import torch.nn as nn
import torch
import os
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from mixk.utils.config import ToExpanduser
from transformers import BertTokenizer
from transformers.modeling_bert import (
    ACT2FN,
    BertAttention,
    BertConfig,
    BertEmbeddings,
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
    BertSelfAttention,
    BertSelfOutput,
)


class GeLU(nn.Module):
    """Implementation of the gelu activation function. For information: OpenAI
    GPT's gelu is slightly different (and gives slightly different results):

    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ACT2FN['gelu'](x)


BertLayerNorm = torch.nn.LayerNorm


class BertCrossattLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.att = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(
            input_tensor,
            encoder_hidden_states=ctx_tensor,
            encoder_attention_mask=ctx_att_mask,
        )[0]
        attention_output = self.output(output, input_tensor)
        return attention_output


"""
---------------------------------------------------------------------------------------
      Above modules are copied from BERT (pytorch-transformer) with modifications.
---------------------------------------------------------------------------------------
"""


class LXRTXLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.lang_self_att = BertAttention(config)
        self.visn_self_att = BertAttention(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        lang_att_output = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention

        lang_att_output = self.lang_self_att(
            lang_input,
            encoder_hidden_states=lang_input,
            encoder_attention_mask=lang_attention_mask,
        )[0]
        visn_att_output = self.visn_self_att(
            visn_input,
            encoder_hidden_states=visn_input,
            encoder_attention_mask=visn_attention_mask,
        )[0]
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats
        lang_att_output, visn_att_output = self.cross_att(
            lang_att_output,
            lang_attention_mask,
            visn_att_output,
            visn_attention_mask,
        )
        lang_att_output, visn_att_output = self.self_att(
            lang_att_output,
            lang_attention_mask,
            visn_att_output,
            visn_attention_mask,
        )
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        return lang_output, visn_output


class VisualFeatEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        feat_dim = config.visual_feat_dim
        pos_dim = config.visual_pos_dim

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        feats, boxes = visn_input

        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        if boxes is not None:
            y = self.box_fc(boxes)
            y = self.box_layer_norm(y)
            output = (x + y) / 2
        else:
            output = x

        output = self.dropout(output)
        return output


class LXRTEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        self.visn_fc = VisualFeatEncoder(config)

        # Number of layers
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(self.num_l_layers)])
        self.x_layers = nn.ModuleList([LXRTXLayer(config) for _ in range(self.num_x_layers)])
        self.r_layers = nn.ModuleList([BertLayer(config) for _ in range(self.num_r_layers)])

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask=None):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        visn_feats = self.visn_fc(visn_feats)

        # Run language layers
        for layer_module in self.layer:
            lang_feats = layer_module(lang_feats, lang_attention_mask)[0]

        # Run relational layers
        for layer_module in self.r_layers:
            visn_feats = layer_module(visn_feats, visn_attention_mask)[0]

        # Run cross-modality layers
        for layer_module in self.x_layers:
            lang_feats, visn_feats = layer_module(
                lang_feats,
                lang_attention_mask,
                visn_feats,
                visn_attention_mask,
            )

        return lang_feats, visn_feats


class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertVisualAnswerHead(nn.Module):

    def __init__(self, config, num_labels):
        super().__init__()
        hid_dim = config.hidden_size

        if config.training_head_type == 'nlvr2':
            in_dim = hid_dim * 2
            out_dim = 2
        else:
            in_dim = hid_dim
            out_dim = config.num_labels

        add_gqa = isinstance(num_labels, list)
        if add_gqa:
            self.logit_gqa = nn.Sequential(
                nn.Linear(in_dim, hid_dim * 2),
                GeLU(),
                nn.LayerNorm(hid_dim * 2, eps=1e-12),
                nn.Linear(hid_dim * 2, num_labels[1]),
            )
            out_dim = num_labels[0]

        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim * 2),
            GeLU(),
            nn.LayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, out_dim),
        )

    def forward(self, hidden_states, name=None):
        if name is None or 'gqa' not in name:
            return self.logit_fc(hidden_states)
        else:
            return self.logit_gqa(hidden_states)


class BertVisualObjHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        visual_losses = ','.join(config.visual_losses)
        self.visual_losses = visual_losses.split(',')

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict(
            {key: nn.Linear(config.hidden_size, config.visual_loss_config[key][0])
             for key in self.visual_losses})

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class LXMERTXLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.lang_self_att = BertAttention(config)
        self.visn_self_att = BertAttention(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        lang_att_output = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)[0]
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)[0]
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats
        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask, visn_att_output,
                                                          visn_attention_mask)
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask, visn_att_output,
                                                         visn_attention_mask)
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        return lang_output, visn_output


class LXMERTModelEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        self.visn_fc = VisualFeatEncoder(config)

        # Number of layers
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(self.num_l_layers)])
        self.x_layers = nn.ModuleList([LXMERTXLayer(config) for _ in range(self.num_x_layers)])
        self.r_layers = nn.ModuleList([BertLayer(config) for _ in range(self.num_r_layers)])

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask=None):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        visn_feats = self.visn_fc(visn_feats)

        # Run language layers
        for layer_module in self.layer:
            lang_feats = layer_module(lang_feats, lang_attention_mask)[0]

        # Run relational layers
        for layer_module in self.r_layers:
            visn_feats = layer_module(visn_feats, visn_attention_mask)[0]

        # Run cross-modality layers
        for layer_module in self.x_layers:
            lang_feats, visn_feats = layer_module(lang_feats, lang_attention_mask, visn_feats, visn_attention_mask)

        return lang_feats, visn_feats


class LXMERTBase(BertPreTrainedModel):
    """LXMERT Model."""

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = LXMERTModelEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        visual_feats=None,
        visual_loc=None,
        visual_attention_mask=None,
        output_all_attention_masks=False,
        output_all_encoded_layers=False,
    ):

        if output_all_encoded_layers:
            raise NotImplementedError
        if output_all_attention_masks:
            raise NotImplementedError

        visual_feats = (visual_feats, visual_loc)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the
        # triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=next(
                self.parameters()).dtype)  # fp16 compatibility
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Run LXMERT backbone
        lang_feats, visn_feats = self.encoder(
            embedding_output,
            extended_attention_mask,
            visn_feats=visual_feats,
            visn_attention_mask=extended_visual_attention_mask,
        )
        pooled_output = self.pooler(lang_feats)

        return (lang_feats, visn_feats), pooled_output


class LXMERTForPretraining(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Configuration
        self.config = config

        # LXMERT backbone
        self.bert = LXMERTBase.from_pretrained(
            self.config.bert_model_name,
            config=BertConfig.from_dict(self.config),
        )

        self.num_labels = config.num_labels
        self.gqa_labels = config.gqa_labels
        self.task_mask_lm = config.task_mask_lm
        self.task_obj_predict = config.task_obj_predict
        self.task_matched = config.task_matched
        self.task_qa = config.task_qa
        self.visual_losses = config.visual_losses
        self.visual_loss_config = config.visual_loss_config

        # Pre-training heads
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)

        if self.task_obj_predict:
            self.obj_predict_head = BertVisualObjHead(config)
        if self.task_qa:
            self.answer_head = BertVisualAnswerHead(config, [self.num_labels, self.gqa_labels])

        # # loss functions
        self.loss_fcts = {
            'l2': SmoothL1Loss(reduction='none'),
            'ce': CrossEntropyLoss(ignore_index=-1, reduction='none'),
            'ce_lang': CrossEntropyLoss(ignore_index=-1),
        }

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.config.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()
                self.cls.apply(self.bert._init_weights)
            self.tie_weights()

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.

        Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder, self.bert.embeddings.word_embeddings)

    def forward(
        self,
        input_ids,  # tokens
        token_type_ids=None,
        attention_mask=None,
        visual_feats=None,
        visual_pos=None,
        visual_attention_mask=None,
        masked_lm_labels=None,
        masked_image_labels=None,
        obj_labels=None,
        matched_label=None,  #
        ans=None,  # qa answer
        num_features=None,  # max num of objects
        name=None,
        output_all_attention_masks=False,
        output_all_encoded_layers=False,
    ):

        (lang_output, visn_output), pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            visual_feats,
            visual_pos,
            visual_attention_mask,
            output_all_attention_masks,
            output_all_encoded_layers,
        )

        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)

        # KEEP TRACK OF OUTPUTS HERE
        output = {}
        if output_all_attention_masks:
            raise NotImplementedError

        if ans is not None and self.task_qa:
            answer_score = self.answer_head(pooled_output, name)
            if name is None or 'gqa' not in name:
                num_labels = self.config.num_labels
            else:
                num_labels = self.config.gqa_labels
            answer_loss = self.loss_fcts['ce_lang'](answer_score.view(-1, num_labels), ans.argmax(-1))
            output['answer_loss'] = answer_loss
        if masked_lm_labels is not None and self.task_mask_lm:
            masked_lm_loss = self.loss_fcts['ce_lang'](
                lang_prediction_scores.view(-1, lang_prediction_scores.size(-1)),
                masked_lm_labels.view(-1),
            )
            output['masked_lm_loss'] = masked_lm_loss
        if matched_label is not None and self.task_matched:
            matched_label = matched_label.to(cross_relationship_score).long()
            matched_loss = self.loss_fcts['ce_lang'](cross_relationship_score.view(-1, 2), matched_label)
            output['matched_loss'] = matched_loss
        if obj_labels is not None and self.task_obj_predict:
            total_visn_loss = 0.0
            visn_prediction_scores_dict = self.obj_predict_head(visn_output)
            for key in self.visual_losses:
                visn_prediction_scores = visn_prediction_scores_dict[key]
                (
                    output_dim,
                    loss_fct_name,
                    label_shape,
                    weight,
                ) = self.visual_loss_config[key]
                if key == 'attr':
                    continue
                elif key == 'obj':
                    temp_obj_labels_dict = obj_labels.max(-1)
                    mask_conf = temp_obj_labels_dict.values
                    visn_loss = self.loss_fcts[loss_fct_name](
                        visn_prediction_scores.view(-1, output_dim),
                        temp_obj_labels_dict.indices.view(-1),
                    )
                elif key == 'feat':
                    if type(masked_image_labels) is None:
                        continue
                    mask_conf = (masked_image_labels == 1).float()
                    visn_loss = self.loss_fcts[loss_fct_name](
                        visn_prediction_scores.view(-1, output_dim),
                        visual_feats.view(-1, output_dim),
                    )
                if visn_loss.dim() > 1:  # Regression Losses
                    visn_loss = visn_loss.mean(1)
                visn_loss = (visn_loss * mask_conf.view(-1)).mean() * weight
                total_visn_loss += visn_loss
            output['visn_loss'] = total_visn_loss

        # # KEEP TRACK OF OUTPUTS HERE
        # output = {
        #     'lang_prediction_scores': lang_prediction_scores,
        #     'cross_relationship_score': cross_relationship_score,
        # }
        #
        # if output_all_attention_masks:
        #     raise NotImplementedError
        #
        # if self.task_qa:
        #     answer_score = self.answer_head(pooled_output, name)
        # else:
        #     # This answer_score would not be used anywhere,
        #     # just to keep a constant return function signature.
        #     answer_score = pooled_output[0][0]
        # output.update({'answer_score': answer_score})
        #
        # if self.task_obj_predict:
        #     visn_prediction_scores_dict = self.obj_predict_head(visn_output)
        #     output.update({'visn_prediction_scores_dict': visn_prediction_scores_dict})

        return output


class LXRTModel(BertPreTrainedModel):
    """LXRT Model."""

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = LXRTEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                visual_feats=None,
                visual_attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the
        # triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=next(
                self.parameters()).dtype)  # fp16 compatibility
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        # Run LXMERT backbone
        lang_feats, visn_feats = self.encoder(
            embedding_output,
            extended_attention_mask,
            visn_feats=visual_feats,
            visn_attention_mask=extended_visual_attention_mask,
        )
        pooled_output = self.pooler(lang_feats)

        return (lang_feats, visn_feats), pooled_output


class LXRTFeatureExtraction(BertPreTrainedModel):
    """BERT model for classification."""

    def __init__(self, config):
        """
            :param config:
            :param mode:  Number of visual layers
            """
        super().__init__(config)
        self.bert = LXRTModel(config)
        # self.mode = mode
        self.apply(self._init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                visual_feats=None,
                visual_attention_mask=None):
        feat_seq, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            visual_feats=visual_feats,
            visual_attention_mask=visual_attention_mask)
        '''
        if 'x' == self.mode:
        return pooled_output
        elif 'x' in self.mode and ('l' in self.mode or 'r' in self.mode):
        return feat_seq, pooled_output
        elif 'l' in self.mode or 'r' in self.mode:
        return feat_seq
        '''
        return pooled_output


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # Keep segment id which allows loading BERT-weights.
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids))
    return features


class LXMERTEncoder(nn.Module):

    def __init__(self, args, mode='x'):
        super().__init__()
        # Build LXRT Model
        self.config = args
        self.max_seq_length = self.config['max_seq_length']
        # set_visual_config(args)

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True,
        )

        # Build LXRT Model
        self.model = LXRTFeatureExtraction.from_pretrained(
            self.config['bert_model_name'],
            config=BertConfig.from_dict(self.config),
            cache_dir=ToExpanduser.modify_path(os.path.join('~/.cache/torch', 'transformers')),
        )
        '''
        if self.config['from_scratch'] is not None:
        print("the model would be trained from scratch")
        self.model.apply(self.model._init_weights)
        '''

    def forward(self, sents, feats, visual_attention_mask=None):
        train_features = convert_sents_to_features(sents, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        output = self.model(
            input_ids, segment_ids, input_mask, visual_feats=feats, visual_attention_mask=visual_attention_mask)
        return output

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join('%s_LXRT.pth' % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print('Load LXMERT pre-trained model from %s' % path)
        state_dict = torch.load(path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[len('module.'):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print('Weights in loaded but not in model:')
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print('Weights in model but not in loaded:')
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)


class ClassificationModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.config = kwargs
        self.num_labels = self.config['num_labels']
        self.training_head_type = self.config['training_head_type']

        # Build LXRT encoder
        self.lxrt_encoder = LXMERTEncoder(self.config)
        hid_dim = self.config['hidden_size']

        if self.training_head_type == 'nlvr2':
            self.logit_fc = nn.Sequential(
                nn.Linear(hid_dim * 2, hid_dim * 2), GeLU(), nn.LayerNorm(hid_dim * 2, eps=1e-12),
                nn.Linear(hid_dim * 2, 2))
        else:
            self.logit_fc = nn.Sequential(
                nn.Linear(hid_dim, hid_dim * 2), GeLU(), nn.LayerNorm(hid_dim * 2, eps=1e-12),
                nn.Linear(hid_dim * 2, self.num_labels))

        self.logit_fc.apply(self.lxrt_encoder.model._init_weights)

    def forward(self, feat, pos, sent):
        if self.training_head_type == 'nlvr2':
            sent = sum(zip(sent, sent), ())
            batch_size, img_num, obj_num, feat_size = feat.size()
            assert img_num == 2 and obj_num == 36 and feat_size == 2048
            feat = feat.view(batch_size * 2, obj_num, feat_size)
            pos = pos.view(batch_size * 2, obj_num, 4)

        x = self.lxrt_encoder(sent, (feat, pos))
        """
        :param feat: b, 2, o, f
        :param pos:  b, 2, o, 4
        :param sent: b, (string)
        :param leng: b, (numpy, int)
        :return:
        """
        # Pairing images and sentences:
        # The input of NLVR2 is two images and one sentence. In batch level, they are saved as
        #   [ [img0_0, img0_1], [img1_0, img1_1], ...] and [sent0, sent1, ...]
        # Here, we flat them to
        #   feat/pos = [ img0_0, img0_1, img1_0, img1_1, ...]
        #   sent     = [ sent0,  sent0,  sent1,  sent1,  ...]
        if self.training_head_type == 'nlvr2':
            x = x.view(-1, x.size(1) * 2)

        logit = self.logit_fc(x)

        output = {}
        reshaped_logits = logit.contiguous().view(-1, self.num_labels)
        output['scores'] = reshaped_logits

        return output
