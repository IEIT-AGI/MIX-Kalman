# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import copy
import logging
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

import numpy as np

from transformers.modeling_bert import (
    BertConfig,
    BertPreTrainedModel,
    BertPredictionHeadTransform,
    BertPooler,
    BertLayer,
    BertIntermediate,
    BertOutput,
    # BertLMPredictionHead,
    # BertEmbeddings,
    # BertEncoder,
    # BertAttention,
    # BertSelfAttention,
    # BertSelfOutput,
    # ACT2FN,
)

logger = logging.getLogger(__name__)


class BertLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the
        square root)."""
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type
    embeddings."""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertBiAttention(nn.Module):

    def __init__(self, config):
        super(BertBiAttention, self).__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention '
                             'heads (%d)' % (config.bi_hidden_size, config.bi_num_attention_heads))

        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(config.bi_hidden_size / config.bi_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']
        v_config = BertConfig.from_dict(config.v_config)

        self.query1 = nn.Linear(v_config.hidden_size, self.all_head_size)
        self.key1 = nn.Linear(v_config.hidden_size, self.all_head_size)
        self.value1 = nn.Linear(v_config.hidden_size, self.all_head_size)
        # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)
        self.dropout1 = nn.Dropout(v_config.attention_probs_dropout_prob)

        t_config = BertConfig.from_dict(config.t_config)
        self.query2 = nn.Linear(t_config.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(t_config.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(t_config.hidden_size, self.all_head_size)
        # self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)
        self.dropout2 = nn.Dropout(t_config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
        co_attention_mask=None,
        use_co_attention_mask=False,
    ):

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1

        if use_co_attention_mask:
            attention_scores1 = attention_scores1 + co_attention_mask.permute(0, 1, 3, 2)

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size, )
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        # Take the dot product between "query1" and "key2" to get the raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # we can comment this line for single flow.
        attention_scores2 = attention_scores2 + attention_mask2
        if use_co_attention_mask:
            attention_scores2 = attention_scores2 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size, )
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        return context_layer1, context_layer2, (attention_probs1, attention_probs2)


class BertBiOutput(nn.Module):

    def __init__(self, config):
        super(BertBiOutput, self).__init__()

        v_config = BertConfig.from_dict(config.v_config)

        self.dense1 = nn.Linear(config.bi_hidden_size, v_config.hidden_size)
        self.LayerNorm1 = BertLayerNorm(v_config.hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(v_config.hidden_dropout_prob)

        # self.q_dense1 = nn.Linear(config.bi_hidden_size, v_config.hidden_size)
        # self.q_dropout1 = nn.Dropout(v_config.hidden_dropout_prob)

        t_config = BertConfig.from_dict(config.t_config)

        self.dense2 = nn.Linear(config.bi_hidden_size, t_config.hidden_size)
        self.LayerNorm2 = BertLayerNorm(t_config.hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(t_config.hidden_dropout_prob)

        # self.q_dense2 = nn.Linear(config.bi_hidden_size, t_config.hidden_size)
        # self.q_dropout2 = nn.Dropout(t_config.hidden_dropout_prob)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):

        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2


class BertConnectionLayer(nn.Module):

    def __init__(self, config):
        super(BertConnectionLayer, self).__init__()
        self.biattention = BertBiAttention(config)
        self.biOutput = BertBiOutput(config)

        v_config = BertConfig.from_dict(config.v_config)
        self.v_intermediate = BertIntermediate(v_config)
        self.v_output = BertOutput(v_config)

        t_config = BertConfig.from_dict(config.t_config)
        self.t_intermediate = BertIntermediate(t_config)
        self.t_output = BertOutput(t_config)

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
        co_attention_mask=None,
        use_co_attention_mask=False,
    ):

        bi_output1, bi_output2, co_attention_probs = self.biattention(
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask,
            use_co_attention_mask,
        )

        attention_output1, attention_output2 = self.biOutput(bi_output2, input_tensor1, bi_output1, input_tensor2)

        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)

        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)

        return layer_output1, layer_output2, co_attention_probs


class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()

        # in the bert encoder, we need to extract three things here.
        # text bert layer: BertLayer
        # vision bert layer: BertImageLayer
        # Bi-Attention: Given the output of two bertlayer, perform bi-directional
        # attention and add on two layers.
        t_config = BertConfig.from_dict(config.t_config)
        v_config = BertConfig.from_dict(config.v_config)

        self.FAST_MODE = config.fast_mode
        self.with_coattention = config.with_coattention
        self.v_biattention_id = v_config.biattention_id
        self.t_biattention_id = t_config.biattention_id
        self.in_batch_pairs = config.in_batch_pairs
        self.fixed_t_layer = config.fixed_t_layer
        self.fixed_v_layer = config.fixed_v_layer

        # layer = BertLayer(config)
        layer = BertLayer(t_config)
        v_layer = BertLayer(v_config)
        connect_layer = BertConnectionLayer(config)

        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(t_config.num_hidden_layers)])
        self.v_layer = nn.ModuleList([copy.deepcopy(v_layer) for _ in range(v_config.num_hidden_layers)])
        self.c_layer = nn.ModuleList([copy.deepcopy(connect_layer) for _ in range(len(v_config.biattention_id))])

    def forward(
        self,
        txt_embedding,
        image_embedding,
        txt_attention_mask,
        image_attention_mask,
        co_attention_mask=None,
        output_all_encoded_layers=True,
        output_all_attention_masks=False,
    ):

        v_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []

        all_attention_mask_t = []
        all_attnetion_mask_v = []
        all_attention_mask_c = []

        batch_size, num_words, t_hidden_size = txt_embedding.size()
        _, num_regions, v_hidden_size = image_embedding.size()

        use_co_attention_mask = False
        for v_layer_id, t_layer_id in zip(self.v_biattention_id, self.t_biattention_id):

            v_end = v_layer_id
            t_end = t_layer_id

            assert self.fixed_t_layer <= t_end
            assert self.fixed_v_layer <= v_end

            for idx in range(v_start, self.fixed_v_layer):
                with torch.no_grad():
                    outputs = self.v_layer[idx](
                        image_embedding,
                        image_attention_mask,
                    )
                    image_embedding, image_attention_probs = outputs[0], outputs[1:]

                    v_start = self.fixed_v_layer

                    if output_all_attention_masks:
                        all_attnetion_mask_v.append(image_attention_probs)

            for idx in range(v_start, v_end):
                outputs = self.v_layer[idx](
                    image_embedding,
                    image_attention_mask,
                )
                image_embedding, image_attention_probs = outputs[0], outputs[1:]

                if output_all_attention_masks:
                    all_attnetion_mask_v.append(image_attention_probs)

            for idx in range(t_start, self.fixed_t_layer):
                with torch.no_grad():
                    outputs = self.layer[idx](txt_embedding, txt_attention_mask)
                    txt_embedding, txt_attention_probs = outputs[0], outputs[1:]
                    t_start = self.fixed_t_layer
                    if output_all_attention_masks:
                        all_attention_mask_t.append(txt_attention_probs)

            for idx in range(t_start, t_end):
                outputs = self.layer[idx](txt_embedding, txt_attention_mask)
                txt_embedding, txt_attention_probs = outputs[0], outputs[1:]

                if output_all_attention_masks:
                    all_attention_mask_t.append(txt_attention_probs)

            if count == 0 and self.in_batch_pairs:
                # new batch size is the batch_size ^2
                image_embedding = image_embedding.unsqueeze(0).expand(batch_size, batch_size, num_regions,
                                                                      v_hidden_size).contiguous().view(
                                                                          batch_size * batch_size, num_regions,
                                                                          v_hidden_size)
                image_attention_mask = image_attention_mask.unsqueeze(0).expand(batch_size, batch_size, 1, 1,
                                                                                num_regions).contiguous().view(
                                                                                    batch_size * batch_size, 1, 1,
                                                                                    num_regions)

                txt_embedding = txt_embedding.unsqueeze(1).expand(batch_size, batch_size, num_words,
                                                                  t_hidden_size).contiguous().view(
                                                                      batch_size * batch_size, num_words, t_hidden_size)
                txt_attention_mask = txt_attention_mask.unsqueeze(1).expand(
                    batch_size, batch_size, 1, 1, num_words).contiguous().view(batch_size * batch_size, 1, 1, num_words)
                co_attention_mask = co_attention_mask.unsqueeze(1).expand(batch_size, batch_size, 1, num_regions,
                                                                          num_words).contiguous().view(
                                                                              batch_size * batch_size, 1, num_regions,
                                                                              num_words)

            if count == 0 and self.FAST_MODE:
                txt_embedding = txt_embedding.expand(
                    image_embedding.size(0),
                    txt_embedding.size(1),
                    txt_embedding.size(2),
                )
                txt_attention_mask = txt_attention_mask.expand(
                    image_embedding.size(0),
                    txt_attention_mask.size(1),
                    txt_attention_mask.size(2),
                    txt_attention_mask.size(3),
                )

            if self.with_coattention:
                # do the bi attention.
                image_embedding, txt_embedding, co_attention_probs = self.c_layer[count](
                    image_embedding,
                    image_attention_mask,
                    txt_embedding,
                    txt_attention_mask,
                    co_attention_mask,
                    use_co_attention_mask,
                )

                if output_all_attention_masks:
                    all_attention_mask_c.append(co_attention_probs)

            v_start = v_end
            t_start = t_end
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(image_embedding)

        for idx in range(v_start, len(self.v_layer)):
            outputs = self.v_layer[idx](
                image_embedding,
                image_attention_mask,
            )
            image_embedding, image_attention_probs = outputs[0], outputs[1:]

            if output_all_attention_masks:
                all_attnetion_mask_v.append(image_attention_probs)

        for idx in range(t_start, len(self.layer)):
            outputs = self.layer[idx](txt_embedding, txt_attention_mask)
            txt_embedding, txt_attention_probs = outputs[0], outputs[1:]

            if output_all_attention_masks:
                all_attention_mask_t.append(txt_attention_probs)

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(image_embedding)

        return (
            all_encoder_layers_t,
            all_encoder_layers_v,
            (all_attention_mask_t, all_attnetion_mask_v, all_attention_mask_c),
        )


class BertTextPooler(BertPooler):

    def __init__(self, config):
        super(BertTextPooler, self).__init__(config)
        self.dense = nn.Linear(config.t_config['hidden_size'], config.bi_hidden_size)
        self.activation = nn.ReLU()


class BertImagePooler(BertPooler):

    def __init__(self, config):
        super(BertImagePooler, self).__init__(config)
        self.dense = nn.Linear(config.v_config['hidden_size'], config.bi_hidden_size)
        self.activation = nn.ReLU()


class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights, hidden_size=768):
        super(BertLMPredictionHead, self).__init__()
        config.hidden_size = hidden_size
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if hidden_size == 768:
            self.decoder = nn.Linear(
                bert_model_embedding_weights.size(1),
                bert_model_embedding_weights.size(0),
                bias=False,
            )
            # print(self.decoder.weight.shape) # 30522, 768
            self.decoder.weight = bert_model_embedding_weights
        else:
            self.decoder = nn.Linear(
                hidden_size,
                bert_model_embedding_weights.size(0),
                bias=False,
            )
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        t_config = BertConfig.from_dict(config.t_config)
        self.causal_predictor_t2v = BertLMPredictionHead(t_config, bert_model_embedding_weights, 768)
        self.causal_predictor_t = BertLMPredictionHead(t_config, bert_model_embedding_weights, 768)
        self.predictions = BertLMPredictionHead(t_config, bert_model_embedding_weights, 768)
        self.bi_seq_relationship = nn.Linear(config.bi_hidden_size, 2)

        v_config = BertConfig.from_dict(config.v_config)
        self.causal_predictor_v2t = BertImagePredictionHead(v_config, 1024)
        self.causal_predictor_v = BertImagePredictionHead(v_config, 2048)  # causal loss，必须放在前面，它修改了config.v_hidden_size
        self.imagePredictions = BertImagePredictionHead(v_config, 1024)  # 类比之前的mask_loss_v
        self.fusion_method = config.fusion_method
        self.dropout = nn.Dropout(0.1)

        self.criterion_v = nn.KLDivLoss(reduction='none')
        self.criterion_t = CrossEntropyLoss(ignore_index=-1)

    def forward(self,
                sequence_output_t,
                sequence_output_v,
                pooled_output_t,
                pooled_output_v,
                causal_sequence_output_v=None,
                causal_sequence_output_v2t=None,
                causal_label_v=None,
                image_target=None,
                causal_sequence_output_t=None,
                causal_sequence_output_t2v=None,
                causal_label_t=None):
        if self.fusion_method == 'sum':
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == 'mul':
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            assert False

        seq_relationship_score = self.bi_seq_relationship(pooled_output)
        prediction_scores_v = self.imagePredictions(sequence_output_v)
        prediction_scores_t = self.predictions(sequence_output_t)
        if image_target is not None:
            causal_prediction_v = self.causal_predictor_v(causal_sequence_output_v)
            causal_prediction_v_loss = self.criterion_v(F.log_softmax(causal_prediction_v, dim=2), image_target)
            causal_prediction_v_loss = torch.sum(causal_prediction_v_loss *
                                                 (causal_label_v == 1).unsqueeze(2).float()) / max(
                                                     torch.sum((causal_label_v == 1)), 1)

            causal_prediction_t = self.causal_predictor_t(causal_sequence_output_t)
            causal_prediction_t_loss = self.criterion_t(causal_prediction_t.view(-1, 30522), causal_label_t.view(-1))

            causal_prediction_v2t = self.causal_predictor_v2t(causal_sequence_output_v2t)
            causal_prediction_v2t_loss = self.criterion_v(F.log_softmax(causal_prediction_v2t, dim=2), image_target)
            causal_prediction_v2t_loss = torch.sum(causal_prediction_v2t_loss *
                                                   (causal_label_v == 1).unsqueeze(2).float()) / max(
                                                       torch.sum((causal_label_v == 1)), 1)

            causal_prediction_t2v = self.causal_predictor_t2v(causal_sequence_output_t2v)
            causal_prediction_t2v_loss = self.criterion_t(
                causal_prediction_t2v.view(-1, 30522), causal_label_t.view(-1))

            return (prediction_scores_t, prediction_scores_v, seq_relationship_score, causal_prediction_v_loss,
                    causal_prediction_t_loss, causal_prediction_v2t_loss, causal_prediction_t2v_loss)
        else:
            return (prediction_scores_t, prediction_scores_v, seq_relationship_score)


class BertImagePredictionHead(nn.Module):

    def __init__(self, config, hidden_size=1024):
        super(BertImagePredictionHead, self).__init__()
        config.hidden_size = hidden_size
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.target_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a
    Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers`
            output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)

        t_config = BertConfig.from_dict(config.t_config)
        v_config = BertConfig.from_dict(config.v_config)

        # initilize word embedding
        self.embeddings = BertEmbeddings(t_config)

        # initlize the vision embedding
        self.v_embeddings = BertImageEmbeddings(v_config)

        self.encoder = BertEncoder(config)
        self.t_pooler = BertTextPooler(config)
        self.v_pooler = BertImagePooler(config)

        self.init_weights()

    def forward(
        self,
        input_txt,
        input_imgs,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
        output_all_encoded_layers=False,
        output_all_attention_masks=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(input_imgs.size(0), input_imgs.size(1)).type_as(input_txt)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_image_attention_mask = extended_image_attention_mask.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        if co_attention_mask is None:
            co_attention_mask = torch.zeros(input_txt.size(0), input_imgs.size(1),
                                            input_txt.size(1)).type_as(extended_image_attention_mask)

        extended_co_attention_mask = co_attention_mask.unsqueeze(1)

        # extended_co_attention_mask = co_attention_mask.unsqueeze(-1)
        extended_co_attention_mask = extended_co_attention_mask * 5.0
        extended_co_attention_mask = extended_co_attention_mask.to(dtype=next(
            self.parameters()).dtype)  # fp16 compatibility

        embedding_output = self.embeddings(input_txt, token_type_ids)
        v_embedding_output = self.v_embeddings(input_imgs, image_loc)
        encoded_layers_t, encoded_layers_v, all_attention_mask = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            extended_co_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )

        sequence_output_t = encoded_layers_t[-1]
        sequence_output_v = encoded_layers_v[-1]

        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_v = encoded_layers_v[-1]

        return (
            encoded_layers_t,
            encoded_layers_v,
            pooled_output_t,
            pooled_output_v,
            all_attention_mask,
        )


class BertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and
    token_type embeddings."""

    def __init__(self, config):
        super(BertImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.feature_size, config.hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_loc):

        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)
        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class Causal_v(nn.Module):

    def __init__(self):
        super(Causal_v, self).__init__()
        self.embedding_size = 1024
        self.Wy = nn.Linear(1024, 1024)
        self.Wz = nn.Linear(2048, 1024)
        self.dic_z = torch.tensor(np.load('./dic/dic_v.npy'), dtype=torch.float).cuda()
        self.prior = torch.tensor(np.load('./dic/prior_v.npy'), dtype=torch.float).cuda()
        nn.init.normal_(self.Wy.weight, std=0.02)
        nn.init.normal_(self.Wz.weight, std=0.02)
        nn.init.constant_(self.Wy.bias, 0)
        nn.init.constant_(self.Wz.bias, 0)

    def forward(self, y):
        temp = []
        # Class = torch.argmax(image_target, 2, keepdim=True)
        # for boxes, cls in zip(y, Class):
        for boxes in y:
            attention = torch.mm(self.Wy(boxes), self.Wz(self.dic_z).t()) / (self.embedding_size**0.5)
            attention = F.softmax(attention, 1)  # torch.Size([box, 1601])
            z_hat = attention.unsqueeze(2) * self.dic_z.unsqueeze(0)  # torch.Size([box, 1601, 2048])
            z = torch.matmul(self.prior.unsqueeze(0),
                             z_hat).squeeze(1)  # torch.Size([box, 1, 2048])->torch.Size([box, 2048])
            temp.append(z)
        temp = torch.stack(temp, 0)
        return temp


class Causal_t(nn.Module):

    def __init__(self):
        super(Causal_t, self).__init__()
        self.embedding_size = 768
        self.Wy = nn.Linear(768, 768)
        self.Wz = nn.Linear(768, 768)
        self.dic_z = torch.tensor(np.load('./dic/dic_t.npy'), dtype=torch.float).cuda()
        self.prior = torch.tensor(np.load('./dic/prior_t.npy'), dtype=torch.float).cuda()
        nn.init.normal_(self.Wy.weight, std=0.02)
        nn.init.normal_(self.Wz.weight, std=0.02)
        nn.init.constant_(self.Wy.bias, 0)
        nn.init.constant_(self.Wz.bias, 0)
        # self.id2class = np.load("./dic/id2class.npy", allow_pickle=True).item()

    def forward(self, y):
        temp = []
        for sentence in y:
            attention = torch.mm(self.Wy(sentence), self.Wz(self.dic_z).t()) / (self.embedding_size**0.5)
            attention = F.softmax(attention, 1)  # torch.Size([box, 1601])
            z_hat = attention.unsqueeze(2) * self.dic_z.unsqueeze(0)  # torch.Size([box, 1601, 2048])
            z = torch.matmul(self.prior.unsqueeze(0),
                             z_hat).squeeze(1)  # torch.Size([box, 1, 2048])->torch.Size([box, 2048])
            temp.append(z)
        temp = torch.stack(temp, 0)
        return temp


class Causal_t2v(nn.Module):

    def __init__(self):
        super(Causal_t2v, self).__init__()
        self.embedding_size = 768
        self.Wy = nn.Linear(768, 768)
        self.Wz = nn.Linear(2048, 768)
        self.t2v = nn.Linear(2048, 768)
        self.dic_z = torch.tensor(np.load('./dic/dic_v.npy'), dtype=torch.float).cuda()
        self.prior = torch.tensor(np.load('./dic/prior_v.npy'), dtype=torch.float).cuda()
        nn.init.normal_(self.Wy.weight, std=0.02)
        nn.init.normal_(self.Wz.weight, std=0.02)
        nn.init.constant_(self.Wy.bias, 0)
        nn.init.constant_(self.Wz.bias, 0)

    def forward(self, y):
        temp = []
        # Class = torch.argmax(image_target, 2, keepdim=True)
        # for boxes, cls in zip(y, Class):
        for boxes in y:
            attention = torch.mm(self.Wy(boxes), self.Wz(self.dic_z).t()) / (self.embedding_size**0.5)
            attention = F.softmax(attention, 1)  # torch.Size([box, 1601])
            z_hat = attention.unsqueeze(2) * self.t2v(self.dic_z).unsqueeze(0)  # torch.Size([box, 1601, 2048])
            z = torch.matmul(self.prior.unsqueeze(0),
                             z_hat).squeeze(1)  # torch.Size([box, 1, 2048])->torch.Size([box, 2048])
            temp.append(z)
        temp = torch.stack(temp, 0)
        return temp


class Causal_v2t(nn.Module):

    def __init__(self):
        super(Causal_v2t, self).__init__()
        self.embedding_size = 1024
        self.Wy = nn.Linear(1024, 1024)
        self.Wz = nn.Linear(768, 1024)
        self.v2t = nn.Linear(768, 1024)
        self.dic_z = torch.tensor(np.load('./dic/dic_t.npy'), dtype=torch.float).cuda()
        self.prior = torch.tensor(np.load('./dic/prior_t.npy'), dtype=torch.float).cuda()
        nn.init.normal_(self.Wy.weight, std=0.02)
        nn.init.normal_(self.Wz.weight, std=0.02)
        nn.init.constant_(self.Wy.bias, 0)
        nn.init.constant_(self.Wz.bias, 0)
        # self.id2class = np.load("./dic/id2class.npy", allow_pickle=True).item()

    def forward(self, y):
        temp = []
        for sentence in y:
            attention = torch.mm(self.Wy(sentence), self.Wz(self.dic_z).t()) / (self.embedding_size**0.5)
            attention = F.softmax(attention, 1)  # torch.Size([box, 1601])
            z_hat = attention.unsqueeze(2) * self.v2t(self.dic_z).unsqueeze(0)  # torch.Size([box, 1601, 2048])
            z = torch.matmul(self.prior.unsqueeze(0),
                             z_hat).squeeze(1)  # torch.Size([box, 1, 2048])->torch.Size([box, 2048])
            temp.append(z)
        temp = torch.stack(temp, 0)
        return temp


class BertForMultiModalPreTraining(BertPreTrainedModel):
    """BERT model with multi modal pre-training heads."""

    def __init__(self, config):
        super(BertForMultiModalPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)

        self.init_weights()
        self.predict_feature = config.predict_feature
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)

        self.causal_v = Causal_v()
        self.causal_t = Causal_t()
        self.causal_t2v = Causal_t2v()
        self.causal_v2t = Causal_v2t()

        print("model's option for predict_feature is ", config.predict_feature)

        if self.predict_feature:
            self.vis_criterion = nn.MSELoss(reduction='none')
        else:
            self.vis_criterion = nn.KLDivLoss(reduction='none')

    def forward(
        self,
        input_ids,
        image_feat,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        masked_lm_labels=None,
        image_label=None,
        image_target=None,
        next_sentence_label=None,
        causal_label_t=None,
        causal_label_v=None,
        output_all_attention_masks=False,
    ):
        # in this model, we first embed the images.
        sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask = self.bert(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=False,
            output_all_attention_masks=output_all_attention_masks,
        )
        # print(sequence_output_t.shape, sequence_output_v.shape) # torch.Size([64, 36, 768]) torch.Size([64, 37, 1024])
        # Added by jt

        if masked_lm_labels is not None and next_sentence_label is not None and image_target is not None:

            causal_sequence_output_v = self.causal_v(sequence_output_v[:, 1:])
            causal_sequence_output_t = self.causal_t(sequence_output_t)

            causal_sequence_output_v2t = self.causal_v2t(sequence_output_v[:, 1:])
            causal_sequence_output_t2v = self.causal_t2v(sequence_output_t)

            (
                prediction_scores_t,
                prediction_scores_v,
                seq_relationship_score,
                causal_prediction_v_loss,
                causal_prediction_t_loss,
                causal_prediction_v2t_loss,
                causal_prediction_t2v_loss,
            ) = self.cls(
                sequence_output_t,
                sequence_output_v,
                pooled_output_t,
                pooled_output_v,
                causal_sequence_output_v,
                causal_sequence_output_v2t,
                causal_label_v,
                image_target,
                causal_sequence_output_t,
                causal_sequence_output_t2v,
                causal_label_t,
            )

        else:
            prediction_scores_t, prediction_scores_v, seq_relationship_score = self.cls(
                sequence_output_t,
                sequence_output_v,
                pooled_output_t,
                pooled_output_v,
            )

        if masked_lm_labels is not None and next_sentence_label is not None and image_target is not None:
            prediction_scores_v = prediction_scores_v[:, 1:]

            if self.predict_feature:
                img_loss = self.vis_criterion(prediction_scores_v, image_target)
                masked_img_loss = torch.sum(img_loss * (image_label == 1).unsqueeze(2).float()) / max(
                    torch.sum((image_label == 1).unsqueeze(2).expand_as(img_loss)), 1)
            else:
                img_loss = self.vis_criterion(F.log_softmax(prediction_scores_v, dim=2), image_target)

                masked_img_loss = torch.sum(img_loss * (image_label == 1).unsqueeze(2).float()) / max(
                    torch.sum((image_label == 1)), 1)

            masked_lm_loss = self.loss_fct(
                prediction_scores_t.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            )

            next_sentence_loss = self.loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return (
                masked_lm_loss,
                masked_img_loss,
                next_sentence_loss,
                causal_prediction_v_loss,
                causal_prediction_t_loss,
                causal_prediction_v2t_loss,
                causal_prediction_t2v_loss,
            )
        else:
            return (
                prediction_scores_t,
                prediction_scores_v,
                seq_relationship_score,
                all_attention_mask,
            )


class SimpleClassifier(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class DeVLBertForVLTasks(BertPreTrainedModel):

    def __init__(self, config, num_labels, dropout_prob=0.1, default_gpu=True):
        super(DeVLBertForVLTasks, self).__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_prob)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.vil_prediction = SimpleClassifier(config.bi_hidden_size, config.bi_hidden_size * 2, num_labels, 0.5)
        self.vil_logit = nn.Linear(config.bi_hidden_size, 1)
        self.vision_logit = nn.Linear(config.v_config['hidden_size'], 1)
        self.linguisic_logit = nn.Linear(config.t_config['hidden_size'], 1)
        self.fusion_method = config.fusion_method
        self.init_weights()

    def forward(
        self,
        input_txt,
        input_imgs,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
        output_all_encoded_layers=False,
    ):
        (sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, _) = self.bert(
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            co_attention_mask,
            output_all_encoded_layers=False,
        )

        linguisic_prediction, vision_prediction, vil_binary_prediction = self.cls(sequence_output_t, sequence_output_v,
                                                                                  pooled_output_t, pooled_output_v)

        if self.fusion_method == 'sum':
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == 'mul':
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            assert False

        vil_prediction = self.vil_prediction(pooled_output)
        vil_logit = self.vil_logit(pooled_output)
        vision_logit = self.vision_logit(self.dropout(sequence_output_v)) + (
            (1.0 - image_attention_mask) * -10000.0).unsqueeze(2).to(dtype=next(self.parameters()).dtype)
        linguisic_logit = self.linguisic_logit(self.dropout(sequence_output_t))

        return (
            vil_prediction,
            vil_logit,
            vil_binary_prediction,
            vision_prediction,
            vision_logit,
            linguisic_prediction,
            linguisic_logit,
        )
