import logging
import torch.nn as nn
from transformers.modeling_bert import BertConfig, BertEmbeddings, BertEncoder, BertPreTrainedModel  # BertLayerNorm,

from ..builder import ENCODER

logger = logging.getLogger(__name__)

TEXT_BERT_HIDDEN_SIZE = 768


@ENCODER.register_module()
class TextBertBase(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        text_bert_init_from_bert_base = kwargs['text_bert_init_from_bert_base']
        params = kwargs['params']
        hidden_size = kwargs['hidden_size']
        self.text_bert_config = BertConfig(**params)

        # TEST FORWARD
        self.text_bert_config.hidden_dropout_prob = 0.0
        self.text_bert_config.attention_probs_dropout_prob = 0.0
        #

        if text_bert_init_from_bert_base:
            self.text_bert = TextBert.from_pretrained('bert-base-uncased', config=self.text_bert_config)
        else:
            logger.info('NOT initializing text_bert from BERT_BASE')
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if hidden_size != TEXT_BERT_HIDDEN_SIZE:
            logger.info(f'Projecting text_bert output to {self.mmt_config.hidden_size} dim')

            self.text_bert_out_linear = nn.Linear(TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size)
        else:
            self.text_bert_out_linear = nn.Identity()

    def forward(self, txt_inds, txt_mask):
        return self.text_bert_out_linear(self.text_bert(txt_inds, txt_mask))


class TextBert(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(encoder_inputs, extended_attention_mask, head_mask=head_mask)
        seq_output = encoder_outputs[0]

        return seq_output
