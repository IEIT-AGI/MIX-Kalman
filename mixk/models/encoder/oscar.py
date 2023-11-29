import math

import torch
import torch.nn as nn
from transformers.modeling_bert import (BertAttention, BertEncoder, BertIntermediate, BertLayer, BertOutput, BertPooler,
                                        BertPreTrainedModel, BertSelfAttention, BertSelfOutput)

from ..builder import ENCODER


class CaptionBertSelfAttention(BertSelfAttention):
    """Modified from BertSelfAtntion to add support for
    output_hidden_states."""

    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)

    def forward(self, hidden_states, attention_mask, head_mask=None, history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer, )
        return outputs


class CaptionBertAttention(BertAttention):
    """Modified from BertAttention to add support for output_hidden_states."""

    def __init__(self, config):
        super(CaptionBertAttention, self).__init__(config)
        self.self = CaptionBertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None, history_state=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask, history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output, ) + self_outputs[1:]  # add attentions if we output them
        return outputs


class CaptionBertLayer(BertLayer):
    """Modified from BertLayer to add support for output_hidden_states."""

    def __init__(self, config):
        super(CaptionBertLayer, self).__init__(config)
        self.attention = CaptionBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None, history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask, history_state)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output, ) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class CaptionBertEncoder(BertEncoder):
    """Modified from BertEncoder to add support for output_hidden_states."""

    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None, encoder_history_states=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            history_state = None if encoder_history_states is None else encoder_history_states[i]
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], history_state)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        outputs = (hidden_states, )
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states, )
        if self.output_attentions:
            outputs = outputs + (all_attentions, )
        return outputs  # outputs, (hidden states), (attentions)


@ENCODER.register_module()
class OSCARBackbone(BertPreTrainedModel):

    def __init__(
        self,
        config,
        visual_embedding_dim=512,
        embedding_strategy='plain',
        bypass_transformer=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        super().__init__(config)
        self.config = config

        self.encoder = CaptionBertEncoder(config)
        self.pooler = BertPooler(config)

        self.apply(self.init_weights)

    def forward(self, embedding_output, extended_attention_mask, head_mask, encoder_history_states):

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            encoder_history_states=encoder_history_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (
            sequence_output,
            pooled_output,
        ) + encoder_outputs[1:]
        return outputs
