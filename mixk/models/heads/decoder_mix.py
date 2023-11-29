import json
from abc import ABCMeta, abstractmethod
from typing import Dict

import torch
import torch.nn as nn

from mixk.models.encoder.visdiag_lstm import DynamicRNN
from ..builder import HEADS, build_loss
from torch.nn.utils import weight_norm


@HEADS.register_module()
class VisualDialogueHead(nn.Module, metaclass=ABCMeta):  # VISDIALPRINCIPLES_HEAD  reanme --> VisualDialogueHead

    def __init__(self, loss_cls: Dict):  # if it has a loss_cls
        super().__init__()

    @abstractmethod
    def forward(self, encoder_output, data):
        pass


@HEADS.register_module()
class DiscQtDecoderHead(VisualDialogueHead):

    def __init__(self, config, vocabulary, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__init_argument(config, vocabulary)

    def __init_argument(self, config, vocabulary):
        self.nhid = config['lstm_hidden_size']
        self.word_embed = nn.Embedding(
            len(vocabulary),
            config['word_embedding_size'],
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.option_rnn = nn.LSTM(
            config['word_embedding_size'],
            config['lstm_hidden_size'],
            config['lstm_num_layers'],
            batch_first=True,
            dropout=config['dropout'],
        )

        self.a2a = nn.Linear(self.nhid * 2, self.nhid)
        self.option_rnn = DynamicRNN(self.option_rnn)

        def read_data_from_json(file):
            with open(file, 'r') as f:
                return json.load(f.read())

        self.count_dict = read_data_from_json('data/qt_scores.json')
        self.qt_file = read_data_from_json('data/qt_count.json')
        self.qt_list = list(self.qt_file.keys())

    def forward(self, encoder_output, batch):
        """Given `encoder_output` + candidate option sequences, predict a score
        for each option sequence.

        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        """

        options = batch['opt']
        batch_size, num_options, max_sequence_length = options.size()
        options = options.contiguous().view(-1, max_sequence_length)

        options_length = batch['opt_len']
        options_length = options_length.contiguous().view(-1)

        options_embed = self.word_embed(options)  # b*100 20 300
        _, (options_feat, _) = self.option_rnn(options_embed, options_length)  # b*100 512
        options_feat = options_feat.view(batch_size, num_options, self.nhid)

        encoder_output = encoder_output.unsqueeze(1).repeat(1, num_options, 1)

        scores = torch.sum(options_feat * encoder_output, -1)
        scores = scores.view(batch_size, num_options)

        qt_score = torch.zeros_like(scores)
        qt_idx = batch['qt']
        opt_idx = batch['opt_idx']
        for b in range(batch_size):
            qt_key = self.qt_list[qt_idx[b]]
            ans_relevance = self.count_dict[qt_key]
            for k in range(100):
                idx_temp = str(opt_idx[b][k].detach().cpu().numpy())
                if idx_temp in ans_relevance.keys():
                    qt_score[b][k] = 1

        return scores, qt_score


@HEADS.register_module()
class DiscByRoundDecoderHead(VisualDialogueHead):

    def __int__(self, vocabulary_len, word_embedding_size, lstm_hidden_size, lstm_num_layers, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nhid = lstm_hidden_size

        self.word_embed = nn.Embedding(vocabulary_len, word_embedding_size, padding_idx=0)
        self.option_rnn = nn.LSTM(
            word_embedding_size,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.a2a = nn.Linear(self.nhid * 2, self.nhid)  # this is useless in this version
        # Options are variable length padded sequences, use DynamicRNN.
        self.option_rnn = DynamicRNN(self.option_rnn)

    def forward(self, encoder_output, data):
        """Given `encoder_output` + candidate option sequences, predict a score
        for each option sequence.

        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        """
        options = data['opt']
        batch_size, num_options, max_sequence_length = options.size()
        options = options.contiguous().view(-1, max_sequence_length)

        options_length = data['opt_len']
        options_length = options_length.contiguous().view(-1)

        options_embed = self.word_embed(options)  # b*100 20 300
        _, (options_feat, _) = self.option_rnn(options_embed, options_length)  # b*100 512
        options_feat = options_feat.view(batch_size, num_options, self.nhid)

        encoder_output = encoder_output.unsqueeze(1).repeat(1, num_options, 1)
        scores = torch.sum(options_feat * encoder_output, -1)
        scores = scores.view(batch_size, num_options)

        return scores


@HEADS.register_module()
class LanguageDecoderHead(nn.Module):

    def __init__(self, loss_cls: Dict):
        super().__init__()
        self.loss_cls = build_loss(loss_cls)

    def forward(self, encoder_output, data):
        return NotImplementedError

    def loss(self):
        pass

    def forward_train(self, *args, **kwargs):
        pass

    def forward_test(self, *args, **kwargs):
        pass


class LanguageDecoder(LanguageDecoderHead):

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.language_lstm = nn.LSTMCell(in_dim + kwargs['hidden_dim'], kwargs['hidden_dim'], bias=True)
        self.fc = weight_norm(nn.Linear(kwargs['hidden_dim'], out_dim))
        self.dropout = nn.Dropout(p=kwargs['dropout'])
        self.init_weights(kwargs['fc_bias_init'])

    def init_weights(self, fc_bias_init):
        self.fc.bias.data.fill_(fc_bias_init)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, weighted_attn):
        # Get LSTM state
        # state = registry.get(f'{weighted_attn.device}_lstm_state')
        state = None
        h1, c1 = state['td_hidden']
        h2, c2 = state['lm_hidden']

        # Language LSTM
        h2, c2 = self.language_lstm(torch.cat([weighted_attn, h1], dim=1), (h2, c2))
        predictions = self.fc(self.dropout(h2))

        # Update hidden state for t+1
        state['lm_hidden'] = (h2, c2)

        return predictions
