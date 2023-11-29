import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..builder import ENCODER

logger = logging.getLogger(__name__)

TEXT_BERT_HIDDEN_SIZE = 768


@ENCODER.register_module()
class VisDialLSTM(nn.Module):

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout):
        super().__init__()

        self.rnn_type = rnn_type
        self.model = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

    def forward(self, emb, hidden):
        return self.model(emb, hidden)


@ENCODER.register_module()
class VisDialPrincipleLSTM(nn.Module):

    def __init__(self, word_embedding_size, lstm_hidden_size, lstm_num_layers, dropout):
        super().__init__()

        self.hist_rnn = nn.LSTM(
            word_embedding_size,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.ques_rnn = nn.LSTM(
            word_embedding_size,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.cap_rnn = nn.LSTM(
            word_embedding_size,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)
        self.cap_rnn = DynamicRNN(self.cap_rnn)

    def forward(self, ques_embed, ques_len, cap_emb, cap_len, his_embed, hist_len):
        q_output, _ = self.ques_rnn(ques_embed, ques_len.view(-1))
        c_output, _ = self.cap_rnn(cap_emb, cap_len.view(-1))
        _, (his_feat, _) = self.hist_rnn(his_embed, hist_len.view(-1))
        return q_output, c_output, his_feat


@ENCODER.register_module()
class VisDialANSEncoder(nn.Module):
    """Given the real/wrong/fake answer, use a RNN (LSTM) to embed the answer.

    answer_encoder
    """

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout, vocab_size):
        super(VisDialANSEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ninp = ninp
        self.d = dropout
        self.vocab_size = vocab_size

        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        self.W1 = nn.Linear(self.nhid, self.nhid)
        self.W2 = nn.Linear(self.nhid, 1)
        self.fc = nn.Linear(nhid, ninp)

    def forward(self, input_feat, idx, hidden):

        output, _ = self.rnn(input_feat, hidden)
        mask = idx.data.eq(0)  # generate the mask
        mask[idx.data == self.vocab_size] = 1  # also set the last token to be 1
        if isinstance(input_feat, Variable):
            mask = Variable(mask, volatile=input_feat.volatile)

        # Doing self attention here.
        # pdb.set_trace()
        atten = self.W2(F.dropout(F.tanh(self.W1(output.view(-1, self.nhid))), self.d,
                                  training=self.training)).view(idx.size())
        atten.masked_fill_(mask, -99999)
        weight = F.softmax(atten.t()).view(-1, 1, idx.size(0))
        feat = torch.bmm(weight, output.transpose(0, 1)).view(-1, self.nhid)
        feat = F.dropout(feat, self.d, training=self.training)
        transform_output = F.tanh(self.fc(feat))

        return transform_output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz,
                                        self.nhid).zero_()), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class DynamicRNN(nn.Module):

    def __init__(self, rnn_model):
        super().__init__()
        self.rnn_model = rnn_model

    def forward(self, seq_input, seq_lens, initial_state=None):
        """A wrapper over pytorch's rnn to handle sequences of variable length.

        Arguments
        ---------
        seq_input : torch.Tensor
            Input sequence tensor (padded) for RNN model.
            Shape: (batch_size, max_sequence_length, embed_size)
        seq_lens : torch.LongTensor
            Length of sequences (b, )
        initial_state : torch.Tensor
            Initial (hidden, cell) states of RNN model.

        Returns
        -------
            Single tensor of shape (batch_size, rnn_hidden_size) corresponding
            to the outputs of the RNN model at the last time step of each input
            sequence.
        """
        max_sequence_length = seq_input.size(1)
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
        sorted_seq_input = seq_input.index_select(0, fwd_order)
        packed_seq_input = pack_padded_sequence(sorted_seq_input, lengths=sorted_len, batch_first=True)

        if initial_state is not None:
            hx = initial_state
            assert hx[0].size(0) == self.rnn_model.num_layers
        else:
            sorted_hx = None

        self.rnn_model.flatten_parameters()
        outputs, (h_n, c_n) = self.rnn_model(packed_seq_input, sorted_hx)

        # pick hidden and cell states of last layer
        h_n = h_n[-1].index_select(dim=0, index=bwd_order)
        c_n = c_n[-1].index_select(dim=0, index=bwd_order)

        outputs = pad_packed_sequence(
            outputs, batch_first=True, total_length=max_sequence_length)[0].index_select(
                dim=0, index=bwd_order)
        # outputs = pad_packed_sequence(
        #     outputs, batch_first=True)[0].index_select(dim=0, index=bwd_order)

        return outputs, (h_n, c_n)

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(lens.contiguous().view(-1), 0, descending=True)
        _, bwd_order = torch.sort(fwd_order)
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order
