import json
import math
from typing import Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..builder import EMBEDDING


class Identity(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


@EMBEDDING.register_module()
class TextEmbedding(nn.Module):

    def __init__(self,
                 emb_type='attention',
                 hidden_dim=1024,
                 num_layers=1,
                 conv1_out=512,
                 conv2_out=2,
                 dropout=0,
                 embedding_dim=300,
                 kernel_size=1,
                 padding=0,
                 **kwargs):
        super().__init__()
        self.model_data_dir = kwargs.get('model_data_dir', None)
        self.embedding_dim = kwargs.get('embedding_dim', None)

        # Update kwargs here
        if emb_type == 'identity':
            self.module = Identity()
            self.module.text_out_dim = self.embedding_dim
        # elif emb_type == 'vocab':
        #     self.module = VocabEmbedding(**kwargs)
        #     self.module.text_out_dim = self.embedding_dim
        # elif emb_type == 'projection':
        #     self.module = ProjectionEmbedding(**kwargs)
        #     self.module.text_out_dim = self.module.out_dim
        # elif emb_type == 'preextracted':
        #     self.module = PreExtractedEmbedding(**kwargs)
        elif emb_type == 'bilstm':
            self.module = BiLSTMTextEmbedding(hidden_dim, embedding_dim, num_layers, dropout, **kwargs)

        elif emb_type == 'visdial':
            ntoken = kwargs['ntoken']
            ninp = kwargs['ninp']
            emb_dim = ninp

            self.module = _netW(ntoken, ninp, dropout, emb_dim)

        elif emb_type == 'attention':
            self.module = AttentionTextEmbedding(hidden_dim, num_layers, conv1_out, conv2_out, dropout, embedding_dim,
                                                 kernel_size, padding, **kwargs)
            self.text_out_dim = self.module.text_out_dim

        elif emb_type == 'mcan':
            self.module = SAEmbedding(hidden_dim, embedding_dim, num_layers, dropout, **kwargs)

        elif emb_type == 'torch':
            vocab_size = kwargs['vocab_size']
            embedding_dim = kwargs['embedding_dim']
            self.module = nn.Embedding(vocab_size, embedding_dim)
            self.module.text_out_dim = self.embedding_dim
        else:
            raise NotImplementedError("Unknown question embedding '%s'" % emb_type)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _netW(nn.Module):

    def __init__(self, ntoken, ninp, dropout, emb_dim):
        super(_netW, self).__init__()
        # self.word_embed = nn.Embedding(ntoken+1, ninp).cuda()
        self.word_embed = nn.Embedding(ntoken + 1, ninp, padding_idx=0)
        # pdb.set_trace()

        self.vocab_size = ntoken

        self.init_pretrained_wemb(emb_dim)

        self.word_embed.weight.data.copy_(torch.from_numpy(self.pretrained_wemb))
        self.Linear = share_Linear(self.word_embed.weight).cuda()
        # self.init_weights()
        self.d = dropout

    """
    def init_weights(self):
        initrange = 0.1
        self.word_embed.weight.data.uniform_(-initrange, initrange)
    """

    def forward(self, input, format='index'):
        if format == 'onehot':
            out = F.dropout(self.Linear(input), self.d, training=self.training)
        elif format == 'index':
            out = F.dropout(self.word_embed(input), self.d, training=self.training)

        return out

    def init_pretrained_wemb(self, emb_dim):
        """From blog.keras.io Initialises words embeddings with pre-trained
        GLOVE embeddings."""

        f = json.load(open('/home/datasets/visual-diag/visdial_params.json', 'r'))
        self.itow = f['itow']

        embeddings_index = {}
        f = open('/home/datasets/visual-diag/glove.6B.300d.txt', 'r')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs
        f.close()

        embedding_mat = np.zeros((self.vocab_size + 1, emb_dim), dtype=np.float32)
        for i, word in self.itow.items():
            embedding_v = embeddings_index.get(word)
            if embedding_v is not None:
                embedding_mat[int(i)] = embedding_v

        self.pretrained_wemb = embedding_mat

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz,
                                        self.nhid).zero_()), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


@EMBEDDING.register_module()
class BiLSTMTextEmbedding(nn.Module):

    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        num_layers,
        dropout,
        bidirectional=False,
        rnn_type='GRU',
    ):
        super().__init__()
        self.text_out_dim = hidden_dim
        self.num_hid = hidden_dim
        self.bidirectional = bidirectional

        if rnn_type == 'LSTM':
            rnn_cls = nn.LSTM
        elif rnn_type == 'GRU':
            rnn_cls = nn.GRU

        self.recurrent_encoder = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.recurrent_encoder(x)
        # Return last state
        if self.bidirectional:
            return out[:, -1]

        forward_ = out[:, -1, :self.num_hid]
        backward = out[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        output, _ = self.recurrent_encoder(x)
        return output


@EMBEDDING.register_module()
class AttentionTextEmbedding(nn.Module):

    def __init__(self, hidden_dim, num_layers, conv1_out, conv2_out, dropout, embedding_dim, kernel_size, padding,
                 **kwargs):
        super().__init__()

        self.text_out_dim = hidden_dim * conv2_out

        bidirectional = kwargs.get('bidirectional', False)

        self.recurrent_unit = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        conv1_out = conv1_out
        conv2_out = conv2_out
        kernel_size = kernel_size
        padding = padding

        self.conv1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        self.recurrent_unit.flatten_parameters()
        # self.recurrent_unit.flatten_parameters()
        lstm_out, _ = self.recurrent_unit(x)  # N * T * hidden_dim
        lstm_drop = self.dropout(lstm_out)  # N * T * hidden_dim
        lstm_reshape = lstm_drop.permute(0, 2, 1)  # N * hidden_dim * T

        qatt_conv1 = self.conv1(lstm_reshape)  # N x conv1_out x T
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)  # N x conv2_out x T

        # Over last dim
        qtt_softmax = nn.functional.softmax(qatt_conv2, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature = torch.bmm(qtt_softmax, lstm_drop)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        return qtt_feature_concat


class share_Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True
    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`
    Attributes:
        weight: the learnable weights of the module of shape (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, weight):
        super(share_Linear, self).__init__()
        self.in_features = weight.size(0)
        self.out_features = weight.size(1)
        self.weight = weight.t()
        self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SAEmbedding(nn.Module):
    """Encoder block implementation in MCAN
    https://arxiv.org/abs/1906.10770."""

    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout, **kwargs):
        super().__init__()
        num_attn = kwargs['num_attn']
        # num_layers = kwargs["num_layers"]
        # dropout = kwargs.get("dropout", 0.1)
        num_attn_pool = kwargs.get('num_attn_pool', 1)
        num_feat = kwargs.get('num_feat', -1)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.self_attns = nn.ModuleList([SelfAttention(hidden_dim, num_attn, dropout) for _ in range(num_layers)])
        self.attn_pool = None
        self.num_feat = num_feat
        self.text_out_dim = hidden_dim
        if num_attn_pool > 0:
            self.attn_pool = AttnPool1d(hidden_dim, num_feat * num_attn_pool)
            self.text_out_dim = hidden_dim * num_attn_pool

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        out, (h, c) = self.lstm(x)
        for self_attn in self.self_attns:
            out = self_attn(out, mask)

        vec = h.transpose(0, 1).contiguous().view(b, 1, -1)
        if self.attn_pool:
            vec = self.attn_pool(out, out, mask).view(b, self.num_feat, -1)

        return out, vec


# Remove this and use torch.nn.MultiHeadAttention
class MovieMcanMultiHeadAttention(nn.Module):
    """Multi-Head Attention implementation from
    https://arxiv.org/abs/1706.03762 used for Movie+MCAN."""

    def __init__(self, dim: int, num_attn: int, dropout: float = 0.1):
        super().__init__()
        self.p_attn = None
        self.h = num_attn
        self.d_k = dim // num_attn
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)

    def qkv_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Type[nn.Dropout] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            from mixk.engine.organizer import get_masked_fill_value
            # scores.data.masked_fill_(mask.unsqueeze(1).unsqueeze(2), -1e9)
            scores.data.masked_fill_(mask.unsqueeze(1).unsqueeze(2), get_masked_fill_value())

        p_attn = nn.functional.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = q.size(0)

        q = self.linears[0](q).view(b, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linears[1](k).view(b, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linears[2](v).view(b, -1, self.h, self.d_k).transpose(1, 2)

        x, self.p_attn = self.qkv_attention(q, k, v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)

        return self.linears[-1](x)


class SelfAttention(nn.Module):

    def __init__(self, dim: int, num_attn: int, dropout: float):
        super().__init__()
        self.multi_head_attn = MovieMcanMultiHeadAttention(dim, num_attn, dropout=0.1)
        self.fcn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4 * dim, dim),
        )
        self.drop_mha = nn.Dropout(p=dropout)
        self.ln_mha = nn.LayerNorm(dim)
        self.drop_fcn = nn.Dropout(p=dropout)
        self.ln_fcn = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.ln_mha(x + self.drop_mha(self.multi_head_attn(x, x, x, x_mask)))
        x = self.ln_fcn(x + self.drop_fcn(self.fcn(x)))

        return x


class AttnPool1d(nn.Module):
    """An attention pooling layer that learns weights using an mlp."""

    def __init__(self, num_features: int, num_attn: int = 1, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(num_features // 2, num_attn),
        )
        self.p_attn = None
        self.num_attn = num_attn

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = query.size(0)
        score = self.linear(query).transpose(-2, -1)
        if mask is not None:
            from mixk.engine.organizer import get_masked_fill_value
            # score.data.masked_fill_(mask.unsqueeze(1), -1e9)
            score.data.masked_fill_(mask.unsqueeze(1), get_masked_fill_value())
        self.p_attn = nn.functional.softmax(score, dim=-1)

        return torch.matmul(self.p_attn, value).view(b, self.num_attn, -1)
