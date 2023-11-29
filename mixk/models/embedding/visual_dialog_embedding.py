from ..builder import EMBEDDING
from torch import nn
import torch
import math


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


@EMBEDDING.register_module()
class VisDiaBertEmbeddingsDialog(nn.Module):

    def __init__(self, config):
        super(VisDiaBertEmbeddingsDialog, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        max_seq_len = 256
        d_model = config.hidden_size
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        self.pe = pe.cuda()
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # add support for additional segment embeddings. Supporting 10 additional embedding as of now
        self.token_type_embeddings_extension = nn.Embedding(10, config.hidden_size)
        # adding specialized embeddings for sep tokens
        self.sep_embeddings = nn.Embedding(50, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, input_ids, sep_indices=None, sep_len=None, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        token_type_ids_extension = token_type_ids - self.config.type_vocab_size
        token_type_ids_extension_mask = (token_type_ids_extension >= 0).float()
        token_type_ids_extension = (token_type_ids_extension.float() * token_type_ids_extension_mask).long()

        token_type_ids_mask = (token_type_ids < self.config.type_vocab_size).float()
        assert torch.sum(token_type_ids_extension_mask + token_type_ids_mask) == \
               torch.numel(token_type_ids) == torch.numel(token_type_ids_mask)
        token_type_ids = (token_type_ids.float() * token_type_ids_mask).long()

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        token_type_embeddings_extension = self.token_type_embeddings_extension(token_type_ids_extension)

        token_type_embeddings = (token_type_embeddings * token_type_ids_mask.unsqueeze(-1)) + \
                                (token_type_embeddings_extension * token_type_ids_extension_mask.unsqueeze(-1))

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


@EMBEDDING.register_module()
class VisDiaBertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and
    token_type embeddings."""

    def __init__(self, config):
        super(VisDiaBertImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)
        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
