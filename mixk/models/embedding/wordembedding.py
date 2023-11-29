from collections import defaultdict

import torch
import torch.nn as nn
from torchtext.vocab import GloVe

from ..builder import EMBEDDING


@EMBEDDING.register_module()
class WordEmbedding(nn.Module):

    def __init__(self, vocab_file=None, embedding_dim=300, **kwargs):

        super().__init__()
        """Vocab class to be used when you want to train word embeddings from
        scratch based on a custom vocab. This will initialize the random
        vectors for the vocabulary you pass. Get the vectors using
        `get_vectors` function. This will also create random embeddings for
        some predefined words like PAD - <pad>, SOS - <s>, EOS - </s>,
        UNK - <unk>.

        Parameters
        ----------
        vocab_file : str
            Path of the vocabulary file containing one word per line
        embedding_dim : int
            Size of the embedding

        """
        self.PAD_TOKEN = '<pad>'
        self.SOS_TOKEN = '<s>'
        self.EOS_TOKEN = '</s>'
        self.UNK_TOKEN = '<unk>'

        self.PAD_INDEX = 0
        self.SOS_INDEX = 1
        self.EOS_INDEX = 2
        self.UNK_INDEX = 3

        self.type = 'base'
        self.word_dict = {}
        self.itos = {}

        self.itos[self.PAD_INDEX] = self.PAD_TOKEN
        self.itos[self.SOS_INDEX] = self.SOS_TOKEN
        self.itos[self.EOS_INDEX] = self.EOS_TOKEN
        self.itos[self.UNK_INDEX] = self.UNK_TOKEN

        self.word_dict[self.SOS_TOKEN] = self.SOS_INDEX
        self.word_dict[self.EOS_TOKEN] = self.EOS_INDEX
        self.word_dict[self.PAD_TOKEN] = self.PAD_INDEX
        self.word_dict[self.UNK_TOKEN] = self.UNK_INDEX

        index = len(self.itos.keys())

        self.total_predefined = len(self.itos.keys())

        if vocab_file is not None:
            with open(vocab_file, 'r') as f:
                for line in f:
                    self.itos[index] = line.strip()
                    self.word_dict[line.strip()] = index
                    index += 1

        self.word_dict[self.SOS_TOKEN] = self.SOS_INDEX
        self.word_dict[self.EOS_TOKEN] = self.EOS_INDEX
        self.word_dict[self.PAD_TOKEN] = self.PAD_INDEX
        self.word_dict[self.UNK_TOKEN] = self.UNK_INDEX
        # Return unk index by default
        self.stoi = defaultdict(self.get_unk_index)
        self.stoi.update(self.word_dict)

        glove_params = kwargs['glove_params']
        embedding = self.init_GloVe(name=glove_params.name, dim=glove_params.dim, cache=glove_params.get('cache', None))

        self.vectors = torch.empty((self.get_size(), len(embedding.vectors[0])), dtype=torch.float)

        self.embedding_dim = len(embedding.vectors[0])

        for i in range(0, 4):
            self.vectors[i] = torch.ones_like(self.vectors[i]) * 0.1 * i

        for i in range(4, self.get_size()):
            word = self.itos[i]
            embedding_index = embedding.stoi.get(word, None)

            if embedding_index is None:
                self.vectors[i] = self.vectors[self.UNK_INDEX]
            else:
                self.vectors[i] = embedding.vectors[embedding_index]

        self.embedding = torch.nn.Embedding(self.get_size(), embedding_dim)
        self.embedding = torch.nn.Embedding.from_pretrained(self.vectors, freeze=False)

        # self.model = torch.nn.Sequential(
        #         [self.embedding, torch.nn.Linear(vector_dim, embedding_dim)]
        #     )

    def get_size(self):
        return len(self.itos)

    def get_unk_index(self):
        return self.UNK_INDEX

    def forward(self, x):
        return self.embedding(x)

    def init_GloVe(self, name, dim, cache=None):
        return GloVe(name, dim, cache=cache)
