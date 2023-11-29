from collections import defaultdict
import os
import torch
from mixk.utils.config import get_mixk_cache_dir, get_mixk_root
from mixk.utils.third_party_libs import PathManager
from mixk.utils.distributed_info import is_main_process, synchronize
from torchtext import vocab

from ..builder import EMBEDDING


class BaseVocab:
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'
    UNK_TOKEN = '<unk>'

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, vocab_file=None, embedding_dim=300, data_dir=None, *args, **kwargs):
        """Vocab class to be used when you want to train word embeddings from
        scratch based on a custom vocab. This will initialize the random
        vectors for the vocabulary you pass. Get the vectors using
        `get_vectors` function. This will also create random embeddings for.

        some predefined words like PAD - <pad>, SOS - <s>, EOS - </s>,
        UNK - <unk>.

        Parameters
        ----------
        vocab_file : str
            Path of the vocabulary file containing one word per line
        embedding_dim : int
            Size of the embedding
        """
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
            if not os.path.isabs(vocab_file) and data_dir is not None:
                mixk_root = get_mixk_root()
                vocab_file = os.path.join(mixk_root, data_dir, vocab_file)
            if not PathManager.exists(vocab_file):
                raise RuntimeError('Vocab not found at ' + vocab_file)

            with PathManager.open(vocab_file, 'r') as f:
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

        self.vectors = torch.FloatTensor(self.get_size(), embedding_dim)

    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi

    def get_size(self):
        return len(self.itos)

    def get_pad_index(self):
        return self.PAD_INDEX

    def get_pad_token(self):
        return self.PAD_TOKEN

    def get_start_index(self):
        return self.SOS_INDEX

    def get_start_token(self):
        return self.SOS_TOKEN

    def get_end_index(self):
        return self.EOS_INDEX

    def get_end_token(self):
        return self.EOS_TOKEN

    def get_unk_index(self):
        return self.UNK_INDEX

    def get_unk_token(self):
        return self.UNK_TOKEN

    def get_vectors(self):
        return getattr(self, 'vectors', None)

    def get_embedding(self, cls, **embedding_kwargs):
        vector_dim = len(self.vectors[0])
        embedding_kwargs['vocab_size'] = self.get_size()

        embedding_dim = embedding_kwargs['embedding_dim']
        embedding_kwargs['embedding_dim'] = vector_dim

        embedding = None

        if cls == torch.nn.Embedding:
            embedding = torch.nn.Embedding(self.get_size(), vector_dim)
        else:
            embedding = cls(**embedding_kwargs)

        if hasattr(embedding, 'embedding'):
            embedding.embedding = torch.nn.Embedding.from_pretrained(self.vectors, freeze=False)
        else:
            embedding = torch.nn.Embedding.from_pretrained(self.vectors, freeze=False)

        if vector_dim == embedding_dim:
            return embedding
        else:
            return torch.nn.Sequential([embedding, torch.nn.Linear(vector_dim, embedding_dim)])


@EMBEDDING.register_module()
class IntersectedVocab(BaseVocab):

    def __init__(self, vocab_file, embedding_name, *args, **kwargs):
        """Use this vocab class when you have a custom vocabulary class but you
        want to use pretrained embedding vectos for it. This will only load the
        vectors which intersect with your vocabulary. Use the embedding_name
        specified in torchtext's pretrained aliases:

        ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d',
         'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d',
         'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
         'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d',
         'glove.6B.200d', 'glove.6B.300d']

        Parameters
        ----------
        vocab_file : str
            Vocabulary file containing list of words with one word per line
            which will be used to collect vectors
        embedding_name : str
            Embedding name picked up from the list of the pretrained aliases
            mentioned above
        """
        super().__init__(vocab_file, *args, **kwargs)

        self.type = 'intersected'

        name = embedding_name.split('.')[0]
        dim = embedding_name.split('.')[2][:-1]
        middle = embedding_name.split('.')[1]

        class_name = 'GloVe'
        params = [middle]

        if name == 'glove':
            params.append(int(dim))

        vector_cache = get_mixk_cache_dir()

        # First test loading the vectors in master so that everybody doesn't
        # download it in case it doesn't exist
        if is_main_process():
            vocab.pretrained_aliases[embedding_name](cache=vector_cache)
        synchronize()

        embedding = getattr(vocab, class_name)(*params, cache=vector_cache)

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

    def get_embedding_dim(self):
        return self.embedding_dim
