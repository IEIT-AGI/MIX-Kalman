import torch
from ..builder import PROCESSOR
from .baseprocessor import BaseProcessor
from ..builder import build_vocab, build_preprocessor


@PROCESSOR.register_module()
class Processor(BaseProcessor):
    """Tokenizes a sentence and processes it.

    Attributes:
        tokenizer (function): Type of tokenizer to be used.
    """

    def __init__(self, *args, **kwargs):
        from mixk.utils.third_party_libs import tokenize

        self.tokenizer = tokenize

    def __call__(self, item, *args, **kwargs):
        return {'text': self.tokenizer(item['text'], *args, **kwargs)}


@PROCESSOR.register_module()
class VocabProcessor(BaseProcessor):
    """Use VocabProcessor when you have vocab file and you want to process
    words to indices. Expects UNK token as "<unk>" and pads sentences using
    "<pad>" token. Config parameters can have ``preprocessor`` property which
    is used to preprocess the item passed and ``max_length`` property which
    points to maximum length of the sentence/tokens which can be convert to
    indices. If the length is smaller, the sentence will be padded. Parameters
    for "vocab" are necessary to be passed.

    **Key**: vocab

    Example Config::

        task_attributes:
            vqa:
                vqa2:
                    processors:
                      text_processor:
                        type: vocab
                        params:
                          max_length: 14
                          vocab:
                            type: intersected
                            embedding_name: glove.6B.300d
                            vocab_file: vocabs/vocabulary_100k.txt

    Args:
        config (DictConfig): node containing configuration parameters of
                             the processor

    Attributes:
        vocab (Vocab): Vocab class object which is abstraction over the vocab
                       file passed.
    """

    MAX_LENGTH_DEFAULT = 50
    PAD_TOKEN = '<pad>'
    PAD_INDEX = 0

    def __init__(self,
                 vocab=dict(
                     type='IntersectedVocab',
                     vocab_file='textvqa/defaults/extras/vocabs/vocabulary_100k.txt',
                     embedding_name='glove.6B.300d'),
                 preprocessor=dict(type='SimpleSentenceProcessor'),
                 *args,
                 **kwargs):

        # self.vocab = Vocab(*args, **config.vocab, **kwargs)
        self.vocab = build_vocab(vocab)
        self.max_length = self.MAX_LENGTH_DEFAULT
        self.preprocessor = build_preprocessor(preprocessor)

        # self._init_extras(config)

    # def _init_extras(self, config, *args, **kwargs):
    #     self.writer = registry.get("writer")
    #     self.preprocessor = None
    #
    #     if hasattr(config, "max_length"):
    #         self.max_length = config.max_length
    #     else:
    #         warnings.warn(
    #             "No 'max_length' parameter in Processor's "
    #             "configuration. Setting to {}.".format(self.MAX_LENGTH_DEFAULT)
    #         )
    #         self.max_length = self.MAX_LENGTH_DEFAULT
    #
    #     if "preprocessor" in config:
    #         self.preprocessor = Processor(config.preprocessor, *args, **kwargs)
    #
    #         if self.preprocessor is None:
    #             raise ValueError(
    #                 f"No text processor named {config.preprocessor} is defined."
    #             )

    def __call__(self, item):
        """Call requires item to have either "tokens" attribute or either
        "text" attribute. If "text" is present, it will tokenized using the
        preprocessor.

        Args:
            item (Dict): Dict containing the "text" or "tokens".

        Returns:
            Dict: Dict containing indices in "text" key, "tokens" in "tokens"
                  key and "length" of the string in "length" key.
        """
        indices = None
        if not isinstance(item, dict):
            raise TypeError('Argument passed to the processor must be '
                            "a dict with either 'text' or 'tokens' as "
                            'keys')
        if 'tokens' in item:
            tokens = item['tokens']
            indices = self._map_strings_to_indices(item['tokens'])
        elif 'text' in item:
            if self.preprocessor is None:
                raise AssertionError('If tokens are not provided, a text ' 'processor must be defined in the config')

            tokens = self.preprocessor({'text': item['text']})['text']
            indices = self._map_strings_to_indices(tokens)
        else:
            raise AssertionError("A dict with either 'text' or 'tokens' keys " 'must be passed to the processor')

        tokens, length = self._pad_tokens(tokens)

        return {'text': indices, 'tokens': tokens, 'length': length}

    def _pad_tokens(self, tokens):
        padded_tokens = [self.PAD_TOKEN] * self.max_length
        token_length = min(len(tokens), self.max_length)
        padded_tokens[:token_length] = tokens[:token_length]
        token_length = torch.tensor(token_length, dtype=torch.long)
        return padded_tokens, token_length

    def get_pad_index(self):
        """Get index of padding <pad> token in vocabulary.

        Returns:
            int: index of the padding token.
        """
        return self.vocab.get_pad_index()

    def get_vocab_size(self):
        """Get size of the vocabulary.

        Returns:
            int: size of the vocabulary.
        """
        return self.vocab.get_size()

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.zeros(self.max_length, dtype=torch.long)
        output.fill_(self.vocab.get_pad_index())

        for idx, token in enumerate(tokens):
            output[idx] = self.vocab.stoi[token]

        return output
