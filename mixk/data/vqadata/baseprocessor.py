class BaseProcessor:
    """Every processor in mixk needs to inherit this class for compatibility
    with mixk. End user mainly needs to implement ``__call__`` function.

    Args:
        config (DictConfig): Config for this processor, containing `type` and
                             `params` attributes if available.
    """

    def __init__(self, config, *args, **kwargs):
        return

    def __call__(self, item, *args, **kwargs):
        """Main function of the processor. Takes in a dict and returns back a
        dict.

        Args:
            item (Dict): Some item that needs to be processed.

        Returns:
            Dict: Processed dict.
        """
        return item


class Processor:
    """Wrapper class used by mixk to initialized processor based on their
    ``type`` as passed in configuration. It retrieves the processor class
    registered in registry corresponding to the ``type`` key and initializes
    with ``params`` passed in configuration. All functions and attributes of
    the processor initialized are directly available via this class.

    Args:
        config (DictConfig): DictConfig containing ``type`` of the processor to
                             be initialized and ``params`` of that procesor.
    """

    def __init__(self, config, *args, **kwargs):
        # self.writer = registry.get('writer')
        self.writer = None

        if not hasattr(config, 'type'):
            raise AttributeError("Config must have 'type' attribute to specify type of processor")

        # processor_class = registry.get_processor_class(config.type)
        processor_class = None

        params = {}
        if not hasattr(config, 'params'):
            self.writer.write("Config doesn't have 'params' attribute to "
                              'specify parameters of the processor '
                              'of type {}. Setting to default {{}}'.format(config.type))
        else:
            params = config.params

        self.processor = processor_class(params, *args, **kwargs)

        self._dir_representation = dir(self)

    def __call__(self, item, *args, **kwargs):
        return self.processor(item, *args, **kwargs)

    def __getattr__(self, name):
        if '_dir_representation' in self.__dict__ and name in self._dir_representation:
            return getattr(self, name)
        elif 'processor' in self.__dict__ and hasattr(self.processor, name):
            return getattr(self.processor, name)
        else:
            raise AttributeError(name)


class Vocab:

    def __init__(self, *args, **params):
        vocab_type = params.get('type', 'pretrained')
        # Stores final parameters extracted from vocab_params

        if vocab_type == 'random':
            if params['vocab_file'] is None:
                raise ValueError('No vocab path passed for vocab')

            # self.vocab = BaseVocab(*args, **params)
            self.vocab = None  # unused

        # elif vocab_type == 'custom':
        #     if params['vocab_file'] is None or params['embedding_file'] is None:
        #         raise ValueError('No vocab path or embedding_file passed for vocab')
        #     self.vocab = CustomVocab(*args, **params)
        #
        # elif vocab_type == 'pretrained':
        #     self.vocab = PretrainedVocab(*args, **params)
        #
        # elif vocab_type == 'intersected':
        #     if params['vocab_file'] is None or params['embedding_name'] is None:
        #         raise ValueError('No vocab path or embedding_name passed for vocab')
        #
        #     self.vocab = IntersectedVocab(*args, **params)
        #
        # elif vocab_type == 'extracted':
        #     if params['base_path'] is None or params['embedding_dim'] is None:
        #         raise ValueError('No base_path or embedding_dim passed for vocab')
        #     self.vocab = ExtractedVocab(*args, **params)
        #
        # elif vocab_type == 'model':
        #     if params['name'] is None or params['model_file'] is None:
        #         raise ValueError('No name or model_file passed for vocab')
        #     if params['name'] == 'fasttext':
        #         self.vocab = ModelVocab(*args, **params)
        else:
            raise ValueError('Unknown vocab type: %s' % vocab_type)

        self._dir_representation = dir(self)

    def __call__(self, *args, **kwargs):
        return self.vocab(*args, **kwargs)

    def __getattr__(self, name):
        if '_dir_representation' in self.__dict__ and name in self._dir_representation:
            return getattr(self, name)
        elif 'vocab' in self.__dict__ and hasattr(self.vocab, name):
            return getattr(self.vocab, name)
        else:
            type_vocab = 'Vocab'
            if 'vocab' in self.__dict__:
                type_vocab = type(self.vocab)

            raise AttributeError(f'{type_vocab} vocab type has no attribute {name}.')
