import os
import warnings
from torch import distributed as dist
import torch
import numpy as np
from ..builder import PROCESSOR
from .vocabprocessor import VocabProcessor
from mixk.utils.third_party_libs import PathManager
from mixk.utils.config import get_mixk_cache_dir


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_nccl_available():
        return
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


class WordToVectorDict:

    def __init__(self, model):
        self.model = model

    def __getitem__(self, word):
        # Check if mean for word split needs to be done here
        return np.mean([self.model.get_word_vector(w) for w in word.split(' ')], axis=0)


@PROCESSOR.register_module()
class FastTextProcessor(VocabProcessor):
    """FastText processor, similar to GloVe processor but returns FastText
    vectors.

    Args:
        config (DictConfig): Configuration values for the processor.
    """

    def __init__(self, max_length, model_file, *args, **kwargs):
        # self._init_extras(config)
        # self.config = config
        self.max_length = max_length
        self.model_file = model_file
        # self._download_initially = config.get('download_initially', True)
        self._download_initially = False
        self._already_downloaded = False
        self._already_loaded = False

        if self._download_initially:
            self._try_download()

    def _try_download(self):
        # _is_master = is_master()
        _is_master = False

        if self._already_downloaded:
            return

        needs_download = False

        if not hasattr(self.config, 'model_file'):
            if _is_master:
                warnings.warn("'model_file' key is required but missing " "from FastTextProcessor's config.")
            needs_download = True

        model_file = self.model_file
        # If model_file is already an existing path don't join to cache dir
        if not PathManager.exists(model_file):
            model_file = os.path.join(get_mixk_cache_dir(), model_file)

        if not PathManager.exists(model_file):
            if _is_master:
                warnings.warn(f'No model file present at {model_file}.')
            needs_download = True

        if needs_download:
            self.writer.write('Downloading FastText bin', 'info')
            model_file = self._download_model()

        self.model_file = model_file
        self._already_downloaded = True
        synchronize()

    def _download_model(self):
        from mixk.utils.distributed_info import is_main_process
        from mixk.utils.config import get_mixk_cache_dir
        _is_master = is_main_process()
        model_file_path = os.path.join(get_mixk_cache_dir(), 'wiki.en.bin')

        if not _is_master:
            return model_file_path

        if PathManager.exists(model_file_path):
            self.writer.write(f'Vectors already present at {model_file_path}.', 'info')
            return model_file_path

        import requests
        from tqdm import tqdm

        FASTTEXT_WIKI_URL = 'https://dl.fbaipublicfiles.com/pythia/pretrained_models/fasttext/wiki.en.bin'
        PathManager.mkdirs(os.path.dirname(model_file_path))
        response = requests.get(FASTTEXT_WIKI_URL, stream=True)

        with PathManager.open(model_file_path, 'wb') as f:
            pbar = tqdm(
                total=int(response.headers['Content-Length']) / 4096,
                miniters=50,
                disable=not _is_master,
            )

            idx = 0
            for data in response.iter_content(chunk_size=4096):
                if data:
                    if idx % 50 == 0:
                        pbar.update(len(data))
                    f.write(data)
                    idx += 1

            pbar.close()

        self.writer.write(f'fastText bin downloaded at {model_file_path}.', 'info')

        return model_file_path

    def _load_fasttext_model(self, model_file):
        if self._already_loaded:
            return

        from fasttext import load_model

        # self.writer.write('Loading fasttext model now from %s' % model_file)

        self.model = load_model(model_file)
        # String to Vector
        self.stov = WordToVectorDict(self.model)
        # self.writer.write('Finished loading fasttext model')

        self._already_loaded = True

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.full(
            (self.max_length, self.model.get_dimension()),
            fill_value=self.PAD_INDEX,
            dtype=torch.float,
        )

        for idx, token in enumerate(tokens):
            output[idx] = torch.from_numpy(self.stov[token])

        return output

    def __call__(self, item):
        self._load_fasttext_model(self.model_file)
        return super().__call__(item)
