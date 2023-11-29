from torch.utils.data import Dataset
import numpy as np
import json
import _pickle as cPickle
import logging


class AnnotationBaseData(Dataset):
    support_file_format = ['json', 'npy', 'pkl']

    def __init__(self, splits, annotation_cfg, *args, **kwargs):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.splits = splits
        self.annotation_cfg = annotation_cfg

        self.annotations = []
        self.item_splits = []

        self._load()

    def _load(self):
        for data in self.splits:
            file = self.annotation_cfg[data]
            file_format = file.split('.')[-1]

            assert file_format in self.support_file_format, f'unknown file format :{file_format}'

            try:
                load_fun = getattr(self, '_load_by_' + file_format)
                annotation = load_fun(path=file)
            except KeyError:
                self.logger.info(f'The expected type are {self.support_file_format},but got type is {file_format}')
                raise KeyError
            else:
                self.annotations.extend(annotation)
                self.item_splits.extend([data] * len(annotation))

    @staticmethod
    def _load_by_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _load_by_npy(path):
        data = np.load(file=path, allow_pickle=True)
        if 'version' in data[0] or 'image_id' not in data[0]:
            return data[1:]
        else:
            return data

    @staticmethod
    def _load_by_pkl(path):
        with open(path, 'rb') as f:
            return cPickle.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations[idx]
        return data


def build_annotations(splits, annotation_cfg):
    annotation_bd = AnnotationBaseData(splits=splits, annotation_cfg=annotation_cfg)
    item_splits = annotation_bd.item_splits
    return annotation_bd, item_splits
