from mixk.utils.registry import Registry, build_from_cfg
from abc import ABCMeta, abstractmethod
import logging

FEATURE_READERS = Registry('FeatureReaders')


def build_feature_reader(cfg, default_args=None):
    return build_from_cfg(cfg=cfg, registry=FEATURE_READERS, default_args=default_args)


class FeatureReader(metaclass=ABCMeta):

    def __init__(self, dataset_type, feat_path=None, max_features=None):
        self.dataset_type = dataset_type
        self.feat_path = feat_path
        self.max_features = max_features
        self.feat_reader = None
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def read(self, img_annotation):
        pass
