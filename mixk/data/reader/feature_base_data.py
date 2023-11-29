from typing import List, Dict
from .annotation_reader import AnnotationBaseData
from torch.utils.data import Dataset
from pathlib import Path
from collections import Counter
from mixk.utils.config import mixkEasyDict
from .feature_reader import build_feature_reader, LMDBFeatureReader, PthFeatureReader
import logging


class FeatureBaseData(Dataset):
    feature_support_type = ['lmdb', 'pth', 'pt', 'pkl', 'csv']

    def __init__(self, cfg, splits, feature_cfg, annotation_bd, *args, **kwargs):
        self.logger = logging.getLogger(__name__)

        self.cfg = cfg
        self.splits = splits
        self.dataset_type = self.cfg.type
        self.feature_cfg = feature_cfg
        self.is_specify_feat_reader = hasattr(cfg, 'image_feature_reader')
        self.annotation_obj = annotation_bd
        self.feature_reader_objs = []
        self._add_feature_reader()

    def _add_feature_reader(self):
        feat_paths = set(self.feature_cfg[d] for d in self.splits)
        for feat_path in feat_paths:
            if self.is_specify_feat_reader:
                feat_reader_obj = build_feature_reader(
                    self.cfg.image_feature_reader, default_args={'features_path': feat_path})
            else:
                feat_reader_obj = self._auto_match_feat_reader(feat_path=feat_path)
            self.feature_reader_objs.append(feat_reader_obj)

    def _auto_match_feat_reader(self, feat_path: str):
        feature_format = self.get_file_format_in_folder(folder=feat_path)
        assert feature_format in self.feature_support_type

        max_features = self.cfg.get('max_features', None)
        if feature_format == 'lmdb':
            obj = LMDBFeatureReader(dataset_type=self.dataset_type, feat_path=feat_path, max_features=max_features)
        elif feature_format in ['pth', 'pt']:
            obj = PthFeatureReader(dataset_type=self.dataset_type, feat_path=feat_path, max_features=max_features)
        elif feature_format in ['pkl', 'csv']:
            self.logger.warning(f'please add support for {feature_format} format')
            raise TypeError('no support feature format')

        return obj

    @staticmethod
    def get_file_format_in_folder(folder: str) -> str:
        feature_format = folder.split('.')[-1]
        if feature_format != folder:
            return feature_format  # such as trainval2014.lmdb
        else:
            assert Path(folder).is_dir()
            feat_files = Path(folder).glob(pattern='*')
            limits = 500
            file_formats = Counter()
            for idx, file in enumerate(feat_files):
                if limits < idx:
                    break
                suffix = str(file.name).split('.')[-1]
                file_formats[suffix] += 1

            most_file_format = file_formats.most_common(1)
            return most_file_format[0][0]

    def __len__(self):
        return len(self.annotation_obj)

    def __getitem__(self, idx):
        img_annotation = self.annotation_obj[idx]
        return self.get_img_feat(img_annotation=img_annotation)

    def get_img_feat(self, img_annotation):  # only get one feature
        feature = None
        for feat_reader_obj in self.feature_reader_objs:
            feat = feat_reader_obj.read(img_annotation)
            if feat is not None:
                feature = feat
                break

        return feature


def build_features(cfg: mixkEasyDict, splits: List, feature_cfg: Dict,
                   annotation_bd: AnnotationBaseData) -> FeatureBaseData:
    return FeatureBaseData(cfg=cfg, splits=splits, feature_cfg=feature_cfg, annotation_bd=annotation_bd)
