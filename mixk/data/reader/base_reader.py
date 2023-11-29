from mixk.utils.config import mixkEasyDict
from .annotation_reader import build_annotations
from .feature_base_data import build_features


class BaseDataReader:

    def __init__(self, cfg):
        # load config: path, name, split ...
        pass

    def load(self):
        pass

    def __getitem__(self, item):
        pass

    def deduplist(self, l):
        return list(set(l))


class mixkDataReader:

    def __init__(self, cfg: mixkEasyDict):
        assert cfg.datasets
        splits = cfg.datasets
        self.splits = [splits] if isinstance(splits, str) else splits
        self.cfg = cfg

        self.use_global_feat = cfg.get('mix_global_features', None)
        self.use_ocr_feat = cfg.get('mix_ocr_features', None)
        self.is_global = cfg.get('is_global', False)

        self.feature_obj = None
        self.global_feature_obj = None
        self.ocr_feature_obj = None

        self._add_annotations()
        self._add_features()

    def _add_annotations(self):
        self.annotations_obj, self.item_splits = build_annotations(
            splits=self.splits, annotation_cfg=self.cfg.mix_annotations)
        self.mix_annotations = self.annotations_obj

    def _add_features(self):
        self.feature_obj = build_features(
            cfg=self.cfg, splits=self.splits, feature_cfg=self.cfg.mix_features, annotation_bd=self.annotations_obj)

        if self.use_global_feat and self.is_global:
            self.global_feature_obj = build_features(
                cfg=self.cfg,
                splits=self.splits,
                feature_cfg=self.cfg.mix_global_features,
                annotation_bd=self.annotations_obj)
        if self.use_ocr_feat:
            self.ocr_feature_obj = build_features(
                cfg=self.cfg,
                splits=self.splits,
                feature_cfg=self.cfg.mix_ocr_features,
                annotation_bd=self.annotations_obj)

        self.feature_txns = self.feature_obj
        self.feature_global_txns = getattr(self, 'global_feature_obj', None)
        self.feature_ocr_txns = getattr(self, 'ocr_feature_obj', None)
