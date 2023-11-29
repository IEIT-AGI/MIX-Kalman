from pathlib import Path
from .feature_reader import FEATURE_READERS, FeatureReader
import torch
from typing import Dict


@FEATURE_READERS.register_module()
class PthFeatureReader(FeatureReader):

    def __init__(self, dataset_type, feat_path=None, max_features=None):
        super().__init__(dataset_type, feat_path=feat_path, max_features=max_features)
        self.img_features = self._load_img_features()

    @staticmethod
    def file_name_to_path(file_dir: str, file_format: str) -> Dict:
        img_feats = {}
        feat_files = Path(file_dir).glob(pattern=file_format)
        for file in feat_files:
            file_name = str(file.name).split('.')[0]
            img_feats[file_name] = str(file)
        return img_feats

    def _load_img_features(self):
        assert Path(self.feat_path).is_dir()
        img_features = self.file_name_to_path(file_dir=self.feat_path, file_format='*.pt')
        if len(img_features) == 0:
            img_features = self.file_name_to_path(file_dir=self.feat_path, file_format='*.pth')

        self.logger.info(f'loading {self.feat_path} features {len(img_features)}')
        return img_features

    def read(self, img_annotation):
        return self._get_img_feat(img_annotation)

    def _get_img_feat(self, img_annotation):
        if self.dataset_type == 'VQAReader':
            key = Path(img_annotation['feature_path']).name.split('.')[0]
            img_feat_path = self.img_features.get(key)
            img_feat = torch.load(img_feat_path)
            return {'features': img_feat[0]}
        else:
            self.logger.error('Please add the image feature reader of the corresponding dataset type!')
            raise TypeError(f'{self.dataset_type} is an unsupported data type')
