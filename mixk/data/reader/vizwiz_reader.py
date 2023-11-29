import numpy as np
import torch

from ..utils.stream import ItemFeature
from .base_reader import mixkDataReader


class VizWizReader(mixkDataReader):

    def __init__(self, cfg):
        super().__init__(cfg)

    def __len__(self):
        return len(self.mix_annotations)

    def __getitem__(self, item):
        annotation = self.mix_annotations[item]
        split = self.item_splits[item]
        item_feature = ItemFeature(annotation)
        item_feature.error = False

        item_feature.tokens = annotation.get('question_tokens')
        item_feature.img_id = annotation.get('image_id')
        if self.default_feature:
            feature_info = self.get_featureinfo_from_txns(self.feature_txns, annotation.get('image_id'))
            if feature_info is None:
                item_feature.error = True
                item_feature.feature = np.random.random((100, 2048))
                return item_feature
            for k, v in feature_info.items():
                item_feature[k] = item_feature.get(k, v)
            return item_feature
        feature_path = self.features_pathes[split + '_' + str(item_feature.img_id)]
        item_feature.feature = torch.load(feature_path)[0]
        return item_feature
