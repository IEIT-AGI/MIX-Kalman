import numpy as np
from ..utils.stream import ItemFeature
from .base_reader import mixkDataReader
from mixk.utils.common_function import update_d1_with_d2


class VQAReader(mixkDataReader):

    def __init__(self, cfg):
        super().__init__(cfg)

    def __len__(self):
        return len(self.mix_annotations)

    def __getitem__(self, idx):
        annotation = self.mix_annotations[idx]

        item_feature = ItemFeature(annotation)
        item_feature.error = False
        item_feature.tokens = annotation['question_tokens']
        item_feature.img_id = annotation['image_id']

        feature = self.feature_obj[idx]
        global_feature = None

        if self.global_feature_obj:
            global_feature = self.global_feature_obj[idx]
            global_feature['global_features'] = global_feature.pop('features')
            global_feature['global_feature_path'] = global_feature.pop('feature_path')

        if self.is_global:
            if None in [feature, global_feature]:
                item_feature.error = True
                item_feature.features = np.random.random((100, 2048))
                item_feature.global_feature = np.random.random((100, 2048))
                return item_feature
        else:
            if feature is None:
                item_feature.error = True
                item_feature.features = np.random.random((100, 2048))
                return item_feature

        update_d1_with_d2(d1=item_feature, d2=feature)
        if self.is_global:
            update_d1_with_d2(d1=item_feature, d2=global_feature)

        return item_feature
