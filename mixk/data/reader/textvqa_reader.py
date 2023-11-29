from ..utils.stream import ItemFeature
from .base_reader import mixkDataReader
from mixk.utils.common_function import update_d1_with_d2


class TextVQAReader(mixkDataReader):

    def __init__(self, cfg):
        super().__init__(cfg)
        # assert self.default_feature, ('Not support non-default features now.')

    def __len__(self):
        return len(self.mix_annotations)

    def __getitem__(self, idx):
        annotation = self.mix_annotations[idx]
        feature = self.feature_obj[idx]
        global_feature, ocr_feature = {}, {}

        item_feature = ItemFeature(annotation)
        item_feature.error = False
        item_feature.tokens = annotation['question_tokens']
        item_feature.img_id = annotation['image_id']

        update_d1_with_d2(d1=item_feature, d2=feature)

        if self.global_feature_obj:
            global_feature = self.global_feature_obj[idx]
            global_feature.update({'features_global': global_feature.pop('features')})
            update_d1_with_d2(d1=item_feature, d2=global_feature)

        if self.ocr_feature_obj:
            ocr_feature = self.ocr_feature_obj[idx]
            ocr_feature.update({'features_ocr': ocr_feature.pop('features')})
            update_d1_with_d2(d1=item_feature, d2=ocr_feature)

        item_feature.error = None in [feature, global_feature, ocr_feature]

        return item_feature
