from .base_loader import BaseLoader
from ..builder import DATASETS
from ..infocomp import STVQAInfoCpler as InfoCpler
from ..reader import STVQAReader as Reader


@DATASETS.register_module()
class STVQADATASET(BaseLoader):

    def __init__(self, reader, info_cpler, limit_nums=None):
        super().__init__(Reader, reader, InfoCpler, info_cpler, limit_nums)

    def __getitem__(self, idx):
        try:
            item_feature = self.reader[idx]
            item_feature = self.infocpler.complete_info(item_feature)
        except Exception:
            item_feature = 0

        item = {
            'feature': item_feature.features,  # feature - feature
            'bbox': item_feature.bbox,  # feature - bbox
            'bbox_normalized': item_feature.bbox_normalized,
            # 'feature_global': item_feature.features_global,
            'feature_ocr': item_feature.features_ocr,
            'bbox_ocr': item_feature.bbox_ocr,
            'bbox_ocr_normalized': item_feature.bbox_ocr_normalized,
            'ocr_vectors_glove': item_feature.ocr_vectors_glove,
            'ocr_vectors_fasttext': item_feature.ocr_vectors_fasttext,
            'ocr_vectors_phoc': item_feature.ocr_vectors_phoc,
            'ocr_vectors_order': item_feature.ocr_vectors_order,
            'input_ids': item_feature.input_ids,  # tokens - ids
            'input_mask': item_feature.input_mask,  # tokens - mask
            'input_segment': item_feature.input_segment,  # tokens - segments
            'input_lm_label_ids': item_feature.input_lm_label_ids,  # tokens - mlm labels
            'question_id': item_feature.question_id,
            'image_id': item_feature.image_id,
        }

        if item_feature.answers_scores is not None:
            item['answers_scores'] = item_feature.answers_scores

        if 'test' in self.splits or 'oneval' in self.splits:
            item['quesid2ans'] = self.infocpler.qa_id2ans
        return item
