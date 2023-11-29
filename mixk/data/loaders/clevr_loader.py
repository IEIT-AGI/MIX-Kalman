from .base_loader import BaseLoader
from ..builder import DATASETS
from ..infocomp import ClevrInfoCpler as InfoCpler
from ..reader import ClevrReader as Reader


@DATASETS.register_module()
class ClevrDATASET(BaseLoader):

    def __init__(self, reader, info_cpler, limit_nums=None):
        super().__init__(Reader, reader, InfoCpler, info_cpler, limit_nums)

    def __getitem__(self, idx):
        item_feature = self.reader[idx]
        item_feature = self.infocpler.complete_info(item_feature)

        item = {
            'feature': item_feature.features,  # feature - feature
            'bbox': item_feature.bbox,  # feature - bbox
            'bbox_normalized': item_feature.bbox_normalized,  # feature - bbox(normalized)
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
