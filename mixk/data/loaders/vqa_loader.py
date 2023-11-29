from ..reader.vqa_reader import VQAReader as Reader
from ..infocomp.vqa_infocpler import VQAInfoCpler as InfoCpler
from ..builder import DATASETS
from .base_loader import BaseLoader


def remove_None_value_elements(input_dict):
    if type(input_dict) is not dict:
        return None
    result = {}
    for key in input_dict:
        tmp = {}
        if input_dict[key] is not None:
            if type(input_dict[key]).__name__ == 'dict':
                tmp.update({key: remove_None_value_elements(input_dict[key])})
            else:
                tmp.update({key: input_dict[key]})
        result.update(tmp)
    return result


@DATASETS.register_module()
class VQADATASET(BaseLoader):

    def __init__(self, reader, info_cpler, limit_nums=None):
        super().__init__(Reader, reader, InfoCpler, info_cpler, limit_nums)

    def __getitem__(self, idx):
        # idx = 0
        item_feature = self.reader[idx]
        item_feature = self.infocpler.completeInfo(item_feature)

        item = {
            'features': item_feature.features,  # feature - feature
            'feature_global': item_feature.global_features,  # feature - global_features
            'cls_prob': item_feature.cls_prob,  # 1601 cls_prob
            'bbox': item_feature.bbox,  # feature - bbox
            'image_dim': item_feature.num_boxes,  # feature - bbox_Num
            'input_ids': item_feature.input_ids,  # tokens - ids
            'input_mask': item_feature.input_mask,  # tokens - mask
            'input_segment': item_feature.input_segment,  # tokens - segments
            'input_lm_label_ids': item_feature.input_lm_label_ids,  # tokens - mlm labels
            'question_id': item_feature.question_id,
            'image_id': item_feature.image_id,
        }

        if not self.reader.is_global:
            item.pop('feature_global')
        if item_feature.answers_scores is not None:
            item['answers_scores'] = item_feature.answers_scores
        # return item_feature.feature, item_feature.input_ids, item_feature.answers_scores, item_feature.input_mask

        if 'test' in self.splits or 'oneval' in self.splits:
            item['quesid2ans'] = self.infocpler.qa_id2ans
        item = remove_None_value_elements(item)
        return item
