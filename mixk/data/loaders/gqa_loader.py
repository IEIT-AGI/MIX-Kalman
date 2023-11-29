from ..reader.gqa_reader import GQAReader as Reader
from ..infocomp.gqa_infocpler import GQAInfoCpler as InfoCpler
from ..builder import DATASETS
import torch
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
class GQADATASET(BaseLoader):

    def __init__(self, reader, info_cpler, limit_nums=None):
        super().__init__(Reader, reader, InfoCpler, info_cpler, limit_nums)

    def __getitem__(self, idx):
        # idx = 0
        item_feature = self.reader[idx]
        item_feature = self.infocpler.completeInfo(item_feature)

        # Only test for GQA LCGN
        feature = torch.zeros([36, 2048], dtype=torch.float)
        bbox = torch.zeros([36, 4], dtype=torch.float)
        for idx in range(item_feature.features.shape[0]):
            bbox[idx] = torch.tensor(item_feature.bbox[idx])
            feature[idx] = torch.tensor(item_feature.features[idx])
        item_feature.bbox = bbox
        item_feature.features = feature

        item = {
            'feature': item_feature.features,  # feature - feature
            'cls_prob': item_feature.cls_prob,  # 1601 cls_prob
            'bbox': item_feature.bbox,  # feature - bbox
            'image_dim': item_feature.num_boxes,  # feature - bbox_Num
            'input_ids': item_feature.input_ids,  # tokens - ids
            'questionLengths': item_feature.tokens_len,
            'input_mask': item_feature.input_mask,  # tokens - mask
            'input_segment': item_feature.input_segment,  # tokens - segments
            'input_lm_label_ids': item_feature.input_lm_label_ids,  # tokens - mlm labels
            'question_id': item_feature.question_id,
            'image_id': item_feature.image_id,
        }

        if item_feature.answers_scores is not None:
            item['answers_scores'] = item_feature.answers_scores
        # return item_feature.feature, item_feature.input_ids, item_feature.answers_scores, item_feature.input_mask

        if 'test' in self.splits or 'oneval' in self.splits:
            item['quesid2ans'] = self.infocpler.qa_id2ans

        return remove_None_value_elements(item)
