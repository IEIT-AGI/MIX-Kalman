from .base_loader import BaseLoader
from ..builder import DATASETS
from ..infocomp.textvqa_infocpler import TextVQAInfoCpler as InfoCpler
from ..reader.textvqa_reader import TextVQAReader as Reader
import torch
import numpy as np
from mixk.utils.config import mixkEasyDict
from collections import OrderedDict


@DATASETS.register_module()
class TEXTVQADATASET(BaseLoader):

    def __init__(self, reader, info_cpler, limit_nums=None):
        super().__init__(Reader, reader, InfoCpler, info_cpler, limit_nums)
        self.max_loc = 100  # feature

    def __getitem__(self, idx):
        item_feature = self.reader[idx]
        # item_feature = self.infocpler.completeInfo(item_feature)
        sample = self.infocpler.complete_info(item_feature)

        sample.question_id = torch.tensor(item_feature['question_id'], dtype=torch.int)
        if isinstance(item_feature['image_id'], int):
            sample.image_id = str(item_feature['image_id'])
        else:
            sample.image_id = item_feature['image_id']

        self.feature_process(item_feature, sample)

        sample.image_info_0_max_features = sample.image_info_0.max_features
        sample.image_info_1_max_features = sample.image_info_1.max_features
        sample.context_info_0_max_features = sample.context_info_0.max_features
        sample.context_info_1_max_features = sample.context_info_1.max_features

        sample.pop('image_id')
        sample.pop('image_info_0')
        sample.pop('image_info_1')
        sample.pop('context_info_0')
        sample.pop('context_info_1')
        sample.pop('sampled_idx_seq')
        return sample

        # if self.infocpler.if_bert:
        #     item = {
        #         'feature': item_feature.features,  # feature - feature
        #         'bbox': item_feature.bbox,  # feature - bbox
        #         'bbox_normalized': item_feature.bbox_normalized,
        #         'feature_global': item_feature.features_global,
        #         'feature_ocr': item_feature.features_ocr,
        #         'bbox_ocr': item_feature.bbox_ocr,
        #         'bbox_ocr_normalized': item_feature.bbox_ocr_normalized,
        #         'ocr_tokens': item_feature.ocr_tokens,
        #         'ocr_vectors_glove': item_feature.ocr_vectors_glove,
        #         'ocr_vectors_fasttext': item_feature.ocr_vectors_fasttext,
        #         'ocr_vectors_phoc': item_feature.ocr_vectors_phoc,
        #         'ocr_vectors_order': item_feature.ocr_vectors_order,
        #         'input_ids': item_feature.input_ids,  # tokens - ids
        #         'input_mask': item_feature.input_mask,  # tokens - mask
        #         'input_segment': item_feature.input_segment,  # tokens - segments
        #         'input_lm_label_ids': item_feature.input_lm_label_ids,  # tokens - mlm labels
        #         'question_id': item_feature.question_id,
        #         'image_id': item_feature.image_id,
        #         'train_prev_inds': item_feature.train_prev_inds,
        #         'train_loss_mask': item_feature.train_loss_mask,
        #         'answers': item_feature.answers,
        #     }
        # else:
        #     item = {
        #         'feature': item_feature.features,  # feature - feature
        #         'bbox': item_feature.bbox,  # feature - bbox
        #         'bbox_normalized': item_feature.bbox_normalized,
        #         'feature_global': item_feature.features_global,
        #         'feature_ocr': item_feature.features_ocr,
        #         'bbox_ocr': item_feature.bbox_ocr,
        #         'bbox_ocr_normalized': item_feature.bbox_ocr_normalized,
        #         'ocr_vectors_glove': item_feature.ocr_vectors_glove,
        #         'ocr_vectors_fasttext': item_feature.ocr_vectors_fasttext,
        #         'ocr_vectors_phoc': item_feature.ocr_vectors_phoc,
        #         'ocr_vectors_order': item_feature.ocr_vectors_order,
        #         'input_ids': item_feature.input_ids,  # tokens - ids
        #         'input_mask': item_feature.input_mask,  # tokens - mask
        #         'question_id': item_feature.question_id,
        #         'image_id': item_feature.image_id,
        #         'train_prev_inds': item_feature.train_prev_inds,
        #         'train_loss_mask': item_feature.train_loss_mask,
        #     }
        #
        # if item_feature.answers_scores is not None:
        #     item['answers_scores'] = item_feature.answers_scores
        #
        # if 'test' in self.splits or 'oneval' in self.splits:
        #     item['quesid2ans'] = self.infocpler.qa_id2ans
        # return item

    def feature_process(self, item_feature, sample):

        def process(image_feature):
            image_info = mixkEasyDict()
            image_loc, image_dim = image_feature.shape
            tmp_image_feat = np.zeros((self.max_loc, image_dim), dtype=np.float32)
            tmp_image_feat[0:image_loc, ] = image_feature[:self.max_loc, :]
            image_info.image_feature = torch.from_numpy(tmp_image_feat)
            image_info.max_features = torch.tensor(image_loc, dtype=torch.long)

            return image_info

        feature_info = process(item_feature.features)
        sample.image_feature_0 = feature_info.image_feature
        sample.image_info_0 = OrderedDict({'max_features': feature_info.max_features})

        ocr_feature_info = process(item_feature.features_ocr)
        sample.image_feature_1 = ocr_feature_info.image_feature
        sample.image_info_1 = OrderedDict({'max_features': ocr_feature_info.max_features})
