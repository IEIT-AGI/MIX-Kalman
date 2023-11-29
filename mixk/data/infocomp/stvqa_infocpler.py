import torch
from ..utils.stream import ItemFeature
from .base_infocpler import BaseInfoCpler


class STVQAInfoCpler(BaseInfoCpler):

    def __init__(self, cfg):
        super().__init__(cfg)

    def complete_info(self, item_feature: ItemFeature):
        tokens = self.tokenizer.tokenize(item_feature.question.strip())
        tokens = self.tokenizer.get_limited_tokens(tokens, self.max_seq_length - 2)
        tokens, input_lm_label_ids = self.tokenizer.random_mask_tokens(tokens, self.word_mask_ratio)
        tokens = [self._CLS_TOKEN] + tokens + [self._SEP_TOEKN]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(tokens)
        input_segment = [0] * len(tokens)
        input_lm_label_ids = [-1] * len(tokens)
        to_extd_length = self.max_seq_length - len(input_ids)
        self.info_extend(to_extd_length, (input_ids, int(self.pad_idx)), (input_mask, 0), (input_segment, 0),
                         (input_lm_label_ids, -1))
        '''
        while len(input_ids) < self.max_seq_length:
            input_ids.append(int(self.pad_idx))
            input_mask.append(0)
            input_segment.append(0)
            input_lm_label_ids.append(-1)
        '''
        # ocr vectors
        ocr_tokens = self.tokenizer.get_limited_tokens(item_feature.ocr_tokens, self.max_ocr_length)
        item_feature.ocr_vectors_glove = self.get_tokens_glove_vectors(ocr_tokens)
        item_feature.ocr_vectors_order = self.get_tokens_order_vectors(ocr_tokens)
        item_feature.ocr_vectors_phoc = self.get_tokens_phoc_vectors(ocr_tokens)
        item_feature.ocr_vectors_fasttext = self.get_tokens_fasttext_vectors(ocr_tokens)

        # ocr features and bboxes
        features_ocr = torch.zeros(
            (self.max_ocr_length,
             item_feature.features_ocr.shape[1] if item_feature.features_ocr is not None else 2048),
            dtype=torch.float)
        bbox_ocr_normalized = torch.zeros(
            (self.max_ocr_length,
             item_feature.ocr_normalized_boxes.shape[1] if item_feature.ocr_normalized_boxes is not None else 4),
            dtype=torch.float)
        if item_feature.features_ocr is not None:
            limit = min(self.max_ocr_length, len(item_feature.features_ocr))
            features_ocr[:limit] = torch.tensor(item_feature.features_ocr[:limit])
            bbox_ocr_normalized[:limit] = torch.tensor(item_feature.ocr_normalized_boxes[:limit])
        item_feature.features_ocr = features_ocr
        item_feature.ocr_normalized_boxes = bbox_ocr_normalized

        # features and bboxes
        img_h = item_feature.image_height
        img_w = item_feature.image_width
        item_feature.bbox = self._get_bbox_from_normalized(item_feature.obj_normalized_boxes, img_h, img_w)
        item_feature.bbox_normalized = item_feature.obj_normalized_boxes
        item_feature.bbox_ocr = self._get_bbox_from_normalized(item_feature.ocr_normalized_boxes, img_h, img_w)
        item_feature.bbox_ocr_normalized = item_feature.ocr_normalized_boxes

        item_feature.input_ids = torch.tensor(input_ids, dtype=torch.long)
        item_feature.input_mask = torch.tensor(input_mask, dtype=torch.int)
        item_feature.input_segment = torch.tensor(input_segment, dtype=torch.int)
        item_feature.input_lm_label_ids = torch.tensor(input_lm_label_ids, dtype=torch.long)
        item_feature.qa_ids = [self.qa_ans2id[ans] for ans in item_feature.answers if ans in self.qa_ans2id]
        # item_feature.qa_allids = [self.qa_ans2id[ans] for ans in item_feature.all_answers if ans in self.qa_ans2id]
        item_feature.answers_scores = self.compute_answers_scores(torch.Tensor(item_feature.qa_ids))
        return item_feature
