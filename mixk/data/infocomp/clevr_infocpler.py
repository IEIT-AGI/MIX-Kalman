import torch

from ..utils.stream import ItemFeature
from .base_infocpler import BaseInfoCpler


class ClevrInfoCpler(BaseInfoCpler):

    def __init__(self, cfg):
        super().__init__(cfg)

    def complete_info(self, item_feature: ItemFeature):
        if self.if_bert:
            return self.complete_bert_info(item_feature)
        else:
            return self.complete_normal_info(item_feature)

    def complete_normal_info(self, item_feature):
        tokens = item_feature.tokens

        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]

        input_ids = [self.stoi[t] for t in tokens]
        input_mask = [1] * len(input_ids)
        '''
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
        '''
        to_extd_length = self.max_seq_length - len(input_ids)
        self.info_extend(to_extd_length, (input_ids, 0), (input_mask, 0))

        item_feature.input_ids = torch.tensor(input_ids, dtype=torch.long)
        item_feature.input_mask = torch.tensor(input_mask, dtype=torch.bool)
        # item_feature.feature_question = torch.stack(list(map(self.get_glove_single_id, input_ids)))

        if item_feature.answers is not None:
            item_feature.answers = self._increase_to_ten(item_feature.answers)
            item_feature.qa_ids = [self.qa_ans2id[ans] for ans in item_feature.answers if ans in self.qa_ans2id]
            item_feature.qa_allids = [self.qa_ans2id[ans] for ans in item_feature.all_answers if ans in self.qa_ans2id]
            item_feature.answers_scores = self.compute_answers_scores(torch.Tensor(item_feature.qa_ids))
        return item_feature

    def complete_bert_info(self, item_feature):
        tokens = self.tokenizer.tokenize(item_feature.question_str.strip())
        tokens = self.tokenizer.get_limited_tokens(tokens, self.max_seq_length - 2)
        tokens, input_lm_label_ids = self.tokenizer.random_mask_tokens(tokens, self.word_mask_ratio)
        tokens = [self._CLS_TOKEN] + tokens + [self._SEP_TOEKN]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(tokens)
        input_segment = [0] * len(tokens)
        input_lm_label_ids = [-1] * len(tokens)
        '''
        while len(input_ids) < self.max_seq_length:
            input_ids.append(int(self.pad_idx))
            input_mask.append(0)
            input_segment.append(0)
            input_lm_label_ids.append(-1)
        '''
        # extend input_xxx
        to_extd_length = self.max_seq_length - len(input_ids)
        self.info_extend(to_extd_length, (input_ids, int(self.pad_idx)), (input_mask, 0), (input_segment, 0),
                         (input_lm_label_ids, -1))

        item_feature.bbox_normalized = torch.tensor(
            self._get_normalized_from_bbox(item_feature.bbox, item_feature.image_height, item_feature.image_width))
        item_feature.input_ids = torch.tensor(input_ids, dtype=torch.long)
        item_feature.input_mask = torch.tensor(input_mask, dtype=torch.int)
        item_feature.input_segment = torch.tensor(input_segment, dtype=torch.int)
        item_feature.input_lm_label_ids = torch.tensor(input_lm_label_ids, dtype=torch.long)
        item_feature.qa_ids = [self.qa_ans2id[ans] for ans in item_feature.answers if ans in self.qa_ans2id]
        item_feature.qa_allids = [self.qa_ans2id[ans] for ans in item_feature.all_answers if ans in self.qa_ans2id]
        item_feature.answers_scores = self.compute_answers_scores(torch.Tensor(item_feature.qa_ids))
        return item_feature
