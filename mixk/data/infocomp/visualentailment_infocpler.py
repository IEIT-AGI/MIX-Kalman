import numpy as np

from ..utils.stream import ItemFeature
from .base_infocpler import BaseInfoCpler


class VisualEntailmentInfoCpler(BaseInfoCpler):

    def __init__(self, cfg):
        self._init_tokens()
        self.default_max_length = cfg.default_max_length

    def complete_info(self, item_feature: ItemFeature):
        tokens1 = self.tokenizer.tokenize(item_feature.text1.strip())
        tokens2 = self.tokenizer.tokenize(item_feature.text2.strip())
        tokens = [self._CLS_TOKEN] + tokens1 + [self._SEP_TOEKN] + tokens2 + [self._SEP_TOEKN]
        input_mask = [1] * len(tokens)
        input_type_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)
        to_extd_length = self.default_max_length - len(tokens)
        self.info_extend(to_extd_length, (tokens, self.PAD_TOKEN), (input_mask, 0), (input_type_ids, 0))
        # while len(tokens) < self.default_max_length:
        #    tokens.append(self._PAD_TOKEN)
        #    input_mask.append(0)
        #    input_type_ids.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = input_mask[:self.default_max_length]
        input_ids = input_ids[:self.default_max_length]
        input_type_ids = input_type_ids[:self.default_max_length]

        input_ids = np.array(input_ids[:self.default_max_length])
        input_mask = np.array(input_mask[:self.default_max_length])
        input_type_ids = np.array(input_type_ids[:self.default_max_length])

        item_feature.input_ids = input_ids
        item_feature.input_mask = input_mask
        item_feature.input_type_ids = input_type_ids

        return item_feature
