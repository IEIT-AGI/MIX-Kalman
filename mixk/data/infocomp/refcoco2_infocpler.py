import numpy as np
from torchvision import transforms as T

from ..utils.stream import ItemFeature
from .base_infocpler import BaseInfoCpler


class RefCOCOInfoCpler(BaseInfoCpler):

    def __init__(self, cfg):
        self._init_tokens()
        self.default_max_length = cfg.default_max_length
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def complete_info(self, item_feature: ItemFeature):
        item_feature.img = self.transform(item_feature.img)
        phrases = item_feature.phrase
        tokenss = [self.tokenizer.tokenize(phrase.strip()) for phrase in phrases]
        tokens_r = [self._CLS_TOKEN]
        input_type_ids = [0]
        for i, tokens in enumerate(tokenss):
            tokens_r += tokens
            tokens_r += [self._SEP_TOEKN]
            input_type_ids += [i] * (len(tokens) + 1)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_r)

        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.default_max_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        input_ids = np.array(input_ids[:self.default_max_length])
        input_mask = np.array(input_mask[:self.default_max_length])
        input_type_ids = np.array(input_type_ids[:self.default_max_length])

        item_feature.input_ids = input_ids
        item_feature.input_mask = input_mask
        item_feature.input_type_ids = input_type_ids

        return item_feature
