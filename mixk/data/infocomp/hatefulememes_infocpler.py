import numpy as np
from torchvision import transforms as T

from ..utils.stream import ItemFeature
from .base_infocpler import BaseInfoCpler


class HatefulMemesInfoCpler(BaseInfoCpler):

    def __init__(self, cfg):
        self._init_tokens()
        self.default_max_length = cfg.default_max_length
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def complete_info(self, item_feature: ItemFeature):
        item_feature.img = self.transform(item_feature.img)
        tokens = self.tokenizer.tokenize(item_feature.text.strip())
        tokens = [self._CLS_TOKEN] + tokens + [self._SEP_TOEKN]
        input_mask = [1] * len(tokens)
        input_type_ids = [0] * self.default_max_length
        '''
        to_extd_length = self.default_max_length - len(tokens)
        tokens.extend([self._PAD_TOKEN] * to_extend)
        input_mask.extend([0] * to_extend)
        '''
        to_extd_length = self.default_max_length - len(tokens)
        self.info_extend(to_extd_length, (tokens, self.PAD_TOKEN), (input_mask, 0))

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = input_mask[:self.default_max_length]
        input_ids = input_ids[:self.default_max_length]

        input_ids = np.array(input_ids[:self.default_max_length])
        input_mask = np.array(input_mask[:self.default_max_length])
        input_type_ids = np.array(input_type_ids[:self.default_max_length])

        item_feature.input_ids = input_ids
        item_feature.input_mask = input_mask
        item_feature.input_type_ids = input_type_ids

        return item_feature
