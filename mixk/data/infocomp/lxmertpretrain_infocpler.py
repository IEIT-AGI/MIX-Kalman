import random

import numpy as np
import torch

from ..utils.stream import ItemFeature
from ..utils.tokenization import BertTokenizer


class LXMERTPreTrainInfoCpler(object):

    def __init__(self, cfg):
        self.if_bert = cfg.if_bert
        self._init_tokens()

        self.max_seq_length = cfg.max_seq_length
        self.word_mask_ratio = cfg.word_mask_ratio
        self.roi_mask_ratio = cfg.roi_mask_ratio

    def completeInfo(self, item_feature: ItemFeature):
        tokens = self.tokenizer.tokenize(item_feature.sent.strip())
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[:(self.max_seq_length - 2)]

        # Ge random words
        masked_tokens, masked_label = self.random_word(tokens)

        # concatenate lm labels and account for CLS, SEP, SEP
        masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)

        # Mask & Segment Word
        lm_label_ids = ([-1] + masked_label + [-1])
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        # while len(input_ids) < self.max_seq_length:
        #    input_ids.append(0)
        #    input_mask.append(0)
        #    segment_ids.append(0)
        #    lm_label_ids.append(-1)

        to_extd_length = self.max_seq_length - len(input_ids)
        self.info_extend(to_extd_length, (input_ids, 0), (input_mask, 0), (segment_ids, 0), (lm_label_ids, -1))
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(lm_label_ids) == self.max_seq_length

        feat, boxes, random_feats = item_feature.features, item_feature.bboxes, item_feature.random_feats
        obj_labels, obj_confs = item_feature.obj_labels, item_feature.obj_confs
        attr_labels, attr_confs = item_feature.attr_labels, item_feature.attr_confs

        # Mask Image Features:
        masked_feat, feat_mask = self.random_feat(feat, random_feats)

        # QA answer label
        if item_feature.label is None or len(item_feature.label) == 0 or item_feature.is_matched != 1:
            # 1. No label 2. Label is pruned 3. unmatched visual + language pair
            ans = -1
        else:
            keys, values = zip(*item_feature.label.items())
            if len(keys) == 1:
                ans = keys[0]
            else:
                value_sum = sum(values)
                prob = [value / value_sum for value in values]
                choice = np.random.multinomial(1, prob).argmax()
                ans = keys[choice]
        item_dict = dict(
            input_ids=torch.tensor(input_ids),
            input_mask=torch.tensor(input_mask),
            segment_ids=torch.tensor(segment_ids),
            lm_label_ids=torch.tensor(lm_label_ids),
            # visual_feats=(masked_feat, boxes),
            feats=torch.from_numpy(masked_feat),
            pos=torch.from_numpy(boxes),
            det_obj_labels=torch.from_numpy(obj_labels),
            det_obj_confs=torch.from_numpy(obj_confs),
            det_attr_labels=torch.from_numpy(attr_labels),
            det_attr_confs=torch.from_numpy(attr_confs),
            det_feat=torch.from_numpy(feat),
            det_feat_mask=torch.from_numpy(feat_mask),
            is_matched=item_feature.is_matched,
            ans=ans,
        )
        return ItemFeature(item_dict)

    def _init_tokens(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.PAD_TOKEN = '<pad>'
        self.SOS_TOKEN = '<s>'
        self.EOS_TOKEN = '</s>'
        self.UNK_TOKEN = '<unk>'
        self.PAD_INDEX = 0
        self.SOS_INDEX = 1
        self.EOS_INDEX = 2
        self.UNK_INDEX = 3
        self._MASK_TOKEN = '[MASK]'
        self._SEP_TOEKN = '[SEP]'
        self._CLS_TOKEN = '[CLS]'
        self._PAD_TOKEN = '[PAD]'
        self.pad_idx = self.tokenizer.vocab[self._PAD_TOKEN]

    def random_word(self, tokens):
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with probability
            if prob < self.word_mask_ratio:
                prob /= self.word_mask_ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = '[MASK]'
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(self.tokenizer.vocab.items()))[0]
                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                try:
                    output_label.append(self.tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(self.tokenizer.vocab['[UNK]'])
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def random_feat(self, feats, random_feats):
        mask_feats = feats.copy()
        feat_mask = np.zeros(len(feats), dtype=np.float32)
        for i in range(len(feats)):
            prob = random.random()
            # mask token with probability
            if prob < self.roi_mask_ratio:
                prob /= self.roi_mask_ratio

                # 80% randomly change token to zero feat
                if prob < 0.8:
                    mask_feats[i, :] = 0.
                # 10% randomly change token to random feat
                elif prob < 0.9:
                    mask_feats[i, :] = random.choice(random_feats)
                # -> rest 10% randomly keep current feat
                # Need to predict this feat
                feat_mask[i] = 1.

        return mask_feats, feat_mask

    def info_extend(self, length, *to_be_extend):
        for info, value in to_be_extend:
            info.extend([value] * length)
