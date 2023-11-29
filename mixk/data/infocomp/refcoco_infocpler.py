import re
import numpy as np
from torchvision import transforms as T

from ..utils.stream import ItemFeature
from .base_infocpler import BaseInfoCpler


class RefCOCOInfoCpler(BaseInfoCpler):

    def __init__(self, cfg):
        self._init_tokens()

        self.vocab_name = cfg.get('vocab_name', 'vocabulart_100k')
        self.vocab_path = self._get_atr_of_atr(cfg, 'mix_vocab', self.vocab_name)
        self.vocab_answer_name = cfg.get('vocab_answer_name', 'answers_vqa')
        self.vocab_answer_path = self._get_atr_of_atr(cfg, 'mix_vocab', self.vocab_answer_name)

        self.default_max_length = cfg.default_max_length
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def read_examples(self, input_line, unique_id):
        """Read a list of `InputExample`s from an input file."""
        examples = []
        # unique_id = 0
        line = input_line  # reader.readline()
        # if not line:
        #     break
        line = line.strip()
        text_a, text_b = None, None

        m = re.match(r'^(.*) \|\|\| (.*)$', line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        examples.append(dict(unique_id=unique_id, text_a=text_a, text_b=text_b))
        # unique_id += 1
        return examples

    def convert_examples_to_features(self, examples, seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example['text_a'])

            tokens_b = None
            if example['text_b']:
                tokens_b = tokenizer.tokenize(example['text_b'])

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                # _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
                pass
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > seq_length - 2:
                    tokens_a = tokens_a[0:(seq_length - 2)]
            tokens = []
            input_type_ids = []
            tokens.append('[CLS]')
            input_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append('[SEP]')
            input_type_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    input_type_ids.append(1)
                tokens.append('[SEP]')
                input_type_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length
            features.append(
                dict(
                    unique_id=example['unique_id'],
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids))
        return features

    def complete_info(self, item_feature: ItemFeature):
        item_feature.img = self.transform(item_feature.img)
        phrases = item_feature.phrase

        examples = self.read_examples(phrases, item_feature.item)
        features = self.convert_examples_to_features(
            examples=examples, seq_length=self.default_max_length, tokenizer=self.tokenizer)
        input_ids = np.array(features[0]['input_ids'], dtype=int)
        input_mask = np.array(features[0]['input_mask'], dtype=int)

        item_feature.input_ids = input_ids
        item_feature.input_mask = input_mask

        return item_feature
