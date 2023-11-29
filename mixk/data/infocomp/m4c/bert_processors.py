# Copyright (c) Facebook, Inc. and its affiliates.

import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from transformers.tokenization_auto import AutoTokenizer
from collections import defaultdict
from mixk.data.builder import PROCESSOR
from dataclasses import dataclass
from mixk.data.vqadata.vocabprocessor import VocabProcessor


class BaseProcessor:
    """Every processor in mixk needs to inherit this class . The end-user
    mainly needs to implement ``__call__`` function.

    Args:
        config (DictConfig): Config for this processor, containing `type` and
                             `params` attributes if available.
    """

    def __init__(self, config: Dict[str, Any], *args, **kwargs):
        return

    def __call__(self, item: Any, *args, **kwargs) -> Any:
        """Main function of the processor. Takes in a dict and returns back a
        dict.

        Args:
            item (Dict): Some item that needs to be processed.

        Returns:
            Dict: Processed dict.
        """
        return item


@PROCESSOR.register_module()
class MaskedTokenProcessor(BaseProcessor):
    _CLS_TOKEN = '[CLS]'
    _SEP_TOKEN = '[SEP]'

    def __init__(self, config, *args, **kwargs):
        tokenizer_config = config.tokenizer_config
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.type, **tokenizer_config.params)

        self._max_seq_length = config.max_seq_length
        self._probability = getattr(config, 'mask_probability', 0.15)

    def get_vocab_size(self) -> int:
        return len(self._tokenizer)

    def tokenize(self, tokens: Union[str, List[str]]) -> List[str]:
        return self._tokenizer.tokenize(tokens)

    def _convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def _convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        return self._tokenizer.convert_ids_to_tokens(ids)

    def _random_word(self, tokens: List[str], probability: float = 0.15) -> Tuple[List[str], List[int]]:
        labels = []
        for idx, token in enumerate(tokens):
            prob = random.random()

            if prob < probability:
                prob /= probability

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[idx] = '[MASK]'
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[idx] = self._convert_ids_to_tokens(
                        torch.randint(self.get_vocab_size(), (1, ), dtype=torch.long))[0]

                # rest 10% keep the original token as it is

                labels.append(self._convert_tokens_to_ids(token))
            else:
                labels.append(-1)

        return tokens, labels

    def _truncate_seq_pair(self, tokens_a: List[str], tokens_b: List[str], max_length: int):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        if tokens_b is None:
            tokens_b = []
        else:
            # _convert_to_indices does [CLS] tokens_a [SEP] tokens_b [SEP]
            max_length -= 1
            assert max_length >= 0, ('Max length should be minimum 2 in case of single sentence' +
                                     ' and 3 in case of two sentences.')

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_to_indices(
        self,
        tokens_a: List[str],
        tokens_b: Optional[List[str]] = None,
        probability: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        tokens_a, label_a = self._random_word(tokens_a, probability=probability)
        tokens = [self._CLS_TOKEN] + tokens_a + [self._SEP_TOKEN]
        segment_ids = [0] + [0] * len(tokens_a) + [0]

        if tokens_b:
            tokens_b, label_b = self._random_word(tokens_b, probability=probability)
            lm_label_ids = [-1] + label_a + [-1] + label_b + [-1]
            assert len(tokens_b) > 0
            tokens += tokens_b + [self._SEP_TOKEN]
            segment_ids += [1] * len(tokens_b) + [1]
        else:
            lm_label_ids = [-1] + label_a + [-1]

        input_ids = self._convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self._max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == self._max_seq_length
        assert len(input_mask) == self._max_seq_length
        assert len(segment_ids) == self._max_seq_length
        assert len(lm_label_ids) == self._max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        lm_label_ids = torch.tensor(lm_label_ids, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'lm_label_ids': lm_label_ids,
            'tokens': tokens,
        }

    def __call__(self, item: Dict[str, Any]):
        text_a = item['text_a']
        text_b = item.get('text_b', None)

        tokens_a = self.tokenize(text_a)
        tokens_b = None

        if text_b:
            tokens_b = self.tokenize(text_b)

        self._truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length - 2)
        output = self._convert_to_indices(tokens_a, tokens_b, probability=self._probability)
        output['is_correct'] = torch.tensor(item['is_correct'], dtype=torch.long)

        return output


@PROCESSOR.register_module()
class BertTokenizerProcessor(MaskedTokenProcessor):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._probability = config.get('mask_probability', 0)

    def __call__(self, item: Dict[str, Any]):

        if 'text' in item:
            text_a = item['text']
        else:
            text_a = ' '.join(item['tokens'])

        if isinstance(text_a, list):
            text_a = ' '.join(text_a)

        tokens_a = self.tokenize(text_a)

        # 'text_b' can be defined in the dataset preparation
        tokens_b = None
        if 'text_b' in item:
            text_b = item['text_b']
            if text_b:
                tokens_b = self.tokenize(text_b)

        self._truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length - 2)
        output = self._convert_to_indices(tokens_a, tokens_b, probability=self._probability)
        output['text'] = output['tokens']
        return output


@PROCESSOR.register_module()
class CopyProcessor(BaseProcessor):
    """Copy boxes from numpy array."""

    def __init__(self, config, *args, **kwargs):
        self.max_length = config.max_length

    def __call__(self, item):
        blob = item['blob']
        final_blob = np.zeros((self.max_length, ) + blob.shape[1:], blob.dtype)
        final_blob[:len(blob)] = blob[:len(final_blob)]

        return {'blob': torch.from_numpy(final_blob)}


# @PROCESSOR.register_module()
class SimpleWordProcessor(BaseProcessor):
    """Tokenizes a word and processes it.

    Attributes:
        tokenizer (function): Type of tokenizer to be used.
    """

    def __init__(self, *args, **kwargs):
        from mixk.utils.third_party_libs import word_tokenize

        self.tokenizer = word_tokenize

    def __call__(self, item, *args, **kwargs):
        return {'text': self.tokenizer(item['text'], *args, **kwargs)}


@PROCESSOR.register_module()
class PhocProcessor(VocabProcessor):
    """Compute PHOC features from text tokens."""

    def __init__(self, config, *args, **kwargs):
        from .phoc import build_phoc

        self._build_phoc = build_phoc
        self._init_extras(config)
        self.config = config

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        phoc_dim = 604
        output = torch.full((self.max_length, phoc_dim), fill_value=self.PAD_INDEX, dtype=torch.float)

        for idx, token in enumerate(tokens):
            output[idx] = torch.from_numpy(self._build_phoc(token))

        return output


@dataclass
class ProcessorConfigType:
    type: str
    params: Dict[str, Any]


@dataclass
class BatchProcessorConfigType:
    processors: ProcessorConfigType


@PROCESSOR.register_module()
class M4CAnswerProcessor(BaseProcessor):
    """Process a TextVQA answer for iterative decoding in M4C."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        from mixk.utils.third_party_libs import VocabDict
        self.answer_vocab = VocabDict(config.vocab_file, *args, **kwargs)
        self.PAD_IDX = self.answer_vocab.word2idx('<pad>')
        self.BOS_IDX = self.answer_vocab.word2idx('<s>')
        self.EOS_IDX = self.answer_vocab.word2idx('</s>')
        self.UNK_IDX = self.answer_vocab.UNK_INDEX

        # make sure PAD_IDX, BOS_IDX and PAD_IDX are valid (not <unk>)
        assert self.PAD_IDX != self.answer_vocab.UNK_INDEX
        assert self.BOS_IDX != self.answer_vocab.UNK_INDEX
        assert self.EOS_IDX != self.answer_vocab.UNK_INDEX
        assert self.PAD_IDX == 0

        # self.answer_preprocessor = Processor(config.preprocessor)
        self.answer_preprocessor = SimpleWordProcessor()
        assert self.answer_preprocessor is not None

        self.num_answers = config.num_answers
        self.max_length = config.max_length
        self.max_copy_steps = config.max_copy_steps
        assert self.max_copy_steps >= 1

        self.match_answer_to_unk = False

    def tokenize(self, sentence):
        return sentence.split()

    def match_answer_to_vocab_ocr_seq(self, answer, vocab2idx_dict, ocr2inds_dict, max_match_num=20):
        """Match an answer to a list of sequences of indices each index
        corresponds to either a fixed vocabulary or an OCR token (in the index
        address space, the OCR tokens are after the fixed vocab)"""
        num_vocab = len(vocab2idx_dict)

        answer_words = self.tokenize(answer)
        answer_word_matches = []
        for word in answer_words:
            # match answer word to fixed vocabulary
            matched_inds = []
            if word in vocab2idx_dict:
                matched_inds.append(vocab2idx_dict.get(word))
            # match answer word to OCR
            # we put OCR after the fixed vocabulary in the answer index space
            # so add num_vocab offset to the OCR index
            matched_inds.extend([num_vocab + idx for idx in ocr2inds_dict[word]])
            if len(matched_inds) == 0:
                if self.match_answer_to_unk:
                    matched_inds.append(vocab2idx_dict.get('<unk>'))
                else:
                    return []
            answer_word_matches.append(matched_inds)

        # expand per-word matched indices into the list of matched sequences
        if len(answer_word_matches) == 0:
            return []
        idx_seq_list = [()]
        for matched_inds in answer_word_matches:
            idx_seq_list = [seq + (idx, ) for seq in idx_seq_list for idx in matched_inds]
            if len(idx_seq_list) > max_match_num:
                idx_seq_list = idx_seq_list[:max_match_num]

        return idx_seq_list

    def get_vocab_size(self):
        answer_vocab_nums = self.answer_vocab.num_vocab
        answer_vocab_nums += self.max_length

        return answer_vocab_nums

    def get_true_vocab_size(self):
        return self.answer_vocab.num_vocab

    def compute_answer_scores(self, answers):
        gt_answers = list(enumerate(answers))
        unique_answers = sorted(set(answers))
        unique_answer_scores = [0] * len(unique_answers)
        for idx, unique_answer in enumerate(unique_answers):
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [item for item in other_answers if item[1] == unique_answer]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[idx] = sum(accs) / len(accs)
        unique_answer2score = {a: s for a, s in zip(unique_answers, unique_answer_scores)}
        return unique_answer2score

    def __call__(self, item):
        answers = item['answers']

        if not answers:
            return {
                'sampled_idx_seq': None,
                'train_prev_inds': torch.zeros(self.max_copy_steps, dtype=torch.long),
            }

        answers = [self.answer_preprocessor({'text': a})['text'] for a in answers]
        assert len(answers) == self.num_answers

        # Step 1: calculate the soft score of ground-truth answers
        unique_answer2score = self.compute_answer_scores(answers)

        # Step 2: fill the first step soft scores for tokens
        scores = torch.zeros(self.max_copy_steps, self.get_vocab_size(), dtype=torch.float)

        # match answers to fixed vocabularies and OCR tokens.
        ocr2inds_dict = defaultdict(list)
        for idx, token in enumerate(item['tokens']):
            ocr2inds_dict[token].append(idx)
        answer_dec_inds = [
            self.match_answer_to_vocab_ocr_seq(a, self.answer_vocab.word2idx_dict, ocr2inds_dict) for a in answers
        ]

        # Collect all the valid decoding sequences for each answer.
        # This part (idx_seq_list) was pre-computed in imdb (instead of online)
        # to save time
        all_idx_seq_list = []
        for answer, idx_seq_list in zip(answers, answer_dec_inds):
            all_idx_seq_list.extend(idx_seq_list)
            # fill in the soft score for the first decoding step
            score = unique_answer2score[answer]
            for idx_seq in idx_seq_list:
                score_idx = idx_seq[0]
                # the scores for the decoding Step 0 will be the maximum
                # among all answers starting with that vocab
                # for example:
                # if "red apple" has score 0.7 and "red flag" has score 0.8
                # the score for "red" at Step 0 will be max(0.7, 0.8) = 0.8
                scores[0, score_idx] = max(scores[0, score_idx], score)

        # train_prev_inds is the previous prediction indices in auto-regressive
        # decoding
        train_prev_inds = torch.zeros(self.max_copy_steps, dtype=torch.long)
        # train_loss_mask records the decoding steps where losses are applied
        train_loss_mask = torch.zeros(self.max_copy_steps, dtype=torch.float)
        if len(all_idx_seq_list) > 0:
            # sample a random decoding answer sequence for teacher-forcing
            idx_seq = all_idx_seq_list[np.random.choice(len(all_idx_seq_list))]
            dec_step_num = min(1 + len(idx_seq), self.max_copy_steps)
            train_loss_mask[:dec_step_num] = 1.0

            train_prev_inds[0] = self.BOS_IDX
            for t in range(1, dec_step_num):
                train_prev_inds[t] = idx_seq[t - 1]
                score_idx = idx_seq[t] if t < len(idx_seq) else self.EOS_IDX
                scores[t, score_idx] = 1.0
        else:
            idx_seq = ()

        answer_info = {
            'answers': answers,
            'answers_scores': scores,
            'sampled_idx_seq': idx_seq,
            'train_prev_inds': train_prev_inds,
            'train_loss_mask': train_loss_mask,
        }
        return answer_info


@PROCESSOR.register_module()
class TransformerBboxProcessor(BaseProcessor):
    """Process a bounding box and returns a array of normalized bbox positions
    and area."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.bbox_key = config.get('bbox_key', 'bbox')
        self.image_width_key = config.get('image_width_key', 'image_width')
        self.image_height_key = config.get('image_height_key', 'image_height')

    def __call__(self, item):
        bbox = item[self.bbox_key]
        image_w = item[self.image_width_key]
        image_h = item[self.image_height_key]
        image_location = torch.zeros((bbox.shape[0], 5), dtype=torch.float)
        image_location[:, :4] = torch.from_numpy(bbox[:, :4])
        image_location[:, 4] = ((image_location[:, 3] - image_location[:, 1]) *
                                (image_location[:, 2] - image_location[:, 0]) / (image_w * image_h))
        image_location[:, 0] = image_location[:, 0] / image_w
        image_location[:, 1] = image_location[:, 1] / image_h
        image_location[:, 2] = image_location[:, 2] / image_w
        image_location[:, 3] = image_location[:, 3] / image_h
        item['bbox'] = image_location
        return item


@PROCESSOR.register_module()
class MaskedRegionProcessor(BaseProcessor):
    """Masks a region with probability `mask_probability`"""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.mask_prob = config.get('mask_probability', 0.15)
        self.mask_region_prob = config.get('mask_region_probability', 0.9)

    def __call__(self, item):
        image_labels = []

        for i in range(item.shape[0]):
            prob = random.random()
            # mask token with 15% probability
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < self.mask_region_prob:
                    item[i] = 0
                image_labels.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                image_labels.append(-1)
        return torch.tensor(image_labels, dtype=torch.long)
