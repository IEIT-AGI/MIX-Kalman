import numpy as np
import torch

from ..builder import build_processor
from mixk.utils.third_party_libs import VocabDict
from ..utils.stream import ItemFeature
from .base_infocpler import BaseInfoCpler
from mixk.utils.config import mixkEasyDict
from mixk.utils.common_function import object_to_byte_tensor
from copy import deepcopy
import json
import os


class TextVQAAnswerProcessor:

    def __init__(self, vocab_file: str):
        self.answer_vocab = VocabDict(vocab_file)
        self.PAD_IDX = self.answer_vocab.word2idx('<pad>')
        self.BOS_IDX = self.answer_vocab.word2idx('<s>')
        self.EOS_IDX = self.answer_vocab.word2idx('</s>')
        self.UNK_IDX = self.answer_vocab.UNK_INDEX
        # make sure PAD_IDX, BOS_IDX and PAD_IDX are valid (not <unk>)
        assert self.PAD_IDX != self.answer_vocab.UNK_INDEX
        assert self.BOS_IDX != self.answer_vocab.UNK_INDEX
        assert self.EOS_IDX != self.answer_vocab.UNK_INDEX
        assert self.PAD_IDX == 0

    def get_true_vocab_size(self):
        return self.answer_vocab.num_vocab


class TextVQAInfoCpler(BaseInfoCpler):

    def __init__(self, cfg):
        self.cfg = cfg

        self.use_ocr = self.cfg.use_ocr
        self.use_ocr_info = self.cfg.use_ocr_info
        self.use_order_vectors = self.cfg.use_order_vectors
        self.return_features_info = self.cfg.return_features_info
        self.phoc_feature_path = getattr(self.cfg, 'phoc_feature_path', None)

        self._init_processors()

    def _init_processors(self):
        self._init_text_processor()
        self._init_copy_processor()
        self._init_ocr_processor()
        self._init_answer_processor()

    def _init_answer_processor(self):
        config = deepcopy(self.cfg.answer_processor)
        self.answer_processor = build_processor(config)

    def _init_copy_processor(self):
        config = deepcopy(self.cfg.copy_processor)
        self.copy_processor = build_processor(config)

    def _init_text_processor(self):
        config = deepcopy(self.cfg.text_processor)
        self.text_processor = build_processor(config)

    def _init_ocr_processor(self):
        ocr_token_processor_cfg = deepcopy(self.cfg.ocr_token_processor)
        self.ocr_token_processor = build_processor(ocr_token_processor_cfg)

        if self.phoc_feature_path is None:
            phoc_cfg = deepcopy(self.cfg.phoc_processor)
            self.phoc_processor = build_processor(phoc_cfg)
        else:
            self.phoc_processor = None

        context_cfg = deepcopy(self.cfg.context_processor)
        self.context_processor = build_processor(context_cfg)

        bbox_cfg = deepcopy(self.cfg.bbox_processor)
        self.bbox_processor = build_processor(bbox_cfg)

    def complete_info(self, item_feature: ItemFeature):
        current_sample = ItemFeature()

        # 1. Load text (question words)
        current_sample = self.add_question_info(item_feature, current_sample)

        # 2. Load object
        # object bounding box information
        current_sample = self.add_object_info(item_feature, current_sample)

        # 3. Load OCR
        current_sample = self.add_ocr_info(item_feature, current_sample)

        # 4. load answer
        current_sample = self.add_answer_info(item_feature, current_sample)
        return current_sample

    def add_question_info(self, item_feature: ItemFeature, sample: ItemFeature):
        question_str = (item_feature['question'] if 'question' in item_feature else item_feature['question_str'])
        text_processor_args = {'text': question_str}

        if 'question_tokens' in item_feature:
            text_processor_args['tokens'] = item_feature['question_tokens']

        processed_question = self.text_processor(text_processor_args)

        if 'input_ids' in processed_question:
            sample.text = processed_question['input_ids']
            sample.text_len = torch.tensor(len(processed_question['tokens']), dtype=torch.long)
        else:
            # For GLoVe based processors
            sample.text = processed_question['text']
            sample.text_len = processed_question['length']

        return sample

    def add_object_info(self, item_feature: ItemFeature, sample: ItemFeature):
        if 'obj_normalized_boxes' in item_feature and hasattr(self, 'copy_processor'):
            sample.obj_bbox_coordinates = self.copy_processor({'blob': item_feature['obj_normalized_boxes']})['blob']

        return sample

    def add_ocr_info(self, item_feature: ItemFeature, sample: ItemFeature):
        sample_info = item_feature

        if not self.use_ocr:
            # remove all OCRs from the sample
            # (i.e. make an empty OCR list)
            sample_info['ocr_tokens'] = []
            sample_info['ocr_info'] = []
            if 'ocr_normalized_boxes' in sample_info:
                sample_info['ocr_normalized_boxes'] = np.zeros((0, 4), np.float32)
            # clear OCR visual features
            if 'image_feature_1' in sample:
                sample.image_feature_1 = torch.zeros_like(sample.image_feature_1)
            return sample

            # Preprocess OCR tokens
        if hasattr(self, 'ocr_token_processor'):
            ocr_tokens = [self.ocr_token_processor({'text': token})['text'] for token in sample_info['ocr_tokens']]
        else:
            ocr_tokens = sample_info['ocr_tokens']
            # Get FastText embeddings for OCR tokens
        context = self.context_processor({'tokens': ocr_tokens})
        sample.context = context['text']
        sample.ocr_tokens = context['tokens']

        sample.context_tokens = object_to_byte_tensor(context['tokens'])
        sample.context_feature_0 = context['text']
        sample.context_info_0 = mixkEasyDict()
        sample.context_info_0.max_features = context['length']

        # Get PHOC embeddings for OCR tokens
        if hasattr(self, 'phoc_processor'):
            if self.phoc_processor is None:
                if item_feature.context_phoc is None:
                    phoc_file_name = f'{item_feature.set_name}_qid_{item_feature.question_id}.json'
                    context_phoc = self.get_phoc_feature(file_name=phoc_file_name)
                else:
                    context_phoc = item_feature.context_phoc

                sample.context_feature_1 = torch.Tensor(context_phoc['text'])
                sample.context_info_1 = mixkEasyDict()
                sample.context_info_1.max_features = torch.tensor(context_phoc['length'])
            else:
                context_phoc = self.phoc_processor({'tokens': ocr_tokens})
                sample.context_feature_1 = context_phoc['text']
                sample.context_info_1 = mixkEasyDict()
                sample.context_info_1.max_features = context_phoc['length']

        # OCR order vectors
        if self.cfg.get('use_order_vectors', False):
            order_vectors = np.eye(len(sample.ocr_tokens), dtype=np.float32)
            order_vectors = torch.from_numpy(order_vectors)
            order_vectors[context['length']:] = 0
            sample.order_vectors = order_vectors

        # OCR bounding box information
        if 'ocr_normalized_boxes' in sample_info and hasattr(self, 'copy_processor'):
            # New imdb format: OCR bounding boxes are already pre-computed
            max_len = self.cfg.answer_processor.config.max_length
            sample.ocr_bbox_coordinates = self.copy_processor({'blob':
                                                               sample_info['ocr_normalized_boxes']})['blob'][:max_len]
        elif self.use_ocr_info and 'ocr_info' in sample_info:
            # Old imdb format: OCR bounding boxes are computed on-the-fly
            # from ocr_info
            sample.ocr_bbox_coordinates = self.bbox_processor({'info': sample_info['ocr_info']})['bbox'].coordinates

        return sample

    def add_answer_info(self, item_feature: ItemFeature, sample: ItemFeature):
        sample_info = item_feature
        answers = sample_info.get('answers', [])
        answer_processor_arg = {'answers': answers}

        answer_processor_arg['tokens'] = sample.pop('ocr_tokens', [])
        processed_answers = self.answer_processor(answer_processor_arg)
        sample.update(processed_answers)
        sample.answers = object_to_byte_tensor(answers)

        if 'answers_scores' in sample:
            sample.targets = sample.pop('answers_scores')

        return sample

    def get_phoc_feature(self, file_name):
        with open(os.path.join(self.phoc_feature_path, file_name), 'r') as f:
            phoc = json.load(f)
            context_phoc = phoc['context_phoc']
            return context_phoc
