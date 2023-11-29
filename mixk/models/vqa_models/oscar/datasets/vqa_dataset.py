from __future__ import absolute_import, division, print_function

import logging
import os
import time
import json
import base64

import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
from ..utils.task_utils import (_truncate_seq_pair, output_modes, processors)
from mixk.data.builder import DATASETS
# import mixk.utils.distributed_info as comm
from .dataset_utils import target_tensor
import sys

sys.path.insert(0, '.')
logger = logging.getLogger(__name__)


def _load_dataset(args, name):
    processor = processors[args.task_name]()
    labels = processor.get_labels(args.label_file)

    if name == 'train':
        if args.data_label_type == 'mask':
            if args.use_vg:
                # examples = processor.get_train_examples(args.data_dir, 'train2014_vg_qla_mrcnn.json')
                examples = processor.get_train_examples(args.txt_data_dir, 'train2014_vg_qla_mrcnn.json')
            else:
                examples = processor.get_train_examples(args.txt_data_dir, 'train2014_qla_mrcnn.json')
        else:
            examples = processor.get_train_examples(args.txt_data_dir, 'train2014_qla.json')
    elif name == 'val':
        if args.data_label_type == 'mask':
            if args.use_vg_dev:
                examples = processor.get_dev_examples(args.txt_data_dir, 'vg_qla_mrcnn.json')
            else:
                examples = processor.get_dev_examples(args.txt_data_dir, 'val2014_qla_mrcnn.json')
        else:
            examples = processor.get_dev_examples(args.txt_data_dir, 'val2014_qla.json')
    elif name == 'train+val':
        if args.data_label_type == 'mask':
            examples = processor.get_train_examples(args.txt_data_dir, 'train+val2014_qla_mrcnn.json')
            # examples = processor.get_train_examples(args.data_dir, 'train+val2014_qla_mrcnn.json')
        else:
            examples = processor.get_train_examples(args.txt_data_dir, 'train+val2014_qla.json')
    elif name == 'test2015':
        if args.data_label_type == 'mask':
            examples = processor.get_test_examples(args.data_dir, 'test2015_qla_mrcnn.json')
        else:
            examples = processor.get_test_examples(args.data_dir, 'test2014_qla.json')
    elif name == 'test-dev2015':
        if args.data_label_type == 'mask':
            examples = processor.get_test_examples(args.data_dir, 'test-dev2015_qla_mrcnn.json')
        else:
            examples = processor.get_test_examples(args.data_dir, 'test2014_qla.json')

    return examples, labels


@DATASETS.register_module()
class OSCAR_VQADataset(Dataset):
    """VQA Dataset."""

    def __init__(self, reader):
        super(OSCAR_VQADataset, self).__init__()

        # if comm.is_main_process():
        #     logger = logging.getLogger(__name__)
        logger.info('start loading vqa data')

        self.args = args = reader
        if isinstance(args.name, list):
            name = args.name[0]
        else:
            name = args.name
        self.name = name

        assert name in ['train', 'val', 'test-dev2015', 'test2015', 'train+val']

        self.tokenizer = BertTokenizer.from_pretrained(
            args.stokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

        # load image features
        t_start = time.time()
        self.img_feature_file = None
        self.img_feat_offset_map = None

        if args.img_feature_type == 'faster_r-cnn':
            if args.img_feat_format == 'pt':
                if args.img_feature_dim == 2048:  # object features
                    self.img_features = torch.load(
                        os.path.join(args.data_dir, '{}_img_frcnn_obj_feats.pt'.format(name)))
                else:  # object + spatial features
                    if args.use_vg_dev:
                        self.img_features = torch.load(os.path.join(args.data_dir, 'train+val_img_frcnn_feats.pt'))
                    else:
                        # self.img_features = torch.load(
                        #     os.path.join(args.data_dir, '{}_img_frcnn_feats.pt'.format(name)))
                        self.img_features_env = None
                        self.img_features = os.path.join(args.data_dir, '{}_img_frcnn_feats.pt'.format(name))
            elif args.img_feat_format == 'tsv':
                self.load_img_tsv_features()
        elif args.img_feature_type == 'mask_r-cnn':
            self.img_features = torch.load(os.path.join(args.data_dir, '{}_img_mask_rcnn_feats.pt'.format(name)))
        elif args.img_feature_type.startswith('dis_code'):  # in ['dis_code', 'dis_code_t']: # discrete code
            self.img_features = torch.load(os.path.join(args.data_dir, 'vqvae',
                                                        '{}.pt'.format(name)))['feats_{}'.format(args.code_level)]
        else:
            self.img_features = torch.load(os.path.join(args.data_dir, '{}_img_feats.pt'.format(name)))

        if args.img_feature_is_folder:
            from pathlib import Path
            img_feat_path = self.img_features
            self.img_features = {}

            feat_file_name = img_feat_path.split('.')[0]
            img_features_path = os.path.join(args.data_dir, feat_file_name)

            feat_files = Path(img_features_path).glob('*.pt')
            for file in feat_files:
                file_name = str(file.name).split('.')[0]
                idx = int(file_name)
                self.img_features[idx] = str(file)

        t_end = time.time()
        logger.info('Info: loading {0} features using {1:.2f} secs'.format(name, (t_end - t_start)))

        self.output_mode = output_modes[args.task_name]

        self.examples, self.labels = _load_dataset(args, name)
        self.label_map = {label: i for i, label in enumerate(self.labels)}

        if self.args.load_fast:
            self.features = self.tensorize(
                args,
                cls_token_at_end=bool(self.args.model_type in ['xlnet']),  # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(self.args.model_type in ['xlnet']),  # pad on the left for xlnet
                pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        else:
            pass

        logger.info('%s Data Examples: %d' % (name, len(self.examples)))

    def tensorize(self,
                  cls_token_at_end=False,
                  pad_on_left=False,
                  cls_token='[CLS]',
                  sep_token='[SEP]',
                  pad_token=0,
                  sequence_a_segment_id=0,
                  sequence_b_segment_id=1,
                  cls_token_segment_id=1,
                  pad_token_segment_id=0,
                  mask_padding_with_zero=True):

        # debug:
        # debug_size = 500
        features = []

        for (ex_index, example) in enumerate(self.examples[0:]):
            if len(example.label) == 0:
                continue
            if ex_index % 10000 == 0:
                logger.info('Tensorizing example %d of %d' % (ex_index, len(self.examples)))

            tokens_a = self.tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = self.tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total length
                # is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > self.args.max_seq_length - 2:
                    tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.args.max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == self.args.max_seq_length
            assert len(input_mask) == self.args.max_seq_length
            assert len(segment_ids) == self.args.max_seq_length

            # self.init_torch_pth_file()
            #
            # # image features
            # img_feat = self.img_features[example.img_key]  # torch
            # # img_feat = self.img_features.item().get(example.img_key)  # numpy

            img_feat_path = self.img_features[example.img_key]
            img_feat = torch.load(img_feat_path)

            if img_feat.shape[0] > self.args.max_img_seq_length:
                img_feat = img_feat[0:self.args.max_img_seq_length, ]
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
            else:
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
                padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
                img_feat = torch.cat((img_feat, padding_matrix), 0)
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                    # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

            if self.args.output_mode == 'classification':
                label_id = [self.label_map[l] for l in example.label]
                score = example.score
            elif self.args.output_mode == 'regression':
                label_id = float(example.label)
            else:
                raise KeyError(self.args.output_mode)

            if ex_index < 5:
                logger.info('*** Example ***')
                logger.info('guid: %s' % (example.guid))
                logger.info('tokens: %s' % ' '.join([str(x) for x in tokens]))
                logger.info('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
                logger.info('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
                logger.info('segment_ids: %s' % ' '.join([str(x) for x in segment_ids]))
                logger.info('label: %s (id = %s)' % (example.label, label_id))
                logger.info('score: %s (score = %s)' % (example.score, score))

            new_scores = target_tensor(len(self.labels), label_id, score)
            # features.append(InputFeat(input_ids=input_ids, input_mask=input_mask, \
            # segment_ids=segment_ids, label_id=label_id, score=score, img_feat=img_feat))
            features.append((torch.tensor(input_ids, dtype=torch.long), torch.tensor(input_mask, dtype=torch.long),
                             torch.tensor(segment_ids, dtype=torch.long), torch.tensor([label_id[0]], dtype=torch.long),
                             torch.tensor(new_scores, dtype=torch.float), img_feat))

        return features

    def tensorize_example(self,
                          example,
                          cls_token_at_end=False,
                          pad_on_left=False,
                          cls_token='[CLS]',
                          sep_token='[SEP]',
                          pad_token=0,
                          sequence_a_segment_id=0,
                          sequence_b_segment_id=1,
                          cls_token_segment_id=1,
                          pad_token_segment_id=0,
                          mask_padding_with_zero=True):

        tokens_a = self.tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.args.max_seq_length - 2:
                tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(segment_ids) == self.args.max_seq_length

        # self.init_torch_pth_file()

        # image features
        if self.args.img_feature_type.startswith('dis_code'):
            img_feat = self.img_features[example.img_key]

            if self.args.img_feature_type == 'dis_code_ln':  # for discrete code image representation
                img_feat = img_feat.reshape(-1, img_feat.shape[0])

            if self.args.img_feature_type == 'dis_code_t':  # transposed
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * 64
            else:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
        else:
            if self.args.img_feat_format == 'pt':
                # img_feat = self.img_features[example.img_key]  # [:, 0:self.args.img_feature_dim]  # torch
                img_feat_path = self.img_features[example.img_key]
                img_feat = torch.load(img_feat_path)
            elif self.args.img_feat_format == 'tsv':
                img_features = self.get_img_feature(str(example.img_key))
                img_feat = torch.from_numpy(img_features)

            if img_feat.shape[0] > self.args.max_img_seq_length:
                img_feat = img_feat[0:self.args.max_img_seq_length, ]
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
            else:
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                    # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
                padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
                img_feat = torch.cat((img_feat, padding_matrix), 0)
                if self.args.max_img_seq_length > 0:
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                    # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

        if self.args.output_mode == 'classification':
            if (example.label is None):
                label_id = [0]
                score = [0]
            elif len(example.label) == 0:
                label_id = [0]
                score = [0]
            else:
                label_id = [self.label_map[l] for l in example.label]
                score = example.score
        elif self.args.output_mode == 'regression':
            if len(example.label) == 0:
                label_id = 0
            else:
                label_id = float(example.label)
        else:
            raise KeyError(self.args.output_mode)

        new_scores = target_tensor(len(self.labels), label_id, score)

        # features.append(InputFeat(input_ids=input_ids, input_mask=input_mask,\
        # segment_ids=segment_ids, label_id=label_id, score=score, img_feat=img_feat))
        if self.args.img_feature_type in ['dis_code', 'dis_code_t']:
            img_feat = img_feat.type(torch.long)
        elif self.args.img_feature_type in ['dis_code_ln']:
            # img_feat = img_feat.reshape(-1, img_feat.shape[0])
            img_feat = img_feat.type(torch.float)

        return (torch.tensor(input_ids, dtype=torch.long), torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long), torch.tensor([label_id[0]], dtype=torch.long),
                torch.tensor(new_scores, dtype=torch.float), img_feat, torch.tensor([example.q_id], dtype=torch.long))

    def __getitem__(self, index):
        if self.args.load_fast:
            example = self.features[index]
        else:
            entry = self.examples[index]
            example = self.tensorize_example(
                entry,
                cls_token_at_end=bool(self.args.model_type in ['xlnet']),  # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(self.args.model_type in ['xlnet']),  # pad on the left for xlnet
                pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        return example

    def __len__(self):
        return len(self.examples)
        # return 64

    # tsv feature loading
    def load_img_tsv_features(self):
        self.check_img_feature_file()
        self.check_img_feature_offset_map()

    def check_img_feature_file(self):
        if self.img_feature_file is None:
            img_feature_path = os.path.join(self.args.img_feat_dir, '{}_img_frcnn_feats.tsv'.format(self.name))
            t_s = time.time()
            self.img_feature_file = open(img_feature_path, 'r')
            t_e = time.time()
            logger.info('Open {} image time: {}'.format(self.name, (t_e - t_s)))

    def check_img_feature_offset_map(self):
        """load the image feature offset map."""
        if self.img_feat_offset_map is None:
            img_feature_path = os.path.join(self.args.img_feat_dir,
                                            '{}_img_frcnn_feats_offset_map.json'.format(self.name))
            t_s = time.time()
            self.img_feat_offset_map = json.load(open(img_feature_path))
            t_e = time.time()
            logger.info('Load {} images: {}, time: {}'.format(self.name, len(self.img_feat_offset_map), (t_e - t_s)))

    def get_img_feature(self, image_id):
        """decode the image feature."""
        self.check_img_feature_file()
        self.check_img_feature_offset_map()

        if image_id in self.img_feat_offset_map:
            img_offset = self.img_feat_offset_map[image_id]
            self.img_feature_file.seek(img_offset, 0)
            arr = [s.strip() for s in self.img_feature_file.readline().split('\t')]
            # num_boxes = int(arr[1])
            feat = np.frombuffer(base64.b64decode(arr[2]), dtype=np.float32).reshape((-1, self.args.img_feature_dim))
            return feat

        return None

    def init_torch_pth_file(self):
        if self.img_features_env is None:
            self.img_features = torch.load(self.img_features)
            self.img_features_env = True
