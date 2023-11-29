from __future__ import absolute_import, division, print_function

import logging
import os
import time
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
        if args.train_data_type == 'bal':
            examples = processor.get_train_examples(args.data_dir, 'gqa_bal_qla_train.json')  # [0: debug_size]
        else:
            examples = processor.get_train_examples(args.data_dir, 'gqa_all_qla_train.json')  # [0: debug_size]
    elif name == 'val':
        if args.eval_data_type == 'bal':
            examples = processor.get_dev_examples(args.data_dir, 'gqa_bal_qla_val.json')  # [0: debug_size]
        else:
            examples = processor.get_dev_examples(args.data_dir, 'gqa_all_qla_val.json')  # [0: debug_size]
    elif name == 'train+val':  # depreciated
        if args.data_label_type == 'mask':
            examples = processor.get_train_examples(args.data_dir, 'train+val2014_qla_mrcnn.json')
        else:
            examples = processor.get_train_examples(args.data_dir, 'train+val2014_qla.json')
    elif name == 'test':  # test-submission
        if args.data_label_type == 'bal':
            examples = processor.get_test_examples(args.data_dir, 'gqa_all_qla_submission.json')
        else:
            examples = processor.get_test_examples(args.data_dir, 'gqa_all_qla_submission.json')
    elif name == 'test-dev':  # test-dev set
        if args.data_label_type == 'bal':
            examples = processor.get_dev_examples(args.data_dir, 'gqa_bal_qla_testdev.json')
        else:
            examples = processor.get_dev_examples(args.data_dir, 'gqa_all_qla_testdev.json')

    return examples, labels


@DATASETS.register_module()
class OSCAR_GQADataset(Dataset):
    """GQA Dataset."""

    # def __init__(self, args, name, img_features, tokenizer, label_pos_feats=None):
    def __init__(self, reader):
        super(OSCAR_GQADataset, self).__init__()

        # if comm.is_main_process():
        #     logger = logging.getLogger(__name__)
        logger.info('start loading gqa data')

        self.args = args = reader
        if isinstance(args.name, list):
            name = args.name[0]
        else:
            name = args.name
        self.name = name

        assert name in ['train', 'val', 'test-dev', 'test', 'train+val']

        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

        self.img_features = self._load_img_features(args)
        self.label_pos_feats = args.label_pos_feats
        self.output_mode = output_modes[args.task_name]
        self.tokenizer = tokenizer

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

            self.init_torch_pth_file()

            # image features
            img_feat = self.img_features[example.img_key]  # torch
            # img_feat = self.img_features.item().get(example.img_key)  # numpy
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
            txt_b_arr = example.text_b.split(';')
            txt_label_ixs = []
            for txt_b_ix, txt_b_ele in enumerate(txt_b_arr):
                tokens_b_ele = self.tokenizer.tokenize(txt_b_ele)
                txt_label_ixs.extend([txt_b_ix] * len(tokens_b_ele))
            txt_b = example.text_b.replace(';', ' ').strip()
            tokens_b = self.tokenizer.tokenize(txt_b)
            assert len(tokens_b) == len(txt_label_ixs)

            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)
            txt_label_ixs = txt_label_ixs[0:len(tokens_b)]

        # original
        # if example.text_b:
        #     txt_b = example.text_b.replace(';', ' ').strip()
        #     tokens_b = self.tokenizer.tokenize(txt_b)
        #     _truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)
        else:  # Account for [CLS] and [SEP] with "- 2"
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
            # img_feat = self.img_features[example.img_key]  # [:, 0:self.args.img_feature_dim]  # torch
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
            if (example.label is None):
                label_id = [0]
                # score = [0]
            elif len(example.label) == 0:
                label_id = [0]
                # score = [0]
            else:
                label_id = [self.label_map[l] for l in example.label]
                # score = example.score
        elif self.args.output_mode == 'regression':
            if len(example.label) == 0:
                label_id = 0
            else:
                label_id = float(example.label)
        else:
            raise KeyError(self.args.output_mode)

        if self.args.img_feature_type in ['dis_code', 'dis_code_t']:
            img_feat = img_feat.type(torch.long)
        elif self.args.img_feature_type in ['dis_code_ln']:
            # img_feat = img_feat.reshape(-1, img_feat.shape[0])
            img_feat = img_feat.type(torch.float)

        return (torch.tensor(input_ids, dtype=torch.long), torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long), torch.tensor([label_id[0]], dtype=torch.long),
                torch.tensor([label_id[0]], dtype=torch.long), img_feat, torch.tensor([example.q_id], dtype=torch.long))

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

    # def _load_img_features(self, args):
    #     t_start = time.time()
    #     if args.img_feature_type == 'faster_r-cnn':
    #         if args.img_feature_dim == 2048:  # object features
    #             feat_file_name = 'gqa_img_frcnn_feats_obj.pt'
    #         else:  # object + spatial features
    #             feat_file_name = 'gqa_img_frcnn_feats.pt'
    #     else:
    #         feat_file_name = 'gqa_img_frcnn_feats.pt'
    #     # img_features = torch.load(os.path.join(args.data_dir, feat_file_name))
    #     self.img_features_env = None
    #     img_features = os.path.join(args.data_dir, feat_file_name)
    #
    #     t_end = time.time()
    #     logger.info('Info: loading {0:s} features using {1:.2f} secs'.format(feat_file_name, (t_end - t_start)))
    #
    #     return img_features

    def _load_img_features(self, args):
        t_start = time.time()
        if args.img_feature_type == 'faster_r-cnn':
            if args.img_feature_dim == 2048:  # object features
                feat_file_name = 'gqa_img_frcnn_feats_obj.pt'
            else:  # object + spatial features
                feat_file_name = 'gqa_img_frcnn_feats.pt'
        else:
            feat_file_name = 'gqa_img_frcnn_feats.pt'
        # img_features = torch.load(os.path.join(args.data_dir, feat_file_name))

        img_features = {}
        if args.img_feature_is_folder:
            feat_file_name = feat_file_name.split('.')[0]
            img_features_path = os.path.join(args.data_dir, feat_file_name)
            from pathlib import Path
            feat_files = Path(img_features_path).glob('*.pt')
            for file in feat_files:
                file_name = str(file.name).split('.')[0]
                img_features[file_name] = str(file)
        else:
            self.img_features_env = None
            img_features = os.path.join(args.data_dir, feat_file_name)

        t_end = time.time()
        logger.info('Info: loading {0:s} features using {1:.2f} secs'.format(feat_file_name, (t_end - t_start)))

        return img_features

    # def init_torch_pth_file(self):
    #     if self.img_features_env is None:
    #         self.img_features = torch.load(self.img_features)
    #         self.img_features_env = True
