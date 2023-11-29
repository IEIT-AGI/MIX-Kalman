# Copyleft 2019 project LXRT.

from collections import defaultdict
import json
import random

import numpy as np
from torch.utils.data import Dataset
from ..lxmert_qa_answer_table import AnswerTable
from .lxmert_vqa import load_obj_tsv
from mixk.data.builder import DATASETS
import logging

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, uid, sent, visual_feats=None, obj_labels=None, attr_labels=None, is_matched=None, label=None):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats
        self.obj_labels = obj_labels
        self.attr_labels = attr_labels
        self.is_matched = is_matched  # whether the visual and obj matched
        self.label = label


class LXMERTDataset:

    def __init__(self, cfg):
        """
        :param splits: The data sources to be loaded
        :param qa_sets: if None, no action
                        o.w., only takes the answers appearing in these dsets
                              and remove all unlabeled data (MSCOCO captions)
        """

        qa_sets = getattr(cfg, 'qa_sets', None)
        if qa_sets is not None:
            qa_sets = set(qa_set.lower().strip() for qa_set in qa_sets.split(','))

        self.datasets = cfg.datasets

        # Loading datasets to data
        self.data = []
        for source in self.datasets:
            with open(cfg.annotations[source]) as f:
                self.data.extend(json.load(f))
        logger.info('Load %d data from %s' % (len(self.data), self.datasets))

        # Create answer table according to the qa_sets
        self.answer_table = AnswerTable(qa_sets)
        logger.info('Load an answer table of size %d.' % (len(self.answer_table.ans2id_map())))

        # Modify the answers
        for datum in self.data:
            labelf = datum['labelf']
            for cat, labels in labelf.items():
                for label in labels:
                    for ans in list(label.keys()):
                        new_ans = self.answer_table.convert_ans(ans)
                        if self.answer_table.used(new_ans):
                            if ans != new_ans:
                                label[new_ans] = label.pop(ans)
                        else:
                            label.pop(ans)

    def __len__(self):
        return len(self.data)


def make_uid(img_id, dset, sent_idx):
    return '%s_%s_%03d' % (img_id, dset, sent_idx),


def random_word(tokens, tokenizer, word_mask_rate=0.15):
    """Masking some random tokens for Language Model task with probabilities as
    in the original BERT paper.

    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with probability
        ratio = word_mask_rate
        if prob < ratio:
            prob /= ratio

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = '[MASK]'

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab['[UNK]'])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def random_feat(feats, obj_mask_rate=0.15):
    mask_feats = feats.copy()
    feat_mask = np.zeros(len(feats), dtype=np.float32)
    for i in range(len(feats)):
        prob = random.random()
        # mask token with probability
        if prob < obj_mask_rate:
            prob /= obj_mask_rate

            # 80% randomly change token to zero feat
            if prob < 0.8:
                mask_feats[i, :] = 0.

            # 10% randomly change token to random feat
            # elif prob < 0.9:
            #     mask_feats[i, :] = train_tuple.torchdset.random_feat()
            # -> rest 10% randomly keep current feat

            # Need to predict this feat
            feat_mask[i] = 1.

    return mask_feats, feat_mask


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, lm_label_ids, visual_feats, obj_labels, is_matched, ans):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids

        self.visual_feats = visual_feats
        self.obj_labels = obj_labels

        self.is_matched = is_matched

        self.ans = ans


def convert_example_to_features(example: InputExample, max_seq_length, tokenizer) -> InputFeatures:
    """Convert a raw sample (pair of sentences as tokenized strings) into a
    proper training sample with IDs, LM labels, input_mask, CLS and SEP tokens
    etc.

    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens = tokenizer.tokenize(example.sent.strip())

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Ge random words
    masked_tokens, masked_label = random_word(tokens, tokenizer)

    # concatenate lm labels and account for CLS, SEP, SEP
    masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    # Mask & Segment Word
    lm_label_ids = ([-1] + masked_label + [-1])
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    feat, boxes = example.visual_feats
    obj_labels, obj_confs = example.obj_labels
    attr_labels, attr_confs = example.attr_labels

    # Mask Image Features:
    masked_feat, feat_mask = random_feat(feat)

    # QA answer label
    if example.label is None or len(example.label) == 0 or example.is_matched != 1:
        # 1. No label 2. Label is pruned 3. unmatched visual + language pair
        ans = -1
    else:
        keys, values = zip(*example.label.items())
        if len(keys) == 1:
            ans = keys[0]
        else:
            value_sum = sum(values)
            prob = [value / value_sum for value in values]
            choice = np.random.multinomial(1, prob).argmax()
            ans = keys[choice]

    features = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        lm_label_ids=lm_label_ids,
        visual_feats=(masked_feat, boxes),
        obj_labels={
            'obj': (obj_labels, obj_confs),
            'attr': (attr_labels, attr_confs),
            'feat': (feat, feat_mask),
        },
        is_matched=example.is_matched,
        ans=ans,
    )
    return features


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""


class LXMERTTorchDataset(Dataset):

    def __init__(self, dataset: LXMERTDataset, topk=-1):
        super().__init__()
        self.raw_dataset = dataset
        self.task_matched = getattr(self.cfg, 'task_matched', False)
        # Load the dataset
        img_data = []
        for source in self.raw_dataset.datasets:
            data = load_obj_tsv(self.cfg.features[source], topk)
            img_data.extend(data)

        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Filter out the dataset
        used_data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                used_data.append(datum)

        # Flatten the dataset (into one sent + one image entries)
        self.data = []
        for datum in used_data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:
                    labels = datum['labelf'][sents_cat]
                else:
                    labels = None
                for sent_idx, sent in enumerate(sents):
                    new_datum = {
                        'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                        'img_id': datum['img_id'],
                        'sent': sent
                    }
                    if labels is not None:
                        new_datum['label'] = labels[sent_idx]
                    self.data.append(new_datum)
        logger.info('Use %d data in torch dataset' % (len(self.data)))

    def __len__(self):
        return len(self.data)

    def random_feat(self):
        """Get a random obj feat from the dataset."""
        datum = self.data[random.randint(0, len(self.data) - 1)]
        img_id = datum['img_id']
        img_info = self.imgid2img[img_id]
        feat = img_info['features'][random.randint(0, 35)]
        return feat

    def __getitem__(self, item: int):
        datum = self.data[item]

        uid = datum['uid']
        img_id = datum['img_id']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        obj_labels = img_info['objects_id'].copy()
        obj_confs = img_info['objects_conf'].copy()
        attr_labels = img_info['attrs_id'].copy()
        attr_confs = img_info['attrs_conf'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 1
        sent = datum['sent']
        if self.task_matched:
            if random.random() < 0.5:
                is_matched = 0
                other_datum = self.data[random.randint(0, len(self.data) - 1)]
                while other_datum['img_id'] == img_id:
                    other_datum = self.data[random.randint(0, len(self.data) - 1)]
                sent = other_datum['sent']

        # Label, convert answer to id
        if 'label' in datum:
            label = datum['label'].copy()
            for ans in list(label.keys()):
                label[self.raw_dataset.answer_table.ans2id(ans)] = label.pop(ans)
        else:
            label = None

        # Create target
        example = InputExample(uid, sent, (feats, boxes), (obj_labels, obj_confs), (attr_labels, attr_confs),
                               is_matched, label)

        return example


class LXMERTEvaluator:

    def __init__(self, dataset: LXMERTDataset):
        self.raw_dataset = dataset

        # Create QA Eval Data
        self.data = []
        for datum in self.raw_dataset.data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:  # A labeled dataset
                    labels = datum['labelf'][sents_cat]
                    for sent_idx, sent in enumerate(sents):
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent': sent,
                            'dset': sents_cat,
                            'label': labels[sent_idx]
                        }
                        self.data.append(new_datum)

        # uid2datum
        self.uid2datum = {}
        for datum in self.data:
            self.uid2datum[datum['uid']] = datum

    def evaluate(self, uid2ans: dict, pprint=False):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:  # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        accu = score / cnt
        dset2accu = {}
        for dset in dset2cnt:
            dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

        if pprint:
            accu_str = 'Overall Accu %0.4f, ' % (accu)
            sorted_keys = sorted(dset2accu.keys())
            for key in sorted_keys:
                accu_str += '%s Accu %0.4f, ' % (key, dset2accu[key])
            print(accu_str)

        return accu, dset2accu

    def dump_result(self, uid2ans: dict, path):
        raise NotImplementedError


@DATASETS.register_module()
class LXMERTPretrainData(LXMERTTorchDataset):

    def __init__(self, cfg, limit_nums=-1):
        self.cfg = cfg

        dataset = self.build_dataset()
        super().__init__(dataset, topk=limit_nums)

    def build_dataset(self):
        return LXMERTDataset(self.cfg)
