import base64
import copy
import csv
import json
import random
import sys
import time

import numpy as np
from tqdm import tqdm

from ..utils.stream import ItemFeature
from .base_reader import mixkDataReader


class LXMERTPretrainReader(mixkDataReader):

    def __init__(self, cfg):
        self.cfg = cfg
        self.datasets = cfg.annotation_splits
        self.training = cfg.training
        self.annotation_splits = cfg.annotation_splits
        self.feat_splits = cfg.feat_splits
        self.is_debug = cfg.is_debug
        self.task_matched = cfg.task_matched
        self.num_random_feats = cfg.num_random_feats
        self.topk = -1 if not self.is_debug else 500

        self.annotations = []
        for split in self.annotation_splits:
            self.annotations.extend(json.load(open(cfg['lxmert_annotation'][split], 'r')))

        self.load_ans()
        self.load_img()
        self.list_qa_data()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        annotation = self.annotations[item]

        uid = annotation['uid']
        img_id = annotation['img_id']

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
        sent = annotation['sent']
        raw_sent = annotation['sent']
        if self.task_matched:
            if random.random() < 0.5:
                is_matched = 0
                other_annotation = self.annotations[random.randint(0, len(self.annotations) - 1)]
                while other_annotation['img_id'] == img_id:
                    other_annotation = self.annotations[random.randint(0, len(self.annotations) - 1)]
                sent = other_annotation['sent']

        # Label, convert answer to id
        if 'label' in annotation:
            label = annotation['label'].copy()
            for ans in list(label.keys()):
                label[self.answer_table.ans2id(ans)] = label.pop(ans)
        else:
            label = None

        # find other random features
        random_feats = [self.random_feat() for i in range(self.num_random_feats)]

        # Create target
        item_dict = {
            'uid': uid,
            'sent': sent,
            'raw_sent': raw_sent,
            'features': feats,
            'random_feats': random_feats,
            'bboxes': boxes,
            'obj_labels': obj_labels,
            'obj_confs': obj_confs,
            'attr_labels': attr_labels,
            'attr_confs': attr_confs,
            'is_matched': is_matched,
            'label': label,
        }
        item_feature = ItemFeature(item_dict)
        return item_feature

    def load_ans(self):
        # Create answer table according to the qa_sets
        self.answer_table = AnswerTable(vocab_ans_path=self.cfg.vocab_ans_path)
        print('Load an answer table of size %d.' % (len(self.answer_table.ans2id_map())))
        # Modify the answers
        for annotation in tqdm(self.annotations):
            labelf = annotation['labelf']
            for cat, labels in labelf.items():
                for label in labels:
                    for ans in list(label.keys()):
                        new_ans = self.answer_table.convert_ans(ans)
                        if self.answer_table.used(new_ans):
                            if ans != new_ans:
                                label[new_ans] = label.pop(ans)
                        else:
                            label.pop(ans)

    def load_img(self):
        # Load the dataset
        img_data = []
        for split in self.cfg['feat_splits']:
            img_data.extend(load_obj_tsv(self.cfg['lxmert_feat'][split], self.topk))

        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Filter out the dataset
        used_annotations = []
        for annotation in self.annotations:
            if annotation['img_id'] in self.imgid2img:
                used_annotations.append(annotation)

        # Flatten the dataset (into one sent + one image entries)
        self.annotations = []
        for datum in tqdm(used_annotations):
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
                        'dset': sents_cat,
                        'sent': sent
                    }
                    if labels is not None:
                        new_datum['label'] = labels[sent_idx]
                    self.annotations.append(new_datum)
        print('Use %d data in torch dataset' % (len(self.annotations)))

    def list_qa_data(self):
        # Create QA Eval Data
        self.qa_annotations = []
        print('Start loading qa dataset')
        for annotation in tqdm(self.annotations):
            if annotation.get('label') is not None:
                new_datum = copy.deepcopy(annotation)
                self.qa_annotations.append(new_datum)
        print('Loaded %d qa dataset' % (len(self.qa_annotations)))

        # uid2qa_annotation
        self.uid2qa_annotation = {}
        for annotation in self.qa_annotations:
            self.uid2qa_annotation[annotation['uid']] = annotation

    def random_feat(self):
        """Get a random obj feat from the dataset."""
        datum = self.annotations[random.randint(0, len(self.annotations) - 1)]
        img_id = datum['img_id']
        img_info = self.imgid2img[img_id]
        feat = img_info['features'][random.randint(0, len(img_info['features']) - 1)]
        return feat


class AnswerTable:
    ANS_CONVERT = {
        'a man': 'man',
        'the man': 'man',
        'a woman': 'woman',
        'the woman': 'woman',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'grey': 'gray',
    }

    def __init__(self, vocab_ans_path=''):
        self.all_ans = json.load(open(vocab_ans_path))
        self.anss = [ans['ans'] for ans in self.all_ans]
        self.ans_set = set(self.anss)

        self._id2ans_map = self.anss
        self._ans2id_map = {ans: ans_id for ans_id, ans in enumerate(self.anss)}

        assert len(self._id2ans_map) == len(self._ans2id_map)
        for ans_id, ans in enumerate(self._id2ans_map):
            assert self._ans2id_map[ans] == ans_id

    def convert_ans(self, ans):
        if len(ans) == 0:
            return ''
        ans = ans.lower()
        if ans[-1] == '.':
            ans = ans[:-1].strip()
        if ans.startswith('a '):
            ans = ans[2:].strip()
        if ans.startswith('an '):
            ans = ans[3:].strip()
        if ans.startswith('the '):
            ans = ans[4:].strip()
        if ans in self.ANS_CONVERT:
            ans = self.ANS_CONVERT[ans]
        return ans

    def ans2id(self, ans):
        return self._ans2id_map[ans]

    def id2ans(self, ans_id):
        return self._id2ans_map[ans_id]

    def ans2id_map(self):
        return self._ans2id_map.copy()

    def id2ans_map(self):
        return self._id2ans_map.copy()

    def used(self, ans):
        return ans in self.ans_set

    def all_answers(self):
        return self.anss.copy()

    @property
    def num_answers(self):
        return len(self.anss)


csv.field_size_limit(sys.maxsize)
FIELDNAMES = [
    'img_id', 'img_h', 'img_w', 'objects_id', 'objects_conf', 'attrs_id', 'attrs_conf', 'num_boxes', 'boxes', 'features'
]


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print('Start to load Faster-RCNN detected objects from %s' % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter='\t')
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print('Loaded %d images in file %s in %d seconds.' % (len(data), fname, elapsed_time))
    return data


def make_uid(img_id, dset, sent_idx):
    return '%s_%s_%03d' % (img_id, dset, sent_idx),
