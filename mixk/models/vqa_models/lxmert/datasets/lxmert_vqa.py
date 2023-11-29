import json
import torch
from torch.utils.data import Dataset
import sys
import csv
import base64
import time
from mixk.data.builder import DATASETS
from mixk.data.utils.stream import ItemFeature

import numpy as np

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


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """

    def __init__(self, cfg):
        splits = cfg.datasets
        if isinstance(splits, str):
            splits = [splits]
            splits = splits.split(',')

        self.name = splits

        self.splits = splits

        # Loading datasets
        self.data = []
        for split in self.splits:
            path = cfg.annotations.get(split, None)
            if path:
                self.data.extend(json.load(open(path)))
        print('Load %d data from split(s) %s.' % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {datum['question_id']: datum for datum in self.data}

        # Answers
        self.ans2label = json.load(open(cfg.answer_2_label))
        self.label2ans = json.load(open(cfg.label_2_answer))
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""


@DATASETS.register_module()
class VQATorchDataset(Dataset):

    def __init__(self, reader):
        super().__init__()
        self.raw_dataset = VQADataset(reader)

        topk = reader.get('topk', None)

        # Loading detection features to img_data
        img_data = []
        for split in self.raw_dataset.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_obj_tsv(reader.img_feature.get(split), topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print('Use %d data in torch dataset' % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        # np.testing.assert_array_less(boxes, 1 + 1e-5)
        # np.testing.assert_array_less(-boxes, 0 + 1e-5)

        item = ItemFeature()
        item.ques_id = ques_id
        item.feats = feats
        item.boxes = boxes
        item.ques = ques

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            # return ques_id, feats, boxes, ques, target
            item.target = target
        else:
            # return ques_id, feats, boxes, ques
            pass

        return item
