# Copyleft 2019 project LXRT.

import json

import numpy as np
import torch
from torch.utils.data import Dataset
from .lxmert_nlvr2 import load_obj_tsv
from mixk.data.utils.stream import ItemFeature
from mixk.data.builder import DATASETS

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000


class GQADataset:
    """
    A GQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """

    def __init__(self, cfg):
        splits = cfg.datasets
        if isinstance(splits, str):
            splits = [splits]
            splits = splits.split(',')

        self.name = splits
        self.splits = splits

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            path = cfg.annotations.get(split, None)
            if path:
                self.data.extend(json.load(open(path)))
        print('Load %d data from split(s) %s.' % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {datum['question_id']: datum for datum in self.data}

        # Answers
        self.ans2label = json.load(open(cfg.answer_2_label))
        self.label2ans = json.load(open(cfg.label_2_answer))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class GQABufferLoader():

    def __init__(self):
        self.key2data = {}

    def load_data(self, path, number):
        key = '%s_%d' % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(path, topk=number)
        return self.key2data[key]


gqa_buffer_loader = GQABufferLoader()
"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""


@DATASETS.register_module()
class GQATorchDataset(Dataset):

    def __init__(self, reader):
        super().__init__()
        self.raw_dataset = GQADataset(reader)
        topk = reader.get('topk', -1)

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []
        # Always loading all the data in testdev
        if 'testdev' in self.raw_dataset.splits or 'testdev_all' in self.raw_dataset.splits:
            path = reader.img_feature.get('testdev', None)
            assert path
            img_data.extend(gqa_buffer_loader.load_data(path, -1))
        else:
            path = reader.img_feature.get('train', None)
            assert path
            img_data.extend(gqa_buffer_loader.load_data(path, topk))

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
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        item = ItemFeature()
        item.ques_id = ques_id
        item.feats = feats
        item.boxes = boxes
        item.ques = ques

        # Create target
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            # return ques_id, feats, boxes, ques, target
            item.target = target
        else:
            # return ques_id, feats, boxes, ques
            pass

        return item
