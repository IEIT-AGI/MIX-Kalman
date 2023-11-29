import json
import os
import pickle

import lmdb
import numpy as np
from PIL import Image

from ..utils.stream import ItemFeature
from .base_reader import mixkDataReader


class HatefulMemesReader(mixkDataReader):

    def __init__(self, cfg):
        self.init_default_params(cfg)
        self.image_dir = cfg.image_dir
        self.imsize = cfg.img_size
        splits = cfg.datasets
        if isinstance(splits, str):
            splits = [splits]
        self.splits = splits
        self.mix_features_pathes = {split: cfg['mix_features'][split] for split in self.splits}
        self.mix_annotations_pathes = {split: cfg['mix_annotations'][split] for split in self.splits}

        self.idx_split_index = []
        self.annotations = []
        for split, mix_annotations_path in self.mix_annotations_pathes.items():
            annotations_tmp = self._load_jsonl(mix_annotations_path)
            self.annotations.extend(annotations_tmp)
            self.idx_split_index.extend([split] * len(annotations_tmp))
        self.feature_txns = list(
            set(
                list({
                    split: lmdb.open(self.mix_features_pathes[split]).begin()
                    for split in list(self.mix_features_pathes.keys())
                }.values())))

        self.annotations = [
            ann for ann in self.annotations if self.get_featureinfo_from_txns(self.feature_txns, ann['id']) is not None
        ]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        annotation = self.annotations[item]
        img_name = annotation['img'].split('/')[-1]
        img_id = annotation['id']
        label = annotation['label']
        text = annotation['text']
        img = self.load_image(img_name)
        features_info = self.get_featureinfo_from_txns(self.feature_txns, img_id)

        item_dict = {}
        item_dict.update(features_info)
        item_dict.update({
            'img_name': img_name,
            'img_id': img_id,
            'label': label,
            'text': text,
            'img': img,
        })

        return ItemFeature(item_dict)

    def _load_jsonl(self, annotation_path):
        with open(annotation_path, 'r') as f:
            vs = [json.loads(s) for s in f]
        return vs

    def get_featureinfo_from_txns(self, txns, key):
        feature_info = None
        key = str(key)
        for txn in txns:
            feature_info = txn.get(key.encode())
            if feature_info is not None:
                break
        return None if feature_info is None else pickle.loads(feature_info)

    def load_image(self, img_name):
        return np.array(
            Image.open(os.path.join(self.image_dir, img_name)).convert('RGB').resize((self.imsize, self.imsize)))
