import os.path as osp
import random

import cv2
import numpy as np
import torch

from ..utils.image_utils import letterbox, random_affine
from ..utils.stream import ItemFeature
from .base_reader import mixkDataReader


class RefCOCOReader(mixkDataReader):

    def __init__(self, cfg):
        self.init_default_params(cfg)
        self.image_dir = cfg.image_dir
        self.imgset_dir = cfg.imgset_dir
        self.data_type = cfg.data_type
        self.imsize = cfg.img_size
        self.idx_split = []
        self.annotations = []
        self.imlist = []
        self.aug = cfg.augment
        self.is_train = cfg.is_train
        self.augment_flip, self.augment_hsv, self.augment_affine = [True, True, True
                                                                    ] if self.is_train else [False, False, False]

        for split in self.splits:
            imgset_file = '{0}_{1}.pth'.format(self.data_type, split)
            imgset_path = osp.join(self.imgset_dir, imgset_file)
            self.annotations += torch.load(imgset_path)

    def __len__(self):
        return len(self.annotations)

    def pull_item(self, idx):

        img_file, _, bbox, phrase, attri = self.annotations[idx]
        # box format: to x1y1x2y2

        bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.image_dir, img_file)
        img = cv2.imread(img_path)
        # duplicate channel if gray image
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)
        return img, phrase, bbox

    def __getitem__(self, item):
        img, phrase, bbox = self.pull_item(item)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()

        img, bbox, phrase, dw, dh = self.augment(img, bbox, phrase)
        bbox = np.array(bbox, dtype=np.float32)

        item_dict = {'img': img, 'bbox': bbox, 'phrase': phrase, 'dw': dw, 'dh': dh, 'item': item}
        item_feature = ItemFeature(item_dict)
        return item_feature

    def augment(self, img, bbox, phrase):
        # h, w = img.shape[0], img.shape[1]
        w = img.shape[1]
        # test forward accuracy without augment
        if self.aug:
            # random horizontal flip
            if self.augment_flip and random.random() > 0.5:
                img = cv2.flip(img, 1)
                bbox[0], bbox[2] = w - bbox[2] - 1, w - bbox[0] - 1
                phrase = phrase.replace('right', '*&^special^&*').replace('left',
                                                                          'right').replace('*&^special^&*', 'left')
            # random intensity, saturation change
            if self.augment_hsv:
                fraction = 0.50
                img_hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)
                a = (random.random() * 2 - 1) * fraction + 1
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)
                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                img = cv2.cvtColor(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
            img, _, ratio, dw, dh = letterbox(img, None, self.imsize)
            bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
            bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh
            # random affine transformation
            if self.augment_affine:
                img, _, bbox, M = random_affine(
                    img, None, bbox, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
        else:  # should be inference, or specified training
            img, _, ratio, dw, dh = letterbox(img, None, self.imsize)
            bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
            bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh
        return img, bbox, phrase, dw, dh
