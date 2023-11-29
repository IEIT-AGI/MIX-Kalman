import os
import random

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat

from ..utils.image_utils import letterbox, random_affine
from ..utils.stream import ItemFeature
from .base_reader import mixkDataReader


class ReferitReader(mixkDataReader):

    def __init__(self, cfg):
        self.init_default_params(cfg)
        self.image_dir = cfg.image_dir
        self.mask_dir = cfg.mask_dir
        self.imfilelist = {split: cfg['annotations']['imlist'][split] for split in self.splits}
        self.querylist = {split: cfg['annotations']['query'][split] for split in self.splits}

        self.idx_split = []
        self.annotations = []
        self.imlist = []
        self.is_train = cfg.is_train
        self.augment_flip, self.augment_hsv, self.augment_affine = [True, True, True
                                                                    ] if self.is_train else [False, False, False]

        for split in self.splits:
            with open(self.querylist[split]) as f:
                querys_tmp = eval(f.read())
            for k, v in querys_tmp.items():
                self.annotations.append({
                    'img_id': k.split('_')[0],
                    'img_ref_idx': k.split('_')[1],
                    'query': v,
                })
                self.idx_split.append(split)

        self.imsize = cfg.img_size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        annotation = self.annotations[item]
        phrase = annotation['query']
        img_id = annotation['img_id']
        img_ref_idx = annotation['img_ref_idx']

        img_path = os.path.join(self.image_dir, img_id + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_id + '_' + img_ref_idx + '.mat')

        img = Image.open(img_path)
        mask = loadmat(mask_path)['segimg_t'].astype(np.uint8)
        # imga = Image.fromarray(np.concatenate(
        # [np.array(img).transpose(2, 0, 1), mask[np.newaxis, :, :]]).transpose(1, 2, 0), mode="RGBA")

        img = np.array(img)
        mask_where = np.where(mask == 0)
        x_1, y_1, x_2, y_2 = mask_where[1].min(), mask_where[0].min(), mask_where[1].max(), mask_where[0].max()
        bbox = [x_1, y_1, x_2, y_2]

        img, bbox, phrase, dw, dh = self.augment(img, bbox, phrase)

        item_dict = {
            'img': img,
            'bbox': bbox,
            'mask': mask,
            'phrase': phrase,
            'dw': dw,
            'dh': dh,
        }
        item_dict.update(annotation)
        item_feature = ItemFeature(item_dict)
        return item_feature

    def augment(self, img, bbox, phrase):
        # h, w = img.shape[0], img.shape[1]
        w = img.shape[1]
        if self.augment:
            # random horizontal flip
            if self.augment_flip and random.random() > 0.5:
                img = cv2.flip(img, 1)
                bbox[0], bbox[2] = w - bbox[2] - 1, w - bbox[0] - 1
                phrase = [
                    p.replace('right', '*&^special^&*').replace('left', 'right').replace('*&^special^&*', 'left')
                    for p in phrase
                ]
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
