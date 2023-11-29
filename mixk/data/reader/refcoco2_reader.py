import json
import os
import pickle
import time

import numpy as np
import pyximport
from external import mask
from PIL import Image

from ..utils.file_utils import StrToBytes
from ..utils.stream import ItemFeature
from .referit_reader import ReferitReader

pyximport.install()


class RefCOCOReader(ReferitReader):

    def __init__(self, cfg):
        self.init_default_params(cfg)

        self.image_dir = cfg.image_dir
        self.instance_path = cfg['annotations']['instances']
        self.ref_card = cfg.get('ref_card', 'unc')
        self.ref_path = cfg['annotations'][self.ref_card]

        tic = time.time()
        self.data = {}
        # load annotations from data/dataset/instances.json
        instances = json.load(open(self.instance_path, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']
        self.data['refs'] = pickle.load(StrToBytes(open(self.ref_path, 'r')))
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

        self.is_train = cfg.is_train
        self.augment_flip, self.augment_hsv, self.augment_affine = [True, True, True
                                                                    ] if self.is_train else [False, False, False]
        self.imsize = cfg.get('img_size', 256)

    def __len__(self):
        return len(self.Refs_ids)

    def __getitem__(self, item):
        ref_id = self.Refs_ids[item]
        ref = self.Refs[ref_id]

        # get image and mask
        img = self.load_image_from_ref(ref)
        mask_info = self.getMask(ref)
        mask = mask_info['mask']
        # area = mask_info['area']

        # get phrases
        phrase_info = ref['sentences']
        phrase = [info['sent'] for info in phrase_info]
        mask_where = np.where(mask > 0)
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
        item_dict.update(ref)
        item_feature = ItemFeature(item_dict)
        return item_feature

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        self.Sents_ids = list(Sents)
        self.Refs_ids = list(Refs)
        print('index created.')

    def getMask(self, ref):
        # return mask, area and mask-center
        ann = self.refToAnn[ref['ref_id']]
        image = self.Imgs[ref['image_id']]
        if type(ann['segmentation'][0]) == list:  # polygon
            rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
        else:
            rle = ann['segmentation']
        m = mask.decode(rle)
        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # compute area
        area = sum(mask.area(rle))  # should be close to ann['area']
        return {'mask': m, 'area': area}

    def load_image_from_ref(self, ref):
        img_path = os.path.join(self.image_dir, '_'.join(ref['file_name'].split('_')[:-1]) + '.jpg')
        return np.array(Image.open(img_path).convert('RGB'))
