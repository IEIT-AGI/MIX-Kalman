"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

VQA dataset
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index
from mixk.data.builder import DATASETS
import logging
from .data import (TxtTokLmdb, ImageLmdbGroup, ConcatDatasetWithLens)
import json

logger = logging.getLogger(__name__)


def _get_vqa_target(example, num_answers):
    target = torch.zeros(num_answers)
    labels = example['target']['labels']
    scores = example['target']['scores']
    if labels and scores:
        target.scatter_(0, torch.tensor(labels), torch.tensor(scores))
    return target


@DATASETS.register_module()
class UNITER_VqaDataset(DetectFeatTxtTokDataset):

    def __init__(self, **kwargs):
        cls_name = self.__class__.__name__
        logger.info('start loading {}'.format(cls_name))

        opts = kwargs['datacfg'].copy()
        train_or_val = kwargs['train_or_val']
        assert train_or_val is not None

        # load DBs and image dirs
        all_img_dbs = ImageLmdbGroup(
            opts['conf_th'],
            opts['max_bb'],
            opts['min_bb'],
            opts['num_bb'],
            False,
        )
        ans2label = json.load(open(opts['ans2label_file']))

        if train_or_val:  # train
            train_datasets = []
            for txt_path, img_path in zip(opts['train_txt_dbs'], opts['train_img_dbs']):
                img_db = all_img_dbs[img_path]
                txt_db = TxtTokLmdb(txt_path, opts['max_txt_len'])
                train_datasets.append(VqaTrainDataset(len(ans2label), txt_db, img_db))
            self.dataset = ConcatDatasetWithLens(train_datasets)
        else:
            val_img_db = all_img_dbs[opts['val_img_db']]
            val_txt_db = TxtTokLmdb(opts['val_txt_db'], -1)
            self.dataset = VqaEvalDataset(len(ans2label), val_txt_db, val_img_db)

        self.collate_fn = vqa_collate if train_or_val else vqa_eval_collate
        logger.info('load {} successfully'.format(cls_name))
        logger.info('Num examples = %d', len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]


class VqaTrainDataset(DetectFeatTxtTokDataset):

    def __init__(self, num_answers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_answers = num_answers
        self.collate_fn = vqa_collate

    def __getitem__(self, i):
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(example['img_fname'])

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        target = _get_vqa_target(example, self.num_answers)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, target


def vqa_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'img_feat': img_feat,
        'img_pos_feat': img_pos_feat,
        'attn_masks': attn_masks,
        'gather_index': gather_index,
        'targets': targets
    }
    return batch


class VqaEvalDataset(VqaTrainDataset):

    def __init__(self, num_answers, *args, **kwargs):
        super().__init__(num_answers, *args, **kwargs)
        self.collate_fn = vqa_eval_collate

    def __getitem__(self, i):
        qid = self.ids[i]
        example = DetectFeatTxtTokDataset.__getitem__(self, i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(example['img_fname'])

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        if 'target' in example:
            target = _get_vqa_target(example, self.num_answers)
        else:
            target = None

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return qid, input_ids, img_feat, img_pos_feat, attn_masks, target


def vqa_eval_collate(inputs):
    (qids, input_ids, img_feats, img_pos_feats, attn_masks, targets) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    if targets[0] is None:
        targets = None
    else:
        targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {
        'qids': qids,
        'input_ids': input_ids,
        'position_ids': position_ids,
        'img_feat': img_feat,
        'img_pos_feat': img_pos_feat,
        'attn_masks': attn_masks,
        'gather_index': gather_index,
        'targets': targets
    }
    return batch
