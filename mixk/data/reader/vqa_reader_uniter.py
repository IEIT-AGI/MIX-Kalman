import json
import os
import io
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import torch
from lz4.frame import compress, decompress
from tqdm import tqdm
from collections import defaultdict

msgpack_numpy.patch()


def _fp16_to_fp32(feat_dict):
    out = {k: arr.astype(np.float32) if arr.dtype == np.float16 else arr for k, arr in feat_dict.items()}
    return out


def compute_num_bb(confs, conf_th, min_bb, max_bb):
    num_bb = max(min_bb, (confs > conf_th).sum())
    num_bb = min(max_bb, num_bb)
    return num_bb


def _get_vqa_target(example, num_answers):
    target = torch.zeros(num_answers)
    labels = example['target']['labels']
    scores = example['target']['scores']
    if labels and scores:
        target.scatter_(0, torch.tensor(labels), torch.tensor(scores))
    return target


class DetectFeatLmdb(object):

    def __init__(self, img_dir, conf_th=0.2, max_bb=100, min_bb=10, num_bb=36, compress=True):
        self.img_dir = img_dir
        if conf_th == -1:
            db_name = f'feat_numbb{num_bb}'
            self.name2nbb = defaultdict(lambda: num_bb)
        else:
            db_name = f'feat_th{conf_th}_max{max_bb}_min{min_bb}'
            nbb = f'nbb_th{conf_th}_max{max_bb}_min{min_bb}.json'
            if not os.path.exists(f'{img_dir}/{nbb}'):
                # nbb is not pre-computed
                self.name2nbb = None
            else:
                self.name2nbb = json.load(open(f'{img_dir}/{nbb}'))
        self.compress = compress
        if compress:
            db_name += '_compressed'

        if self.name2nbb is None:
            if compress:
                db_name = 'all_compressed'
            else:
                db_name = 'all'
        # only read ahead on single node training
        self.env = lmdb.open(
            f'{img_dir}/{db_name}',
            readonly=True,
            create=False,
            # readahead=not _check_distributed())
            readahead=True)
        self.txn = self.env.begin(buffers=True)
        if self.name2nbb is None:
            self.name2nbb = self._compute_nbb()

    def _compute_nbb(self):
        name2nbb = {}
        fnames = json.loads(self.txn.get(key=b'__keys__').decode('utf-8'))
        for fname in tqdm(fnames, desc='reading images'):
            dump = self.txn.get(fname.encode('utf-8'))
            if self.compress:
                with io.BytesIO(dump) as reader:
                    img_dump = np.load(reader, allow_pickle=True)
                    confs = img_dump['conf']
            else:
                img_dump = msgpack.loads(dump, raw=False)
                confs = img_dump['conf']
            name2nbb[fname] = compute_num_bb(confs, self.conf_th, self.min_bb, self.max_bb)

        return name2nbb

    def __del__(self):
        self.env.close()

    def get_dump(self, file_name):
        # hack for MRC
        dump = self.txn.get(file_name.encode('utf-8'))
        nbb = self.name2nbb[file_name]
        if self.compress:
            with io.BytesIO(dump) as reader:
                img_dump = np.load(reader, allow_pickle=True)
                img_dump = _fp16_to_fp32(img_dump)
        else:
            img_dump = msgpack.loads(dump, raw=False)
            img_dump = _fp16_to_fp32(img_dump)
        img_dump = {k: arr[:nbb, ...] for k, arr in img_dump.items()}
        return img_dump

    def __getitem__(self, file_name):
        dump = self.txn.get(file_name.encode('utf-8'))
        nbb = self.name2nbb[file_name]
        if self.compress:
            with io.BytesIO(dump) as reader:
                img_dump = np.load(reader, allow_pickle=True)
                img_dump = {'features': img_dump['features'], 'norm_bb': img_dump['norm_bb']}
        else:
            img_dump = msgpack.loads(dump, raw=False)
        img_feat = torch.tensor(img_dump['features'][:nbb, :]).float()
        img_bb = torch.tensor(img_dump['norm_bb'][:nbb, :]).float()
        return img_feat, img_bb


class TxtLmdb(object):

    def __init__(self, db_dir, readonly=True):
        self.readonly = readonly
        if readonly:
            # training
            self.env = lmdb.open(
                db_dir,
                readonly=True,
                create=False,
                # readahead=not _check_distributed())
                readahead=True)
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            # prepro
            self.env = lmdb.open(db_dir, readonly=False, create=True, map_size=4 * 1024**4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0

    def __del__(self):
        if self.write_cnt:
            self.txn.commit()
        self.env.close()

    def __getitem__(self, key):
        return msgpack.loads(decompress(self.txn.get(key.encode('utf-8'))), raw=False)

    def __setitem__(self, key, value):
        # NOTE: not thread safe
        if self.readonly:
            raise ValueError('readonly text DB')
        ret = self.txn.put(key.encode('utf-8'), compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        return ret


class TxtTokLmdb(object):

    def __init__(self, db_dir, max_txt_len=60):
        if max_txt_len == -1:
            self.id2len = json.load(open(f'{db_dir}/id2len.json'))
        else:
            self.id2len = {
                id_: len_
                for id_, len_ in json.load(open(f'{db_dir}/id2len.json')).items() if len_ <= max_txt_len
            }
        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

    def __getitem__(self, id_):
        txt_dump = self.db[id_]
        return txt_dump

    def combine_inputs(self, *inputs):
        input_ids = [self.cls_]
        for ids in inputs:
            input_ids.extend(ids + [self.sep])
        return torch.tensor(input_ids)

    @property
    def txt2img(self):
        txt2img = json.load(open(f'{self.db_dir}/txt2img.json'))
        return txt2img

    @property
    def img2txts(self):
        img2txts = json.load(open(f'{self.db_dir}/img2txts.json'))
        return img2txts
