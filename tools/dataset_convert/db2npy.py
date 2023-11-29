import json

import lmdb
import msgpack
from lz4.frame import decompress

db_dir = '/home/datasets/UNITER/vqa_train.db'

id2len = json.load(open(f'{db_dir}/id2len.json'))

env = lmdb.open(db_dir, readonly=True, create=False, readahead=True)

txn = env.begin(buffers=True)
ids = []
for id_ in list(id2len.keys()):
    ids.append(id_)
for key in ids:
    res = msgpack.loads(decompress(txn.get(key.encode('utf-8'))), raw=False)
    print('haha')
