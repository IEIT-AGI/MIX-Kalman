import os
import pickle

import lmdb
import numpy as np
from mixk.utils.config import ToExpanduser

feats_dir = '~/.cache/torch/mixk/data/datasets/gqa/defaults/features/new_output'
feats_dir = ToExpanduser.modify_path(feats_dir)

lmdb_dir = '~/.cache/torch/mixk/data/datasets/gqa/defaults/features/gqa'
lmdb_dir = ToExpanduser.modify_path(lmdb_dir)
env = lmdb.open(lmdb_dir, map_size=1099511627776)

txn = env.begin(write=True)
max_num_bbox = 0
for file in os.listdir(feats_dir):
    img_id = file.split('.')[0]
    res = np.load(os.path.join(feats_dir, file), allow_pickle=True).item()
    res['features'] = res['featrues_obj']
    res['bbox'] = res['boxes_obj']
    res['num_boxes'] = res['num_boxes_obj']
    if res['num_boxes_obj'] > max_num_bbox:
        max_num_bbox = res['num_boxes_obj']
    res['image_width'] = res['img_w']
    res['image_height'] = res['img_h']

    res_pkl = pickle.dumps(res)
    txn.put(key=img_id.encode(), value=res_pkl)

txn.commit()
env.close()
print(max_num_bbox)
