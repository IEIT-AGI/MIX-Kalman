import torch
import time
import os
from pathlib import Path
import tqdm


class VinVLImgFeature:

    def __init__(self, src_feat_file: str, save_path: str):
        self.src_feature_file = src_feat_file
        self.save_path = Path(save_path)
        if not self.save_path.is_dir():
            self.save_path.mkdir()

    def convert(self):
        data = torch.load(self.src_feature_file)
        print('processing -> {} files:{}'.format(self.src_feature_file, len(data)))
        time.sleep(1)
        for k, v in tqdm.tqdm(data.items()):
            if isinstance(k, str):
                file_name = k + '.pt'
                file_name = os.path.join(self.save_path, file_name)
                torch.save(v, file_name)
            elif isinstance(k, int):
                file_name = str(k) + '.pt'
                file_name = os.path.join(self.save_path, file_name)
                torch.save(v, file_name)
            else:
                pass


def gqa():
    data_root = '/home/datasets/mix_data/vinvl/datasets/gqa'
    img_feat_name = 'gqa_img_frcnn_feats.pt'
    save_dir = img_feat_name.split('.')[0]

    feat_path = os.path.join(data_root, img_feat_name)
    feat_path = os.path.normpath(feat_path)

    save_path = os.path.join(data_root, save_dir)

    feat_obj = VinVLImgFeature(src_feat_file=feat_path, save_path=save_path)
    feat_obj.convert()


def nlvr2():
    data_root = '/home/datasets/mix_data/vinvl/datasets/nlvr2'
    img_feat_name = 'nlvr2_img_frcnn_feats.pt'
    save_dir = img_feat_name.split('.')[0]

    feat_path = os.path.join(data_root, img_feat_name)
    feat_path = os.path.normpath(feat_path)

    save_path = os.path.join(data_root, save_dir)

    feat_obj = VinVLImgFeature(src_feat_file=feat_path, save_path=save_path)
    feat_obj.convert()


def vqa():
    data_root = '/home/datasets/mix_data/vinvl/datasets/vqa'
    img_feat_name = 'train_img_frcnn_feats.pt'
    # img_feat_name = 'val_img_frcnn_feats.pt'
    save_dir = img_feat_name.split('.')[0]

    feat_path = os.path.join(data_root, img_feat_name)
    feat_path = os.path.normpath(feat_path)

    save_path = os.path.join(data_root, save_dir)

    feat_obj = VinVLImgFeature(src_feat_file=feat_path, save_path=save_path)
    feat_obj.convert()


if __name__ == '__main__':
    # nlvr2()
    # gqa()
    vqa()
