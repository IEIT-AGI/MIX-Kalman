import torch
import os
import tqdm
from pathlib import Path
import time


class OscarImgFeature:

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


# def GQA():
#     data_root = '/home/datasets/mix_data/oscar/datasets/GQA/0.4true'
#     save_dir = 'gqa_img_frcnn_feats'
#     img_feat_name = 'gqa_img_frcnn_feats.pt'  # [37,2054]
#
#     feat_path = os.path.join(data_root, img_feat_name)
#     feat_path = os.path.normpath(feat_path)
#
#     save_path = os.path.join(data_root, save_dir)
#
#     data = torch.load(feat_path)
#     print('{} files:{}'.format(img_feat_name, len(data)))
#     for k, v in tqdm.tqdm(data.items()):
#         file_name = k + '.pt'
#         file_name = os.path.join(save_path, file_name)
#         torch.save(v, file_name)


def gqa():
    data_root = '/home/datasets/mix_data/oscar/datasets/GQA/0.4true'
    save_dir = 'gqa_img_frcnn_feats'
    img_feat_name = 'gqa_img_frcnn_feats.pt'  # [37,2054]

    feat_path = os.path.join(data_root, img_feat_name)
    feat_path = os.path.normpath(feat_path)

    save_path = os.path.join(data_root, save_dir)

    feat_obj = OscarImgFeature(src_feat_file=feat_path, save_path=save_path)
    feat_obj.convert()


def vqa():
    data_root = '/home/datasets/mix_data/oscar/datasets/vqa/2k'
    # img_feat_name = 'train_img_frcnn_feats.pt'
    # img_feat_name = 'train+val_img_frcnn_feats.pt'
    img_feat_name = 'val_img_frcnn_feats.pt'
    save_dir = img_feat_name.split('.')[0]

    feat_path = os.path.join(data_root, img_feat_name)
    feat_path = os.path.normpath(feat_path)

    save_path = os.path.join(data_root, save_dir)

    feat_obj = OscarImgFeature(src_feat_file=feat_path, save_path=save_path)
    feat_obj.convert()


def nlvr2():
    data_root = '/home/datasets/mix_data/oscar/datasets/nlvr2/ft_corpus'
    img_feat_name = 'nlvr2_img_frcnn_feats.pt'
    save_dir = img_feat_name.split('.')[0]

    feat_path = os.path.join(data_root, img_feat_name)
    feat_path = os.path.normpath(feat_path)

    save_path = os.path.join(data_root, save_dir)

    feat_obj = OscarImgFeature(src_feat_file=feat_path, save_path=save_path)
    feat_obj.convert()


if __name__ == '__main__':
    vqa()
    # nlvr2()
