import os
import numpy as np
from copy import deepcopy
import json
import tqdm
import torch
from shutil import copyfile


class IMDB_OCR_EN:
    """add ocr tokens and context_phoc to imbd_xxx_ocr_en.npy."""

    def __init__(self, imdb_file: str, phoc_dir: str):
        self.imbd_file = imdb_file
        self.phoc_dir = phoc_dir

    def process(self):
        phoc_files = self.get_all_phoc_files()
        imdb = self.read_imdb_file()
        imdb_new = deepcopy(imdb)
        for idx in tqdm.tqdm(range(1, imdb.shape[0])):
            ann = imdb[idx]
            set_name = ann['set_name']
            qid = ann['question_id']
            name_prefix = f'{set_name}_{idx - 1}_qid_{qid}.json'
            if name_prefix in phoc_files:
                with open(file=os.path.join(self.phoc_dir, name_prefix)) as f:
                    data = json.load(f)

                if qid == data['question_id'] and idx - 1 == data['idx']:
                    imdb_new[idx]['context_phoc'] = data['context_phoc']
                    imdb_new[idx]['phoc_ocr_tokens'] = data['ocr_tokens']
                    ann['ocr_tokens'] = [a.lower() for a in ann['ocr_tokens']]
                else:
                    print('ocr_tokens nomatch: ')
                # if ann['ocr_tokens'] == data['ocr_tokens']:
                #     imdb_new[idx]['context_phoc'] = data['context_phoc']
                # else:
                #     print('ocr_tokens nomatch: ')
                #     print('src:{}'.format(ann['ocr_tokens']))
                #     print('dst:{}'.format(data['ocr_tokens']))
                #     return
            else:
                print(f'no find {name_prefix}')
        save_file_name = os.path.join(self.imbd_file.split('.')[0] + '_phoc_feature.npy')
        np.save(save_file_name, imdb_new)
        print(save_file_name)

    def add_phoc_path(self):
        imdb = self.read_imdb_file()
        imdb_new = deepcopy(imdb)

        save_file_name = os.path.join(self.imbd_file.split('.')[0] + '_phoc_feature.npy')
        np.save(save_file_name, imdb_new)
        print(save_file_name)

    def get_all_phoc_files(self):
        files = os.listdir(self.phoc_dir)
        if 'train' in self.imbd_file:
            return [file_name for file_name in files if file_name.find('train') != -1]
        else:
            return [file_name for file_name in files if file_name.find('val') != -1]

    def read_imdb_file(self):
        return np.load(self.imbd_file, allow_pickle=True)


def train_set():
    data_root = '/home/datasets/mix_data/mixk/'
    annotation_path = 'data/datasets/textvqa/defaults/annotations/'
    train_en = os.path.join(data_root, annotation_path, 'imdb_train_ocr_en.npy')
    imdb_file = os.path.normpath(train_en)
    phoc_dir = '~/text_vqa_phoc'
    imdb_ocr_en = IMDB_OCR_EN(imdb_file=imdb_file, phoc_dir=phoc_dir)
    imdb_ocr_en.process()


def val_set():
    data_root = '/home/datasets/mix_data/mixk/'
    annotation_path = 'data/datasets/textvqa/defaults/annotations/'
    val_en = os.path.join(data_root, annotation_path, 'imdb_val_ocr_en.npy')
    imdb_file = os.path.normpath(val_en)
    phoc_dir = '~/text_vqa_phoc'
    imdb_ocr_en = IMDB_OCR_EN(imdb_file=imdb_file, phoc_dir=phoc_dir)
    imdb_ocr_en.process()


def build_small_dataset(data_name, size=20):
    src_data = np.load(data_name, allow_pickle=True)
    small_data = src_data[:size + 1]

    small_dataset_name = data_name.split('.')[0] + f'_small_{size}.npy'
    np.save(small_dataset_name, small_data)
    print(small_dataset_name)


def build_dataset():
    data_root = '/home/datasets/mix_data/mixk/'
    annotation_path = 'data/datasets/textvqa/defaults/annotations/'
    train_en = os.path.join(data_root, annotation_path, 'imdb_train_ocr_en_phoc_feature.npy')
    imdb_file = os.path.normpath(train_en)
    build_small_dataset(imdb_file, size=128)

    val_en = os.path.join(data_root, annotation_path, 'imdb_val_ocr_en_phoc_feature.npy')
    imdb_file = os.path.normpath(val_en)
    build_small_dataset(imdb_file, size=128)


def npy_to_pth():
    from mixk.models.vqa_models.m4c import M4C
    ann_path = '/home/datasets/mix_data/mixk/data/datasets/textvqa/defaults/annotations/'
    npy_file = 'imdb_val_ocr_en_phoc_feature.npy'
    npy_file = os.path.join(ann_path, npy_file)
    data = np.load(npy_file, allow_pickle=True)
    data = M4C.cpu_to_cuda(data, 'cpu')
    torch.save(data, 't.pt')


def modify_file_name():
    data_path = '/home/datasets/mix_data/mixk/data/datasets/textvqa/defaults/text_vqa_phoc'
    files = os.listdir(data_path)
    for file in tqdm.tqdm(files):
        name_split = file.split('_')
        del name_split[1]
        new_file_name = '_'.join(name_split)
        copyfile(os.path.join(data_path, file), os.path.join(data_path, new_file_name))


if __name__ == '__main__':
    # train_set()
    # val_set()

    # build_dataset()

    # npy_to_pth()
    modify_file_name()
