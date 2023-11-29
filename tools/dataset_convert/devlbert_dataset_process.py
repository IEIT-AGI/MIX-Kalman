import os
import _pickle as cPickle
from pathlib import Path
from torch import Tensor
import tqdm
from typing import Dict


class DevlBertDatasetPickle:

    def __init__(self, dataset_path: str, pattern: str):
        self.dataset_path = dataset_path
        self.patter = pattern

    def convert(self):
        annotations_files = Path(self.dataset_path).glob(self.patter)
        for file in annotations_files:
            file = str(file)
            print('processing: {}'.format(file))

            with open(file, 'rb') as f:
                data = cPickle.load(f)
                data = self.tesnor2list(data)

                file_name = file.split('.')[0] + '_tolist.' + file.split('.')[-1]
                self.save_data(data, file_name)

    @staticmethod
    def tesnor2list(data: list):
        new_data = []
        for d in tqdm.tqdm(data):
            tmp = {}
            for k, v in d.items():
                if isinstance(v, Tensor):
                    tmp[k] = v.tolist()
                elif isinstance(v, Dict):
                    tmp[k] = DevlBertDatasetPickle.dict2convert(v)
                else:
                    tmp[k] = v
            new_data.append(tmp)

        return new_data

    @staticmethod
    def dict2convert(data: Dict):
        for k, v in data.items():
            if isinstance(v, Tensor):
                data[k] = v.tolist()
            else:
                data[k] = v

        return data

    @staticmethod
    def save_data(data, file_name):
        with open(file_name, 'wb') as f:
            cPickle.dump(data, f)
        print(file_name)
        print('==' * 20)


def vqa():
    vilbert_dataset_root = '/home/datasets/mix_data/DeVLBert/'
    dataset_dir = 'vqa/cache'
    dataset_path = os.path.join(vilbert_dataset_root, dataset_dir)
    dataset_path = os.path.normpath(dataset_path)

    # pattern = 'VQA_*_16.pkl.bk'
    pattern = 'VQA_*_16.pkl'

    vbd_pickle = DevlBertDatasetPickle(dataset_path=dataset_path, pattern=pattern)
    vbd_pickle.convert()


def refcoco_plus():
    vilbert_dataset_root = '/home/datasets/mix_data/DeVLBert/'
    dataset_dir = 'referExpression/cache'
    dataset_path = os.path.join(vilbert_dataset_root, dataset_dir)
    dataset_path = os.path.normpath(dataset_path)

    pattern = 'refcoco+_*_20_100.pkl'

    vbd_pickle = DevlBertDatasetPickle(dataset_path=dataset_path, pattern=pattern)
    vbd_pickle.convert()


def vcr():
    vilbert_dataset_root = '/home/datasets/mix_data/DeVLBert/'
    dataset_dir = 'vcr/cache'
    dataset_path = os.path.join(vilbert_dataset_root, dataset_dir)
    dataset_path = os.path.normpath(dataset_path)

    pattern = '*.pkl'

    vbd_pickle = DevlBertDatasetPickle(dataset_path=dataset_path, pattern=pattern)
    vbd_pickle.convert()


if __name__ == '__main__':
    # vqa()
    # refcoco_plus()
    vcr()
