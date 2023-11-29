import os
import _pickle as cPickle
from pathlib import Path
from torch import Tensor
import tqdm
from typing import Dict


class VilBertDatasetPickle:

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
                    tmp[k] = VilBertDatasetPickle.dict2convert(v)
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

    # @staticmethod
    # def tensor2list(data):
    #     if isinstance(data, list):
    #         pass
    #     elif isinstance(data, Dict)
    #         pass
    #     elif isinstance(data, Tensor):
    #         pass
    #     else:
    #         pass

    @staticmethod
    def save_data(data, file_name):
        with open(file_name, 'wb') as f:
            cPickle.dump(data, f)
        print(file_name)
        print('==' * 20)


def visual7w():
    vilbert_dataset_root = '/home/datasets/mix_data/vilbert/datasets'
    dataset_dir = 'visual7w/cache'
    dataset_path = os.path.join(vilbert_dataset_root, dataset_dir)
    dataset_path = os.path.normpath(dataset_path)

    pattern = '*cleaned.pkl'

    vbd_pickle = VilBertDatasetPickle(dataset_path=dataset_path, pattern=pattern)
    vbd_pickle.convert()


def refcocog():
    vilbert_dataset_root = '/home/datasets/mix_data/vilbert/datasets'
    dataset_dir = 'refcoco/cache'
    dataset_path = os.path.join(vilbert_dataset_root, dataset_dir)
    dataset_path = os.path.normpath(dataset_path)

    pattern = 'refcocog_*_cleaned.pkl'

    vbd_pickle = VilBertDatasetPickle(dataset_path=dataset_path, pattern=pattern)
    vbd_pickle.convert()


def Retrivalcoco():
    vilbert_dataset_root = '/home/datasets/mix_data/vilbert/datasets'
    dataset_dir = 'cocoRetreival/cache'
    dataset_path = os.path.join(vilbert_dataset_root, dataset_dir)
    dataset_path = os.path.normpath(dataset_path)

    pattern = 'RetrievalCOCO_*_cleaned.pkl'
    vbd_pickle = VilBertDatasetPickle(dataset_path=dataset_path, pattern=pattern)
    vbd_pickle.convert()


def RetrievalFlickr30k():
    vilbert_dataset_root = '/home/datasets/mix_data/vilbert/datasets'
    dataset_dir = 'flickr30k/cache'
    dataset_path = os.path.join(vilbert_dataset_root, dataset_dir)
    dataset_path = os.path.normpath(dataset_path)

    pattern = 'RetrievalFlickr30k_*_cleaned.pkl'
    vbd_pickle = VilBertDatasetPickle(dataset_path=dataset_path, pattern=pattern)
    vbd_pickle.convert()


def VisualEntailment():
    vilbert_dataset_root = '/home/datasets/mix_data/vilbert/datasets'
    dataset_dir = 'visual_entailment/cache'
    dataset_path = os.path.join(vilbert_dataset_root, dataset_dir)
    dataset_path = os.path.normpath(dataset_path)

    pattern = 'VisualEntailment_*_cleaned.pkl'
    vbd_pickle = VilBertDatasetPickle(dataset_path=dataset_path, pattern=pattern)
    vbd_pickle.convert()


def GuessWhat():
    vilbert_dataset_root = '/home/datasets/mix_data/vilbert/datasets'
    dataset_dir = 'guesswhat/cache'
    dataset_path = os.path.join(vilbert_dataset_root, dataset_dir)
    dataset_path = os.path.normpath(dataset_path)

    pattern = 'GuessWhat_*_25.pkl'
    vbd_pickle = VilBertDatasetPickle(dataset_path=dataset_path, pattern=pattern)
    vbd_pickle.convert()


def GuessWhatPointing():
    vilbert_dataset_root = '/home/datasets/mix_data/vilbert/datasets'
    dataset_dir = 'guesswhat/cache'
    dataset_path = os.path.join(vilbert_dataset_root, dataset_dir)
    dataset_path = os.path.normpath(dataset_path)

    pattern = 'GuessWhatPointing_*_256_306_cleaned.pkl'
    vbd_pickle = VilBertDatasetPickle(dataset_path=dataset_path, pattern=pattern)
    vbd_pickle.convert()


if __name__ == '__main__':
    # refcocog()
    # Retrivalcoco()
    # RetrievalFlickr30k()
    # VisualEntailment()
    # GuessWhat()
    GuessWhatPointing()
