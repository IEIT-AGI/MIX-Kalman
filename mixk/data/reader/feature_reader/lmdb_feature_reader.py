import lmdb
import os
import pickle
from .feature_reader import FeatureReader, FEATURE_READERS


@FEATURE_READERS.register_module()
class LMDBFeatureReader(FeatureReader):

    def __init__(self, dataset_type, feat_path=None, max_features=None):
        super().__init__(dataset_type, feat_path=feat_path, max_features=max_features)
        self.env_db = None

    def _init_env_db(self):
        self.env_db = lmdb.open(
            path=self.feat_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=os.path.isdir(self.feat_path))
        with self.env_db.begin(write=False, buffers=True) as txn:
            keys = txn.get(b'keys')
            if keys is None:
                self.img_ids = None
                self.img_ids_index = None
            else:
                self.img_ids = pickle.loads(keys)
                self.img_ids_index = {self.img_ids[idx]: idx for idx in range(0, len(self.img_ids))}

    def _get_feature(self, key):
        with self.env_db.begin(write=False, buffers=True) as txn:
            try:
                feature = pickle.loads(txn.get(key.encode()))
            except TypeError:
                feature = None

        return feature

    def read(self, img_annotation):
        if self.env_db is None:
            self._init_env_db()

        if self.dataset_type == 'TEXTVQAREADER':
            dataset_name = img_annotation.get('set_name', None)
            img_name = img_annotation.get('image_name', None)

            # the val-set comes from train-set
            if img_annotation['image_path'].find(dataset_name) == -1:
                image_path = img_annotation['image_path']
                # self.logger.warning(f'{dataset_name} does not match {image_path} , it will be replaced')
                dataset_name = image_path.split('/')[0]
            if None in [dataset_name, img_name]:
                return None
            else:
                key = dataset_name + '/' + img_name
        elif self.dataset_type in ['GQAReader', 'VQAReader']:
            img_name = img_annotation.get('image_name', None)
            if img_name is None:
                return None
            else:
                key = img_name
        else:
            self.logger.warning(f'please add support for {self.dataset_type} type')
            raise TypeError('no support dataset type')

        return self.read_by_path(img_feat_path=key)

    def read_by_path(self, img_feat_path):
        img_feature = self._get_feature(key=img_feat_path)
        return img_feature
