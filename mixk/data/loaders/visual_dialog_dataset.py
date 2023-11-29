from torch.utils.data import Dataset
from mixk.data.reader.visual_dialog_reader import VisDiaReader
from mixk.data.infocomp.visual_dialog_infocpler import VisDiaInfoCpler, VisualDialogDenseInfoCpler
from mixk.data.builder import DATASETS
from mixk.utils.config import mixkEasyDict
from mixk.utils.common_function import update_d1_with_d2


@DATASETS.register_module()
class VisDialDataset(Dataset):

    def __init__(self, reader, info_cpler, limit_nums=None):
        self.reader = VisDiaReader(reader)
        self.info_cpler = VisDiaInfoCpler(info_cpler)
        self._limit_sample_nums = limit_nums
        self._splits = self.reader.splits
        self._val_dense_ann = self._add_dense_annotation(reader) if 'val' in self._splits else None

    def __len__(self):
        if self._limit_sample_nums and self._limit_sample_nums > 0:
            return min(len(self.reader), self._limit_sample_nums)
        return len(self.reader)

    def __getitem__(self, idx):
        item_feature = self.reader[idx]
        if self._val_dense_ann is not None:
            self._add_dense_info(item_feature, idx)

        item = self.info_cpler.complete_info(item_feature=item_feature, split=self._splits[0])
        return item

    @staticmethod
    def _add_dense_annotation(reader_cfg: mixkEasyDict):
        dense_file = reader_cfg.mix_annotations.dense
        assert dense_file
        import json
        with open(dense_file, 'r') as f:
            return json.load(f)

    def _add_dense_info(self, item, idx):
        dense_info = self._val_dense_ann[idx]
        assert dense_info['image_id'] == item.image_id
        update_d1_with_d2(item, dense_info)


@DATASETS.register_module()
class VisualDialogDatasetDense(Dataset):

    def __init__(self, reader, info_cpler, limit_nums=None):
        self.reader = VisDiaReader(cfg=reader)
        self.info_cpler = VisualDialogDenseInfoCpler(cfg=info_cpler)
        self._limit_sample_nums = limit_nums
        self._splits = self.reader.splits
        self._dense_annotation = self._add_dense_annotation(reader_cfg=reader)

    @staticmethod
    def _add_dense_annotation(reader_cfg: mixkEasyDict):
        dense_file = getattr(reader_cfg.mix_annotations, 'dense', None)
        assert dense_file

        import json
        with open(dense_file, 'r') as f:
            return json.load(f)

    def __len__(self):
        if self._limit_sample_nums and self._limit_sample_nums > 0:
            return min(len(self.reader), self._limit_sample_nums)
        return len(self.reader)

    def _add_dense_info(self, item, idx):
        dense_info = self._dense_annotation[idx]

        assert dense_info['image_id'] == item.image_id
        update_d1_with_d2(item, dense_info)

    def __getitem__(self, idx):
        item_feature = self.reader[idx]
        self._add_dense_info(item_feature, idx)
        item = self.info_cpler.complete_info(item_feature=item_feature, split=self._splits[0])
        return item
