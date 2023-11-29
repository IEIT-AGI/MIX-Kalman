'''
author: shihzh
created time: 2021/3/20
'''

from torch.utils.data import Dataset
import logging
import mixk.utils.distributed_info as comm


class BaseLoader(Dataset):

    def __init__(self, reader_cls, reader, info_cpler_cls, info_cpler, limit_nums=None):
        if comm.is_main_process():
            cls_name = self.__class__.__name__
            logger = logging.getLogger(__name__)
            logger.info('start loading' + cls_name)

        self.reader = reader_cls(reader)
        self.infocpler = info_cpler_cls(info_cpler)
        self._limit_sample_nums = limit_nums
        self.splits = reader.datasets
        if comm.is_main_process():
            logger.info('load {} {} successfully'.format(cls_name, reader.datasets))

    def __len__(self):
        if self._limit_sample_nums and self._limit_sample_nums > 0:
            return min(len(self.reader), self._limit_sample_nums)
        return len(self.reader)

    def __getitem__(self, idx):
        raise NotImplementedError
