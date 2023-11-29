from .base_loader import BaseLoader
from ..builder import DATASETS
from ..infocomp import RefCOCOInfoCpler as InfoCpler
from ..reader import RefCOCOReader as Reader


@DATASETS.register_module()
class RefCOCODATASET(BaseLoader):

    def __init__(self, reader, info_cpler, limit_nums=None):
        super().__init__(Reader, reader, InfoCpler, info_cpler, limit_nums)
        '''
        if comm.is_main_process():
            logger = logging.getLogger(__name__)
            logger.info('start loading data')

        self.reader = Reader(reader)
        self.infocpler = InfoCpler(info_cpler)
        self._limit_sample_nums = limit_nums
        self.splits = reader.datasets
        if comm.is_main_process():
            logger.info('load data {} successfully'.format(reader.datasets))
        '''

    '''
    def __len__(self):
        if self._limit_sample_nums and self._limit_sample_nums > 0:
            return min(len(self.reader), self._limit_sample_nums)
        return len(self.reader)
    '''

    def __getitem__(self, idx):
        # idx = 0
        item_feature = self.reader[idx]
        item_feature = self.infocpler.complete_info(item_feature)

        item = {
            'image': item_feature.img,
            'bbox': item_feature.bbox,
            'input_ids': item_feature.input_ids,
            'input_mask': item_feature.input_mask,
        }
        return item
