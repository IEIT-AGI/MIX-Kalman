from .base_loader import BaseLoader
from ..builder import DATASETS
from ..infocomp import VisualEntailmentInfoCpler as InfoCpler


@DATASETS.register_module()
class VisualEntailmentDATASET(BaseLoader):

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
        item_feature = self.reader[idx]
        item_feature = self.infocpler.complete_info(item_feature)

        item = {
            'features': item_feature.features,
            'bbox': item_feature.bbox,
            'input_ids': item_feature.input_ids,
            'input_mask': item_feature.input_mask,
            'input_type_ids': item_feature.input_type_ids,
            'image_width': item_feature.image_width,
            'image_height': item_feature.image_height,
            'objects': item_feature.objects,
        }
        if item_feature.get('label') is not None:
            item['label'] = item_feature['label']
        return item
