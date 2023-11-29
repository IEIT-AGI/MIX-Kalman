from .base_loader import BaseLoader
from ..builder import DATASETS
from ..infocomp.vcr_infocpler import VCRInfoCpler as InfoCpler
from ..reader.vcr_reader import VCRReader as Reader


@DATASETS.register_module()
class VCRDATASET(BaseLoader):

    def __init__(self, reader, info_cpler, limit_nums=None):
        super().__init__(Reader, reader, InfoCpler, info_cpler, limit_nums)
        '''
        if comm.is_main_process():
            logger = logging.getLogger(__name__)
            logger.info('start loading vcrdata')

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
            'questions_embeddings': item_feature.questions_embeddings,
            'questions_masks': item_feature.questions_masks,
            'questions_obj_tags': item_feature.questions_obj_tags,
            'answers_embeddings': item_feature.answers_embeddings,
            'answers_masks': item_feature.answers_masks,
            'answers_obj_tags': item_feature.answers_obj_tags,
            'label': item_feature.label,
            'segms': item_feature.segms,
            'objects': item_feature.objects,
            'boxes': item_feature.boxes,
            'image': item_feature.image,
            'max_num': item_feature.question_max_num,
            'bbox_num': item_feature.bbox_num,
        }

        return item
