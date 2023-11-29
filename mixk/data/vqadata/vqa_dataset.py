import logging

from torch.utils.data import Dataset

import mixk.utils.distributed_info as comm
from ..builder import DATASETS
from ..infocomp.vqa_infocpler import VQAInfoCpler
from ..reader.vqa_reader import VQAReader


@DATASETS.register_module()
class VQADATASET(Dataset):

    def __init__(self, vqa_reader, vqa_info_cpler, limit_nums=None):
        if comm.is_main_process():
            logger = logging.getLogger(__name__)
            logger.info('start loading vqadata')

        self.reader = VQAReader(vqa_reader, vqa_reader.datasets)
        self.infocpler = VQAInfoCpler(vqa_info_cpler)
        self._limit_sample_nums = limit_nums
        self._split = vqa_reader.datasets
        if comm.is_main_process():
            logger.info('load vqadata {} successfully'.format(vqa_reader.datasets))

    def __len__(self):
        if self._limit_sample_nums and self._limit_sample_nums > 0:
            return min(len(self.reader), self._limit_sample_nums)
        return len(self.reader)

    def __getitem__(self, idx):
        item_feature = self.reader[idx]
        item_feature = self.infocpler.completeInfo(item_feature)

        item = {
            'feature': item_feature.feature,
            'input_ids': item_feature.input_ids,
            'input_mask': item_feature.input_mask,
            'question_id': item_feature.question_id,
            'image_id': item_feature.image_id
        }

        if item_feature.answers_scores is not None:
            item['answers_scores'] = item_feature.answers_scores
        # return item_feature.feature, item_feature.input_ids, item_feature.answers_scores, item_feature.input_mask

        if 'test' in self._split or 'oneval' in self._split:
            item['quesid2ans'] = self.infocpler.qa_id2ans
        return item
