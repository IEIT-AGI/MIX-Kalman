"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Visual entailment dataset # NOTE: basically reuse VQA dataset
"""
from .vqa import (
    UNITER_VqaDataset,
    VqaTrainDataset,
    VqaEvalDataset,
    vqa_collate,
    vqa_eval_collate,
)
from mixk.data.builder import DATASETS
import logging
from .data import (TxtTokLmdb, DetectFeatLmdb)

logger = logging.getLogger(__name__)


def create_dataloader(img_path, txt_path, is_train, dset_cls, opts):
    img_db = DetectFeatLmdb(
        img_path,
        opts['conf_th'],
        opts['max_bb'],
        opts['min_bb'],
        opts['num_bb'],
        False,
    )
    txt_db = TxtTokLmdb(txt_path, opts['max_txt_len'] if is_train else -1)
    return dset_cls(txt_db, img_db)


@DATASETS.register_module()
class UNITER_VeDataset(UNITER_VqaDataset):

    def __init__(self, **kwargs):
        cls_name = self.__class__.__name__
        logger.info('start loading {}'.format(cls_name))

        # load DBs and image dirs
        opts = kwargs['datacfg'].copy()
        train_or_val = kwargs['train_or_val']
        assert train_or_val is not None
        if train_or_val:  # train
            self.dataset = create_dataloader(
                opts['train_img_db'],
                opts['train_txt_db'],
                True,
                VeTrainDataset,
                opts,
            )
            self.collate_fn = ve_collate
        else:
            self.dataset = create_dataloader(
                opts['val_img_db'],
                opts['val_txt_db'],
                False,
                VeEvalDataset,
                opts,
            )
            self.collate_fn = ve_eval_collate

        logger.info('load {} successfully'.format(cls_name))
        logger.info('Num examples = %d', len(self.dataset))


class VeTrainDataset(VqaTrainDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class VeEvalDataset(VqaEvalDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


ve_collate = vqa_collate
ve_eval_collate = vqa_eval_collate
