from .vqa_dataset import OSCAR_VQADataset
from .gqa_dataset import OSCAR_GQADataset
from .nlvr2_dataset import OSCAR_NLVR2Dataset
# from .retrieval_dataset import OSCAR_RetrievalDataset
# from .captioning_dataset import OSCAR_CaptioningDataset
from .oscarplus_pretrain_dataset import OscarPretrainDataset

__all__ = [
    'OSCAR_VQADataset',
    'OSCAR_GQADataset',
    'OSCAR_NLVR2Dataset',
    'OscarPretrainDataset',
    # 'OSCAR_RetrievalDataset',
    # 'OSCAR_CaptioningDataset',
]
