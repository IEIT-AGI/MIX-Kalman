from .train_nlvr2 import UNITER_NLVR
from .train_vqa import UNITER_VQA
from .train_ve import UNITER_VE
from .train_vcr import UNITER_VCR
from .postprocess_evaluator import UNITER_AccuracyMetric
from .postprocess_evaluator import UNITER_DatasetConverter

from .datasets import (
    UNITER_NLVR2Dataset,
    UNITER_VcrDataset,
    UNITER_VeDataset,
    UNITER_VqaDataset,
)

__all__ = [
    'UNITER_NLVR',
    'UNITER_VQA',
    'UNITER_VCR',
    'UNITER_VE',
]
