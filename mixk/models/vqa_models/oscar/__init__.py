# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .run_vqa import OSCAR
from .run_gqa import OSCAR_GQA
from .run_nlvr import OSCAR_NLVR
# from .run_retrieval import OSCAR_RETRIEVAL
# from .run_captioning import OSCAR_CAPTIONING

from .datasets import OSCAR_VQADataset
from .datasets import OSCAR_GQADataset
from .datasets import OSCAR_NLVR2Dataset
from .datasets import OscarPretrainDataset
# from .datasets import OSCAR_RetrievalDataset
# from .datasets import OSCAR_CaptioningDataset

from .postprocess_evaluator import OSCAR_DatasetConverter
from .postprocess_evaluator import OSCAR_AccuracyMetric

from .modeling.oscarplus_pretrain import OscarPreTraining

__all__ = [
    'OSCAR',
    'OSCAR_GQA',
    'OSCAR_NLVR',
    # 'OSCAR_RETRIEVAL',
    # 'OSCAR_CAPTIONING',
    'OSCAR_VQADataset',
    'OSCAR_GQADataset',
    'OSCAR_NLVR2Dataset',
    # 'OSCAR_RetrievalDataset',
    # 'OSCAR_CaptioningDataset',
    'OSCAR_RetrievalDataset',
    'OSCAR_DatasetConverter',
    'OSCAR_AccuracyMetric',
    'OscarPreTraining',
    'OscarPretrainDataset',
]
