# __all__ = ["MetadataCatelog",]

from .builder import build_mixk_test_loader, build_mixk_train_loader
from .loaders.clevr_loader import ClevrDATASET
from .loaders.gqa_loader import GQADATASET
from .loaders.hatefulmemes_loader import HatefulMemesDATASET
from .loaders.ocrvqa_loader import OCRVQADATASET
from .loaders.refclef_loader import RefClefDATASET
# from .loaders.referit_loader import ReferitDATASET
from .loaders.refcoco_loader import RefCOCODATASET
from .loaders.refcocog_loader import RefCOCOgDATASET
from .loaders.refcocop_loader import RefCOCOpDATASET
from .loaders.stvqa_loader import STVQADATASET
from .loaders.textvqa_loader import TEXTVQADATASET
from .loaders.vcr_loader import VCRDATASET
from .loaders.visualentailment_loader import VisualEntailmentDATASET

from .loaders.visual_dialog_dataset import VisDialDataset, VisualDialogDatasetDense
from .loaders.vqa_loader import VQADATASET
from .loaders.vizwiz_loader import VizWizDATASET

__all__ = [
    'VQADATASET',
    'GQADATASET',
    'VizWizDATASET',
    'ClevrDATASET',
    'TEXTVQADATASET',
    'STVQADATASET',
    'OCRVQADATASET',
    'VCRDATASET',
    'RefCOCODATASET',
    'RefCOCOpDATASET',
    'RefCOCOgDATASET',
    'RefClefDATASET',
    'HatefulMemesDATASET',
    'VisualEntailmentDATASET',
    'build_mixk_train_loader',
    'build_mixk_test_loader',
    'VisDialDataset',
    'VisualDialogDatasetDense',
]
