# from .classifier import ClassifierLayer
# from .decoder import VISDIALPRINCIPLES_HEAD
#
# __all__ = ['ClassifierLayer', 'VISDIALPRINCIPLES_HEAD']

from .classifier_mix import (BertClassifierHead, ClassifierHead, LCGNClassiferHead, LogitClassifierHead,
                             MLPClassiferHead, R2CHead, TripleLinearHead, WeightNormClassifierHead)
from .decoder_mix import (DiscByRoundDecoderHead, DiscQtDecoderHead, LanguageDecoder, LanguageDecoderHead,
                          VisualDialogueHead)

__all__ = [
    'ClassifierHead', 'BertClassifierHead', 'MLPClassiferHead', 'LogitClassifierHead', 'LCGNClassiferHead',
    'TripleLinearHead', 'VisualDialogueHead', 'DiscQtDecoderHead', 'DiscByRoundDecoderHead', 'WeightNormClassifierHead',
    'LanguageDecoderHead', 'LanguageDecoder', 'R2CHead'
]
