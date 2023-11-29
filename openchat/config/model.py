from dataclasses import dataclass
from mixk.utils.registry import Registry
from .default import root_path
import os

MODEL_CFG = Registry('model_cfg')


@dataclass
class ModelConfig:
    model: dict
    task: list


@dataclass
@MODEL_CFG.register_module()
class LXMERT(ModelConfig):
    model = dict(type='LxmertBot', weight=os.path.join(root_path, 'model_pth/lxmert_vqa.pth'))
    task = [
        dict(type='vqa', answer_table='/home/datasets/mix_data/lxmert/vqa/trainval_label2ans.json'),
    ]
