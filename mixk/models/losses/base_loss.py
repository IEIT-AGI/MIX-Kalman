from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.distributed as dist

from ..builder import build_loss


class BaseLoss(torch.nn.Module, metaclass=ABCMeta):
    loss_name = 'base_loss'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __str__(self):
        return self.loss_name


class Losser:

    def __init__(self, losses_cfg):
        self._loss_list: list = self.build_losses(losses_cfg)

    @classmethod
    def build_losses(cls, losses_cfg):
        loss_objs = build_loss(losses_cfg)
        return loss_objs if isinstance(loss_objs, list) else [loss_objs]

    def __call__(self, model_output: Dict):
        losses_dict = {}
        for loss_obj in self._loss_list:
            # losses = {str(loss_obj): loss_obj.forward(*args, **kwargs)}
            losses = loss_obj.forward(model_output)
            if isinstance(losses, Dict):
                losses.pop(str(loss_obj))
            else:
                losses = {str(loss_obj): losses}
            losses_dict.update(losses)

        losses, losses_log = self.parse_losses(losses=losses_dict)
        output = losses_log
        output['loss'] = losses
        return output

    @classmethod
    def parse_losses(cls, losses: Dict) -> Tuple[torch.Tensor, Dict]:

        losses_log = OrderedDict()
        for name, value in losses.items():
            losses_log[name] = value.mean()

        loss = sum(v for k, v in losses_log.items() if 'loss' in k)
        losses_log['total_loss'] = loss
        for name, value in losses_log.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = value.data.clone()
                value = loss_value.div_(dist.get_world_size())
                dist.all_reduce(value)

            losses_log[name] = value.item()
        return loss, losses_log
