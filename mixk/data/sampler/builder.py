from mixk.utils.registry import Registry, build_from_cfg
import mixk.utils.distributed_info as dist_info
import logging
import torch
import inspect
from torch.utils.data.sampler import Sampler
from typing import Dict, Optional
from .sampler_adaptor import SamplerAdaptor

SAMPLER = Registry('SAMPLER')
BATCH_SAMPLER = Registry('BATCH_SAMPLER')


def register_torch_sampler():
    torch_sampler = []
    torch_batch_sampler = []
    data_module = torch.utils.data
    for name in dir(data_module):
        if name.startswith('__') or 'Sampler' not in name:
            continue
        _sampler = getattr(data_module, name)
        if inspect.isclass(_sampler) and issubclass(_sampler, Sampler):
            if name == 'BatchSampler':
                BATCH_SAMPLER.register_module()(_sampler)
                torch_batch_sampler.append(_sampler)
            else:
                SAMPLER.register_module()(_sampler)
                torch_sampler.append(_sampler)

    return torch_sampler, torch_batch_sampler


TORCH_SAMPLER, TORCH_BATCH_SAMPLER = register_torch_sampler()


def build_sampler(cfg, default_args: Optional[Dict] = None):
    world_size = dist_info.get_world_size()
    if world_size > 1:
        rank = dist_info.get_local_rank()
        default_args.update({'world_size': world_size, 'rank': rank})
        sampler_type = getattr(cfg, 'type', None)
        if sampler_type != 'DistributedSampler':
            logger = logging.getLogger(__name__)
            logger.warning(
                f'because world_size equal {world_size},expected DistributedSampler type,but got {sampler_type}')
            cfg.type = 'DistributedSampler'
    elif world_size == 1:
        sampler_type = getattr(cfg, 'type', None)
        if sampler_type == 'DistributedSampler':
            logger = logging.getLogger(__name__)
            logger.warning(f'because world_size equal {world_size},and the preset sampling is {sampler_type}, '
                           f'here it will set sampler is None')
            return None

    SamplerAdaptor.adaptor(cfg=cfg, default_args=default_args)
    return build_from_cfg(cfg, SAMPLER, default_args)


def build_batch_sampler(cfg, default_args: Optional[Dict] = None):
    SamplerAdaptor.adaptor(cfg=cfg, default_args=default_args)
    return build_from_cfg(cfg, BATCH_SAMPLER, default_args)
