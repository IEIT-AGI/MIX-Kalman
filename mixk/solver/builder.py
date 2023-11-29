import copy
import inspect
import logging
import torch
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from typing import List
from ..utils.registry import Registry, build_from_cfg
from mixk.utils.config import mixkEasyDict

OPTIMIZERS = Registry('optimizer')
OPTIMIZER_BUILDERS = Registry('optimizer builder')
LR_SCHEDULERS = Registry('lr scheduler')


def register_torch_optimizers() -> List:
    torch_optimizers = []
    for name in dir(torch.optim):
        if name.startswith('__'):
            continue
        else:
            optim_cls = getattr(torch.optim, name)
            if inspect.isclass(optim_cls) and issubclass(optim_cls, Optimizer):
                OPTIMIZERS.register_module()(optim_cls)
                torch_optimizers.append(optim_cls)
    return torch_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()


def register_torch_lr_schedulers() -> List:
    torch_lr_schedulers = []
    for name in dir(lr_scheduler):
        if name.startswith('__'):
            continue
        else:
            lr_cls = getattr(lr_scheduler, name)
            if inspect.isclass(lr_cls) and issubclass(lr_cls, LRScheduler):
                LR_SCHEDULERS.register_module()(lr_cls)
                torch_lr_schedulers.append(lr_cls)
    return torch_lr_schedulers


TORCH_LR_SCHEDULERS = register_torch_lr_schedulers()


def build_optimizer_constructor(cfg):
    return build_from_cfg(cfg, OPTIMIZER_BUILDERS)


def build_optimizer(cfg, model):

    def get_param() -> mixkEasyDict:
        param = mixkEasyDict()
        param.optimizer_cfg = copy.deepcopy(cfg)
        param.type = param.optimizer_cfg.pop('constructor', 'DefaultOptimizerConstructor')
        param.paramwise_cfg = param.optimizer_cfg.pop('paramwise_cfg', None)
        return param

    obj = build_optimizer_constructor(cfg=get_param())
    optimizer = obj(model)
    return optimizer


def build_lr_scheduler(lr_config, optimizer):
    logger = logging.getLogger(__name__)
    try:
        assert 'policy' in lr_config
        lr_cfg = copy.deepcopy(lr_config)
        lr_cfg.type = lr_cfg.pop('policy')
        lr_cfg.optimizer = optimizer
        lr_scheduler = build_from_cfg(lr_cfg, LR_SCHEDULERS)
    except Exception:
        logger.exception('exception during build_lr_scheduler')
        raise
    else:
        logger.info('Success in building learn rate scheduler')
        return lr_scheduler
    finally:
        logger.info('build_lr_scheduler completion')
