from torch import nn
from torch.utils.data import DataLoader
from mixk.utils.config import mixkEasyDict, seed_all_rng
from mixk.utils.registry import Registry, build_from_cfg
from .sampler import build_sampler, build_batch_sampler
from typing import Optional
import torch
import logging

VOCAB = Registry('vocab')
PREPROCESSOR = Registry('preprocessor')
DATASETS = Registry('dataset')
PROCESSOR = Registry('processor')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_vocab(cfg):
    """Build vocab."""
    return build_from_cfg(cfg, VOCAB)


def build_preprocessor(cfg):
    """Build preprocessor."""
    return build_from_cfg(cfg, PREPROCESSOR)


def build_processor(cfg, defualt_args=None):
    return build_from_cfg(cfg, PROCESSOR, defualt_args)


def build_dataset(dataset_cfg, default_args=None):
    return build_from_cfg(dataset_cfg, DATASETS, default_args)


class BatchCollator:
    """From a list of samples from the dataset, returns the images and
    targets."""

    def __call__(self, batch):
        return list(zip(*batch))


def build_data_loader_by_epoch(dataset, cfg, is_training=True):

    def get_cfg_param(data_cfg):
        params = mixkEasyDict()

        params.batch_size = getattr(data_cfg, 'samples_per_gpu')
        params.num_workers = getattr(data_cfg, 'workers_per_gpu')
        params.drop_last = getattr(data_cfg, 'drop_last', False)
        params.pin_memory = getattr(data_cfg, 'pin_memory', False)
        params.sampler_cfg = getattr(data_cfg, 'sampler', None)
        params.batch_sampler_cfg = getattr(data_cfg, 'batch_sampler', None)
        params.shuffle = getattr(data_cfg, 'shuffle', False)
        params.collate_fn = getattr(data_cfg, 'collate_fn', None)
        params.worker_init_fn = worker_init_fn

        return params

    params = get_cfg_param(cfg.train_data if is_training else cfg.test_data)
    sampler_cfg, batch_sampler_cfg = params.sampler_cfg, params.batch_sampler_cfg

    dataloader_param = {
        'dataset': dataset,
        'pin_memory': params.pin_memory,
        'num_workers': params.num_workers,
        'collate_fn': eval(params.collate_fn) if params.collate_fn else None,
    }

    if batch_sampler_cfg:
        batch_sampler = build_batch_sampler(batch_sampler_cfg, default_args={'dataset': dataset})
        dataloader_param.update({'batch_sampler': batch_sampler})
    else:
        if sampler_cfg:
            sampler_cfg = mixkEasyDict({'type': sampler_cfg}) if isinstance(sampler_cfg, str) else sampler_cfg
            sampler = build_sampler(sampler_cfg, default_args={'dataset': dataset})
            dataloader_param.update({'sampler': sampler})

        dataloader_param.update({
            'batch_size': params.batch_size,
            'drop_last': params.drop_last,
            'shuffle': params.shuffle,
        })

    return DataLoader(**dataloader_param)


def batch_collate_fn(batch):
    return batch


def worker_init_fn(worker_id):
    initial_seed = torch.initial_seed() % 2**31
    return seed_all_rng(initial_seed + worker_id)


def build_mixk_train_loader(cfg):
    """A data loader is created  by the following steps:

    1. Use the dataset names in config to create dataset
    2. Build PyTorch DataLoader
    """

    dataset = build_dataset(cfg.train_data.data)
    if not hasattr(cfg.train_data, 'shuffle') and getattr(cfg.train_data, 'sampler', None) != 'RandomSampler':
        cfg.train_data.shuffle = True
        logger = logging.getLogger(__name__)
        logger.info('the shuffle value of the train_data set to True')

    return build_data_loader_by_epoch(dataset, cfg, is_training=True)


def build_mixk_test_loader(cfg, dataset_name: Optional[str] = ''):
    dataset = build_dataset(cfg.test_data.data)
    return build_data_loader_by_epoch(dataset, cfg, is_training=False)
