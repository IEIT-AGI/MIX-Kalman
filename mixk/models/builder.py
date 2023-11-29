from torch import nn
import torch
from mixk.utils.registry import Registry, build_from_cfg

EMBEDDING = Registry('embedding')
ENCODER = Registry('encoder')
BACKBONES = Registry('backbone')
COMBINE_LAYERS = Registry('combine_layers')
POOLERS = Registry('pooler')
HEADS = Registry('head')
LOSSES = Registry('loss')
VQA_MODELS = Registry('vqa_models')


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


def build_embedding(cfg):
    """Build neck."""
    return build(cfg, EMBEDDING)


def build_encoder(cfg):
    """Build roi extractor."""
    return build(cfg, ENCODER)


def build_pooler(cfg):
    return build(cfg, POOLERS)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_combine_layer(cfg):
    """Build shared head."""
    return build(cfg, COMBINE_LAYERS)


def build_head(cfg, default_args=None):
    """Build head."""
    return build(cfg, HEADS, default_args)


def build_loss(cfg):
    """Build loss."""
    from mixk.evaluation.evaluator_mixk import build as list_build
    return list_build(cfg, LOSSES)

    # return build(cfg, LOSSES)


def build_vqa_models(cfg):
    """Build vqa_models."""
    return build(cfg.model, VQA_MODELS)


def build_model(cfg):
    """Build models based on different input type."""
    model = build_vqa_models(cfg)
    model.to(torch.device(cfg.model_device))

    return model
