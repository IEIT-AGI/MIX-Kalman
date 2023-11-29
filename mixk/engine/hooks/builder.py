from mixk.utils.registry import Registry, build_from_cfg
from typing import Optional, Dict

HOOKS = Registry('hook')


def build_hook(cfg, default_args: Optional[Dict] = None):
    return build_from_cfg(cfg=cfg, registry=HOOKS, default_args=default_args)
