from .base_hook import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class MomentumSchedulerHook(HookBase):
    pass
