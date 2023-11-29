from .base_hook import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class EMAIterHook(HookBase):

    def __init__(self):
        super().__init__()

    def before_train(self):
        if hasattr(self.trainer.model, 'module'):
            self.trainer.model.module.init_ema()
        else:
            self.trainer.model.init_ema()

    def after_train_iter(self):
        if hasattr(self.trainer.model, 'module'):
            self.trainer.model.module.update_ema()
        else:
            self.trainer.model.update_ema()


@HOOKS.register_module()
class EMAEpochHook(HookBase):

    def __init__(self):
        super().__init__()
        self.epochId = 0

    def before_train(self):
        self.savePath = self.trainer.work_dir

    def after_train_epoch(self):
        if hasattr(self.trainer.model, 'module'):
            self.trainer.model.module.load_bkp_state_dict()
            self.trainer.model.module.save_checkpoint_ema(self.savePath, self.epochId)
        else:
            self.trainer.model.load_bkp_state_dict()
            self.trainer.model.save_checkpoint_ema(self.savePath, self.epochId)
        self.epochId += 1
