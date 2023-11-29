import mixk.utils.distributed_info as comm
from mixk.utils.checkpoint import PeriodicCheckpointer
from ..base_hook import HookBase
from ..builder import HOOKS


@HOOKS.register_module()
class CheckPointHook(PeriodicCheckpointer, HookBase):

    def __init__(self, *args, **kwargs):
        self.prefix = kwargs.pop('prefix')
        self.iter_period = kwargs.get('iter_period', None)
        HookBase.__init__(self)
        PeriodicCheckpointer.__init__(self, *args, **kwargs)
        self.curr_checkpoint_name = None

    def before_train(self):
        self.max_iter = self.trainer.max_iter
        if self.trainer.by_epoch:
            self.max_epoch = self.trainer.max_epoch

    def after_train_iter(self):
        if self.trainer.by_epoch:
            if self._is_save_epoch_state():
                self._save_epoch_state()
        else:
            iter_other_info = {'by_epoch': self.trainer.by_epoch}
            self.curr_checkpoint_name = self.record_iter_checkpoint(self.trainer.iter, self.prefix, **iter_other_info)

    def after_train_epoch(self):
        self._save_epoch_state(is_epoch=True)

    def _is_save_epoch_state(self):
        if self.iter_period is None:
            return False
        else:
            next_iter = self.trainer.iter + 1
            is_divide = (self.iter_period > 0 and next_iter % self.iter_period == 0)
            is_not_epoch = (next_iter % len(self.trainer.data_loader) != 0)
            is_not_final = (next_iter != self.trainer.max_iter)

            return is_divide and is_not_epoch and is_not_final

    def _save_epoch_state(self, is_epoch=False):
        epoch_other_info = {
            'epoch_inner_iter': self.trainer.inner_iter,
            'epoch_iter': self.trainer.iter,
            'by_epoch': self.trainer.by_epoch,
            'is_epoch': is_epoch,
        }
        self.curr_checkpoint_name = self.record_epoch_checkpoint(self.trainer.epoch, self.prefix, **epoch_other_info)

    def _multi_gpus_sync(self):
        comm.synchronize()
