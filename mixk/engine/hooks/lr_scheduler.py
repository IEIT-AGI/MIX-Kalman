from collections import Counter
import torch
from .base_hook import HookBase, PriorityStatus
from .builder import HOOKS


@HOOKS.register_module()
class LRSchedulerHook(HookBase):
    """A hook which executes a torch builtin LR scheduler and summarizes the
    LR."""

    def __init__(self, optimizer: torch.optim.Optimizer, scheduler):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        super().__init__()
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._best_param_group_idx = self._get_best_parm_group_idx()
        self._level = PriorityStatus.HIGH

    def after_train_iter(self):

        if self.trainer.is_lr_accumulation and (self.trainer.iter + 1) % self.trainer.gradient_accumulation_steps == 0:
            is_step = True
        elif self.trainer.is_lr_accumulation is False:
            is_step = True
        else:
            is_step = False

        if is_step:
            self._record_lr_log()
            self._scheduler.step()

    def _get_best_parm_group_idx(self):
        longest = max(len(pg['params']) for pg in self._optimizer.param_groups)
        if longest == 1:
            lr_counter = Counter([pg['lr'] for pg in self._optimizer.param_groups])
            lr_largest = lr_counter.most_common()[0][0]
            for idx, pg in enumerate(self._optimizer.param_groups):
                if pg['lr'] == lr_largest:
                    return idx
        else:
            for idx, pg in enumerate(self._optimizer.param_groups):
                if len(pg['params']) == longest:
                    return idx

    def _record_lr_log(self):
        lr = self._optimizer.param_groups[self._best_param_group_idx]['lr']
        self.trainer.log_buffer.put_scalar('lr', lr, smoothing_hint=False)
