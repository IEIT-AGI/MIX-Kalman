import datetime
import logging

from mixk.utils.Timer import Timer
from .base_hook import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class IterationTimerHook(HookBase):
    """Track the time spent for each iteration (each run_train_iter call in the
    CommonEngine).

    Print a summary in the end of training.
    """

    def __init__(self, warmup_iter: int = 0):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude from timing.
        """
        super().__init__()
        self._warmup_iter = warmup_iter
        self._iter_start_time = None
        self._epoch_start_time = None
        self._total_start_time = None
        self._total_end_time = None

    def before_train(self):
        self._total_start_time = Timer.now()

    def after_train(self):
        self._total_end_time = Timer.now()
        self._write_log()

    def before_train_iter(self):
        self._iter_start_time = Timer.now()

    def after_train_iter(self):
        iter_seconds = Timer.passed_seconds(start=self._iter_start_time, end=Timer.now())
        self.trainer.log_buffer.put_scalar('iter_time', iter_seconds)

    def before_train_epoch(self):
        self._epoch_start_time = Timer.now()

    def after_train_epoch(self):
        epoch_sec = Timer.passed_seconds(self._epoch_start_time, Timer.now())
        self.trainer.log_buffer.put_scalar('epoch_time', epoch_sec, is_epoch=True)

    def _write_log(self):
        logger = logging.getLogger(__name__)
        total_time = Timer.passed_seconds(self._total_start_time, self._total_end_time)
        num_iter = self.trainer.iter + 1 - self.trainer.start_iter - self._warmup_iter
        if num_iter > 0 and total_time > 0:
            logger.info('The Whole training speed:{} iterations in {} ({:.4f} sec/iter)'.format(
                num_iter, str(datetime.timedelta(seconds=int(total_time))), total_time / num_iter))
        if self.trainer.by_epoch:
            epochs = self.trainer.max_epoch
            logger.info('The Whole training speed:{} epochs in {} ({} /epoch)'.format(
                epochs, str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(total_time / epochs)))))

        logger.info('total training time:{}'.format(str(datetime.timedelta(seconds=int(total_time)))))
