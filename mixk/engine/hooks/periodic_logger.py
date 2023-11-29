from .base_hook import HookBase
from .builder import HOOKS
from .periods import LogBufferWriter
from .periods.tensorboard_logger import TensorboardLoggerHook


@HOOKS.register_module()
class PeriodicLogger(HookBase):
    """Used to write data every log_config_period times."""

    def __init__(self, loggers, log_config_period):
        super().__init__()
        for logger in loggers:
            assert isinstance(logger, LogBufferWriter), logger
        self._loggers = loggers
        self._period_iter = log_config_period

    def before_train(self):
        self._period_iter = self.trainer.cfg.log_config.period

    def after_train_iter(self):
        assert self._period_iter, self._period_iter
        write_flag = False
        if self.trainer.by_epoch is True:
            if (self.trainer.inner_iter + 1) % self._period_iter == 0:
                write_flag = True
        else:
            if (self.trainer.iter + 1) % self._period_iter == 0 or (self.trainer.iter == self.trainer.max_iter - 1):
                write_flag = True
        if write_flag:
            for logger in self._loggers:
                logger.write()

    def after_train(self):
        for logger in self._loggers:
            logger.close()

    def after_train_epoch(self):
        for logger in self._loggers:
            if isinstance(logger, TensorboardLoggerHook):
                logger.write()
