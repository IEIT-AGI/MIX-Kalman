import weakref
from abc import ABCMeta, abstractmethod
from mixk.engine.hooks import HookBase
import logging


class EngineBase(metaclass=ABCMeta):
    """Base class for mixk engine."""

    def __init__(self):
        self._hooks: list[HookBase] = []
        self.logger = logging.getLogger(__name__)

    def register_hooks(self, engine_hooks):
        for hk in engine_hooks:
            if hk is not None:
                assert isinstance(hk, HookBase), 'the current hook object must be a HookBase subclass!'
                hk.trainer = weakref.proxy(self)
                self._hooks.append(hk)

    def before_train(self):
        for hk in self._hooks:
            hk.before_train()

    def after_train(self):
        for hk in self._hooks:
            hk.after_train()

    def before_train_iter(self):
        for hk in self._hooks:
            hk.before_train_iter()

    def after_train_iter(self):
        for hk in self._hooks:
            hk.after_train_iter()
        self.log_buffer.step()
        if self.by_epoch:
            self.log_buffer.epoch_iter(self.inner_iter)

    def before_train_epoch(self):
        for hk in self._hooks:
            hk.before_train_epoch()

    def after_train_epoch(self):
        for hk in self._hooks:
            hk.after_train_epoch()
        self.log_buffer.epoch_step()

    @abstractmethod
    def run_train_epoch(self):
        pass

    @abstractmethod
    def run_train_iter(self):
        pass
