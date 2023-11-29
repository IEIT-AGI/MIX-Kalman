import logging
from .builder import HOOKS
from .base_hook import HookBase


@HOOKS.register_module()
class TextLoggerHook(HookBase):

    def __init__(self, text=''):
        super().__init__()
        self.text = text
        self.logger = logging.getLogger(__name__)

    def before_train(self):
        self.logger.info('before_train:{}'.format(self.text))

    def after_train(self):
        self.logger.info('after_train:{}'.format(self.text))

    def after_train_iter(self):
        self.logger.info('after_train_iter:{}'.format(self.text))
