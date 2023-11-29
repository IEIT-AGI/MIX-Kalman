import time

from torch.cuda.amp.autocast_mode import autocast
from .base_engine import EngineBase
from .hooks.periods import write_metrics
from .organizer import Organizer, is_mixed_precision
from mixk.engine.hooks.periods import LogBufferStorage
from mixk.utils.Timer import batch_iter


class CommonEngine(EngineBase):

    def __init__(self, model, data_loader, optimizer, loss_fn, batch_processor=None):
        super(CommonEngine, self).__init__()

        model.train()

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_processor = batch_processor

        self.data_loader = data_loader
        self.mixked_precision = False

    def run_train_iter(self, batch_data=None, data_time=None):
        assert self.model.training, '[CommonEngine] model was changed to eval model!'

        if self.batch_processor is not None:
            self.output = self.batch_processor(batch_data)
        else:
            with autocast(enabled=is_mixed_precision()):
                self.model_output = self.model(
                    batch_data,
                    cur_epoch=getattr(self, 'epoch', None),
                    cur_iter=self.iter,
                    inner_iter=getattr(self, 'inner_iter', None))
                self.output = self.loss_fn(self.model_output)

        metrics_dict = {'data_time': data_time}
        metrics_dict.update(self.output)
        write_metrics(metrics_dict)

    def run_train_epoch(self):
        assert self.model.training, '[CommonEngine] model was changed to eval model!'
        time.sleep(2)  # prevent possible deadlockduring epoch transition
        for i, batch_data, data_time in batch_iter(self.data_loader):
            self.inner_iter = i
            self.before_train_iter()
            self.run_train_iter(batch_data, data_time)
            self.after_train_iter()
            self.iter += 1

    def train_by_iter(self, start_iter: int, max_iter: int) -> None:
        self.start_iter = start_iter
        self.iter = start_iter
        self.max_iter = max_iter

        with LogBufferStorage(start_iter, by_epoch=False) as self.log_buffer:
            try:
                self.before_train()
                for self.iter, batch_data, data_time in batch_iter(self.data_loader):
                    self.before_train_iter()
                    self.run_train_iter(batch_data, data_time)
                    self.after_train_iter()
            except Exception:
                self.logger.exception('exception during training ')
                raise
            finally:
                time.sleep(1)  # wait for some hooks like logger to finish
                self.after_train()

    def train_by_epoch(self, start_epoch: int, max_epoch: int) -> None:
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.iter = self.start_iter

        with LogBufferStorage(self.start_iter, len(self.data_loader), self.start_epoch) as self.log_buffer:
            try:
                self.before_train()
                for self.epoch in range(self.start_epoch, self.max_epoch):
                    self.before_train_epoch()
                    self.run_train_epoch()
                    self.after_train_epoch()
            except Exception:
                self.logger.exception('exception during training')
                raise
            finally:
                time.sleep(1)  # wait for some hooks like logger to finish
                self.after_train()


class mixkEngine(CommonEngine):

    def __init__(self, cfg):
        self.organizer = Organizer(cfg)
        super(mixkEngine, self).__init__(
            model=self.organizer.model,
            data_loader=self.organizer.train_data_loader,
            optimizer=self.organizer.optimizer,
            loss_fn=self.organizer.losses_fn)

        self.start_iter = self.organizer.start_iter
        self.max_iter = self.organizer.max_iter
        self.start_epoch = self.organizer.start_epoch
        self.max_epoch = self.organizer.max_epoch
        self.cfg = self.organizer.cfg
        self.by_epoch = self.organizer.by_epoch
        self.is_lr_accumulation = self.organizer.is_lr_accumulation
        self.gradient_accumulation_steps = self.organizer.gradient_accumulation_steps

        self.mixked_precision = self.organizer.mixked_precision if hasattr(self.organizer,
                                                                           'mixked_precision') else False
        self.work_dir = cfg.work_dir

        self.register_hooks(self.organizer.hooks)

    def run_by_iter(self):
        super(mixkEngine, self).train_by_iter(self.start_iter, self.max_iter)

    def run_by_epoch(self):
        super(mixkEngine, self).train_by_epoch(self.start_epoch, self.max_epoch)

    def train(self):
        if self.by_epoch:
            self.run_by_epoch()
        else:
            self.run_by_iter()
