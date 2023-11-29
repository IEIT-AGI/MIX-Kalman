import datetime
import logging
from typing import NamedTuple, Optional

import torch

from mixk.utils.Timer import Timer
from .log_buffer_mixk import LogBufferWriter, get_log_buffer

_RecorderTime = NamedTuple(
    '_RecorderTime',
    [('iteration', int), ('time', float)],
)


class CommonMetricLoggerHook(LogBufferWriter):
    """Terminal output related information: iteration time , loss ,learning
    rate ,ETA."""

    def __init__(self, max_iter: int, max_epoch: Optional[int] = None):
        self._max_iter = max_iter
        self._max_epoch = max_epoch
        self.logger = logging.getLogger(__name__)
        self._recorder_iter_time: _RecorderTime = None

    def process_buffer_data(self):
        if self.log_buffer.by_epoch:
            self.__epoch_metric()
        else:
            self.__iter_metric()

    @staticmethod
    def __get_used_max_memory():
        if torch.cuda.is_available():
            max_used_memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            return 'max used memory:{:.0f}M'.format(max_used_memory)
        else:
            return ''

    def __get_lr(self):
        try:
            lr = '{:.6e}'.format(self.log_buffer.history('lr').latest())
        except KeyError:
            lr = 'N/A'
        finally:
            return lr

    def __get_data_time(self):
        try:
            data_time = self.log_buffer.history('data_time').avg(20)
        except KeyError:
            data_time = None
        finally:
            if data_time is not None:
                return '{:.4f}'.format(data_time)
            else:
                return ''

    def __get_by_iter_train_time(self):
        try:
            iter_time = self.log_buffer.history('iter_time').global_mean
            eta_time = self.log_buffer.history('iter_time').median(1000) * (self._max_iter - self.log_buffer.iter)
            eta_time_str = str(datetime.timedelta(seconds=int(eta_time)))
            iter_time_str = str(iter_time)
        except KeyError:
            eta_time_str = ''
            iter_time_str = ''
            if self._recorder_iter_time is not None:
                estimate_iter_time = (Timer.now() - self._recorder_iter_time.time) / (
                    self.log_buffer.iter - self._recorder_iter_time.iteration)
                eta_time = estimate_iter_time * (self._max_iter - self.log_buffer.iter)
                eta_time_str = str(datetime.timedelta(seconds=int(eta_time)))
                iter_time_str = str(datetime.timedelta(seconds=int(estimate_iter_time)))

            self._recorder_iter_time = _RecorderTime(self.log_buffer.iter, Timer.now())
        finally:
            return iter_time_str, eta_time_str

    def __collect_common_log_info(self):
        data_time = self.__get_data_time()
        iter_time, eta_time = self.__get_by_iter_train_time()
        lr = self.__get_lr()
        max_used_memory = self.__get_used_max_memory()

        common_log_info = '{losses} \t lr:{lr} \t' \
                          'load_data_time:{data_time} \t iter_time:{iter_time} \t ' \
                          '{memory} \t eta:{eta}'.format(losses=self.__get_losses_log_info(), lr=lr,
                                                         data_time=data_time,
                                                         iter_time=iter_time,
                                                         memory=max_used_memory,
                                                         eta=eta_time
                                                         )
        return common_log_info

    def __get_losses_log_info(self):
        losses_info = []
        for k, v in self.log_buffer.histories().items():
            if 'loss' in k:
                # loss_str = '{}: {:.3f}'.format(k, v.median(20))
                loss_str = '{}: {:.3f}'.format(k, v.latest())
                losses_info.append(loss_str)

        return '  '.join(losses_info)

    def __epoch_metric(self):
        epoch = self.log_buffer.epoch
        iteration = self.log_buffer.iter + 1
        single_epoch_iters = self.log_buffer.single_epoch_iters
        epoch_inner_iter = self.log_buffer.iter % single_epoch_iters + 1
        epoch_log_info_0 = 'epoch:{epoch} \t {inner_iter}/{single_epoch_iters} \t ' \
                           'current_iter:{iter} \t ' \
                           'max_iter:{max_iter} \t ' \
                           'max_epoch:{max_epoch} \t '.format(epoch=epoch,
                                                              inner_iter=epoch_inner_iter,
                                                              single_epoch_iters=single_epoch_iters,
                                                              iter=iteration,
                                                              max_iter=self._max_iter,
                                                              max_epoch=self._max_epoch)

        epoch_log_info_1 = self.__collect_common_log_info()
        self.logger.info(epoch_log_info_0 + epoch_log_info_1)

    def __iter_metric(self):
        iteration = self.log_buffer.iter + 1
        epoch_log_info_0 = 'current_iter:{iter} \t max_iter:{max_iter} \t '.format(
            iter=iteration, max_iter=self._max_iter)

        epoch_log_info_1 = self.__collect_common_log_info()
        self.logger.info(epoch_log_info_0 + epoch_log_info_1)


def write_metrics(data: dict):
    if 'loss' in data.keys():
        data.pop('loss')

    logger_buffer = get_log_buffer()
    for k, v in data.items():
        logger_buffer.put_scalar(k, v)
