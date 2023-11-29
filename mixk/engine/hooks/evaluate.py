from .base_hook import HookBase
from .builder import HOOKS
import mixk.utils.distributed_info as comm
from mixk.utils.third_party_libs import PathManager
import logging
import json
import os

import torch
from operator import itemgetter
from shutil import copyfile
import copy
from mixk.utils.config import get_mixk_work_dir
from .periods.checkpoint import CheckPointHook
from mixk.utils.distributed_info import master_only_run
from dataclasses import dataclass


@HOOKS.register_module()
class EvaluateHook(HookBase):
    """Run an evaluation function periodically or at the end of training.

    It is executed every ``eval_iter_period`` iterations if is_run_eval == True and after epoch or the last iteration.
    """

    def __init__(self, eval_function, eval_iter_period=None, is_run_eval=True, eval_json_file='eval_result.json'):
        """
        Args:
            eval_iter_period (int): the period to run `eval_function` if is_run_eval is True
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
            is_run_eval(bool): when training based on epoch,if is_run_eval is True, eval_function will be run every
                                eval_iter_period times
            eval_json_file(str): the file used to save the evaluation metrics
        """

        super().__init__()
        self.is_run_eval = is_run_eval  # Dose it need to be evaluated when training ,if True, it will be evaluated
        self._eval_iter_period = eval_iter_period
        self._func = eval_function
        self._file_handle = PathManager.open(os.path.join(get_mixk_work_dir(), eval_json_file), 'w')
        self._all_eval_results = []

        self.mulit_metircs_dataset_type = ['VisDialDataset']

    def _do_eval(self):
        results = self._func()
        comm.synchronize()
        return results

    def after_train_iter(self):
        if not self.is_run_eval:
            return

        def is_run():
            if self._eval_iter_period is None:
                return False

            next_iter = self.trainer.iter + 1
            is_final = next_iter == self.trainer.max_iter

            is_run_flag = (self._eval_iter_period > 0 and next_iter % self._eval_iter_period == 0)
            if self.trainer.by_epoch:
                is_run_flag &= (next_iter % len(self.trainer.data_loader) != 0)
            else:
                is_run_flag |= is_final

            return is_run_flag

        if is_run():
            self._run_eval(is_epoch=True if self.trainer.by_epoch else False)

    def after_train(self):
        if hasattr(self, '_file_handle'):
            self._file_handle.close()
        if len(self._all_eval_results):
            self._best_eval_result()
        del self._func

    def after_train_epoch(self):
        if not self.is_run_eval:
            return

        self._run_eval(is_epoch=True)

    def _run_eval(self, is_epoch=False):
        results = self._do_eval()
        self._write_eval_result(results)
        self._write_to_tensorboard(results, is_epoch=is_epoch)

    @master_only_run
    def _write_to_tensorboard(self, eval_result, is_epoch=False):
        for k, v in eval_result.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, dict):
                for ki, vi in v.items():
                    self.trainer.log_buffer.put_scalar(ki, vi, is_epoch=is_epoch)
            else:
                self.trainer.log_buffer.put_scalar(k, v, is_epoch=is_epoch)

    @master_only_run
    def _write_eval_result(self, results):

        data = self._train_info()
        data.update(self._eval_result(results))

        for hk in self.trainer._hooks:
            if isinstance(hk, CheckPointHook):
                ck_name = hk.curr_checkpoint_name
                if ck_name:
                    data.update({'model_name': ck_name})
                break

        self._all_eval_results.append(data)

        self._file_handle.write(json.dumps(data, sort_keys=False) + '\n')
        self._file_handle.flush()

        self.mulit_metircs_process(data)
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def _train_info(self):
        train_info = {'iter': self.trainer.iter, 'max_iter': self.trainer.max_iter}
        if self.trainer.by_epoch:
            train_info.update({'epoch': self.trainer.epoch})
            train_info.update({'max_epoch': self.trainer.max_epoch})

        return train_info

    @staticmethod
    def _eval_result(eval_result):

        def value(v):
            return v.item() if isinstance(v, torch.Tensor) else v

        result = {k: value(v) for k, v in eval_result.items()}
        return result

    def _best_eval_result(self):

        def get_sored_key() -> list:
            keys = list(self._all_eval_results[0].keys())
            keys.remove('max_iter')
            keys.remove('iter')
            if 'model_name' in keys:
                keys.remove('model_name')

            if self.trainer.by_epoch:
                keys.remove('epoch')
                keys.remove('max_epoch')
            return keys

        def absolute_path(file_name):
            return os.path.join(self.trainer.work_dir, file_name)

        key_sort = get_sored_key()
        results = sorted(self._all_eval_results, key=itemgetter(*key_sort), reverse=True)
        best_result = copy.deepcopy(results[0])

        best_model_name = best_result.pop('model_name')

        # best info log ouput
        if self.trainer.by_epoch:
            ind1, ind2 = best_model_name.find('epoch'), best_model_name.find('_model.pth')
            if ind1 != -1 and ind2 != -1 and ind1 < ind2:
                best_info = best_model_name[ind1:ind2]
            else:
                best_iter = best_result.pop('iter')
                best_result.pop('max_iter')
                best_result.pop('max_epoch')
                best_epoch = best_result.pop('epoch')
                best_info = 'epoch{}th_iter{}'.format(best_epoch, best_iter)

        else:
            best_result.pop('max_iter')
            best_iter = best_result.pop('iter')
            best_info = 'iter{}th'.format(best_iter)

        logger = logging.getLogger(__name__)
        logger.info('In {},got the highest score{}'.format(best_info, best_result))

        # copy ck file
        copyfile(src=absolute_path(best_model_name), dst=absolute_path('best_result.pth'))

        if hasattr(self.trainer.model, 'module'):
            model = self.trainer.model.module
        else:
            model = self.trainer.model

        model_best_name = 'model_best.pth'
        logger.info('Saving the best model checkpoint to {}'.format(model_best_name))
        torch.save(model, absolute_path(model_best_name))

    def mulit_metircs_process(self, single_eval_info: dict):
        dataset_type = self.trainer.cfg.test_data.data.type
        if dataset_type in self.mulit_metircs_dataset_type:
            key = [k for k in list(single_eval_info.keys()) if k.find('metric') != -1]
            instance = eval(f'{dataset_type}Metric()')
            for k in key:
                metric_info = single_eval_info[k]
                single_eval_info[k] = metric_info[instance.metric_name]


@dataclass
class MultiMetricDataset:
    metric_name: str = None  # the most important metric name


@dataclass
class VisDialDatasetMetric(MultiMetricDataset):
    metric_name: str = 'ndcg'
