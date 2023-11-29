import logging
import os
from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel

import mixk.engine.hooks as hooks
import mixk.utils.distributed_info as comm
from mixk.models import build_loss, build_model
from mixk.solver import build_lr_scheduler, build_optimizer
from mixk.utils.mixk_checkpoint import mixkCheckpointer
from mixk.utils.logger import setup_logger
from ..data import build_mixk_test_loader, build_mixk_train_loader
from ..models.losses.base_loss import Losser
from ..evaluation import inference_on_dataset

_AUTOMATIC_mixkED_PRECISION = False
_BY_ITER_TRAIN = False


def is_mixed_precision():
    return _AUTOMATIC_mixkED_PRECISION


def get_masked_fill_value():
    if is_mixed_precision():
        return torch.finfo(torch.float16).min
    else:
        return -1e9


def is_multi_gpus_mixked_precision():
    if comm.get_world_size() > 1 and is_mixed_precision():
        return True
    else:
        return False


def is_by_iter():
    return _BY_ITER_TRAIN


class Organizer:

    def __init__(self, cfg):
        assert cfg, 'cfg must be non-empty!'

        logger = logging.getLogger('mixk')
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        self.cfg = cfg
        self.gradient_accumulation_steps = self.cfg.get('gradient_accumulation_steps', 1)
        self.is_lr_accumulation = self.cfg.get('is_lr_accumulation', True)
        self._model_name = self.cfg.model.type
        self._dataset_name = self.cfg.dataset_type

        self.model = self.build_model(cfg)
        self.losses_fn = Losser(cfg.loss)

        self.train_data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[comm.get_local_rank()],
                output_device=comm.get_local_rank(),
                find_unused_parameters=True)

        self.adaptor_optimizer(cfg.optimizer)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        self.checkpointer = mixkCheckpointer(
            self.model, cfg.work_dir, optimizer=self.optimizer, scheduler=self.scheduler)

        self._by_epoch = False if hasattr(cfg, 'by_iter') else True
        self.start_epoch = 0
        self.start_iter = 0
        self.max_iter = cfg.total_epochs * len(self.train_data_loader) if self.by_epoch else cfg.max_iter
        if self.is_lr_accumulation and self.gradient_accumulation_steps > 1:
            self.max_iter //= self.gradient_accumulation_steps

        self.max_epoch = cfg.total_epochs if self.by_epoch else 0

        self.hooks = self.build_hooks()
        if cfg.get('custom_hooks', None):
            self.add_custom_hooks(custom_hooks_cfg=cfg.custom_hooks)

        self.set_by_iter()

        if cfg.get('load_from', None) is not None:
            self.resume_or_load(cfg.load_from, resume=False)
        elif cfg.get('resume_from', None) is not None:
            self.resume_or_load(cfg.resume_from, resume=True)

        logger.info('Created Organizer')

    @classmethod
    def build_model(cls, cfg):

        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info('build model:\n {} '.format(model))

        return model

    @classmethod
    def build_loss(cls, cfg):
        losses_fn = build_loss(cfg.loss)
        return losses_fn

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg.optimizer, model)

    def adaptor_optimizer(self, optimizer):
        if optimizer.type == 'BertAdam':
            optimizer.t_total = len(self.train_data_loader) * self.cfg.total_epochs

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        if hasattr(cfg, 'lr_config'):
            return build_lr_scheduler(cfg.lr_config, optimizer)
        else:
            return None

    @classmethod
    def build_train_loader(cls, cfg):
        return build_mixk_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_mixk_test_loader(cfg, dataset_name)

    def build_hooks(self):

        def warmup_iter():
            if hasattr(self.cfg, 'lr_config'):
                lr_cfg = self.cfg.lr_config
                return lr_cfg.get('warmup_iterations', 0) if lr_cfg.get('use_warmup', False) else 0
            else:
                if hasattr(self.optimizer, 'warmup'):
                    return len(self.train_data_loader) * self.cfg.total_epochs * self.optimizer.warmup
                else:
                    return 0

        cfg = self.cfg
        hook_list = []

        if hasattr(self.cfg, 'fp16'):
            hook_list.append(hooks.Fp16OptimizerHook(self.cfg.optimizer_config.grad_clip, self.cfg.fp16))
            self.set_mixked_precision(True)
        else:
            hook_list.append(hooks.OptimizerHook(self.cfg.optimizer_config.grad_clip))

        hook_list.append(hooks.LRSchedulerHook(self.optimizer, self.scheduler))
        hook_list.append(hooks.IterationTimerHook(warmup_iter=warmup_iter()))

        if comm.is_main_process():
            # add periodcLogger
            hook_list.append(hooks.PeriodicLogger(self.build_writers(), cfg.log_config.period))

            # add checkpointHook
            kwargs = {'prefix': f'{self.model_name}_{self.dataset_name}'}
            if hasattr(self.cfg, 'checkpoint_config'):
                kwargs.update(self.cfg.checkpoint_config)
            hook_list.append(hooks.CheckPointHook(self.checkpointer, **kwargs))

        if hasattr(cfg, 'test_data') and getattr(cfg.test_data, 'is_run_eval', True):
            hook_list.append(self.add_evaluate_hook())

        hook_list.sort(key=lambda obj: obj.level.value)
        return hook_list

    def build_writers(self):
        cm_hk = hooks.CommonMetricLoggerHook(self.max_iter, self.max_epoch if self.by_epoch else None)
        json_hk = hooks.JSONLoggerHook(os.path.join(self.cfg.work_dir, 'training_status.json'))
        tb_hk = hooks.TensorboardLoggerHook(self.cfg.work_dir)

        writers = [cm_hk, json_hk, tb_hk]
        return writers

    def __getattr__(self, item):
        return self.hooks

    @classmethod
    def test(cls, cfg, model):
        """
                    Args:
                        cfg (CfgNode):
                        model (nn.Module):
                    Returns:
                        dict: a dict of result metrics
                    """
        logger = logging.getLogger(__name__)

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.test_datasets):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            results_i = inference_on_dataset(cfg, model, data_loader)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i,
                    dict), 'Evaluator must return a dict on the main process. Got {} instead.'.format(results_i)
                logger.info('Evaluation results for {} in csv format:'.format(dataset_name))

        if len(results) == 1:
            results = list(results.values())[0]

        logger.info('test finish')
        return results

    def add_evaluate_hook(self):

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        def evaluate_hook_param():
            kwargs = {'eval_function': test_and_save_results, 'eval_json_file': 'eval_result.json'}
            if hasattr(self.cfg, 'eval_iter_period') and hasattr(self.cfg, 'checkpoint_config') and hasattr(
                    self.cfg.checkpoint_config, 'iter_period'):
                iter_period = self.cfg.checkpoint_config.iter_period
                eval_iter_period = self.cfg.eval_iter_period
                if iter_period % eval_iter_period == 0:
                    kwargs.update({'eval_iter_period': eval_iter_period})
                else:
                    logger = logging.getLogger(__name__)
                    msg = 'eval_iter_period:{} is not equal to iter_period {} in checkpoint_config,'.format(
                        eval_iter_period, iter_period)
                    msg += 'it will be assign the value iter_period to eval_iter_period'
                    logger.warning(msg=msg)
                    self.cfg.eval_iter_period = iter_period
                    kwargs.update({'eval_iter_period': iter_period})

            if hasattr(self.cfg.test_data, 'is_run_eval'):
                kwargs.update({'is_run_eval': self.cfg.test_data.is_run_eval})

            return kwargs

        return hooks.EvaluateHook(**evaluate_hook_param())

    @property
    def by_epoch(self):
        return self._by_epoch

    def set_mixked_precision(self, enable=False):
        global _AUTOMATIC_mixkED_PRECISION
        _AUTOMATIC_mixkED_PRECISION = enable

    def set_by_iter(self):
        global _BY_ITER_TRAIN
        _BY_ITER_TRAIN = False if self._by_epoch else True

    def resume_or_load(self, path: str, resume: bool) -> None:
        """If resume is True, and path checkpoint exists, resume from it(eg. optimizer and scheduler)
        and update start_iter or start_epoch (if by_epoch = True)
        counter.

        Otherwise, load the model specified by the config( skip optimizer and scheduler) and start from
        the first iteration.

        Args:
            resume (bool): whether to do resume or not
        """
        logger = logging.getLogger(__name__)
        if not os.path.isfile(path):
            logger.warning(f'{path} checkpoint does not exists')
            return

        logger.info(f'loading : {path}')
        checkpoint = self.checkpointer.resume_or_load(path, resume=resume)

        if resume:
            by_epoch = checkpoint.get('by_epoch', False)
            if by_epoch:
                self.start_iter = checkpoint.get('epoch_iter', 0)
                self.start_epoch = checkpoint.get('epoch', -1) + 1
                logger.info(f'current epoch: {self.start_epoch}')
            else:
                self.start_iter = checkpoint.get('iteration', 0)

            logger.info(f'current iteration :{self.start_iter}')

        logger.info('checkpoint loaded')

    @property
    def model_name(self):
        return self._model_name

    @property
    def dataset_name(self):
        return self._dataset_name

    @model_name.setter
    def model_name(self, name):
        self._model_name = name

    @dataset_name.setter
    def dataset_name(self, name):
        self._dataset_name = name

    def add_custom_hooks(self, custom_hooks_cfg):
        assert isinstance(custom_hooks_cfg, list), f'custom_hook expect list type,but got {type(custom_hooks_cfg)} '
        from mixk.engine.hooks import build_hook, PriorityStatus

        def get_insert_idx(hook):
            for idx, hk in enumerate(self.hooks[::-1]):
                if hk.level.value <= hook.level.value:
                    return -idx

        for hk_cfg in custom_hooks_cfg:
            assert isinstance(hk_cfg, dict), f' hook expect dict type,but got{type(hk_cfg)}'
            level = hk_cfg.pop('level', PriorityStatus.NORMAL)
            hook = build_hook(hk_cfg)
            hook.level = level
            idx = get_insert_idx(hook)
            if idx < 0:
                self.hooks.insert(idx, hook)
            else:  # == 0, the last one
                self.hooks.append(hook)

        # print all hook
        logger = logging.getLogger(__name__)
        for hk in self.hooks:
            logger.info(f'hook name:{type(hk)}  ->   level name and value:{hk.level.name, hk.level.value}')
