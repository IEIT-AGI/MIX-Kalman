import math
from bisect import bisect_right, bisect
from typing import List
from functools import lru_cache

import torch
from .builder import LR_SCHEDULERS
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler, ReduceLROnPlateau
from .optimization import BertAdam
import mixk.utils.distributed_info as comm
import logging
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


@LR_SCHEDULERS.register_module()
class WarmupMultiStepLR(_LRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        *,
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = 'linear',
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of' ' increasing integers. Got {}', milestones)

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = WarmupCosineLR.get_warmup_factor_at_iter(self.warmup_method, self.last_epoch, self.warmup_iters,
                                                                 self.warmup_factor)

        @lru_cache
        def calculate_lr(base_lr):
            return base_lr * warmup_factor * self.gamma**bisect_right(self.milestones, self.last_epoch)

        return [calculate_lr(base_lr) for base_lr in self.base_lrs]

    def _compute_values(self) -> List[float]:
        return self.get_lr()


@LR_SCHEDULERS.register_module()
class ReduceOnPlateauSchedule(ReduceLROnPlateau):

    def __init__(self, optimizer: torch.optim.Optimizer, **kwargs):
        self.factor = kwargs['factor']
        self.mode = kwargs['mode']
        self.patience = kwargs['patience']
        self.verbose = kwargs['verbose']
        self.cooldown = kwargs['cooldown']
        super().__init__(
            optimizer,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            verbose=self.verbose,
            cooldown=self.cooldown)

    def get_lr(self):
        return self.get_last_lr()


@LR_SCHEDULERS.register_module()
class WarmupCosineLR(_LRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        *,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = 'linear',
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = self.get_warmup_factor_at_iter(self.warmup_method, self.last_epoch, self.warmup_iters,
                                                       self.warmup_factor)

        @lru_cache
        def calculate_lr(base_lr):
            return base_lr * warmup_factor * 0.5 * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))

        return [calculate_lr(base_lr) for base_lr in self.base_lrs]

    def _compute_values(self) -> List[float]:
        return self.get_lr()

    @staticmethod
    def get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int, warmup_factor: float) -> float:
        """Return the learning rate warmup factor at a specific iteration. See
        :paper:`in1k1h` for more details.

        Args:
            method (str): warmup method; either "constant" or "linear".
            iter (int): iteration at which to calculate the warmup factor.
            warmup_iters (int): the number of warmup iterations.
            warmup_factor (float): the base warmup factor (the meaning changes according
                to the method used).

        Returns:
            float: the effective warmup factor at the given iteration.
        """
        if iter >= warmup_iters:
            return 1.0
        support_method = ['constant', 'linear']

        def constant_method():
            return warmup_factor

        def linear_method():
            alpha = iter / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        if method in support_method:
            return eval(method + '_method')()
        else:
            raise ValueError('Unknown warmup method: {}'.format(method))


@LR_SCHEDULERS.register_module()
class PythiaScheduler(LambdaLR):

    def __init__(self, optimizer, *args, **kwargs):
        self._lambda_func = PythiaScheduler.lr_lambda_update

        from mixk.utils.config import mixkEasyDict
        self._global_config = mixkEasyDict({'lr_config': {k: v for k, v in kwargs.items()}})
        kwargs = {}
        super().__init__(optimizer, self.lr_lambda, *args, **kwargs)

    def lr_lambda(self, step):
        return self._lambda_func(step, self._global_config)

    @staticmethod
    def lr_lambda_update(i_iter, cfg):
        if cfg.lr_config.use_warmup is True and i_iter <= cfg.lr_config.warmup_iterations:
            alpha = float(i_iter) / float(cfg.lr_config.warmup_iterations)
            return cfg.lr_config.warmup_factor * (1.0 - alpha) + alpha
        else:
            idx = bisect(cfg.lr_config.lr_steps, i_iter)
            return pow(cfg.lr_config.lr_ratio, idx)


@LR_SCHEDULERS.register_module()
class MultiStepScheduler(PythiaScheduler):

    def __init__(self, optimizer, *args, **kwargs):
        self.use_warmup = kwargs['use_warmup']
        self.lr_steps = kwargs['lr_steps']
        self.lr_ratio = kwargs['lr_ratio']
        self.warmup_iterations = kwargs['warmup_iterations'] if self.use_warmup else 0
        self.warmup_factor = kwargs['warmup_factor']
        assert self.warmup_iterations < self.lr_steps[0]
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iterations and self.use_warmup is True:
            alpha = float(self.last_epoch) / float(self.warmup_iterations)
            lr_ratio = self.warmup_factor * (1.0 - alpha) + alpha

            return [base_lr * lr_ratio for base_lr in self.base_lrs]
        else:

            @lru_cache
            def calculate_lr(base_lr):
                return base_lr * self.lr_ratio**bisect_right(self.lr_steps, self.last_epoch)

            return [calculate_lr(base_lr) for base_lr in self.base_lrs]


@LR_SCHEDULERS.register_module()
class WarmupLinearScheduleNonZero(_LRScheduler):
    """Linear warmup and then linear decay. Linearly increases learning rate
    from 0 to max_lr over `warmup_steps` training steps.

    Linearly decreases learning rate linearly to min_lr over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, t_total, warmup_iterations=0, use_warmup=False, min_lr=1e-5, last_epoch=-1):
        self.use_warmup = use_warmup
        self.warmup_iters = warmup_iterations
        self.t_total = t_total
        self.min_lr = min_lr
        super(WarmupLinearScheduleNonZero, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_iters:
            lr_factor = float(step) / float(max(1, self.warmup_iters))
        else:
            lr_factor = max(0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_iters)))

        return [
            base_lr * lr_factor if (base_lr * lr_factor) > self.min_lr else self.min_lr for base_lr in self.base_lrs
        ]


@LR_SCHEDULERS.register_module()
class BertWarmupLinearLR(_LRScheduler):
    """Implements BERT version of Warmup Linear lr algorithm
  Params:
      warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
      t_total: total number of training steps for the learning
          rate schedule, -1  means constant learning rate. Default: -1
      schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
  """
    SCHEDULES = [
        'warmup_cosine',
        'warmup_constant',
        'warmup_linear',
    ]

    def __init__(
        self,
        optimizer: BertAdam,
        max_iters: int,
        warmup: float = -1,
        warmup_method: str = 'warmup_linear',
        last_epoch: int = -1,
    ):
        if warmup_method not in self.SCHEDULES:
            raise ValueError('Invalid schedule parameter: {}'.format(warmup_method))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError('Invalid warmup: {} - should be in [0.0, 1.0[ or -1'.format(warmup))

        self.max_iters = max_iters
        self.warmup = warmup
        self.warmup_method = warmup_method
        self.warned_for_t_total = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.max_iters != -1:
            if comm.is_main_process():
                logger = logging.getLogger(__name__)

            schedule_fct = getattr(self, self.warmup_method)
            progress = self.last_epoch / self.max_iters
            lr_cur = [base_lr * schedule_fct(progress, self.warmup) for base_lr in self.base_lrs]
            # warning for exceeding t_total (only active with warmup_linear
            if self.warmup_method == 'warmup_linear' and progress > 1. and not self.warned_for_t_total:
                if comm.is_main_process():
                    logger.info(
                        "Training beyond specified 't_total' steps with schedule '{}'. Learning rate set to {}. "
                        "Please set 't_total' of {} correctly.".format(self.warmup_method, lr_cur,
                                                                       self.__class__.__name__))
                self.warned_for_t_total = True
            # end warning
        else:
            lr_cur = [base_lr for base_lr in self.base_lrs]

        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return lr_cur

    def _compute_values(self) -> List[float]:
        return self.get_lr()

    @staticmethod
    def warmup_cosine(x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 0.5 * (1.0 + torch.cos(math.pi * x))

    @staticmethod
    def warmup_constant(x, warmup=0.002):
        """Linearly increases learning rate over `warmup`*`t_total` (as
        provided to BertAdam) training steps.

        Learning rate is 1. afterwards.
        """

        return x / warmup if x < warmup else 1.0

    @staticmethod
    def warmup_linear(x, warmup=0.002):
        """Specifies a triangular learning rate schedule where peak is reached
        at `warmup`*`t_total`-th (as provided to BertAdam) training step.

        After `t_total`-th training step, learning rate is zero.
        """
        if x < warmup:
            return x / warmup
        return max((x - 1.) / (warmup - 1.), 0)


@LR_SCHEDULERS.register_module()
class ConstantSchedule(LambdaLR):

    def __new__(cls, optimizer, *args, **kwargs):
        return get_constant_schedule(optimizer, *args, **kwargs)


@LR_SCHEDULERS.register_module()
class WarmupConstantSchedule(LambdaLR):

    def __new__(cls, optimizer, *args, **kwargs):
        return get_constant_schedule_with_warmup(optimizer, *args, **kwargs)


@LR_SCHEDULERS.register_module()
class WarmupLinearSchedule(LambdaLR):
    """Linear warmup and then linear decay. Linearly increases learning rate
    from 0 to 1 over `warmup_steps` training steps.

    Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __new__(cls, optimizer, *args, **kwargs):
        return get_linear_schedule_with_warmup(optimizer, *args, **kwargs)


@LR_SCHEDULERS.register_module()
class WarmupCosineSchedule(LambdaLR):

    def __new__(cls, optimizer, *args, **kwargs):
        return get_cosine_schedule_with_warmup(optimizer, *args, **kwargs)


@LR_SCHEDULERS.register_module()
class WarmupCosineWithHardRestartsSchedule(LambdaLR):

    def __new__(cls, optimizer, *args, **kwargs):
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, *args, **kwargs)


@LR_SCHEDULERS.register_module()
class WarmupPolynomialSchedule(LambdaLR):

    def __new__(cls, optimizer, *args, **kwargs):
        return get_polynomial_decay_schedule_with_warmup(optimizer, *args, **kwargs)
