from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad

from .base_hook import HookBase, PriorityStatus
from .builder import HOOKS


@HOOKS.register_module()
class OptimizerHook(HookBase):

    def __init__(self, grad_clip=None):
        super().__init__()
        self._grad_clip = grad_clip
        self._level = PriorityStatus.HIGH
        self._user_scale = False if self._grad_clip is None else getattr(self._grad_clip, 'use_scale', False)
        self._scaler = GradScaler(enabled=False) if self._user_scale else None

    def _clip_grad_norm(self) -> None:
        clip_norm_params = list(
            filter(lambda parm: parm.requires_grad and parm.grad is not None, self.trainer.model.parameters()))
        if len(clip_norm_params) == 0:
            return
        else:
            if hasattr(self._grad_clip, 'clip_norm_mode'):
                scale = self._scaler.get_scale() if self._user_scale else 1.0
                max_norm = self._grad_clip.max_grad_l2_norm * scale
                grad_norm = clip_grad.clip_grad_norm(clip_norm_params, max_norm)
            else:
                grad_norm = clip_grad.clip_grad_norm_(clip_norm_params, **self._grad_clip)
            self.trainer.log_buffer.put_scalar('grad_norm', float(grad_norm))

    def after_train_iter(self):
        self.trainer.output['loss'] /= self.trainer.gradient_accumulation_steps
        self.trainer.output['loss'].backward()
        if self._grad_clip is not None:
            self._clip_grad_norm()

        if (self.trainer.iter + 1) % self.trainer.gradient_accumulation_steps == 0:
            if self._user_scale:
                self._scaler.step(self.trainer.optimizer)
                self._scaler.update()
            else:
                self.trainer.optimizer.step()

    def before_train_iter(self):
        if self.trainer.iter == 0:
            is_clean = True
        elif self.trainer.iter % self.trainer.gradient_accumulation_steps == 0:
            is_clean = True
        else:
            is_clean = False

        if is_clean:
            self.trainer.optimizer.zero_grad()


@HOOKS.register_module()
class Fp16OptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None, grad_scaler_config=None):
        super().__init__(grad_clip)
        self._grad_scaler_config = grad_scaler_config
        self._scaler = None

    def before_train(self):
        if self._grad_scaler_config is None:
            self._scaler = GradScaler()
        else:
            self._scaler = GradScaler(**self._grad_scaler_config)

    def after_train_iter(self):
        loss = self.trainer.output['loss'] / self.trainer.gradient_accumulation_steps
        self._scaler.scale(loss).backward()
        if self._grad_clip is not None:
            self._scaler.unscale_(self.trainer.optimizer)
            self._clip_grad_norm()

        if (self.trainer.iter + 1) % self.trainer.gradient_accumulation_steps == 0:
            self._scaler.step(self.trainer.optimizer)
            self._scaler.update()
