# Tutorial 4: Customize Schedule and Runtime Settings

## Customize optimization schedules

### Customize optimizer supported by Pytorch

We already support to use all the optimizers implemented by PyTorch, and the only modification is to change the `optimizer` field of config files.
For example, if you want to use `Adam`, the modification could be as the following.

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

To modify the learning rate of the model, the users only need to modify the `lr` in the config of optimizer in the `configs/_base_/schedules/schedule***.py`.

### Customize self-implemented optimizer

1. Define a new optimizer (e.g.BertAdam)

   Create a new optimizer in the file `mixk/solver/optimization.py`.

   ```python
   # import OPTIMIZERS
   from .builder import OPTIMIZERS

   # register the OPTIMIZERS by decorator and inherit from the parent class
   @OPTIMIZERS.register_module()
   class BertAdam(Optimizer):

       def __init__(self,  *args, **kwargs):  # add input parameters
           pass

       def step(self, closure=None):  # Performs a single optimization step
           pass
   ```

2. Import the module

   You can add the following line to `mixk/solver/__init__.py`, and add `'BertAdam'` in `__all__`.

   ```python
   from .optimization import BertAdam

   __all__ = [
       'BertAdam', ...
   ]
   ```

3. Use the BertAdam in your config file

   ```python
   optimizer = dict(
       type='BertAdam', lr=1e-5, weight_decay=0.01, eps=1e-6, betas=[0.9, 0.999], training_encoder_lr_multiply=1)
   ```

### Additional settings

Tricks not implemented by the optimizer should be implemented through optimizer constructor (e.g. set parameter-wise learning rates) or hooks. We list some common settings that could stabilize the training or accelerate the training (e.g.`schedule_gqa_lxmert.py`).

- __Use gradient clip to stabilize training__:
  Some models need gradient clip to clip the gradients to stabilize the training process. An example is as below:

  ```python
  optimizer_config = dict(
      grad_clip=dict(max_norm=35, norm_type=2))

  optimizer_config = dict(grad_clip=dict(max_norm=5))
  ```


## Customize FP16 schedules

In MIX-Kalman,  we support mixed precision training by FP16. Mixed-precision training is to use half-precision floating-point numbers to accelerate training speed while minimizing the loss of precision.

It is easy to use FP16 by setting the config in `configs/_base_/schedules/schedule***.py`, which is as follows:

```python
fp16 = dict(
    init_scale=2.**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)
```

If you set up the `fp` config, it will generate `Fp16OptimizerHook` when build hooks, otherwise the `OptimizerHook` in `mixk/engine/orgnizer.py`.

```python
class Organizer:

    def build_hooks(self):

        ...
        hook_list = []
        if hasattr(self.cfg, 'fp16'):
            hook_list.append(hooks.Fp16OptimizerHook(self.cfg.optimizer_config.grad_clip, self.cfg.fp16))
            self.set_mixked_precision(True)
        else:
            hook_list.append(hooks.OptimizerHook(self.cfg.optimizer_config.grad_clip))
```

## Customize LR schedules

We support many other learning rate schedule, such as `ReduceOnPlateauSchedule`, `BertWarmupLinearLR` and `MultiStepScheduler` schedule. Use `ReduceOnPlateauSchedule` as an example:

```python
lr_config = dict(
    policy='ReduceOnPlateauSchedule',
    use_warmup=False,
    factor=0.5,
    mode='max',
    patience=1,
    verbose=True,
    cooldown=2)
```

## Customize train mode

Before training, you can set `by_iter` in the `configs/_base_/schedules/schedule***.py` to determine  `train by epoch` or `train by iter`.

- `train by epoch` mode

  ```python
  # by_iter = True  # the mode is train by epoch if annotated
  total_epochs = 8  # the num of epoch
  ```

- `train by iter` mode

  ```python
  by_iter = True  # the mode is train by iter if true
  total_iters = 8  # the num of iter
  ```

## Modify default runtime settings

### Checkpoint config

It will use `checkpoint_config` to initialize `CheckpointHook` when build hooks in `mixk/engine/orgnizer.py`.

```python
eval_iter_period = 5000
checkpoint_config = dict(iter_period=eval_iter_period)
```

The users could set `checkpoint_config` and `eval_iter_period` to decide how many iters to save a model checkpoint and compute an evaluation. If the two params are not specified, it will save model checkpoint and compute evaluation by epoch.

### Log config

The `log_config` enables to set intervals of the frequency to print log information.

```python
log_config = dict(period=100,)
```

### Additional settings

We list some common settings which used in training or test, which are as follows:

```python
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs'  # the dir to save logs and models
load_from = '/home/datasets/mix_data/model/visdial_model_mixk/vqa_weights.pth'
seed = 13
CUDNN_BENCHMARK = False
model_device = 'cuda'
find_unused_parameters = True
```

You can add other parameters if needed.
