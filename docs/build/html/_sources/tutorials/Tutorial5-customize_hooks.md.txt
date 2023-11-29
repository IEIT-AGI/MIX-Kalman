# Tutorial 5: Customize hooks

In MIX-Kalman, we encapsulates most of the calculation in the deep learning process into hooks, such as `autograd_anomally_detect`, `lr_scheduler`, `iteration_time`,`periodic_logger`,  `evaluate` and so on. Those hooks will be called by mixkEngine.

In EngineBase( parent class of mixkEngine), we use `weakref.prox` to transfer mixkEngine object to each hook as `hk.trainer`. Since `hk.trainer` contains all hooks, data can also be transferred between each hooks.

In MIX-Kalman, the whole process of deep learning process is abstracted and modularized into several stages. Different hook run in different stage depending on its functionality, as mentioned in [Engine](Tutorial-engine.md). All the default hooks supported by MIX-Kalman are shown below.

| hook name                 | hook function                                                |
| :------------------------ | :----------------------------------------------------------- |
| AutogradAnomalyDetectHook | anomaly detection for the autograd engine                    |
| AutogradProfilerHook      | run `torch.autograd.profiler.profile`                        |
| CheckPointHook            | save model or model_dict                                     |
| EvaluateHook              | run an evaluation function periodically, and at the end of training |
| IterationTimerHook        | track each iteration time and print a summary in the end of training. |
| LRSchedulerHook           | executes a torch builtin LR scheduler and summarizes the LR  |
| OptimizerHook             | execute `clip_grad_norm`                                     |
| Fp16OptimizerHook         | inherit from OptimizerHook and execute Fp16 training         |
| PeriodicLogger            | record train result every interval time                      |
| CommonMetricLoggerHook    | print iteration time, loss, lr and ETA on the terminal       |
| JSONLoggerHook            | save scalars like data_time, loss ,iteration... to a json file |
| TensorboardLoggerHook     | save all scalars to a tensorboard file                       |

### Customize self-implemented hooks

There are some occasions users might need to implement a new hook. mixk supports customized hooks. (e.g. `EMAIterHook`)

1. Define a new hook

   Create a new file `mixk/engine/hooks/ema.py`.

   ```python
   from .base_hook import HookBase  # import HookBase
   from .builder import HOOKS  # import HOOKS register

   # register the EMAIterHook by decorator and inherit from the parent class
   @HOOKS.register_module()
   class EMAIterHook(HookBase):

       def __init__(self):
           super().__init__()

       def before_train(self):
           if hasattr(self.trainer.model, 'module'):
               self.trainer.model.module.init_ema()
           else:
               self.trainer.model.init_ema()

       def after_train_iter(self):
           if hasattr(self.trainer.model, 'module'):
               self.trainer.model.module.update_ema()
           else:
               self.trainer.model.update_ema()
   ```

   **NOTE**:

   - import the `HOOKS` register and `HookBase`;

   - decorate the `EMAIterHook` class by `@HOOKS.register_module()` to register it;

   - the class `EMAIterHook` should inherit from the parent class `HookBase` ;

   - choose `before_train()`and `after_train_iter` , rewrite the function;

     We have six functions in `HookBase` , which are`before_train`, `after_train`, `before_train_iter`, `after_train_iter`, `before_train_epoch` and  `after_train_epoch`.  Each function represents different  stage of the deep learning process. Depending on the functionality of the hook, the users need to rewrite corresponding functions to decide in which stage hook runs.

2. Import the module

   You can add the following line to `mixk/engine/hooks/__init__.py`, and add `'EMAIterHook'` in `__all__`.

   ```python
   from .lr_scheduler import LRSchedulerHook

   __all__ = [
       'EMAIterHook', ...
   ]
   ```

3. Prepare a hook config file

   Create a new hook config file `mixk/configs/_base_/custom_hook/devlbert_ema_hook.py`.

   ```python
   custom_hooks = [
       dict(
           type='EMAIterHook',
           level=30,  # set the hook level,  level type : PriorityStatus, str, int
       ),
       dict(
           type='EMAEpochHook',
           level=40,
       ),
   ]
   ```

**NOTE**:

+ each customized hook should have a parameter `dict`, which gives hook class type and hook level. And put all customized hook dicts in a list `custom_hooks`.

+ set the hook level. The level type can be str or int. For the reason that we have many hooks in each stage, so we should set the hook level to decide the priority of  the hook. We have seven levels in total for you to choose, which is listed as below (the lower the number, the higher the priority). If the parameter is not given,  it will be set to level `NORMAL` in default.

| PriorityStatus   | level  |
| :--------------- | :----- |
| HIGHEST          | 0      |
| HIGHER           | 10     |
| HIGH             | 20     |
| NORMAL           | 30     |
| LOW              | 40     |
| LOWER            | 50     |
| LOWEST           | 60     |

+ when sorting hooks, we compare the priority level(the lower the number, the higher the ranking). If the customized hook level same with the default hook, it append behind the default hook.
