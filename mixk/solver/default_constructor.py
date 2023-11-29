import torch
import json
from torch.nn import GroupNorm, LayerNorm, Module
from .builder import OPTIMIZER_BUILDERS, OPTIMIZERS
from ..utils.registry import build_from_cfg
import logging
from typing import Optional, Any
from mixk.utils.config import mixkEasyDict


@OPTIMIZER_BUILDERS.register_module()
class DefaultOptimizerConstructor:
    """Used for parameters setting related to gradient updating: such as
    learning rate and delay. By default, the settings of every layer in the
    model is shared configured in the configure file related to optimizer. If
    argument ``paramwise_cfg`` is set, then everything can happened to every
    layer in the model. Specifically, ``paramwise_cfg`` is a dict and contains
    the following keys:

        - ``custom_keys`` (dict): Use keys representing layers in model to specify
          the configuration of somelayers. The key in the ``custom_keys``  is a substring
          of the layer name in the model. Learning rate and decay will be adopted to the
          spcified layer. If there are many names contain the specified key string, then the key
          with lower alphabet order will be chosen.
          ``custom_keys[key]`` should be a dict and may contain fields ``lr_mult``
          and ``decay_mult``. See Example 2 below.
        - ``bias_lr_mult`` (float): Used to be multiplied to the learning
          rate for all bias parameters (except for those in normalization
          layers).
        - ``bias_decay_mult`` (float): Used to be multiplied to the weight
          decay for all bias parameters (except for those in
          normalization layers and depthwise conv layers).
        - ``norm_decay_mult`` (float): Used be multiplied to the weight
          decay for all weight and bias parameters of normalization
          layers.
        - ``dwconv_decay_mult`` (float): Used be multiplied to the weight
          decay for all weight and bias parameters of depthwise conv
          layers.
        - ``bypass_duplicate`` (bool): If true, the duplicate parameters
          are not added into optimizer. Default: False.

        Args:
            model (:obj:`nn.Module`): The model with parameters to be optimized.
            optimizer_cfg (dict): The config dict for the optimizer.
                Positional fields are
                    - `type`: class name of the optimizer.
                Optional fields are
                    - any arguments of the corresponding optimizer type, e.g.,
                      lr, weight_decay, momentum, etc.
                      refer to: https://pytorch.org/docs/stable/optim.html#module-torch.optim
            paramwise_cfg (dict, optional): Settings to specifiy configure to layers separately.

        Example 1:
            >>> model = torch.nn.modules.Conv1d(1, 1, 1)
            >>> optimizer_cfg = dict(type='Adagrad', lr=0.01, lr_decay=0.9, weight_decay=0.0001)
            >>> paramwise_cfg = dict(norm_decay_mult=0.)
            >>> optim_builder = DefaultOptimizerConstructor(optimizer_cfg, paramwise_cfg)
            >>> optimizer = optim_builder(model)

        Example 2:
            >>> # assume model have attribute model.extract_feature and model.object_detection
            >>> optimizer_cfg = dict(type='SGD', lr=0.01, weight_decay=0.95)
            >>> paramwise_cfg = dict(custom_keys={'.extract_feature': dict(lr_mult=0.1, decay_mult=0.9)})
            >>> optim_builder = DefaultOptimizerConstructor(optimizer_cfg, paramwise_cfg)
            >>> optimizer = optim_builder(model)
            >>> # Then the `lr` and `weight_decay` for model.extract_feature is
            >>> # (0.01 * 0.1, 0.95 * 0.9). `lr` and `weight_decay` for
            >>> # model.object_detection is (0.01, 0.95).
    """

    def __init__(self, optimizer_cfg: mixkEasyDict, paramwise_cfg: Optional[mixkEasyDict] = {}):
        paramwise_cfg = {} if paramwise_cfg is None else paramwise_cfg

        self.check_var_type(optimizer_cfg, 'optimizer_cfg', {})
        self.check_var_type(paramwise_cfg, 'paramwise_cfg', {})

        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = paramwise_cfg
        self.base_lr = getattr(optimizer_cfg, 'lr', None)
        self.base_wd = getattr(optimizer_cfg, 'weight_decay', None)

        self._validate_paramwise_cfg()
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def check_var_type(var: Any, var_name: str, desired_var_type: Any) -> None:
        msg = f'{var_name} should be a {type(desired_var_type).__name__} type ,but got {type(var).__name__} type'
        assert isinstance(var, type(desired_var_type)), TypeError(msg)

    def _validate_paramwise_cfg(self):
        if 'custom_keys' in self.paramwise_cfg:
            self.check_var_type(self.paramwise_cfg['custom_keys'], 'custom_keys', {})
            if self.base_wd is None:
                for key in self.paramwise_cfg['custom_keys']:
                    if 'decay_mult' in self.paramwise_cfg['custom_keys'][key]:
                        raise ValueError('base_wd should not be None')

        is_bias_mult = 'bias_decay_mult' in self.paramwise_cfg
        is_norm_mult = 'norm_decay_mult' in self.paramwise_cfg
        is_dwconv_mult = 'dwconv_decay_mult' in self.paramwise_cfg

        # if mult is specified , weight_decay must be explicitly specified
        if is_bias_mult or is_norm_mult or is_dwconv_mult:
            assert self.base_wd, ValueError('base_wd should not be None')

    def _is_in(self, param_group, param_group_list):
        assert isinstance(param_group_list, dict)
        param_set, param = set(), set(param_group['params'])
        for group in param_group_list:
            param_set.update(set(group['params']))

        return not param.isdisjoint(param_set)

    def _paramwise_cfg_params(self):
        paramwise = mixkEasyDict()

        paramwise.bias_lr_mult = getattr(self.paramwise_cfg, 'bias_lr_mult', 1.0)
        paramwise.bias_decay_mult = getattr(self.paramwise_cfg, 'bias_decay_mult', 1.0)
        paramwise.norm_decay_mult = getattr(self.paramwise_cfg, 'norm_decay_mult', 1.0)
        paramwise.dwconv_decay_mult = getattr(self.paramwise_cfg, 'dwconv_decay_mult', 1.0)
        paramwise.bypass_duplicate = getattr(self.paramwise_cfg, 'bypass_duplicate', False)

        paramwise.custom_keys = getattr(self.paramwise_cfg, 'custom_keys', {})
        paramwise.sorted_keys = sorted(sorted(paramwise.custom_keys.keys()), key=len, reverse=True)

        return paramwise

    def add_params(self, params: mixkEasyDict, module: Module, prefix: str = ''):

        def match_custom_keys(paramwise: mixkEasyDict):
            p = paramwise
            is_custom = False
            for key in p.sorted_keys:
                if key in f'{prefix}.{name}':
                    is_custom = True
                    lr_mult = p.custom_keys[key].get('lr_mult', 1.)
                    param_group['lr'] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = p.custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                    break
            return is_custom

        p = self._paramwise_cfg_params()
        is_norm = isinstance(module, (GroupNorm, LayerNorm))
        is_dwconv = (isinstance(module, torch.nn.Conv2d) and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}

            if not param.requires_grad:
                params.append(param_group)
                continue
            if p.bypass_duplicate and self._is_in(param_group, params):
                msg = f'{prefix} is duplicate. It is skipped since ' f'bypass_duplicate={p.bypass_duplicate}'
                self.logger.warning(msg)
                continue

            if not match_custom_keys():
                # bias_lr_mult affects all bias parameters except for norm.bias
                if name == 'bias' and not is_norm:
                    param_group['lr'] = self.base_lr * p.bias_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    if is_norm:
                        param_group['weight_decay'] = self.base_wd * p.norm_decay_mult  # norm decay
                    elif is_dwconv:
                        param_group['weight_decay'] = self.base_wd * p.dwconv_decay_mult  # depth-wise conv
                    elif name == 'bias':
                        param_group['weight_decay'] = self.base_wd * p.bias_decay_mult  # bias lr and decay
            params.append(param_group)

        for child_name, child_module in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(params, child_module, prefix=child_prefix)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        if self.paramwise_cfg:  # specified paramwise_cfg option
            # set param-wise lr and weight decay recursively
            params = []
            self.add_params(params, model)
            optimizer_cfg['params'] = params
        else:
            optimizer_cfg['params'] = model.parameters()
            optimizer_cfg.pop('training_encoder_lr_multiply')

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)


@OPTIMIZER_BUILDERS.register_module()
class BertOptimizerConstructor(DefaultOptimizerConstructor):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        self.language_weights_file = paramwise_cfg.pop('language_weights_file', None)
        super().__init__(optimizer_cfg=optimizer_cfg, paramwise_cfg=paramwise_cfg)
        self.no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    def modify_params(self, params, optimizer_cfg, module, prefix=''):
        langauge_weights = self.load_language_weight(file=self.language_weights_file)

        for key, value in dict(module.named_parameters()).items():
            if value.requires_grad:
                if key in langauge_weights:
                    lr = optimizer_cfg['lr']
                else:
                    lr = optimizer_cfg['image_lr']

                if any(nd in key for nd in self.no_decay):
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0}]

                if not any(nd in key for nd in self.no_decay):
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.01}]

    @staticmethod
    def load_language_weight(file):
        with open(file) as f:
            return json.load(f)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        params = []
        self.modify_params(params, self.optimizer_cfg, model)
        optimizer_cfg['params'] = params

        optimizer_cfg.pop('image_lr')
        optimizer_cfg.pop('training_encoder_lr_multiply')

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)


@OPTIMIZER_BUILDERS.register_module()
class VilbertOptimizerConstructor(DefaultOptimizerConstructor):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        self.language_weights_file = paramwise_cfg.pop('language_weights_file')
        self.vision_scratch = paramwise_cfg.pop('vision_scratch')
        super().__init__(optimizer_cfg=optimizer_cfg, paramwise_cfg=paramwise_cfg)
        self.no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    def modify_params(self, params, optimizer_cfg, model, prefix=''):
        langauge_weights = self.load_language_weight(file=self.language_weights_file)

        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'vil_' in key:
                    lr = 1e-4
                else:
                    if self.vision_scratch:
                        if key[12:] in langauge_weights:
                            lr = optimizer_cfg['lr']
                        else:
                            lr = 1e-4
                    else:
                        lr = optimizer_cfg['lr']
                if any(nd in key for nd in self.no_decay):
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0}]
                if not any(nd in key for nd in self.no_decay):
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.01}]

        self.logger.info('len(model.named_parameters)={} len(params)={}'.format(
            len(list(model.named_parameters())), len(params)))

    @staticmethod
    def load_language_weight(file):
        with open(file) as f:
            return json.load(f)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        params = []
        self.modify_params(params, optimizer_cfg, model)
        optimizer_cfg['params'] = params

        optimizer_cfg.pop('training_encoder_lr_multiply')

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)


@OPTIMIZER_BUILDERS.register_module()
class OscarOptimizerConstructor(DefaultOptimizerConstructor):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        self.weight_decay = paramwise_cfg.pop('weight_decay')
        super().__init__(optimizer_cfg=optimizer_cfg, paramwise_cfg=paramwise_cfg)
        self.no_decay = ['bias', 'LayerNorm.weight']

    def modify_params(self, params, optimizer_cfg, model, prefix=''):
        # Prepare optimizer (decay)
        params += [{
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in self.no_decay)],
            'weight_decay': self.weight_decay
        }]
        params += [{
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in self.no_decay)],
            'weight_decay': 0.0
        }]

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        params = []
        self.modify_params(params, optimizer_cfg, model)
        optimizer_cfg['params'] = params

        optimizer_cfg.pop('training_encoder_lr_multiply')

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)


@OPTIMIZER_BUILDERS.register_module()
class OscarPreTrainOptimizerConstructor(OscarOptimizerConstructor):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        super().__init__(optimizer_cfg, paramwise_cfg)
        self.no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']


@OPTIMIZER_BUILDERS.register_module()
class DevlbertOptimizerConstructor(VilbertOptimizerConstructor):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        super().__init__(optimizer_cfg=optimizer_cfg, paramwise_cfg=paramwise_cfg)

    def modify_params(self, params, optimizer_cfg, model, prefix=''):
        langauge_weights = self.load_language_weight(file=self.language_weights_file)

        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'vil_prediction' in key:
                    lr = 1e-4
                else:
                    if self.vision_scratch:
                        if key[12:] in langauge_weights:
                            lr = optimizer_cfg['lr']
                        else:
                            lr = 1e-4
                    else:
                        lr = optimizer_cfg['lr']
                if any(nd in key for nd in self.no_decay):
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.01}]
                if not any(nd in key for nd in self.no_decay):
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0}]

        self.logger.info('len(model.named_parameters)={} len(params)={}'.format(
            len(list(model.named_parameters())), len(params)))


@OPTIMIZER_BUILDERS.register_module()
class UniterOptimizerConstructor(OscarOptimizerConstructor):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        super().__init__(optimizer_cfg=optimizer_cfg, paramwise_cfg=paramwise_cfg)
        self.no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']


@OPTIMIZER_BUILDERS.register_module()
class UniterVQAOptimizerConstructor(UniterOptimizerConstructor):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        self.lr_mul = paramwise_cfg.pop('lr_mul')
        self.key_named_param = paramwise_cfg.pop('key_named_param')
        super().__init__(optimizer_cfg=optimizer_cfg, paramwise_cfg=paramwise_cfg)

    def modify_params(self, params, optimizer_cfg, model, prefix=''):
        base_lr = optimizer_cfg['lr']

        param_optimizer = [(n, p) for n, p in model.named_parameters() if self.key_named_param not in n]
        param_top = [(n, p) for n, p in model.named_parameters() if self.key_named_param in n]
        params += [{
            'params': [p for n, p in param_top if not any(nd in n for nd in self.no_decay)],
            'lr': base_lr * self.lr_mul,
            'weight_decay': self.weight_decay
        }, {
            'params': [p for n, p in param_top if any(nd in n for nd in self.no_decay)],
            'lr': base_lr * self.lr_mul,
            'weight_decay': 0.0
        }, {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in self.no_decay)],
            'lr': base_lr,
            'weight_decay': self.weight_decay
        }, {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in self.no_decay)],
            'lr': base_lr,
            'weight_decay': 0.0
        }]
