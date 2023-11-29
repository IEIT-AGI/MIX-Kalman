import copy
import itertools
import logging
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch.nn.modules import Module
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mixk.utils.third_party_libs import PathManager

__all__ = ['Checkpointer', 'PeriodicCheckpointer']

_IncorrectShape = NamedTuple(
    '_IncorrectShape',
    [('key', str), ('checkpoint_model_shape', Tuple), ('model_shape', Tuple)],
)

# _IncompatibleKeys = NamedTuple('IncompatibleKeys',
#                                [
#                                    ('missing_keys', List[str]),
#                                    ('unexpected_keys', List[str]),
#                                    ('incorrect_shapes', List[_IncorrectShape]),
#                                ],
#                                )


class IncompatibleKeys:

    def __init__(self, model: Module, missing_keys: List[str], unexpected_keys: List[str],
                 incorrect_shapes: List[_IncorrectShape]):
        self._model = model
        self.missing_keys = missing_keys
        self.unexpected_keys = unexpected_keys
        self.incorrect_shapes = incorrect_shapes
        self.logger = logging.getLogger(__name__)

    def log_all_info(self):
        self.__log_incorrect_shapes()
        self.__log_missing_keys()
        self.__log_unexpected_keys()

    def __log_incorrect_shapes(self):
        for s in self.incorrect_shapes:
            key_msg = 'Due to model matching issues,some parameters {} will be skipped'.format(s.key)
            ck_shape_msg = 'checkpoint_model_shapes:{}'.format(s.checkpoint_model_shape)
            model_shape_msg = 'model_shapes:{}'.format(s.model_shape)
            msg = key_msg + ck_shape_msg + ' VS ' + model_shape_msg
            self.logger.warning(msg)

    def __log_missing_keys(self):
        if self.missing_keys:
            missing_keys = self.__remove_reused_missing_keys()
            missing_keys_info = self.list_to_dict(missing_keys)
            msg = 'Some model parameters are not found in the  state_dict checkpoint:'
            msg += '\n'.join('  ' + k + self.list_to_str(v) for k, v in missing_keys_info.items())
            self.logger.warning(msg)

    def __log_unexpected_keys(self):
        if self.unexpected_keys:
            unexpected_keys = self.list_to_dict(self.unexpected_keys)
            msg = 'Some keys contained in the state_dict checkpoint that are not used by the model:'
            msg += '\n'.join(' ' + k + self.list_to_str(v) for k, v in unexpected_keys.items())
            self.logger.warning(msg)

    def __remove_reused_missing_keys(self) -> List[str]:
        unique_keys = set(self.missing_keys)
        param_map_name = defaultdict(set)
        for model_prefix, model in self.__named_modules_with_duplicated(self._model):
            for name, param, in itertools.chain(
                    model.named_buffers(recurse=False), model.named_parameters(recurse=False)):
                param_of_name = model_prefix + ('.' if model_prefix else '') + name
                param_map_name[param].add(param_of_name)

        param_map_of_names = param_map_name.values()
        for names in param_map_of_names:
            if not all(name in unique_keys for name in names) and any(name in unique_keys for name in names):
                for name in names:
                    if name in unique_keys:
                        unique_keys.remove(name)

        return list(unique_keys)

    @staticmethod
    def __named_modules_with_duplicated(model: Module, model_prefix: str = '') -> Iterable[Tuple[str, Module]]:
        yield model_prefix, model
        for name, module in model._modules.items():
            if module is None:
                continue
            submodule_prefix = model_prefix + ('.' if model_prefix else '') + name
            yield from IncompatibleKeys.__named_modules_with_duplicated(model=module, model_prefix=submodule_prefix)

    @staticmethod
    def list_to_str(list_array: List[str]) -> str:
        _len = len(list_array)
        if _len == 0:
            return ''
        elif _len == 1:
            return '.' + list_array[0]
        else:
            return '.{' + ', '.join(list_array) + '}'

    @staticmethod
    def list_to_dict(list_array: List[str]) -> Dict[str, List[str]]:
        result = defaultdict(list)
        for l in list_array:
            pos = l.rfind('.')
            if pos >= 0:
                key, value = l[:pos], [l[pos + 1:]]
            else:
                key, value = l, list()
            result[key] = value
        return result

    @classmethod
    def generate_incompatible_keys(cls, **kwargs):
        yield cls(**kwargs)


class Checkpointer:

    def __init__(self,
                 model: Module,
                 save_dir: str = '',
                 *,
                 is_save_disk: bool = True,
                 is_record_ck: bool = True,
                 **other_train_info: object):
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module
        self.model = model
        self.save_dir = save_dir
        self.is_save_disk = is_save_disk
        self.other_train_info = copy.copy(other_train_info)
        self.logger = logging.getLogger(__name__)
        self._is_record_ck = is_record_ck
        self._checkpoint_files: List[str] = []

    @property
    def is_record_ck(self) -> bool:
        return self._is_record_ck

    @is_record_ck.setter
    def is_record_ck(self, value: bool):
        self._is_record_ck = value

    @property
    def checkpoint_files(self) -> List[str]:
        return self._checkpoint_files

    def remove_first_from_checkpoint(self):
        if len(self._checkpoint_files) > 0:
            remove_file_name = self._checkpoint_files.pop(0)
            remove_file_path = osp.join(self.save_dir, remove_file_name)
            if not remove_file_name.endswith('model_final.pth') and PathManager.exists(
                    osp.join(self.save_dir, remove_file_path)):
                PathManager.rm(remove_file_path)
                self.logger.info('Removing {} from checkpoint_files'.format(remove_file_name))

    def save(self, checkpoint_name: str, **kwargs: Dict[str, str]) -> None:
        if not (self.is_save_disk and self.save_dir):
            return
        self._write_data(checkpoint_name, self._prepare_data(**kwargs))

    def _prepare_data(self, **kwargs: Dict[str, str]) -> Dict:
        data = {'model': self.model.state_dict()}
        data.update(kwargs)
        for k, v in self.other_train_info.items():
            data[k] = v.state_dict()

        return data

    def _write_data(self, file_name: str, data: Dict) -> None:
        ck_file_name = '{}.pth'.format(file_name)
        ck_file_path = osp.join(self.save_dir, ck_file_name)
        self.logger.info('Saving checkpoint to {}'.format(ck_file_path))
        with PathManager.open(ck_file_path, 'wb') as fwb:
            torch.save(data, fwb)

        if self._is_record_ck:
            self._checkpoint_files.append(ck_file_name)

    def load_checkpoint(self, path: str, specify_load_keys: Optional[List[str]] = None) -> object:
        if not path:
            self.logger.info('path:{} is empty'.format(path))
            raise ValueError('path is empty, (path:{})'.format(path))

        if not osp.isfile(path):
            path = PathManager.get_local_path(path)
            if not osp.isfile(path):
                raise ValueError('path is {}, no checkpoint found'.format(path))

        self.logger.info('Loading checkpoint form {}'.format(path))

        checkpoint = Checkpointer._load_file_from_path(file_path=path)
        incompatible_keys = self._load_model(checkpoint)
        if incompatible_keys is not None:
            incompatible_keys.log_all_info()

        if specify_load_keys is not None and len(specify_load_keys):
            self.other_train_info = specify_load_keys

        if specify_load_keys is None:  # resume
            for key in self.other_train_info:
                if key in checkpoint:
                    value = self.other_train_info[key]
                    value.load_state_dict(checkpoint.pop(key))

        return checkpoint

    @staticmethod
    def _load_file_from_path(file_path, map_location='cpu'):
        return torch.load(file_path, map_location=map_location)

    def _load_model(self, checkpoint):
        ck_model_state_dict = checkpoint.pop('model')
        self._state_dict_to_tensor(ck_model_state_dict)
        self.__remove_prefix_from_state_dict(ck_model_state_dict, 'module.')

        model_state_dict = self.model.state_dict()
        incorrect_shapes = []
        for key in list(ck_model_state_dict.keys()):
            if key in model_state_dict:
                model_shape = tuple(model_state_dict[key].shape)
                ck_model_shape = tuple(ck_model_state_dict[key].shape)
                if model_shape != ck_model_shape:
                    incorrect_shapes.append(_IncorrectShape(key, ck_model_shape, model_shape))
                    ck_model_state_dict.pop(key)

        incompatible = self.model.load_state_dict(ck_model_state_dict, strict=False)

        return IncompatibleKeys(
            model=self.model,
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes)

    @staticmethod
    def _state_dict_to_tensor(state_dict: Dict[str, Any]) -> None:
        for k, v in state_dict.items():
            assert isinstance(v, (np.ndarray, torch.Tensor)), ValueError(
                '{}:{} is an unsupported type in checkpoint'.format(k, type(v)))
            if isinstance(v, np.ndarray):
                state_dict[k] = torch.from_numpy(v)

    @staticmethod
    def __remove_prefix_from_state_dict(state_dict: Dict[str, Any], prefix: str) -> None:
        all_key = state_dict.keys()
        for key in all_key:
            if key.startswith(prefix):
                _key = key[len(prefix):]
                state_dict[_key] = state_dict.pop(key)

    def resume_or_load(self, path, resume=True):  # tmp function
        return self.load_checkpoint(path=path, specify_load_keys=None if resume else [])


class PeriodicCheckpointer:

    def __init__(self,
                 checkpointer: Checkpointer,
                 *,
                 iter_period: int = None,
                 epoch_period: int = 1,
                 max_iter: Optional[int] = None,
                 max_epoch: Optional[int] = None,
                 max_num_checkpoints: Optional[int] = None):

        assert isinstance(checkpointer, Checkpointer), TypeError(
            'checkpointer is {} type,but the desired type is Checkpointer'.format(type(checkpointer)))

        self.checkpointer = checkpointer
        self.iter_period = iter_period
        self.epoch_period = epoch_period

        if max_iter is not None:
            assert max_iter > 0, ValueError('max_iter{} < 0'.format(max_iter))
        if max_epoch is not None:
            assert max_epoch > 0, ValueError('max_epoch{} < 0'.format(max_epoch))
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.max_num_checkpoints = max_num_checkpoints

        if self.max_num_checkpoints is not None:
            self.checkpointer.is_record_ck = True

    def record_iter_checkpoint(self, iteration: int, prefix: str = '', **kwargs: Any) -> str:
        state_dict = dict(iteration=iteration)
        state_dict.update(kwargs)
        file_name = None
        if (iteration + 1) % self.iter_period == 0:
            file_name = '{}_iter{:07d}_model'.format(prefix, iteration)
            self.save(file_name, **state_dict)
            if self.max_num_checkpoints is not None and len(
                    self.checkpointer.checkpoint_files) > self.max_num_checkpoints:
                self.checkpointer.remove_first_from_checkpoint()

        if iteration + 1 >= self.max_iter:
            self.save('{}_model_final'.format(prefix), **state_dict)

        return file_name + '.pth'

    def record_epoch_checkpoint(self, epoch: int, prefix: str = '', **kwargs: Any) -> str:
        is_epoch = kwargs.pop('is_epoch')
        state_dict = dict(epoch=epoch)
        state_dict.update(kwargs)
        file_name = None
        if (epoch + 1) % self.epoch_period == 0:
            if is_epoch:
                file_name = '{}_epoch{}_model'.format(prefix, epoch)
            else:
                file_name = '{}_epoch{}_iter{}_model'.format(prefix, epoch, kwargs.get('epoch_iter') + 1)
            self.save(file_name, **state_dict)
            if self.max_num_checkpoints is not None and len(
                    self.checkpointer.checkpoint_files) > self.max_num_checkpoints:
                self.checkpointer.remove_first_from_checkpoint()

        if epoch + 1 >= self.max_epoch:
            self.save('{}_model_final'.format(prefix), **state_dict)

        return file_name + '.pth'

    def save(self, file_name, **kwargs: Any) -> None:
        self.checkpointer.save(file_name, **kwargs)
