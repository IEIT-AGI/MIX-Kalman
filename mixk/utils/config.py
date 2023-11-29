import ast
import os
import os.path as osp
import shutil
import sys
import tempfile
from importlib import import_module
from typing import Dict
import torch
from datetime import datetime
import numpy as np
import random

import regex
from easydict import EasyDict
import logging

logger = logging.getLogger(__name__)

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
RESERVED_KEYS = ['filename', 'text', 'pretty_text']
SUPPORTED_FILE_EXT = ['.json', '.py', '.yaml', '.yml']

_CACHE_DIR = osp.expanduser('~') + '/cache'
_mixk_WORK_DIR = './work_dir'


class mixkEasyDict(EasyDict):

    def pop(self, k, d=None):
        if k not in self.keys():
            return d
        else:
            return super().pop(k, d)


class ToExpanduser:  # ~  ->  /home/xxxx/

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def to_expanduser(self):
        return self._traverse_dict(self.cfg)

    def _traverse_dict(self, instance_dict):
        output = {}
        for k, v in instance_dict.items():
            output[k] = self._traverse(k, v)
        return output

    def _traverse(self, key, value):
        if isinstance(value, ToExpanduser):
            return value.to_expanduser()
        elif isinstance(value, dict):
            return self._traverse_dict(value)
        elif isinstance(value, list):
            return [self._traverse(key, i) for i in value]
        elif hasattr(value, '__dict__'):
            return self._traverse_dict(value.__dict__)
        elif isinstance(value, str):
            return self.modify_path(value)
        else:
            return value

    @classmethod
    def generate_obj(cls, args):
        return cls(args)

    @staticmethod
    def modify_path(path: str) -> str:
        if path.startswith('~'):
            new_path = path.replace('~', osp.expanduser('~'))
            return new_path
        else:
            return path


def file2buffer(filename: str, file_ext_name: str, use_predefined_var: bool = True) -> (dict, str):
    file_content = filename + ':\n'
    with open(filename, 'r') as rf:
        file_content += rf.read()

    cfg_dict = dict()
    with tempfile.TemporaryDirectory() as tmp_cfg_dir:
        tmp_cfg_file = tempfile.NamedTemporaryFile(suffix=file_ext_name, dir=tmp_cfg_dir)
        tmp_cfg_file_name = os.path.basename(tmp_cfg_file.name)
        if use_predefined_var:
            _copy_file_content_by_predefined_vars(filename, tmp_cfg_file.name)
        else:
            shutil.copyfile(filename, tmp_cfg_file.name)

        if file_ext_name == '.py':
            tmp_module_name = os.path.splitext(tmp_cfg_file_name)[0]
            sys.path.insert(0, tmp_cfg_dir)
            _validate_py_syntax(filename)
            tmp_module = import_module(tmp_module_name)
            cfg_dict = dict()
            for k, v in tmp_module.__dict__.items():
                if not k.startswith('__'):
                    cfg_dict[k] = v

            sys.path.pop(0)
            del sys.modules[tmp_module_name]

        tmp_cfg_file.close()

    return cfg_dict, file_content


def _validate_py_syntax(py_file):
    with open(py_file, 'r') as rf:
        py_content = rf.read()

    try:
        ast.parse(py_content)
    except SyntaxError:
        raise SyntaxError('In the {} ,there are some syntax errors', format(py_file))


def _copy_file_content_by_predefined_vars(filename, tmp_file_name):
    base_name = os.path.basename(filename)
    dir_name = os.path.dirname(filename)
    file_name_no_ext, ext_name = os.path.splitext(base_name)
    support_template = dict(
        DirName=dir_name, BaseName=base_name, fileBaseNameNoExt=file_name_no_ext, FileExtName=ext_name)
    with open(filename, 'r') as rf:
        cfg_content = rf.read()

    for k, v in support_template.items():
        pattern = r'\{\{\s*' + str(k) + r'\s*\}\}'
        repl = v.replace('\\', '/')
        cfg_content = regex.sub(pattern=pattern, repl=repl, string=cfg_content)

    with open(tmp_file_name, 'w') as wf:
        wf.write(cfg_content)


class Config:

    @staticmethod
    def _file2dict(file_name: str, use_predefined_var: bool = True) -> tuple:
        file_path = os.path.abspath(os.path.expandvars(file_name))
        if not os.path.exists(file_path):
            raise FileNotFoundError('{} does not exist'.format(file_path))
        ext_name = os.path.splitext(file_path)[-1]
        assert ext_name in SUPPORTED_FILE_EXT, 'Only json/py/yaml/yml extensions are supported,' \
                                               'but get extension is {}'.format(ext_name)
        cfg_dict, cfg_content = file2buffer(
            filename=file_name, file_ext_name=ext_name, use_predefined_var=use_predefined_var)

        if BASE_KEY in cfg_dict:
            file_dir = os.path.dirname(file_path)
            basekey_filename = cfg_dict.pop(BASE_KEY)
            basekey_filename = basekey_filename if isinstance(basekey_filename, list) else [basekey_filename]

            dict_list = []
            content_list = []
            for file in basekey_filename:
                single_dict, single_content = Config._file2dict(os.path.join(file_dir, file))
                dict_list.append(single_dict)
                content_list.append(single_content)

            base_dict = dict()
            for dl in dict_list:
                common_keys = base_dict.keys() & dl.keys()
                if len(common_keys) > 0:
                    common_base = [base_dict[k] for k in common_keys]
                    common_dl = [dl[k] for k in common_keys]
                    if common_base != common_dl:
                        logger.info('hold the same key {} with value{}'.format(common_keys, common_base))
                        # raise KeyError('base_dict and dl have the same key, but different values!')
                    for k in common_keys:
                        dl.pop(k)
                base_dict.update(dl)

            base_dict = Config._merge_x_2_y(cfg_dict, base_dict)
            content_list.append(cfg_content)

            cfg_dict = base_dict
            cfg_content = '\n'.join(content_list)

        return cfg_dict, cfg_content

    @staticmethod
    def _merge_x_2_y(x: dict, y: dict):
        y = y.copy()
        for key, value in x.items():
            if key in y and isinstance(value, dict) and value.pop(DELETE_KEY, False) is False:
                if isinstance(y[key], dict):
                    y[key] = Config._merge_x_2_y(value, y[key])
            else:
                y[key] = value

        return y

    @staticmethod
    def fromfile(file_name: str, use_predefined_var: bool = True):
        cfg_dict, cfg_content = Config._file2dict(file_name, use_predefined_var)

        cfg_dict = ToExpanduser.generate_obj(cfg_dict).to_expanduser()
        return Config(cfg_dict, cfg_content=cfg_content, file_name=file_name)

    def __init__(self, cfg_dict: dict = None, cfg_content: str = None, file_name: str = None):
        if cfg_dict is None:
            cfg_dict = dict()
        assert isinstance(cfg_dict, dict), 'cfg_dict must be  a dict type, but got {}'.format(type(cfg_dict))
        for k in cfg_dict:
            if k in RESERVED_KEYS:
                raise KeyError(f'the {k} in  cfg_dict is a reserved field!')

        super().__setattr__('_file_name', file_name)
        super().__setattr__('_cfg_eDict', mixkEasyDict(cfg_dict))

        if cfg_content is None and file_name is not None:
            with open(file_name, 'r') as rf:
                cfg_content = rf.read()
        super().__setattr__('_cfg_content', cfg_content)

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = mixkEasyDict(value)
        self._cfg_eDict.__setattr__(key, value)

    __setitem__ = __setattr__

    def __getattr__(self, key):
        return getattr(self._cfg_eDict, key)

    def __len__(self):
        return len(self._cfg_eDict)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = mixkEasyDict(value)
        self._cfg_eDict.__setitem__(key, value)

    def __repr__(self):
        return 'Config path:{} \n content:{}'.format(self.filename, self._cfg_eDict.__repr__())

    @property
    def filename(self):
        return self._file_name

    @property
    def cfg_content(self):
        return self._cfg_content

    def __iter__(self):
        return iter(self._cfg_eDict)


def get_mixk_cache_dir():
    return _CACHE_DIR


# @master_only_run
def set_mixk_cache_dir(cache_dir):
    global _CACHE_DIR
    _CACHE_DIR = ToExpanduser.modify_path(cache_dir)


def get_mixk_root():
    mixk_root = os.path.dirname(os.path.abspath(__file__))
    mixk_root = os.path.dirname(mixk_root)
    mixk_root = os.path.abspath(os.path.join(mixk_root, '..'))

    return mixk_root


# @master_only_run
def set_mixk_work_dir(work_dir):
    global _mixk_WORK_DIR
    _mixk_WORK_DIR = ToExpanduser.modify_path(work_dir)


def get_mixk_work_dir():
    return _mixk_WORK_DIR


def seed_all_rng(seed=None):
    logger = logging.getLogger(__name__)

    if seed is None:
        dt = datetime.now()
        seed = int(dt.strftime('%S%f'))
        seed += os.getpid()

        logger.info(f'Using a generated random seed {seed}')

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # a fixed seed for generating the hash()

    logger.info(f'seed : {seed}')
