import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import PIL
import torch
from tabulate import tabulate
import re

__all__ = ['collect_env_info']


class EnvironmentInfo:

    def __init__(self):
        self._data = []

    def _gpu_info(self):
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME is not None:
            try:
                nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
                nvcc = subprocess.check_output("'{}' -V".format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip().split('\n')[-1]
            except subprocess.SubprocessError:
                nvcc = 'Not found'

            self._data.append(('CUDA_HOME', str(CUDA_HOME)))
            self._data.append(('CUDA compiler', nvcc))

            devices = defaultdict(list)
            for k in range(torch.cuda.device_count()):
                devices[torch.cuda.get_device_name(k)].append(str(k))
            for name, devids in devices.items():
                self._data.append(('GPU ' + ','.join(devids), name))

            cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
            if cuda_arch_list:
                self._data.append(('TORCH_CUDA_ARCH_LIST', cuda_arch_list))

    def _related_lib_info(self):
        self._data.append(('Python', sys.version.replace('\n', '')))
        self._data.append(('numpy', np.__version__))
        self._data.append(('Pillow', PIL.__version__))

    def _latform_info(self):
        self._data.append(('sys.platform', sys.platform))
        if sys.platform != 'win32':
            try:
                # this is how torch/utils/cpp_extensions.py choose compiler
                cxx = os.environ.get('CXX', 'c++')
                cxx = subprocess.check_output("'{}' --version".format(cxx), shell=True)
                cxx = cxx.decode('utf-8').strip().split('\n')[0]
            except subprocess.SubprocessError:
                cxx = 'Not found'
            self._data.append(('Compiler', cxx))

    def _torch_info(self):
        self._data.append(('PyTorch', torch.__version__ + ' @' + os.path.dirname(torch.__file__)))
        self._data.append(('PyTorch debug build', torch.version.debug))
        self._data.append(('PyTorch configuration info', torch.__config__.show()))

    def _env_module(self):
        self._data.append(('mixk_ENV_MODULE', os.environ.get('mixk_ENV_MODULE', '<not set>')))

    def _compute_compatibility_info(self, CUDA_HOME, so_file):
        try:
            cuobjdump = os.path.join(CUDA_HOME, 'bin', 'cuobjdump')
            if os.path.isfile(cuobjdump):
                output = subprocess.check_output("'{}' --list-elf '{}'".format(cuobjdump, so_file), shell=True)
                output = output.decode('utf-8').strip().split('\n')
                sm = []
                for line in output:
                    line = re.findall(r'\.sm_[0-9]*\.', line)[0]
                    sm.append(line.strip('.'))
                sm = sorted(set(sm))
                return ', '.join(sm)
            else:
                return so_file + '; cannot find cuobjdump'
        except Exception:
            # unhandled failure
            return so_file

    def get_env_info(self):
        if torch.cuda.is_available():
            self._data.append(('GPU available', True))
            self._gpu_info()
        else:
            self._data.append(('GPU available', False))

        self._related_lib_info()
        self._latform_info()
        self._torch_info()
        self._env_module()

    @property
    def env_info(self):
        return self._data


def collect_env_info():
    env_info = EnvironmentInfo()
    env_info.get_env_info()
    return tabulate(env_info.env_info)
