import torch
from torch.distributed import init_process_group, new_group  # destroy_process_group
import torch.multiprocessing.spawn as spawn
import logging
import random
import mixk.utils.distributed_info as dist_info
import os

__all__ = ['launch']


def launch(run_fn,
           gpus,
           *,
           machines=1,
           node_rank=0,
           master_addr='tcp://127.0.0.1',
           master_port='auto',
           run_fn_args=()) -> None:
    """
          Args:
              run_fn: a function that will be called by `run_fn(*args)`
              machines (int): the total number of machines
              node_rank (int): the rank of this machine (one per machine)
              master_addr (str): url to connect to for distributed jobs, including protocol
                             e.g. "tcp://127.0.0.1:8686".
                             Can be set to "auto" to automatically select a free port on localhost
              run_fn_args (tuple): arguments passed to run_fn
      """
    if torch.cuda.is_available() is False:
        raise RuntimeError('cuda is not available,please check if it is installed!')

    if master_port == 'auto':
        master_port = random.randint(2**14, 1**16)
    dist_url = f'{master_addr}:{master_port}'

    dist_world_size = gpus * machines
    set_pytorch_env_var(master_addr=master_addr, master_port=master_port, world_size=dist_world_size)

    if dist_world_size > 1:
        try:
            if machines > 1:
                is_right_dist_url(dist_url)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(e)
        else:
            spawn_fn_args = (run_fn, dist_world_size, gpus, node_rank, dist_url, run_fn_args)
            spawn(fn=_distributed_main, nprocs=gpus, args=spawn_fn_args)
        # finally:
        #     destroy_process_group()
    else:
        run_fn(*run_fn_args)


def is_right_dist_url(dist_url):
    if dist_url.startswith('tcp://') is False:
        raise Exception('dist_url dose not support the format{},please use {}'.format(dist_url, 'tcp://'))


def _distributed_main(local_rank, run_fn, dist_world_size, gpus, node_rank, dist_url, run_fn_args):
    if gpus > torch.cuda.device_count():
        raise RuntimeError('The number of GPUs per machine is greater than the number of  GPUs available')

    if dist_info._LOCAL_PROCESS_GROUP is not None:
        raise RuntimeError('_LOCAL_PROCESS_GROUP is {},unable to set up local process group'.format(
            dist_info._LOCAL_PROCESS_GROUP))

    global_rank_idx = gpus * node_rank + local_rank
    try:
        init_process_group(backend='nccl', world_size=dist_world_size, rank=global_rank_idx, init_method=dist_url)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(e)
    else:
        dist_info.synchronize()
        torch.cuda.set_device(local_rank)
        machine_nums = dist_world_size // gpus
        for idx in range(machine_nums):
            ranks_list = list(range(idx * gpus, (idx + 1) * gpus))
            dist_group = new_group(ranks_list)
            if idx == node_rank:
                dist_info._LOCAL_PROCESS_GROUP = dist_group
        run_fn(*run_fn_args)


class PytorchEnv:

    def __init__(self):
        self._current_env = os.environ.copy()

    def has(self, name):
        return name in self._current_env.keys()

    def __getattr__(self, name):
        if self.has(name):
            return self._current_env[name]
        else:
            raise AttributeError('there is no such {} in the current pytorch environment variable'.format(name))

    def __setattr__(self, name, value):
        try:
            if name in ['_current_env']:
                super().__setattr__(name, value)
            else:
                self._current_env[name] = value
        except Exception as e:
            raise e

    def __len__(self):
        return len(self._current_env)


def set_pytorch_env_var(master_addr, master_port, world_size, omp_num_threads=1):
    current_env_var = PytorchEnv()
    current_env_var.MASTER_ADDR = master_addr
    current_env_var.MASTER_PORT = str(master_port)
    current_env_var.WORLD_SIZE = str(world_size)
    current_env_var.OMP_NUM_THREADS = str(omp_num_threads)
