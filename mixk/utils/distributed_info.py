import enum
import functools
import pickle

import numpy as np
import torch
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""

G_NCCL = 'nccl'
G_GLOO = 'gloo'
G_CUDA = 'cuda'


class DistributedStatus(enum.Enum):
    AVAILABLE = 1
    INITIALIZED = 2
    AVAILABLE_AND_INITIALIZED = 3
    NO_AVAILABLE_INITIALIZED = 4


def get_dist_status():
    dist_package_status = dist.is_available()
    pg_init_status = dist.is_initialized()

    if dist_package_status and pg_init_status:
        return DistributedStatus.AVAILABLE_AND_INITIALIZED
    else:
        if dist_package_status:
            return DistributedStatus.AVAILABLE
        elif pg_init_status:
            return DistributedStatus.INITIALIZED
        else:
            return DistributedStatus.NO_AVAILABLE_INITIALIZED


def get_world_size() -> int:
    status = get_dist_status()
    if status is DistributedStatus.AVAILABLE_AND_INITIALIZED:
        return dist.get_world_size()
    if status in (DistributedStatus.AVAILABLE, DistributedStatus.INITIALIZED):
        return 1


def get_rank() -> int:
    status = get_dist_status()
    if status is DistributedStatus.AVAILABLE_AND_INITIALIZED:
        return dist.get_rank()
    if status in (DistributedStatus.AVAILABLE, DistributedStatus.INITIALIZED):
        return 0


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    status = get_dist_status()
    if status in (DistributedStatus.AVAILABLE, DistributedStatus.INITIALIZED):
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    if status is DistributedStatus.AVAILABLE_AND_INITIALIZED:
        return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
          i.e. the number of processes per machine.
    """
    status = get_dist_status()
    if status in (DistributedStatus.AVAILABLE, DistributedStatus.INITIALIZED):
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    if status is DistributedStatus.AVAILABLE_AND_INITIALIZED:
        return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


def master_only_run(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper


def synchronize() -> None:
    """Helper function to synchronize (barrier) among all processes when using
    distributed training."""
    status = get_dist_status()
    if status in (DistributedStatus.AVAILABLE, DistributedStatus.INITIALIZED):
        return
    num_processes = dist.get_world_size()
    if num_processes == 1:
        return
    else:
        dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """Return a process group based on gloo backend, containing all the ranks
    The result is cached."""
    if dist.get_backend() == G_NCCL:
        return dist.new_group(backend=G_GLOO)
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):  # serialize2tensor -> is_serialize
    backend = dist.get_backend(group=group)
    assert backend in [G_GLOO, G_NCCL]
    device = torch.device('cpu' if backend is G_GLOO else 'cuda')
    bytes_data = pickle.dumps(data)
    b2s = torch.ByteStorage.from_buffer(bytes_data)
    s2t = torch.ByteTensor(b2s).to(device=device)
    return s2t


def _pad_to_largest_tensor(tensor: torch.Tensor, group) -> tuple:
    """
          Returns:
              list[int]: size of the tensor, on each rank
              Tensor: padded tensor that has the max size
    """

    world_size = dist.get_world_size(group=group)
    if world_size < 1:
        raise Exception('dist.gather/all_gather must be called from ranks with in the given group', world_size)

    dtype = torch.int64
    device = tensor.device
    local_tensor_size = torch.tensor([tensor.numel()], dtype=dtype, device=device)
    tensor_sizes = [torch.zeros([1], dtype=dtype, device=device) for _ in range(world_size)]
    dist.all_gather(tensor_sizes, local_tensor_size, group=group)
    tensor_sizes = [int(size.item()) for size in tensor_sizes]
    max_tensor_size = max(tensor_sizes)

    if local_tensor_size != max_tensor_size:
        pad_size = max_tensor_size - local_tensor_size
        pad = torch.zeros((pad_size, ), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, pad), dim=0)

    return tensor, tensor_sizes


@functools.lru_cache()
def is_single_processes(group=None) -> bool:
    if get_world_size() == 1:
        return True
    else:
        if group is None:
            group = _get_global_gloo_group()
        if dist.get_world_size(group=group) == 1:
            return True
        else:
            return False


def all_gather(data, group=None) -> list:
    """Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """

    if is_single_processes(group):
        return [data]

    if group is None:
        group = _get_global_gloo_group()

    tensor = _serialize_to_tensor(data, group)
    tensor, tensor_sizes = _pad_to_largest_tensor(tensor, group)
    max_tensor_size = max(tensor_sizes)
    tensor_list = [torch.empty((max_tensor_size, ), dtype=torch.uint8, device=tensor.device) for _ in tensor_sizes]

    dist.all_gather(tensor_list, tensor, group=group)
    datum = []
    for length, tensor in zip(tensor_sizes, tensor_list):
        single_data = tensor.cpu().numpy().tobytes()
        single_data = single_data[:length]
        datum.append(pickle.loads(single_data))
    return datum


def gather(data, *, dst_rank=0, group=None) -> list:
    """Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """

    if is_single_processes(group=group):
        return [data]
    if group is None:
        group = _get_global_gloo_group()

    tensor = _serialize_to_tensor(data, group)
    tensor, tensor_sizes = _pad_to_largest_tensor(tensor, group)

    if dist.get_rank(group=group) == dst_rank:
        max_tensor_size = max(tensor_sizes)
        tensor_list = [torch.empty((max_tensor_size, ), dtype=torch.uint8, device=tensor.device) for _ in tensor_sizes]
        dist.gather(tensor, tensor_list, dst=dst_rank, group=group)

        datum = []
        for length, tensor in zip(tensor_sizes, tensor_list):
            single_data = tensor.cpu().numpy().tobytes()
            single_data = single_data[:length]
            datum.append(pickle.loads(single_data))
        return datum
    else:
        dist.gather(tensor, [], dst=dst_rank, group=group)
        return []


def shared_random_seed(low=2**31, select_idx=0) -> int:
    """
    Returns:
      int: a random number that is the same across all workers.
          If workers need a shared RNG, they can use this shared seed to
          create one.

    All workers must call this function, otherwise it will deadlock.
    """
    random_ints = np.random.randint(low)
    all_random_ints = all_gather(random_ints)
    if len(all_random_ints) < select_idx:
        return all_random_ints[0]
    else:
        return all_random_ints[select_idx]


def reduce_dict(input_dict: dict, is_average: bool = True) -> dict:
    """Reduce the values in the dictionary from all processes so that process
    with rank 0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        is_average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        values = torch.stack(list(input_dict.values()), dim=0)
        dist.reduce(values, dst=0)
        if is_main_process() and is_average:
            values /= world_size
        output_dict = {k: v for k, v in zip(input_dict.keys(), values)}
        return output_dict
