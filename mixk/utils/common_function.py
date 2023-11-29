from typing import Dict
import torch
import pickle

MAX_SIZE_LIMIT = 65533


def update_d1_with_d2(d1: Dict, d2: Dict):  # update d1 with d2
    if d2 is None:
        return

    for k, v in d2.items():
        d1[k] = d1.get(k, v)


def object_to_byte_tensor(obj, max_size=4094):
    """Encode Python objects to PyTorch byte tensors."""
    assert max_size <= MAX_SIZE_LIMIT
    byte_tensor = torch.zeros(max_size, dtype=torch.uint8)

    obj_enc = pickle.dumps(obj)
    obj_size = len(obj_enc)
    if obj_size > max_size:
        raise Exception(f'objects too large: object size {obj_size}, max size {max_size}')

    byte_tensor[0] = obj_size // 256
    byte_tensor[1] = obj_size % 256
    byte_tensor[2:2 + obj_size] = torch.ByteTensor(list(obj_enc))
    return byte_tensor


def byte_tensor_to_object(byte_tensor, max_size=4094):
    """Decode PyTorch byte tensors to Python objects."""
    assert max_size <= MAX_SIZE_LIMIT

    obj_size = byte_tensor[0].item() * 256 + byte_tensor[1].item()
    obj_enc = bytes(byte_tensor[2:2 + obj_size].tolist())
    obj = pickle.loads(obj_enc)
    return obj
