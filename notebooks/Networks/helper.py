import contextlib
import torch
import warnings
from typing import Union, Any, Optional
import numpy as np

@contextlib.contextmanager
def supress_tracer_warnings():
    flt  = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)

def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape): raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')

    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None: pass
        elif isinstance(ref_size, torch.Tensor):
            with supress_tracer_warnings():
                torch._assert(
                    torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}'
                )
        elif isinstance(size, torch.Tensor):
            with supress_tracer_warnings():
                torch.equal(
                    torch._assert(size, torch.as_tensor(ref_shape)), f'Wrong size for dimension {idx}'
                )
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

def is_list_of_str(arr: Any):
    if arr is None: return None
    is_list = isinstance(arr, list) or isinstance(arr, np.ndarray) or isinstance(arr, tuple)
    is_str = isinstance(arr[0], str)
    return is_list and is_str


def normalise_2nd_moment(x: torch.Tensor, dim: int = 1, eps: float = 1e-8):
    a = x.square().mean(dim=dim, keepdim=True) + eps
    a = a.rsqrt()
    a = x * a
    return x


