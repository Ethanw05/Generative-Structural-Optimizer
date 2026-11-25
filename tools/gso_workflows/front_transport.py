"""Use the native monotone level-set update on the exact CAD-constrained material field."""
import ctypes
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np


@lru_cache(maxsize=None)
def native_front(executable):
    name = {'win32':'gso_field.dll', 'darwin':'libgso_field.dylib'}.get(sys.platform, 'libgso_field.so')
    library = ctypes.CDLL(str(Path(executable).resolve().with_name(name)))
    arrays = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags='C_CONTIGUOUS')
    function = library.gsoAdvanceFront
    function.argtypes = [ctypes.c_size_t]*3 + [arrays]*4 + [ctypes.c_float, arrays]
    function.restype = ctypes.c_int
    return function


def advance(field, speed, retained, restriction, spacing, executable):
    arrays = [np.ascontiguousarray(item, dtype=np.float32) for item in (field, speed, retained, restriction)]
    if arrays[0].ndim != 3 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('Front fields require a common three-dimensional grid')
    result = np.empty_like(arrays[0])
    status = native_front(executable)(*result.shape, *arrays, spacing, result)
    if status == 1:
        raise ValueError('Invalid native front field or spacing')
    if status != 0:
        raise RuntimeError('Unable to allocate or transport the front')
    return result
