"""Call the native evolutionary driver with application-owned CAD and mechanics callbacks."""
import ctypes
from functools import lru_cache
from pathlib import Path
import sys

_Bytes = ctypes.POINTER(ctypes.c_ubyte)
_Update = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, _Bytes, _Bytes)
_Digest = ctypes.c_ubyte * 32


@lru_cache(maxsize=None)
def _operator(executable):
    filename = {'win32':'gso_evolution.dll','darwin':'libgso_evolution.dylib'}.get(sys.platform,'libgso_evolution.so')
    library = ctypes.CDLL(str(Path(executable).resolve().with_name(filename)))
    run = library.gsoFeedbackSweep
    run.argtypes = [ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint64, _Bytes, _Update, _Bytes]
    run.restype = ctypes.c_int
    return run


def run_sweep(executable, scenarios, seed, sweep, initial, update):
    names = sorted(scenarios)
    error = []

    @_Update
    def step(index, ordinal, parent, child):
        try:
            next_geometry = bytes.fromhex(update(names[index], ordinal, bytes(parent[:32]).hex()))
            if len(next_geometry) != 32:
                raise ValueError('A mechanical update returned an invalid geometry digest')
            ctypes.memmove(child, next_geometry, len(next_geometry))
            return 0
        except BaseException as caught:
            # Propagate through Python after the native driver has unwound.
            error.append(caught)
            return 1

    output = _Digest()
    first = _Digest.from_buffer_copy(bytes.fromhex(initial))
    status = _operator(executable)(len(names), seed, sweep, first, step, output)
    if error:
        raise error[0]
    if status:
        raise RuntimeError(f'Native feedback sweep failed ({status})')
    return bytes(output).hex()
