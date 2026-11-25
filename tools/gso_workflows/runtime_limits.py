"""Optional resource limits scoped to one optimizer run and its child processes."""
import ctypes
import os
import sys


def configure_cpu_limit(count):
    if count < 1:
        raise ValueError('The CPU limit must be positive')
    if sys.platform == 'win32':
        api = ctypes.windll.kernel32
        process = ctypes.c_void_p(-1)
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not api.GetProcessAffinityMask(process, ctypes.byref(allowed), ctypes.byref(system)):
            raise ctypes.WinError()
        bits = [1 << i for i in range(allowed.value.bit_length()) if allowed.value & (1 << i)]
        count = min(count, len(bits))
        if not api.SetProcessAffinityMask(process, ctypes.c_size_t(sum(bits[:count]))):
            raise ctypes.WinError()
        if not api.SetPriorityClass(process, 0x4000):  # BELOW_NORMAL_PRIORITY_CLASS
            raise ctypes.WinError()
    elif hasattr(os, 'sched_setaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        count = min(count, len(allowed))
        os.sched_setaffinity(0, allowed[:count])
    for key in ('GSO_WORKERS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'VTK_SMP_MAX_THREADS'):
        os.environ[key] = str(count)
    return count
