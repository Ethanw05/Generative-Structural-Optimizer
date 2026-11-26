"""Scoped oneMKL out-of-core Cholesky factors for positive-definite elastic stiffness."""
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock

import numpy as np
from scipy import sparse


# oneMKL selects its factor-file prefix from process-wide environment variables.
_environment_lock=RLock()


@contextmanager
def disk_backed_stiffness(matrix,directory,ram_mb):
    from pypardiso import PyPardisoSolver
    if not isinstance(ram_mb,int) or ram_mb<=0:
        raise ValueError('The numerical factor memory budget must be a positive integer in MB')
    base=Path(directory).resolve()
    base.mkdir(parents=True,exist_ok=True)
    with _environment_lock,TemporaryDirectory(dir=base,prefix='factors-') as work:
        settings=dict(MKL_PARDISO_OOC_FILE_NAME=str(Path(work)/'stiffness'),
                      MKL_PARDISO_OOC_MAX_CORE_SIZE=str(ram_mb),
                      MKL_PARDISO_OOC_MAX_SWAP_SIZE='0',MKL_PARDISO_OOC_KEEP_FILE='0')
        previous={key:os.environ.get(key) for key in settings}
        os.environ.update(settings)
        try:
            solver=PyPardisoSolver(mtype=2)
            # Load the installed runtime's defaults before changing just its storage mode.
            initialise=solver.libmkl.pardisoinit
            initialise.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int32),ctypes.c_void_p]
            initialise.restype=None
            initialise(solver.pt.ctypes.data,ctypes.byref(ctypes.c_int32(2)),solver.iparm.ctypes.data)
            solver.set_iparm(1,1)
            solver.set_iparm(60,2)
            upper=sparse.triu(matrix,format='csr')
            upper.sort_indices()
            try:
                solver.factorize(upper)
                yield lambda rhs:solver.solve(upper,np.asfortranarray(rhs))
            finally:
                solver.free_memory(everything=True)
        finally:
            for key,value in previous.items():
                if value is None:
                    os.environ.pop(key,None)
                else:
                    os.environ[key]=value
