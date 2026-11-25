"""Bound complete blocks of records without discarding any possible local stress maximum."""
import ctypes
from functools import lru_cache
import numpy as np
from .history_solver import history_library


@lru_cache(maxsize=None)
def envelope_operator(executable):
    f=history_library(executable).gsoHistoryEnvelope
    real=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    index=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    f.argtypes=[ctypes.c_int]*5+[real,real,real,index,real,index]
    f.restype=ctypes.c_int
    return f


class HistoryEnvelope:
    """Share one ordering and upper bounds across all stress and displacement measurements."""
    def __init__(self,states,executable):
        states=np.asarray(states,dtype=float)
        if states.ndim!=2 or min(states.shape)<1 or not np.isfinite(states).all():
            raise ValueError('Invalid modal displacement records')
        _,axes=np.linalg.eigh(states.T@states)
        self.axes=axes[:,::-1].copy()
        projected=states@self.axes
        self.order=np.ascontiguousarray(np.argsort(-np.abs(projected[:,0]),kind='stable'),dtype=np.int64)
        self.states=np.ascontiguousarray(projected[self.order])
        self.block_size=24
        self.bounds=np.maximum.reduceat(np.abs(self.states),np.arange(0,len(states),self.block_size),axis=0)
        self.apply=envelope_operator(executable)

    def measure(self,coefficients):
        maps=np.ascontiguousarray(coefficients@self.axes)
        if not np.isfinite(maps).all():
            raise ValueError('Invalid local response coefficients')
        size,components,modes=maps.shape
        maximum=np.empty(size)
        when=np.empty(size,dtype=np.int64)
        status=self.apply(size,components,modes,len(self.states),self.block_size,maps,self.states,
                          self.bounds,self.order,maximum,when)
        if status or not np.isfinite(maximum).all():
            raise ValueError('Native history envelope recovery failed')
        return maximum,when
