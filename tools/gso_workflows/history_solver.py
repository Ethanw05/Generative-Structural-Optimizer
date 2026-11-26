"""State histories at physical event times from the native nonlinear-contact Newmark integrator."""
import ctypes
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np


@lru_cache(maxsize=None)
def history_library(executable):
    name = {'win32':'gso_mechanics.dll','darwin':'libgso_mechanics.dylib'}.get(sys.platform,'libgso_mechanics.so')
    return ctypes.CDLL(str(Path(executable).resolve().with_name(name)))


@lru_cache(maxsize=None)
def history_operator(executable, accelerations=False):
    library=history_library(executable)
    matrix=np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS')
    vector=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    number=ctypes.c_double
    index=ctypes.c_int
    f=library.gsoResponseKinematicsAt if accelerations else library.gsoResponseHistoryAt
    f.argtypes=[index,index,vector,matrix,matrix,vector,index,matrix,matrix,index,matrix,vector,vector,number,number,matrix,vector]
    if accelerations: f.argtypes = [*f.argtypes,matrix]
    f.restype=ctypes.c_int
    return f


def integrate(executable,stiffness,mass,initial_velocity,step,forces,histories,normals,gaps,springs,damping=(0.,0.), *, record_acceleration=False):
    k,m,v,f,h,n,g,c=[np.ascontiguousarray(a,dtype=float) for a in
                      (stiffness,mass,initial_velocity,forces,histories,normals,gaps,springs)]
    size=len(v)
    if (k.shape!=(size,size) or m.shape!=k.shape or f.shape!=(len(h),size) or h.ndim!=2
            or h.shape[1]<2 or n.shape!=(len(g),size) or g.shape!=c.shape):
        raise ValueError('Inconsistent time history dimensions')
    times=(np.arange(h.shape[1],dtype=float)*step if np.ndim(step)==0
           else np.ascontiguousarray(step,dtype=float))
    if times.shape!=(h.shape[1],) or times[0]!=0. or not np.isfinite(times).all() or np.any(np.diff(times)<=0.):
        raise ValueError('Recording times must strictly increase from zero')
    states=np.zeros((h.shape[1],size))
    results=np.zeros(3)
    acceleration = np.empty_like(states) if record_acceleration else None
    extra = (acceleration,) if record_acceleration else ()
    status=history_operator(executable,record_acceleration)(size,len(states),times,k,m,v,len(f),f,h,len(g),n,g,c,
                                        *damping,states,results,*extra)
    if status or not np.isfinite(states).all():
        raise ValueError(f'Native transient equilibrium failed ({status})')
    report = dict(peak_energy_Nmm=results[0],peak_contact_N=results[1],residual=results[2])
    if record_acceleration:
        if not np.isfinite(acceleration).all():
            raise ValueError('Nonfinite transient acceleration')
        return states,report,acceleration
    return states,report
