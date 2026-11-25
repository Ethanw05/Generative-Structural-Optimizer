"""Force and moment transfer on selected CAD material through the common C++ wrench operator."""
import ctypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
from .surface_transfer import native_samples, project_face


@dataclass(frozen=True)
class SurfaceWrench:
    """Resultant about an explicit origin, distributed over one or more CAD face selections."""
    faces: tuple
    origin_mm: np.ndarray
    force_n: np.ndarray
    moment_nmm: np.ndarray


@lru_cache(maxsize=None)
def wrench_operator(executable):
    name = {'win32':'gso_mechanics.dll','darwin':'libgso_mechanics.dylib'}.get(sys.platform,'libgso_mechanics.so')
    library = ctypes.CDLL(str(Path(executable).resolve().with_name(name)))
    function = library.gsoDistributeWrench
    vector = np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    array = np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS')
    function.argtypes = [ctypes.c_size_t,array,vector,vector,vector,vector,array]
    function.restype = ctypes.c_int
    return function


def forces_at(load, points, areas, executable):
    points = np.ascontiguousarray(points,dtype=float)
    areas = np.ascontiguousarray(areas,dtype=float)
    origin, force, moment = [np.ascontiguousarray(v,dtype=float) for v in (load.origin_mm,load.force_n,load.moment_nmm)]
    if (points.ndim!=2 or points.shape[1]!=3 or areas.shape!=(len(points),)
            or any(v.shape!=(3,) for v in (origin,force,moment))):
        raise ValueError('Invalid wrench geometry or components')
    forces = np.empty_like(points)
    if wrench_operator(executable)(len(points),points,areas,origin,force,moment,forces):
        raise ValueError('Selected material cannot transmit the specified wrench')
    return forces


def grid_wrench(load,origin,shape,pitch,node_map,executable):
    from .design_space import surface_interpolation
    points=[];areas=[];ranges=[]
    for face in load.faces:
        q,a=native_samples(face,pitch*.5)
        points.append(q);areas.append(a);ranges.append(len(q))
    if not points:
        raise ValueError('A wrench requires at least one receiving face')
    forces=forces_at(load,np.vstack(points),np.concatenate(areas),executable)
    nodes=shape+1
    corners=np.indices((2,2,2)).reshape(3,8).T
    output=np.zeros((int(node_map.max())+1,3))
    offset=0
    for face,samples,count in zip(load.faces,points,ranges,strict=True):
        fallback=surface_interpolation(face,origin,shape,pitch,node_map)
        for q,force in zip((samples-origin)/pitch,forces[offset:offset+count],strict=True):
            cell=np.clip(np.floor(q).astype(int),0,shape-1)
            frac=np.clip(q-cell,0.,1.)
            weights=np.prod(np.where(corners,frac,1.-frac),axis=1)
            ids=node_map[np.ravel_multi_index((cell+corners).T,nodes)]
            live=weights>1.e-12
            if np.any(ids[live]<0):
                ids,weights=fallback(q)
            else:
                ids,weights=ids[live],weights[live]
            np.add.at(output,ids,weights[:,None]*force)
        offset+=count
    return output.ravel()


def boundary_wrench(load,local,skin,tolerance,executable):
    from .cad_patch import TrianglePatch,project_patch
    from .surface_locator import SurfaceLocator
    locator=SurfaceLocator(local,skin) if any(isinstance(face,TrianglePatch) for face in load.faces) else None
    ids=[];weights=[];points=[];areas=[];reports=[]
    for face in load.faces:
        nodes,area,report=(project_patch(locator,face,tolerance) if isinstance(face,TrianglePatch)
                           else project_face(local,skin,face,tolerance))
        sample_area=area.sum(axis=1)
        bary=area/sample_area[:,None]
        points.append(np.einsum('ij,ijk->ik',bary,local[nodes]))
        ids.append(nodes);weights.append(bary);areas.append(sample_area);reports.append(report)
    if not points:
        raise ValueError('A wrench requires at least one receiving face')
    forces=forces_at(load,np.vstack(points),np.concatenate(areas),executable)
    out=np.zeros_like(local)
    np.add.at(out,np.vstack(ids).ravel(),(np.vstack(weights)[:,:,None]*forces[:,None,:]).reshape(-1,3))
    return out,float(np.concatenate(areas).sum()),reports
