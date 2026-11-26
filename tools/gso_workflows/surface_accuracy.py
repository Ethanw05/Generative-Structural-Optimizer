"""Check both directions of a proposed surface discretization at vertices and face samples."""
import numpy as np
from meshlib import mrmeshpy as mr, mrmeshnumpy as arrays
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
from vtkmodules.vtkFiltersGeneral import vtkSampleImplicitFunctionFilter
from .cad_motion import polydata


def sample_deviation(first, second, limit):
    native = [arrays.meshFromFacesVerts(np.asarray(m.faces, dtype=np.int32),
                                        np.asarray(m.vertices, dtype=float)) for m in (first, second)]
    # The capped native BVH query rejects out-of-tolerance geometry before scanning
    # the face interiors. A cap above the limit distinguishes a failed query from a tie.
    maximum = np.sqrt(mr.findMaxDistanceSq(native[0], native[1], None, (2.*limit)**2))
    if maximum > limit:
        return maximum
    for source, target in ((first, second), (second, first)):
        distance = vtkImplicitPolyDataDistance()
        distance.SetInput(polydata(target.vertices, target.faces))
        def check(points):
            sample = vtkSampleImplicitFunctionFilter()
            sample.SetInputData(polydata(points))
            sample.SetImplicitFunction(distance)
            sample.Update()
            return float(np.max(np.abs(vtk_to_numpy(sample.GetOutput().GetPointData().GetScalars()))))
        for start in range(0, len(source.faces), 32768):
            tri = source.vertices[source.faces[start:start+32768]]
            samples = np.concatenate((tri.mean(axis=1), ((tri + np.roll(tri, 1, axis=1)) / 2.).reshape(-1, 3)))
            maximum = max(maximum, check(samples))
            if maximum > limit:
                return maximum
    return maximum


def sample_offsets(source,target,count,seed):
    """Measure signed distances from area-uniform source samples to the exact target triangles."""
    if count<1:
        raise ValueError('At least one area sample is required')
    rng=np.random.default_rng(seed)
    area=source.area_faces
    indices=rng.choice(len(area),size=count,p=area/area.sum())
    triangle=source.vertices[source.faces[indices]]
    uv=rng.random((count,2))
    flip=uv.sum(axis=1)>1.
    uv[flip]=1.-uv[flip]
    points=triangle[:,0]+uv[:,:1]*(triangle[:,1]-triangle[:,0])+uv[:,1:]*(triangle[:,2]-triangle[:,0])
    distance=vtkImplicitPolyDataDistance()
    distance.SetInput(polydata(target.vertices,target.faces))
    sample=vtkSampleImplicitFunctionFilter()
    sample.SetInputData(polydata(points))
    sample.SetImplicitFunction(distance)
    sample.Update()
    offsets=vtk_to_numpy(sample.GetOutput().GetPointData().GetScalars()).astype(float)
    if offsets.shape!=(count,) or not np.isfinite(offsets).all():
        raise ValueError('Incomplete surface-distance measurements')
    return offsets
