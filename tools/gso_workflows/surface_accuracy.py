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
