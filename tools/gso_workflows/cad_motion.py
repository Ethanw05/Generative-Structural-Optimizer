"""Linearized one-sided CAD bounds for a boundary velocity. Exact solids also constrain extraction."""
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
from vtkmodules.vtkFiltersGeneral import vtkSampleImplicitFunctionFilter


def polydata(vertices, faces=None):
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices, dtype=float), deep=True))
    surface = vtkPolyData()
    surface.SetPoints(points)
    if faces is not None:
        cells = vtkCellArray()
        indices = np.ascontiguousarray(faces.ravel(), dtype=np.int64)
        offsets = np.arange(0, len(indices) + 1, 3, dtype=np.int64)
        cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                      numpy_to_vtkIdTypeArray(indices, deep=True))
        surface.SetPolys(cells)
    return surface


def _signed_distance(function, points):
    sample = vtkSampleImplicitFunctionFilter()
    sample.SetInputData(polydata(points))
    sample.SetImplicitFunction(function)
    sample.Update()
    return vtk_to_numpy(sample.GetOutput().GetPointData().GetScalars()).copy()


def cad_bounds(space, vertices, faces, move, tolerance):
    tri = vertices[faces]
    vectors = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths == 0.) or tolerance <= 0. or move <= 0.:
        raise ValueError('CAD motion requires a nondegenerate surface and a positive step')
    # A level-set front can sweep laterally beside a wall at a geometric crease.
    # Keep each incident face direction separate until every CAD bound has been applied.
    nodes = faces.ravel()
    normals = np.repeat(vectors/lengths[:, None], 3, axis=0)
    allowed = np.ones((len(nodes), 2), dtype=bool)
    for kind, solids in (('preserved', space.preserved), ('excluded', space.exclusions),
                         ('envelope', (space.envelope,))):
        inward = kind == 'preserved'
        column = 0 if inward else 1
        sense = -1. if inward else 1.
        sign = -1. if kind == 'envelope' else 1.
        for solid in solids:
            if solid.is_empty():
                continue
            box = np.asarray(solid.bounding_box()).reshape(2, 3)
            nearby = np.all((vertices >= box[0]-move-tolerance) & (vertices <= box[1]+move+tolerance), axis=1)
            sites = np.flatnonzero(nearby)
            if not len(sites):
                continue
            native = solid.to_mesh64()
            function = vtkImplicitPolyDataDistance()
            function.SetInput(polydata(native.vert_properties[:, :3], native.tri_verts))
            clearance = np.full(len(vertices), np.inf)
            clearance[sites] = sign*_signed_distance(function, vertices[sites])
            corners = np.flatnonzero((clearance[nodes] <= tolerance) & allowed[:, column])
            if not len(corners):
                continue
            ids = nodes[corners]
            ahead = sign*_signed_distance(function, vertices[ids]+sense*tolerance*normals[corners])
            # A trace already within the CAD sampling error may move tangentially
            # or back into the admissible side. Exact solids also clip the transported front.
            roundoff = 64.*np.finfo(float).eps*max(1., np.abs(box).max())
            allowed[corners, column] = ahead >= np.minimum(0., clearance[ids])-roundoff
    movable = np.zeros((len(vertices), 2), dtype=bool)
    np.logical_or.at(movable, nodes, allowed)
    return movable*np.array([-move, move])
