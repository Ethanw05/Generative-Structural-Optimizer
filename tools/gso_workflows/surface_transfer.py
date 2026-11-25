"""Transfer native CAD quadrature to the corresponding outward-facing material boundary."""
import numpy as np
import trimesh
from .mesh_quality import InadmissibleBoundary
from .surface_locator import SurfaceLocator


def native_samples(face,max_spacing,with_normals=False):
    xyz=[];weights=[];normals=[]
    for tri in face.triangles:
        length=np.linalg.norm(tri-np.roll(tri,1,axis=0),axis=1).max()
        count=max(1,int(np.ceil(length/max_spacing)))
        uv=[(i+1/3,j+1/3) for i in range(count) for j in range(count-i)]
        uv.extend((i+2/3,j+2/3) for i in range(count-1) for j in range(count-1-i))
        xyz.append(tri[0]+np.asarray(uv)/count@(tri[1:]-tri[0]))
        area=np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2
        weights.append(np.full(len(uv),area/count**2))
        if with_normals:
            normal=np.cross(tri[1]-tri[0],tri[2]-tri[0])/(2.*area)
            normals.append(np.broadcast_to(normal,(len(uv),3)))
    if with_normals:return np.vstack(xyz),np.concatenate(weights),np.vstack(normals)
    return np.vstack(xyz),np.concatenate(weights)


def _receiver(points,skin,face,tolerance):
    points=np.asarray(points)
    triangles=points[skin]
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    lengths=np.linalg.norm(normal,axis=1)
    # A triangle reaching the CAD plane can have a remote centroid on a
    # graded or oblique mesh. Its normal interval is a conservative broad phase.
    plane_distance = (triangles-face.origin) @ face.normal
    eligible = ((normal @ face.normal > .35*lengths)
                & (plane_distance.min(axis=1) <= tolerance)
                & (plane_distance.max(axis=1) >= -tolerance))
    if not np.any(eligible):
        raise InadmissibleBoundary('Missing prescribed face on realized solid')
    return SurfaceLocator(points, skin[eligible])


def _quadrature(locator, face, tolerance, spacing):
    samples, areas = native_samples(face, spacing)
    cells, projected = locator.nearest(samples)
    nodes = locator.faces[cells]
    barycentric = trimesh.triangles.points_to_barycentric(locator.vertices[nodes], projected)
    distance = np.linalg.norm(projected-samples, axis=1)
    worst = float(distance.max())
    rms = float(np.sqrt(np.average(distance**2, weights=areas)))
    if worst > tolerance:
        raise InadmissibleBoundary(f'Prescribed face misses material: maximum {worst:.4f} mm, RMS {rms:.4f} mm')
    drift = np.average(projected-samples, axis=0, weights=areas)
    return nodes, areas[:,None]*barycentric, dict(max_mm=worst, rms_mm=rms, centroid_drift_mm=drift.tolist())


def project_face(points, skin, face, tolerance, spacing=.8):
    locator = _receiver(points, skin, face, tolerance)
    return _quadrature(locator, face, tolerance, spacing)


def project_fixture(points, skin, face, tolerance, spacing=.8):
    """Recover Dirichlet support on the receiving sheet of the selected CAD face."""
    locator = _receiver(points, skin, face, tolerance)
    nodes, weights, distances = _quadrature(locator, face, tolerance, spacing)
    epsilon = 64.*np.finfo(float).eps
    supported = nodes[weights > epsilon * weights.sum(axis=1)[:, None]]

    # Quadrature alone can undersample fine triangles. Trace each candidate
    # node back to the selected CAD patch and check which sheet receives it.
    nearby = np.unique(locator.faces)
    nearby = nearby[face.contains_projection(points[nearby], tolerance)]
    relative = (points[nearby] - face.origin) @ face.normal
    query = points[nearby] - relative[:,None]*face.normal
    cells, hits = locator.nearest(query)
    first_sheet = np.any(locator.faces[cells] == nearby[:,None], axis=1)
    roundoff = epsilon * max(1., np.max(np.abs(points)))
    # Shared edges can have several equally close receiving triangles.
    tied = np.abs(relative) <= np.linalg.norm(query-hits,axis=1) + roundoff
    return np.union1d(supported, nearby[first_sheet | tied]), float(weights.sum()), distances
