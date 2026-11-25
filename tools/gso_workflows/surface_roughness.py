"""Area-sampled departures from local quadratic patches, in model units."""
import numpy as np
from scipy.spatial import cKDTree


def surface_residuals(mesh, radius_mm, count=2048, seed=31567):
    """Return a local RMS height residual per sampled site and the fraction measured.

    This samples the actual triangles on an oriented solid. Sites with insufficient
    same-facing samples for a full quadratic fit are excluded from the statistic.
    CAD edges can contribute to the tail, so use the same radius on comparative parts.
    """
    if not np.isfinite(radius_mm) or radius_mm <= 0. or count < 1:
        raise ValueError('Sampling needs a positive length and site count')
    weights = np.cumsum(mesh.area_faces)
    if not np.isfinite(weights[-1]) or weights[-1] <= 0.:
        raise ValueError('Expected a finite surface area')
    rng = np.random.default_rng(seed)

    def sample(size):
        u, v, w = rng.random((3, size))
        face = np.searchsorted(weights, u*weights[-1], side='right')
        tri = mesh.vertices[mesh.faces[face]]
        fold = v+w > 1.
        v[fold], w[fold] = 1.-v[fold], 1.-w[fold]
        return tri[:,0] + v[:,None]*(tri[:,1]-tri[:,0]) + w[:,None]*(tri[:,2]-tri[:,0]), face

    sites, site_faces = sample(count)
    # A flat circular neighborhood has 64 support samples in expectation.
    support_count = max(64, int(np.ceil(64.*weights[-1]/(np.pi*radius_mm**2))))
    samples, faces = sample(support_count)
    normals = mesh.face_normals
    tree = cKDTree(samples)
    residuals = []
    for site, face, ids in zip(sites, site_faces, tree.query_ball_point(sites,radius_mm), strict=True):
        normal = normals[face]
        ids = np.asarray(ids, dtype=np.intp)
        ids = ids[normals[faces[ids]] @ normal > .5]
        if len(ids) < 12:
            continue
        offset = (samples[ids]-site)/radius_mm
        tangent = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
        tangent /= np.linalg.norm(tangent)
        u, v = offset @ tangent, offset @ np.cross(normal,tangent)
        heights = offset @ normal
        basis = np.column_stack((np.ones(len(u)),u,v,u*u,u*v,v*v))
        radial = np.maximum(0.,1.-np.einsum('ij,ij->i',offset,offset))
        kernel = np.sqrt(radial)
        fit, _, rank, _ = np.linalg.lstsq(basis*kernel[:,None], heights*kernel, rcond=None)
        if rank == 6:
            error = np.average((heights-basis@fit)**2, weights=radial)
            residuals.append(radius_mm*np.sqrt(error))
    if not residuals:
        raise ValueError('No site supported a full local quadratic fit')
    return np.array(residuals), len(residuals)/count


def crease_length(mesh, angle_degrees):
    """Total length of all internal mesh edges sharper than a specified dihedral."""
    if not np.isfinite(angle_degrees) or not 0. < angle_degrees < 180.:
        raise ValueError('The dihedral threshold must lie between 0 and 180 degrees')
    sharp = mesh.face_adjacency_angles > np.deg2rad(angle_degrees)
    edges = mesh.face_adjacency_edges[sharp]
    return float(np.linalg.norm(mesh.vertices[edges[:,1]]-mesh.vertices[edges[:,0]],axis=1).sum())
