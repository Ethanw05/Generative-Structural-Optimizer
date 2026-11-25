"""Read-only acceptance checks for a candidate boundary."""
import numpy as np
import pymeshlab
from scipy.spatial import cKDTree
from .triangle_contact import improper_contact


class InadmissibleBoundary(ValueError):
    """A proposed physical part cannot be evaluated as a solid."""


class DisconnectedMaterial(InadmissibleBoundary):
    """The exported single-part candidate contains multiple material bodies."""


def intersection_count(vertices, triangles):
    check=pymeshlab.MeshSet()
    check.add_mesh(pymeshlab.Mesh(vertices,triangles))
    check.compute_selection_by_self_intersections_per_face()
    flagged=np.flatnonzero(check.current_mesh().face_selection_array())
    if not len(flagged):return 0
    xyz=vertices[triangles]
    low,high=xyz.min(1),xyz.max(1)
    centers=.5*(low+high)
    radius=np.linalg.norm(high-centers,axis=1)
    tree=cKDTree(centers)
    intersections=set()
    tested=set()
    for i in flagged:
        ids=np.asarray(tree.query_ball_point(centers[i],radius[i]+radius.max()),int)
        ids=ids[np.all((low[ids]<=high[i])&(high[ids]>=low[i]),axis=1)]
        for j in ids:
            pair=(min(i,j),max(i,j))
            if i!=j and pair not in tested:
                tested.add(pair)
                if improper_contact(vertices,triangles[i],triangles[j]):
                    intersections.update(pair)
    return len(intersections)


def validate_single_part(mesh):
    if not mesh.is_volume or not np.all(np.isfinite(mesh.vertices)):
        raise InadmissibleBoundary('The candidate boundary is not an oriented solid')
    if mesh.body_count!=1:
        shells=mesh.split(only_watertight=False)
        pieces=sum(shell.volume>0. for shell in shells)
        if pieces!=1:
            positive=sorted((s for s in shells if s.volume>0.),key=lambda s:s.volume,reverse=True)
            islands=[(float(s.volume),s.bounds.mean(0).round(2).tolist()) for s in positive[:8]]
            raise DisconnectedMaterial(f'Candidate contains {pieces} material bodies (mm3, center mm): {islands}')
    count=intersection_count(mesh.vertices,mesh.faces)
    if count:
        raise InadmissibleBoundary(f'Candidate contains {count} intersecting triangles')
