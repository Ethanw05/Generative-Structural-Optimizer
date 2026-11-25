"""Retain enclosed voids from the generated geometry as explicit volume-meshing constraints."""
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from .mesh_quality import InadmissibleBoundary,validate_single_part


def material_shells(mesh):
    if mesh.body_count==1:
        return mesh,()
    faces=np.asarray(mesh.faces)
    # Label vertex-connected components without welding, healing, or reorienting triangles.
    graph=sparse.coo_matrix((np.ones(2*len(faces),dtype=bool),
                            (np.r_[faces[:,0],faces[:,1]],np.r_[faces[:,1],faces[:,2]])),
                           shape=(len(mesh.vertices),)*2)
    count,labels=connected_components(graph,directed=False)
    group=labels[faces[:,0]]
    shells=[]
    for index in range(count):
        selected=faces[group==index]
        if not len(selected):continue
        vertices,remap=np.unique(selected,return_inverse=True)
        shells.append(trimesh.Trimesh(mesh.vertices[vertices],remap.reshape(-1,3),process=False))
    material=[part for part in shells if part.volume>0.]
    if len(material)!=1 or any(part.volume==0. for part in shells):
        raise InadmissibleBoundary('A single optimized part requires one positive exterior shell')
    return material[0],tuple(part for part in shells if part.volume<0.)


def topology(shells):
    exterior,voids=shells
    return int(exterior.euler_number),tuple(sorted(int(cavity.euler_number) for cavity in voids))


def constrained_boundary(source_shells,discrete_surface):
    outer,voids=material_shells(discrete_surface)
    reference,cavities=source_shells
    if abs(outer.volume/reference.volume-1.)>.003:
        raise InadmissibleBoundary('Analysis changed the volume enclosed by the exterior surface')
    if outer.euler_number!=reference.euler_number:
        raise InadmissibleBoundary('Analysis changed the topology of the external material surface')
    if not cavities:
        if voids:raise InadmissibleBoundary('Analysis introduced an enclosed void')
        return None
    # The outer trace supplies the requested analysis spacing. Every original
    # interior triangle remains a PLC facet, even for a pore smaller than that spacing.
    boundary=trimesh.util.concatenate([outer,*cavities])
    validate_single_part(boundary)
    return boundary

def verify_constraints(reference_shells,discrete_surface):
    actual=material_shells(discrete_surface)
    if topology(reference_shells)!=topology(actual):
        raise InadmissibleBoundary('Volume discretization changed a through-hole or an enclosed cavity')
    original=reference_shells[1]
    if not original:return
    from scipy.optimize import linear_sum_assignment
    before=np.array([part.bounds.ravel() for part in original])
    after=np.array([part.bounds.ravel() for part in actual[1]])
    cost=np.linalg.norm(before[:,None]-after[None,:],axis=2)
    left,right=linear_sum_assignment(cost)
    tolerance=1024.*np.finfo(float).eps*max(1.,np.abs(reference_shells[0].bounds).max())
    for i,j in zip(left,right,strict=True):
        if cost[i,j]>4.*tolerance or abs(original[i].volume-actual[1][j].volume)>tolerance*original[i].area:
            raise InadmissibleBoundary(
                f'Enclosed boundary at {before[i].tolist()} moved by {cost[i,j]:.12g} mm '
                f'(limit {4.*tolerance:.12g}); volume changed by '
                f'{abs(original[i].volume-actual[1][j].volume):.12g} mm3 '
                f'(limit {tolerance*original[i].area:.12g})')
