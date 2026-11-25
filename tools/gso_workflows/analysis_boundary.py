"""Verify the material boundary of a tetrahedral analysis discretization before using its response."""
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from .mesh_quality import InadmissibleBoundary,validate_single_part


def exterior(t):
    faces=np.concatenate([t[:,[0,2,1]],t[:,[0,1,3]],t[:,[0,3,2]],t[:,[1,2,3]]])
    _,first,counts=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_counts=True)
    if np.any(counts>2):
        raise InadmissibleBoundary('Analysis volume contains a tetrahedral face with more than two owners')
    first=first[counts==1]
    return faces[first],first%len(t)


def require_closed_fans(faces):
    """Every boundary edge must have two owners and every vertex a single circular fan."""
    corners=np.arange(faces.size).reshape(-1,3)
    ends=np.roll(corners,-1,axis=1).ravel()
    starts=corners.ravel()
    vertices=faces.ravel()
    edges=np.sort(np.column_stack((vertices[starts],vertices[ends])),axis=1)
    _,group,count=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    if np.any(count!=2):
        raise InadmissibleBoundary(f'Analysis boundary has {np.count_nonzero(count!=2)} nonmanifold edges')
    pairs=np.argsort(group,kind='stable').reshape(-1,2)
    first,second=pairs.T
    if np.any(vertices[starts[first]]!=vertices[ends[second]]):
        raise InadmissibleBoundary('Analysis boundary has inconsistent face winding')
    links=(np.r_[starts[first],ends[first]],np.r_[ends[second],starts[second]])
    graph=sparse.coo_matrix((np.ones(len(links[0]),dtype=bool),links),shape=(faces.size,faces.size))
    fans=connected_components(graph,directed=False,return_labels=False)
    if fans!=len(np.unique(vertices)):
        raise InadmissibleBoundary('Analysis boundary has a disconnected vertex fan')


def analysis_skin(points,cells):
    faces,owner=exterior(cells)
    tri=points[faces]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    inward=np.einsum('ij,ij->i',normal,tri.mean(1)-points[cells[owner]].mean(1))<0.
    faces[inward]=faces[inward,::-1]
    require_closed_fans(faces)
    used,index=np.unique(faces,return_inverse=True)
    skin=trimesh.Trimesh(points[used],index.reshape(-1,3),process=False)
    validate_single_part(skin)
    return skin
