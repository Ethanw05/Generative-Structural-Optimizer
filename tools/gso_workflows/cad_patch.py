"""Selections on tessellated CAD faces with arbitrary normals, including bore walls."""
from functools import cached_property
import numpy as np
from scipy.spatial import cKDTree
import trimesh
from .surface_locator import SurfaceLocator
from .mesh_quality import InadmissibleBoundary
from .surface_transfer import native_samples


class TrianglePatch:
    def __init__(self,triangles):
        self.triangles=np.ascontiguousarray(triangles,dtype=float)
        if self.triangles.shape[1:]!=(3,3) or not len(self.triangles):
            raise ValueError('Empty CAD selection')

    @cached_property
    def locator(self):
        vertices=self.triangles.reshape(-1,3)
        return SurfaceLocator(vertices,np.arange(len(vertices)).reshape(-1,3))

    def contains_projection(self,points,tolerance):
        result=np.zeros(len(points),dtype=bool)
        bounds=np.array([self.triangles.min(axis=(0,1)),self.triangles.max(axis=(0,1))])
        near=np.flatnonzero(np.all((points>=bounds[0]-tolerance)&(points<=bounds[1]+tolerance),axis=1))
        if len(near):
            _,projected=self.locator.nearest(points[near])
            result[near]=np.linalg.norm(projected-points[near],axis=1)<=tolerance
        return result


def bore_wall_patches(mesh,face):
    """Select the first native wall band adjoining each inside loop of a picked face."""
    tree=cKDTree(mesh.vertices)
    edges=mesh.edges_sorted
    edge_owners={}
    for row in range(len(edges)):
        key=tuple(edges[row])
        edge_owners.setdefault(key,[]).append(row//3)
    group_by_face={int(index):group for group in mesh.facets for index in group}
    result=[]
    for loop in face.loops:
        if np.sum(loop[:,0]*np.roll(loop[:,1],-1)-loop[:,1]*np.roll(loop[:,0],-1))>=0.:continue
        distances,rim=tree.query(loop@face.basis[:2]+face.origin)
        if distances.max()>face.tolerance:raise ValueError('CAD rim did not match its source')
        wall=set()
        for a,b in zip(rim,np.roll(rim,-1),strict=True):
            owners=edge_owners.get(tuple(sorted((a,b))),())
            adjacent=[i for i in owners if mesh.face_normals[i]@face.normal < 1.-1.e-6]
            if len(adjacent)!=1:raise ValueError('Ambiguous bore wall adjoining the selected face')
            i=adjacent[0]
            wall.update(group_by_face.get(i,(i,)))
        result.append(TrianglePatch(mesh.triangles[sorted(wall)]))
    if not result:raise ValueError('Selected face has no bore wall')
    return tuple(result)


def project_patch(locator,patch,tolerance,spacing=.8):
    query,areas,normals=native_samples(patch,spacing,with_normals=True)
    cell,hit=locator.nearest(query)
    ids=locator.faces[cell]
    tri=locator.vertices[ids]
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    distance=np.linalg.norm(hit-query,axis=1)
    if distance.max()>tolerance or np.any(np.einsum('ij,ij->i',n,normals)<=0.):
        raise InadmissibleBoundary('Selected CAD patch lost its material support or orientation')
    bary=trimesh.triangles.points_to_barycentric(tri,hit)
    return ids,areas[:,None]*bary,dict(max_mm=float(distance.max()),
        rms_mm=float(np.sqrt(np.average(distance**2,weights=areas))),
        centroid_drift_mm=np.average(hit-query,axis=0,weights=areas).tolist())
