"""Integrate a piecewise linear FE trace over the area of a selected CAD patch."""
from types import SimpleNamespace
import numpy as np
import shapely
import trimesh
from .cad_patch import TrianglePatch, project_patch
from .load_transfer import face_footprint
from .mesh_quality import InadmissibleBoundary
from .surface_locator import SurfaceLocator
from .surface_transfer import _receiver, _quadrature


def coplanar_weights(points, skin, face):
    """On each clipped receiving triangle, an affine field integrates at its area centroid."""
    tris=points[skin]
    eps=128.*np.finfo(float).eps*max(1.,np.max(np.abs(points)))
    distance=(tris-face.origin)@face.normal
    normal=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0])
    ids=skin[(np.max(np.abs(distance),axis=1)<=eps) & (normal@face.normal>0.)]
    if not len(ids):
        return None
    region=face_footprint(face)
    xy=(points[ids]-face.origin)@face.basis[:2].T
    lo,hi=np.asarray(region.bounds).reshape(2,2)
    nearby=np.all((xy.max(axis=1)>=lo) & (xy.min(axis=1)<=hi),axis=1)
    if not np.any(nearby):
        return None
    ids=ids[nearby]
    pieces=shapely.intersection(shapely.polygons(xy[nearby]),region)
    area=shapely.area(pieces)
    occupied=area>0.
    # A mixture of offset and coplanar receiving surfaces requires full projection.
    # Coverage and the summed area independently detect gaps and duplicate coverage.
    area_eps=max(eps**2,256.*np.finfo(float).eps*region.area)
    if abs(area.sum()-region.area)>area_eps or not np.any(occupied):
        return None
    if region.difference(shapely.union_all(pieces[occupied])).area>area_eps:
        return None
    centroids=shapely.get_coordinates(shapely.centroid(pieces[occupied]))
    nodes=ids[occupied]
    samples=centroids@face.basis[:2]+face.origin
    bary=trimesh.triangles.points_to_barycentric(points[nodes],samples)
    weights=np.zeros(len(points))
    np.add.at(weights,nodes.ravel(),(area[occupied,None]*bary).ravel())
    return weights


def batches(triangles,spacing):
    # This limits temporary memory while still integrating the full CAD selection.
    batch=[]
    count=0
    pending=list(triangles)
    while pending:
        tri=pending.pop()
        length=np.linalg.norm(tri-np.roll(tri,1,axis=0),axis=1).max()
        samples=max(1,int(np.ceil(length/spacing)))**2
        if samples>32768:
            a,b,c=tri
            ab,bc,ca=(a+b)/2.,(b+c)/2.,(c+a)/2.
            pending.extend(np.asarray(((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))))
            continue
        if batch and count+samples>32768:
            yield SimpleNamespace(triangles=np.asarray(batch))
            batch=[]
            count=0
        batch.append(tri)
        count+=samples
    if batch:
        yield SimpleNamespace(triangles=np.asarray(batch))


def integrated_weights(points,skin,face,tolerance,relative_tolerance):
    if not isinstance(face,TrianglePatch):
        exact=coplanar_weights(points,skin,face)
        if exact is not None:
            return exact
        locator=_receiver(points,skin,face,tolerance)
        project=lambda patch,spacing:_quadrature(locator,patch,tolerance,spacing)
    else:
        locator=SurfaceLocator(points,skin)
        project=lambda patch,spacing:project_patch(locator,patch,tolerance,spacing)
    lengths=np.linalg.norm(face.triangles-np.roll(face.triangles,1,axis=1),axis=2).max(axis=1)
    spacing=min(.8,float(np.max(lengths)))
    previous=None
    stable=0
    for _ in range(12):
        weights=np.zeros(len(points))
        for patch in batches(face.triangles,spacing):
            nodes,areas,_=project(patch,spacing)
            np.add.at(weights,nodes.ravel(),areas.ravel())
        if previous is not None:
            change=np.abs(weights-previous).sum()/weights.sum()
            stable=stable+1 if change<=relative_tolerance else 0
            if stable>=2:
                return weights
        previous=weights
        spacing*=.5
    raise InadmissibleBoundary('CAD receiver integration did not converge')
