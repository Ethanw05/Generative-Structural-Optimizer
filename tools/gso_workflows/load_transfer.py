"""Integrate the actual traction carried through a retained planar interface's receiving side."""
import numpy as np
import shapely


def face_footprint(face):
    region=shapely.GeometryCollection()
    for loop in face.loops:
        region=shapely.symmetric_difference(region,shapely.Polygon(loop))
    return region


def section_response(local,t,sigma,study,load,depth):
    front,back=load.face,load.transfer_face
    if front.normal@back.normal>-.999:
        raise ValueError('Load spreading requires an opposed planar receiving face')
    heights=(local-back.origin)@back.normal-depth
    ht=heights[t]
    selected=np.flatnonzero((ht.min(1)<0.)&(ht.max(1)>=0.))
    h=ht[selected];tet=local[t[selected]]
    pieces=[];owners=[]
    for i in range(1,15):
        pattern=(h>=0)@np.array([1,2,4,8])
        ids=np.flatnonzero(pattern==i)
        if not len(ids):continue
        points=[]
        for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
            if bool(i&(1<<a))==bool(i&(1<<b)):continue
            u=h[ids,a]/(h[ids,a]-h[ids,b])
            points.append(tet[ids,a]+u[:,None]*(tet[ids,b]-tet[ids,a]))
        xy=(np.stack(points,axis=1)-front.origin)@front.basis[:2].T
        centered=xy-xy.mean(1)[:,None,:]
        order=np.argsort(np.arctan2(centered[:,:,1],centered[:,:,0]),axis=1)
        polys=shapely.polygons(np.take_along_axis(xy,order[:,:,None],axis=1))
        pieces.extend(polys);owners.extend(selected[ids])
    region=face_footprint(front)
    if not owners:
        return dict(depth_mm=float(depth),section_area_mm2=0.,face_area_mm2=float(region.area),
                    effective_area_mm2=0.,transmitted_force_N=0.,traction_mismatch=1.)
    cut=shapely.intersection(pieces,region)
    area=shapely.area(cut)
    # The normal facing the applied side gives positive transmitted load under pure tension.
    normal_world=front.normal@study.world_rotation
    force_world=load.force_n@study.world_rotation
    traction=(sigma[np.asarray(owners)]@normal_world)@(force_world/np.linalg.norm(force_world))
    force=float(area@traction)
    square=float(area@(traction**2))
    target_area=float(region.area);pressure=np.linalg.norm(force_world)/target_area
    mismatch=(square-2.*pressure*force+pressure**2*target_area)/(pressure**2*target_area)
    return dict(depth_mm=float(depth),section_area_mm2=float(area.sum()),face_area_mm2=target_area,
                effective_area_mm2=float(force**2/square) if square>0 else 0.,
                transmitted_force_N=force,traction_mismatch=float(max(mismatch,0.)))


def loaded_neighborhood(local,cells,sample_faces,vm,volume,order):
    centers=local[cells].mean(1)
    nearby=np.zeros(len(cells),dtype=bool)
    radii=[]
    for face in sample_faces:
        triangle=face.triangles
        area=np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1)
        center=np.average(triangle.mean(1),axis=0,weights=area)
        radius=1.5*np.linalg.norm(triangle-center,axis=2).max()
        nearby |= np.linalg.norm(centers-center,axis=1)<=radius
        radii.append(float(radius))
    subset=order[nearby[order]]
    return dict(radius_mm=max(radii),volume_mm3=float(volume[subset].sum()),
        p99_MPa=float(np.interp(.99*volume[subset].sum(),np.cumsum(volume[subset]),vm[subset])),
        peak_MPa=float(vm[subset].max()))
