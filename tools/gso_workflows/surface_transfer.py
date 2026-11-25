"""Transfer native CAD quadrature to the corresponding outward-facing material boundary."""
import numpy as np
import trimesh
from .mesh_quality import InadmissibleBoundary
from scipy.spatial import cKDTree


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


def project_face(points,skin,face,tolerance,spacing=.8):
    points=np.asarray(points)
    triangles=points[skin]
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    lengths=np.linalg.norm(normal,axis=1)
    center=triangles.mean(axis=1)
    pick=((normal@face.normal)>.35*lengths)&(np.abs((center-face.origin)@face.normal)<2.*tolerance)
    eligible=np.flatnonzero(pick)
    if not len(eligible):raise InadmissibleBoundary('Missing prescribed face on realized solid')
    tree=cKDTree(center[eligible])
    samples,areas=native_samples(face,spacing)
    ids=[];weights=[];distances=[];positions=[]
    for start in range(0,len(samples),512):
        query=samples[start:start+512];a=areas[start:start+512]
        k=min(24,len(eligible))
        nearby=eligible[tree.query(query,k=k)[1]].reshape(len(query),k)
        tri=triangles[nearby]
        closest=trimesh.triangles.closest_point(tri.reshape(-1,3,3),np.repeat(query,k,axis=0)).reshape(len(query),k,3)
        distance=np.linalg.norm(closest-query[:,None,:],axis=2)
        which=np.argmin(distance,axis=1);row=np.arange(len(query))
        chosen=nearby[row,which];q=closest[row,which]
        bary=trimesh.triangles.points_to_barycentric(triangles[chosen],q)
        ids.append(skin[chosen]);weights.append(a[:,None]*bary);distances.append(distance[row,which]);positions.append(q)
    distance=np.concatenate(distances)
    worst=float(distance.max());rms=float(np.sqrt(np.average(distance**2,weights=areas)))
    if worst>tolerance:
        raise InadmissibleBoundary(f'Prescribed face misses material: maximum {worst:.4f} mm, RMS {rms:.4f} mm')
    return np.vstack(ids),np.vstack(weights),dict(max_mm=worst,rms_mm=rms,
            centroid_drift_mm=(np.average(np.vstack(positions)-samples,axis=0,weights=areas)).tolist())
