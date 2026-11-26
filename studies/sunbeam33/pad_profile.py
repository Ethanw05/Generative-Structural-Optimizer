"""Measure the exposed motor-pad edge from radial sections of the two actual surfaces."""
from pathlib import Path
import json
import sys
from geometry import WORK, ORIGIN, ROT, build_study
import numpy as np
import trimesh
from shapely.geometry import GeometryCollection,Polygon


def section_radii(mesh, center, depth, angles):
    segments = trimesh.intersections.mesh_plane(
        mesh, [0., 0., 1.], center - [0., 0., depth])[:, :, :2] - center[:2]
    start = segments[:, 0]
    edge = segments[:, 1] - start
    radii = []
    for angle in np.deg2rad(angles):
        x, y = np.cos(angle), np.sin(angle)
        divisor = x * edge[:, 1] - y * edge[:, 0]
        valid = np.abs(divisor) > 1.e-14
        ray = (start[valid, 0]*edge[valid, 1] - start[valid, 1]*edge[valid, 0]) / divisor[valid]
        t = (start[valid, 0]*y - start[valid, 1]*x) / divisor[valid]
        hits = ray[(t >= 0.) & (t <= 1.) & (ray > 0.)]
        if not len(hits):
            raise ValueError('Section missed the pad')
        radii.append(hits.max())
    return np.array(radii)


def local_surface(path):
    with np.load(path) as data:
        return trimesh.Trimesh((data['v']-ORIGIN) @ ROT.T, data['f'], process=False)


def underside_sections(mesh,study):
    load=study.loads['lift']
    face=load.face
    outer=max(face.loops,key=lambda loop:np.ptp(loop,axis=0).prod())
    perimeter=outer@face.basis[:2]+face.origin
    footprint=Polygon(perimeter[:,:2])
    underside=load.transfer_face.origin[2]
    rows=[]
    for depth in (.15,.55,2.2,4.4,8.8):
        cut=mesh.section(plane_normal=[0.,0.,1.],plane_origin=[0.,0.,underside-depth])
        region=GeometryCollection()
        if cut is not None:
            for ring in cut.discrete:
                if len(ring)>3:region=region.symmetric_difference(Polygon(ring[:,:2]))
        material=footprint.intersection(region)
        rows.append(dict(depth_mm=depth,area_mm2=float(material.area),
                         distance_from_far_rim_mm=float(perimeter[:,0].max()-material.bounds[2]) if not material.is_empty else None))
    return rows


def measure(reference, candidates):
    study = build_study()
    # The exterior loop of the recorded native lift face defines the circular pad's center.
    face = study.loads['lift'].face
    outer = max(face.loops,key=lambda q: np.ptp(q,axis=0).prod())
    center = (outer @ face.basis[:2] + face.origin).mean(0)
    # Restrict the measured arc to the distal side, away from the arm junction.
    angles = np.arange(-80,81)
    for depth in (.45,2.75,5.):
        native = section_radii(reference,center,depth,angles)
        metrics = {'depth_mm':depth,'rev2_radius_variation_mm': np.ptp(native)}
        for name,mesh in candidates.items():
            delta = section_radii(mesh,center,depth,angles)-native
            metrics[name] = {'min_mm':delta.min(),'max_mm':delta.max(),'median_mm':np.median(delta)}
        print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    measure(local_surface(WORK/'native-arm.npz'),
            {Path(p).stem:local_surface(Path(p)) for p in sys.argv[1:]})
