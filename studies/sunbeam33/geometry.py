"""Recorded face picks for the front-left ABS lift study. All picks are in native CAD millimetres."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'build/sunbeam33'
sys.path[:0]=[str(WORK/'meshlib-runtime'),str(WORK/'deps'),str(ROOT/'tools')]
import numpy as np
import manifold3d as md
import trimesh
from gso_workflows.design_space import DesignSpace,Face,closed_solid

ORIGIN=np.array([-13.5,-14.,13.5])
SPAN_AXIS=np.array([-138.9,0.,148.425]);SPAN_AXIS/=np.linalg.norm(SPAN_AXIS)
UP=np.array([0.,1.,0.])
ROT=np.stack([SPAN_AXIS,np.cross(UP,SPAN_AXIS),UP])


REV2_REFERENCE=Path(__file__).resolve().parent/'inputs'/'rev2'/'installed-arm.obj'


def build_study(obj=REV2_REFERENCE,assembly_geometry=None):
    from gso_workflows.volume_study import VolumeStudy,SurfaceLoad
    native=trimesh.load_mesh(obj,process=True)
    native.vertices=(native.vertices-ORIGIN)@ROT.T
    solid=closed_solid(native)
    def pick(position,normal):
        return Face.pick(native,(np.asarray(position)-ORIGIN)@ROT.T,np.asarray(normal)@ROT.T)
    load_face=pick([-155.3655,0.,165.8887],UP)
    motor_back=pick([-157.9551,-5.5,166.9293],-UP)
    hub_top=pick([-2.3139,-6.,11.8632],UP)
    hub_bottom=pick([-14.5155,-22.,1.8844],-UP)
    preserved=[];voids=[]
    for front,back in ((load_face,motor_back),(hub_top,hub_bottom)):
        thickness=(back.origin-front.origin)@front.normal
        retained,internal_void=front.retained_and_internal_void(back,solid)
        preserved.append(retained)
        # Preserve the CAD bores and internal voids inside the selected interface volume.
        voids.append(internal_void)
        voids.extend(front.open_hole_clearances(thickness,np.linalg.norm(native.extents),solid))
    low,high=native.bounds
    envelope=md.Manifold.cube(high-low).translate(low)
    if assembly_geometry is not None:
        from gso_workflows.assembly_geometry import obstacle_solids,occurrences
        from scipy.spatial import cKDTree
        occurrence='Sunbeam Optimized Arm - Front Left-1'
        placed=next(m for name,m in occurrences(assembly_geometry) if name==occurrence)
        native_points=native.vertices@ROT+ORIGIN
        translation=native_points.min(0)-placed.bounds[0]
        aligned=placed.vertices+translation
        if cKDTree(native_points).query(aligned)[0].max()>.01 or cKDTree(aligned).query(native_points)[0].max()>.01:
            raise ValueError('Assembly occurrence placement changed; select the verified native placement')
        transform=np.eye(4);transform[:3,:3]=ROT;transform[:3,3]=ROT@(translation-ORIGIN)
        voids.extend(obstacle_solids(assembly_geometry,occurrence,transform,envelope))
    space=DesignSpace(envelope,tuple(preserved),tuple(voids))
    return VolumeStudy(space,(hub_top,hub_bottom),{'lift':SurfaceLoad(load_face,33.*(UP@ROT.T),motor_back)},
                       young_mpa=2200.,poisson=.35,volume_limit_mm3=solid.volume(),
                       world_rotation=ROT,world_origin=ORIGIN,reference_surface_local=native)
