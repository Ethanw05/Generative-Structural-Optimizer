"""Bolt-group and translational load definitions in the recorded arm's local CAD frame."""
import numpy as np
import trimesh
from gso_workflows.cad_patch import bore_wall_patches
from gso_workflows.cad_wrench import SurfaceWrench


def motor_mount_moment(study,moment_nmm):
    native=study.reference_surface_local
    if native is None:raise ValueError('Bolt-wall selection requires the imported part boundary')
    face=study.loads['lift'].face
    walls=bore_wall_patches(native,face)
    centers=[np.average(p.triangles.mean(axis=1),axis=0,weights=trimesh.triangles.area(p.triangles)) for p in walls]
    origin=np.mean(centers,axis=0)
    return SurfaceWrench(walls,origin,np.zeros(3),face.normal*moment_nmm)


def motor_hardware(study,assembly_mass_kg,center_height_mm,central_inertia_kg_mm2):
    """Six virtual-work ports attach the casing's spatial inertia to the selected bolt group."""
    from dataclasses import replace
    from gso_workflows.cad_history import AttachedInertia
    from gso_workflows.rigid_carrier import RigidCarrier
    unit=motor_mount_moment(study,1.)
    axis=study.loads['lift'].face.normal
    origin=unit.origin_mm
    directions=tuple(replace(unit,force_n=d,moment_nmm=np.zeros(3)) for d in np.eye(3))
    rotations=tuple(replace(unit,force_n=np.zeros(3),moment_nmm=d) for d in np.eye(3))
    fixture=RigidCarrier(origin,origin+center_height_mm*axis,assembly_mass_kg,central_inertia_kg_mm2,
                         np.zeros(3),np.zeros(3))
    inertia=AttachedInertia(directions+rotations,fixture.spatial_mass())
    return unit,inertia


def braking_case(study,rpm,stop_s,rotating_inertia_kg_m2,assembly_mass_kg,center_height_mm,
                 central_inertia_kg_mm2,observation_s,step_s):
    """Smooth prescribed shaft stop with exactly the requested discrete angular impulse."""
    from gso_workflows.cad_history import HistoryCase,TimeExcitation
    if min(rpm,stop_s,rotating_inertia_kg_m2,step_s)<=0. or observation_s<stop_s:
        raise ValueError('Invalid motor stop conditions')
    unit,inertia=motor_hardware(study,assembly_mass_kg,center_height_mm,central_inertia_kg_mm2)
    knots=np.linspace(0.,stop_s,int(np.ceil(stop_s/step_s))+1)
    torque=np.sin(np.pi*knots/stop_s);torque[[0,-1]]=0.
    impulse=1000.*rotating_inertia_kg_m2*(rpm*2.*np.pi/60.)
    integral=float((np.diff(knots)*(torque[:-1]+torque[1:])*.5).sum())
    if integral<=0.:raise ValueError('Time step cannot resolve the braking pulse')
    torque*=impulse/integral
    load=TimeExcitation(unit,knots,torque)
    return HistoryCase(observation_s,step_s,(load,),inertias=(inertia,))
