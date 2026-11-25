"""Motor-first impact study inputs. Assembly properties and contact law are supplied by the caller."""
from dataclasses import replace
import numpy as np
from gso_workflows.cad_history import HistoryCase,FoundationContact
from gso_workflows.rigid_carrier import RigidCarrier


def motor_first_impact(study,mount,hardware_inertia,carrier_mass_kg,carrier_center_local_mm,
                      carrier_central_inertia_kg_mm2,contact_local_mm,normal_local,velocity_local_mm_s,
                      stiffness_n_mm,duration_s,step_s):
    normal=np.asarray(normal_local,dtype=float)
    if abs(np.linalg.norm(normal)-1.)>1.e-9:
        raise ValueError('Impact normal must have unit length')
    rotation=study.world_rotation
    center=np.asarray(carrier_center_local_mm)@rotation+study.world_origin
    carrier=RigidCarrier(center,center,carrier_mass_kg,rotation.T@carrier_central_inertia_kg_mm2@rotation,
                         np.asarray(velocity_local_mm_s)@rotation,np.zeros(3))
    # Virtual work at the casing contact is transmitted through the declared bolt group.
    # The contact origin is expressed in the same local coordinates as the load selection.
    port=replace(mount,force_n=normal,moment_nmm=np.cross(np.asarray(contact_local_mm)-mount.origin_mm,normal))
    contact=FoundationContact(port,0.,stiffness_n_mm)
    return HistoryCase(duration_s,step_s,(),contacts=(contact,),inertias=(hardware_inertia,),carrier=carrier)
