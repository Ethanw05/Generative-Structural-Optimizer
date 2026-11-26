"""Service assumptions for the recorded front-left-arm comparison, in local CAD coordinates."""
from dataclasses import replace
import numpy as np
from motor_load import braking_case,motor_mount_moment
from impact_cases import motor_first_impact


def directional_loads(study,force_n=33.,torque_nmm=1000.):
    lift=study.loads['lift']
    motor_torque=motor_mount_moment(study,torque_nmm)
    return {'span':replace(lift,force_n=np.array([force_n,0.,0.])),
            'lateral':replace(lift,force_n=np.array([0.,force_n,0.])),
            'torque':motor_torque,'span twist':replace(motor_torque,moment_nmm=np.array([torque_nmm,0.,0.]))}


def service_portfolio(study):
    brake=braking_case(study,15000.,.1,5.e-5,.09,15.,np.diag([29.,29.,50.]),.15,.002)
    brake=replace(brake,max_orders=20,relative_tolerance=.035,max_time_levels=11)
    mount=brake.loads[0].attachment
    mass=1.6
    inertia=mass*np.diag(np.square([100.,100.,135.]))
    center=np.mean([face.triangles.mean(axis=(0,1)) for face in study.fixed_faces],axis=0)
    histories={'brake':brake}
    for label,normal,speed,stiffness in (('drop',np.array([-.5,0.,-np.sqrt(.75)]),np.sqrt(2.*9810.*250.),25.),
                                         ('strike',np.array([-.8,.6,0.]),3000.,100.)):
        point=mount.origin_mm+np.array([0.,0.,18.])-16.75*normal
        impact=motor_first_impact(study,mount,brake.inertias[0],mass,center,inertia,point,normal,
                                  -speed*normal,stiffness,.014,.00025)
        histories[label]=replace(impact,max_orders=22,relative_tolerance=.04,max_time_levels=11)
    lift=study.loads['lift']
    loads={'lift':lift,**directional_loads(study)}
    if ((loads.keys()-{'lift'}) & study.loads.keys()) or histories.keys() & study.histories.keys():
        raise ValueError('Recorded and additional cases need distinct names')
    return replace(study,loads={**loads,**study.loads},histories={**histories,**study.histories},
                   density_kg_m3=1040. if study.density_kg_m3 is None else study.density_kg_m3)
