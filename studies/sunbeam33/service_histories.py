"""Measure the real braking transient and a lateral contact strike on the registered arm geometry."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from receiver_comparison import WORK,ROOT,PLA,build_study,contact_history,service_portfolio
from service_scenarios import (BRAKING_RPM,BRAKING_STOP_S,ROTOR_INERTIA_KG_M2,CARRIER_MASS_KG,
                               STRIKE_SPEED_MM_S,STRIKE_STIFFNESS_N_MM)
from gso_workflows.analysis_mesh import read_mesh
from gso_workflows.cad_history import export_history
from gso_workflows.reference_regions import ReferenceRegions
from gso_workflows.shock_receivers import AccelerationReceiver
from gso_workflows.study_materials import material_portfolio
from gso_workflows.tetra_elasticity import assemble
from gso_workflows.volume_study import linear_solver
import numpy as np


def measure(mesh,workspace,strike_window,names=('brake','strike')):
    with np.load(mesh) as archive:
        if int(archive['verified_boundary'])!=2:
            raise ValueError('Expected a verified conforming mesh')
        identity=str(archive['source'])
    p,t=read_mesh(mesh,identity)
    study=service_portfolio(build_study())
    receiver=AccelerationReceiver('hub',study.fixed_faces)
    histories={'brake':study.histories['brake'],'strike':replace(study.histories['strike'],receivers=(receiver,))}
    study=material_portfolio(replace(study,loads={},histories=histories,mechanics_executable=ROOT/'build/Release/gso.exe'),(PLA,))
    system=assemble(p,t,study.for_case('PLA / brake'),.36)
    solver,handle=linear_solver(WORK)
    workspace.mkdir(parents=True,exist_ok=True)
    result={}
    for name in names:
        label='PLA / '+name
        active=study.for_case(label)
        path=workspace/name
        if name=='strike':
            row=contact_history(system,active,path,solver,strike_window,label)
        else:
            row=export_history(system,active,active.histories[label],path,solver,ReferenceRegions(),.36,label=label)
        result[name]={key:row[key] for key in ('scenario','n_tets','peak_MPa','regional_peak_rms_MPa',
                      'peak_elastic_displacement_mm','peak_energy_Nmm','peak_contact_N','peak_acceleration_g@hub',
                      'step_s','window_s','peak_at_window_end','contact_histories','all_active_contacts_released_at_end') if key in row}
        print(workspace.name,name,round(row['peak_MPa'],3),'MPa',flush=True)
    return dict(geometry=identity,results=result)


def align_strike_windows(results,meshes,work,maximum):
    from math import isclose
    while True:
        target=max(part['results']['strike']['window_s'] for part in results.values())
        if not 0.<target<=maximum:
            raise ValueError('Existing contact window lies outside the requested observation bound')
        shorter=[name for name,part in results.items()
                 if not isclose(part['results']['strike']['window_s'],target,rel_tol=1.e-12)]
        if not shorter:
            return
        for name in shorter:
            longer=measure(meshes[name],work/name,(target,maximum),names=('strike',))
            if longer['geometry']!=results[name]['geometry'] or longer['results']['strike']['window_s']<target:
                raise ValueError('The extended strike evidence does not match the part or window')
            results[name]['results']['strike']=longer['results']['strike']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--part',nargs=2,action='append',required=True,metavar=('LABEL','VERIFIED_MESH'))
    # The 28 ms exploratory window ended during renewed contact on the reference part.
    parser.add_argument('--strike-window',type=float,nargs=2,default=(.056,.112),metavar=('INITIAL_S','MAXIMUM_S'),
                        help='Observe repeat contact until the last recorded contact is open')
    parser.add_argument('--match-existing',action='store_true',help='Recheck shorter windows in this completed comparison')
    args=parser.parse_args()
    if not np.isfinite(args.strike_window).all() or not 0.<args.strike_window[0]<=args.strike_window[1]:
        parser.error('Choose finite positive observation windows in increasing order')
    meshes={}
    for label,mesh in args.part:
        if label in meshes or Path(label).name!=label:
            parser.error('Choose unique part labels without path components')
        meshes[label]=Path(mesh)
    inputs=dict(material='PLA',brake=dict(initial_rpm=BRAKING_RPM,stop_s=BRAKING_STOP_S,rotating_inertia_kg_m2=ROTOR_INERTIA_KG_M2),
                strike=dict(speed_m_s=STRIKE_SPEED_MM_S/1000.,contact_stiffness_N_per_mm=STRIKE_STIFFNESS_N_MM,
                            carrier_mass_kg=CARRIER_MASS_KG))
    if args.match_existing:
        record=json.loads(args.output.read_text())
        if any(record[key]!=value for key,value in inputs.items()):
            parser.error('Different physical inputs cannot share a comparison')
        results=record['parts']
        if results.keys()!=meshes.keys():
            parser.error('The supplied meshes must identify every part in the existing comparison')
        for name,path in meshes.items():
            with np.load(path) as evidence:
                if int(evidence['verified_boundary'])!=2 or str(evidence['source'])!=results[name]['geometry']:
                    parser.error('The supplied analysis mesh does not match the recorded geometry')
    else:
        results={name:measure(path,args.work/name,tuple(args.strike_window)) for name,path in meshes.items()}
    align_strike_windows(results,meshes,args.work,args.strike_window[1])
    meta=dict(**inputs,parts=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(meta,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
