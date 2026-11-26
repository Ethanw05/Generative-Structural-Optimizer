"""Compare an installed arm and a generated arm under the same lift and motor-first impact."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from gso_workflows.runtime_limits import configure_cpu_limit
configure_cpu_limit(int(os.environ.get('GSO_WORKERS','2')))
from geometry import WORK,build_study
import numpy as np
from filaments import PLA
from service_scenarios import service_portfolio
from gso_workflows.analysis_mesh import read_mesh
from gso_workflows.cad_history import export_history
from gso_workflows.design_targets import YieldSafety,ResponseLimit,assessments
from gso_workflows.reference_regions import ReferenceRegions
from gso_workflows.shock_receivers import AccelerationReceiver
from gso_workflows.study_materials import material_portfolio
from gso_workflows.surface_objectives import EvaluatedSurface
from gso_workflows.tetra_elasticity import assemble
from gso_workflows.tetra_check import solve
from gso_workflows.volume_study import linear_solver


def simulate(study,mesh,folder,solver,*,rebound_window=None):
    with np.load(mesh) as archive:
        identity=str(archive['source'])
        if int(archive['verified_boundary'])!=2:
            raise ValueError('Expected a verified conforming solid mesh')
    p,t=read_mesh(mesh,identity)
    folder.mkdir(parents=True,exist_ok=True)
    static=study.for_case('PLA / lift')
    stat=solve(p,t,replace(static,loads={'PLA / lift':static.loads['PLA / lift']}),folder/'lift',
               solver,surface_tol=.36,report_to_console=False)
    print(folder.name,'lift',stat['peak_MPa'],flush=True)
    dynamic=study.for_case('PLA / drop')
    system=assemble(p,t,dynamic,.36)
    dyn=drop_history(system,dynamic,folder/'drop',solver,rebound_window)
    print(folder.name,'drop',dyn['peak_MPa'],dyn['peak_acceleration_g@hub'],flush=True)
    return identity,[stat,dyn]


def drop_history(system,study,output,solver,rebound_window=None):
    from gso_workflows.cad_history import UnresolvedHistory
    case=study.histories['PLA / drop']
    if rebound_window is not None:
        initial,maximum=rebound_window
        if not 0.<initial<=maximum:
            raise ValueError('Rebound verification needs a positive initial and maximum observation time')
        case=replace(case,duration_s=initial)
    while True:
        row=export_history(system,study,case,output,solver,ReferenceRegions(),.36,label='PLA / drop')
        touched=[contact for contact in row['contact_histories'] if contact['first_contact_s'] is not None]
        separated=bool(touched) and all(contact['first_release_s'] is not None and contact['final_force_N']==0.
                                        for contact in touched)
        row['all_active_contacts_released_at_end']=separated
        if rebound_window is None or separated:
            return row
        if case.duration_s>=maximum:
            raise UnresolvedHistory('Motor contact persisted at the maximum rebound verification time')
        case=replace(case,duration_s=min(2.*case.duration_s,maximum))


def study_definition():
    study=service_portfolio(build_study(WORK/'installed-arm.obj'))
    receiver=AccelerationReceiver('hub',study.fixed_faces)
    drop=replace(study.histories['drop'],receivers=(receiver,))
    study=replace(study,loads={'lift':study.loads['lift']},histories={'drop':drop},mechanics_executable=ROOT/'build/Release/gso.exe')
    study=material_portfolio(study,(PLA,))
    return study


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('installed_mesh',type=Path)
    cli.add_argument('candidate_mesh',type=Path)
    cli.add_argument('output',type=Path)
    cli.add_argument('--work',type=Path,required=True)
    cli.add_argument('--rebound-window',type=float,nargs=2,metavar=('INITIAL_S','MAXIMUM_S'),
                     help='Extend observation until the motor has separated from the impact surface')
    args=cli.parse_args()
    study=study_definition()
    solver,handle=linear_solver(WORK)
    comparison={}
    for label,mesh in (('installed',args.installed_mesh),('candidate',args.candidate_mesh)):
        identity,rows=simulate(study,mesh,args.work/label,solver,rebound_window=args.rebound_window)
        if label=='installed':
            limits=(YieldSafety(study.scenarios(),PLA.tensile_yield_mpa,2.,PLA.source),
                    ResponseLimit(('PLA / drop',),'peak_acceleration_g@hub',rows[1]['peak_acceleration_g@hub']))
        results=[{k:r[k] for k in ('scenario','n_tets','peak_MPa','peak_contact_N','compliance_Nmm',
                                   'mean_lift_mm','peak_acceleration_g@hub','peak_rigid_rotation_rad','step_s',
                                   'contact_histories','peak_at_window_end','window_s','all_active_contacts_released_at_end') if k in r} for r in rows]
        comparison[label]=dict(geometry=identity,results=results,
                               checks=assessments(EvaluatedSurface(None,[rows],[],label),limits,require_all=True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(analysis='one matched conforming mesh per shape, linear elasticity',
        shock_limit_basis='installed arm peak hub acceleration under the same drop',
        material=PLA.assumptions,parts=comparison),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
