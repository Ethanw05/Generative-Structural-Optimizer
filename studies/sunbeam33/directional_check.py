"""Matched horizontal-force, motor-axis torque and torsion tests on verified CAD analysis meshes."""
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
from filaments import ABS,PLA,PETG
from service_scenarios import directional_loads
from gso_workflows.analysis_mesh import registered_analysis_mesh
from gso_workflows.cad_wrench import SurfaceWrench
from gso_workflows.design_targets import YieldSafety
from gso_workflows.study_materials import material_portfolio
from gso_workflows.tetra_check import solve_static_cases
from gso_workflows.volume_study import linear_solver


def reversed_load(load):
    if isinstance(load,SurfaceWrench):
        return replace(load,force_n=-load.force_n,moment_nmm=-load.moment_nmm)
    return replace(load,force_n=-load.force_n)


def mesh_statistics(points,cells,tolerance):
    # The surface envelope and tetrahedron sizes are different physical quantities.
    longest_sq=np.zeros(len(cells))
    for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
        delta=points[cells[:,a]]-points[cells[:,b]]
        np.maximum(longest_sq,np.einsum('ij,ij->i',delta,delta),out=longest_sq)
    longest=np.sqrt(longest_sq)
    return dict(nodes=len(points),effective_surface_tolerance_mm=float(tolerance),
                tetrahedron_longest_edge_mm=dict(median=float(np.median(longest)),
                                                 p95=float(np.quantile(longest,.95)),maximum=float(longest.max())))


def measure(mesh,workspace,materials=(ABS,PLA,PETG),force_n=33.,torque_nmm=1000.,fixed_surface_edge_mm=None,factor_ram_mb=None):
    study=build_study(WORK/'installed-arm.obj')
    loads=directional_loads(study,force_n,torque_nmm)
    signed={mode+' '+symbol:action for mode,load in loads.items()
            for symbol,action in (('positive',load),('negative',reversed_load(load)))}
    study=material_portfolio(replace(study,loads=signed,mechanics_executable=ROOT/'build/Release/gso.exe'),materials)
    storage=(workspace,factor_ram_mb) if factor_ram_mb is not None else None
    solver,handle=linear_solver(WORK,factor_storage=storage)
    workspace.mkdir(parents=True,exist_ok=True)
    key,points,cells,metadata=registered_analysis_mesh(mesh,workspace,fixed_surface_edge_mm)
    if 'refinement' in metadata:
        print(workspace.name,'fixed surface refinement:',len(cells),'tetrahedra',flush=True)
    paths={label:workspace/label.replace(' / ','-') for label in study.loads}
    raw=solve_static_cases(points,cells,study,paths,solver,surface_tol=.36)
    report={}
    for spec in materials:
        modes={}
        for mode,load in loads.items():
            values={}
            for sign,symbol in ((1,'positive'),(-1,'negative')):
                label=f'{spec.name} / {mode} {symbol}'
                row=raw[label]
                np.testing.assert_allclose(row['force_N'],sign*load.force_n@study.world_rotation,atol=1.e-7)
                np.testing.assert_allclose(np.add(row['force_N'],row['reaction_N']),0.,atol=1.e-6)
                np.testing.assert_allclose(row['moment_residual_Nmm'],0.,atol=1.e-4)
                response=dict(peak_MPa=row['peak_MPa'],regional_peak_rms_MPa=row['regional_peak_rms_MPa'],
                              required_yield_MPa_for_factor_two=2.*row['peak_MPa'],peak_at_world_mm=row['peak_at_world_mm'],
                              force_world_N=row['force_N'],reaction_world_N=row['reaction_N'],
                              moment_balance_world_Nmm=row['moment_residual_Nmm'],compliance_Nmm=row['compliance_Nmm'])
                if isinstance(load,SurfaceWrench):
                    np.testing.assert_allclose(row['applied_moment_local_Nmm'],sign*load.moment_nmm,atol=1.e-6)
                    response.update(applied_moment_local_Nmm=row['applied_moment_local_Nmm'],
                                    signed_rotation_deg=sign*np.rad2deg(row['mean_rotation_rad']),
                                    stiffness_Nmm_per_rad=row['torsional_stiffness_Nmm_rad'])
                else:
                    response.update(signed_displacement_mm=sign*row['mean_lift_mm'],stiffness_N_per_mm=row['stiffness_N_mm'])
                if spec.tensile_yield_mpa is not None:
                    response['yield_check']=YieldSafety((label,),spec.tensile_yield_mpa,2.,spec.source).assessment(row)
                values[symbol]=response
            for scalar in ('peak_MPa','compliance_Nmm'):
                np.testing.assert_allclose(values['positive'][scalar],values['negative'][scalar],rtol=1.e-7)
            modes[mode]=values
            print(workspace.name,spec.name,mode,round(values['positive']['peak_MPa'],3),'MPa',flush=True)
        report[spec.name]=modes
    mesh_info=mesh_statistics(points,cells,metadata['effective_surface_tolerance_mm'])
    mesh_info.update(metadata)
    return dict(geometry=key,n_tets=len(cells),mesh=mesh_info,modes=report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--part',nargs=2,action='append',required=True,metavar=('LABEL','VERIFIED_MESH'))
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--force-n',type=float,default=33.)
    parser.add_argument('--torque-nmm',type=float,default=1000.)
    parser.add_argument('--fixed-surface-edge-mm',type=float,help='Refine the registered exterior with a regular-tetrahedron equivalent volume bound')
    parser.add_argument('--factor-ram-mb',type=int,help='Store numerical stiffness factors on disk with this oneMKL RAM budget')
    parser.add_argument('--material',choices=('ABS','PLA','PETG'),action='append',
                        help='Restrict a refinement study to the named materials')
    args=parser.parse_args()
    catalog={spec.name:spec for spec in (ABS,PLA,PETG)}
    materials=tuple(catalog[key] for key in dict.fromkeys(args.material or catalog))
    if min(args.force_n,args.torque_nmm)<=0. or not np.isfinite([args.force_n,args.torque_nmm]).all():
        parser.error('Supply finite positive test amplitudes')
    if args.fixed_surface_edge_mm is not None and not (np.isfinite(args.fixed_surface_edge_mm) and args.fixed_surface_edge_mm>0.):
        parser.error('Choose a finite positive refinement edge')
    if args.factor_ram_mb is not None and args.factor_ram_mb<=0:
        parser.error('Choose a positive factor memory budget in MB')
    parts={}
    for label,path in args.part:
        if label in parts or Path(label).name!=label:
            parser.error('Choose distinct part labels without path components')
        parts[label]=measure(Path(path),args.work/label,materials=materials,force_n=args.force_n,torque_nmm=args.torque_nmm,
                             fixed_surface_edge_mm=args.fixed_surface_edge_mm,factor_ram_mb=args.factor_ram_mb)
    record=dict(linear_elastic=True,force_N=args.force_n,torque_Nmm=args.torque_nmm,frame='arm-local CAD',
                force_application='distributed on motor pad',moment_application='pure couple on motor bolt bores',
                fixtures='same two hub attachment faces',parts=parts)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
