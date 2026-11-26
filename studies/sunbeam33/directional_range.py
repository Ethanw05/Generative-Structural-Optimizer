"""Find and independently solve the worst horizontal force and bolt-group moment orientations."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from directional_check import ROOT,WORK,PLA,build_study,directional_loads
from gso_workflows.analysis_mesh import registered_analysis_mesh
from gso_workflows.cad_wrench import SurfaceWrench
from gso_workflows.elastic_load_plane import equivalent_stress_squared,compliance_plane
from gso_workflows.tetra_check import solve_static_cases
from gso_workflows.volume_study import linear_solver
import numpy as np


def combination(first,second,angle):
    c,s=np.cos(angle),np.sin(angle)
    force=c*first.force_n+s*second.force_n
    if isinstance(first,SurfaceWrench):
        if first.faces is not second.faces or not np.array_equal(first.origin_mm,second.origin_mm):
            raise ValueError('The recorded bolt group and moment origin must match')
        return replace(first,force_n=force,moment_nmm=c*first.moment_nmm+s*second.moment_nmm)
    if first.face is not second.face:
        raise ValueError('The recorded force basis must occupy the same selected pad')
    return replace(first,force_n=force)


def measure(mesh_path,work,force,torque,fixed_surface_edge_mm=None,factor_ram_mb=None):
    ident,points,cells,metadata=registered_analysis_mesh(mesh_path,work,fixed_surface_edge_mm)
    study=PLA.apply(build_study())
    loads=directional_loads(study,force,torque)
    study=replace(study,loads=loads,mechanics_executable=ROOT/'build/Release/gso.exe')
    work.mkdir(parents=True,exist_ok=True)
    storage=(work,factor_ram_mb) if factor_ram_mb is not None else None
    solver,handle=linear_solver(WORK,factor_storage=storage)
    fields={}
    def record(label,stress,force,motion):
        fields[label]=(stress,force,motion)
    paths={label:work/label for label in loads}
    solve_static_cases(points,cells,study,paths,solver,surface_tol=.36,observe_linear=record)

    checks={}
    result={}
    for family,a,b in (('horizontal','span','lateral'),('torque plane','torque','span twist')):
        sa,fa,ua=fields[a]
        sb,fb,ub=fields[b]
        stress=equivalent_stress_squared(sa,sb)
        strain_work=compliance_plane(fa,ua,fb,ub)
        maxima,angles=stress.maximum()
        index=int(np.argmax(maxima))
        peak=float(np.sqrt(maxima[index]))
        stress_angle=float(angles[index])
        max_work,work_angle=strain_work.maximum()
        result[family]=dict(first_axis=a,second_axis=b,stress_angle_deg=float(np.rad2deg(stress_angle)),
                            governing_element=index,peak_at_world_mm=points[cells[index]].mean(axis=0).tolist(),
                            peak_MPa=peak,required_yield_MPa_for_factor_two=2.*peak,
                            compliance_angle_deg=float(np.rad2deg(work_angle)),maximum_compliance_Nmm=float(max_work),
                            peak_curve_MPa=[float(np.sqrt(stress.at(np.deg2rad(angle)).max())) for angle in range(181)])
        checks[family+' stress']=combination(loads[a],loads[b],stress_angle)
        checks[family+' compliance']=combination(loads[a],loads[b],work_angle)
    fields.clear()
    del sa,sb,fa,fb,ua,ub
    direct=solve_static_cases(points,cells,replace(study,loads=checks),{label:work/label for label in checks},
                              solver,surface_tol=.36)
    for family,row in result.items():
        stress=direct[family+' stress']
        flexible=direct[family+' compliance']
        np.testing.assert_allclose(stress['peak_MPa'],row['peak_MPa'],rtol=1.e-7)
        np.testing.assert_allclose(flexible['compliance_Nmm'],row['maximum_compliance_Nmm'],rtol=1.e-7)
        for label in (family+' stress',family+' compliance'):
            solution=direct[label]
            np.testing.assert_allclose(np.add(solution['force_N'],solution['reaction_N']),0.,atol=1.e-6)
            np.testing.assert_allclose(solution['moment_residual_Nmm'],0.,atol=1.e-4)
            if isinstance(checks[label],SurfaceWrench):
                np.testing.assert_allclose(solution['applied_moment_local_Nmm'],checks[label].moment_nmm,atol=1.e-6)
        row.update(direct_peak_MPa=stress['peak_MPa'],direct_maximum_compliance_Nmm=flexible['compliance_Nmm'],
                   regional_peak_rms_at_stress_angle_MPa=stress['regional_peak_rms_MPa'],
                   yield_factor_PLA=PLA.tensile_yield_mpa/row['peak_MPa'])
        if family=='horizontal':
            row['maximum_directional_displacement_mm']=flexible['mean_lift_mm']
        else:
            row['maximum_directional_rotation_deg']=float(np.rad2deg(flexible['mean_rotation_rad']))
        print(work.name,family,round(row['stress_angle_deg'],2),'degrees',round(row['peak_MPa'],3),'MPa',flush=True)
    return dict(geometry=ident,n_tets=len(cells),mesh=metadata,planes=result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--part',nargs=2,action='append',required=True,metavar=('LABEL','VERIFIED_MESH'))
    parser.add_argument('--force-n',type=float,default=33.)
    parser.add_argument('--torque-nmm',type=float,default=1000.)
    parser.add_argument('--fixed-surface-edge-mm',type=float,help='Refine the registered exterior using a regular-tetrahedron equivalent volume bound')
    parser.add_argument('--factor-ram-mb',type=int,help='oneMKL numerical factor memory budget in MB')
    args=parser.parse_args()
    if args.factor_ram_mb is not None and args.factor_ram_mb<=0:
        parser.error('Choose a positive factor memory budget in MB')
    if not np.isfinite([args.force_n,args.torque_nmm]).all() or min(args.force_n,args.torque_nmm)<=0.:
        parser.error('Choose positive finite load magnitudes')
    parts={}
    for name,mesh in args.part:
        if name in parts or Path(name).name!=name:
            parser.error('Choose distinct labels without path components')
        parts[name]=measure(Path(mesh),args.work/name,args.force_n,args.torque_nmm,args.fixed_surface_edge_mm,args.factor_ram_mb)
    out=dict(material='PLA',force_N=args.force_n,torque_Nmm=args.torque_nmm,angle_frame='first axis toward second axis',
             range_deg=[0.,360.],linear_reversal_period_deg=180.,plot_spacing_deg=1.,
             method='elementwise exact quadratic stress envelope with independent direct solves at critical orientations',
             parts=parts)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
