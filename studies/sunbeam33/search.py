"""Run the recorded 33 N ABS study using native geometry and the shared design workflow."""
import argparse
import subprocess
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('part',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('pitch_mm',type=float,nargs='?',default=2.)
    parser.add_argument('iterations',type=int,nargs='?',default=72)
    parser.add_argument('--assembly',type=Path)
    start=parser.add_mutually_exclusive_group()
    start.add_argument('--refine-generated-obj',type=Path)
    start.add_argument('--resume-feedback',type=Path,metavar='RUN_DIRECTORY',
                       help='Start a new pass on the latest evaluated working mesh with the current load cases')
    parser.add_argument('--surface-generations',type=int,default=2,help='Complete shuffled passes through the load cases')
    parser.add_argument('--service-cases',action='store_true',
                        help='Use the recorded three 33 N directions, bolt torque, motor braking, drop and strike conditions')
    parser.add_argument('--filament',choices=('ABS','PLA','PETG'),action='append',
                        help='Apply every case to this reference filament on the same evolving arm; repeat for a material portfolio')
    parser.add_argument('--finite-strain',action='append',default=[],metavar='CASE',
                        help='Use finite-deformation elasticity for this named force or history')
    parser.add_argument('--plastic-law',action='append',nargs=3,default=[],metavar=('CASE','YIELD_MPA','H_MPA'),
                        help='Use finite-strain J2 with initial Kirchhoff yield stress and isotropic hardening modulus')
    parser.add_argument('--sequence-seed',type=int,help='Replay the order of the load cases')
    parser.add_argument('--target',choices=('stiffness','maximum-stress'),action='append')
    parser.add_argument('--cpu-limit',type=int,help='Limit this run and its workers to this many CPUs')
    parser.add_argument('--force-local-n',nargs=3,type=float,action='append',metavar=('FX','FY','FZ'),
                        help='Add a named force case on the selected load face in local CAD axes (N)')
    parser.add_argument('--bolt-moment-nmm',type=float,action='append',
                        help='Add a torque about the motor axis on the native bolt bore walls (N mm)')
    parser.add_argument('--brake-rpm',type=float)
    parser.add_argument('--stop-s',type=float)
    parser.add_argument('--spinning-inertia-kg-m2',type=float)
    parser.add_argument('--motor-assembly-kg',type=float)
    parser.add_argument('--motor-center-height-mm',type=float)
    parser.add_argument('--central-inertia-kg-mm2',type=float,nargs=3,metavar=('IXX','IYY','IZZ'))
    parser.add_argument('--history-window-s',type=float)
    parser.add_argument('--history-step-s',type=float)
    parser.add_argument('--density-kg-m3',type=float)
    parser.add_argument('--impact',type=float,nargs=12,action='append',
                        metavar=('NX','NY','NZ','VX','VY','VZ','X','Y','Z','K','WINDOW','DT'),
                        help='Motor casing impact. Unit ground normal, initial velocity (mm/s), casing contact point in local CAD axes (mm), normal stiffness (N/mm), window and step (s)')
    parser.add_argument('--carrier-mass-kg',type=float)
    parser.add_argument('--carrier-center-mm',type=float,nargs=3)
    parser.add_argument('--carrier-inertia-kg-mm2',type=float,nargs=3,metavar=('IXX','IYY','IZZ'))
    args=parser.parse_args()
    if args.cpu_limit is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
        from gso_workflows.runtime_limits import configure_cpu_limit
        configure_cpu_limit(args.cpu_limit)
    from geometry import ROOT,WORK,build_study
    from gso_workflows.volume_study import generate
    from gso_workflows.surface_search import ShapeTargets
    names=args.target or ('stiffness','maximum-stress')
    targets=ShapeTargets('stiffness' in names,'maximum-stress' in names)
    executable=ROOT/'build/Release/gso.exe'
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(dir=WORK) as directory:
        source=Path(directory)/'part.obj'
        subprocess.run([executable,'import',args.part.resolve(),source],check=True)
        scene=None
        if args.assembly:
            scene=Path(directory)/'assembly.obj'
            subprocess.run([executable,'import',args.assembly.resolve(),scene],check=True)
        study=build_study(source,scene)
    from dataclasses import replace
    import numpy as np
    from motor_load import motor_mount_moment,braking_case,motor_hardware
    loads=dict(study.loads)
    for i,force in enumerate(args.force_local_n or ()):
        loads[f'force {i+1}']=replace(study.loads['lift'],force_n=np.asarray(force))
    for i,moment in enumerate(args.bolt_moment_nmm or ()):
        loads[f'bolt group torque {i+1}']=motor_mount_moment(study,moment)
    histories={}
    brake=(args.brake_rpm,args.stop_s,args.spinning_inertia_kg_m2,args.motor_assembly_kg,
           args.motor_center_height_mm,args.central_inertia_kg_mm2,args.history_window_s,args.history_step_s)
    if any(x is not None for x in (*brake[:3],*brake[6:])):
        if any(x is None for x in brake) or args.density_kg_m3 is None:
            parser.error('Motor braking needs RPM, stop time, rotating inertia, attached mass, center height, central inertia, observation window, time step and material density')
        histories['motor stop']=braking_case(study,*brake[:5],np.diag(brake[5]),*brake[6:])
    if args.impact:
        from impact_cases import motor_first_impact
        hardware=(args.motor_assembly_kg,args.motor_center_height_mm,args.central_inertia_kg_mm2)
        carrier=(args.carrier_mass_kg,args.carrier_center_mm,args.carrier_inertia_kg_mm2)
        if any(x is None for x in (*hardware,*carrier,args.density_kg_m3)):
            parser.error('Impact cases require motor mass, center height, central inertia, carrier mass, center, principal inertia and material density')
        mount,inertia=motor_hardware(study,hardware[0],hardware[1],np.diag(hardware[2]))
        for i,values in enumerate(args.impact):
            normal,velocity,point=np.asarray(values[:9]).reshape(3,3)
            histories[f'impact {i+1}']=motor_first_impact(
                study,mount,inertia,carrier[0],carrier[1],np.diag(carrier[2]),point,normal,velocity,*values[9:])
    study=replace(study,loads=loads,histories=histories,density_kg_m3=args.density_kg_m3,mechanics_executable=executable)
    if args.service_cases:
        from service_scenarios import service_portfolio
        study=service_portfolio(study)
    from gso_workflows.finite_solid import FiniteStrain
    laws={name:FiniteStrain() for name in args.finite_strain}
    for name,yield_mpa,hardening_mpa in args.plastic_law:
        if name in laws and laws[name].initial_yield_mpa is not None:parser.error('Duplicate plastic law for '+name)
        try:laws[name]=FiniteStrain(float(yield_mpa),float(hardening_mpa))
        except ValueError as error:parser.error(str(error))
    base_names=set(study.scenarios())
    study=replace(study,solid_laws={k:v for k,v in laws.items() if k in base_names})
    if args.filament:
        from filaments import ABS,PLA,PETG
        from gso_workflows.study_materials import material_portfolio
        catalog={m.name:m for m in (ABS,PLA,PETG)}
        study=material_portfolio(study,[catalog[name] for name in args.filament])
    qualified={k:v for k,v in laws.items() if k not in base_names}
    if qualified.keys()-set(study.scenarios()):parser.error('Unknown nonlinear scenario. Available: '+', '.join(study.scenarios()))
    study=replace(study,solid_laws={**study.solid_laws,**qualified})
    baseline=working=None
    if args.resume_feedback:
        from gso_workflows.scenario_feedback import resume_pair
        baseline,working=resume_pair(args.resume_feedback)
    elif args.refine_generated_obj:
        import trimesh
        baseline = trimesh.load_mesh(args.refine_generated_obj.resolve(), process=True)
    print(generate(study,args.pitch_mm,args.iterations,executable,WORK,args.output.resolve(),targets=targets,
                   initial_surface=baseline,working_surface=working,surface_generations=args.surface_generations,sequence_seed=args.sequence_seed),flush=True)


if __name__=='__main__':
    main()
