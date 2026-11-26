"""Recover lift stress on two verified conforming meshes with identical fixtures and traction."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from gso_workflows.runtime_limits import configure_cpu_limit
configure_cpu_limit(int(os.environ.get('GSO_WORKERS','2')))
from geometry import WORK,build_study
from figures import make_figures
from gso_workflows.analysis_mesh import read_mesh
from gso_workflows.tetra_check import solve
from gso_workflows.volume_study import linear_solver
import numpy as np
import trimesh


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_geometry',type=Path)
    parser.add_argument('candidate_mesh',type=Path)
    parser.add_argument('installed_mesh',type=Path)
    parser.add_argument('work',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--installed-geometry',type=Path,default=WORK/'native-arm.npz')
    args=parser.parse_args()
    study=build_study(WORK/'installed-arm.obj')
    solver,handle=linear_solver(WORK)
    args.work.mkdir(parents=True,exist_ok=True)
    args.output.mkdir(parents=True,exist_ok=True)
    solids={'candidate':args.candidate_geometry,'installed':args.installed_geometry}
    responses={}
    for name,source in (('installed',args.installed_mesh),('candidate',args.candidate_mesh)):
        with np.load(source) as data:
            identity=str(data['source'])
            if int(data['verified_boundary'])!=2:
                raise ValueError('A verified conforming analysis mesh is required')
        from gso_workflows.analysis_mesh import digest
        with np.load(solids[name]) as data:
            part=trimesh.Trimesh(data['v'],data['f'],process=False)
        if digest(part)!=identity:
            raise ValueError(f'Geometry and stress mesh differ for {name}')
        p,t=read_mesh(source,identity)
        path=args.work/(name+'-lift')
        row=solve(p,t,study,path,solver,surface_tol=.36,report_to_console=False)
        responses[name]=dict(geometry=identity,mean_pad_lift_mm=row['mean_lift_mm'],peak_MPa=row['peak_MPa'],
                             compliance_Nmm=row['compliance_Nmm'],pad_peak_MPa=row['loaded_face_neighborhood']['peak_MPa'],
                             required_yield_MPa_for_factor_two=2.*row['peak_MPa'],n_tets=row['n_tets'],
                             force_N=row['force_N'],reaction_N=row['reaction_N'])
        print(name, responses[name],flush=True)
    (args.output/'Matched-lift.json').write_text(json.dumps(responses,indent=2,allow_nan=False))
    make_figures(solids['candidate'],solids['installed'],args.work/'candidate-lift.npz',args.work/'installed-lift.npz',
                 args.output,mesh_caption='Matched mesh')


if __name__=='__main__':
    main()
