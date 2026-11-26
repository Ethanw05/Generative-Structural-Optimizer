"""Apply identical native face selections, ABS properties, and resultant to either exported surface."""
import argparse
import hashlib
import json
from pathlib import Path
from geometry import WORK,build_study
import numpy as np
from gso_workflows.volume_study import linear_solver
from gso_workflows.tetra_check import solve
import tetgen
import trimesh


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('obj',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('volume',type=float)
    parser.add_argument('--tetra',type=Path)
    args=parser.parse_args()
    study=build_study()
    cache=WORK/('tet-'+hashlib.sha256(args.obj.read_bytes()).hexdigest()[:16]+f'-{args.volume}.npz')
    if args.tetra or cache.exists():
        with np.load(args.tetra or cache) as d:p,t=d['p'],d['tet']
    else:
        surf=trimesh.load_mesh(args.obj,process=False,maintain_order=True)
        generator=tetgen.TetGen(surf.vertices,surf.faces.astype(np.int32))
        p,t,*_=generator.tetrahedralize(switches=f'pq1.6a{args.volume}Q',steinerleft=-1)
        np.savez_compressed(cache,p=p,tet=t)
    solver,handle=linear_solver(WORK)
    summary=solve(p,t,study,args.output,solver)
    summary['source_obj']=str(args.obj.resolve())
    summary['source_obj_sha256']=hashlib.sha256(args.obj.read_bytes()).hexdigest()
    summary['tetra_file']=str((args.tetra or cache).resolve())
    args.output.with_suffix('.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
