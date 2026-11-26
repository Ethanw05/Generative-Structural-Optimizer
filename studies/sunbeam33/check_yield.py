"""Screen saved service-case responses against material yield, with geometry identity checks."""
import argparse
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'build/sunbeam33/deps'),str(ROOT/'build/sunbeam33/meshlib-runtime')]
import numpy as np
import trimesh
from gso_workflows.analysis_mesh import digest
from gso_workflows.design_targets import YieldSafety


def screen(source, mesh_file, factor):
    with np.load(mesh_file) as data:
        model=trimesh.Trimesh(data['v'],data['f'],process=False)
    identity=digest(model)
    rows=[]
    for path in sorted(source.glob('*.json')):
        response=json.loads(path.read_text())
        spec=response.get('material_spec')
        if spec is None or 'geometry' not in response or not path.stem.startswith(spec['name']+'-'):
            continue
        if response['geometry']!=identity:
            raise ValueError(f'Geometry mismatch in {path}')
        label=path.stem.removeprefix(spec['name']+'-')
        strength=spec['tensile_yield_mpa']
        row=dict(material=spec['name'],case=label,n_tets=response['n_tets'],peak_MPa=response['peak_MPa'],
                 required_yield_MPa=factor*response['peak_MPa'],strength_MPa=strength)
        if strength is not None:
            check=YieldSafety((label,),strength,factor,spec['source']+' '+spec['assumptions']).assessment(response)
            row.update(factor=check['achieved'],met=check['utilization']<=1.,basis=check['strength_basis'])
        rows.append(row)
    return dict(geometry=identity,target=factor,analysis='single-mesh linear elastic screening',
                evaluated_cases=len(rows),rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('response_folder',type=Path)
    parser.add_argument('geometry',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--factor',type=float,default=2.)
    args=parser.parse_args()
    result=screen(args.response_folder,args.geometry,args.factor)
    if not result['rows']:
        raise ValueError('No material-tagged service responses were available')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
