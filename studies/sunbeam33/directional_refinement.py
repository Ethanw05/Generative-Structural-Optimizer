"""Compare matched force and torque results across independently verified analysis meshes."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'build/sunbeam33/deps')]
import numpy as np
from filaments import ABS,PLA,PETG
from gso_workflows.design_targets import YieldSafety,assessments
from gso_workflows.surface_objectives import EvaluatedSurface


def stress_locations(workspaces,name,mode,parts,rows,study):
    result=[]
    for folder,part,expected in zip(workspaces,parts,rows,strict=True):
        field=json.loads((folder/name/('ABS-'+mode+' positive.json')).read_text())
        if field['n_tets']!=part['n_tets'] or field['scenario']!='ABS / '+mode+' positive':
            raise ValueError('Incompatible directional field or mesh')
        for key in ('peak_MPa','peak_at_world_mm','compliance_Nmm'):
            np.testing.assert_allclose(field[key],expected[key],rtol=1.e-12,atol=1.e-12)
        point=(np.asarray(field['peak_at_world_mm'])-study.world_origin)@study.world_rotation.T
        result.append(dict(peak_at_world_mm=field['peak_at_world_mm'],p99_by_volume_MPa=field['p99_MPa'],
                           near_load=field['loaded_face_neighborhood'],
                           peak_distance_to_fixture_perimeter_mm=min(f.perimeter_distance(point) for f in study.fixed_faces)))
    return result


def compare(records,workspaces=(),study=None):
    reference=records[0]
    cases={}
    for other in records[1:]:
        if any(other[k]!=reference[k] for k in ('force_N','torque_Nmm','frame','fixtures','moment_application','force_application')):
            raise ValueError('Compare identical load definitions and supports')
        if other['parts'].keys()!=reference['parts'].keys():
            raise ValueError('Every refinement needs the same set of geometries')
    for name,base in reference['parts'].items():
        parts=[entry['parts'][name] for entry in records]
        if any(part['geometry']!=base['geometry'] for part in parts):
            raise ValueError('A refinement refers to a different CAD geometry')
        results={}
        for mode in base['modes']['ABS']:
            axial=base['modes']['ABS'][mode]['positive']
            motion='signed_rotation_deg' if 'signed_rotation_deg' in axial else 'signed_displacement_mm'
            rows=[part['modes']['ABS'][mode]['positive'] for part in parts]
            sample=EvaluatedSurface(None,[[dict(row,scenario=mode)] for row in rows],[],name,verify_margin=True)
            materials={}
            for material in (PLA,PETG):
                if material.poisson!=ABS.poisson:
                    raise ValueError('Elastic stress transfer requires a common Poisson ratio')
                # At fixed Poisson ratio, K scales with E and force-driven stress is invariant to E.
                # Direct solves on the starting mesh must independently corroborate that relation.
                for sign in ('positive','negative'):
                    polymer=base['modes'][material.name][mode][sign]['peak_MPa']
                    np.testing.assert_allclose(polymer,base['modes']['ABS'][mode][sign]['peak_MPa'],rtol=1.e-7)
                limits=(YieldSafety((mode,),material.tensile_yield_mpa,2.,material.source),)
                materials[material.name]=assessments(sample,limits,require_all=True)[0]
            peaks=[r['peak_MPa'] for r in rows]
            movement=[r[motion] for r in rows]
            results[mode]=dict(peak_MPa=peaks,motion_metric=motion,movement=movement,
                               peak_change_percent=100.*(peaks[-1]/peaks[0]-1.),
                               motion_change_percent=100.*(movement[-1]/movement[0]-1.),
                               required_ABS_yield_MPa_for_factor_two=2.*max(peaks),yield_screen=materials)
            if workspaces:
                results[mode]['stress_location']=stress_locations(workspaces,name,mode,parts,rows,study)
        cases[name]=dict(geometry=base['geometry'],tetrahedra=[part['n_tets'] for part in parts],
                         meshes=[part['mesh'] for part in parts],modes=results)
    return dict(force_N=reference['force_N'],torque_Nmm=reference['torque_Nmm'],parts=cases,
                stress_transfer='homogeneous linear elasticity, equal Poisson ratios, force/moment controlled; directly cross-checked on first mesh')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    cli.add_argument('levels',nargs='+',type=Path)
    cli.add_argument('--fields',type=Path,action='append',default=[],help='Field directory per mesh level to locate the peaks relative to CAD fixture edges')
    args=cli.parse_args()
    if len(args.levels)<2:
        cli.error('Supply at least two registered mesh levels')
    study=None
    if args.fields:
        if len(args.fields)!=len(args.levels):
            cli.error('Provide one field directory per mesh level')
        from geometry import WORK,build_study
        study=build_study(WORK/'installed-arm.obj')
    report=compare([json.loads(path.read_text()) for path in args.levels],args.fields,study)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
