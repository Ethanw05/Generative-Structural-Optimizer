"""Recheck static safety on a smaller independent surface mesh of the published geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare import ROOT,necked_link,impact_mount
import trimesh
from gso_workflows.analysis_mesh import digest,volume_mesh
from gso_workflows.design_targets import YieldSafety
from gso_workflows.tetra_check import solve
from gso_workflows.volume_study import linear_solver
from dataclasses import replace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=('necked-link','impact-mount'))
    parser.add_argument('executable',type=Path)
    parser.add_argument('--edge-mm',type=float,default=.26)
    args=parser.parse_args()
    factory=necked_link if args.case=='necked-link' else impact_mount
    study,stock,targets=factory(args.executable.resolve())
    work=ROOT/'build/target-studies'/args.case/'holdout'
    work.mkdir(parents=True,exist_ok=True)
    out=ROOT/'runs/Target-studies'/args.case
    solver,handle=linear_solver(work)
    rows={}
    identities={}
    receipts={}
    compared=json.loads((out/'comparison.json').read_text())['comparisons']
    with np.load(work.parent/'generated.npz') as data:
        selected=trimesh.Trimesh(data['v'],data['f'],process=False)
    for name,source in (('original',stock),('generated',selected)):
        obj=out/(name+'.obj')
        mesh=trimesh.load_mesh(obj,process=False,maintain_order=True)
        design_id=digest(source)
        if design_id!=compared[name]['geometry']:
            raise ValueError(f'Study result no longer describes the published {name} geometry')
        np.testing.assert_array_equal(mesh.faces,source.faces)
        tolerance=5.e-14+np.finfo(float).eps*np.max(np.abs(source.vertices))
        np.testing.assert_allclose(mesh.vertices,source.vertices,atol=tolerance,rtol=0.)
        receipts[name]=dict(selected_geometry=design_id,obj_lf_sha256=hashlib.sha256(obj.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
                            max_coordinate_roundoff_mm=float(np.max(np.abs(mesh.vertices-source.vertices))))
        identities[name]=digest(mesh)
        p,t=volume_mesh(mesh,work,args.edge_mm,args.edge_mm/20.)
        results={}
        for label,load in study.loads.items():
            single=replace(study,loads={label:load})
            row=solve(p,t,single,work/(name+'-'+label),solver,surface_tol=.216,report_to_console=False)
            limit=next(item for item in targets.limits if isinstance(item,YieldSafety) and label in item.cases)
            results[label]=dict(tetrahedra=len(t),peak_MPa=row['peak_MPa'],compliance_Nmm=row['compliance_Nmm'],
                peak_at_world_mm=row['peak_at_world_mm'],safety=limit.assessment(row))
            print(name,label,results[label]['peak_MPa'],flush=True)
        rows[name]=results
    data=dict(geometries=identities,export_verification=receipts,independent_surface_edge_mm=args.edge_mm,surface_envelope_mm=args.edge_mm/20.,static_cases=rows)
    (out/'independent-refinement.json').write_text(json.dumps(data,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
