"""Recheck static safety on a smaller independent surface mesh of the published geometry."""
import argparse
import json
from pathlib import Path
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
    for name,mesh in (('original',stock),('generated',trimesh.load_mesh(out/'generated.obj',process=False))):
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
    data=dict(geometries=identities,independent_surface_edge_mm=args.edge_mm,surface_envelope_mm=args.edge_mm/20.,static_cases=rows)
    (out/'independent-refinement.json').write_text(json.dumps(data,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
