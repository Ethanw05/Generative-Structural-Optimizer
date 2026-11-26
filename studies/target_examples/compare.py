"""Run equal-budget before/after verification on two deliberately inefficient original parts."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'build/sunbeam33/meshlib-runtime'), str(ROOT/'build/sunbeam33/deps')]
from gso_workflows.runtime_limits import configure_cpu_limit
configure_cpu_limit(int(os.environ.get('GSO_WORKERS', '2')))

import manifold3d as md
import numpy as np
import trimesh
from gso_workflows.cad_history import FoundationContact, HistoryCase
from gso_workflows.design_space import DesignSpace, Face
from gso_workflows.design_targets import YieldSafety, ResponseLimit, MinimizeResponse
from gso_workflows.rigid_carrier import RigidCarrier
from gso_workflows.shock_receivers import AccelerationReceiver
from gso_workflows.surface_objectives import ShapeTargets
from gso_workflows.volume_study import SurfaceLoad, VolumeStudy, generate


def mesh(solid):
    raw = solid.to_mesh64()
    return trimesh.Trimesh(raw.vert_properties[:,:3], raw.tri_verts, process=False)


def centered_box(width, depth, height, z):
    return md.Manifold.cube([width,depth,height]).translate([-width/2.,-depth/2.,z])


def necked_link(executable):
    # Wide end blocks waste material; an abrupt shoulder transfers load into the waist.
    solid = centered_box(3.4,2.4,1.4,0.) + centered_box(1.45,1.35,2.84,1.38) + centered_box(3.4,2.4,1.4,4.2)
    original = mesh(solid)
    base = Face.pick(original,[0.,0.,0.],[0.,0.,-1.])
    cap = Face.pick(original,[0.,0.,5.6],[0.,0.,1.])
    space = DesignSpace(centered_box(3.8,2.8,5.6,0.), (base.prism(-.25),cap.prism(-.25)), ())
    loads = {'tension': SurfaceLoad(cap,np.array([0.,0.,18.])),
             'oblique': SurfaceLoad(cap,np.array([1.4,.6,12.]))}
    study = VolumeStudy(space,(base,),loads,2100.,.33,solid.volume(),np.eye(3),np.zeros(3),executable)
    return study,original,ShapeTargets(limits=(YieldSafety(tuple(loads),30.,2.,'synthetic isotropic yield'),))


def impact_mount(executable):
    # A square plate supports a narrow, off-centre raised contact button.
    plate = centered_box(5.,4.,1.,0.)
    button = md.Manifold.cylinder(.72,.85,.85,32).translate([.55,0.,.98])
    solid = plate + button
    original = mesh(solid)
    fixed = tuple(Face.pick(original,[sign*2.5,0.,.5],[sign,0.,0.]) for sign in (-1.,1.))
    cap = Face.pick(original,[.55,0.,1.7],[0.,0.,1.])
    keep = tuple(face.prism(-.22) for face in (*fixed,cap))
    space = DesignSpace(centered_box(5.,4.6,2.2,-.5),keep,())
    press = SurfaceLoad(cap,np.array([0.,0.,-14.]))
    normal = SurfaceLoad(cap,np.array([0.,0.,-1.]))
    payload = RigidCarrier(np.zeros(3),np.zeros(3),.005,np.diag([.04,.04,.04]),
                           np.array([0.,0.,40.]),np.zeros(3))
    receiver = AccelerationReceiver('payload',fixed)
    crash = HistoryCase(.0023,.000012,(),contacts=(FoundationContact(normal,.001,14.),),
                        carrier=payload,receivers=(receiver,),relative_tolerance=.035,max_orders=22)
    study = VolumeStudy(space,fixed,{'press':press},2100.,.33,solid.volume(),np.eye(3),np.zeros(3),executable,
                        histories={'impact':crash},density_kg_m3=1050.)
    limits = (YieldSafety(('press','impact'),30.,2.,'synthetic isotropic yield'),
              ResponseLimit(('impact',),receiver.metric,7.))
    return study,original,ShapeTargets(limits=limits,extra_objectives=(MinimizeResponse(('impact',),receiver.metric),))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=('necked-link','impact-mount'))
    parser.add_argument('executable',type=Path)
    parser.add_argument('--sweeps',type=int,default=2)
    parser.add_argument('--continue-selected',action='store_true',help='Resume the previously exported part at full precision of its OBJ')
    parser.add_argument('--verify-edge-mm',type=float,action='append',default=[])
    args=parser.parse_args()
    executable=args.executable.resolve(strict=True)
    factory=necked_link if args.case=='necked-link' else impact_mount
    study,original,targets=factory(executable)
    out=ROOT/'runs/Target-studies'/args.case
    out.mkdir(parents=True,exist_ok=True)
    work=ROOT/'build/target-studies'/args.case
    work.mkdir(parents=True,exist_ok=True)
    original.export(out/'original.obj',digits=13)
    from dataclasses import replace
    targets=replace(targets,verification_edges_mm=tuple(args.verify_edge_mm))
    previous=trimesh.load_mesh(out/'generated.obj',process=False) if args.continue_selected else None
    selected=generate(study,.72,0,executable,work,out/'generated.obj',initial_surface=original,working_surface=previous,
                      targets=targets,surface_generations=args.sweeps,sequence_seed=193)
    report=json.loads((work/'generated-surface/selection.json').read_text())
    from gso_workflows.design_targets import assessments
    from gso_workflows.surface_objectives import EvaluatedSurface
    comparisons={}
    for title,key in (('original','initial'),('generated',report['selected'])):
        entry=next(v for v in report['evaluations'] if v['name']==key and 'responses' in v)
        rows=entry['responses']
        extra=entry.get('limit_evidence',{})
        evaluated=EvaluatedSurface(None,rows,[],key,limit_evidence=extra,verify_margin=True)
        checks=assessments(evaluated,targets.limits,require_all=True)
        responses=[[{k:v for k,v in response.items() if k in
                    ('scenario','n_tets','peak_MPa','compliance_Nmm','peak_contact_N','peak_elastic_displacement_mm',
                     'peak_acceleration_g@payload','peak_time_s','peak_at_window_end','step_s')} for response in level]
                    for level in rows]
        extra_rows={name:[{k:v for k,v in item.items() if k in ('scenario','verification_edge_mm','peak_MPa',
                        'compliance_Nmm','n_tets','peak_acceleration_g@payload')} for item in cases]
                    for name,cases in extra.items()}
        comparisons[title]=dict(geometry=entry['geometry'],checks=checks,responses=responses,extra_verification=extra_rows)
    result=dict(example=args.case,material=dict(young_MPa=2100.,poisson=.33,yield_MPa=30.,basis='synthetic isotropic'),
                analysis_pitch_mm=.72,verification=report['verification'],additional_edges_mm=args.verify_edge_mm,seed=193,
                original_volume_mm3=original.volume,selection=selected,
                comparisons=comparisons,lineage=[{k:event[k] for k in ('sweep','case','parent','child','status','proposal_edge_mm','direction') if k in event}
                for event in report['lineage']])
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(selected),flush=True)


if __name__=='__main__':
    main()
