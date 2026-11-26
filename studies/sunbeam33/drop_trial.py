"""Propose a solid from the governing lift/drop field and measure it before the full service comparison."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from receiver_comparison import ROOT,WORK,study_definition,simulate,linear_solver,PLA
import trimesh
from gso_workflows.analysis_mesh import digest,volume_mesh
from gso_workflows.boundary_shape import propose
from gso_workflows.design_targets import MinimizeResponse,ResponseLimit,YieldSafety,assessments
from gso_workflows.mesh_quality import InadmissibleBoundary,validate_single_part
from gso_workflows.scenario_feedback import mutation_weights
from gso_workflows.surface_objectives import EvaluatedSurface,ShapeTargets,pareto,compromise_scores,objective_scales
from gso_workflows.volume_study import publish_surface


def measured_proposal(study,parent,field_path,label,weights,work,spacing,step,resume):
    source=dict(parent=digest(parent),field_sha256=hashlib.sha256(field_path.read_bytes()).hexdigest(),
                case=label,spacing_mm=spacing,initial_step_mm=step,weights=list(weights),
                volume_limit_mm3=study.volume_limit_mm3)
    saved=work/'candidate.npz'
    receipt=work/'candidate-evidence.json'
    if resume:
        entry=json.loads(receipt.read_text())
        if entry['source']!=source:
            raise ValueError('Saved mutation inputs differ from the requested parent, field or step')
        with np.load(saved) as data:
            child=trimesh.Trimesh(data['v'],data['f'],process=False)
        validate_single_part(child)
        if digest(child)!=entry['geometry'] or child.volume>study.volume_limit_mm3*(1.+1.e-6):
            raise ValueError('Saved boundary differs from the verified mutation')
        return child
    move=step
    with np.load(field_path) as field:
        while move>=spacing:
            try:
                child=propose(study,parent,[field],study.mechanics_executable,work,spacing,move,weights,step)
                if child is None:
                    break
                validate_single_part(child)
                identity=digest(child)
                if identity==source['parent'] or child.volume>study.volume_limit_mm3*(1.+1.e-6):
                    raise InadmissibleBoundary('No distinct admissible boundary at this step')
                temporary=work/'candidate-writing.npz'
                np.savez(temporary,v=child.vertices,f=child.faces)
                temporary.replace(saved)
                record=work/'candidate-evidence.tmp'
                record.write_text(json.dumps(dict(source=source,geometry=identity,move_mm=move),indent=2,allow_nan=False))
                record.replace(receipt)
                return child
            except InadmissibleBoundary as error:
                print('Rejected boundary at',move,'mm:',error,flush=True)
                move*=.5
    raise InadmissibleBoundary('No admissible boundary from this field at the declared sampling resolution')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent_npz',type=Path)
    parser.add_argument('driving_results',type=Path)
    parser.add_argument('matched_results',type=Path,help='Same case portfolio observed through separation on the parent')
    parser.add_argument('matched_summary',type=Path,help='Installed reference and parent results through separation')
    parser.add_argument('work',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--spacing-mm',type=float,default=.24)
    parser.add_argument('--step-mm',type=float,default=2.)
    parser.add_argument('--resume-proposal',action='store_true',help='Verify and resume the saved boundary from the same stress field')
    args=parser.parse_args()
    if not np.isfinite([args.spacing_mm,args.step_mm]).all() or not 0.<args.spacing_mm<=args.step_mm:
        parser.error('Supply a positive sampling spacing no greater than the initial boundary step')
    args.work.mkdir(parents=True,exist_ok=True)
    with np.load(args.parent_npz) as data:
        parent=trimesh.Trimesh(data['v'],data['f'],process=False)
    identity=digest(parent)
    if (args.driving_results/'identity.txt').read_text()!=identity:
        raise ValueError('The driving stress field belongs to a different geometry')
    study=study_definition(WORK/'study-assembly.obj')
    study=replace(study,volume_limit_mm3=min(study.volume_limit_mm3,parent.volume))
    rows=[json.loads((args.driving_results/(name+'.json')).read_text()) for name in ('lift','drop')]
    safety=YieldSafety(study.scenarios(),PLA.tensile_yield_mpa,2.,PLA.source)
    sensor='peak_acceleration_g@hub'
    reference_impact=json.loads((args.driving_results.parent/'installed/drop.json').read_text())
    if reference_impact['window_s']!=rows[1]['window_s']:
        raise ValueError('Compare the driving cases across an identical observation window')
    source_limits=(safety,ResponseLimit(('PLA / drop',),sensor,reference_impact[sensor]))
    driving=max(assessments(EvaluatedSurface(parent,[rows],[],'parent'),source_limits,require_all=True),
                key=lambda row:row['normalized_selection_excess'])
    label=driving['scenario']
    filename={'PLA / lift':'lift','PLA / drop':'drop'}[label]
    weights=mutation_weights(label in study.histories,ShapeTargets(),label)
    print('Driving case',label,driving['quantity'],driving['achieved'],flush=True)
    child=measured_proposal(study,parent,args.driving_results/(filename+'.npz'),label,weights,args.work,
                            args.spacing_mm,args.step_mm,args.resume_proposal)
    child_id=digest(child)
    print('Changed solid',child_id,'volume',child.volume,flush=True)
    cache=WORK/'joint-validation'
    volume_mesh(child,cache,1.9,.06)
    mesh=cache/f'conforming-surface-{child_id[:24]}-1.9-0.06.npz'
    solver,handle=linear_solver(WORK)
    verified,trial=simulate(study,mesh,args.work/'verification',solver,rebound_window=(.028,.056))
    if verified!=child_id:
        raise ValueError('The verification mesh belongs to a different trial')
    summary=json.loads(args.matched_summary.read_text())
    if summary['parts']['candidate']['geometry']!=identity:
        raise ValueError('The matched reference is a different arm')
    reference=[json.loads((args.matched_results/(name+'.json')).read_text()) for name in ('lift','drop')]
    installed_peak=summary['parts']['installed']['results'][1][sensor]
    targets=ShapeTargets(False,False,(safety,ResponseLimit(('PLA / drop',),sensor,installed_peak)),(
        MinimizeResponse(('PLA / lift',),'compliance_Nmm'),MinimizeResponse(study.scenarios(),'peak_MPa'),
        MinimizeResponse(('PLA / drop',),sensor)))
    candidates=[EvaluatedSurface(mesh,[row],[],name) for mesh,row,name in
                ((parent,reference,'parent'),(child,trial,'trial'))]
    archive=pareto(candidates,targets)
    ratings=compromise_scores(archive,targets,objective_scales(candidates[0],targets))
    selected=min(archive,key=lambda item:ratings[item.name])
    results={}
    for item in candidates:
        checks=assessments(item,targets.limits,require_all=True)
        results[item.name]=dict(geometry=digest(item.mesh),volume_mm3=float(item.mesh.volume),checks=checks,
                                observed_targets_met=all(r['met'] for r in checks),
                                compliance_Nmm=item.responses[0][0]['compliance_Nmm'],
                                drop_peak_MPa=item.responses[0][1]['peak_MPa'],hub_peak_g=item.responses[0][1][sensor])
    report=dict(driving_case=label,driver_window_s=rows[1]['window_s'],verification='matched single-mesh screening through separation',
                selected=selected.name,parts=results,reference_hub_g=installed_peak)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'Drop-feedback.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    publish_surface(child,args.output/'Front-Left-drop-trial.obj',args.work)
    print('Portfolio selection',selected.name,flush=True)


if __name__=='__main__':
    main()
