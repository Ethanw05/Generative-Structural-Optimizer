"""Realized-solid verification and Pareto selection across static and transient load cases."""
from dataclasses import replace
from pathlib import Path
import secrets
from .scenario_feedback import FeedbackAnalysis,feedback_step
from .native_feedback import run_sweep
import json
import numpy as np
from .boundary_shape import propose
from .surface_objectives import ShapeTargets,EvaluatedSurface,objective_bounds,coordinates,pareto,objective_scales,compromise_scores
from .cad_history import UnresolvedHistory
from .finite_solid import UnresolvedSolid
from .analysis_boundary import analysis_skin
from .analysis_mesh import volume_mesh,digest
from .mesh_quality import InadmissibleBoundary,validate_single_part
from .tetra_check import solve
from .reference_regions import ReferenceRegions, compare_regions


def evaluate(study,mesh,pitch,spacing,work,solver,name,regions=None,feedback=None):
    if regions is None:
        regions = {label:ReferenceRegions() for label in study.scenarios()}
    levels=[];fields=[];coarse_fields=[]
    skin=None
    geometry_id=digest(mesh) if feedback is not None else None
    mesh_cache=feedback.workspace if feedback is not None else work
    for level,(edge,eps) in enumerate(((.95*pitch,.25*spacing),(.575*pitch,.21875*spacing),
                                      (.04375*pitch**3,0.))):
        if level==2:
            p,t=volume_mesh(skin,mesh_cache,edge,eps,'interior')
        else:
            p,t=volume_mesh(mesh,mesh_cache,edge,eps)
        if level==0:
            skin=analysis_skin(p,t)
        results=[];case_fields=[]
        if level==0 and feedback is not None:
            measured=[feedback.field(mesh,geometry_id,label) for label in study.scenarios()]
            levels.append([entry.response for entry in measured])
            coarse_fields=[entry.path for entry in measured]
            continue
        for index,(label,load) in enumerate(study.loads.items()):
            single=replace(study.for_case(label),loads={label:load})
            path=work/f'{name}-level{level}-case{index}'
            result=solve(p,t,single,path,solver,surface_tol=1.5*spacing, local_regions=regions[label])
            results.append(result)
            case_fields.append(path.with_suffix('.npz'))
        if study.histories:
            from .cad_history import export_history
            from .tetra_elasticity import assemble
            system=None
            previous=None
            for index,(label,case) in enumerate(study.histories.items(),start=len(study.loads)):
                active=study.for_case(label)
                if active.density_kg_m3 is None or active.mechanics_executable is None:
                    raise ValueError("History cases require material density and the native mechanics executable")
                elastic=(active.young_mpa,active.poisson)
                if elastic!=previous:
                    system=assemble(p,t,active,1.5*spacing)
                    previous=elastic
                path=work/f"{name}-level{level}-case{index}"
                results.append(export_history(system,active,case,path,solver,regions[label],1.5*spacing,law=active.solid_laws.get(label)))
                case_fields.append(path.with_suffix(".npz"))
            del system
        levels.append(results)
        if level==0:coarse_fields=case_fields
        if level==1:fields=case_fields
    return EvaluatedSurface(mesh,levels,fields,name,coarse_fields,regions)


def evolve_surface(study,baseline,pitch,exe,work,solver,targets=None,generations=2,initial=None,
                   sequence_seed=None,case_updates=None,working_surface=None):
    """One generation visits every active case on the latest admissible descendant."""
    targets=targets or ShapeTargets()
    targets.names()
    if not isinstance(generations,int) or generations<1:
        raise ValueError('A positive number of case sweeps is required')
    if not np.isfinite(pitch) or pitch<=0.:
        raise ValueError('Evolution requires a positive finite analysis pitch')
    seed=secrets.randbits(64) if sequence_seed is None else sequence_seed
    if not isinstance(seed,int) or not 0<=seed<2**64:
        raise ValueError('The scenario sequence seed must be an unsigned 64-bit integer')
    work=Path(work)
    work.mkdir(parents=True,exist_ok=True)
    spacing=min(.24,pitch/5.)
    validate_single_part(baseline)
    if initial is not None and digest(initial.mesh)!=digest(baseline):
        raise ValueError('Initial mechanical responses belong to a different part')
    # The same geometry can have changed loads, material, or fixtures on a resumed run.
    # Only evidence produced in this study invocation is eligible for reuse.
    analysis=FeedbackAnalysis(study,baseline,pitch,spacing,work,solver)
    current=baseline if working_surface is None else working_surface
    identity=analysis.baseline_id if current is baseline else digest(current)
    checkpoints={'initial':baseline}
    if identity!=analysis.baseline_id:
        validate_single_part(current)
        if current.volume>study.volume_limit_mm3*(1.+1.e-6):
            raise ValueError('The resumed working solid exceeds the material budget')
        checkpoints['resumed']=current
    analysis.start(current)
    verified={}
    trace=[]
    lineage=[]
    expanded=[]
    termination='generation_budget'
    radius=pitch
    trust_history=[]
    print(f'Evolution seed {seed}, run {analysis.run}',flush=True)
    for generation in range(1,generations+1):
        if case_updates is not None:
            added=case_updates(generation,study)
            if added is not None and (added.loads or added.histories):
                study=added.apply(study)
                analysis.study=study
                # Recompare every stored endpoint on the expanded portfolio, including
                # previously dominated endpoints that can excel on the added case.
                verified.clear()
        first_event=len(lineage)
        start=identity
        sweep_start=current
        expanded.append(start)
        def advance(label,ordinal,parent_id):
            nonlocal current,identity
            if parent_id!=identity:
                raise RuntimeError('Native sweep and mechanical state refer to different geometries')
            try:
                current,identity,event=feedback_step(analysis,current,identity,label,targets,exe,radius,spacing,
                                                     propose,generation,ordinal)
            except (InadmissibleBoundary,UnresolvedHistory,UnresolvedSolid) as error:
                if identity==analysis.baseline_id:
                    raise
                previous=identity
                current=sweep_start if start!=identity else baseline
                identity=digest(current)
                try:
                    analysis.field(current,identity,label)
                except (InadmissibleBoundary,UnresolvedHistory,UnresolvedSolid):
                    current,identity=baseline,analysis.baseline_id
                rollback=dict(sweep=generation,step=ordinal,case=label,status='rollback',parent=previous,
                              child=identity,reason=str(error))
                analysis.checkpoint(current,rollback)
                analysis.record(rollback)
                lineage.append(rollback)
                current,identity,event=feedback_step(analysis,current,identity,label,targets,exe,radius,spacing,
                                                     propose,generation,ordinal)
            lineage.append(event)
            print(f"Sweep {generation}, step {ordinal}, {label}: {event['status']} "
                  f"{event['parent'][:12]} -> {identity[:12]}",flush=True)
            return identity
        identity=run_sweep(exe,study.scenarios(),seed,generation,identity,advance)
        name=f'g{generation}'
        if identity!=start and all(identity!=digest(m) for m in checkpoints.values()):
            checkpoints[name]=current
        for tag,mesh in list(checkpoints.items()):
            if tag in verified:
                continue
            try:
                item=evaluate(study,mesh,pitch,spacing,analysis.run,solver,tag,analysis.regions,analysis)
            except (InadmissibleBoundary,UnresolvedHistory,UnresolvedSolid) as error:
                if tag=='initial':
                    raise
                trace.append(dict(name=tag,rejected=str(error)))
                del checkpoints[tag]
                if digest(mesh)==identity:
                    front=pareto(list(verified.values()),targets)
                    ratings=compromise_scores(front,targets,objective_scales(verified['initial'],targets))
                    fallback=min(front,key=lambda c:ratings[c.name])
                    current=fallback.mesh
                    identity=digest(current)
                    rollback=dict(sweep=generation,status='rollback',parent=digest(mesh),child=identity,rejected=tag)
                    analysis.checkpoint(current,rollback)
                    analysis.record(rollback)
                    lineage.append(rollback)
                continue
            verified[tag]=item
            trace.append(dict(name=tag,scenarios=list(study.scenarios()),geometry=digest(mesh),
                              responses=item.responses))
        # Portfolio checks change the next move size on the working lineage. A load
        # which has just run cannot erase regressions under a different load.
        front=pareto(list(verified.values()),targets)
        ratings=compromise_scores(front,targets,objective_scales(verified['initial'],targets))
        champion=min(front,key=lambda item:ratings[item.name])
        old_radius=radius
        if identity!=start:
            if identity==digest(champion.mesh):
                radius=min(max(pitch,2.*current.volume/current.area),2.*radius)
            else:
                radius=max(spacing,.5*radius)
        trust_history.append(dict(sweep=generation,working=identity,best_verified=champion.name,
                                  used_mm=old_radius,next_mm=radius))
        if case_updates is None and all(e['status']=='retained' for e in lineage[first_event:]):
            termination='no_distinct_feasible_candidate'
            break
    initial_surface=verified['initial']
    baseline_values=coordinates(initial_surface,targets)
    response_scale=objective_scales(initial_surface,targets)
    archive=pareto(list(verified.values()),targets)
    scores=compromise_scores(archive,targets,response_scale)
    parent=min(archive,key=lambda item:scores[item.name])
    for item in archive:
        path=analysis.run/(item.name+'.obj')
        item.mesh.export(path,digits=13)
    mesh_change={item.name:[
        {metric:float(np.ptp([r[metric] for r in cases])/max(np.finfo(float).tiny,max(r[metric] for r in cases))) for metric in targets.names(cases[1])}
        for cases in zip(*item.responses)]
        for item in archive}
    comparisons={}
    for item in archive:
        ranks={}
        for case_index,name in enumerate(study.scenarios()):
            ranks[name]={}
            for metric in targets.names(item.responses[0][case_index]):
                old=np.array([row[case_index][metric] for row in initial_surface.responses])
                new=np.array([row[case_index][metric] for row in item.responses])
                delta=np.divide(new-old,old,out=np.zeros_like(new,dtype=float),where=old>0.)
                reported=[float(value) if initial>0. else None for value,initial in zip(delta,old,strict=True)]
                # Magnitudes relative to a zero reference are undefined. Direction is measurable.
                delta[(old==0.) & (new>0.)]=np.inf
                if np.all(np.abs(delta)<=1.e-8):status='unchanged'
                elif np.all(delta<=0.):status='lower_on_every_mesh'
                elif np.all(delta>=0.):status='higher_on_every_mesh'
                else:status='ranking_changes_with_mesh'
                ranks[name][metric]=dict(comparison=status,relative_change_by_mesh=reported)
            if targets.maximum_stress:
                ranks[name]['tracked_local_peaks'] = compare_regions(
                    [level[case_index] for level in initial_surface.responses],
                    [level[case_index] for level in item.responses])
        comparisons[item.name]=ranks
    report=dict(verification=('coarse','fine','fixed-interface refinement'),relative_mesh_spread=mesh_change,comparison_with_start=comparisons,scenarios=list(study.scenarios()),selected=parent.name,nondominated=[c.name for c in archive],
                local_regions={name: region.anchors.tolist() for name,region in initial_surface.regions.items()},
                termination=termination,expanded=expanded,sequence_seed=seed,lineage=lineage,trust_history=trust_history,
                solid_models={name:study.solid_model(name) for name in study.scenarios()},
                working_geometry=identity,feedback_directory=str(analysis.run),objective_names=targets.names(),
                selection_method='minimum worst normalized regret from the observed per-objective best',
                compromise_scores=scores,objective_scales=response_scale.tolist(),
                objectives_by_scenario={name:targets.names(case) for name,case in
                                        zip(study.scenarios(),initial_surface.responses[0],strict=True)},
                baseline=baseline_values.tolist(),evaluations=trace)
    report_path=analysis.run/'selection.json'
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False))
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(dir=work) as temporary:
        latest=Path(temporary)/'selection.json'
        latest.write_bytes(report_path.read_bytes())
        latest.replace(work/'selection.json')
    return parent,report
