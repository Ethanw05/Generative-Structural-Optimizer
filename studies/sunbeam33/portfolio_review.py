"""Rank identical solid designs across matched lift, horizontal, torque and transient tests."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'build/sunbeam33/deps')]
import numpy as np

from filaments import PLA
from refined_cases import refined_cases
from gso_workflows.design_targets import MinimizeResponse,ResponseLimit,YieldSafety,assessments
from gso_workflows.surface_objectives import EvaluatedSurface,ShapeTargets,pareto,compromise_scores,objective_scales


def select_by_identity(reports,identity):
    entries=[part for report in reports for part in report['parts'].values() if part['geometry']==identity]
    if len(entries)!=1:
        raise ValueError(f'Expected one independent study record for solid {identity} and got {len(entries)}')
    return entries[0]


def released_contact(row):
    contacts=[event for event in row['contact_histories'] if event['first_contact_s'] is not None]
    return bool(contacts) and all(event['first_release_s'] is not None and event['final_force_N']==0. for event in contacts)


def collect(label,identity,folder,directions,drop_reports,service_reports,range_reports):
    part=select_by_identity(directions,identity)
    paired=select_by_identity(drop_reports,identity)
    rows=[]
    for case in ('lift','drop'):
        row=json.loads((folder/(case+'.json')).read_text())
        if row['scenario']!='PLA / '+case or row['n_tets']!=part['n_tets']:
            raise ValueError(f'Mismatched scenario or tetrahedra for {label} {case}')
        if 'results' in paired:
            recorded=next(r for r in paired['results'] if r['scenario']==row['scenario'])
        elif case=='lift':
            recorded=dict(compliance_Nmm=paired['compliance_Nmm'])
        else:
            recorded=dict(peak_MPa=paired['drop_peak_MPa'])
            recorded['peak_acceleration_g@hub']=paired['hub_peak_g']
        for key in ('peak_MPa','compliance_Nmm','peak_acceleration_g@hub'):
            if key in recorded:
                np.testing.assert_allclose(row[key],recorded[key],rtol=1.e-10,atol=0.)
        if case=='drop' and not released_contact(row):
            raise ValueError(f'Unresolved drop window for {label}')
        rows.append(row)
    for mode,pair in sorted(part['modes']['PLA'].items()):
        forward,reverse=pair['positive'],pair['negative']
        for key in ('peak_MPa','compliance_Nmm','regional_peak_rms_MPa'):
            np.testing.assert_allclose(forward[key],reverse[key],rtol=1.e-7)
        rows.append(dict(forward,scenario='PLA / '+mode))
    if range_reports:
        envelope=select_by_identity(range_reports,identity)
        if envelope['n_tets']!=part['n_tets']:
            raise ValueError('Directional range and axis tests used different meshes')
        for family,pair in sorted(envelope['planes'].items()):
            rows.append(dict(scenario='PLA / '+family+' range',**{k:pair[k] for k in
                             ('peak_MPa','maximum_compliance_Nmm','regional_peak_rms_at_stress_angle_MPa',
                              'stress_angle_deg','compliance_angle_deg')}))
    if service_reports:
        observed=select_by_identity(service_reports,identity)
        for key in ('brake','strike'):
            row=observed['results'][key]
            if row['scenario']!='PLA / '+key or row['n_tets']!=part['n_tets']:
                raise ValueError(f'Mismatched history for {label} {key}')
            if key=='strike' and not released_contact(row):
                raise ValueError(f'Unresolved strike for {label}')
            rows.append(row)
    return EvaluatedSurface(None,[rows],[],label)


def measured_rows(item,sensor):
    fields=('scenario','n_tets','peak_MPa','regional_peak_rms_MPa','compliance_Nmm','peak_elastic_displacement_mm',
            sensor,'mean_lift_mm','signed_rotation_deg','signed_displacement_mm','window_s','contact_histories',
            'maximum_compliance_Nmm','regional_peak_rms_at_stress_angle_MPa','stress_angle_deg','compliance_angle_deg')
    measured=[]
    for row in item.responses[0]:
        stored={k:row[k] for k in fields if k in row}
        refined=item.limit_evidence.get(row['scenario'],())
        if refined:
            stored['refinement']=[{k:r[k] for k in fields if k in r} for r in refined]
        measured.append(stored)
    return measured


def review(candidates,baseline):
    ref=next(item for item in candidates if item.name==baseline)
    rows=ref.responses[0]
    labels=tuple(row['scenario'] for row in rows)
    for item in candidates:
        if tuple(row['scenario'] for row in item.responses[0])!=labels:
            raise ValueError('Candidates require the same cases in matching order')
        for baseline_row,trial_row in zip(rows,item.responses[0],strict=True):
            if 'window_s' in baseline_row and not np.isclose(baseline_row['window_s'],trial_row['window_s'],rtol=1.e-12,atol=0.):
                raise ValueError('Observe every geometry over the same time window in '+baseline_row['scenario'])
    sensor='peak_acceleration_g@hub'
    limits=[YieldSafety(labels,PLA.tensile_yield_mpa,2.,PLA.source)]
    objectives=[MinimizeResponse(labels,'peak_MPa')]
    for metric in ('regional_peak_rms_MPa','regional_peak_rms_at_stress_angle_MPa','maximum_compliance_Nmm'):
        measured_on=tuple(row['scenario'] for row in rows if metric in row)
        if measured_on:
            objectives.append(MinimizeResponse(measured_on,metric))
    statics=tuple(row['scenario'] for row in rows if 'compliance_Nmm' in row)
    if statics:
        objectives.append(MinimizeResponse(statics,'compliance_Nmm'))
    for row in rows:
        name=(row['scenario'],)
        if sensor in row:
            limits.append(ResponseLimit(name,sensor,row[sensor]))
            objectives.append(MinimizeResponse(name,sensor))
        if 'peak_elastic_displacement_mm' in row:
            objectives.append(MinimizeResponse(name,'peak_elastic_displacement_mm'))
    targets=ShapeTargets(False,False,tuple(limits),tuple(objectives))
    options=[item for item in candidates if item.name!=baseline]
    if not options:
        raise ValueError('Supply at least one generated part to compare with the installed reference')
    archive=pareto(options,targets)
    scores=compromise_scores(archive,targets,objective_scales(ref,targets))
    selected=min(archive,key=lambda item:scores[item.name])
    parts={}
    for item in candidates:
        checks=assessments(item,targets.limits,require_all=True)
        parts[item.name]=dict(requirements_met=all(r['met'] for r in checks),checks=checks,
                              results=measured_rows(item,sensor))
    return dict(reference=baseline,selected_generated=selected.name,generated_nondominated=[c.name for c in archive],
                material=PLA.assumptions,verification='matched single verified tetrahedral mesh per geometry',
                static_signs='both independently solved, positive response used once in ranking',parts=parts)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--part',action='append',nargs=3,metavar=('LABEL','GEOMETRY_SHA256','LIFT_DROP_FOLDER'),required=True)
    parser.add_argument('--directions',type=Path,action='append',required=True)
    parser.add_argument('--drop',type=Path,action='append',required=True)
    parser.add_argument('--service',type=Path,action='append',default=[])
    parser.add_argument('--range',dest='ranges',type=Path,action='append',default=[])
    parser.add_argument('--refined-axes',type=Path,action='append',default=[])
    parser.add_argument('--refined-range',type=Path,action='append',default=[])
    parser.add_argument('--reference',required=True)
    args=parser.parse_args()
    read=lambda paths:[json.loads(path.read_text()) for path in paths]
    directional,drop,service,ranges=map(read,(args.directions,args.drop,args.service,args.ranges))
    # The two-case mutation record repeats the parent already reported by the matched comparison.
    for record in drop:
        if 'driving_case' in record:
            record['parts']={'trial':record['parts']['trial']}
    for record in (*directional,*ranges):
        if (record['force_N'],record['torque_Nmm'])!=(directional[0]['force_N'],directional[0]['torque_Nmm']):
            parser.error('Different directional amplitudes cannot share a comparison')
    if len({name for name,_,_ in args.part})!=len(args.part) or len({key for _,key,_ in args.part})!=len(args.part):
        parser.error('Use distinct part names and geometries')
    inputs=[collect(name,key,Path(folder),directional,drop,service,ranges) for name,key,folder in args.part]
    refined_axes,refined_ranges=map(read,(args.refined_axes,args.refined_range))
    resolutions=set()
    for item,(_,identity,_) in zip(inputs,args.part,strict=True):
        evidence,edge=refined_cases(identity,directional,refined_axes,refined_ranges)
        if set(evidence)-{row['scenario'] for row in item.responses[0]}:
            parser.error('Every refined case needs a matching baseline test')
        item.limit_evidence=evidence
        item.verify_margin=bool(evidence)
        resolutions.add(edge)
    if len(resolutions)!=1:
        parser.error('Rank all geometries at the same refinement setting')
    report=review(inputs,args.reference)
    if refined_axes:
        report['verification']='matched portfolio with additional fixed-exterior directional checks'
        report['directional_refinement_equivalent_edge_mm']=resolutions.pop()
    for name,key,_ in args.part:
        report['parts'][name]['geometry']=key
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Selected generated design:',report['selected_generated'],flush=True)
