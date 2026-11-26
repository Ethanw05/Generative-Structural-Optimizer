"""The public generate path automatically scores stiffness and peak stress on a real solid."""
from pathlib import Path
import sys
import tempfile
import json

root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'build/sunbeam33/meshlib-runtime'),str(root/'build/sunbeam33/deps'),str(root/'tools')]
import numpy as np
import manifold3d as md
import trimesh
from gso_workflows.design_space import DesignSpace,Face
from gso_workflows.volume_study import VolumeStudy,SurfaceLoad,generate
from gso_workflows.surface_search import ShapeTargets,EvaluatedSurface,pareto


def result(name,work,peak):
    return EvaluatedSurface(None,[[{'compliance_Nmm':work,'peak_MPa':peak,'regional_peak_rms_MPa':peak}]],[],name)


solutions=[result('stiff',1.,5.),result('low stress',2.,2.),result('dominated',3.,6.)]
assert [s.name for s in pareto(solutions,ShapeTargets())]==['stiff','low stress']
assert pareto(solutions,ShapeTargets(True,False))[0].name=='stiff'
assert pareto(solutions,ShapeTargets(False,True))[0].name=='low stress'

# A lower global maximum and lower regional RMS can still hide a worsening local joint.
old = result('reference', 1., 10.)
changed = result('tradeoff', .9, 8.)
old.responses[0][0]['local_peaks_MPa'] = [10., 4.]
changed.responses[0][0]['local_peaks_MPa'] = [8., 6.]
assert len(pareto([old, changed], ShapeTargets())) == 2

# A status quo reference cannot win automatically whenever any single objective rises.
from gso_workflows.surface_search import compromise_scores,coordinates,objective_scales
reference=result('starting design',10.,100.)
balanced=result('compromise',10.2,80.)
goals=ShapeTargets()
ratings=compromise_scores([reference,balanced],goals,coordinates(reference,goals))
assert ratings[balanced.name]<ratings[reference.name]
assert ratings==compromise_scores([balanced,reference],goals,coordinates(reference,goals))

# Losing centroid coverage must never produce a fictitious zero-stress improvement.
missing=result('unsampled',.9,8.)
missing.responses[0][0].update(local_peaks_MPa=[8.,0.],local_region_volumes_mm3=[1.,0.])
assert len(pareto([old,missing],ShapeTargets()))==2
# MPa has the same physical weight at the clamp and at a previously unstressed location.
old.responses[0][0]['local_peaks_MPa']=[10.,.0001]
np.testing.assert_array_equal(objective_scales(old,ShapeTargets()),[1.,10.,10.,10.,10.])

cube=md.Manifold.cube([7.,4.,4.])
native=cube.to_mesh64()
surface=trimesh.Trimesh(native.vert_properties[:,:3],native.tri_verts,process=False)
left=Face.pick(surface,[0,2,2],[-1,0,0])
right=Face.pick(surface,[7,2,2],[1,0,0])
space=DesignSpace(cube,(left.prism(-1.),right.prism(-1.)),())
study=VolumeStudy(space,(left,),{'vertical':SurfaceLoad(right,np.array([0.,0.,2.])),
                    'lateral':SurfaceLoad(right,np.array([0.,1.,0.]))},2000.,.3,90.,np.eye(3),np.zeros(3))
# A later state is reachable only through a worse active response. A load added
# in the next sweep must also re-score every previously verified endpoint.
from unittest.mock import patch
from gso_workflows.surface_search import evolve_surface
from gso_workflows.scenario_feedback import CaseField,ScenarioAdditions,case_order
from gso_workflows.analysis_mesh import digest
from types import SimpleNamespace
exe=Path(sys.argv[1]).resolve()

# A finer analysis overturns a provisional pass, and its field becomes the next proposal's resolution.
from gso_workflows.design_targets import YieldSafety, assessments
from gso_workflows.limit_verification import verify_limits
near=EvaluatedSurface(trimesh.creation.box(),[[dict(scenario='pull',peak_MPa=s)] for s in (13.06,14.19,14.11)],[],'near')
queries=[]
def measured_field(mesh,identity,label,resolution):
    queries.append(resolution)
    return SimpleNamespace(response=dict(scenario=label,peak_MPa=15.12))
probe=SimpleNamespace(analysis_pitch=.72,edge=.684,envelope=.036,proposal_resolution={},field=measured_field)
constraint=YieldSafety(('pull',),30.,2.,'test')
verify_limits(near,ShapeTargets(limits=(constraint,)),probe)
assert len(queries)==1 and probe.proposal_resolution['pull']==queries[0]
assert assessments(near,(constraint,))[0]['status']=='exceeded'

with tempfile.TemporaryDirectory(dir=root/'build') as directory:
    work=Path(directory)
    coupon=trimesh.creation.box(extents=[2.,2.,2.])
    visited=[]
    def fields(analysis, mesh, identity, label):
        assert identity==digest(mesh)
        state=int(round(mesh.centroid[0]))
        path=work/f'{identity}-{label}.npz'
        np.savez(path, geometry=identity, label=label, state=state)
        analysis.regions.setdefault(label,SimpleNamespace(anchors=np.empty((0,3))))
        return CaseField(identity,label,{'compliance_Nmm':[2.,3.,1.][state]},path)
    def mutate(study,mesh,evidence,*args):
        field,=evidence
        assert str(field['geometry'])==digest(mesh)
        state=int(field['state'])
        if state==2:return None
        child=mesh.copy()
        child.apply_translation([1.,0.,0.])
        return child
    def score(study,mesh,pitch,spacing,work,solver,name,regions,feedback):
        visited.append((name,tuple(study.scenarios())))
        value=[fields(feedback,mesh,digest(mesh),label).response for label in study.scenarios()]
        return EvaluatedSurface(mesh,[value]*3,[],name,regions=regions)
    from dataclasses import replace
    one=replace(study,loads={'vertical': study.loads['vertical']})
    def add(sweep, model):
        if sweep==2:return ScenarioAdditions(loads={'lateral':study.loads['lateral']})
    with patch('gso_workflows.scenario_feedback.FeedbackAnalysis.field', fields), \
         patch('gso_workflows.surface_search.propose',mutate), patch('gso_workflows.surface_search.evaluate',score):
        chosen,audit=evolve_surface(one,coupon,1.,exe,work,None,ShapeTargets(True,False),
                                    generations=3,sequence_seed=177,case_updates=add)
        from gso_workflows.scenario_feedback import resume_pair
        original,working=resume_pair(audit['feedback_directory'])
        assert digest(original)==digest(coupon) and digest(working)==audit['working_geometry']
        revisited,second=evolve_surface(add(2,one).apply(one),original,1.,exe,work,None,ShapeTargets(True,False),
                                       generations=1,sequence_seed=1,working_surface=working)
        assert second['selected']=='resumed' and second['lineage'][0]['parent']==digest(working)
        assert second['termination']=='no_distinct_feasible_candidate'
        assert digest(resume_pair(second['feedback_directory'])[1])==digest(working)
    assert chosen.responses[0][0]['compliance_Nmm']==1.
    assert audit['selected'] in audit['nondominated']
    assert audit['lineage'][0]['selection']=='exploratory'
    assert audit['lineage'][1]['parent']==audit['lineage'][0]['child']
    assert ('g1',('vertical','lateral')) in visited
    assert ('initial',('vertical','lateral')) in visited
    for sweep in (2,3):
        sequence=[s['case'] for s in audit['lineage'] if s['sweep']==sweep]
        assert sequence==case_order(exe, ['lateral','vertical'],177,sweep)
        assert sorted(sequence)==['lateral','vertical']

with tempfile.TemporaryDirectory(dir=root/'build') as temporary:
    work=Path(temporary)
    destination=work/'coupon.obj'
    response=generate(study,.5,12,Path(sys.argv[1]).resolve(),work,destination,sequence_seed=389)
    external=trimesh.load_mesh(destination,process=False)
    print('OBJ solid, boundary shells, volume:',external.is_volume,external.body_count,external.volume,flush=True)
    from gso_workflows.mesh_quality import validate_single_part
    validate_single_part(external)
    assert external.volume<=90.0001
    report=json.loads((work/'coupon-surface/selection.json').read_text())
    assert set(report['objective_names'])=={'compliance_Nmm','peak_MPa','regional_peak_rms_MPa'}
    assert report['selected'] in report['nondominated']
    from gso_workflows.scenario_feedback import resume_geometry
    assert digest(resume_geometry(report['feedback_directory']))==report['working_geometry']
    assert len(report['baseline'])>18
    advanced=[step for step in report['lineage'] if step['status']=='advanced']
    assert len(advanced)>=2 and len(set(step['child'] for step in advanced))>=2
    assert all(a['child']==b['parent'] for a,b in zip(report['lineage'],report['lineage'][1:]))
    assert all(step['parent']==step['field_geometry'] for step in advanced)
    assert all('local_peaks_MPa' in r for e in report['evaluations'] if 'responses' in e for level in e['responses'] for r in level)
    assert any('responses' in entry for entry in report['evaluations'])
    print('Two-load automatic surface refinement passed',response)
