"""Case-scoped feasibility, verified worst stress and independently selected shock targets."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'tools'),str(root/'build/sunbeam33/deps')]
from gso_workflows.design_targets import YieldSafety, ResponseLimit, MinimizeResponse, MaximizeResponse, assessments
from gso_workflows.surface_objectives import ShapeTargets, EvaluatedSurface, pareto, compromise_scores, objective_scales


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected invalid or missing evidence to fail')


def part(name, stresses, acceleration, work):
    levels=[]
    for stress in stresses:
        levels.append([dict(scenario='pull',peak_MPa=stress,compliance_Nmm=work,regional_peak_rms_MPa=1.),
                       dict(scenario='drop',time_targets=[],peak_MPa=2.,regional_peak_rms_MPa=1.,
                            peak_elastic_displacement_mm=.2,**{'peak_acceleration_g@hub':acceleration})])
    return EvaluatedSurface(None,levels,[],name)


safety=YieldSafety(('pull',),30.,2.,'assumed isotropic yield')
shock=ResponseLimit(('drop',),'peak_acceleration_g@hub',10.)
extra=MinimizeResponse(('drop',),shock.metric)
targets=ShapeTargets(limits=(safety,shock),extra_objectives=(extra,))
study=SimpleNamespace(scenarios=lambda:('pull','drop'),histories={'drop':SimpleNamespace(
    receivers=(SimpleNamespace(metric=shock.metric),),transmissions=())})
targets.validate_cases(study)
rejects(lambda:ShapeTargets(limits=(ResponseLimit(('pull',),shock.metric,1.),)).validate_cases(study))
rejects(lambda:ShapeTargets(limits=(ResponseLimit(('missing',),'peak_MPa',3.),)).validate_cases(study))
for invalid in (0.,-1.,np.nan,np.inf):
    rejects(lambda:YieldSafety(('pull',),invalid,2.,'test'))
    rejects(lambda:ResponseLimit(('drop',),'peak_MPa',invalid))
a=part('feasible',(12.,15.),9.,10.)
b=part('stiffer',(14.,16.),8.,1.)
c=part('shock fails',(10.,11.),12.,1.)
assert [x.name for x in pareto([c,b,a],targets)]==['feasible']
assert [x.name for x in pareto([c,b],targets)]==['stiffer']
checks=assessments(a,targets.limits,require_all=True)
assert checks[0]['achieved']==2. and checks[0]['level']==1 and all(r['met'] for r in checks)
assert shock.metric not in targets.names(a.responses[0][0]) and shock.metric in targets.names(a.responses[0][1])
limits_only=ShapeTargets(False,False,(safety,shock))
ranks=compromise_scores([a,b,c],limits_only,objective_scales(a,limits_only))
assert ranks['feasible']<ranks['stiffer']<ranks['shock fails']
yielded=safety.assessment({'peak_MPa':3.,'peak_equivalent_plastic_strain':.001,'solid_model':{'young_MPa':2200.}})
assert yielded['plastic_flow_detected'] and yielded['achieved'] is None and yielded['violation']>0.
zero=safety.assessment({'peak_MPa':0.})
assert zero['unbounded'] and zero['achieved'] is None
json.dumps(zero,allow_nan=False)
rejects(lambda:shock.assessment({'scenario':'drop'}))
scope=EvaluatedSurface(None,[[a.responses[0][0]]],[],'partial')
rejects(lambda:assessments(scope,targets.limits,require_all=True))
maximize=ShapeTargets(False,False,(),(MaximizeResponse(('drop',),shock.metric),))
assert [x.name for x in pareto([a,b,c],maximize)]==['shock fails']
opposed=ShapeTargets(False,False,(),(extra,MaximizeResponse(('drop',),shock.metric)))
rejects(lambda:pareto([a,b],opposed))
narrow=part('near limit',(13.06,14.19,14.11),8.,1.)
narrow.verify_margin=True
assert assessments(narrow,(safety,))[0]['status']=='margin_unresolved'
narrow.limit_evidence={'pull':[dict(scenario='pull',peak_MPa=15.12)]}
check=assessments(narrow,(safety,))[0]
assert check['status']=='exceeded' and check['level']==3
narrow.limit_evidence={'pull':[dict(scenario='pull',peak_MPa=14.20)]}
assert assessments(narrow,(safety,))[0]['met']
rejects(lambda:ShapeTargets(verification_edges_mm=(.3,.4)))
print('Scoped yield, shock, missing evidence and feasibility ordering passed')


# Scope extra evidence to its physical case and respect each objective direction.
from gso_workflows.surface_objectives import coordinates
proof=EvaluatedSurface(None,[[dict(scenario='pull',peak_MPa=10.,stiffness=12.),
                             dict(scenario='twist',peak_MPa=8.)]],[],'verified',
                      limit_evidence={'pull':[dict(peak_MPa=17.,stiffness=9.)]})
requests=ShapeTargets(False,False,extra_objectives=(MinimizeResponse(('pull','twist'),'peak_MPa'),
                                                   MaximizeResponse(('pull',),'stiffness')))
np.testing.assert_allclose(coordinates(proof,requests),[17.,-9.,8.])
