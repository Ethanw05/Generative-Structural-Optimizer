"""Check CAD time recovery against the exact modal step response of the full 3-D FE system."""
from pathlib import Path
from itertools import permutations,product
import sys,tempfile
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'build/sunbeam33/meshlib-runtime'),str(root/'build/sunbeam33/deps'),str(root/'tools')]
import numpy as np
import trimesh
import manifold3d as md
from scipy.linalg import eigh
from scipy.sparse.linalg import spsolve
from gso_workflows.design_space import Face,DesignSpace
from gso_workflows.volume_study import SurfaceLoad,VolumeStudy
from gso_workflows.tetra_elasticity import assemble,applied
from gso_workflows.ritz_response import consistent_mass,stress_envelope
from gso_workflows.cad_history import HistoryCase,TimeExcitation,measure_history
from gso_workflows.history_solver import integrate

exe=Path(sys.argv[1]).resolve()
from gso_workflows.history_envelope import HistoryEnvelope
rng=np.random.default_rng(193)
states=rng.normal(size=(257,5))
states[53]=states[11]*12.
states[91]=states[53]
maps=rng.normal(size=(31,6,5))
exact=np.linalg.norm(maps@states.T,axis=1)
maximum,indices=HistoryEnvelope(states,exe).measure(maps)
np.testing.assert_allclose(maximum,exact.max(axis=1),rtol=1.e-12)
np.testing.assert_array_equal(indices,exact.argmax(axis=1))
size=np.array([3,1,1])
points=np.array(list(product(*(range(n+1) for n in size))),dtype=float)
cells=[]
for x in product(*(range(n) for n in size)):
    for sequence in permutations(range(3)):
        chain=np.vstack((np.zeros(3,dtype=int),np.eye(3,dtype=int)[list(sequence)].cumsum(axis=0)))
        cells.append(np.ravel_multi_index((chain+x).T,size+1))
native=md.Manifold.cube(size.tolist())
packed=native.to_mesh64()
mesh=trimesh.Trimesh(packed.vert_properties[:,:3],packed.tri_verts,process=False)
left=Face.pick(mesh,[0.,.5,.5],[-1.,0.,0.])
right=Face.pick(mesh,[3.,.5,.5],[1.,0.,0.])
load=SurfaceLoad(right,np.array([.35,.2,1.]))
study=VolumeStudy(DesignSpace(native,(),()),(left,),{'step':load},2200.,.35,3.,np.eye(3),np.zeros(3),exe)
system=assemble(points,np.asarray(cells),study,.01)
free=np.tile(~system.fixed,3)
k=system.stiffness[free][:,free].toarray()
m=consistent_mass(system,1040.)[free][:,free].toarray()
lam,phi=eigh(k,m)
force=applied(study,load,system.local,system.skin,.01)[0].T.ravel()[free]
duration=4.e-6
time=np.linspace(0.,duration,801)
coordinates=(1.-np.cos(time[:,None]*np.sqrt(lam)))*((phi.T@force)/lam)
basis=np.zeros((len(free),len(lam)));basis[free]=phi
truth,_=stress_envelope(system,basis,coordinates)
case=HistoryCase(duration,duration/50.,(TimeExcitation(load,np.array([0.]),np.array([1.])),))
response,vm,when=measure_history(system,study,case,spsolve,exe,1040.,.01,relative_tolerance=.003,max_orders=32)
np.testing.assert_allclose(vm,truth,rtol=.012)
assert np.max(when)<=duration and response['modes']<=len(lam)

# A free mass strikes a linear unilateral foundation and rebounds. The force and
# separation follow the closed-form compression half-wave.
step=.002
states,out=integrate(exe,[[0.]],[[1.]],[-2.],step,np.empty((0,1)),np.empty((0,551)),[[1.]],[.5],[16.])
np.testing.assert_allclose(out['peak_contact_N'],8.,rtol=5.e-5)
assert states[-1,0]>-.5

# An off-grid triangular pulse lies entirely between two nominal recording times.
# Free-mass momentum after the pulse equals its analytical force impulse.
from gso_workflows.cad_history import event_times,FoundationContact
pulse=TimeExcitation(load,np.array([0.,.011,.012,.013,.1]),np.array([0.,0.,1.,0.,0.]))
timeline=event_times(HistoryCase(.1,.05,(pulse,)),0)
assert set(pulse.times_s)<=set(timeline)
states,_=integrate(exe,[[0.]],[[1.]],[0.],timeline,[[1.]],[pulse.at(timeline)],np.empty((0,1)),[],[])
np.testing.assert_allclose((states[-1,0]-states[-2,0])/(timeline[-1]-timeline[-2]),.001,rtol=1.e-10)

from gso_workflows.rigid_carrier import RigidCarrier
contact_face=Face.pick(mesh,[1.5,.5,0.],[0.,0.,-1.])
carrier=RigidCarrier(np.array([0.,.5,.5]),np.array([0.,.5,.5]),.005,np.diag([.01,.01,.01]),
                     np.array([0.,0.,-5.]),np.zeros(3))
contact=FoundationContact(SurfaceLoad(contact_face,np.array([0.,0.,1.])),.0002,20.)
drop=HistoryCase(.0009,.000009,(),contacts=(contact,),carrier=carrier)
impact,vm,t=measure_history(system,study,drop,spsolve,exe,1040.,.01,relative_tolerance=.025,max_orders=24)
total_mass=.005+system.volume.sum()*1040.e-9
assert 0.<impact['peak_contact_N']<=1.03*np.sqrt(20.*total_mass*.001*25.)
assert vm.max()>0. and impact['peak_rigid_rotation_rad']<.01
trace=impact['contact_histories'][0]
assert trace['impulse_Ns']>0. and 0.<trace['first_contact_s']<=trace['peak_time_s']<=drop.duration_s
np.testing.assert_allclose(trace['peak_N'],impact['peak_contact_N'])

# Exercise the public candidate scoring path with the same CAD solid under a
# static load and a real time history. The full FE benchmark above checks accuracy.
from dataclasses import replace
from unittest.mock import patch
from gso_workflows.surface_search import evaluate,coordinates,ShapeTargets
with tempfile.TemporaryDirectory(dir=root/'build') as tmp:
    folder=Path(tmp)
    mild=HistoryCase(2.e-6,.5e-7,(TimeExcitation(load,np.array([0.,2.e-6]),np.array([0.,1.])),),
                     relative_tolerance=.06,max_orders=34,targets=('peak_energy_Nmm',))
    mixed=replace(study,histories={'ramp':mild},density_kg_m3=1040.)
    with patch('gso_workflows.surface_search.volume_mesh',return_value=(points,np.asarray(cells))):
        scored=evaluate(mixed,mesh,.5,.04,folder,spsolve,'mixed')
    from gso_workflows.tetra_check import solve
    from gso_workflows.reference_regions import ReferenceRegions
    unloaded=replace(study,loads={'unloaded':SurfaceLoad(right,np.zeros(3))})
    zero=solve(points,np.asarray(cells),unloaded,folder/'unloaded',spsolve,surface_tol=.01,local_regions=ReferenceRegions())
    assert zero['peak_MPa']==zero['compliance_Nmm']==zero['relative_regional_stress']==0.
    for static,history in scored.responses:
        assert 'compliance_Nmm' in static and history['peak_energy_Nmm']>0.
        assert 'compliance_Nmm' not in history and len(history['local_peaks_MPa'])>0
    assert len(coordinates(scored,ShapeTargets()))>18
    with np.load(scored.evidence[1]) as field:
        assert field['transient'] and np.max(field['peak_time_s'])<=mild.duration_s
