"""Exact finite-extension equilibrium, modal limit, and a rigid-carrier impact."""
from pathlib import Path
from itertools import permutations,product
import sys,tempfile
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'build/sunbeam33/meshlib-runtime'),str(root/'build/sunbeam33/deps'),str(root/'tools')]
import manifold3d as md
import numpy as np
import trimesh
from scipy.optimize import brentq
from scipy.linalg import eigh
from scipy.sparse.linalg import spsolve
from dataclasses import replace
from gso_workflows.design_space import Face,DesignSpace
from gso_workflows.volume_study import VolumeStudy,SurfaceLoad
from gso_workflows.tetra_elasticity import applied,assemble
from gso_workflows.tetra_check import solve
from gso_workflows.finite_solid import FiniteStrain,FiniteSolid
from gso_workflows.nonlinear_equilibrium import static_path
from gso_workflows.cad_history import HistoryCase,TimeExcitation,FoundationContact,export_history
from gso_workflows.reference_regions import ReferenceRegions
from gso_workflows.ritz_response import consistent_mass,stress_envelope
from gso_workflows.rigid_carrier import RigidCarrier

exe=Path(sys.argv[1]).resolve()
points=np.array(list(product(range(2),repeat=3)),dtype=float)
cells=[]
for axes in permutations(range(3)):
    chain=np.vstack((np.zeros(3,dtype=int),np.eye(3,dtype=int)[list(axes)].cumsum(axis=0)))
    cells.append(np.ravel_multi_index(chain.T,(2,2,2)))
cells=np.asarray(cells)
cube=md.Manifold.cube([1.,1.,1.])
data=cube.to_mesh64()
mesh=trimesh.Trimesh(data.vert_properties[:,:3],data.tri_verts,process=False)
left=Face.pick(mesh,[0.,.5,.5],[-1.,0.,0.])
right=Face.pick(mesh,[1.,.5,.5],[1.,0.,0.])
load=SurfaceLoad(right,np.array([15.,0.,0.]))
law=FiniteStrain()
study=VolumeStudy(DesignSpace(cube,(),()),(left,),{'pull':load},100.,0.,1.,np.eye(3),np.zeros(3),
                  exe,density_kg_m3=1040.,solid_laws={'pull':law})
system=assemble(points,cells,study,.01)
force=applied(study,load,system.local,system.skin,.01)[0].T.ravel()
solid=FiniteSolid(system,law,exe)
# Apply a consistent nodal traction on the FE boundary for the analytical patch.
# Projecting a separately triangulated CAD face can use a different diagonal.
tri=system.skin[np.all(system.points[system.skin,0]==1.,axis=1)]
traction=np.zeros_like(system.points)
for face in tri:traction[face,0]+=15./6.
traction=traction.T.ravel()
q,field,residual,_=static_path(solid,[traction],spsolve)[0]
stretch=brentq(lambda x:100.*np.log(x)/x-15.,1.,np.e)
np.testing.assert_allclose(q.reshape(3,-1)[0],(stretch-1.)*system.points[:,0],atol=1.e-8)
np.testing.assert_allclose(field.measures[:,0],15.,atol=1.e-6)
assert residual<1.e-7
plastic=FiniteSolid(system,FiniteStrain(3.,15.),exe)
loaded,unloaded=static_path(plastic,[.5*traction,np.zeros_like(traction)],spsolve)
assert loaded[1].state[:,9].max()>.01 and np.linalg.norm(unloaded[0])>.01
with tempfile.TemporaryDirectory(dir=root/'build') as tmp:
    path=Path(tmp)
    output=solve(points,cells,study,path/'finite',spsolve,local_regions=ReferenceRegions(),report_to_console=False)
    assert output['solid_model']['kinematics']=='finite strain' and output['peak_MPa']>0.
    small=replace(load,force_n=np.array([1.e-3,0.,0.]))
    duration=4.e-6
    case=HistoryCase(duration,duration/12,(TimeExcitation(small,np.array([0.]),np.array([1.])),),
                     relative_tolerance=.025,max_time_levels=6)
    history=export_history(system,study,case,path/'finite-history',spsolve,ReferenceRegions(),.01,law=law)
    free=np.tile(~system.fixed,3)
    k=system.stiffness[free][:,free].toarray()
    m=consistent_mass(system,1040.)[free][:,free].toarray()
    lam,phi=eigh(k,m)
    timed_force=applied(study,small,system.local,system.skin,.01)[0].T.ravel()[free]
    times=np.linspace(0.,duration,401)
    states=(1.-np.cos(times[:,None]*np.sqrt(lam)))*((phi.T@timed_force)/lam)
    basis=np.zeros((len(free),len(lam)));basis[free]=phi
    exact,_=stress_envelope(system,basis,states)
    np.testing.assert_allclose(history['peak_MPa'],exact.max(),rtol=.012)
    reversal=HistoryCase(8.e-5,4.e-6,(TimeExcitation(load,np.array([0.,3.e-5,6.e-5,8.e-5]),
                         np.array([0.,.42,0.,0.])),),relative_tolerance=.04,max_time_levels=7)
    yielded=export_history(system,study,reversal,path/'yielded',spsolve,ReferenceRegions(),.01,law=FiniteStrain(3.,50.))
    assert yielded['peak_equivalent_plastic_strain']>.005
    with np.load(path/'yielded.npz') as evidence:assert evidence['equivalent_plastic_strain'].max()>.005
    carrier=RigidCarrier(np.array([0.,.5,.5]),np.array([0.,.5,.5]),.002,np.diag([.005,.005,.005]),
                         np.array([-7.,0.,0.]),np.zeros(3))
    normal=SurfaceLoad(right,np.array([1.,0.,0.]))
    from gso_workflows.shock_receivers import AccelerationReceiver
    receiver=AccelerationReceiver('carrier',(left,),(1.,0.,0.))
    drop=HistoryCase(.0016,.000025,(),contacts=(FoundationContact(normal,.0001,25.),),carrier=carrier,receivers=(receiver,),
                     relative_tolerance=.04,max_time_levels=6)
    impact=export_history(system,study,drop,path/'impact',spsolve,ReferenceRegions(),.01,law=FiniteStrain(2.,10.))
    assert impact[receiver.metric]>0.
    assert impact['peak_MPa']>0. and impact['contact_histories'][0]['impulse_Ns']>0.
    assert impact['contact_histories'][0]['first_release_s'] is not None and impact['final_linear_momentum_Ns'][0]>0.
    np.testing.assert_allclose(impact['momentum_balance_Ns'],0.,atol=1.e-10)
    assert impact['maximum_equilibrium_residual']<=law.equilibrium_tolerance
    print('Finite extension, unloading, modal limit and nonlinear contact passed',flush=True)

    # Doubling both stiffness and density preserves modal frequencies. For a fixed
    # force the displacement halves and the small-load stress field stays equal.
    from unittest.mock import patch
    from gso_workflows.study_materials import IsotropicMaterial,material_portfolio
    from gso_workflows.surface_search import evaluate
    proto=replace(study,loads={'pull':small},histories={'pulse':case},solid_laws={'pulse':law})
    soft=IsotropicMaterial('soft',100.,0.,1040.)
    hard=IsotropicMaterial('hard',200.,0.,2080.)
    portfolio=material_portfolio(proto,(soft,hard))
    with patch('gso_workflows.surface_search.volume_mesh',return_value=(points,cells)):
        actual=evaluate(portfolio,mesh,.5,.04,path,spsolve,'materials')
    for row in actual.responses:
        a,b,c,d=(dict(zip(portfolio.scenarios(),row,strict=True))[key] for key in
                  ('soft / pull','hard / pull','soft / pulse','hard / pulse'))
        np.testing.assert_allclose(a['compliance_Nmm'],2.*b['compliance_Nmm'],rtol=1.e-9)
        np.testing.assert_allclose(c['peak_relative_displacement_mm'],2.*d['peak_relative_displacement_mm'],rtol=.002)
        np.testing.assert_allclose(c['peak_MPa'],d['peak_MPa'],rtol=.002)
    print('Material-specific stiffness and inertia reach every verification level',flush=True)
