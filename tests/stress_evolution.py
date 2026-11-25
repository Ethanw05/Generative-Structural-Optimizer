"""Independent differential and load-rotation checks for the actual cubature/adjoint search path."""
import sys
import tempfile
from itertools import product
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'build/sunbeam33/deps'),str(root/'tools')]
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from gso_workflows.stress_regions import support_edges
from gso_workflows.stress_evolution import evaluate_design,refine

shape=(4,3,3)
nodes=np.array(shape)+1
cells=np.array(list(product(*(range(i) for i in shape))))
corners=np.array(list(product(range(2),repeat=3)))
ids=np.ravel_multi_index((cells[:,None]+corners).transpose(2,0,1),nodes)
points=np.array(list(product(*(range(i) for i in nodes))))
index=np.cumsum(points[:,0]>0)-1
index[points[:,0]==0]=-1
conn=index[ids];n=3*(index.max()+1)
dof=(3*conn[:,:,None]+np.arange(3)).reshape(len(cells),24)
row=np.broadcast_to(dof[:,:,None],(len(cells),24,24))
col=np.broadcast_to(dof[:,None,:],row.shape)
valid=(row>=0)&(col>=0)
with tempfile.TemporaryDirectory(dir=root/'build') as temp:
    def kernel(young,nu):
        path=Path(temp)/'stiffness.txt'
        subprocess.run([sys.argv[1],'mls-stencil','1',str(young),str(nu),path],check=True)
        return np.loadtxt(path)
    ke=kernel(2200.,.35)
    shear=2200/(2*1.35)
    sm=shear**2*(5*kernel(2,0)-2*kernel(2.5,.25))
def assemble(weights):
    entries=(weights[:,None,None]*ke)[valid]
    return sparse.coo_matrix((entries,(row[valid],col[valid])),shape=(n,n)).tocsr()
def case(axis):
    f=np.zeros(n)
    end=index[points[:,0]==shape[0]]
    f[3*end+axis]=1./len(end)
    return {'end-load':f}
edges=support_edges(np.ones(shape,bool))
def evaluate(x,axis):
    return evaluate_design(x,ke,sm,conn,assemble,spsolve,case(axis),3.5,edges)
rng=np.random.default_rng(72)
x=rng.uniform(.35,.9,len(cells));d=rng.normal(size=x.size);d/=np.linalg.norm(d)
r=evaluate(x,2)['end-load']
def metrics(r):return np.array([r.compliance,r.stress_mpa,r.regional_mpa])
fd=(metrics(evaluate(x+1e-5*d,2)['end-load'])-metrics(evaluate(x-1e-5*d,2)['end-load']))/2e-5
np.testing.assert_allclose(fd,[r.compliance_gradient@d,r.stress_gradient@d,r.regional_gradient@d],rtol=1e-5,atol=1e-8)
free=np.ones(shape,bool)
protected=(cells[:,0]==3)&(cells[:,1]==1)&(cells[:,2]==1)
initial=np.full(len(cells),.62);initial[protected]=1.
def response_to(axis):
    history=[]
    def record(i,x,rs):history.append(metrics(rs['end-load']))
    final=refine(initial.copy(),free,protected,.55,lambda x:evaluate(x,axis),6,record)
    assert len(history)>1 and np.all(history[-1]<history[0])
    np.testing.assert_allclose(final[protected],1.)
    np.testing.assert_allclose(final.sum(),initial.sum(),rtol=1e-12)
    return final.reshape(shape),history
vertical,history=response_to(2);lateral,_=response_to(1)
np.testing.assert_allclose(vertical,lateral.swapaxes(1,2),atol=1e-7)
assert np.max(abs(vertical-lateral))>.05
print('Relative change in compliance and stress:',history[-1]/history[0]-1)
print('Max field difference after changing load direction:',np.max(abs(vertical-lateral)))
