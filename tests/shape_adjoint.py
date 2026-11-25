"""Verify the complete realized-shape gradient against geometry perturbation and re-equilibration."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'build/sunbeam33/deps'),str(root/'tools')]
import numpy as np
from gso_workflows.shape_adjoint import stress_shape_gradient

p=np.array([[0,0,0],[0,1,0],[0,0,1],[1.2,.2,.1],[1.5,.8,.7]],float)
t=np.array([[0,1,2,3],[1,2,3,4]])
free=np.tile(np.arange(5)>=3,3)
rhs=np.zeros(15);rhs[4]=1.7;rhs[9]=-.6;rhs[14]=2.1
mu,lam=720.,1130.
def response(points, basins=None, reference=None):
    jac=points[t[:,1:]]-points[t[:,:1]]
    vol=abs(np.linalg.det(jac))/6.
    g=np.linalg.inv(jac).transpose(0,2,1)
    g=np.concatenate((-g.sum(1)[:,None,:],g),axis=1)
    k=np.zeros((15,15))
    for e,ids in enumerate(t):
        for a in range(4):
            for b in range(4):
                block=(mu*np.eye(3)*(g[e,a]@g[e,b])+lam*np.outer(g[e,a],g[e,b])
                       +mu*np.outer(g[e,b],g[e,a]))*vol[e]
                k[np.ix_(ids[a]+5*np.arange(3),ids[b]+5*np.arange(3))]+=block
    mat=k[np.ix_(free,free)]
    u=np.zeros(15);u[free]=np.linalg.solve(mat,rhs[free])
    du=u.reshape(3,5).T[t].transpose(0,2,1)@g
    return stress_shape_gradient(points,t,g,vol,du,mu,lam,free,mat,np.linalg.solve,basins=basins,reference=reference)
rng=np.random.default_rng(57)
for basins,reference in ((None,None), (np.array([0, 1]),None), (np.array([0,1]),np.array([2.,3.]))):
    value,gradient=response(p,basins,reference)
    for i in range(8):
        direction=rng.normal(size=p.shape)
        epsilon=1.e-6
        measured=(response(p+epsilon*direction,basins,reference)[0]-response(p-epsilon*direction,basins,reference)[0])/(2.*epsilon)
        predicted=np.sum(gradient*direction)
        np.testing.assert_allclose(predicted,measured,rtol=2.e-7,atol=5.e-8)
    np.testing.assert_allclose(gradient.sum(0),0.,atol=1.e-11)
print('Global, regional and reference-normalized adjoints matched 24 re-equilibrated directional differences.')
