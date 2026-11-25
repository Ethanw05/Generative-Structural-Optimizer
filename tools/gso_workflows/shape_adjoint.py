"""Discrete P1 shape derivative of a volume-normalized von Mises p-mean."""
import numpy as np
from .stress_regions import regional_stress
from .reference_regions import relative_stress


def stress_shape_gradient(points,tets,grad,vol,du,shear,lame,free,matrix,solver,power=8.,basins=None,reference=None):
    eye=np.eye(3)
    strain=.5*(du+du.transpose(0,2,1))
    trace=np.trace(strain,axis1=1,axis2=2)[:,None,None]
    stress=2.*shear*strain+lame*trace*eye
    deviator=2.*shear*(strain-trace*eye/3.)
    vm=np.sqrt(1.5*np.einsum('eij,eij->e',deviator,deviator))
    maximum=float(vm.max())
    if maximum==0.:return 0.,np.zeros_like(points)
    total=vol.sum()
    if basins is None:
        metric=maximum*(np.average((vm/maximum)**power,weights=vol))**(1./power)
        partial=(vm/metric)**(power-1.)/total
        direct=metric*((vm/metric)**power-1.)/(power*total)
    elif reference is None:
        metric,partial,direct=regional_stress(vm,vol,basins,power)
    else:
        metric,partial,direct=relative_stress(vm,vol,basins,reference,power)
    factor=np.divide(3.*shear*partial,vm,out=np.zeros_like(vm),where=vm>0.)
    sensitivity=factor[:,None,None]*deviator
    rhs=np.zeros_like(points)
    local_rhs=vol[:,None,None]*(grad@sensitivity)
    np.add.at(rhs,tets.ravel(),local_rhs.reshape(-1,3))
    vector=rhs.T.ravel();adjoint=np.zeros_like(vector)
    adjoint[free]=solver(matrix,vector[free])
    err=np.linalg.norm(matrix@adjoint[free]-vector[free])
    if err>1.e-6*max(np.linalg.norm(vector[free]),1.e-30):
        raise ValueError('Stress shape adjoint failed equilibrium')
    dz=adjoint.reshape(3,-1).T[tets].transpose(0,2,1)@grad
    ez=.5*(dz+dz.transpose(0,2,1))
    sz=2.*shear*ez+lame*np.trace(ez,axis1=1,axis2=2)[:,None,None]*eye
    lagrangian=direct-np.einsum('eij,eij->e',stress,ez)
    tensor=lagrangian[:,None,None]*eye+du.transpose(0,2,1)@(sz-sensitivity)+dz.transpose(0,2,1)@stress
    derivative=vol[:,None,None]*(grad@tensor.transpose(0,2,1))
    by_node=np.zeros_like(points)
    np.add.at(by_node,tets.ravel(),derivative.reshape(-1,3))
    return metric,by_node


def normal_descent(points,skin,derivative):
    triangles=points[skin]
    n=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    area=.5*np.linalg.norm(n,axis=1)
    n/=2.*area[:,None]
    nodal_area=np.zeros(len(points))
    np.add.at(nodal_area,skin.ravel(),np.repeat(area/3.,3))
    shape_force=np.divide(derivative,nodal_area[:,None],out=np.zeros_like(derivative),where=nodal_area[:,None]>0.)
    return -np.einsum('fi,fi->f',shape_force[skin].mean(1),n)
