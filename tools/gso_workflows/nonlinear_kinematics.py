"""Exact rigid attachment maps for total-Lagrangian solids and plane-contact points."""
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation


def cross_matrix(v):
    v=np.asarray(v)
    out=np.zeros(v.shape[:-1]+(3,3))
    out[...,0,1]=-v[...,2];out[...,0,2]=v[...,1]
    out[...,1,0]=v[...,2];out[...,1,2]=-v[...,0]
    out[...,2,0]=-v[...,1];out[...,2,1]=v[...,0]
    return out


def rigid_inertia_points(carrier):
    """Six masses whose total, center and inertia tensor equal the supplied rigid body."""
    carrier.spatial_mass()
    mass=carrier.mass_kg
    inertia=np.asarray(carrier.inertia_center_kg_mm2)
    if mass==0.:
        if np.any(inertia):raise ValueError('A massless rigid carrier cannot have independent inertia')
        return np.empty((0,3)),np.empty(0)
    cov=(.5*np.trace(inertia)*np.eye(3)-inertia)/mass
    values,axes=np.linalg.eigh(cov)
    if values.min() < -1.e-12*max(1.,abs(values).max()):
        raise ValueError('The carrier principal inertias violate the triangle inequality')
    offsets=(axes*np.sqrt(3.*np.maximum(0.,values))).T
    return carrier.center_world_mm+np.vstack((offsets,-offsets)),np.full(6,mass*.001/6.)


class AttachmentMap:
    def __init__(self,system,case):
        self.carrier=case.carrier
        self.free=np.flatnonzero(np.tile(~system.fixed,3))
        n=len(system.points)
        self.solid_dofs=3*n
        if self.carrier is None:
            if case.rigid_contacts:raise ValueError('A rigid contact needs a carrier')
            extra=np.empty((0,3))
            self.masses=np.empty(0)
        else:
            extra,self.masses=rigid_inertia_points(self.carrier)
            extra=np.vstack((extra,np.asarray([c.point_world_mm for c in case.rigid_contacts]).reshape(-1,3)))
        self.extra=len(extra)
        self.size=3*(n+self.extra)
        fixed=np.flatnonzero(system.fixed)
        self.rows=np.vstack((np.column_stack([fixed+i*n for i in range(3)]),
                             3*n+np.arange(3*self.extra).reshape(-1,3)))
        self.offsets=(np.vstack((system.points[fixed],extra))-self.carrier.origin_world_mm
                      if self.carrier is not None else np.empty((0,3)))
        nq=len(self.free)+(6 if self.carrier is not None else 0)
        self.direct=sparse.csr_matrix((np.ones(len(self.free)),(self.free,np.arange(len(self.free)))),
                                      shape=(self.size,nq))

    def rotation(self,theta,old):
        return Rotation.from_rotvec(theta).as_matrix()@old

    def rotational_derivative(self,theta,old):
        angle=np.linalg.norm(theta)
        skew=cross_matrix(theta)
        if angle<1.e-4:
            a=.5-angle**2/24.;b=1./6.-angle**2/120.
        else:
            a=(1.-np.cos(angle))/angle**2;b=(angle-np.sin(angle))/angle**3
        left=np.eye(3)+a*skew+b*(skew@skew)
        arms=self.offsets@self.rotation(theta,old).T
        return -cross_matrix(arms)@left

    def position(self,q,old):
        out=np.zeros(self.size)
        out[self.free]=q[:len(self.free)]
        if self.carrier is not None:
            out[self.rows]=q[-6:-3]+self.offsets@(self.rotation(q[-3:],old)-np.eye(3)).T
        return out

    def jacobian(self,q,old):
        if self.carrier is None:return self.direct
        dr=self.rotational_derivative(q[-3:],old)
        blocks=np.concatenate((np.broadcast_to(np.eye(3),dr.shape),dr),axis=2)
        rows=np.broadcast_to(self.rows[:,:,None],blocks.shape).ravel()
        cols=np.broadcast_to(len(self.free)+np.arange(6),blocks.shape).ravel()
        return self.direct+sparse.coo_matrix((blocks.ravel(),(rows,cols)),shape=self.direct.shape).tocsr()

    def curvature(self,q,old,force):
        if self.carrier is None:return 0.
        # Only three rotation coordinates have curvature. Differentiate their
        # analytical velocity map with a centered perturbation in radians.
        step=np.cbrt(np.finfo(float).eps)
        raw=np.empty((3,3))
        for axis in range(3):
            delta=np.eye(3)[axis]*step
            change=self.rotational_derivative(q[-3:]+delta,old)-self.rotational_derivative(q[-3:]-delta,old)
            raw[:,axis]=np.einsum('nij,ni->j',change,force[self.rows])/(2.*step)
        start=len(q)-3
        rows,cols=np.indices((3,3))
        return sparse.coo_matrix((.5*(raw+raw.T).ravel(),(start+rows.ravel(),start+cols.ravel())),
                                 shape=(len(q),len(q))).tocsr()
