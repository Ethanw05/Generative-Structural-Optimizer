"""Exact directional envelopes for a plane of normalized linear static load combinations."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class QuadraticPlane:
    first: object
    second: object
    cross: object

    def at(self,radians):
        c,s=np.cos(radians),np.sin(radians)
        return c*c*self.first+2.*s*c*self.cross+s*s*self.second

    def maximum(self):
        # Largest eigenpair of the symmetric two-by-two Gram matrix at every element.
        value=.5*(self.first+self.second)+np.hypot(.5*(self.first-self.second),self.cross)
        angle=np.mod(.5*np.arctan2(2.*self.cross,self.first-self.second),np.pi)
        return value,angle


def equivalent_stress_squared(first,second):
    if first.ndim!=3 or first.shape[1:]!=(3,3) or second.shape!=first.shape or not len(first):
        raise ValueError('One symmetric deviatoric stress tensor per common element is required')
    for field in (first,second):
        scale=max(1.,float(np.max(np.abs(field))))
        if (not np.isfinite(field).all() or np.max(np.abs(field-field.transpose(0,2,1)))>1.e-10*scale
            or np.max(np.abs(field.trace(axis1=1,axis2=2)))>1.e-10*scale):
            raise ValueError('Expected finite symmetric deviatoric stresses in common axes')
    def product(a,b):
        return 1.5*np.einsum('ijk,ijk->i',a,b)
    return QuadraticPlane(product(first,first),product(second,second),product(first,second))


def compliance_plane(load_a,motion_a,load_b,motion_b):
    if not (load_a.shape==motion_a.shape==load_b.shape==motion_b.shape):
        raise ValueError('Force and displacement must use the same degrees of freedom')
    aa=float(np.vdot(load_a,motion_a))
    bb=float(np.vdot(load_b,motion_b))
    ab=float(np.vdot(load_a,motion_b))
    ba=float(np.vdot(load_b,motion_a))
    if not np.isfinite([aa,bb,ab,ba]).all() or min(aa,bb)<0.:
        raise ValueError('Nonfinite or negative elastic work')
    tolerance=1.e-7*max(np.sqrt(aa*bb),np.finfo(float).tiny)
    if abs(ab-ba)>tolerance or max(abs(ab),abs(ba))>np.sqrt(aa*bb)+tolerance:
        raise ValueError('The solutions violate elastic reciprocity or positive semidefiniteness')
    return QuadraticPlane(aa,bb,.5*(ab+ba))
