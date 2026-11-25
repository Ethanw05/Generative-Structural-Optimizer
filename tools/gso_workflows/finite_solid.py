"""Total-Lagrangian tetrahedra using the native finite-strain material operator."""
import ctypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
from scipy import sparse


class UnresolvedSolid(ValueError):
    """A candidate failed equilibrium, time refinement, or positive material volume."""


@dataclass(frozen=True)
class FiniteStrain:
    initial_yield_mpa: float | None = None
    hardening_mpa: float = 0.
    equilibrium_tolerance: float = 1.e-7
    minimum_increment: float = 1./4096.

    def __post_init__(self):
        if self.initial_yield_mpa is not None and not (np.isfinite(self.initial_yield_mpa) and self.initial_yield_mpa>0.):
            raise ValueError('The initial Kirchhoff yield stress must be positive')
        if not np.isfinite(self.hardening_mpa) or self.hardening_mpa<0. or (self.initial_yield_mpa is None and self.hardening_mpa!=0.):
            raise ValueError('Isotropic hardening requires a nonnegative modulus and a yield stress')
        if not (0.<self.equilibrium_tolerance<1. and 0.<self.minimum_increment<=1.):
            raise ValueError('Invalid nonlinear equilibrium controls')

    def description(self):
        return dict(kinematics='finite strain',material='Hencky elastic' if self.initial_yield_mpa is None
                    else 'multiplicative J2 with linear isotropic hardening',
                    initial_Kirchhoff_yield_MPa=self.initial_yield_mpa,hardening_MPa=self.hardening_mpa)


@lru_cache(maxsize=None)
def _operator(executable):
    name={'win32':'gso_mechanics.dll','darwin':'libgso_mechanics.dylib'}.get(sys.platform,'libgso_mechanics.so')
    library=ctypes.CDLL(str(Path(executable).resolve().with_name(name)))
    op=library.gsoSolidPoints
    op.argtypes=[ctypes.c_size_t]+[ctypes.c_void_p]*7
    op.restype=ctypes.c_int
    return op


@dataclass
class EquilibriumField:
    internal: np.ndarray
    tangent: object
    state: np.ndarray
    measures: np.ndarray
    first_piola: np.ndarray | None


class FiniteSolid:
    def __init__(self, system, law, executable, *, capture_stress=False):
        self.system=system
        self.law=law
        self.capture_stress=capture_stress
        self.properties=np.array([system.shear,system.lame+2.*system.shear/3.,
                                  law.initial_yield_mpa or 0.,law.hardening_mpa])
        self.evaluate_points=_operator(executable)
        self.size=len(system.points)
        self.rows=np.repeat(system.cells,4,axis=1).ravel()
        self.cols=np.tile(system.cells,(1,4)).ravel()

    def virgin(self):
        return np.tile(np.r_[np.eye(3).ravel(),0.],(len(self.system.cells),1))

    def evaluate(self, displacement, committed, tangent):
        system=self.system
        count=len(system.cells)
        displacement=np.asarray(displacement,dtype=float)
        committed=np.ascontiguousarray(committed,dtype=float)
        if displacement.shape!=(3*self.size,) or committed.shape!=(count,10):
            raise ValueError('The displacement or material state belongs to a different solid')
        nodal=displacement.reshape(3,self.size).T
        internal=np.zeros((self.size,3))
        state=np.empty_like(committed)
        measures=np.empty((count,2))
        stresses=np.empty((count,3,3)) if self.capture_stress else None
        blocks=np.empty((3,3,count,4,4)) if tangent else None
        for first in range(0,count,4096):
            sl=slice(first,min(count,first+4096))
            nodes=system.cells[sl]
            grad=system.gradients[sl]
            volume=system.volume[sl]
            f=np.ascontiguousarray(np.eye(3)+nodal[nodes].transpose(0,2,1)@grad)
            piola=np.empty_like(f)
            derivative=np.empty((len(f),3,3,3,3)) if tangent else None
            status=self.evaluate_points(len(f),self.properties.ctypes.data,f.ctypes.data,committed[sl].ctypes.data,
                                        state[sl].ctypes.data,piola.ctypes.data,measures[sl].ctypes.data,
                                        derivative.ctypes.data if tangent else None)
            if status==2:raise UnresolvedSolid('A trial deformation inverted a material element')
            if status:raise ValueError('The native finite-strain operator rejected a material state')
            if stresses is not None:stresses[sl]=piola
            force=volume[:,None,None]*(grad@piola.transpose(0,2,1))
            for node in range(4):
                np.add.at(internal,nodes[:,node],force[:,node])
            if tangent:
                local=np.einsum('eaj,eijml,ebl,e->imeab',grad,derivative,grad,volume,optimize=True)
                blocks[:,:,sl]=local
        matrix=None
        if tangent:
            parts=[[sparse.coo_matrix((block.ravel(),(self.rows,self.cols)),
                                      shape=(self.size,self.size)).tocsr() for block in row] for row in blocks]
            matrix=sparse.bmat(parts,format='csr')
        return EquilibriumField(internal.T.ravel(),matrix,state,measures,stresses)
