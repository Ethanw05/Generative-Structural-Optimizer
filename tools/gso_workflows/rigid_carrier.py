"""Six rigid-body coordinates coupled inertially to an elastic component."""
from dataclasses import dataclass
import numpy as np


def rigid_modes(points,origin):
    offsets=points-np.asarray(origin)
    basis=np.zeros((len(points),3,6))
    basis[:,:,:3]=np.eye(3)
    for axis in range(3):
        basis[:,:,axis+3]=np.cross(np.eye(3)[axis],offsets)
    return basis.transpose(1,0,2).reshape(3*len(points),6)


@dataclass(frozen=True)
class RigidCarrier:
    origin_world_mm: np.ndarray
    center_world_mm: np.ndarray
    mass_kg: float
    inertia_center_kg_mm2: np.ndarray
    velocity_world_mm_s: np.ndarray
    angular_velocity_world_rad_s: np.ndarray

    def spatial_mass(self):
        inertia=np.asarray(self.inertia_center_kg_mm2,dtype=float)
        if (not np.isfinite(self.mass_kg) or self.mass_kg<0. or inertia.shape!=(3,3)
                or not np.allclose(inertia,inertia.T) or np.linalg.eigvalsh(inertia).min() < 0.):
            raise ValueError('Invalid rigid assembly mass or central inertia')
        h=rigid_modes(np.asarray(self.center_world_mm)[None,:],self.origin_world_mm)
        mass=self.mass_kg*(h.T@h)
        mass[3:,3:]+=inertia
        return mass*1.e-3


@dataclass(frozen=True)
class RigidContact:
    point_world_mm: np.ndarray
    normal_world: np.ndarray
    gap_mm: float
    stiffness_n_mm: float

    def direction(self,origin):
        if not np.isclose(np.linalg.norm(self.normal_world),1.):
            raise ValueError('A carrier contact needs a unit plane normal')
        return self.normal_world@rigid_modes(np.asarray(self.point_world_mm)[None,:],origin)
