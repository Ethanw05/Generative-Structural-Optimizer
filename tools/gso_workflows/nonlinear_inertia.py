"""Mass, dead loads and unilateral contact in the common ambient coordinate system."""
import numpy as np
from scipy import sparse
from .ritz_response import consistent_mass
from .tetra_elasticity import applied


class DynamicPorts:
    def __init__(self,system,study,case,mapping,tolerance):
        n=mapping.solid_dofs
        def port(action):
            f=applied(study,action,system.local,system.skin,tolerance)[0].T.ravel()
            return sparse.csr_matrix(f[None,:])
        mass=consistent_mass(system,study.density_kg_m3)
        for item in case.inertias:
            h=sparse.vstack([port(load) for load in item.unit_loads],format='csr')
            m=np.asarray(item.matrix,dtype=float)
            if (m.shape!=(h.shape[0],)*2 or not np.isfinite(m).all()
                    or not np.allclose(m,m.T) or np.linalg.eigvalsh(m).min()<0.):
                raise ValueError('Invalid attached inertia')
            mass+=h.T@sparse.csr_matrix(m)@h
        extra_mass=np.zeros(mapping.extra)
        extra_mass[:len(mapping.masses)]=mapping.masses
        self.extra=sparse.csr_matrix((3*mapping.extra,)*2)
        self.mass=sparse.block_diag((mass,sparse.diags(np.repeat(extra_mass,3))),format='csr')
        self.loads=sparse.vstack([port(load.attachment) for load in case.loads],format='csr') if case.loads else sparse.csr_matrix((0,n))
        self.loads=sparse.hstack((self.loads,sparse.csr_matrix((self.loads.shape[0],mapping.size-n))),format='csr')
        contacts=[]
        for contact in case.contacts:
            if not np.isclose(np.linalg.norm(contact.unit_normal_load.force_n),1.):
                raise ValueError('Contact needs a unit normal force')
            contacts.append(sparse.hstack((port(contact.unit_normal_load),sparse.csr_matrix((1,mapping.size-n)))))
        for i,c in enumerate(case.rigid_contacts):
            c.direction(case.carrier.origin_world_mm)
            cols=n+3*(len(mapping.masses)+i)+np.arange(3)
            contacts.append(sparse.csr_matrix((c.normal_world,(np.zeros(3,dtype=int),cols)),shape=(1,mapping.size)))
        self.normals=sparse.vstack(contacts,format='csr') if contacts else sparse.csr_matrix((0,mapping.size))
        all_contacts=(*case.contacts,*case.rigid_contacts)
        self.gaps=np.asarray([c.gap_mm for c in all_contacts])
        self.spring=np.asarray([c.stiffness_n_mm for c in all_contacts])
        if (np.any(self.gaps<0.) or np.any(self.spring<=0.) or not np.isfinite(self.gaps).all()
                or not np.isfinite(self.spring).all()):raise ValueError('Invalid contact law')
        if len(case.damping)!=2 or any(not np.isfinite(x) or x<0. for x in case.damping):
            raise ValueError('Invalid Rayleigh damping')
        self.damping=case.damping[0]*self.mass+case.damping[1]*sparse.block_diag((system.stiffness,self.extra),format='csr')

    def load(self,case,time):
        factors=np.asarray([item.at(time) for item in case.loads])
        return self.loads.T@factors

    def contact(self,x):
        penetration=np.maximum(0.,-self.gaps-self.normals@x)
        return self.spring*penetration


class DynamicPreconditioner:
    """Reference tangent used only for predicting Newton steps; acceptance uses full nonlinear forces."""
    def __init__(self,mapping,ports,system,solve):
        self.jac=mapping.jacobian(np.zeros(mapping.direct.shape[1]),np.eye(3))
        self.stiffness=sparse.block_diag((system.stiffness,ports.extra),format='csr')
        self.ports=ports
        self.solve=solve
        self.factor=None
        self.step=None
        self.active=None

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        if self.factor is not None:self.factor.__exit__(*exc)

    def get(self,h,active):
        from .sparse_solve import fixed_matrix_solver
        if self.factor is not None:
            if abs(h-self.step)<=1.e-12*h and np.array_equal(active,self.active):return self.apply
            self.factor.__exit__(None,None,None)
            self.factor=None
        ports=self.ports
        contact=ports.normals.T@ports.normals.multiply((ports.spring*active)[:,None])
        operator=self.stiffness+contact+(4./h**2)*ports.mass+(2./h)*ports.damping
        matrix=(self.jac.T@operator@self.jac).tocsr()
        manager=fixed_matrix_solver(matrix,self.solve)
        self.apply=manager.__enter__()
        self.factor=manager
        self.step,self.active=h,active.copy()
        return self.apply
