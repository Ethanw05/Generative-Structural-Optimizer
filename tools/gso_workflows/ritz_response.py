"""Mass-orthogonal load-dependent Ritz reduction for a constrained elastic component."""
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from .sparse_solve import fixed_matrix_solver


def consistent_mass(system,density_kg_m3):
    if not np.isfinite(density_kg_m3) or density_kg_m3<=0.:
        raise ValueError('Material density must be positive')
    t=system.cells
    # N, mm, s. A kilogram has dynamic mass 0.001 N s²/mm.
    weights=density_kg_m3*1.e-12*system.volume/20.
    blocks=weights[:,None,None]*(np.ones((4,4))+np.eye(4))
    indices=(np.repeat(t,4,axis=1).ravel(),np.tile(t,(1,4)).ravel())
    scalar=sparse.coo_matrix((blocks.ravel(),indices),shape=(len(system.points),)*2).tocsr()
    return sparse.block_diag([scalar]*3,format='csr')


def ritz_spaces(stiffness,mass_apply,spatial_forces,solve,max_orders):
    """Grow nested spaces by inverse iteration driven by each load and its inertia."""
    if max_orders<1 or spatial_forces.ndim!=2 or not spatial_forces.shape[1]:
        raise ValueError('At least one spatial load or contact direction is required')
    with fixed_matrix_solver(stiffness,solve) as solve_block:
        yield from _ritz_blocks(stiffness,mass_apply,spatial_forces,solve_block,max_orders)


def _ritz_blocks(stiffness,mass_apply,spatial_forces,solve_block,max_orders):
    basis=np.empty((stiffness.shape[0],0))
    mass_basis=basis.copy()
    rhs=spatial_forces
    for order in range(max_orders):
        block=solve_block(rhs)
        residual=np.linalg.norm(stiffness@block-rhs,axis=0)
        if np.any(residual>1.e-6*np.maximum(np.linalg.norm(rhs,axis=0),np.finfo(float).tiny)):
            raise ValueError('Ritz elastic solve failed equilibrium')
        # Scale each input separately. A force and a moment have different units.
        mb=mass_apply(block)
        norm=np.sqrt(np.maximum(0.,np.einsum('ij,ij->j',block,mb)))
        live=norm>0.
        if not np.any(live):return
        block=block[:,live]/norm[live]
        mb=mb[:,live]/norm[live]
        for _ in range(2):
            coeff=basis.T@mb
            block-=basis@coeff
            mb=mass_apply(block)
        gram=block.T@mb
        value,axes=eigh((gram+gram.T)/2.)
        live=value>np.finfo(float).eps*max(1.,value.max())*len(value)*8.
        if not np.any(live):return
        added=block@(axes[:,live]/np.sqrt(value[live]))
        rhs=mass_apply(added)
        basis=np.column_stack((basis,added))
        mass_basis=np.column_stack((mass_basis,rhs))
        yield basis,mass_basis


def stress_envelope(system,basis,histories,recovery=None):
    """Recover the maximum elemental von Mises stress and its time sample."""
    count=len(system.cells)
    envelope=np.zeros(count)
    instant=np.zeros(count,dtype=np.intp)
    modes=basis.reshape(3,len(system.points),-1).transpose(1,0,2)
    # Keep the temporary stress history below approximately 32 MiB.
    block_size=max(1,min(4096,4_000_000//(6*len(histories))))
    if recovery is not None:
        block_size=max(1,min(4096,1_000_000//basis.shape[1]))
    for left in range(0,count,block_size):
        region=slice(left,min(left+block_size,count))
        gradient=np.einsum('eicm,eij->ecjm',modes[system.cells[region]],system.gradients[region],optimize=True)
        solid=system.shear*(gradient+gradient.swapaxes(1,2))
        trace=solid[:,0,0]+solid[:,1,1]+solid[:,2,2]
        columns=[(solid[:,i,i]-trace/3.)*np.sqrt(1.5) for i in range(3)]
        columns.extend(solid[:,a,b]*np.sqrt(3.) for a,b in ((0,1),(0,2),(1,2)))
        coefficients=np.stack(columns,axis=1)
        if recovery is not None:
            envelope[region],instant[region]=recovery.measure(coefficients)
            continue
        values=(coefficients.reshape(-1,basis.shape[1])@histories.T).reshape(len(coefficients),6,-1)
        squared=np.einsum('eit,eit->et',values,values)
        sample=np.argmax(squared,axis=1)
        envelope[region]=np.sqrt(squared[np.arange(len(sample)),sample])
        instant[region]=sample
    return envelope,instant
