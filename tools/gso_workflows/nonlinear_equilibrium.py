"""Incremental equilibrium. State is committed only after residual convergence."""
from dataclasses import dataclass
import numpy as np
from .finite_solid import UnresolvedSolid


@dataclass
class NewtonResult:
    coordinates: np.ndarray
    field: object
    residual: float
    iterations: int


def equilibrium(initial, evaluate, solve, force_scale, tolerance, max_iterations=32, predictor=None):
    x=initial.copy()
    scale=max(float(force_scale),np.finfo(float).tiny)
    threshold=tolerance*scale
    residual,_,field=evaluate(x,False)
    chord=predictor
    for iteration in range(max_iterations+1):
        norm=np.linalg.norm(residual)
        if not np.isfinite(norm):raise UnresolvedSolid('Nonfinite nonlinear equilibrium residual')
        if norm<=threshold:return NewtonResult(x,field,norm/scale,iteration)
        if iteration==max_iterations:break
        if chord is None:
            residual,tangent,_=evaluate(x,True)
            dx=np.asarray(solve(tangent,-residual),dtype=float)
        else:
            dx=np.asarray(chord(-residual),dtype=float)
        if dx.shape!=x.shape or not np.isfinite(dx).all():raise UnresolvedSolid('Invalid nonlinear direction')
        for cut in range(16):
            candidate=x+2.**(-cut)*dx
            try:trial,_,next_field=evaluate(candidate,False)
            except UnresolvedSolid:continue
            trial_norm=np.linalg.norm(trial)
            if trial_norm<=(1.-1.e-4*2.**(-cut))*norm:
                x,residual,field=candidate,trial,next_field
                if trial_norm>.5*norm or iteration>=4:chord=None
                break
        else:
            if chord is None:raise UnresolvedSolid('No admissible residual-reducing Newton step')
            chord=None
    raise UnresolvedSolid('Nonlinear equilibrium iteration limit reached')


def static_path(solid, forces, solve):
    """Advance a prescribed load path, preserving the material history through reversals."""
    free=np.tile(~solid.system.fixed,3)
    if not free.any() or free.all():raise ValueError('The static model needs a fixture and a free degree of freedom')
    from .sparse_solve import fixed_matrix_solver
    with fixed_matrix_solver(solid.system.stiffness[free][:,free].tocsr(),solve) as predictor:
        return _advance_loads(solid,forces,solve,free,predictor)


def _advance_loads(solid,forces,solve,free,predictor):
    reference=np.zeros(len(free))
    state=solid.virgin()
    x=np.zeros(free.sum())
    records=[]
    def full(y):
        q=np.zeros(len(free))
        q[free]=y
        return q
    for target in forces:
        target=np.asarray(target,dtype=float)
        if target.shape!=reference.shape or not np.isfinite(target).all():raise ValueError('Invalid nonlinear load')
        fraction=0.
        increment=1.
        steps=0
        scale=max(np.linalg.norm(reference[free]),np.linalg.norm(target[free]),np.finfo(float).tiny)
        while fraction<1.:
            end=min(1.,fraction+increment)
            load=reference+(target-reference)*end
            def residual(y,tangent):
                field=solid.evaluate(full(y),state,tangent)
                return (field.internal-load)[free],field.tangent[free][:,free] if tangent else None,field
            try:result=equilibrium(x,residual,solve,scale,solid.law.equilibrium_tolerance,predictor=predictor)
            except UnresolvedSolid:
                increment=.5*(end-fraction)
                if increment<solid.law.minimum_increment:raise
                continue
            x,state=result.coordinates,result.field.state
            fraction=end
            steps+=1
            increment=min(1.,2.*increment)
        records.append((full(x),result.field,result.residual,steps))
        reference=target
    return records
