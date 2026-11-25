"""Fixed-volume Pareto refinement of elastic stiffness and stress concentration across load cases."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter
from .stress_regions import ascent_basins, regional_stress

EQUILIBRIUM_TOLERANCE = 1.e-5


@dataclass(frozen=True)
class CaseResponse:
    compliance: float
    stress_mpa: float
    compliance_gradient: np.ndarray
    stress_gradient: np.ndarray
    regional_mpa: float
    regional_gradient: np.ndarray


def stress_response(density, local_u, stress_matrix, exponent=8.0, relaxation=0.5, edges=None):
    """Volume-normalized p-mean of density-relaxed RMS von Mises stress in equal-volume supports."""
    stress_times_u = local_u @ stress_matrix
    squared = np.maximum(np.einsum('ij,ij->i', local_u, stress_times_u), 0.)
    relaxed = density**relaxation * np.sqrt(squared)
    maximum = relaxed.max()
    if maximum == 0.:
        return 0., np.zeros_like(density), np.zeros_like(local_u)
    if edges is None:
        aggregate = maximum * np.mean((relaxed / maximum)**exponent)**(1./exponent)
        influence = (relaxed / aggregate)**(exponent-1.) / len(density)
        volume_partial = 0.
    else:
        # Upland differences smaller than the admissible equilibrium residual
        # cannot reliably choose a direction on an independently oriented solve.
        basins = ascent_basins(relaxed, *edges, relative_tolerance=EQUILIBRIUM_TOLERANCE)
        aggregate, partial, volume_partial = regional_stress(relaxed, density, basins, exponent)
        influence = density * partial
    direct = influence * relaxation * relaxed / density + volume_partial
    scale = np.divide(influence * density**relaxation, np.sqrt(squared),
                      out=np.zeros_like(density), where=squared > 0.)
    return aggregate, direct, scale[:, None] * stress_times_u


def evaluate_design(density, stiffness, stress_matrix, connections, assemble_matrix, solve, cases, power, edges=None):
    floor = 0.00012
    matrix = assemble_matrix(floor+(1.-floor)*density**power)
    derivative = (1.-floor)*power*density**(power-1.)
    responses = {}
    def equilibrium(rhs):
        result = solve(matrix, rhs)
        if not np.all(np.isfinite(result)) or np.linalg.norm(matrix @ result-rhs)>EQUILIBRIUM_TOLERANCE*np.linalg.norm(rhs):
            raise RuntimeError('Elastic equilibrium or adjoint failed')
        return result
    def gather(vector):
        with_zero = np.vstack((vector.reshape(-1,3), np.zeros((1,3))))
        return with_zero[connections].reshape(len(density), -1)
    for name, force in cases.items():
        u = equilibrium(force)
        local = gather(u)
        elastic_force = local @ stiffness
        compliance_grad = -derivative * np.einsum('ij,ij->i',local,elastic_force)
        def stress_adjoint(neighbors):
            stress, direct, displacement_grad = stress_response(density, local, stress_matrix, edges=neighbors)
            rhs = np.zeros((len(force)//3+1, 3))
            np.add.at(rhs, connections.ravel(), displacement_grad.reshape(-1,3))
            adjoint = gather(equilibrium(rhs[:-1].ravel()))
            sensitivity = direct-derivative*np.einsum('ij,ij->i', adjoint, elastic_force)
            return float(stress), sensitivity
        stress, stress_grad = stress_adjoint(None)
        regional, regional_grad = stress_adjoint(edges) if edges is not None else (stress, stress_grad)
        responses[name] = CaseResponse(float(force@u), stress, compliance_grad, stress_grad, regional, regional_grad)
    return responses


def projected_trial(density, cost, free, move):
    """Euclidean projection onto the original material budget and per-variable move bounds."""
    scale=np.max(np.abs(cost[free]))
    if scale==0.:
        return density.copy()
    delta=-move*cost[free]/scale
    x=density[free]
    lower=np.maximum(.015,x-move); upper=np.minimum(1.,x+move)
    lo,hi=np.min(lower-x-delta),np.max(upper-x-delta)
    for _ in range(55):
        shift=.5*(lo+hi)
        if np.clip(x+delta+shift,lower,upper).sum()>x.sum():
            hi=shift
        else:
            lo=shift
    trial=density.copy()
    trial[free]=np.clip(x+delta+.5*(lo+hi),lower,upper)
    return trial


def refine(density, allowed, protected, filter_sigma, evaluate, iterations, record, *, stiffness=True, maximum_stress=True):
    """Keep steps only after independent re-equilibration of every supplied load case."""
    if not np.any(~protected) or not np.all(np.isfinite(density)) or np.any(density<=0):
        raise ValueError('Invalid material field or empty design region')
    field=np.zeros(allowed.shape)
    weight=gaussian_filter(allowed.astype(float),filter_sigma,mode='constant')[allowed]
    responses=evaluate(density)
    if not (stiffness or maximum_stress):
        raise ValueError('Select at least one structural objective')
    def metrics(results):
        return np.array([([r.compliance] if stiffness else []) +
                         ([r.stress_mpa,r.regional_mpa] if maximum_stress else []) for r in results.values()])
    baseline=metrics(responses)
    if np.any(baseline<=0):
        raise ValueError('Every optimization scenario requires a nonzero structural response')
    record(0,density,responses)
    for generation in range(1,iterations+1):
        gradients=[]
        for i,r in enumerate(responses.values()):
            fields = ([r.compliance_gradient] if stiffness else []) + (
                [r.stress_gradient, r.regional_gradient] if maximum_stress else [])
            gradients.extend(g/scale for g,scale in zip(fields,baseline[i],strict=True))
        accepted=False
        # Search different stiffness/stress tradeoffs. Acceptance checks every individual response.
        fractions = (.5,.8,.2) if stiffness and maximum_stress else (float(maximum_stress),)
        for stress_weight in fractions:
            weights = ([1.-stress_weight] if stiffness else []) + ([stress_weight/2.]*2 if maximum_stress else [])
            grad=np.average(gradients,axis=0,weights=weights*len(responses))
            field[allowed]=grad
            smooth=gaussian_filter(field,filter_sigma,mode='constant')[allowed]/weight
            for move in (.12,.06,.03,.015,.0075):
                trial=projected_trial(density,smooth,~protected,move)
                if np.max(np.abs(trial-density))<1e-8:
                    continue
                if any(g@(trial-density)>0 for g in gradients):
                    continue
                results=evaluate(trial)
                change=(metrics(results)-metrics(responses))/baseline
                if np.max(change)<=1.e-9 and np.min(change)<-1.e-7:
                    density,responses=trial,results
                    record(generation,density,responses)
                    accepted=True
                    break
            if accepted:
                break
        if not accepted:
            break
    return density
