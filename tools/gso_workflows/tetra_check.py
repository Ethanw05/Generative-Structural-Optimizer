"""Independent small-strain P1 elasticity with native face patch loads, fixtures and element stresses."""
import json
from pathlib import Path
import numpy as np
from .tetra_elasticity import assemble, applied, exterior
from .cad_wrench import SurfaceWrench
from .stress_regions import stress_basins, basin_peaks
from .reference_regions import moments
from .sparse_solve import fixed_matrix_solver
from .shape_adjoint import normal_descent,stress_shape_gradient
from .load_transfer import loaded_neighborhood,section_response


def solve(p,t,study,output,linear_solve,*,surface_tol=.35,local_regions=None,report_to_console=True):
    if len(study.loads)!=1:raise ValueError('A single named force case is required for this field export')
    label=next(iter(study.loads))
    study=study.for_case(label)
    system=assemble(p,t,study,surface_tol)
    law=study.solid_laws.get(label)
    if law is not None:
        from .nonlinear_response import static_response
        result=static_response(system,study,label,law,output,linear_solve,surface_tol,local_regions)
        if report_to_console:print(json.dumps(result),flush=True)
        return result
    free=np.tile(~system.fixed,3)
    stiffness=system.stiffness[free][:,free].tocsr()
    stiffness.sort_indices()
    with fixed_matrix_solver(stiffness,linear_solve) as factorized:
        def solve_system(matrix,rhs):
            # Reference-region recovery has its own scalar projection matrix.
            return factorized(rhs) if matrix is stiffness else linear_solve(matrix,rhs)
        result=_response(system,study,output,surface_tol,local_regions,free,stiffness,solve_system)
    if report_to_console:
        print(json.dumps(result),flush=True)
    return result


def _response(system,study,output,surface_tol,local_regions,free,A,linear_solve):
    (p,t,volume,grad,shear,lame,matrix,skin,owner,local,fixed,fixture_area,fixture_transfer)=system
    n=len(p)
    load=next(iter(study.loads.values()))
    sample_faces=load.faces if isinstance(load,SurfaceWrench) else (load.face,)
    direction=load.force_n@study.world_rotation
    force,load_area,load_transfer=applied(study,load,local,skin,surface_tol)
    rhs=force.T.ravel();u=np.zeros(3*n)
    u[free]=linear_solve(A,rhs[free])
    balance=matrix@u-rhs
    residual=np.linalg.norm(balance[free])/max(np.linalg.norm(rhs),np.finfo(float).tiny)
    if not np.isfinite(residual) or residual>1e-6:raise ValueError(f'Equilibrium residual {residual}')
    displacement=u.reshape(3,n).T
    du=displacement[t].transpose(0,2,1)@grad
    strain=.5*(du+du.transpose(0,2,1))
    dev=strain-np.trace(strain,axis1=1,axis2=2)[:,None,None]*np.eye(3)/3.
    vm=2*shear*np.sqrt(1.5*np.einsum('ijk,ijk->i',dev,dev))
    order=np.argsort(vm)
    def percentile(q):return float(np.interp(q*volume.sum(),np.cumsum(volume[order]),vm[order]))
    fn=float(np.linalg.norm(direction));work=float(rhs@u)
    summary=dict(scenario=next(iter(study.loads)),solid_model=study.solid_model(next(iter(study.loads))),stress_measure='element von Mises',volume_mm3=volume.sum(),n_tets=len(t),n_nodes=n,
        force_N=force.sum(0).tolist(),reaction_N=balance.reshape(3,n).sum(1).tolist(),
        load_transfer=load_transfer,fixture_transfer=fixture_transfer,loaded_area_mm2=load_area,fixed_area_mm2=fixture_area,compliance_Nmm=work,
        p95_MPa=percentile(.95),p99_MPa=percentile(.99),
        peak_MPa=float(vm.max()),residual=residual)
    # Integrate stress by material volume near the loaded interface as well as over the whole part.
    if not isinstance(load,SurfaceWrench):
        if fn>0.:
            summary['mean_lift_mm']=work/fn
            if work>0.:summary['stiffness_N_mm']=fn**2/work
    else:
        moment = np.cross(local-load.origin_mm,force@study.world_rotation.T).sum(axis=0)
        summary['applied_moment_local_Nmm'] = moment.tolist()
        magnitude = np.linalg.norm(load.moment_nmm)
        if fn == 0. and magnitude > 0.:
            summary['mean_rotation_rad']=work/magnitude
            if work>0.:summary['torsional_stiffness_Nmm_rad']=magnitude**2/work
    summary['loaded_face_neighborhood']=loaded_neighborhood(local,t,sample_faces,vm,volume,order)
    summary['peak_at_world_mm']=p[t[np.argmax(vm)]].mean(0).tolist()
    forces_at_supports=balance.reshape(3,n).T.copy()
    forces_at_supports[~fixed]=0.
    moment=np.cross(p-study.world_origin,force+forces_at_supports).sum(0)
    summary['moment_residual_Nmm']=moment.tolist()
    active,indices=np.unique(skin,return_inverse=True)
    energy=2.*shear*np.einsum('ijk,ijk->i',strain,strain)+lame*np.trace(strain,axis1=1,axis2=2)**2
    aggregate,derivative=stress_shape_gradient(p,t,grad,volume,du,shear,lame,free,A,linear_solve)
    descent=normal_descent(p,skin,derivative)
    basins = stress_basins(t, vm)
    regional, gradient = stress_shape_gradient(p,t,grad,volume,du,shear,lame,free,A,linear_solve,basins=basins)
    local_descent = normal_descent(p,skin,gradient)
    summary['regional_stress_MPa'] = float(regional)
    maxima, basin_volumes = basin_peaks(vm, volume, basins)
    summary['stress_basin_count'] = len(maxima)
    summary['regional_peak_rms_MPa'] = float(np.sqrt(np.average(vm[maxima]**2, weights=basin_volumes)))
    np.savez_compressed(output.with_name(output.name+'-peaks.npz'), world_mm=p[t[maxima]].mean(axis=1),
                        peak_MPa=vm[maxima], basin_volume_mm3=basin_volumes)

    summary['stress_p8_MPa']=float(aggregate)
    if not isinstance(load,SurfaceWrench) and load.transfer_face is not None and fn>0.:
        sigma=2.*shear*strain+lame*np.trace(strain,axis1=1,axis2=2)[:,None,None]*np.eye(3)
        thickness=abs((load.transfer_face.origin-load.face.origin)@load.face.normal)
        summary['transfer_sections']=[section_response(local,t,sigma,study,load,thickness*fraction)
                                      for fraction in (.10,.40,.80)]
    additional = {}
    if local_regions is not None:
        labels = local_regions.assign(p,t,grad,volume,vm,skin,linear_solve)
        peaks, _, region_volumes = moments(vm,volume,labels,len(local_regions.scale),8.)
        summary['local_peaks_MPa'] = peaks.tolist()
        summary['local_region_volumes_mm3'] = region_volumes.tolist()
        balanced, deriv = stress_shape_gradient(p,t,grad,volume,du,shear,lame,free,A,linear_solve,
                                                basins=labels,reference=local_regions.scale)
        summary['relative_regional_stress'] = float(balanced)
        additional['balanced_descent'] = normal_descent(p,skin,deriv)
    np.savez_compressed(output.with_suffix('.npz'),v=p[active],f=indices.reshape(-1,3),
                        vm=vm[owner],elastic_work_density=energy[owner],stress_descent=descent,regional_descent=local_descent,**additional)
    output.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    return summary
