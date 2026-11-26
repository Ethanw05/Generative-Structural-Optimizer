"""Full-order nonlinear Newmark dynamics with committed material state and exact rigid fixtures."""
import numpy as np
from scipy import sparse
from .finite_solid import FiniteSolid,UnresolvedSolid
from .nonlinear_equilibrium import equilibrium
from .nonlinear_inertia import DynamicPorts,DynamicPreconditioner
from .nonlinear_kinematics import AttachmentMap
from .reference_regions import moments
from .shock_receivers import receiver_patches, shock_metrics


def _history(system,case,solid,ports,mapping,times,solve,preconditioner,receivers):
    n=mapping.solid_dofs
    state=solid.virgin()
    u=np.zeros(mapping.size)
    v=np.zeros_like(u)
    rotation=np.eye(3)
    translation=np.zeros(3)
    q=np.zeros(mapping.direct.shape[1])
    carrier=mapping.carrier
    centripetal=np.zeros_like(u)
    if carrier is not None:
        omega=carrier.angular_velocity_world_rad_s
        arms=system.points-carrier.origin_world_mm
        v[:n]=(carrier.velocity_world_mm_s+np.cross(omega,arms)).T.ravel()
        v[mapping.rows]=carrier.velocity_world_mm_s+np.cross(omega,mapping.offsets)
        centripetal[mapping.rows]=np.cross(omega,np.cross(omega,mapping.offsets))
    jac=mapping.jacobian(q,rotation)
    a=jac@solve((jac.T@ports.mass@jac).tocsr(),jac.T@(ports.load(case,0.)+ports.normals.T@ports.contact(u)
                 -ports.damping@v-ports.mass@centripetal))+centripetal
    peaks = np.array([r.magnitude_g(r.average(a[:n])) for r in receivers])
    vm=np.zeros(len(system.cells));when=np.zeros_like(vm);energy=np.zeros_like(vm)
    translations=np.vstack((np.repeat(np.eye(3),len(system.points),axis=0),np.tile(np.eye(3),(mapping.extra,1))))
    original_momentum=translations.T@(ports.mass@v)
    previous_external=ports.load(case,0.)+ports.normals.T@ports.contact(u)-ports.damping@v
    external_impulse=np.zeros(3)
    max_energy=max_disp=angle=residual=0.
    contacts=np.zeros((len(times),len(ports.gaps)))
    initial_momentum=np.linalg.norm(ports.mass@v)/times[-1]
    length=np.linalg.norm(np.ptp(system.points,axis=0))
    force_roundoff=128.*np.finfo(float).eps*system.shear*length**2
    for step,(begin,end) in enumerate(zip(times[:-1],times[1:],strict=True),1):
        h=end-begin
        prediction=u+h*v+.25*h*h*a
        velocity_prediction=v+.5*h*a
        accel_factor=4./h**2
        speed_factor=2./h
        external=ports.load(case,end)
        constant=accel_factor*ports.mass+speed_factor*ports.damping
        q[:len(mapping.free)]=prediction[mapping.free]
        if carrier is not None:q[-6:]=np.r_[translation,np.zeros(3)]
        def balance(y,tangent):
            x=mapping.position(y,rotation)
            j=mapping.jacobian(y,rotation)
            acceleration=accel_factor*(x-prediction)
            speed=velocity_prediction+.5*h*acceleration
            field=solid.evaluate(x[:n],state,tangent)
            contact=ports.contact(x)
            ambient=ports.mass@acceleration+ports.damping@speed-external-ports.normals.T@contact
            ambient[:n]+=field.internal
            matrix=None
            if tangent:
                contact_k=ports.normals.T@ports.normals.multiply((ports.spring*(contact>0.))[:,None])
                ambient_k=constant+contact_k+sparse.block_diag((field.tangent,ports.extra),format='csr')
                matrix=(j.T@ambient_k@j+mapping.curvature(y,rotation,ambient)).tocsr()
            return j.T@ambient,matrix,field
        scale=max(np.linalg.norm(external),np.linalg.norm(ports.mass@a),initial_momentum,
                  np.linalg.norm(ports.normals.T@ports.contact(prediction)),force_roundoff/solid.law.equilibrium_tolerance)
        predictor=preconditioner.get(h,ports.contact(prediction)>0.)
        answer=equilibrium(q,balance,solve,scale,solid.law.equilibrium_tolerance,predictor=predictor)
        x=mapping.position(answer.coordinates,rotation)
        a=accel_factor*(x-prediction)
        v=velocity_prediction+.5*h*a
        for index,r in enumerate(receivers):
            peaks[index] = max(peaks[index],r.magnitude_g(r.average(a[:n])))
        u=x
        state=answer.field.state
        values=answer.field.measures
        greater=values[:,0]>vm
        vm[greater]=values[greater,0];when[greater]=end
        np.maximum(energy,values[:,1],out=energy)
        max_energy=max(max_energy,float(system.volume@values[:,1]))
        displacement=u[:n].reshape(3,-1).T
        if carrier is not None:
            translation=answer.coordinates[-6:-3]
            rotation=mapping.rotation(answer.coordinates[-3:],rotation)
            displacement=displacement-translation-(system.points-carrier.origin_world_mm)@(rotation-np.eye(3)).T
            angle=max(angle,float(np.arccos(np.clip((np.trace(rotation)-1.)*.5,-1.,1.))))
        max_disp=max(max_disp,float(np.linalg.norm(displacement,axis=1).max()))
        contacts[step]=ports.contact(x)
        net=external+ports.normals.T@contacts[step]-ports.damping@v
        external_impulse+=.5*h*(translations.T@(previous_external+net))
        previous_external=net
        residual=max(residual,answer.residual)
    report=dict(peak_energy_Nmm=max_energy,peak_relative_displacement_mm=max_disp,
                peak_contact_N=float(contacts.max(initial=0.)),peak_rigid_rotation_rad=angle,
                maximum_equilibrium_residual=residual,solid_model=solid.law.description(),contact_histories=[])
    if carrier is not None:
        final_momentum=translations.T@(ports.mass@v)
        report.update(initial_linear_momentum_Ns=original_momentum.tolist(),final_linear_momentum_Ns=final_momentum.tolist(),
                      external_impulse_Ns=external_impulse.tolist(),
                      momentum_balance_Ns=(final_momentum-original_momentum-external_impulse).tolist())
    for values in contacts.T:
        active=np.flatnonzero(values>0.)
        released=np.flatnonzero((values[:-1]>0.)&(values[1:]==0.))+1
        report['contact_histories'].append(dict(peak_N=float(values.max()),peak_time_s=float(times[np.argmax(values)]),
            first_contact_s=float(times[active[0]]) if len(active) else None,
            first_release_s=float(times[released[0]]) if len(released) else None,final_force_N=float(values[-1]),
            impulse_Ns=float(np.sum(np.diff(times)*(values[:-1]+values[1:])*.5))))
    report.update(shock_metrics(case,peaks))
    return report,vm,when,energy,state[:,9]


def measure_nonlinear(system,study,case,solve,tolerance,regions,law):
    from .cad_history import event_times,UnresolvedHistory
    if not (0.<case.relative_tolerance<1. and 0.<case.step_s<=case.duration_s and case.max_time_levels>=3):
        raise ValueError('Invalid nonlinear time integration settings')
    receivers=receiver_patches(case,system,tolerance)
    mapping=AttachmentMap(system,case)
    ports=DynamicPorts(system,study,case,mapping,tolerance)
    if not case.loads and not len(ports.gaps):raise ValueError('A history needs a load or contact')
    solid=FiniteSolid(system,law,study.mechanics_executable)
    previous=None;stable=0
    for level in range(case.max_time_levels):
        time=event_times(case,level)
        try:
            with DynamicPreconditioner(mapping,ports,system,solve) as preconditioner:
                report,vm,when,energy,plastic=_history(system,case,solid,ports,mapping,time,solve,preconditioner,receivers)
        except UnresolvedSolid:
            stable=0;previous=None
            if level+1==case.max_time_levels:raise
            continue
        metrics=np.r_[report['peak_energy_Nmm'],report['peak_relative_displacement_mm'],report['peak_contact_N'],vm,plastic,
                      [report[r.metric] for r in (*case.receivers,*case.transmissions)]]
        if previous is not None:
            scale=np.maximum(np.maximum(np.abs(metrics),np.abs(previous)),1.e-12)
            stable=stable+1 if np.all(abs(metrics-previous)<=case.relative_tolerance*scale) else 0
        if stable>=2:
            if case.require_contact and len(ports.gaps) and report['peak_contact_N']==0.:
                raise UnresolvedHistory('No contact occurred during the nonlinear history')
            labels=regions.assign(system.points,system.cells,system.gradients,system.volume,vm,system.skin,solve)
            peaks,_,volumes=moments(vm,system.volume,labels,len(regions.scale),8.)
            report.update(peak_MPa=float(vm.max()),peak_time_s=float(when[np.argmax(vm)]),
                          local_peaks_MPa=peaks.tolist(),local_region_volumes_mm3=volumes.tolist(),
                          step_s=float(np.diff(time).max()),window_s=float(time[-1]),time_refinements=level,
                          peak_at_window_end=bool(when[np.argmax(vm)]==time[-1]))
            if law.initial_yield_mpa is not None:report['peak_equivalent_plastic_strain']=float(plastic.max())
            return report,vm,when,energy,plastic
        previous=metrics
    raise UnresolvedHistory('Full-order nonlinear history did not converge under time refinement')
