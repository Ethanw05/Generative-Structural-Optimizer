"""CAD time histories with shared 3-D elasticity, load-dependent Ritz vectors and native contact."""
from dataclasses import dataclass
import numpy as np
from .shock_receivers import receiver_patches, shock_metrics
from .rigid_carrier import rigid_modes
from .history_solver import integrate
from .history_envelope import HistoryEnvelope
from .ritz_response import consistent_mass,ritz_spaces,stress_envelope
from .tetra_elasticity import applied
from .reference_regions import moments,ReferenceRegions


class UnresolvedHistory(ValueError):
    """A configured transient response could not be established for this geometry."""

@dataclass(frozen=True)
class TimeExcitation:
    attachment: object
    times_s: np.ndarray
    factors: np.ndarray

    def at(self,times):
        t=np.asarray(self.times_s);f=np.asarray(self.factors)
        if t.ndim!=1 or f.shape!=t.shape or not len(t) or t[0]!=0. or np.any(np.diff(t)<=0.) or not np.isfinite(f).all() or not np.isfinite(t).all():
            raise ValueError('Invalid physical load history')
        return np.interp(times,t,f)


@dataclass(frozen=True)
class FoundationContact:
    unit_normal_load: object
    gap_mm: float
    stiffness_n_mm: float


@dataclass(frozen=True)
class AttachedInertia:
    unit_loads: tuple
    matrix: np.ndarray


@dataclass(frozen=True)
class HistoryCase:
    duration_s: float
    step_s: float
    loads: tuple
    contacts: tuple = ()
    inertias: tuple = ()
    damping: tuple = (0.,0.)
    carrier: object | None = None
    rigid_contacts: tuple = ()
    targets: tuple = ()
    relative_tolerance: float = .02
    max_orders: int = 12
    max_time_levels: int = 8
    require_contact: bool = True
    receivers: tuple = ()
    transmissions: tuple = ()


def event_times(case,level):
    """Bisect every physical load interval, retaining even pulses shorter than the nominal step."""
    knots=[0.,case.duration_s]
    for excitation in case.loads:
        excitation.at([0.])  # Validate the entire record before taking the analysis window.
        knots.extend(t for t in excitation.times_s if 0.<t<case.duration_s)
    events=np.unique(knots)
    pieces=[]
    for first,last in zip(events[:-1],events[1:],strict=True):
        count=max(1,int(np.ceil((last-first)/case.step_s-1.e-12)))*2**level
        pieces.append(np.linspace(first,last,count,endpoint=False))
    return np.concatenate([*pieces,events[-1:]])


def measure_history(system,study,case,solver,executable,density_kg_m3,surface_tol,reference=None,
                    relative_tolerance=.02,max_orders=8,max_time_levels=8):
    if any(abs(np.linalg.norm(x.unit_normal_load.force_n)-1.)>1.e-9 for x in case.contacts):
        raise ValueError("Contact ports must be unit normal forces")
    if not (0.<relative_tolerance<1. and 0.<case.step_s<=case.duration_s and max_orders>=2 and max_time_levels>=3):
        raise ValueError('Invalid dynamic study resolution or convergence settings')
    free=np.tile(~system.fixed,3)
    stiffness=system.stiffness[free][:,free].tocsr()
    component_mass=consistent_mass(system,density_kg_m3)
    def port(action):
        force,_,_=applied(study,action,system.local,system.skin,surface_tol)
        return force.T.ravel()
    loads=[port(item.attachment) for item in case.loads]
    normal=[port(item.unit_normal_load) for item in case.contacts]
    if not loads and not normal and not case.rigid_contacts:raise ValueError('An excitation or contact is required')
    if case.rigid_contacts and case.carrier is None:raise ValueError('Rigid contact requires a carrier')
    inertias=[]
    for item in case.inertias:
        h=np.asarray([port(load) for load in item.unit_loads])
        inertia=np.asarray(item.matrix,dtype=float)
        if inertia.shape!=(len(h),len(h)) or not np.allclose(inertia,inertia.T) or np.linalg.eigvalsh(inertia).min()<0.:
            raise ValueError('Invalid attached inertia')
        inertias.append((h,inertia))
    def full_mass(x):
        product=component_mass@x
        for h,inertia in inertias:product+=h.T@(inertia@(h@x))
        return product
    def supported(v):
        output=np.zeros((len(free),v.shape[1]))
        output[free]=v
        return output
    def mass_apply(x):
        return full_mass(supported(x))[free]
    seeds=[v[free] for v in [*loads,*normal]]
    carrier=case.carrier
    rigid=np.empty((len(free),0)) if carrier is None else rigid_modes(system.points,carrier.origin_world_mm)
    if carrier is not None:
        seeds.extend(full_mass(rigid)[free].T)
    channels=np.column_stack(seeds)
    all_contacts=(*case.contacts,*case.rigid_contacts)
    reference=reference or ReferenceRegions()
    patches = receiver_patches(case,system,surface_tol)
    previous_order=None
    stable_orders=0
    def close(a,b):
        scale=np.maximum(np.maximum(abs(a),abs(b)),np.finfo(float).eps)
        return np.all(abs(a-b)<=relative_tolerance*scale)
    for order,(basis,mb) in enumerate(ritz_spaces(stiffness,mass_apply,channels,solver,max_orders),start=1):
        full=supported(basis)
        motion=np.column_stack((rigid,full))
        dimension=motion.shape[1]
        m=motion.T@full_mass(motion)
        k=np.zeros_like(m)
        k[rigid.shape[1]:,rigid.shape[1]:]=basis.T@(stiffness@basis)
        velocity=np.zeros(dimension)
        if carrier is not None:
            m[:6,:6]+=carrier.spatial_mass()
            velocity[:6]=np.r_[carrier.velocity_world_mm_s,carrier.angular_velocity_world_rad_s]
        k=(k+k.T)/2.;m=(m+m.T)/2.
        force=(np.asarray(loads)@motion if loads else np.empty((0,dimension)))
        normal_rows=[v@motion for v in normal]
        normal_rows.extend(np.r_[c.direction(carrier.origin_world_mm),np.zeros(len(k)-6)] for c in case.rigid_contacts)
        normals=np.asarray(normal_rows).reshape(-1,dimension)
        sensors = [patch.translation_map(motion) for patch in patches]
        previous_time=None
        stable_times=0
        for power in range(max_time_levels):
            time=event_times(case,power)
            series=np.vstack([load.at(time) for load in case.loads]) if loads else np.empty((0,len(time)))
            recording=integrate(executable,k,m,velocity,time,force,series,normals,
                                    [c.gap_mm for c in all_contacts],[c.stiffness_n_mm for c in all_contacts],case.damping,
                                    record_acceleration=bool(patches))
            states,result = recording[:2]
            shock = {}
            if patches:
                acceleration = recording[2]
                peaks = [patch.magnitude_g(acceleration @ sensor.T).max()
                         for patch,sensor in zip(patches,sensors,strict=True)]
                shock = shock_metrics(case,peaks)
                result.update(shock)
            elastic_states=states[:,rigid.shape[1]:]
            recovery=HistoryEnvelope(elastic_states,executable)
            vm,instant=stress_envelope(system,full,elastic_states,recovery)
            nodal_map=full.reshape(3,len(system.points),-1).transpose(1,0,2)
            motion_peak,_=recovery.measure(nodal_map)
            result['peak_elastic_displacement_mm']=float(motion_peak.max())
            metrics=np.r_[result['peak_energy_Nmm'],result['peak_contact_N'],result['peak_elastic_displacement_mm'],vm,list(shock.values())]
            stable_times = stable_times+1 if previous_time is not None and close(metrics,previous_time) else 0
            if stable_times>=2:break
            if power+1==max_time_levels:
                scale=np.maximum(np.maximum(abs(metrics),abs(previous_time)),np.finfo(float).eps)
                worst=int(np.argmax(abs(metrics-previous_time)/scale))
                name=('elastic energy in N mm','contact force in N','elastic displacement in mm')[worst] if worst<3 else f'element {worst-3} stress in MPa'
                if worst >= 3+len(vm): name=list(shock)[worst-3-len(vm)]
                raise UnresolvedHistory(f'Temporal refinement unresolved at Ritz order {order}, '
                                 f'dt={np.diff(time).max():.6g} s, {stable_times}/2 stable levels. '
                                 f'Largest relative change in {name}: {previous_time[worst]:.8g} to {metrics[worst]:.8g}')
            previous_time=metrics
        stable_orders = stable_orders+1 if previous_order is not None and close(metrics,previous_order) else 0
        # An invariant subspace containing all load directions is exact for this linear component.
        kb=stiffness@basis
        reduced=basis.T@kb
        remainder=kb@np.linalg.solve(reduced,basis.T@mb)-mb
        invariant=np.all(np.linalg.norm(remainder,axis=0)<=1.e-8*np.linalg.norm(mb,axis=0))
        if stable_orders>=2 or invariant or basis.shape[1]==stiffness.shape[0]:
            if all_contacts and case.require_contact and result["peak_contact_N"]<=0.:
                raise UnresolvedHistory("Impact history ended without contact")
            if all_contacts:
                compression=np.maximum(0.,-(states@normals.T+[c.gap_mm for c in all_contacts]))
                contact_force=compression*np.asarray([c.stiffness_n_mm for c in all_contacts])
                result['contact_histories']=[]
                for values in contact_force.T:
                    touched=np.flatnonzero(values>0.)
                    released=np.flatnonzero((values[:-1]>0.) & (values[1:]==0.))+1
                    result['contact_histories'].append(dict(
                        peak_N=float(values.max()),peak_time_s=float(time[np.argmax(values)]),
                        first_contact_s=float(time[touched[0]]) if len(touched) else None,
                        first_release_s=float(time[released[0]]) if len(released) else None,
                        final_force_N=float(values[-1]),
                        impulse_Ns=float(np.sum(np.diff(time)*(values[1:]+values[:-1])*.5))))
            if carrier is not None:
                result["peak_rigid_rotation_rad"]=float(np.linalg.norm(states[:,3:6],axis=1).max())
            labels=reference.assign(system.points,system.cells,system.gradients,system.volume,vm,system.skin,solver)
            peaks,_,occupied=moments(vm,system.volume,labels,len(reference.scale),8.)
            result.update(orders=order,modes=basis.shape[1],step_s=float(np.diff(time).max()),peak_MPa=float(vm.max()),
                          peak_time_s=float(time[instant[np.argmax(vm)]]),window_s=float(time[-1]),
                          peak_at_window_end=bool(instant[np.argmax(vm)]==len(time)-1),local_peaks_MPa=peaks.tolist(),
                          local_region_volumes_mm3=occupied.tolist())
            return result,vm,time[instant]
        previous_order=metrics
    raise UnresolvedHistory('Elastic basis refinement did not resolve all stress regions')


def export_history(system,study,case,path,solver,regions,tolerance,*,law=None,label=None):
    import json
    from .stress_regions import stress_basins,basin_peaks
    additional={}
    if law is None:
        response,vm,times=measure_history(system,study,case,solver,study.mechanics_executable,study.density_kg_m3,
                                          tolerance,regions,case.relative_tolerance,case.max_orders,case.max_time_levels)
    else:
        from .nonlinear_dynamics import measure_nonlinear
        response,vm,times,energy,plastic=measure_nonlinear(system,study,case,solver,tolerance,regions,law)
        additional=dict(finite_strain=True,elastic_work_density=energy[system.owner],
                        equivalent_plastic_strain=plastic[system.owner])
    basins=stress_basins(system.cells,vm)
    indices,volumes=basin_peaks(vm,system.volume,basins)
    response['regional_peak_rms_MPa']=float(np.sqrt(np.average(vm[indices]**2,weights=volumes)))
    if law is None:response['solid_model']=dict(kinematics='small strain',material='linear isotropic elastic')
    response['solid_model'].update(young_MPa=study.young_mpa,poisson=study.poisson)
    if label is not None:
        response['scenario'] = label
        response['solid_model'].update(study.solid_model(label))
    response['time_targets']=list(case.targets)
    response['n_tets']=len(system.cells)
    response['local_peak_count']=len(indices)
    np.savez_compressed(path.with_name(path.name+'-peaks.npz'),world_mm=system.points[system.cells[indices]].mean(axis=1),
                        peak_MPa=vm[indices],time_s=times[indices],basin_volume_mm3=volumes)
    used,inv=np.unique(system.skin,return_inverse=True)
    np.savez_compressed(path.with_suffix('.npz'),v=system.points[used],f=inv.reshape(-1,3),
                        vm=vm[system.owner],peak_time_s=times[system.owner],transient=True,**additional)
    path.with_suffix('.json').write_text(json.dumps(response,indent=2))
    return response
