"""Static nonlinear stress, displacement and plastic strain on the realized CAD solid."""
import json
import numpy as np
from .cad_wrench import SurfaceWrench
from .finite_solid import FiniteSolid
from .load_transfer import loaded_neighborhood,section_response
from .nonlinear_equilibrium import static_path
from .reference_regions import ReferenceRegions,moments
from .stress_regions import basin_peaks,stress_basins
from .tetra_elasticity import applied


def static_response(system,study,label,law,path,solve,tolerance,reference):
    load=study.loads[label]
    force,area,transfer=applied(study,load,system.local,system.skin,tolerance)
    rhs=force.T.ravel()
    solid=FiniteSolid(system,law,study.mechanics_executable,capture_stress=True)
    displacement,field,residual,steps=static_path(solid,[rhs],solve)[0]
    vm,energy=field.measures.T
    volume=system.volume
    basins=stress_basins(system.cells,vm)
    maxima,weights=basin_peaks(vm,volume,basins)
    ref=reference if reference is not None else ReferenceRegions()
    labels=ref.assign(system.points,system.cells,system.gradients,volume,vm,system.skin,solve)
    peaks,_,occupied=moments(vm,volume,labels,len(ref.scale),8.)
    centers=system.points[system.cells].mean(axis=1)
    result=dict(scenario=label,solid_model=study.solid_model(label),stress_measure='Cauchy element von Mises',
                peak_MPa=float(vm.max()),compliance_Nmm=float(rhs@displacement),
                stored_elastic_energy_Nmm=float(volume@energy),volume_mm3=float(volume.sum()),
                regional_peak_rms_MPa=float(np.sqrt(np.average(vm[maxima]**2,weights=weights))),
                local_peaks_MPa=peaks.tolist(),local_region_volumes_mm3=occupied.tolist(),
                reaction_N=(field.internal-rhs).reshape(3,-1)[:,system.fixed].sum(axis=1).tolist(),
                force_N=force.sum(axis=0).tolist(),loaded_area_mm2=float(area),load_transfer=transfer,
                residual=float(residual),load_increments=steps,n_tets=len(volume),
                peak_at_world_mm=centers[np.argmax(vm)].tolist())
    sample_faces=load.faces if isinstance(load,SurfaceWrench) else (load.face,)
    result['loaded_face_neighborhood']=loaded_neighborhood(system.local,system.cells,sample_faces,vm,volume,np.argsort(vm))
    magnitude=np.linalg.norm(load.force_n)
    if magnitude>0. and not isinstance(load,SurfaceWrench):
        result['mean_lift_mm']=result['compliance_Nmm']/magnitude
        if load.transfer_face is not None:
            thickness=abs((load.transfer_face.origin-load.face.origin)@load.face.normal)
            # P times the undeformed normal gives force per reference area.
            result['transfer_sections']=[section_response(system.local,system.cells,field.first_piola,study,load,thickness*f)
                                         for f in (.10,.40,.80)]
    if law.initial_yield_mpa is not None:
        result['peak_equivalent_plastic_strain']=float(field.state[:,9].max())
    active,inv=np.unique(system.skin,return_inverse=True)
    np.savez_compressed(path.with_suffix('.npz'),v=system.points[active],f=inv.reshape(-1,3),vm=vm[system.owner],
                        elastic_work_density=energy[system.owner],equivalent_plastic_strain=field.state[system.owner,9],
                        finite_strain=True)
    np.savez_compressed(path.with_name(path.name+'-peaks.npz'),world_mm=centers[maxima],peak_MPa=vm[maxima],
                        basin_volume_mm3=weights)
    path.with_suffix('.json').write_text(json.dumps(result,allow_nan=False,indent=2))
    return result
