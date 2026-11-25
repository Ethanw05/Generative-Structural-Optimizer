"""Comparable mechanical objectives for independently evaluated solid designs."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ShapeTargets:
    stiffness: bool = True
    maximum_stress: bool = True

    def names(self,response=None):
        if response is not None and "time_targets" in response:
            metrics=list(response["time_targets"])
            if self.stiffness:metrics.append("peak_relative_displacement_mm" if "peak_relative_displacement_mm" in response
                                             else "peak_elastic_displacement_mm")
            if self.maximum_stress:metrics.extend(("peak_MPa","regional_peak_rms_MPa"))
            if any(key not in ("peak_energy_Nmm","peak_contact_N","peak_elastic_displacement_mm","peak_relative_displacement_mm","peak_MPa","regional_peak_rms_MPa") for key in metrics):
                raise ValueError("Unknown transient objective")
            if self.maximum_stress and 'peak_equivalent_plastic_strain' in response:metrics.append('peak_equivalent_plastic_strain')
            missing=set(metrics)-response.keys()
            if missing:raise ValueError('The history does not provide the selected objectives: '+', '.join(sorted(missing)))
            return tuple(dict.fromkeys(metrics))
        names=tuple(name for active,name in ((self.stiffness,'compliance_Nmm'),
                    (self.maximum_stress,'peak_MPa'), (self.maximum_stress,'regional_peak_rms_MPa')) if active)
        if not names:
            raise ValueError('Choose at least one structural objective')
        if self.maximum_stress and response is not None and 'peak_equivalent_plastic_strain' in response:
            names+=('peak_equivalent_plastic_strain',)
        return names


@dataclass
class EvaluatedSurface:
    mesh: object
    responses: list
    evidence: list
    name: str
    first_evidence: list | None = None
    regions: dict | None = None


def objective_bounds(candidate,targets):
    """Bound an unsampled region by zero and the global maximum of this discrete stress field."""
    lower=[]
    upper=[]
    for level in candidate.responses:
        for result in level:
            primary=[result[metric] for metric in targets.names(result)]
            lower.extend(primary)
            upper.extend(primary)
            if targets.maximum_stress:
                peaks=np.asarray(result.get('local_peaks_MPa',()),dtype=float)
                volume=np.asarray(result.get('local_region_volumes_mm3',np.ones_like(peaks)))
                if volume.shape!=peaks.shape or np.any(volume<0.) or not np.isfinite(volume).all():
                    raise ValueError('Invalid regional sampling coverage')
                lower.extend(np.where(volume>0.,peaks,0.))
                upper.extend(np.where(volume>0.,peaks,result['peak_MPa']))
    return np.asarray(lower),np.asarray(upper)


def coordinates(candidate,targets):
    return objective_bounds(candidate,targets)[1]



def pareto(candidates,targets):
    if targets.maximum_stress and candidates and any(c.regions is not candidates[0].regions for c in candidates):
        raise ValueError('Stress regions belong to different reference geometries')
    bounds=[objective_bounds(candidate,targets) for candidate in candidates]
    if any(np.any(~np.isfinite(pair)) or np.any(np.asarray(pair)<0.) for pair in bounds):
        raise ValueError('Invalid structural optimization response')
    return [candidate for candidate,(lower,_) in zip(candidates,bounds,strict=True)
            if not any(np.all(upper<=lower) and np.any(upper<lower) for _,upper in bounds)]


def objective_scales(reference,targets):
    """Use one physical stress unit per case across every mesh and every local maximum."""
    case_scales=[]
    for index,response in enumerate(reference.responses[0]):
        cases=[row[index] for row in reference.responses]
        stress_scale=max((case.get('peak_MPa',0.) for case in cases),default=0.)
        # With zero baseline stress the study unit (MPa) is the available physical scale.
        stress_scale=stress_scale if stress_scale>0. else 1.
        primary=[]
        for key in targets.names(response):
            magnitude=stress_scale if key in ('peak_MPa','regional_peak_rms_MPa') else max(c[key] for c in cases)
            if key=='peak_equivalent_plastic_strain':
                material=response['solid_model']
                magnitude=max(magnitude,material['initial_Kirchhoff_yield_MPa']/material['young_MPa'])
            primary.append(magnitude if magnitude>0. else 1.)
        regions=len(response.get('local_peaks_MPa',())) if targets.maximum_stress else 0
        case_scales.extend([*primary,*([stress_scale]*regions)])
    return np.tile(case_scales,len(reference.responses))


def compromise_scores(candidates,targets,scale):
    """Worst normalized regret from the best observed bound for each requested objective."""
    values=np.stack([coordinates(item,targets) for item in candidates])
    scale=np.asarray(scale,dtype=float)
    if scale.shape!=(values.shape[1],) or np.any(scale<=0.) or not np.isfinite(scale).all():
        raise ValueError('Invalid comparative objective scales')
    regret=(values-values.min(axis=0))/scale
    # A finer mesh can reveal more regional peaks. Give each scenario the same
    # tie-breaking weight regardless of how many peaks it contains.
    scenario_means=[]
    offset=0
    for level in candidates[0].responses:
        for response in level:
            primary=len(targets.names(response))
            regions=len(response.get('local_peaks_MPa',())) if targets.maximum_stress else 0
            categories=list(regret[:,offset:offset+primary].T)
            if regions:
                categories.append(regret[:,offset+primary:offset+primary+regions].mean(axis=1))
            if categories:scenario_means.append(np.mean(categories,axis=0))
            offset+=primary+regions
    if offset!=values.shape[1] or not scenario_means:
        raise ValueError('Incomplete objective layout')
    average=np.mean(scenario_means,axis=0)
    return {item.name:(float(worst),float(mean)) for item,worst,mean in
            zip(candidates,regret.max(axis=1),average,strict=True)}


