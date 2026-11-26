"""Comparable mechanical objectives for independently evaluated solid designs."""
from dataclasses import dataclass,field
import numpy as np
from .design_targets import YieldSafety, violation, measured


@dataclass(frozen=True)
class ShapeTargets:
    stiffness: bool = True
    maximum_stress: bool = True
    limits: tuple = ()
    extra_objectives: tuple = ()
    verification_refinements: int = 2
    verification_edges_mm: tuple = ()

    def __post_init__(self):
        if not isinstance(self.verification_refinements,int) or self.verification_refinements<0:
            raise ValueError('The additional target refinement budget must be a nonnegative integer')
        edges=self.verification_edges_mm
        if any(not np.isfinite(edge) or edge<=0. for edge in edges) or any(a<=b for a,b in zip(edges,edges[1:])):
            raise ValueError('Independent verification edge lengths must be positive and strictly decreasing')

    def validate_cases(self, study):
        labels = set(study.scenarios())
        for request in (*self.limits, *self.extra_objectives):
            missing = set(request.cases) - labels
            if missing:
                raise ValueError('Unknown target scenarios: ' + ', '.join(sorted(missing)))
            metric = getattr(request, 'metric', '')
            if '@' in metric:
                for label in request.cases:
                    case = study.histories.get(label)
                    if case is None or metric not in {s.metric for s in (*case.receivers,*case.transmissions)}:
                        raise ValueError(f'No configured receiver for {metric} in {label}')

    def direction(self, response, metric):
        senses={goal.direction for goal in self.extra_objectives
                if response.get('scenario') in goal.cases and metric==goal.metric}
        built_in=(self.stiffness and metric in ('compliance_Nmm','peak_relative_displacement_mm','peak_elastic_displacement_mm')
                  or self.maximum_stress and metric in ('peak_MPa','regional_peak_rms_MPa','peak_equivalent_plastic_strain')
                  or metric in response.get('time_targets',()))
        if built_in:
            senses.add(1)
        if len(senses)>1:
            raise ValueError(f'Opposite directions requested for {metric} in {response.get("scenario")}')
        return next(iter(senses),1)

    def uses_stress(self, label=None):
        return self.maximum_stress or any(isinstance(limit,YieldSafety) and
            (label is None or label in limit.cases) for limit in self.limits)

    def names(self,response=None):
        extra = []
        if response is not None:
            for goal in self.extra_objectives:
                if response.get('scenario') in goal.cases:
                    measured(response, goal.metric)
                    extra.append(goal.metric)
        if response is not None and "time_targets" in response:
            metrics=list(response["time_targets"])
            if self.stiffness:metrics.append("peak_relative_displacement_mm" if "peak_relative_displacement_mm" in response
                                             else "peak_elastic_displacement_mm")
            if self.maximum_stress:metrics.extend(("peak_MPa","regional_peak_rms_MPa"))
            transient_metrics = {"peak_energy_Nmm","peak_contact_N","peak_elastic_displacement_mm",
                                 "peak_relative_displacement_mm","peak_MPa","regional_peak_rms_MPa"}
            if any(key not in transient_metrics and not key.startswith(("peak_acceleration_g@","peak_transmission@"))
                   for key in metrics):
                raise ValueError("Unknown transient objective")
            if self.maximum_stress and 'peak_equivalent_plastic_strain' in response:metrics.append('peak_equivalent_plastic_strain')
            missing=set(metrics)-response.keys()
            if missing:raise ValueError('The history does not provide the selected objectives: '+', '.join(sorted(missing)))
            return tuple(dict.fromkeys([*metrics,*extra]))
        names=tuple(name for active,name in ((self.stiffness,'compliance_Nmm'),
                    (self.maximum_stress,'peak_MPa'), (self.maximum_stress,'regional_peak_rms_MPa')) if active)
        if not names and not self.limits and not self.extra_objectives:
            raise ValueError('Choose at least one structural objective')
        if self.maximum_stress and response is not None and 'peak_equivalent_plastic_strain' in response:
            names+=('peak_equivalent_plastic_strain',)
        return tuple(dict.fromkeys([*names,*extra]))


@dataclass
class EvaluatedSurface:
    mesh: object
    responses: list
    evidence: list
    name: str
    first_evidence: list | None = None
    regions: dict | None = None
    limit_evidence: dict = field(default_factory=dict)
    verify_margin: bool = False


def objective_bounds(candidate,targets):
    """Bound an unsampled region by zero and the global maximum of this discrete stress field."""
    lower=[]
    upper=[]
    for level in candidate.responses:
        for result in level:
            extra=candidate.limit_evidence.get(result.get('scenario'),())
            primary=[]
            for metric in targets.names(result):
                sense=targets.direction(result,metric)
                # Retain the least favorable measured value of each objective when a
                # target recheck also measured stiffness, stress, or receiver motion.
                refined=[sense*measured(row,metric) for row in extra if metric in row]
                primary.append(max([sense*measured(result,metric),*refined]))
            lower.extend(primary)
            upper.extend(primary)
            if targets.maximum_stress:
                peaks=np.asarray(result.get('local_peaks_MPa',()),dtype=float)
                if np.any(peaks<0.):
                    raise ValueError('Invalid local stress magnitude')
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
    if targets.limits and candidates:
        violations = [violation(item, targets.limits) for item in candidates]
        lowest = min(violations)
        candidates = [item for item,v in zip(candidates,violations,strict=True) if v == lowest]
    bounds=[objective_bounds(candidate,targets) for candidate in candidates]
    if any(np.any(~np.isfinite(pair)) for pair in bounds):
        raise ValueError('Invalid structural optimization response')
    return [candidate for candidate,(lower,_) in zip(candidates,bounds,strict=True)
            if not any(np.all(upper<=lower) and np.any(upper<lower) for _,upper in bounds)]


def objective_scales(reference,targets):
    """Use one physical stress unit per case across every mesh and every local maximum."""
    case_scales=[]
    for index,response in enumerate(reference.responses[0]):
        cases=[row[index] for row in reference.responses]
        checks=reference.limit_evidence.get(response.get('scenario'),())
        stress_scale=max((case.get('peak_MPa',0.) for case in (*cases,*checks)),default=0.)
        # With zero baseline stress the study unit (MPa) is the available physical scale.
        stress_scale=stress_scale if stress_scale>0. else 1.
        primary=[]
        for key in targets.names(response):
            if key in ('peak_MPa','regional_peak_rms_MPa'):
                magnitude=stress_scale
            else:
                magnitude=max([c[key] for c in cases]+[c[key] for c in checks if key in c])
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
    if offset!=values.shape[1]:
        raise ValueError('Incomplete objective layout')
    average=np.mean(scenario_means,axis=0) if scenario_means else np.zeros(len(candidates))
    worst=regret.max(axis=1,initial=0.)
    scores = {}
    for item,w,a in zip(candidates,worst,average,strict=True):
        rank = (float(w),float(a))
        scores[item.name] = (violation(item,targets.limits), *rank) if targets.limits else rank
    return scores


