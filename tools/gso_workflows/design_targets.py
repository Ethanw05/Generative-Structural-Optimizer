"""Case-scoped hard limits and scalar objectives on verified solid responses."""
from dataclasses import dataclass
from math import isfinite


def validate_scope(cases):
    if not isinstance(cases, tuple) or not cases or any(not isinstance(c,str) or not c for c in cases):
        raise ValueError('Select one or more named study scenarios')
    if len(set(cases)) != len(cases):
        raise ValueError('Scenario names in a target must be distinct')


def positive(value):
    if not isfinite(value) or value <= 0.:
        raise ValueError('Target values must be finite and positive')


def measured(response, key):
    if key not in response:
        raise ValueError(f'Scenario {response.get("scenario")} did not measure {key}')
    value = float(response[key])
    if not isfinite(value) or value < 0.:
        raise ValueError(f'Invalid magnitude for {key}')
    return value


@dataclass(frozen=True)
class YieldSafety:
    cases: tuple
    yield_mpa: float
    minimum: float
    strength_basis: str

    def __post_init__(self):
        validate_scope(self.cases)
        positive(self.yield_mpa)
        positive(self.minimum)
        if not self.strength_basis.strip():
            raise ValueError('Identify the material and basis of its yield strength')

    def assessment(self, response):
        stress = measured(response, 'peak_MPa')
        plastic = measured(response,'peak_equivalent_plastic_strain') if 'peak_equivalent_plastic_strain' in response else 0.
        factor = self.yield_mpa/stress if stress else None
        utilization = self.minimum*stress/self.yield_mpa
        excess = max(0.,utilization-1.)
        if plastic:
            # Stress redistribution can hide earlier yield. A yielded candidate
            # cannot qualify on an elastic strength-to-stress margin.
            modulus = measured(response['solid_model'],'young_MPa')
            positive(modulus)
            excess = max(excess,plastic/(self.yield_mpa/modulus))
        return dict(quantity='whole-part yield factor of safety', minimum=self.minimum,
                    yield_MPa=self.yield_mpa, strength_basis=self.strength_basis, peak_MPa=stress,
                    achieved=factor if not plastic else None, unbounded=stress==0. and plastic==0.,
                    plastic_flow_detected=plastic>0.,utilization=utilization,violation=excess)


@dataclass(frozen=True)
class ResponseLimit:
    cases: tuple
    metric: str
    maximum: float

    def __post_init__(self):
        validate_scope(self.cases)
        positive(self.maximum)
        if not self.metric:
            raise ValueError('Name the measured response to limit')

    def assessment(self, response):
        value = measured(response, self.metric)
        utilization=value/self.maximum
        return dict(quantity=self.metric, achieved=value, maximum=self.maximum,
                    utilization=utilization,violation=max(0.,utilization-1.))


@dataclass(frozen=True)
class ScalarObjective:
    cases: tuple
    metric: str

    def __post_init__(self):
        validate_scope(self.cases)
        if not self.metric:
            raise ValueError('Name the measured response to minimize')


class MinimizeResponse(ScalarObjective):
    direction = 1


class MaximizeResponse(ScalarObjective):
    direction = -1


def assessments(candidate, limits, *, require_all=False):
    """The most heavily utilized mesh controls each case and limit independently."""
    if limits and any('scenario' not in row for level in candidate.responses for row in level):
        raise ValueError('Case-scoped limits require labeled mechanical results')
    rows = []
    for limit in limits:
        for label in limit.cases:
            checks = []
            for level, responses in enumerate(candidate.responses):
                matches = [r for r in responses if r.get('scenario') == label]
                if len(matches) == 1:
                    checks.append(dict(level=level, scenario=label, **limit.assessment(matches[0])))
                elif matches or require_all:
                    raise ValueError(f'Expected exactly one verified response for {label} on level {level}')
            if checks:
                if len(checks) != len(candidate.responses):
                    raise ValueError(f'Incomplete mesh verification for {label}')
                worst = max(checks, key=lambda check:(check['violation'],check['utilization']))
                rows.append(dict(**worst, met=worst['violation'] == 0.,
                                 utilization_by_mesh=[r['utilization'] for r in checks]))
    return rows


def violation(candidate, limits):
    return max((r['violation'] for r in assessments(candidate, limits)),default=0.)
