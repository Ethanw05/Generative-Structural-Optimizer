"""Area-averaged absolute acceleration on named CAD selections, in standard gravity."""
from dataclasses import dataclass
import numpy as np
from .surface_average import integrated_weights


@dataclass(frozen=True)
class AccelerationReceiver:
    name: str
    faces: tuple
    axis_world: tuple | None = None

    def __post_init__(self):
        if not self.name or not self.faces:
            raise ValueError('A receiver needs a name and CAD faces')
        if self.axis_world is not None:
            axis = np.asarray(self.axis_world, dtype=float)
            if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1.):
                raise ValueError('A directional receiver needs a world unit vector')

    @property
    def metric(self):
        return 'peak_acceleration_g@' + self.name


@dataclass(frozen=True)
class ShockTransmission:
    name: str
    source: str
    receiver: str

    @property
    def metric(self):
        return 'peak_transmission@' + self.name


class ReceiverPatch:
    def __init__(self, definition, system, tolerance, carrier, integration_tolerance):
        self.definition = definition
        self.node_count = len(system.points)
        area = np.zeros(self.node_count)
        for face in definition.faces:
            area += integrated_weights(system.local,system.skin,face,tolerance,integration_tolerance)
        self.nodes = np.flatnonzero(area > 0.)
        if not len(self.nodes) or (carrier is None and np.all(system.fixed[self.nodes])):
            raise ValueError(f'Receiver {definition.name} has no mobile material in the inertial frame')
        self.weight = area[self.nodes] / area[self.nodes].sum()

    def translation_map(self, basis):
        components = basis.reshape(3, self.node_count, -1)
        return np.einsum('j,ijk->ik', self.weight, components[:,self.nodes,:])

    def average(self, nodal):
        return nodal.reshape(3, self.node_count)[:,self.nodes] @ self.weight

    def magnitude_g(self, vectors):
        axis = self.definition.axis_world
        magnitude = np.linalg.norm(vectors,axis=-1) if axis is None else np.abs(vectors @ np.asarray(axis))
        return magnitude / 9806.65


def receiver_patches(case, system, tolerance):
    names = [item.name for item in case.receivers]
    if len(names) != len(set(names)):
        raise ValueError('Receiver names must be unique within each history')
    ratios = set()
    for pair in case.transmissions:
        if not pair.name or pair.name in ratios or pair.source == pair.receiver or {pair.source,pair.receiver}-set(names):
            raise ValueError('A transfer ratio needs a unique name and two distinct configured receivers')
        ratios.add(pair.name)
    integration_tolerance=min(.005,case.relative_tolerance*.1)
    return [ReceiverPatch(item,system,tolerance,case.carrier,integration_tolerance) for item in case.receivers]


def shock_metrics(case, peaks):
    result = {r.metric:float(value) for r,value in zip(case.receivers,peaks,strict=True)}
    for pair in case.transmissions:
        source = result['peak_acceleration_g@'+pair.source]
        if source <= 0.:
            raise ValueError(f'Undefined shock transfer ratio at zero source acceleration: {pair.source}')
        result[pair.metric] = result['peak_acceleration_g@'+pair.receiver] / source
    return result
