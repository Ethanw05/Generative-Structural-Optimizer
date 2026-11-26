"""Case-by-case mechanical feedback on a single evolving solid."""
from dataclasses import dataclass, field, replace
from pathlib import Path
import json
import subprocess
import tempfile
import numpy as np

from .analysis_mesh import digest, volume_mesh
from .cad_history import export_history, UnresolvedHistory
from .finite_solid import UnresolvedSolid
from .surface_objectives import EvaluatedSurface, objective_scales, compromise_scores, pareto
from .mesh_quality import InadmissibleBoundary, validate_single_part
from .reference_regions import ReferenceRegions
from .tetra_check import solve
from .tetra_elasticity import assemble


@dataclass
class ScenarioAdditions:
    loads: dict = field(default_factory=dict)
    histories: dict = field(default_factory=dict)
    solid_laws: dict = field(default_factory=dict)
    case_materials: dict = field(default_factory=dict)

    def apply(self, study):
        new = set(self.loads) | set(self.histories)
        if new & set(study.scenarios()) or self.loads.keys() & self.histories.keys():
            raise ValueError('A new load or history needs a unique name')
        if (self.solid_laws.keys()|self.case_materials.keys())-new:raise ValueError('Assign material definitions to new scenarios')
        materials={**study.case_materials,**self.case_materials}
        return replace(study, case_materials=materials, solid_laws={**study.solid_laws, **self.solid_laws}, loads={**study.loads, **self.loads}, histories={**study.histories, **self.histories})


def case_order(executable, labels, seed, sweep):
    names = sorted(labels)
    result = subprocess.run([executable, 'scenario-order', str(len(names)), str(seed), str(sweep)],
                            capture_output=True, text=True, check=True)
    indices = [int(index) for index in result.stdout.split()]
    if sorted(indices) != list(range(len(names))):
        raise RuntimeError('The case sequence is incomplete')
    return [names[index] for index in indices]


@dataclass(frozen=True)
class CaseField:
    geometry: str
    label: str
    response: dict
    path: Path


class FeedbackAnalysis:
    """Case fields are valid only for the geometry and immutable case that produced them."""
    def __init__(self, study, baseline, pitch, spacing, work, solver):
        self.study = study
        self.baseline = baseline
        self.analysis_pitch = pitch
        self.edge = .95*pitch
        self.envelope = .25*spacing
        self.surface_tol = 1.5*spacing
        self.workspace = Path(work)
        self.run = Path(tempfile.mkdtemp(prefix='feedback-', dir=work))
        self.solver = solver
        self.regions = {}
        self.fields = {}
        self.proposal_resolution = {}
        self.baseline_id = digest(baseline)

    def proposal_field(self,mesh,identity,label):
        if label in self.proposal_resolution:
            return self.field(mesh,identity,label,self.proposal_resolution[label])
        return self.field(mesh,identity,label)

    def field(self, mesh, identity, label, resolution=None):
        resolution=resolution or (self.edge,self.envelope)
        key = (identity,label,resolution)
        if key in self.fields:
            return self.fields[key]
        if identity != self.baseline_id and (self.baseline_id,label,resolution) not in self.fields:
            self.field(self.baseline,self.baseline_id,label,resolution)
        if label not in self.regions:
            self.regions[label] = ReferenceRegions()
        print(f'Analyzing {label} on {identity[:12]}', flush=True)
        p, t = volume_mesh(mesh,self.workspace,*resolution)
        path = self.run/f'field-{len(self.fields)}'
        region = self.regions[label]
        active=self.study.for_case(label)
        if label in self.study.histories:
            system = assemble(p, t, active, self.surface_tol)
            response = export_history(system, active, active.histories[label], path,
                                      self.solver, region, self.surface_tol, law=self.study.solid_laws.get(label), label=label)
        else:
            single = replace(active, loads={label: active.loads[label]})
            response = solve(p, t, single, path, self.solver, surface_tol=self.surface_tol, local_regions=region,
                             report_to_console=False)
        response['verification_edge_mm']=resolution[0]
        evidence = CaseField(identity, label, response, path.with_suffix('.npz'))
        self.fields[key] = evidence
        return evidence

    def record(self, event):
        with (self.run/'lineage.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, allow_nan=False)+'\n')

    def start(self, current):
        np.savez(self.run/'reference.npz', v=self.baseline.vertices, f=self.baseline.faces,
                 geometry=self.baseline_id)
        self.checkpoint(current, dict(child=digest(current), sweep=0))

    def checkpoint(self, mesh, event):
        with tempfile.TemporaryDirectory(dir=self.run) as temporary:
            path = Path(temporary)/'current.npz'
            np.savez(path, v=mesh.vertices, f=mesh.faces, geometry=event['child'],
                     parent=event.get('parent',''), sweep=event['sweep'], label=event.get('case',''))
            path.replace(self.run/'current.npz')


def _read_geometry(path):
    import trimesh
    with np.load(path, allow_pickle=False) as data:
        mesh = trimesh.Trimesh(data['v'], data['f'], process=False)
        if 'geometry' in data and digest(mesh) != str(data['geometry']):
            raise ValueError('The stored mesh does not match its recorded geometry')
    validate_single_part(mesh)
    return mesh


def resume_geometry(directory):
    """Read the last working mesh, including the initial state of a stationary run."""
    return _read_geometry(Path(directory)/'current.npz')


def resume_pair(directory):
    """Retain the original reference for comparisons when a restarted study adds cases."""
    current = resume_geometry(directory)
    reference = Path(directory)/'reference.npz'
    return (_read_geometry(reference) if reference.is_file() else current), current


def mutation_weights(history, targets, label=None):
    if history:
        # Exploratory dynamic stress-energy envelope. Each changed solid receives a fresh history solve.
        return (0., 0., 0., 1.)
    stiffness = float(targets.stiffness)
    stress = float(targets.uses_stress(label))
    if not stiffness and not stress:
        # Explore mechanically active boundary directions for user-defined scalar responses.
        stiffness,stress = 1.,1.
    return (stiffness, .5*stress, 0., 0., .5*stress)


def lower_rank(delta, tolerance=1.e-9):
    """Lexicographic decrease of normalized ranks, ignoring differences at solver roundoff."""
    for value in delta:
        if not np.isfinite(value):
            return bool(value < 0.)
        if value < -tolerance:
            return True
        if value > tolerance:
            return False
    return False


def feedback_step(analysis, current, identity, label, targets, exe, pitch, spacing, propose, sweep, ordinal):
    parent = analysis.proposal_field(current, identity, label)
    if parent.geometry != identity or parent.label != label:
        raise ValueError('Mutation evidence does not belong to the current geometry and case')
    event = dict(sweep=sweep, step=ordinal, case=label, parent=identity, field_geometry=parent.geometry,
                 parent_peak_MPa=parent.response.get('peak_MPa'),proposal_edge_mm=parent.response.get('verification_edge_mm'))
    history = label in analysis.study.histories
    directions = [('requested', mutation_weights(history, targets,label))]
    if history:
        quantities = [*targets.names(parent.response),
                      *(getattr(limit,'metric','') for limit in targets.limits if label in limit.cases)]
        if any(key.startswith(('peak_acceleration_g@','peak_transmission@')) for key in quantities):
            # Either stiffening or softening a stressed region can reduce receiver shock.
            # The actual transient on each proposed shape determines its ranking.
            directions.append(('reverse dynamic energy',(0.,0.,0.,-1.)))
    if not history and targets.stiffness and targets.maximum_stress:
        # Step reduction changes magnitude only. Probe the separate shape
        # derivatives if the combined direction trades away another objective.
        directions.extend([
            ('stress', mutation_weights(False, replace(targets, stiffness=False))),
            ('stiffness', mutation_weights(False, replace(targets, maximum_stress=False,limits=()))),
        ])

    def candidate(field, name):
        return EvaluatedSurface(None, [[field.response]], [], name)

    before = candidate(parent, 'parent')
    reference = analysis.proposal_field(analysis.baseline, analysis.baseline_id, label)
    scale = objective_scales(candidate(reference, 'reference'), targets)
    best = None
    seen = {}
    event['trials'] = []
    event['directions_searched'] = []
    with np.load(parent.path) as evidence:
        for direction, weights in directions:
            event['directions_searched'].append(direction)
            move = pitch
            dominates = False
            while move >= spacing:
                try:
                    child = propose(analysis.study, current, [evidence], exe, analysis.run, spacing,
                                    move, weights, analysis.analysis_pitch)
                    if child is None:
                        break
                    if child.volume > analysis.study.volume_limit_mm3*(1.+1.e-6):
                        raise InadmissibleBoundary('Candidate exceeds the material budget')
                    child_id = digest(child)
                    if child_id != identity:
                        if child_id not in seen:
                            result = analysis.proposal_field(child, child_id, label)
                            after = candidate(result, 'trial')
                            front = pareto([before, after], targets)
                            scores = compromise_scores([before, after], targets, scale)
                            delta = tuple(a-b for a,b in zip(scores['trial'],scores['parent'],strict=True))
                            improves = any(item is after for item in front) and lower_rank(delta)
                            seen[child_id] = (improves, len(front)==1 and front[0] is after)
                            event['trials'].append(dict(geometry=child_id, direction=direction,
                                                       move_mm=move, regret=scores))
                            if best is None or lower_rank(tuple(a-b for a,b in zip(delta,best[0],strict=True))):
                                best = (delta, child, result, move, direction)
                        improves, dominates = seen[child_id]
                        if improves:
                            break
                except (InadmissibleBoundary, UnresolvedHistory, UnresolvedSolid) as error:
                    event['last_rejection'] = str(error)
                move *= .5
            if dominates:
                break
    if best is not None:
        delta, child, result, move, direction = best
        event.update(status='advanced', child=result.geometry, move_mm=move, direction=direction,
                     selection='improving' if lower_rank(delta) else 'exploratory',
                     child_peak_MPa=result.response.get('peak_MPa'))
        analysis.checkpoint(child, event)
        analysis.record(event)
        return child, result.geometry, event
    event.update(status='retained', child=identity)
    analysis.record(event)
    return current, identity, event
