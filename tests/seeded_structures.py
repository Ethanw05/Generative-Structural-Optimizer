"""Seeded solid studies with independent mechanics and load-transfer checks."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'build/sunbeam33/meshlib-runtime'),
                str(ROOT/'build/sunbeam33/deps')]

import manifold3d as md
import numpy as np
import trimesh
from scipy.sparse.linalg import spsolve

from gso_workflows.analysis_mesh import digest, volume_mesh
from gso_workflows.boundary_shape import propose
from gso_workflows.cad_wrench import SurfaceWrench
from gso_workflows.design_space import DesignSpace, Face
from gso_workflows.mesh_quality import validate_single_part
from gso_workflows.scenario_feedback import FeedbackAnalysis, feedback_step
from gso_workflows.surface_objectives import (ShapeTargets, EvaluatedSurface, pareto,
                                              objective_scales, compromise_scores)
from gso_workflows.tetra_check import solve
from gso_workflows.volume_study import SurfaceLoad, VolumeStudy


def surface(solid):
    native = solid.to_mesh64()
    return trimesh.Trimesh(native.vert_properties[:, :3], native.tri_verts, process=False)


def cylinder(radius, height, z=0.):
    return md.Manifold.cylinder(height, radius, radius, 40).translate([0., 0., z])


def shaft(rng, executable):
    radius = rng.uniform(1.85, 2.2)
    bore = rng.uniform(.62, .85)
    length = rng.uniform(5.1, 5.9)
    moment = rng.uniform(7., 14.)
    young, poisson = 2100., .32
    hole = cylinder(bore, length+2., -1.)
    solid = cylinder(radius, length) - hole
    mesh = surface(solid)
    mid = .5*(radius+bore)
    bottom = Face.pick(mesh, [mid, 0., 0.], [0., 0., -1.])
    top = Face.pick(mesh, [mid, 0., length], [0., 0., 1.])
    envelope = cylinder(radius+.7, length)
    space = DesignSpace(envelope, (bottom.prism(-.3), top.prism(-.3)), (hole,))
    torque = SurfaceWrench((top,), np.array([0.,0.,length]), np.zeros(3), np.array([0.,0.,moment]))
    study = VolumeStudy(space, (bottom,), {'twist':torque}, young, poisson,
                        solid.volume(), np.eye(3), np.zeros(3), executable)
    polar = .5*np.pi*(radius**4 - bore**4)
    exact_work = moment**2 * length / ((young/(2.*(1.+poisson))) * polar)
    return study, mesh, exact_work


def diaphragm(rng, executable):
    width, thickness = rng.uniform(5.4, 6.3), rng.uniform(1.1, 1.4)
    offset = rng.uniform(-.7, .7)
    boss_height, boss_radius = .65, rng.uniform(.75, 1.05)
    plate = md.Manifold.cube([width, width, thickness]).translate([-width/2., -width/2., 0.])
    boss = cylinder(boss_radius, boss_height, thickness).translate([offset,0.,0.])
    solid = plate + boss
    mesh = surface(solid)
    fixed = []
    for axis in (0, 1):
        for sign in (-1., 1.):
            normal = np.eye(3)[axis]*sign
            fixed.append(Face.pick(mesh, normal*(width/2.)+[0.,0.,thickness/2.], normal))
    top = Face.pick(mesh, [offset,0.,thickness+boss_height], [0.,0.,1.])
    envelope = md.Manifold.cube([width,width,thickness+boss_height+1.]).translate([-width/2.,-width/2.,-.5])
    space = DesignSpace(envelope, tuple(f.prism(-.25) for f in [*fixed,top]), ())
    load = SurfaceLoad(top, np.array([0.,0.,-rng.uniform(3.,8.)]))
    study = VolumeStudy(space, tuple(fixed), {'press':load}, 1800., .34,
                        solid.volume(), np.eye(3), np.zeros(3), executable)
    return study, mesh, None


def equilibrium(response):
    np.testing.assert_allclose(np.add(response['reaction_N'],response['force_N']), 0., atol=2.e-7)
    np.testing.assert_allclose(response['moment_residual_Nmm'], 0., atol=2.e-6)
    assert response['compliance_Nmm'] > 0. and response['peak_MPa'] > 0.


def exercise(factory, rng, executable, directory):
    directory.mkdir()
    study, original, exact_work = factory(rng, executable)
    label, load = next(iter(study.loads.items()))
    pitch, spacing = .72, .144
    analysis = FeedbackAnalysis(study, original, pitch, spacing, directory, spsolve)
    analysis.start(original)
    identity = digest(original)
    # A missed impact supplies an exactly zero transient stress envelope. Its
    # position in the event sequence must not mutate this already curved part.
    quiet = dict(v=original.vertices, f=original.faces, transient=True, vm=np.zeros(len(original.faces)))
    assert propose(study, original, [quiet], executable, directory, spacing, pitch,
                   (0., 0., 0., 1.), pitch) is None
    stock = analysis.field(original, identity, label)
    equilibrium(stock.response)
    if exact_work is not None:
        error = stock.response['compliance_Nmm']/exact_work - 1.
        print(f'Annular torsion baseline compliance error {error:+.2%}', flush=True)
        assert abs(error) < .10, error
    else:
        p, t = volume_mesh(original, directory, analysis.edge, analysis.envelope)
        doubled = replace(study, loads={label:replace(load, force_n=2.*load.force_n)})
        twice = solve(p,t,doubled,directory/'twice',spsolve,surface_tol=analysis.surface_tol,
                      report_to_console=False)
        np.testing.assert_allclose(twice['compliance_Nmm'], 4.*stock.response['compliance_Nmm'], rtol=1.e-8)
        np.testing.assert_allclose(twice['peak_MPa'], 2.*stock.response['peak_MPa'], rtol=1.e-8)
    changed, new_id, event = feedback_step(analysis, original, identity, label, ShapeTargets(), executable,
                                           pitch, spacing, propose, 1, 1)
    assert event['parent'] == event['field_geometry'] == identity
    primary = [trial for trial in event['trials'] if trial['direction'] == 'requested']
    if primary and all(tuple(t['regret']['trial']) >= tuple(t['regret']['parent']) for t in primary):
        assert 'stress' in event['directions_searched']
    print('Searched directions:',event['directions_searched'],'chosen:',event.get('direction'),flush=True)
    if new_id == identity:
        assert event['status'] == 'retained'
        print(factory.__name__, 'retained', event.get('last_rejection','stationary'), flush=True)
        return False
    assert digest(changed) == new_id and changed.volume <= study.volume_limit_mm3
    validate_single_part(changed)
    decision, rows = verified_pair(analysis, label, original, changed, directory, pitch, spacing)
    print(f'{factory.__name__} ({event["selection"]}), fine stock / trial [N mm, MPa]: {rows}; '
          f'two-mesh selection: {decision}', flush=True)
    if exact_work is not None:
        assert abs(rows[0][0]/exact_work - 1.) < .06
    return True


def verified_pair(analysis, label, original, changed, directory, pitch, spacing):
    # Validate both geometry hypotheses using the same reference stress regions
    # on independently remeshed solids and the production compromise selector.
    verified = []
    rows = []
    for name, part in (('stock', original), ('trial', changed)):
        coarse = analysis.field(part, digest(part), label)
        p,t = volume_mesh(part,directory,.60*pitch,.18*spacing)
        fine = solve(p,t,analysis.study,directory/f'fine-{name}',spsolve,
                     surface_tol=analysis.surface_tol,local_regions=analysis.regions[label],
                     report_to_console=False)
        equilibrium(fine)
        rows.append([fine['compliance_Nmm'],fine['peak_MPa']])
        verified.append(EvaluatedSurface(part,[[coarse.response],[fine]],[],name,regions=analysis.regions))
    targets = ShapeTargets()
    archive = pareto(verified, targets)
    scores = compromise_scores(archive, targets, objective_scales(verified[0], targets))
    selection = min(archive, key=lambda item:scores[item.name]).name
    return selection, rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('executable', type=Path)
    parser.add_argument('--seed', type=int, default=70241)
    parser.add_argument('--case', choices=['all','shaft','diaphragm'], default='all')
    args = parser.parse_args()
    print('Seed', args.seed, flush=True)
    trials = []
    with tempfile.TemporaryDirectory(dir=ROOT/'build') as tmp:
        for shape, child_seed in zip((shaft, diaphragm), np.random.SeedSequence(args.seed).spawn(2), strict=True):
            if args.case not in ('all',shape.__name__):
                continue
            trials.append(exercise(shape,np.random.default_rng(child_seed),
                                   args.executable.resolve(),Path(tmp)/shape.__name__))
    assert any(trials), 'No changed geometry reached an independent solve'
