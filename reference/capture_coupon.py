"""Record the density stage of the tests/design_inputs.py coupon as a reference for the C++ port.

This script does not change the pipeline. It builds the same study as tests/design_inputs.py,
wraps a few module-level functions to record what passes through them, and writes everything
to reference/coupon/. It is deleted together with the rest of the Python once the port is done.

Usage: python reference/capture_coupon.py build/Release/gso.exe
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
import manifold3d as md
import trimesh
from gso_workflows import volume_study
from gso_workflows.design_space import DesignSpace, Face
from gso_workflows.volume_study import VolumeStudy, SurfaceLoad, optimize, export_design

OUT = ROOT / 'reference' / 'coupon'
WORK = OUT / 'work'


def solid_to_obj(solid, path):
    raw = solid.to_mesh64()
    trimesh.Trimesh(raw.vert_properties[:, :3], raw.tri_verts, process=False).export(path, digits=17)


def build_study():
    """Same geometry, faces, material and budget as tests/design_inputs.py."""
    box = md.Manifold.cube([6., 4., 4.])
    void = md.Manifold.cube([1., .5, .5]).translate([5., 1.75, 1.75])
    surface = (box - void).to_mesh64()
    mesh = trimesh.Trimesh(surface.vert_properties[:, :3], surface.tri_verts, process=False)
    mounted = Face.pick(mesh, [0, 2, 2], [-1, 0, 0])
    end = Face.pick(mesh, [6, 1, 1], [1, 0, 0])
    frame = end.prism(-1.)
    preserved = frame - void
    space = DesignSpace(box, (preserved,), (void,))
    force = np.array([0., 0., 1.])
    study = VolumeStudy(space, (mounted,), {'load A': SurfaceLoad(end, force)},
                        2000., .3, 70., np.eye(3), np.zeros(3))
    solid_to_obj(box, OUT / 'envelope.obj')
    solid_to_obj(preserved, OUT / 'preserved.obj')
    solid_to_obj(void, OUT / 'excluded.obj')
    mesh.export(OUT / 'part.obj', digits=17)
    definition = dict(
        envelope='envelope.obj', preserved=['preserved.obj'], excluded=['excluded.obj'],
        fixed_faces=[dict(pick_point=[0, 2, 2], pick_normal=[-1, 0, 0])],
        loads={'load A': dict(face=dict(pick_point=[6, 1, 1], pick_normal=[1, 0, 0]),
                              force_n=force.tolist())},
        young_mpa=2000., poisson=.3, volume_limit_mm3=70.,
        pitch_mm=.5, iterations=12, refinement_steps=6)
    (OUT / 'study.json').write_text(json.dumps(definition, indent=2))
    return study, definition


def install_recorders(record):
    """Wrap the functions whose inputs and outputs the C++ port must reproduce."""
    original_distribute = volume_study.distribute_surface_force
    original_stencil = volume_study.stencil
    original_refine = volume_study.refine
    original_sample = DesignSpace.sample
    original_savez = np.savez_compressed

    def distribute(face, force, origin, shape, pitch, lookup):
        result = original_distribute(face, force, origin, shape, pitch, lookup)
        record['force'] = result.copy()
        record['lookup'] = lookup.copy()
        return result

    def stencil(executable, work, pitch, young, poisson):
        result = original_stencil(executable, work, pitch, young, poisson)
        record[f'stencil_{pitch}_{young}_{poisson}'] = result.copy()
        return result

    def sample(self, origin, shape, pitch):
        allowed, protected = original_sample(self, origin, shape, pitch)
        record['allowed'], record['protected'] = allowed.copy(), protected.copy()
        record['grid_origin'], record['grid_shape'] = np.asarray(origin), np.asarray(shape)
        return allowed, protected

    def refine(density, *args, **kwargs):
        record['density_before_refine'] = density.copy()
        result = original_refine(density, *args, **kwargs)
        record['density_after_refine'] = result.copy()
        return result

    def savez(path, **arrays):
        if 'generation' in arrays:
            record.setdefault('density_by_generation', []).append(arrays['density'].copy())
        return original_savez(path, **arrays)

    volume_study.distribute_surface_force = distribute
    volume_study.stencil = stencil
    volume_study.refine = refine
    DesignSpace.sample = sample
    volume_study.np.savez_compressed = savez


def main():
    executable = Path(sys.argv[1]).resolve(strict=True)
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    for stale in WORK.glob('*'):
        stale.unlink()
    study, definition = build_study()
    record = {}
    install_recorders(record)
    design = optimize(study, definition['pitch_mm'], definition['iterations'], executable, WORK,
                      refinement_steps=definition['refinement_steps'])
    summary = export_design(study, design, OUT / 'surface.obj')
    arrays = {key: value for key, value in record.items() if key != 'density_by_generation'}
    arrays['density_by_generation'] = np.stack(record['density_by_generation'])
    arrays['final_grid'] = design.field
    np.savez_compressed(OUT / 'density_stage.npz', **arrays)
    (OUT / 'summary.json').write_text(json.dumps(dict(
        surface=summary, arrays={key: list(np.shape(value)) for key, value in arrays.items()}), indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
