"""Transfer analysis sensitivities to the native surface solver and extend its normal velocity."""
import subprocess

import numpy as np
import trimesh
from .surface_locator import SurfaceLocator
from .cad_motion import cad_bounds
from .cad_wrench import SurfaceWrench


class SurfaceMotion:
    """Piecewise-linear motion on the analysis triangles, evaluated at the closest surface point."""

    def __init__(self, vertices, faces, values):
        self.vertices = vertices
        self.faces = faces
        self.values = values
        self.locator = SurfaceLocator(vertices, faces)

    def sample(self, points):
        if len(points) == 0:
            return np.empty(0)
        cells, projected = self.locator.nearest(points)
        triangles = self.faces[cells]
        barycentric = trimesh.triangles.points_to_barycentric(self.vertices[triangles], projected)
        return np.einsum('ij,ij->i', self.values[triangles], barycentric)


def prescribed_faces(study):
    loads = [*study.loads.values(), *(item.attachment for history in study.histories.values()
                                      for item in history.loads)]
    faces = [*study.fixed_faces, *(face for load in loads for face in
                                   (load.faces if isinstance(load, SurfaceWrench) else (load.face,)))]
    return {id(face): face for face in faces}.values()


def solve_motion(study, evidence, executable, work, move, spacing, weights, available_volume, regularization_mm=0.):
    vertices = (evidence[0]['v'] - study.world_origin) @ study.world_rotation.T
    faces = evidence[0]['f']
    triangles = vertices[faces]
    area = trimesh.triangles.area(triangles)
    centers = triangles.mean(axis=1)
    if np.any(area <= 0.):
        raise ValueError('Degenerate analysis boundary')
    locked = np.zeros(len(faces), dtype=bool)
    for face in prescribed_faces(study):
        locked |= face.contains_projection(centers, spacing / 2.)

    drive = np.zeros(len(faces))
    for case in evidence:
        if not (np.array_equal(case['v'], evidence[0]['v']) and np.array_equal(case['f'], faces)):
            raise ValueError('Scenario sensitivities require a common boundary')
        if "finite_strain" in case:
            # Finite-strain energy and stress envelopes provide a proposal direction.
            # Every accepted candidate is evaluated with the same nonlinear solid law.
            contributions = [(weights[0],case['elastic_work_density'])] if len(weights)==5 else []
            stress_weight=weights[1]+weights[4] if len(weights)==5 else weights[3]
            contributions.append((stress_weight,case['vm']**2))
            if 'equivalent_plastic_strain' in case:
                contributions.append((stress_weight,case['equivalent_plastic_strain']))
            for weight,signal in contributions:
                norm=np.average(signal,weights=area)
                if weight and norm>0.:drive+=weight*signal/(norm*len(evidence))
            continue
        if "transient" in case:
            # The spatial envelope supplies an exploratory energy direction. The full
            # history is resolved again on each proposal before Pareto selection.
            if len(weights)>=4 and weights[3]>0.:
                energy=case["vm"]**2
                norm=np.average(energy,weights=area)
                if norm>0.:drive+=weights[3]*energy/(norm*len(evidence))
            continue
        keys = ('elastic_work_density', 'stress_descent', 'regional_descent', 'distortional_energy', 'balanced_descent')
        if len(weights) not in (3, 4, 5):
            raise ValueError('Invalid boundary objective weights')
        for weight, key in zip(weights, keys[:len(weights)], strict=True):
            if weight == 0.:
                continue
            signal = case['vm']**2 if key == 'distortional_energy' else case[key]
            if signal.shape != drive.shape or not np.all(np.isfinite(signal)):
                raise ValueError('Invalid objective sensitivity')
            scale = np.average(np.abs(signal), weights=area)
            if scale > 0.:
                drive += (weight / len(evidence)) * (signal / scale)

    source = work / 'surface-drive.txt'
    destination = work / 'surface-motion.txt'
    with source.open('w') as stream:
        stream.write(f'{len(vertices)} {len(faces)}\n')
        np.savetxt(stream, vertices, fmt='%.17g')
        np.savetxt(stream, np.column_stack((faces, drive, locked)), fmt=('%d', '%d', '%d', '%.17g', '%d'))
        stream.write('bounds\n')
        np.savetxt(stream, cad_bounds(study.space, vertices, faces, move, spacing), fmt='%.17g')
        stream.write(f'regularization_mm {regularization_mm:.17g}\n')
    subprocess.run([executable, 'surface-motion', source, destination, str(move), str(available_volume)],
                   check=True)
    values = np.loadtxt(destination, ndmin=1)
    if values.shape != (len(vertices),) or not np.all(np.isfinite(values)):
        raise ValueError('Invalid native surface motion')
    return SurfaceMotion(vertices, faces, values)
