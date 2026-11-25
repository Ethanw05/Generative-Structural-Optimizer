"""Regression for multiple peaks, constant plateaus, and regional and global shape objectives."""
from pathlib import Path
import subprocess
import sys
import tempfile
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root/'build/sunbeam33/deps'), str(root/'tools')]
import numpy as np
from gso_workflows.stress_regions import stress_basins, regional_stress


def check_stress_regions():
    heights = np.array([1., 5., 5., 1., 9., 2., 6., 1., 7., 3., 100., 1.])
    tets = np.arange(len(heights))[:, None] + np.arange(4)
    basins = stress_basins(tets, heights)
    peaks = np.zeros(basins.max() + 1)
    np.maximum.at(peaks, basins, heights)
    np.testing.assert_equal(np.sort(peaks), [5., 6., 7., 9., 100.])
    np.testing.assert_equal(basins[1], basins[2])
    weights = np.linspace(.8, 1.3, len(heights))
    value, signal, volume_partial = regional_stress(heights, weights, basins, 8.)
    for i in range(len(weights)):
        dx = np.eye(len(weights))[i] * 1.e-5
        plus = regional_stress(heights + dx, weights, basins, 8.)[0]
        minus = regional_stress(heights - dx, weights, basins, 8.)[0]
        np.testing.assert_allclose((plus-minus)/2.e-5, weights[i]*signal[i], atol=2.e-8)
        plus = regional_stress(heights, weights + dx, basins, 8.)[0]
        minus = regional_stress(heights, weights - dx, basins, 8.)[0]
        np.testing.assert_allclose((plus-minus)/2.e-5, volume_partial[i], atol=2.e-8)
    np.testing.assert_allclose(weights @ volume_partial, 0., atol=1.e-12)
    assert value < peaks.max()
    rng = np.random.default_rng(8)
    perm = rng.permutation(len(tets))
    reordered = stress_basins(tets[perm], heights[perm])[np.argsort(perm)]
    np.testing.assert_equal(basins[:, None] == basins, reordered[:, None] == reordered)
    plateau = stress_basins(tets[:5], np.array([1., 2., 2., 2., 3.]))
    assert np.unique(plateau).size == 1
    tied = stress_basins(tets[:3], np.array([5., 1., 5.]))
    np.testing.assert_allclose(tied.toarray(), [[1., 0.], [.5, .5], [0., 1.]])


def check_motion(executable):
    # One connected strip, driven in five places. Raising the largest peak must
    # preserve movement at the four smaller peaks. The last triangle is locked.
    vertices = np.array([[x, y, 0.] for x in range(43) for y in (0., 1.)])
    faces = np.array([(a, a+2, a+1) if half == 0 else (a+1, a+2, a+3)
                      for a in range(0, 84, 2) for half in (0, 1)])
    x = vertices[faces, 0].mean(1)
    signal = np.zeros(len(faces))
    for center in (4, 12, 20, 28, 36):
        signal[np.abs(x-center) < 1.5] = 3.
    signal[np.abs(x-20) < 1.5] = 300.
    locked = np.zeros(len(faces), bool)
    locked[-1] = True
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'build') as folder:
        table, output = (Path(folder)/n for n in ('motion.in', 'motion.out'))
        def evaluate(drive, bounds=None):
            with table.open('w') as stream:
                stream.write(f'{len(vertices)} {len(faces)}\n')
                np.savetxt(stream, vertices)
                np.savetxt(stream, np.column_stack((faces,drive,locked)), fmt=('%d','%d','%d','%.17g','%d'))
                if bounds is not None:
                    stream.write('bounds\n')
                    np.savetxt(stream, bounds)
            subprocess.run([executable,'surface-motion',table,output,'1','0'],check=True)
            return np.loadtxt(output)
        motion = evaluate(signal)
        np.testing.assert_allclose(motion[faces[-1]], 0.)
        for center in (4, 12, 20, 28, 36):
            assert motion[vertices[:,0] == center].min() > .25
        assert abs(motion).max() <= 1.
        assert np.mean(motion[faces],axis=1).sum() <= 1.e-5
        np.testing.assert_allclose(evaluate(np.ones_like(signal)), 0., atol=3.e-6)

        limits = np.tile([-1., 1.], (len(vertices), 1))
        limits[vertices[:,0] == 4, 1] = 0.
        limits[vertices[:,0] == 12, 0] = 0.
        constrained = evaluate(signal, limits)
        assert np.max(constrained[vertices[:,0] == 4]) <= 0.
        assert np.min(constrained[vertices[:,0] == 12]) >= 0.


def check_normal_transport(executable):
    from gso_workflows.front_transport import advance
    x = np.indices((25, 5, 5), dtype=np.float32)[0]
    free = np.full_like(x, 1000.)
    initial = x - 6.
    speed = np.full_like(x, 2.)
    translated = advance(initial, speed, free, -free, 1., executable)
    # Six upwind substeps have a six-cell numerical domain of dependence.
    np.testing.assert_allclose(translated[7:18], initial[7:18]-2., atol=2.e-6)
    contracted = advance(initial, -speed, free, -free, 1., executable)
    np.testing.assert_allclose(contracted[7:18], initial[7:18]+2., atol=2.e-6)
    gap = np.minimum(x-4., 20.-x)
    isolated = np.where(np.abs(x-12.) < 2., 15., 0.).astype(np.float32)
    after = advance(gap, isolated, free, -free, 1., executable)
    assert np.all(after[9:16] > 0.)
    blocked = advance(initial, speed*4., free, x-8., 1., executable)
    assert np.all(blocked[8:] >= 0.)


if __name__ == '__main__':
    check_stress_regions()
    check_motion(sys.argv[1])
    check_normal_transport(sys.argv[1])

# An unsampled region has no measured improvement on that mesh.
from gso_workflows.reference_regions import compare_regions
old_region = {'local_peaks_MPa':[8.,5.,1.], 'local_region_volumes_mm3':[1.,1.,1.]}
new_region = {'local_peaks_MPa':[3.,6.,0.], 'local_region_volumes_mm3':[1.,1.,0.]}
comparison = compare_regions([old_region],[new_region])
assert comparison['lower_on_all_meshes'] == 1 and comparison['higher_on_all_meshes'] == 1
assert comparison['changes_MPa_by_mesh'] == [[-5.,1.,None]]

# A newly loaded, formerly quiet region cannot outweigh a much larger stress in MPa.
from gso_workflows.reference_regions import relative_stress
labels = np.array([0, 0, 1, 1])
volume = np.array([0.7, 1.3, 0.9, 1.1])
reference = np.array([100., 1.e-6])
stress = np.array([90., 95., 0.05, 0.07])
value, signal, volume_partial = relative_stress(stress, volume, labels, reference, 8.)
assert .9 < value < .95 and np.max(signal[2:]) < np.min(signal[:2])
for i in range(len(stress)):
    delta = np.zeros_like(stress)
    delta[i] = 1.e-4
    plus = relative_stress(stress + delta, volume, labels, reference, 8.)[0]
    minus = relative_stress(stress - delta, volume, labels, reference, 8.)[0]
    np.testing.assert_allclose((plus-minus)/2.e-4, volume[i]*signal[i], atol=1.e-9)
    plus = relative_stress(stress, volume + delta, labels, reference, 8.)[0]
    minus = relative_stress(stress, volume - delta, labels, reference, 8.)[0]
    np.testing.assert_allclose((plus-minus)/2.e-4, volume_partial[i], atol=1.e-9)
