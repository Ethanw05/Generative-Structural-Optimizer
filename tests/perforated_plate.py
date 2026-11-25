"""Four through-holes in a loaded plate exercise the full local-stress and surface-motion path."""
from itertools import permutations, product
from pathlib import Path
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root/'build/sunbeam33/meshlib-runtime'), str(root/'build/sunbeam33/deps'), str(root/'tools')]

import manifold3d as md
import numpy as np
import trimesh
from scipy.sparse.linalg import spsolve
from gso_workflows.design_space import DesignSpace, Face
from gso_workflows.surface_motion import solve_motion
from gso_workflows.reference_regions import ReferenceRegions
from gso_workflows.tetra_check import solve
from gso_workflows.volume_study import SurfaceLoad, VolumeStudy


def test_plate(executable):
    size = np.array([30, 7, 2])
    hole_starts = (4, 11, 18, 25)
    points = np.array(list(product(*(range(n+1) for n in size))), dtype=float)
    cells = []
    axes = np.eye(3, dtype=int)
    for corner in product(*(range(n) for n in size)):
        x, y, _ = corner
        if y in (2, 3) and any(h <= x < h+2 for h in hole_starts):
            continue
        for sequence in permutations(range(3)):
            chain = np.vstack((np.zeros(3, dtype=int), axes[list(sequence)].cumsum(axis=0)))
            cells.append(np.ravel_multi_index((chain+corner).T, size+1))
    solid = md.Manifold.cube(size.tolist())
    for h in hole_starts:
        solid -= md.Manifold.cube([2.,2.,4.]).translate([h,2.,-1.])
    reference = solid.to_mesh64()
    surface = trimesh.Trimesh(reference.vert_properties[:, :3], reference.tri_verts, process=False)
    left = Face.pick(surface, [0, 3.5, 1], [-1, 0, 0])
    right = Face.pick(surface, [30, 3.5, 1], [1, 0, 0])
    # The analysis starts with holes. This test leaves their walls free to evolve.
    space = DesignSpace(md.Manifold.cube([32, 9, 4]).translate([-1, -1, -1]), (), ())
    load = SurfaceLoad(right, np.array([14., 0., 0.]))
    study = VolumeStudy(space, (left,), {'tension':load}, 2200., .35,
                        float(len(cells))/6., np.eye(3), np.zeros(3))
    with tempfile.TemporaryDirectory(dir=root/'build') as folder:
        work = Path(folder)
        regions = ReferenceRegions()
        result = solve(points, np.asarray(cells), study, work/'plate', spsolve, surface_tol=.01, local_regions=regions)
        assert len(result['local_peaks_MPa']) >= len(hole_starts)
        np.testing.assert_allclose(max(result['local_peaks_MPa']), result['peak_MPa'])
        np.testing.assert_allclose(sum(result['local_region_volumes_mm3']), result['volume_mm3'])
        np.testing.assert_allclose(result['relative_regional_stress'], 1.)

        with np.load(work/'plate.npz') as field, np.load(work/'plate-peaks.npz') as peaks:
            motion = solve_motion(study, [field], executable, work, .2, .05, (0.,0.,1.), 0.)
            for h in hole_starts:
                local = np.abs(peaks['world_mm'][:, 0] - (h+1)) < 2.
                assert np.max(peaks['peak_MPa'][local]) > 1.5
                nearby = np.abs(motion.vertices[:, 0] - (h+1)) < 2.
                assert np.max(motion.values[nearby]) > .01
        assert result['peak_MPa'] >= result['regional_peak_rms_MPa'] >= result['regional_stress_MPa']
        np.testing.assert_allclose(result['reaction_N'], [-14.,0.,0.], atol=1.e-9)

        # A pure moment must do work while its resultant force remains zero.
        # Use the same attachment and native wrench operator in both design stages.
        from dataclasses import replace
        from gso_workflows.cad_wrench import SurfaceWrench, grid_wrench
        torque = SurfaceWrench((right,), np.array([30.,3.5,1.]), np.zeros(3), np.array([15.,0.,0.]))
        twisted = replace(study,loads={'torsion':torque},mechanics_executable=executable)
        response = solve(points,np.asarray(cells),twisted,work/'twist',spsolve,surface_tol=.01)
        np.testing.assert_allclose(response['force_N'],np.zeros(3),atol=1.e-10)
        np.testing.assert_allclose(response['applied_moment_local_Nmm'],torque.moment_nmm,atol=1.e-10)
        np.testing.assert_allclose(response['moment_residual_Nmm'],np.zeros(3),atol=1.e-8)
        assert response['torsional_stiffness_Nmm_rad']>0. and 'mean_lift_mm' not in response
        lookup=np.arange(int(np.prod(size+1)))
        grid_force=grid_wrench(torque,np.zeros(3),size,1.,lookup,executable).reshape(-1,3)
        np.testing.assert_allclose(grid_force.sum(0),np.zeros(3),atol=1.e-10)
        np.testing.assert_allclose(np.cross(points-torque.origin_mm,grid_force).sum(0),torque.moment_nmm,atol=1.e-9)

        # Arbitrary selected inner loops on a different CAD part exercise nonplanar loading.
        from gso_workflows.cad_patch import bore_wall_patches
        top = Face.pick(surface,[1.,1.,2.],[0.,0.,1.])
        bores = bore_wall_patches(surface,top)
        assert len(bores)==len(hole_starts)
        bore_torque=SurfaceWrench(bores,np.array([15.,3.,1.]),np.zeros(3),np.array([0.,0.,18.]))
        case=replace(study,loads={'hole group':bore_torque},mechanics_executable=executable)
        r=solve(points,np.asarray(cells),case,work/'hole-twist',spsolve,surface_tol=.01)
        np.testing.assert_allclose(r['applied_moment_local_Nmm'],bore_torque.moment_nmm,atol=1.e-9)
        np.testing.assert_allclose(r['moment_residual_Nmm'],np.zeros(3),atol=1.e-8)
        assert all(x['max_mm']<1.e-10 for x in r['load_transfer'])


if __name__ == '__main__':
    test_plate(Path(sys.argv[1]).resolve())
