"""A fine patch beside a long supporting triangle must not steal a CAD load."""
from pathlib import Path
from types import SimpleNamespace
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root/'tools'), str(root/'build/sunbeam33/deps')]
import numpy as np
from scipy.spatial.transform import Rotation
from gso_workflows.surface_transfer import project_face, native_samples

triangles = [[[0.,0.,0.],[10.,0.,0.],[0.,10.,0.]]]
for x in np.linspace(-.2, -.025, 8):
    for y in np.arange(10)*.04:
        a, b = [x,y,0.], [x+.025,y,0.]
        c, d = [x+.025,y+.04,0.], [x,y+.04,0.]
        triangles.extend(([a,b,c], [a,c,d]))
triangles = np.array(triangles)
patch = np.array([[[.006,.198,0.],[.012,.198,0.],[.006,.207,0.]]])
rng = np.random.default_rng(2511)
for rotation in [np.eye(3), *Rotation.random(3,random_state=rng).as_matrix()]:
    origin = rng.uniform(-50.,50.,3)
    points = (triangles @ rotation.T + origin).reshape(-1,3)
    faces = np.arange(len(points)).reshape(-1,3)
    selection = SimpleNamespace(triangles=patch@rotation.T+origin, origin=origin, normal=rotation[:,2])
    expected, areas = native_samples(selection, 1.)
    nodes, weights, report = project_face(points, faces, selection, .001)
    np.testing.assert_array_equal(nodes, [[0,1,2]])
    np.testing.assert_allclose(weights.sum(), areas.sum(), rtol=1.e-12)
    # Virtual work against any rigid rotation depends on the first moment.
    np.testing.assert_allclose(np.einsum('ij,ijk->k',weights,points[nodes]), areas @ expected, atol=1.e-13)
    assert report['max_mm'] < 1.e-11
print('Graded CAD patch transferred area and first moment in four orientations.')

# Two sheet faces overlap in projection and lie inside the transfer tolerance.
# Only the top face of the selected sheet transmits the prescribed fixture.
from itertools import permutations, product
from gso_workflows.design_space import Face
from gso_workflows.tetra_elasticity import assemble
import trimesh

base = np.array(list(product(range(2),repeat=3)), dtype=float)
base[:,2] *= .04
two_sheets = np.vstack((base, base+[0.,0.,.12]))
cells = []
for order in permutations(range(3)):
    corners = np.vstack((np.zeros(3,dtype=int),np.eye(3,dtype=int)[list(order)].cumsum(axis=0)))
    cells.append(np.ravel_multi_index(corners.T,(2,2,2)))
cells = np.array(cells)
upper = trimesh.creation.box(extents=(1.,1.,.04))
upper.apply_translation([.5,.5,.14])
fixture = Face.pick(upper,[.5,.5,.16],[0.,0.,1.])
study = SimpleNamespace(world_origin=np.zeros(3),world_rotation=np.eye(3),
                        fixed_faces=(fixture,),young_mpa=1800.,poisson=.32)
system = assemble(two_sheets,np.vstack((cells,cells+8)),study,.15)
np.testing.assert_array_equal(system.fixed, np.isclose(two_sheets[:,2], .16))
print('Only the selected sheet carries fixture reactions.')
