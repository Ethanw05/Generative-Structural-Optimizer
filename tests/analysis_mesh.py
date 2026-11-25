"""Reject artificial connections and retain enclosed material boundaries in analysis meshes."""
from pathlib import Path
from itertools import permutations,product
from unittest.mock import patch
import sys,tempfile
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'build/sunbeam33/meshlib-runtime'),str(root/'build/sunbeam33/deps'),str(root/'tools')]
import numpy as np
from gso_workflows.material_shells import material_shells,verify_constraints
from gso_workflows.analysis_mesh import volume_mesh,digest
# A conforming volume can have an invalid material surface when cells touch only
# along an edge or vertex. Both must be rejected before stress recovery.
from gso_workflows.analysis_boundary import analysis_skin
from gso_workflows.mesh_quality import InadmissibleBoundary
sites=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,-1,0],[0,0,-1],[-1,0,0]],dtype=float)
for pair in ([[0,1,2,3],[0,1,4,5]],[[0,1,2,3],[0,4,5,6]]):
    try:analysis_skin(sites,np.array(pair))
    except InadmissibleBoundary:pass
    else:raise AssertionError('Touching tetrahedra accepted as a manifold analysis boundary')
assert analysis_skin(sites,np.array([[0,1,2,3]])).is_volume

def grid(size,void=None):
    points=np.array(list(product(*(range(n+1) for n in size))),dtype=float)
    cells=[]
    for x in product(*(range(n) for n in size)):
        if x==void:continue
        for sequence in permutations(range(3)):
            offsets=np.vstack((np.zeros(3,dtype=int),np.eye(3,dtype=int)[list(sequence)].cumsum(axis=0)))
            cells.append(np.ravel_multi_index((offsets+x).T,size+1))
    cells=np.asarray(cells)
    reverse=np.linalg.det(points[cells[:,1:]]-points[cells[:,:1]])<0.
    cells[reverse]=cells[reverse][:,[1,0,2,3]]
    return points,cells


with tempfile.TemporaryDirectory(dir=root/'build') as tmp:
    folder=Path(tmp)
    points,cells=grid(np.array([3,1,1]))
    mesh=analysis_skin(points,cells)
    # An old cache can pass a volume check and still have a false edge connection.
    from gso_workflows.surface_search import volume_mesh,digest
    from subprocess import CompletedProcess
    key=digest(mesh)
    cache=folder/f'surface-{key[:24]}-0.5-0.04.npz'
    np.savez(cache,source=key,p=sites*(9.**(1./3.)),tet=np.array([[0,1,2,3],[0,1,4,5]]))
    native_cells=np.asarray(cells).copy()
    negative=np.linalg.det(points[native_cells[:,1:]]-points[native_cells[:,:1]])<0.
    native_cells[negative]=native_cells[negative][:,[1,0,2,3]]
    def conforming_worker(command,**options):
        assert float(command[-2])==.02
        np.savez(command[-4],p=points,tet=native_cells)
        return CompletedProcess(command,0,'','')
    with patch('gso_workflows.analysis_mesh.subprocess.run',side_effect=conforming_worker):
        regenerated=volume_mesh(mesh,folder,.5,.04)
    np.testing.assert_array_equal(regenerated[1],native_cells)
    with np.load(folder/('conforming-'+cache.name)) as data:
        assert int(data['verified_boundary'])==2 and float(data['effective_tolerance'])==.02

    # The old exterior-only mesh fills one cubic millimetre of empty space.
    # Reuse its exterior trace but constrain the actual inner wall in TetGen.
    points,full=grid(np.array([3,3,3]))
    _,hollow=grid(np.array([3,3,3]),(1,1,1))
    original=analysis_skin(points,hollow)
    key=digest(original)
    np.savez(folder/f'surface-{key[:24]}-0.8-0.04.npz',source=key,p=points,tet=full)
    p,t=volume_mesh(original,folder,.8,.04)
    recovered=analysis_skin(p,t)
    verify_constraints(material_shells(original),recovered)
    np.testing.assert_allclose(recovered.volume,26.,atol=1.e-9)
    assert len(material_shells(recovered)[1])==1
