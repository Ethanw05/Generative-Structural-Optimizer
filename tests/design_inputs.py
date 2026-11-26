"""Exercise the public study path with a rectangular coupon and independently rotated loading."""
from pathlib import Path
import sys
import tempfile
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'build/sunbeam33/meshlib-runtime'),str(root/'build/sunbeam33/deps'),str(root/'tools')]
import numpy as np
import manifold3d as md
import trimesh
from gso_workflows.design_space import Face,DesignSpace,distribute_surface_force
from gso_workflows.volume_study import VolumeStudy,SurfaceLoad,optimize,export_design

from gso_workflows.mesh_quality import DisconnectedMaterial,validate_single_part
from gso_workflows.reconstruction import free_material_field
from gso_workflows.triangle_contact import improper_contact
from gso_workflows.boundary_shape import admissible_growth
grid=np.indices((9,9,9),dtype=float)
baseline=grid[2]-1.
normal=np.zeros((9,9,3));normal[:,:,2]=1.
travel=np.full((9,9),5.)
wall_parallel=np.abs(grid[0]-2.)-.6
wall_crossed=np.abs(grid[2]-3.)-.6
assert admissible_growth(4,baseline,normal,wall_parallel,travel.copy(),1.)[4,5]==5.
assert admissible_growth(4,baseline,normal,wall_crossed,travel.copy(),1.)[4,5]==0.

triangles=np.array([[0,0,0],[2,0,0],[0,2,0],[1,0,-1],[1,1,1],[1,2,-1],
                    [2,2,0],[.3,.3,0],[.7,.3,0],[.3,.7,0]],float)
assert improper_contact(triangles,[0,1,2],[3,4,5])
assert improper_contact(triangles,[0,1,2],[7,8,9])
assert not improper_contact(triangles,[0,1,2],[2,1,6])
assert not improper_contact(triangles,[0,1,2],[1,6,4])


# A locked CAD island in empty design space has no free density to extrude past its rim.
locked=np.zeros((11,9,7),bool)
locked[4:7,2:7,2:5]=True
proposal=locked.astype(float)
np.testing.assert_allclose(free_material_field(proposal,locked),0.)
linear=np.broadcast_to(np.arange(11.)[:,None,None],locked.shape).copy()
proposal=linear.copy();proposal[locked]=1.
np.testing.assert_allclose(free_material_field(proposal,locked),linear,atol=1.e-12)
first=trimesh.creation.box()
second=first.copy();second.apply_translation([1.1,0.,0.])
try:
    validate_single_part(trimesh.util.concatenate((first,second)))
except DisconnectedMaterial:
    pass
else:
    raise AssertionError('Two watertight plates with a gap were accepted as one part')

# At the exposed face of a retained block material can grow outward, while the
# reverse move would erode protected CAD. An intersecting keepout reverses that bound.
from gso_workflows.cad_motion import cad_bounds
patch = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
faces = np.array([[0, 1, 2]])
large = md.Manifold.cube([10., 10., 10.], center=True)
lower = md.Manifold.cube([4., 4., 2.], center=True)
upper = md.Manifold.cube([4., 4., 2.], center=True).translate([0., 0., 2.])
np.testing.assert_allclose(cad_bounds(DesignSpace(large,(lower,),()),patch,faces,1.,.02),
                           [[0.,1.]]*3)
np.testing.assert_allclose(cad_bounds(DesignSpace(large,(),(upper,)),patch,faces,1.,.02),
                           [[-1.,0.]]*3)
np.testing.assert_allclose(cad_bounds(DesignSpace(large,(),()),patch,faces,1.,.02),
                           [[-1.,1.]]*3)

# A top face and a vertical side meet at the edge of the envelope. The side
# may advance along the ceiling. A second vertical wall removes that option.
crease=np.vstack((patch, [0.,0.,0.]))
crease_faces=np.array([[0,1,2],[0,3,1]])
ceiling=md.Manifold.cube([10.,10.,10.],center=True).translate([0.,0.,-4.])
bounds=cad_bounds(DesignSpace(ceiling,(),()),crease,crease_faces,1.,.02)
np.testing.assert_allclose(bounds[:,1],[1.,1.,0.,1.])
wall=md.Manifold.cube([10.,10.,10.],center=True).translate([0.,-5.,0.])
bounds=cad_bounds(DesignSpace(ceiling,(),(wall,)),crease,crease_faces,1.,.02)
np.testing.assert_allclose(bounds[:2,1],0.)

box=md.Manifold.cube([6.,4.,4.])
void=md.Manifold.cube([1.,.5,.5]).translate([5.,1.75,1.75])
surface=(box-void).to_mesh64()
mesh=trimesh.Trimesh(surface.vert_properties[:,:3],surface.tri_verts,process=False)
mounted=Face.pick(mesh,[0,2,2],[-1,0,0])
end=Face.pick(mesh,[6,1,1],[1,0,0])
np.testing.assert_allclose(mounted.perimeter_distance([0.,2.,2.]),2.)
np.testing.assert_allclose(end.perimeter_distance([6.3,2.,2.]),np.hypot(.25,.3))
np.testing.assert_allclose(end.perimeter_distance([6.,0.,2.]),0.,atol=1.e-12)
approx_end=Face.pick(mesh,[6.+3.e-8,1.,1.],[1.,1.e-8,0.])
np.testing.assert_allclose(approx_end.origin[0],6.,atol=1.e-13,rtol=0.)
np.testing.assert_allclose(approx_end.normal,[1.,0.,0.],atol=1.e-13)
rear=Face.pick(mesh,[-3.e-8,1.,1.],[-1.,0.,0.])
retained,internal=approx_end.retained_and_internal_void(rear,box-void)
assert abs(retained.volume()-(box-void).volume())<1.e-7
from gso_workflows.reconstruction import distance_samples
shape1=np.array([1,1,1])
assert distance_samples(internal,np.array([6.01,.2,.2]),shape1,.1).item()>1.
assert distance_samples(internal,np.array([5.5,2.,2.]),shape1,.1).item()<0.
# A hollow protected feature exercises both preservation and a selected geometric exclusion.
frame=end.prism(-1.)
space=DesignSpace(box,(frame-void,),(void,))
shape=np.array([6,4,4]);origin=np.zeros(3);unit_pitch=1.
idx=np.arange(np.prod(shape+1))
xyz=np.stack(np.unravel_index(idx,shape+1),axis=1).astype(float)
force=np.array([1.,3.,-2.])
f=distribute_surface_force(end,force,origin,shape,unit_pitch,idx).reshape(-1,3)
np.testing.assert_allclose(f.sum(0),force,atol=1e-12)
np.testing.assert_allclose(np.cross(xyz,f).sum(0),np.cross([6,2,2],force),atol=1e-12)
# A body with a differently named load sees the same physics and identical density updates.
with tempfile.TemporaryDirectory(dir=root/'build') as temp:
    def run(name,direction):
        study=VolumeStudy(space,(mounted,),{name:SurfaceLoad(end,np.array(direction,dtype=float))},
                          2000.,.3,70.,np.eye(3),np.zeros(3))
        return study,optimize(study,.5,12,Path(sys.argv[1]).resolve(),Path(temp),refinement_steps=6)
    study,a=run('load A',[0,0,1])
    _,b=run('different label',[0,0,1])
    np.testing.assert_allclose(a.field,b.field,atol=1e-12)
    _,c=run('load A',[0,1,0])
    assert np.max(np.abs(a.field-c.field))>.02
    np.testing.assert_allclose(a.field,c.field.swapaxes(1,2),atol=1.e-7)
    np.testing.assert_allclose(a.field[a.protected],1.)
    out=Path(temp)/'coupon.obj'
    results=export_design(study,a,out)
    assert out.is_file() and results['volume_mm3']<=70.0001
    reconstructed=trimesh.load_mesh(out)
    assert reconstructed.is_volume and reconstructed.body_count==1
    print('Independent coupon export, volume',results['volume_mm3'])
