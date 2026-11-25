"""Build one boundary from the material field and the signed distances of the declared CAD regions."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates,spline_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from meshlib import mrmeshpy as mr,mrmeshnumpy as arrays
import trimesh


def distance_samples(solid,first,shape,spacing):
    native=solid.to_mesh64()
    mesh=arrays.meshFromFacesVerts(np.asarray(native.tri_verts,dtype=np.int32),
                                    np.asarray(native.vert_properties,dtype=float)[:,:3])
    settings=mr.MeshToDistanceVolumeParams()
    settings.dist.signMode=mr.SignDetectionMode.HoleWindingRule
    settings.vol.dimensions=mr.Vector3i(*map(int,shape))
    settings.vol.voxelSize=mr.Vector3f(*([float(spacing)]*3))
    # MeshLib evaluates cell centers, whereas gridToMesh interpolates lattice vertices.
    settings.vol.origin=mr.Vector3f(*map(float,first-.5*spacing))
    return arrays.getNumpy3Darray(mr.meshToDistanceVolume(mesh,settings)).copy()


def region_distances(solids,first,shape,h,band):
    distance=np.full(tuple(shape),band,np.float32)
    for solid in solids:
        if solid.is_empty() or solid.volume()<=0.:continue
        low,high=np.asarray(solid.bounding_box()).reshape(2,3)
        a=np.maximum(np.floor((low-band-first)/h).astype(int),0)
        b=np.minimum(np.ceil((high+band-first)/h).astype(int)+1,shape)
        if np.any(a>=b):continue
        sl=tuple(slice(i,j) for i,j in zip(a,b))
        patch=distance_samples(solid,first+a*h,b-a,h)
        np.minimum(distance[sl],patch,out=distance[sl])
    return distance


def free_material_field(field,protected):
    """Continue the designable density through fixed CAD before interpolating it.

    Locked samples describe the presence of exact CAD, not a proposal to add material
    to its outside. The discrete harmonic extension uses adjoining design variables
    only, with zero normal flux where the study grid ends.
    """
    if not np.any(protected):
        return field
    if np.all(protected):
        return np.zeros_like(field)
    points=np.argwhere(protected)
    indices=np.full(field.shape,-1,dtype=np.int32)
    indices[protected]=np.arange(len(points))
    rows=[];cols=[];coeff=[]
    diagonal=np.zeros(len(points))
    rhs=np.zeros(len(points))
    for axis in range(3):
        for step in (-1,1):
            neighbor=points.copy()
            neighbor[:,axis]+=step
            inside=(neighbor[:,axis]>=0)&(neighbor[:,axis]<field.shape[axis])
            row=np.flatnonzero(inside)
            pos=tuple(neighbor[inside].T)
            col=indices[pos]
            diagonal[row]+=1.
            internal=col>=0
            rows.extend(row[internal]);cols.extend(col[internal]);coeff.extend(-np.ones(internal.sum()))
            rhs[row[~internal]]+=field[pos][~internal]
    row=np.arange(len(points))
    rows.extend(row);cols.extend(row);coeff.extend(diagonal)
    matrix=sparse.coo_matrix((coeff,(rows,cols)),shape=(len(points),len(points))).tocsr()
    result=field.copy()
    result[protected]=spsolve(matrix,rhs)
    return result


@dataclass
class MaterialBoundary:
    density: np.ndarray
    restriction: np.ndarray
    preserve: np.ndarray
    first: np.ndarray
    spacing: float
    pitch: float

    @classmethod
    def sample(cls,design,space,spacing):
        box=np.asarray(space.envelope.bounding_box()).reshape(2,3)
        # An offset sampling lattice avoids exact zero values at coplanar CAD/grid coincidences.
        first=spacing*(np.floor(box[0]/spacing)-2.371)
        shape=np.ceil((box[1]-first)/spacing).astype(int)+4
        axes=[first[i]+np.arange(shape[i])*spacing for i in range(3)]
        yz=np.meshgrid(*axes[1:],indexing='ij')
        yz=[(axis-design.origin[i+1])/design.pitch-.5 for i,axis in enumerate(yz)]
        free=free_material_field(design.field,design.protected)
        coeff=spline_filter(free,order=3,mode='nearest')
        density=np.empty(tuple(shape),np.float32)
        for i,x in enumerate(axes[0]):
            indices=np.array([np.full(yz[0].shape,(x-design.origin[0])/design.pitch-.5),*yz])
            density[i]=map_coordinates(coeff,indices,order=3,mode='nearest',prefilter=False)
        band=2.*design.pitch
        restriction=distance_samples(space.envelope,first,shape,spacing)
        excluded=region_distances(space.exclusions,first,shape,spacing,band)
        np.maximum(restriction,-excluded,out=restriction)
        preserved=region_distances(space.preserved,first,shape,spacing,band)
        return cls(density,restriction,preserved,first,spacing,design.pitch)

    def surface(self,level,adaptivity):
        # Negative is material. Each lattice vertex has one material decision. There are
        # no independently tessellated pieces to weld, intersect or repair afterwards.
        signed=(level-self.density)*self.pitch
        np.minimum(signed,self.preserve,out=signed)
        np.maximum(signed,self.restriction,out=signed)
        return surface_from_signed(signed,self.first,self.spacing,self.pitch,adaptivity)

    def at_budget(self,volume_limit,adaptivity=0.):
        # Bracket using occupancy before integrating the volume of actual candidate boundaries.
        lo,hi=.005,1.02
        for _ in range(14):
            mid=.5*(lo+hi)
            count=np.count_nonzero(((self.density>mid)|(self.preserve<0.))&(self.restriction<0.))
            if count*self.spacing**3>volume_limit:lo=mid
            else:hi=mid
        level=.5*(lo+hi);lo,hi=.005,1.02
        best=None
        for _ in range(14):
            mesh=self.surface(level,adaptivity)
            volume=mesh.volume
            if volume>volume_limit:lo=level
            else:
                hi=level;best=mesh
                if volume_limit-volume<.0003*volume_limit:break
            level=.5*(lo+hi)
        if best is None:raise ValueError('Cannot meet requested volume with prescribed material')
        return best


def surface_from_signed(signed,first,spacing,band,adaptivity=0.):
    grid=mr.simpleVolumeToDenseGrid(arrays.simpleVolumeFrom3Darray(signed),band)
    settings=mr.GridToMeshSettings()
    settings.voxelSize=mr.Vector3f(*([spacing]*3))
    settings.adaptivity=adaptivity
    settings.relaxDisorientedTriangles=False
    boundary=mr.gridToMesh(grid,settings)
    vertices=arrays.getNumpyVerts(boundary).astype(float)+first
    faces=arrays.getNumpyFaces(boundary.topology).copy()
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    if mesh.volume<0.:mesh.invert()
    return mesh

