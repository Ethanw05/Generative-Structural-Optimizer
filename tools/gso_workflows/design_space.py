"""CAD selections and a common design space for analysis and surface reconstruction (millimetres)."""
from dataclasses import dataclass
import numpy as np
import manifold3d as md
from matplotlib.path import Path as Polygon
from scipy.spatial import cKDTree
import trimesh


def closed_solid(mesh):
    solid=md.Manifold(md.Mesh64(np.ascontiguousarray(mesh.vertices,dtype=float),
                                np.ascontiguousarray(mesh.faces,dtype=np.uint64)))
    if solid.is_empty() or solid.status()!=md.Error.NoError or solid.volume()<=0:
        raise ValueError('Selection does not define an oriented solid')
    return solid


@dataclass
class Face:
    triangles: np.ndarray
    normal: np.ndarray
    loops: list
    basis: np.ndarray
    origin: np.ndarray
    tolerance: float = 1.e-4

    @classmethod
    def pick(cls,mesh,point,normal,tolerance=1e-4):
        point=np.asarray(point,dtype=float);normal=np.asarray(normal,dtype=float)
        normal=normal/np.linalg.norm(normal)
        # A pick resolves one connected planar CAD patch, including its inner loops.
        accepted=[]
        for group in mesh.facets:
            if np.dot(mesh.face_normals[group[0]],normal)<1.-1e-6:
                continue
            tri=mesh.triangles[group]
            if np.max(np.abs((tri-point)@normal))>tolerance:
                continue
            xyz=trimesh.triangles.closest_point(tri,np.broadcast_to(point,(len(tri),3)))
            if np.linalg.norm(xyz-point,axis=1).min()<=tolerance:
                accepted.append(group)
        if len(accepted)!=1:raise ValueError(f'CAD face pick resolved {len(accepted)} patches at {point}')
        group=accepted[0];tri=mesh.faces[group]
        # A pick locates a native patch. Its rounded coordinates and requested direction
        # must never define the physical plane used by subsequent solid operations.
        vertices=mesh.vertices[tri]
        native_normal=np.sum(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=0)
        native_normal/=np.linalg.norm(native_normal)
        normal=native_normal if native_normal@normal>0. else -native_normal
        point=point+normal*((vertices.mean(axis=(0,1))-point)@normal)
        edges=tri[:,[0,1,1,2,2,0]].reshape(-1,2)
        _,first,count=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True)
        boundary=edges[first[count==1]]
        successors=dict(boundary)
        if len(successors)!=len(boundary):raise ValueError('Branched boundary in selected face')
        u=np.cross(np.eye(3)[np.argmin(np.abs(normal))],normal);u/=np.linalg.norm(u)
        basis=np.array([u,np.cross(normal,u),normal])
        loops=[]
        while successors:
            start=next(iter(successors));chain=[start];nxt=successors.pop(start)
            while nxt!=start:
                chain.append(nxt);nxt=successors.pop(nxt)
            loops.append(((mesh.vertices[chain]-point)@basis.T)[:,:2])
        return cls(mesh.triangles[group],normal,loops,basis,point,tolerance)

    def prism(self,depth,outer_only=False,side_inset=0.):
        loops=self.loops
        if outer_only:
            # A connected planar face has one positively oriented exterior and zero or more holes.
            loops=[q for q in loops if np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))>0.]
            if len(loops)!=1:raise ValueError('Selected face has an ambiguous outer boundary')
        section=md.CrossSection(loops,md.FillRule.EvenOdd)
        if side_inset:section=section.offset(-side_inset)
        local=section.extrude(abs(depth)).translate([0.,0.,min(depth,0.)])
        return local.transform(np.column_stack((self.basis.T,self.origin)))

    def retained_and_internal_void(self,back,solid):
        depth=(back.origin-self.origin)@self.normal
        tolerance=max(self.tolerance,back.tolerance)
        if abs(depth)<=2.*tolerance:
            raise ValueError('Unresolved depth between selected native faces')
        region=self.prism(depth,outer_only=True)
        retained=region^solid
        # Only a void within the feature is forbidden material. Its exterior faces are
        # attachable. Evaluate the Boolean in the open interior at CAD selection tolerance
        # so coplanar tessellation residue cannot create an artificial sheet obstacle.
        inset=np.sign(depth)*tolerance
        interior=self.prism(depth-2.*inset,outer_only=True,side_inset=tolerance)
        interior=interior.translate(inset*self.normal)
        return retained,interior-solid

    def open_hole_clearances(self,depth,reach,original):
        """Keep existing axial access to the through bores of a retained CAD interface."""
        solids=[]
        for loop in self.loops:
            area=np.sum(loop[:,0]*np.roll(loop[:,1],-1)-loop[:,1]*np.roll(loop[:,0],-1))
            if area>=0.:continue
            section=md.CrossSection([loop],md.FillRule.EvenOdd)
            transform=np.column_stack((self.basis.T,self.origin))
            inside=section.extrude(abs(depth)).translate([0.,0.,min(depth,0.)]).transform(transform)
            if (inside^original).volume()>max(1.e-6,1.e-6*inside.volume()):continue
            end=depth+np.sign(depth)*reach
            prism=section.extrude(abs(end-depth)).translate([0.,0.,min(depth,end)])
            column=prism.transform(np.column_stack((self.basis.T,self.origin)))
            solids.append(column-original)
        return solids

    def contains_projection(self,points,tolerance):
        xyz=(points-self.origin)@self.basis.T
        on_plane=np.abs(xyz[:,2])<=tolerance
        query=xyz[on_plane,:2]
        inside=np.zeros(len(query),bool);edge=np.zeros(len(query),bool)
        for loop in self.loops:
            inside ^= Polygon(loop).contains_points(query)
            eps=max(1.,np.ptp(loop,axis=0).max())*1e-9
            for start,end in zip(loop,np.roll(loop,-1,axis=0)):
                delta=end-start;length2=delta@delta
                if length2==0.:continue
                local=query-start
                t=np.clip((local@delta)/length2,0.,1.)
                edge |= np.sum((local-t[:,None]*delta)**2,axis=1)<=eps**2
        selected=np.zeros(len(points),bool)
        selected[on_plane]=inside|edge
        return selected


@dataclass
class DesignSpace:
    envelope: object
    preserved: tuple
    exclusions: tuple

    def free_envelope(self):
        result=self.envelope
        for region in self.exclusions:result-=region
        return result

    def sample(self,origin,shape,pitch):
        from .reconstruction import distance_samples,region_distances
        first=np.asarray(origin)+.5*pitch
        # Use exactly the same CAD membership functions as the realized solid.
        tolerance=1e-6*pitch
        envelope=distance_samples(self.envelope,first,shape,pitch)
        void=region_distances(self.exclusions,first,shape,pitch,2.*pitch)
        retained=region_distances(self.preserved,first,shape,pitch,2.*pitch)
        allowed=(envelope<=tolerance)&(void>tolerance)
        return allowed,allowed&(retained<=tolerance)


def surface_interpolation(face,origin,shape,pitch,node_map):
    nodes=np.asarray(shape)+1
    support_index=None
    def transfer_to_material(point):
        nonlocal support_index
        if support_index is None:
            used=np.where(node_map>=0)[0]
            locations=np.stack(np.unravel_index(used,nodes),axis=1)
            selected=face.contains_projection(origin+pitch*locations,.51*pitch)
            locations,used=locations[selected],used[selected]
            support_index=(cKDTree(locations),locations,node_map[used])
        tree,locations,mapped=support_index
        nearby=tree.query_ball_point(point,2.5)
        if not nearby:raise ValueError('Load is outside the local material cloud')
        delta=locations[nearby]-point
        basis=np.column_stack((np.ones(len(nearby)),delta)).T
        distance=np.linalg.norm(delta,axis=1)/2.5
        weight=np.maximum(1.-distance,0.)**6*(1.+6.*distance+35.*distance**2/3.)
        wanted=np.array([1.,0.,0.,0.])
        coefficient=np.linalg.lstsq((basis*weight)@basis.T,wanted,rcond=1e-12)[0]
        values=weight*(basis.T@coefficient)
        if not np.allclose(basis@values,wanted,atol=1e-8):
            raise ValueError('Attachment cannot transmit the resultant and moment on this cloud')
        return mapped[nearby],values
    return transfer_to_material


def distribute_surface_force(face,resultant,origin,shape,pitch,node_map):
    """Integrate a uniform resultant using triangle quadrature and trilinear grid interpolation."""
    shape=np.asarray(shape);nodes=shape+1
    free_count=int(node_map.max())+1
    force=np.zeros((free_count,3))
    areas=trimesh.triangles.area(face.triangles);total=areas.sum()
    if not np.isfinite(total) or total<=0:raise ValueError('Invalid selected area')
    resultant=np.asarray(resultant,dtype=float)
    corner=np.indices((2,2,2)).reshape(3,8).T
    transfer_to_material=surface_interpolation(face,origin,shape,pitch,node_map)
    def clip(polygon,axis,limit,sign):
        if len(polygon)==0:return polygon
        output=[];last=polygon[-1];previous=sign*(last[axis]-limit)
        for point in polygon:
            distance=sign*(point[axis]-limit)
            if (distance>=0)!=(previous>=0):
                output.append(last+(point-last)*(previous/(previous-distance)))
            if distance>=0:output.append(point)
            last,previous=point,distance
        return output
    quadrature=[];weights_q=[]
    for a,w in ((.445948490915965,.223381589678011),(.091576213509771,.109951743655322)):
        for j in range(3):
            bary=np.full(3,a);bary[j]=1.-2*a;quadrature.append(bary);weights_q.append(w)
    quadrature=np.array(quadrature);weights_q=np.array(weights_q)
    for triangle in face.triangles:
        lattice=(triangle-origin)/pitch
        rounded=np.rint(lattice)
        lattice=np.where(np.abs(lattice-rounded)<1e-8,rounded,lattice)
        lower=np.clip(np.floor(lattice.min(0)).astype(int),0,shape-1)
        upper=np.clip(np.ceil(lattice.max(0)).astype(int)-1,lower,shape-1)
        for offset in np.ndindex(tuple(upper-lower+1)):
            cell=lower+offset
            polygon=lattice
            for axis in range(3):
                polygon=clip(polygon,axis,cell[axis],1)
                polygon=clip(polygon,axis,cell[axis]+1,-1)
            for j in range(1,len(polygon)-1):
                sub=np.array([polygon[0],polygon[j],polygon[j+1]])
                area=.5*pitch**2*np.linalg.norm(np.cross(sub[1]-sub[0],sub[2]-sub[0]))
                if area<=1e-16:continue
                frac=np.clip(quadrature@sub-cell,0.,1.)
                values=np.prod(np.where(corner[None,:,:],frac[:,None,:],1.-frac[:,None,:]),axis=2)
                weights=weights_q@values
                mapped=node_map[np.ravel_multi_index((cell+corner).T,nodes)]
                active=weights>1e-12
                if np.any(mapped[active]<0):
                    # The affine weights vary non-polynomially at cut cells. Subdivide
                    # once so the load is insensitive to the CAD face diagonal.
                    middle=(sub+np.roll(sub,-1,axis=0))*.5
                    children=[np.array([sub[k],middle[k],middle[k-1]]) for k in range(3)]
                    for child in [*children,middle]:
                        for point,qweight in zip(quadrature@child,weights_q):
                            ids,local_weights=transfer_to_material(point)
                            np.add.at(force,ids,local_weights[:,None]*(qweight*area/(4.*total))*resultant)
                else:
                    np.add.at(force,mapped[active],weights[active,None]*(area/total)*resultant)
    if not np.allclose(force.sum(0),resultant,rtol=1.e-9,atol=1.e-10):raise ValueError('Surface force balance failed')
    return force.ravel()
