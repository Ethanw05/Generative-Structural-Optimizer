"""Shared linear elastic CAD operators and work-equivalent interface loads."""
from typing import NamedTuple
import numpy as np
from scipy import sparse
from .surface_transfer import project_face, project_fixture
from .analysis_boundary import exterior,require_closed_fans
from .cad_wrench import SurfaceWrench, boundary_wrench


class ElasticSystem(NamedTuple):
    points: object
    cells: object
    volume: object
    gradients: object
    shear: float
    lame: float
    stiffness: object
    skin: object
    owner: object
    local: object
    fixed: object
    fixture_area: float
    fixture_transfer: list


def assemble(points,cells,study,surface_tol):
    p,t=points,cells
    used,t=np.unique(t,return_inverse=True)
    p=p[used];t=t.reshape(-1,4).astype(np.int32)
    n=len(p);tet_points=p[t];jac=tet_points[:,1:]-tet_points[:,:1]
    volume=abs(np.linalg.det(jac))/6
    grad=np.linalg.inv(jac).transpose(0,2,1)
    grad=np.concatenate((-grad.sum(1)[:,None,:],grad),axis=1)
    shear=study.young_mpa/(2.*(1.+study.poisson))
    lame=2.*shear*study.poisson/(1.-2.*study.poisson)
    row=np.repeat(t,4,axis=1).ravel();col=np.tile(t,(1,4)).ravel()
    products=grad@grad.transpose(0,2,1)
    k=[[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(i,3):
            outer=grad[:,:,i,None]*grad[:,None,:,j]
            elastic=lame*outer+shear*outer.transpose(0,2,1)
            if i==j:elastic+=shear*products
            k[i][j]=sparse.coo_matrix(((volume[:,None,None]*elastic).ravel(),(row,col)),shape=(n,n)).tocsr()
            if i!=j:k[j][i]=k[i][j].T
    matrix=sparse.bmat(k,format='csr')
    del k,products,row,col,jac,tet_points
    skin,owner=exterior(t)
    local=(p-study.world_origin)@study.world_rotation.T
    # Exterior winding follows the material, independent of the tetrahedralizer's convention.
    tri=local[skin]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    inward=np.einsum('ij,ij->i',normal,tri.mean(1)-local[t[owner]].mean(1))<0.
    skin[inward]=skin[inward,::-1]
    require_closed_fans(skin)
    fixed=np.zeros(n,bool);fixture_area=0.;fixture_transfer=[]
    for face in study.fixed_faces:
        nodes,area,dist=project_fixture(local,skin,face,surface_tol)
        fixed[nodes]=True
        fixture_area+=area
        fixture_transfer.append(dist)
    return ElasticSystem(p,t,volume,grad,shear,lame,matrix,skin,owner,local,fixed,fixture_area,fixture_transfer)


def applied(study,load,local,skin,surface_tol):
    direction=load.force_n@study.world_rotation
    if isinstance(load,SurfaceWrench):
        if study.mechanics_executable is None:
            raise ValueError('CAD wrench transfer requires the mechanics library beside the study executable')
        local_force,load_area,load_transfer=boundary_wrench(load,local,skin,surface_tol,study.mechanics_executable)
        force=local_force@study.world_rotation
    else:
        ids,weights,load_transfer=project_face(local,skin,load.face,surface_tol)
        load_area=weights.sum()
        force=np.zeros_like(local)
        np.add.at(force,ids.ravel(),weights.ravel()[:,None]*direction/load_area)
    return force,load_area,load_transfer
