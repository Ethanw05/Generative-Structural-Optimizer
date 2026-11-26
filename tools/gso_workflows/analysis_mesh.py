"""Conforming analysis tetrahedra for a generated solid, including its enclosed CAD boundaries."""
from pathlib import Path
import hashlib
import os
import sys
import subprocess
import tempfile
import numpy as np
from .analysis_boundary import analysis_skin
from .material_shells import material_shells,constrained_boundary,verify_constraints
from .mesh_quality import InadmissibleBoundary


def digest(mesh):
    key=hashlib.sha256()
    for array,dtype in ((mesh.vertices,'<f8'),(mesh.faces,'<i4')):
        key.update(np.ascontiguousarray(array,dtype=dtype).tobytes())
    return key.hexdigest()


def enclosed_void_seeds(mesh):
    if mesh.body_count==1:return np.empty((0,3))
    seeds=[]
    for shell in mesh.split(only_watertight=False):
        if shell.volume>=0.:continue
        tri=shell.triangles
        i=np.argmax(shell.area_faces)
        length=np.sqrt(shell.area_faces[i])
        def enclosed(point):
            a,b,c=(tri-point).transpose(1,0,2)
            la,lb,lc=np.linalg.norm(tri-point,axis=2).T
            numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
            denominator=la*lb*lc+lc*np.einsum('ij,ij->i',a,b)+la*np.einsum('ij,ij->i',b,c)+lb*np.einsum('ij,ij->i',c,a)
            return abs(np.arctan2(numerator,denominator).sum())>np.pi
        possibilities=[shell.center_mass,tri[i].mean(0)+1.e-5*length*shell.face_normals[i]]
        for point in possibilities:
            if enclosed(point):
                seeds.append(point);break
        else:raise InadmissibleBoundary('Cannot establish analysis void interior')
    return np.asarray(seeds).reshape(-1,3)


def checked_surface(points,cells,expected_volume):
    det=np.linalg.det(points[cells[:,1:]]-points[cells[:,:1]])
    volume=np.abs(det).sum()/6.
    wrong_volume=expected_volume is not None and abs(volume/expected_volume-1.)>.003
    if not (np.all(det>0.) or np.all(det<0.)) or wrong_volume:
        raise InadmissibleBoundary(f'Invalid volume discretization: {volume} vs {expected_volume} mm3')
    return analysis_skin(points,cells)


def read_mesh(path,key):
    with np.load(path) as data:
        if str(data['source'])!=key:raise ValueError('Volume mesh provenance mismatch')
        return data['p'],data['tet']


def registered_analysis_mesh(path, workspace, fixed_surface_edge_mm=None):
    """Load a verified CAD analysis mesh and optionally refine its interior at fixed exterior geometry."""
    with np.load(path) as data:
        if int(data['verified_boundary']) != 2:
            raise ValueError('Expected a verified conforming analysis boundary')
        identity = str(data['source'])
        tolerance = float(data['effective_tolerance'])
        points, cells = data['p'], data['tet']
    metadata = dict(effective_surface_tolerance_mm=tolerance)
    if fixed_surface_edge_mm is not None:
        if not np.isfinite(fixed_surface_edge_mm) or fixed_surface_edge_mm <= 0.:
            raise ValueError('Choose a finite positive regular-tetrahedron equivalent edge')
        skin = analysis_skin(points, cells)
        maximum_volume = fixed_surface_edge_mm**3 / (6.*np.sqrt(2.))
        metadata['refinement'] = dict(method='refine tetrahedra on registered exterior', source_tetrahedra=len(cells),
                                      source_skin=digest(skin), regular_tet_equivalent_edge_mm=fixed_surface_edge_mm,
                                      maximum_tetrahedron_volume_mm3=maximum_volume)
        workspace.mkdir(parents=True, exist_ok=True)
        points, cells = volume_mesh(skin, workspace, maximum_volume, 0., 'interior')
    return identity, points, cells, metadata


def save_mesh(path,key,p,t,tolerance):
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        cache=Path(temporary)/'verified.npz'
        np.savez_compressed(cache,source=key,p=p,tet=t,verified_boundary=2,effective_tolerance=tolerance)
        cache.replace(path)


def native_mesh(source,work,method,edge,tolerance,threads):
    with tempfile.TemporaryDirectory(dir=work) as temporary:
        folder=Path(temporary)
        input_file=folder/'source.npz'
        out=folder/'result.npz'
        np.savez(input_file,v=source.vertices,f=source.faces,
                 holes=enclosed_void_seeds(source) if method=='interior' else [])
        env=os.environ.copy()
        env['PYTHONPATH']=os.pathsep.join(str(Path(d).resolve()) for d in sys.path if d)
        worker=Path(__file__).with_name('tetra_worker.py')
        counts=(1,) if method=='interior' else (min(8,threads,max(1,len(source.faces)//10000)),1)
        for count in dict.fromkeys(counts):
            out.unlink(missing_ok=True)
            run=subprocess.run([sys.executable,'-B','-X','faulthandler',str(worker),method,str(input_file),
                                str(out),str(edge),str(tolerance),str(count)],env=env,cwd=folder,
                                capture_output=True,text=True,errors='replace')
            if run.returncode==0 and out.exists():
                with np.load(out) as data:return data['p'],data['tet']
        raise InadmissibleBoundary(f'Native {method} meshing failed: {run.stderr[-1500:]}')


def volume_mesh(mesh,root,edge,tolerance,method='surface'):
    if method not in ('surface','interior'):raise ValueError('Unknown volume discretization method')
    key=digest(mesh)
    name=f'{method}-{key[:24]}-{edge:.8g}-{tolerance:.8g}.npz'
    path=root/('conforming-'+name)
    if path.is_file():return read_mesh(path,key)
    shells=material_shells(mesh)
    threads=int(os.environ.get('GSO_WORKERS','8'))
    if threads<1:raise ValueError('GSO_WORKERS must be positive')

    def certify(p,t):
        boundary=checked_surface(p,t,None if method=='surface' else mesh.volume)
        if method=='surface':
            trace=constrained_boundary(shells,boundary)
            if trace is not None:
                # edge is the edge of a regular tetrahedron in millimetres.
                # The PLC mesher accepts a maximum volume in cubic millimetres.
                p,t=volume_mesh(trace,root,edge**3/(6.*np.sqrt(2.)),0.,'interior')
                boundary=checked_surface(p,t,mesh.volume)
        verify_constraints(shells,boundary)
        return p,t

    previous=root/name
    last_error=None
    if previous.is_file():
        try:
            with np.load(previous) as data:
                effective=float(data['effective_tolerance']) if 'effective_tolerance' in data else tolerance
            p,t=certify(*read_mesh(previous,key))
            save_mesh(path,key,p,t,effective)
            return p,t
        except InadmissibleBoundary as error:
            last_error=error
    factors=((.5,.25) if last_error is not None else (1.,.5,.25)) if method=='surface' else (1.,)
    source=shells[0] if method=='surface' else mesh
    for factor in factors:
        try:
            p,t=certify(*native_mesh(source,root,method,edge,tolerance*factor,threads))
        except InadmissibleBoundary as error:
            last_error=error
            continue
        save_mesh(path,key,p,t,tolerance*factor)
        return p,t
    raise InadmissibleBoundary(f'No valid analysis discretization: {last_error}')
