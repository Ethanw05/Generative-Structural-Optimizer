"""General density proposal and stress refinement workflow using the C++ MLS cubature kernel."""
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import subprocess
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import spsolve
import trimesh

from .design_space import DesignSpace,Face,distribute_surface_force
from .mesh_quality import DisconnectedMaterial,validate_single_part
from .reconstruction import MaterialBoundary
from .cad_wrench import SurfaceWrench, grid_wrench
from .stress_regions import support_edges
from .stress_evolution import evaluate_design,refine


@dataclass
class SurfaceLoad:
    face: Face
    force_n: np.ndarray
    transfer_face: Face | None = None


@dataclass
class VolumeStudy:
    space: DesignSpace
    fixed_faces: tuple
    loads: dict
    young_mpa: float
    poisson: float
    volume_limit_mm3: float
    world_rotation: np.ndarray
    world_origin: np.ndarray
    mechanics_executable: Path | None = None
    histories: dict = field(default_factory=dict)
    density_kg_m3: float | None = None
    reference_surface_local: trimesh.Trimesh | None = None

    solid_laws: dict = field(default_factory=dict)
    case_materials: dict = field(default_factory=dict)

    def scenarios(self):
        if self.loads.keys() & self.histories.keys():
            raise ValueError('Static and history scenarios need distinct names')
        names=(*self.loads,*self.histories)
        if (self.solid_laws.keys()|self.case_materials.keys())-set(names):raise ValueError('A material refers to an absent case')
        return names

    def for_case(self,label):
        spec=self.case_materials.get(label)
        return spec.apply(self) if spec is not None else self

    def solid_model(self,label):
        law=self.solid_laws.get(label)
        description=(law.description() if law is not None else
                     dict(kinematics='small strain',material='linear isotropic elastic'))
        spec=self.case_materials.get(label)
        if spec is not None:
            description.update(material_name=spec.name,source=spec.source,assumptions=spec.assumptions,
                               horizontal_tensile_yield_MPa=spec.tensile_yield_mpa)
        active=self.for_case(label)
        return dict(**description,young_MPa=active.young_mpa,poisson=active.poisson,density_kg_m3=active.density_kg_m3)

    def seed_loads(self):
        seeds=dict(self.loads)
        for name,case in self.histories.items():
            ports=[x.attachment for x in case.loads]+[x.unit_normal_load for x in case.contacts]
            for i,port in enumerate(ports):
                key=f"{name} / excitation {i+1}"
                if key in seeds:raise ValueError("Duplicate excitation name")
                seeds[key]=port
        return seeds



@dataclass
class Design:
    field: np.ndarray
    protected: np.ndarray
    origin: np.ndarray
    pitch: float
    workspace: Path


def linear_solver(work):
    configured=os.environ.get('PYPARDISO_MKL_RT')
    dll=Path(configured) if configured else next((work/'intel').rglob('mkl_rt*.dll'),None)
    if dll is None:
        return spsolve,None
    dll=dll.resolve(strict=True)
    os.environ['PYPARDISO_MKL_RT']=str(dll)
    os.environ.setdefault('MKL_NUM_THREADS','6')
    handle=os.add_dll_directory(str(dll.parent)) if os.name=='nt' else None
    from .sparse_solve import PardisoLinearSolver
    return PardisoLinearSolver(),handle


def stencil(executable,work,pitch,young,poisson):
    path=work/f'mls-{pitch}-{young}-{poisson}.txt'
    subprocess.run([executable,'mls-stencil',str(pitch),str(young),str(poisson),path],check=True)
    return np.loadtxt(path)


def block_assembler(conn,nfree,stiffness):
    corners=np.array(list(product(range(2),repeat=3)))
    directions=(corners[None,:,:]-corners[:,None,:]+1)@np.array([9,3,1])
    occupied=np.zeros((nfree,27),bool)
    columns=np.zeros((nfree,27),np.int32)
    mapping=[]
    for i,j in product(range(8),repeat=2):
        row,col=conn[:,i],conn[:,j]
        valid=(row>=0)&(col>=0)
        direction=directions[i,j]
        occupied[row[valid],direction]=True
        columns[row[valid],direction]=col[valid]
        mapping.append((np.flatnonzero(valid),row[valid]*27+direction,i,j))
    permutation=np.full(nfree*27,-1,np.int32)
    permutation[np.flatnonzero(occupied.ravel())]=np.arange(occupied.sum(),dtype=np.int32)
    mapping=[(ix,permutation[slot],stiffness[3*i:3*i+3,3*j:3*j+3]) for ix,slot,i,j in mapping]
    indptr=np.r_[0,occupied.sum(1).cumsum()].astype(np.int32)
    indices=columns[occupied]
    def assemble(multiplier):
        blocks=np.zeros((len(indices),3,3))
        for ix,pos,block in mapping:blocks[pos]+=multiplier[ix,None,None]*block
        matrix=sparse.bsr_matrix((blocks,indices,indptr),shape=(3*nfree,3*nfree)).tocsr()
        matrix.sort_indices()
        return matrix
    return assemble


def optimize(study,pitch,iterations,executable,work,*,refinement_steps=24,filter_length_mm=None,targets=None):
    if study.histories:
        study=replace(study,loads=study.seed_loads())
    if pitch<=0 or not np.isfinite(pitch) or iterations<1 or not study.loads:
        raise ValueError('Expected positive resolution, iteration budget, and at least one load')
    sigma=1.2 if filter_length_mm is None else filter_length_mm/pitch
    bounds=np.array(study.space.envelope.bounding_box()).reshape(2,3)
    origin=pitch*np.floor(bounds[0]/pitch)
    shape=np.ceil((bounds[1]-origin)/pitch-1.e-10).astype(int)
    allowed,protected=study.space.sample(origin,shape,pitch)
    hard=protected[allowed]
    target=study.volume_limit_mm3/pitch**3
    cells=np.argwhere(allowed)
    if not (hard.sum()+.015*(len(cells)-hard.sum())<=target<len(cells)):
        raise ValueError('Volume budget outside feasible material bounds of this design space')
    corners=np.array(list(product(range(2),repeat=3)))
    nodes=shape+1
    cell_nodes=np.ravel_multi_index((cells[:,None,:]+corners).transpose(2,0,1),nodes)
    used=np.unique(cell_nodes)
    xyz=origin+np.stack(np.unravel_index(used,nodes),axis=1)*pitch
    fixed=np.zeros(len(used),bool)
    for face in study.fixed_faces:
        selected=face.contains_projection(xyz,pitch*.5+1.e-7*pitch)
        if not np.any(selected):raise ValueError('Unresolved CAD fixture, refine analysis spacing')
        fixed|=selected
    free=used[~fixed];lookup=np.full(int(nodes.prod()),-1,np.int32)
    lookup[free]=np.arange(len(free),dtype=np.int32)
    connection=lookup[cell_nodes]
    cases={name:(grid_wrench(load,origin,shape,pitch,lookup,executable) if isinstance(load,SurfaceWrench)
                 else distribute_surface_force(load.face,load.force_n,origin,shape,pitch,lookup))
           for name,load in study.loads.items()}
    ke=stencil(executable,work,pitch,study.young_mpa,study.poisson)
    assemble=block_assembler(connection,len(free),ke)
    solve,handle=linear_solver(work)
    density=np.full(len(cells),(target-hard.sum())/(len(cells)-hard.sum()))
    density[hard]=1.
    grid=np.zeros(tuple(shape))
    weights=gaussian_filter(allowed.astype(float),sigma,mode='constant')[allowed]
    signature=hashlib.sha256(b'volume-study-1')
    for array in (allowed,protected,lookup,ke,np.array([target,pitch,sigma]),*cases.values()):
        signature.update(np.ascontiguousarray(array).tobytes())
    tag=signature.hexdigest()[:24]
    checkpoint=work/f'screen-{tag}.npz'
    start=0
    if checkpoint.exists():
        with np.load(checkpoint) as prior:
            start=int(prior['generation']);density=prior['density'].copy()
            if start>iterations:raise ValueError('Requested fewer iterations than stored in this checkpoint')
    for generation in range(start,iterations):
        power=min(3.5,1.5+generation*.12)
        matrix=assemble(.00012+.99988*density**power)
        gain=np.zeros(len(density));scores={}
        for name,force in cases.items():
            u=solve(matrix,force)
            if not np.all(np.isfinite(u)) or np.linalg.norm(matrix@u-force)>1e-5*np.linalg.norm(force):
                raise RuntimeError('Screening equilibrium failed in '+name)
            disp=np.vstack([u.reshape(-1,3),np.zeros((1,3))])[connection].reshape(len(cells),24)
            energy=np.einsum('ij,ij->i',disp@ke,disp)
            work_nmm=float(force@u);scores[name]=work_nmm
            gain+=.99988*power*density**(power-1.)*energy/work_nmm
        grid[allowed]=gain*density
        gain=gaussian_filter(grid,sigma,mode='constant')[allowed]/(weights*np.maximum(density,.08))
        lo,hi=0.,max(gain.max()*1e4,1.)
        for _ in range(55):
            mid=(lo+hi)/2
            trial=np.clip(np.clip(density*np.sqrt(np.maximum(gain,0.)/mid),density-.135,density+.135),.015,1.)
            trial[hard]=1.
            if trial.sum()>target:lo=mid
            else:hi=mid
        density=trial
        temp=checkpoint.with_suffix('.writing.npz')
        np.savez_compressed(temp,generation=generation+1,density=density)
        temp.replace(checkpoint)
        print('screen',generation+1,scores,flush=True)
    shear=study.young_mpa/(2*(1.+study.poisson))
    # Two unit-shear MLS forms isolate distortional strain energy for RMS von Mises recovery.
    stress_matrix=shear**2*(5*stencil(executable,work,pitch,2.,0.)-2*stencil(executable,work,pitch,2.5,.25))/pitch**3
    edges=support_edges(allowed)
    def evaluate(x):return evaluate_design(x,ke,stress_matrix,connection,assemble,solve,cases,3.5,edges)
    def record(i,x,rs):
        print('verified proposal',i,{k:(r.compliance,r.stress_mpa) for k,r in rs.items()},flush=True)
    density=refine(density,allowed,hard,sigma,evaluate,refinement_steps,record,
                   stiffness=targets is None or targets.stiffness,
                   maximum_stress=targets is None or targets.maximum_stress)
    grid[allowed]=density
    np.savez_compressed(work/f'design-{tag}.npz',field=grid,protected=protected,origin=origin,pitch=pitch)
    return Design(grid,protected,origin,pitch,work)


def candidate_surface(study,design,boundary_spacing=None):
    spacing=boundary_spacing or min(.24,design.pitch/5.)
    material=MaterialBoundary.sample(design,study.space,spacing)
    mesh=material.at_budget(study.volume_limit_mm3)
    mesh.vertices=mesh.vertices@study.world_rotation+study.world_origin
    validate_single_part(mesh)
    if mesh.volume>study.volume_limit_mm3*(1.+1e-6):
        raise ValueError('Candidate exceeds its material budget')
    return mesh


def export_design(study,design,destination,verify=None):
    mesh=candidate_surface(study,design)
    if verify is not None:verify(mesh)
    return publish_surface(mesh,destination,design.workspace)


def publish_surface(mesh,destination,workspace):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_suffix('.writing.obj')
    mesh.export(temp,digits=13)
    try:
        actual=trimesh.load_mesh(temp,process=False,maintain_order=True)
        validate_single_part(actual)
        if abs(actual.volume-mesh.volume)>max(1.e-9,1.e-9*mesh.volume):
            raise ValueError('Export changed the volume of the verified part')
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    np.savez(workspace/(destination.stem+'.npz'),v=mesh.vertices,f=mesh.faces)
    return dict(volume_mm3=float(mesh.volume),triangles=len(mesh.faces))

def generate(study,pitch,iterations,executable,work,destination,*,refinements=1,targets=None,
             initial_surface=None,surface_generations=2,sequence_seed=None,case_updates=None,working_surface=None):
    """Evolve and verify the realized part before atomically publishing the selected OBJ."""
    from .surface_search import evolve_surface
    study=replace(study,mechanics_executable=Path(executable).resolve())
    if surface_generations < 1:
        raise ValueError('A positive surface-generation budget is required')
    if initial_surface is not None:
        baseline = initial_surface
        validate_single_part(baseline)
        if baseline.volume > study.volume_limit_mm3 * (1.+1.e-6):
            raise ValueError('The supplied starting solid exceeds the material budget')
    else:
        for attempt in range(refinements+1):
            design=optimize(study,pitch,iterations,executable,work,targets=targets)
            try:
                baseline=candidate_surface(study,design)
                break
            except DisconnectedMaterial:
                if attempt==refinements:
                    raise
                pitch*=.5
                print('Rejected disconnected candidate; refining the physical design to',pitch,'mm',flush=True)
    solver,handle=linear_solver(work)
    selected,report=evolve_surface(study,baseline,pitch,executable,
                                   work/(Path(destination).stem+'-surface'),solver,targets,generations=surface_generations,
                                   sequence_seed=sequence_seed,case_updates=case_updates,working_surface=working_surface)
    result=publish_surface(selected.mesh,destination,work)
    return dict(**result,selected=report['selected'],nondominated=report['nondominated'])
