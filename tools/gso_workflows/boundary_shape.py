"""Evolve the material boundary using its mechanical sensitivities and declared CAD constraints."""
import numpy as np
from meshlib import mrmeshpy as mr, mrmeshnumpy as arrays
from scipy.ndimage import map_coordinates
import trimesh
from .mesh_quality import InadmissibleBoundary, validate_single_part
from .reconstruction import distance_samples, region_distances, surface_from_signed
from .surface_motion import prescribed_faces, solve_motion
from .front_transport import advance


def local_signed_surface(mesh, first, shape, spacing, band):
    surface=arrays.meshFromFacesVerts(np.asarray(mesh.faces,np.int32),np.asarray(mesh.vertices,float))
    cfg=mr.MeshToDistanceVolumeParams()
    cfg.dist.signMode=mr.SignDetectionMode.HoleWindingRule
    cfg.dist.maxDistSq=band**2
    cfg.dist.nullOutsideMinMax=False
    cfg.vol.dimensions=mr.Vector3i(*map(int,shape))
    cfg.vol.voxelSize=mr.Vector3f(*([spacing]*3))
    cfg.vol.origin=mr.Vector3f(*map(float,first-.5*spacing))
    return arrays.getNumpy3Darray(mr.meshToDistanceVolume(surface,cfg)).copy()


def pin_selected_faces(study, first, spacing, field, retained, restriction):
    """Hold both interpolation endpoints of prescribed CAD faces throughout evolution."""
    shape = np.array(field.shape)
    halo = np.sqrt(3.) * spacing
    for face in prescribed_faces(study):
        vertices = face.triangles.reshape(-1, 3)
        low = np.maximum(0, np.floor((vertices.min(axis=0) - halo - first) / spacing).astype(int))
        high = np.minimum(shape, np.ceil((vertices.max(axis=0) + halo - first) / spacing).astype(int) + 1)
        if np.any(high <= low):
            continue
        # Slice by the longest bounding-box axis to bound temporary storage on large faces.
        axis = int(np.argmax(high - low))
        axes = [np.arange(lo, hi) for lo, hi in zip(low, high)]
        for station in axes[axis]:
            section = list(axes)
            section[axis] = np.array([station])
            indices = np.stack(np.meshgrid(*section, indexing='ij'), axis=-1).reshape(-1, 3)
            selected = face.contains_projection(first + spacing * indices, halo)
            sites = tuple(indices[selected].T)
            fixed = np.maximum(np.minimum(field[sites], retained[sites]), restriction[sites])
            retained[sites] = restriction[sites] = fixed


def admissible_growth(i,baseline,gradient,clearance,motion,spacing):
    """A front may travel beside an exclusion. Stop only a path passing through one."""
    distance=baseline[i]
    test=(distance>0.)&(distance<motion+spacing)&(clearance[i]<distance+spacing)&(motion>0.)
    j,k=np.nonzero(test)
    if not len(j):return motion
    normal=gradient[j,k].copy()
    length=np.linalg.norm(normal,axis=1)
    normal/=np.maximum(length[:,None],1.e-12)
    travel=distance[j,k]+.25*spacing
    origin=np.column_stack((np.full(len(j),i),j,k))
    blocked=length<.25
    steps=max(1,int(np.ceil(travel.max()/(.5*spacing))))
    for step in range(steps+1):
        q=origin-normal*(travel*(step/steps)/spacing)[:,None]
        value=map_coordinates(clearance,q.T,order=1,mode='nearest')
        blocked|=value<-.0001*spacing
    motion[j[blocked],k[blocked]]=0.
    return motion


def propose(study, source, evidence, exe, work, spacing, move, weights, regularization_mm=None):
    if regularization_mm is None:
        regularization_mm = move
    slack = max(0., study.volume_limit_mm3 - source.volume)
    motion = solve_motion(study, evidence, exe, work, move, spacing, weights, slack, regularization_mm)
    # The heat length is expressed in model units and scales down on line search.
    # It is independent of the sampling lattice used to realize a trial.
    diffusion_mm2 = .5 * move * regularization_mm
    if np.max(np.abs(motion.values)) <= 64. * np.finfo(float).eps * move:
        return None

    local = trimesh.Trimesh((source.vertices - study.world_origin) @ study.world_rotation.T,
                            source.faces, process=False)
    bounds = np.asarray(study.space.envelope.bounding_box()).reshape(2, 3)
    first = spacing * (np.floor(bounds[0] / spacing) - 2.371)
    shape = np.ceil((bounds[1] - first) / spacing).astype(int) + 4
    band = move + 3. * spacing
    baseline = local_signed_surface(local, first, shape, spacing, band)
    if not np.all(np.isfinite(baseline)) or baseline.min() >= 0.:
        raise ValueError('Unable to recover the signed material boundary')
    restriction = distance_samples(study.space.envelope, first, shape, spacing)
    clearance = region_distances(study.space.exclusions, first, shape, spacing, band)
    np.maximum(restriction, -clearance, out=restriction)
    retained = region_distances(study.space.preserved, first, shape, spacing, band)
    pin_selected_faces(study, first, spacing, baseline, retained, restriction)
    boundary_velocity = np.zeros_like(baseline)
    for i, section in enumerate(baseline):
        j, k = np.nonzero(abs(section) <= move + spacing)
        if not len(j):
            continue
        points = first + spacing * np.column_stack((np.full_like(j, i), j, k))
        velocity = np.zeros_like(section)
        velocity[j, k] = motion.sample(points)
        if study.space.exclusions:
            left, right = max(0, i - 1), min(shape[0] - 1, i + 1)
            dx = (baseline[right] - baseline[left]) / ((right - left) * spacing)
            dy, dz = np.gradient(section, spacing)
            admissible_growth(i, baseline, np.stack((dx, dy, dz), axis=-1), clearance, velocity, spacing)
        boundary_velocity[i] = velocity
    del clearance
    retreat = np.minimum(boundary_velocity, 0.)
    outward = np.maximum(boundary_velocity, 0.)
    del boundary_velocity

    def field_at(growth_fraction):
        return advance(baseline, retreat + growth_fraction * outward, retained, restriction, spacing, exe,
                       diffusion_mm2, move)

    def extract(field, adaptivity=0.):
        return surface_from_signed(field, first, spacing, band, adaptivity=adaptivity)

    realized_field = field_at(1.)
    candidate = extract(realized_field)
    if candidate.volume > study.volume_limit_mm3:
        # Reduce only already-proposed outward travel. Never inflate an under-budget
        # result, or remove unrelated material, just to hit the budget after extraction.
        realized_field = field_at(0.)
        candidate = extract(realized_field)
        if candidate.volume > study.volume_limit_mm3:
            raise InadmissibleBoundary('Resolved surface cannot meet the material budget at this step')
        low, high = 0., 1.
        for _ in range(24):
            fraction = (low + high) / 2.
            trial_field = field_at(fraction)
            trial = extract(trial_field)
            if trial.volume > study.volume_limit_mm3:
                high = fraction
            else:
                low, candidate, realized_field = fraction, trial, trial_field
            if (high - low) * move <= 1.e-4 * spacing:
                break
    validate_single_part(candidate)
    from .surface_accuracy import sample_deviation
    adaptivity = .01
    while adaptivity >= np.finfo(np.float32).eps:
        adaptive = extract(realized_field, adaptivity=adaptivity)
        if len(adaptive.faces) >= len(candidate.faces):
            break
        same_topology = (adaptive.is_volume and adaptive.euler_number == candidate.euler_number
                         and adaptive.body_count == candidate.body_count)
        volume_error = abs(adaptive.volume - candidate.volume)
        if (same_topology and adaptive.volume <= study.volume_limit_mm3
                and volume_error <= spacing * candidate.area / 100.):
            if sample_deviation(candidate, adaptive, spacing / 4.) <= spacing / 4.:
                try:
                    validate_single_part(adaptive)
                    candidate = adaptive
                    break
                except InadmissibleBoundary:
                    pass
        adaptivity /= 4.
    candidate.vertices = candidate.vertices @ study.world_rotation + study.world_origin
    return candidate
