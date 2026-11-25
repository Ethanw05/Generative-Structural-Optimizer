"""Follow the same physical stress regions while the boundary and analysis mesh change."""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from .stress_regions import memberships, stress_basins, basin_peaks


def moments(stress, volume, labels, size, power):
    row, region, part = memberships(labels)
    measure = volume[row]*part
    volume_by_region = np.bincount(region, weights=measure, minlength=size)
    maximum = np.zeros(size)
    np.maximum.at(maximum, region, stress[row])
    scaled = np.divide(stress[row], maximum[region], out=np.zeros(len(row)), where=maximum[region]>0.)
    sums = np.bincount(region, weights=measure*scaled**power, minlength=size)
    mean = np.divide(sums, volume_by_region, out=np.zeros(size), where=volume_by_region>0.)
    return maximum, maximum*mean**(1./power), volume_by_region


def relative_stress(stress, volume, labels, reference, power):
    """Normalize the p-mean of regional stresses by the same metric on the reference part."""
    row, region, part = memberships(labels)
    _, regional, size = moments(stress, volume, labels, len(reference), power)
    # All locations in one physical scenario share a stress scale. A tiny
    # reference prestress must not magnify a benign local change in MPa.
    reference_max = float(np.max(reference))
    if reference_max <= 0.:
        raise ValueError("Expected positive stress normalization for the load case")
    scale = reference_max*np.mean((reference/reference_max)**power)**(1./power)
    relative = regional/scale
    maximum = relative.max()
    if maximum == 0.:
        return 0., np.zeros_like(stress), np.zeros_like(stress)
    value = maximum*np.mean((relative/maximum)**power)**(1./power)
    ratio = stress[row]/(value*scale)
    weight = part / (len(reference)*size[region])
    derivative = weight*ratio**(power-1.)/scale
    volume_derivative = weight*value/power * (ratio**power-(relative[region]/value)**power)
    return (value, np.bincount(row,weights=derivative,minlength=len(stress)),
            np.bincount(row,weights=volume_derivative,minlength=len(stress)))


class ReferenceRegions:
    """Partition the reference material once, then measure elemental peaks in the same regions."""

    def __init__(self):
        self.tree = None
        self.labels = None
        self.scale = None
        self.anchors = None

    def assign(self, points, tets, grad, volume, vm, skin, solve, power=8.):
        centers = points[tets].mean(1)
        if self.tree is not None:
            _, nearest = self.tree.query(centers)
            return self.labels[nearest]

        # Smooth only the field used to locate catchments. The acceptance measurements
        # below use the original, unsmoothed three-dimensional element stresses.
        tri = points[skin]
        area = .5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()
        length = 2.*volume.sum()/area
        mass = np.bincount(tets.ravel(),weights=np.repeat(volume/4.,4),minlength=len(points))
        forcing = np.bincount(tets.ravel(),weights=np.repeat(volume*vm**2/4.,4),minlength=len(points))
        laplace = (volume*length**2)[:,None,None]*(grad @ grad.transpose(0,2,1))
        rows = np.repeat(tets,4,axis=1).ravel()
        cols = np.tile(tets,(1,4)).ravel()
        matrix = sparse.coo_matrix((laplace.ravel(),(rows,cols)),shape=(len(points),len(points))).tocsr()
        matrix += sparse.diags(mass)
        filtered = solve(matrix,forcing)
        if np.linalg.norm(matrix@filtered-forcing) > 1.e-7*np.linalg.norm(forcing):
            raise ValueError('Stress region recovery failed equilibrium')
        self.labels = stress_basins(tets, np.sqrt(np.maximum(0.,filtered[tets].mean(1))))
        self.tree = cKDTree(centers)
        indices, _ = basin_peaks(vm,volume,self.labels)
        self.anchors = centers[indices]
        n = int(self.labels.max())+1 if isinstance(self.labels,np.ndarray) else self.labels.shape[1]
        _, scales, _ = moments(vm,volume,self.labels,n,power)
        peak=float(vm.max())
        # An entirely unloaded reference has no stress scale. Use the study stress unit (MPa).
        floor=np.finfo(float).eps**.5*peak if peak>0. else 1.
        self.scale=np.maximum(scales,floor)
        return self.labels


def compare_regions(reference, candidate):
    """Only rate a regional change on meshes which sample material in both shapes."""
    old = np.array([case['local_peaks_MPa'] for case in reference])
    new = np.array([case['local_peaks_MPa'] for case in candidate])
    present = (np.array([case['local_region_volumes_mm3'] for case in reference]) > 0.) & (
        np.array([case['local_region_volumes_mm3'] for case in candidate]) > 0.)
    sampled = present.all(axis=0)
    delta = new-old
    return dict(sampled_in_both_on_all_meshes=int(sampled.sum()),
                lower_on_all_meshes=int(np.sum(sampled & np.all(delta<0.,axis=0))),
                higher_on_all_meshes=int(np.sum(sampled & np.all(delta>0.,axis=0))),
                changes_MPa_by_mesh=[[float(value) if valid else None for value,valid in zip(row,mask,strict=True)]
                                    for row,mask in zip(delta,present,strict=True)])
