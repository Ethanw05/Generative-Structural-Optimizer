"""Face-connected stress basins and a differentiable regional stress objective."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def stress_basins(tets, stress):
    """All ascent basins. A connected plateau has one outlet or one terminal maximum."""
    count = len(tets)
    if count == 0 or stress.shape != (count,) or not np.all(np.isfinite(stress)):
        raise ValueError('Invalid element stress field')
    faces = np.sort(tets[:, ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))].reshape(-1, 3), axis=1)
    order = np.lexsort(faces.T)
    matched = np.all(faces[order[1:]] == faces[order[:-1]], axis=1)
    if np.any(matched[1:] & matched[:-1]):
        raise ValueError('Nonmanifold tetrahedral adjacency')
    a, b = order[:-1][matched] // 4, order[1:][matched] // 4
    return ascent_basins(stress, a, b)


def support_edges(allowed):
    ids = np.full(allowed.shape, -1, dtype=np.intp)
    ids[allowed] = np.arange(np.count_nonzero(allowed))
    left, right = [], []
    for axis in range(allowed.ndim):
        grid = np.moveaxis(ids, axis, 0)
        near, far = grid[:-1], grid[1:]
        live = (near >= 0) & (far >= 0)
        left.append(near[live])
        right.append(far[live])
    return np.concatenate(left), np.concatenate(right)


def ascent_basins(stress, a, b, relative_tolerance=None):
    """Steepest ascent with equal outlet sharing on symmetric watershed divides."""
    count = len(stress)
    resolution = np.finfo(float).eps**.75 if relative_tolerance is None else relative_tolerance
    tolerance = resolution * np.max(np.abs(stress))
    equal = np.abs(stress[a] - stress[b]) <= tolerance
    if np.any(equal):
        graph = coo_matrix((np.ones(equal.sum(), dtype=bool), (a[equal], b[equal])), shape=(count, count))
        n, group = connected_components(graph, directed=False)
    else:
        n, group = count, np.arange(count)
    height = np.full(n, -np.inf)
    np.maximum.at(height, group, stress)
    a, b = group[a], group[b]
    rising = height[a] < height[b]
    distinct = height[a] != height[b]
    lower = np.where(rising, a, b)[distinct]
    higher = np.where(rising, b, a)[distinct]
    outlet_height = np.full(n, -np.inf)
    np.maximum.at(outlet_height, lower, height[higher])
    best = height[higher] >= outlet_height[lower] - tolerance
    outlets = coo_matrix((np.ones(best.sum(), dtype=bool), (lower[best], higher[best])), shape=(n,n)).tocsr()
    degree = np.diff(outlets.indptr)
    terminal = np.flatnonzero(degree == 0)
    if degree.max() <= 1:
        successor = np.arange(n)
        successor[degree == 1] = outlets.indices
        while np.any(successor != successor[successor]):
            successor = successor[successor]
        return np.searchsorted(terminal, successor[group])
    # A plateau with two equally high exits belongs fractionally to both. Choosing
    # one exit by node ID would make a symmetric part change when rotated/reindexed.
    row, col = outlets.nonzero()
    transition = coo_matrix((np.r_[1./degree[row], np.ones(len(terminal))],
                             (np.r_[row, terminal], np.r_[col, terminal])), shape=(n,n)).tocsr()
    distance = 1
    while distance < n:
        transition = transition @ transition
        distance *= 2
    return transition[:,terminal][group].tocsr()


def memberships(basins):
    if isinstance(basins, np.ndarray):
        return np.arange(len(basins)), basins, np.ones(len(basins))
    matrix = basins.tocoo()
    return matrix.row, matrix.col, matrix.data


def basin_peaks(stress, volume, basins):
    row, group, fraction = memberships(basins)
    size = np.bincount(group, weights=fraction*volume[row])
    order = np.lexsort((-stress[row], group))
    _, first = np.unique(group[order], return_index=True)
    return row[order[first]], size


def regional_stress(stress, volume, basins, power):
    """RMS of basin p-means and its partials for fixed watershed memberships."""
    if np.any(volume <= 0.) or power < 2. or np.any(stress < 0.):
        raise ValueError('Invalid stress aggregation data')
    row, group, fraction = memberships(basins)
    weight = volume[row]*fraction
    size = np.bincount(group, weights=weight)
    peak = np.zeros(len(size))
    np.maximum.at(peak, group, stress[row])
    scaled = np.divide(stress[row], peak[group], out=np.zeros(len(row)), where=peak[group] > 0.)
    moment = np.bincount(group, weights=weight * scaled**power) / size
    local = peak * moment**(1. / power)
    total = volume.sum()
    value = np.sqrt(size @ (local**2) / total)
    if value == 0.:
        return 0., np.zeros_like(stress), np.zeros_like(stress)
    local = local[group]
    ratio = np.divide(stress[row], local, out=np.zeros(len(row)), where=local > 0.)
    sensitivity = local / (value * total) * ratio**(power - 1.)
    direct = (local**2 - value**2 + (2. / power) * local**2 * (ratio**power - 1.)) / (2. * value * total)
    return (value, np.bincount(row, weights=fraction*sensitivity, minlength=len(volume)),
            np.bincount(row, weights=fraction*direct, minlength=len(volume)))
