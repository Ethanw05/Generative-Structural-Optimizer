"""Manufactured interfaces for coupled normal transport and area regularization."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'tools'), str(root / 'build/sunbeam33/deps')]
import numpy as np
from gso_workflows.front_transport import advance

executable = Path(sys.argv[1])
diffusion = .25  # mm^2 of mean-curvature flow in one proposal


def evolve(field, h, speed=0., retained=None, restriction=None):
    field = np.ascontiguousarray(field, dtype=np.float32)
    free = np.full_like(field, 100.)
    return advance(field, np.broadcast_to(speed, field.shape),
                   free if retained is None else retained,
                   -free if restriction is None else restriction, h, executable, diffusion, 1., conserve_volume=False)


def interface(profile, positions):
    return np.interp(0., profile, positions)


def check(h):
    axis = np.arange(-5., 5. + .1*h, h)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij', sparse=True)
    mid = len(axis) // 2
    # A tilted flat front moves by its mechanical speed at both resolutions.
    normal = np.array([1., -2., 3.]) / np.sqrt(14.)
    plane = x*normal[0] + y*normal[1] + z*normal[2]
    swept = evolve(plane, h, speed=.25)
    interior = (abs(x)<2.) & (abs(y)<2.) & (abs(z)<2.) & (abs(plane)<.25)
    np.testing.assert_allclose(swept[interior], (plane-.25)[interior], atol=2.e-5)

    # A sphere under pure area descent has radius sqrt(R^2 - 4*t).
    sphere = np.sqrt(x*x + y*y + z*z) - 3.
    curved = evolve(sphere, h)
    radius = interface(curved[mid, mid, mid:], axis[mid:])
    radius_error = abs(radius - np.sqrt(9.-4.*diffusion))
    assert radius_error < .025, (h, radius, radius_error)

    # On a small-amplitude graph the linearized solution is the heat equation.
    period = 2.
    wave = np.cos(2.*np.pi*x/period)
    corrugated = np.broadcast_to(z - .04*wave, plane.shape).copy()
    corrected = evolve(corrugated, h)
    amplitude = -interface(corrected[mid+round(period/(2*h)), mid], axis)
    expected = .04*np.exp(-diffusion*(2.*np.pi/period)**2)
    assert abs(amplitude-expected) < .002, (h, amplitude, expected)

    # One-sided CAD constraints and a fixed patch constrain every substep.
    retained = np.minimum(100.+0.*sphere, sphere)
    assert np.all(evolve(sphere, h, retained=retained) <= retained + 2.e-6)
    limit = np.full_like(sphere, -100.)
    cap = -sphere
    assert np.all(evolve(cap, h, restriction=cap) >= cap - 2.e-6)
    mask = (x < -1.) & (abs(z) < 2*h) & (abs(y) < 1.)
    ceiling = np.where(mask, corrugated, 100.)
    floor = np.where(mask, corrugated, limit)
    held = evolve(corrugated, h, retained=ceiling, restriction=floor)
    np.testing.assert_array_equal(held[mask], corrugated[mask].astype(np.float32))
    print(f'{h:g} mm lattice: sphere radius {radius:.6f} mm, ripple amplitude {amplitude:.6f} mm')
    return radius_error, abs(amplitude-expected)


# At a symmetry center the normal vanishes. A quadratic defining function
# for a shrinking sphere has an exact increment even at this critical point.
h = .125
q = np.arange(-12, 13, dtype=np.float32)*h
x, y, z = np.meshgrid(q, q, q, indexing='ij', sparse=True)
potential = (x*x + y*y + z*z - 1.)/2.
small_time = h*h/8.
free = np.full_like(potential, 100.)
after = advance(potential, np.zeros_like(potential), free, -free, h, executable, small_time, 1., conserve_volume=False)
np.testing.assert_allclose(after[12,12,12] - potential[12,12,12], 2.*small_time, atol=1.e-7)

# The production step preserves the requested mechanical volume, including
# active CAD constraints. A stationary sphere should not shed material simply
# because it is curved, and a simultaneous outward load must still add it.
radius = np.sqrt(x*x+y*y+z*z) - 1.
clip = np.broadcast_to(x - .88, radius.shape)
for travel in (0., .12, -.12):
    speed = np.full_like(radius, travel)
    raw = advance(radius, speed, free, clip, h, executable)
    constrained = advance(radius, speed, free, clip, h, executable, .02, .4)
    measure = lambda field: np.clip(.5 - field.astype(float)/(2.*h), 0., 1.).sum()*h**3
    np.testing.assert_allclose(measure(constrained), measure(raw), atol=2.e-6)
    assert np.all(constrained >= clip)

coarse = check(.25)
fine = check(.125)
assert fine[0] < coarse[0] and fine[1] < coarse[1], (coarse, fine)

# Validate the whole-surface diagnostic on actual triangles and a flat control.
import trimesh
from gso_workflows.surface_roughness import surface_residuals

u, v = np.meshgrid(np.linspace(-3.,3.,81), np.linspace(-3.,3.,81), indexing='ij')
ids = np.arange(u.size).reshape(u.shape)
a,b,c,d = (ids[:-1,:-1],ids[1:,:-1],ids[1:,1:],ids[:-1,1:])
triangles = np.concatenate([np.stack(face,axis=-1).reshape(-1,3) for face in ((a,b,c),(a,c,d))])
heights = (np.zeros_like(u), .08*np.cos(6.*u)*np.cos(6.*v))
for height, corrugated in zip(heights,(False,True),strict=True):
    mesh = trimesh.Trimesh(np.column_stack((u.ravel(),v.ravel(),height.ravel())),triangles,process=False)
    residual, coverage = surface_residuals(mesh, .8, count=256)
    assert coverage > .98
    if corrugated:
        assert np.median(residual) > .015
    else:
        assert np.max(residual) < 1.e-12
