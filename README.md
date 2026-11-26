# Generative Structural Optimizer (GSO)

Give it a design space, fixed faces, loads and a material budget; it grows a part
inside that space and writes a mesh (OBJ). See `ProjectGoal.md` for where this is headed.

## How a part is generated

`generate()` in `tools/gso_workflows/volume_study.py` runs two stages on one part:

1. **Density stage** (`optimize()` in `volume_study.py`, `stress_evolution.py`).
   The design space is filled with voxels of partial density. Each iteration solves
   every static load case and moves material to where it carries load, inside the
   volume budget. This is where holes and members appear.
2. **Boundary stage** (`surface_search.py`, `scenario_feedback.py`, `boundary_shape.py`).
   The density field becomes a closed surface. That surface is then moved in and out,
   one load case at a time, using stress and stiffness sensitivities computed on a
   tetrahedral mesh. Transient cases (braking, drop, strike) are evaluated here.

The result is the best surface found, written as an OBJ with a `selection.json` report.

## Layout

| Path | What it is |
|---|---|
| `tools/gso_workflows/` | The Python pipeline (both stages, meshing, tetra solver, transient and nonlinear solvers) |
| `src/`, `include/` | C++ helpers the pipeline calls: SOLIDWORKS import, voxel cell stiffness, surface-motion solve, level-set update, Newmark time integration, finite-strain material law |
| `studies/sunbeam33/` | The drone arm study: `search.py` is the entry point, `geometry.py` holds the face picks |
| `studies/target_examples/compare.py` | Two small synthetic parts generated from scratch |
| `tests/` | Tests that check against something independent (finite differences, exact fields), and one smoke test of the density stage |

The C++ builds one executable (`gso`) and three libraries (`gso_field`, `gso_mechanics`,
`gso_evolution`) that Python loads with ctypes. They must sit next to the executable.

## Build and run (Windows, MSVC)

```
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release
pip install -r requirements.txt
python studies/sunbeam33/search.py "studies/sunbeam33/inputs/rev2/Sunbeam Optimized Arm - Front Left.SLDPRT" out/arm.obj --service-cases
```

`gso import` needs SOLIDWORKS installed. Several scripts add `build/sunbeam33/deps` and
`build/sunbeam33/meshlib-runtime` to the Python path; that was a local dependency folder
and is not needed if `requirements.txt` is installed normally.

## Assumptions to check

These are the modelling decisions currently in the code. They decide the result more
than the optimizer does.

**Not modelled at all**
- **Vibration.** There is no oscillating-load, natural-frequency or fatigue case. The
  original arm failed from oscillating motors, so this is the main gap.
- **Print direction.** Every material is isotropic. A printed part is much weaker
  across layers than along them.

**Study setup (`studies/sunbeam33/geometry.py`)**
- The design space is the bounding box of the existing arm.
- The volume budget equals the existing arm's volume, so the generator can only
  redistribute material. It cannot make the arm heavier, even if that is what it needs.
- The hub faces are held perfectly rigid, which overstates stress next to them.
- The base load is 33 N on the motor pad. The source of that number is not recorded.

**Service cases (`service_scenarios.py`)**
- Side and span loads reuse the same 33 N. Motor torque is 1000 N mm.
- Braking: 15,000 rpm to zero in 0.1 s with rotor inertia 5e-5 kg m2. That is a
  half-sine torque pulse peaking at about 1230 N mm.
- Drop: 250 mm (2.2 m/s at impact) onto a 25 N/mm contact. Strike: 3 m/s onto 100 N/mm.
  Both contact stiffnesses are assumed, and impact results depend strongly on them.
- Airframe: 1.6 kg with assumed radii of gyration of 100, 100 and 135 mm. Motor: 0.09 kg.

**Materials (`filaments.py`)**
- ABS: 2200 MPa, Poisson 0.35, 1040 kg/m3, no yield strength recorded.
- PLA and PETG moduli and yield strengths are Prusament data-sheet values for
  horizontally printed coupons. Poisson ratios are assumed.

**Solver**
- Stresses come from 4-node (linear) tetrahedra. SOLIDWORKS Simulation defaults to
  10-node elements, so peak stresses will differ until meshes are fine.
- No result has been compared with SOLIDWORKS Simulation yet.
- For transient and finite-strain cases the boundary stage uses stress-squared as its
  search direction. That is a heuristic, not a true gradient.
- Each candidate is re-solved on three meshes before it is accepted. This is thorough
  and slow, and is a candidate for simplification.
