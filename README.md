# Generative Structural Optimizer (GSO)

Give it a part or a design space, fixtures, loads, materials and targets (safety factors,
mass, stiffness, frequencies, fatigue life and so on). It generates the part that best
meets them and certifies the result. See `ProjectGoal.md` for where this is headed.

## Status

GSO is being rebuilt as one self-contained C++ program on the `cpp-rewrite` branch.
`docs/port-plan.md` is the design and build plan; `docs/study-format.md` describes study
files.

| Phase | What | State |
|---|---|---|
| 0 | Foundation: build, core, study files, OBJ and SOLIDWORKS import, GPU discovery | done |
| 1 | Static analysis on curved 10-node tetrahedra | next |
| 2 | Modal, harmonic, fatigue, transient and nonlinear analysis | |
| 3 | Topology stage (GPU multigrid) | |
| 4 | Shape stage | |
| 5 | Pareto front, certification report, resume | |
| 6 | Sunbeam drone-arm study | |
| 7 | Remove the Python | |

Until phase 7, the previous Python generator in `tools/gso_workflows` remains as a
reference only. Its captured results are in `reference/`. It runs against the old
binaries kept in `reference/python-bin/`, which are not in git.

## Build (Windows)

Requirements:
- Visual Studio 2022 Build Tools (C++ workload);
- CMake 3.28 or newer;
- CUDA Toolkit 13;
- an NVIDIA GPU.

Every other dependency is fetched and pinned by CMake.

```
cmake --preset windows
cmake --build --preset release
ctest --preset release
```

The first configure downloads the dependencies. Open a new terminal after installing the
CUDA Toolkit so that `CUDA_PATH` is set.

## Use

```
gso info                                    # version, CUDA, GPU
gso check studies/necked-link/study.json    # validate a study file
gso import part.SLDPRT part.obj             # convert a part (needs SOLIDWORKS for SLDPRT/SLDASM)
```

## Layout

| Path | What it is |
|---|---|
| `include/gso/<module>/`, `src/<module>/` | The library, one folder per module: `core`, `geometry`, `solvers`, `study`, `cad` |
| `apps/gso/` | Command-line front end |
| `studies/` | Study files (`*/study.json`) and the parts they use. `studies/sunbeam33/` still holds the Python version of the drone-arm study |
| `tests/` | Verification tests against independent answers |
| `docs/` | Plan and file formats |
| `reference/` | Results of the Python generator, for comparison during the rewrite |

## Assumptions to check

These are the modelling decisions in the current Sunbeam study (`studies/sunbeam33/`).
They decide the result more than the optimizer does, and are being verified before the
Sunbeam study is rebuilt.

**Not modelled at all (yet)**
- **Vibration.** There is no oscillating-load, natural-frequency or fatigue case. The
  original arm failed from oscillating motors, so this is the main gap (phase 2).
- **Print direction.** Every material is isotropic, but a printed part is much weaker
  across layers than along them. The new material model supports this (Tsai-Wu,
  transversely isotropic).

**Study setup (`studies/sunbeam33/geometry.py`)**
- The design space is the bounding box of the existing arm.
- The volume budget equals the existing arm's volume. The new generator makes mass a free
  target instead.
- The hub faces are held perfectly rigid, which overstates stress next to them.
- The base load is 33 N on the motor pad. The source of that number is not recorded.

**Service cases (`service_scenarios.py`)**
- Side and span loads reuse the same 33 N. Motor torque is 1000 N mm.
- **Braking:**
  - 15,000 rpm to zero in 0.1 s.
  - Rotor inertia 5e-5 kg m2, which looks high for a small prop.
  - This is a half-sine torque pulse peaking at about 1230 N mm.
- **Drop and strike:**
  - Drop: 250 mm (2.2 m/s at impact) onto a 25 N/mm contact.
  - Strike: 3 m/s onto 100 N/mm.
  - Both contact stiffnesses are assumed, and impact results depend strongly on them.
- **Masses:** airframe 1.6 kg, with assumed radii of gyration of 100, 100 and 135 mm.
  Motor 0.09 kg.

**Materials (`filaments.py`)**
- ABS: 2200 MPa, Poisson 0.35, 1040 kg/m3, no yield strength recorded.
- PLA and PETG moduli and yield strengths are Prusament data-sheet values for
  horizontally printed coupons. Poisson ratios are assumed.

**Previous solver (the Python reference)**
- Stresses come from 4-node tetrahedra, whose peak stresses do not converge at sharp
  edges. On necked-link, utilisation rose from 0.85 to 1.05 under refinement. The new
  solver uses 10-node tetrahedra and critical-distance stress.
- No result has been compared with SOLIDWORKS Simulation yet.
