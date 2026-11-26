# GSO: solver design and build plan

Status: **draft for review, revision 2** (2026-11-25). No code has been written yet.

Revision 1 was a port of the Python method. You rejected it: the solver must combine the
best current research into one method of our own, with one decision per problem and no
fallback or legacy code. The Python and its saved runs are now only reference results for
checking that the new solver gives better parts. They are not a design to reproduce.

Decisions you have made, which this plan builds on:
- **Mass is free.** It can go up or down; it is decided by the targets.
- **Targets are general.** Any number can be combined, for example a target weight or
  weight removal alongside a safety factor.
- **Any material type:** metals, composites and plastics.
- **10-node tetrahedra are the default element** for analysis.
- **One mesh per candidate during the search.** A convergence check runs only on the
  starting and final parts.

---

## 1. The method in one page

Generation runs in two stages on one geometry, followed by a certification step.

```
 CAD part / design space ──► STAGE 1  TOPOLOGY            ──► STAGE 2  SHAPE              ──► CERTIFY
 fixtures, loads, targets    sparse voxel grid, density      body-fitted 10-node tets,        independent re-analysis,
                             robust projection, all cases     level-set boundary moved by      two meshes + error
                             and targets as constraints       exact shape derivatives          estimate, report
                             (GPU multigrid)                  (AMG-preconditioned CG)
```

### Stage 1: topology, deciding where material goes

- **Geometry.**
  - A density field lives on a sparse voxel grid. Only cells in the design space are
    stored, as in the narrow-band grids of Liu et al. 2018.
  - The density is smoothed by a PDE (Helmholtz) filter and sharpened by a threshold
    projection.
  - It is evaluated in three versions, **eroded, nominal and dilated**: the robust
    formulation of Wang, Lazarov & Sigmund 2011. This gives a guaranteed minimum member
    size and fillet radius, and near-solid/void designs. It also models manufacturing
    tolerance.
- **Optimizer.**
  - An augmented-Lagrangian outer loop handles the millions of local stress and fatigue
    constraints without lumping them into one p-norm. This is the current state of the
    art for stress-constrained design (Senhora, Giraldo-Londoño, Menezes & Paulino 2020;
    PolyStress 2021; the dynamic extension 2021).
  - An MMA inner solver (Svanberg 1987/2002) handles the few global targets: mass,
    frequency, stiffness and displacement.
  - The OC update, which can only handle one constraint, goes.
- **Solver.**
  - Matrix-free geometric multigrid preconditions CG on the GPU. The preconditioner runs
    in FP32 and the outer CG in FP64.
  - **Block CG** solves all load cases at once, and every solve warm-starts from the
    previous design's solution.
  - **LOBPCG** with the same multigrid preconditioner gives the natural frequencies.
  - Sources: Aage et al. 2017 (Nature); Liu et al. 2018; Träff et al. 2023.
- **Starting point.** Either the design space alone, so the part is generated purely
  from the fixtures, forces and constraints you give, or an existing part, whose
  material becomes the initial density.

### Stage 2: shape, making the boundary smooth and accurate

- The projected density becomes a signed-distance level set.
- It is meshed body-fitted with **10-node tets curved onto the surface**, so stresses are
  evaluated on the true boundary rather than a staircase. The search runs on one mesh per
  candidate.
- The boundary moves by exact shape derivatives of every target, computed by adjoint and
  smoothed (H¹-regularised) into a normal velocity. It is remeshed after each move.
- The same augmented-Lagrangian/MMA optimizer runs, so stage 2 refines with exactly the
  constraints stage 1 used. This is the body-fitted level-set approach of Allaire,
  Dapogny & Frey 2014. It replaces the current ad-hoc candidate acceptance.

### Certification

- The final part, and the starting part when there is one, are analysed on **two
  meshes**, with a Zienkiewicz–Zhu error estimate for each limit.
- A limit counts as met only if `value + estimated error ≤ allowable`.
- The report gives every target, the margin on it and the error bound.
- Rejected alternatives for this step, which would give the same answer by other means:
  the three-mesh check, and SOLIDWORKS-style "converged" checks.

### Evolving a part: re-running with new targets

You can feed any part back in and optimize it again for new or changed targets. The part
can be one this generator produced, or one that exists already (OBJ, STL, or SLDPRT
through SOLIDWORKS).

- **Every run saves its design state.** That means the level set, the density field,
  the certified mesh, the study it was generated for, its results, and a link to its
  parent run.
- A new study names a `start` (a previous run, or a part file) and a **freedom**:
  - **`shape`**: the topology is kept; only stage 2 moves the boundary. Use this to tune
    a design you like.
  - **`topology`**: stage 1 reopens. The part becomes the starting density, and
    projection sharpness is reset so members can thin, vanish or appear. Holes can open
    where the new loads want them, using a topological derivative.
- **Limits on change.**
  - A maximum deviation from the parent.
  - Regions that must not change.
  - Mass change relative to the parent.

  These are ordinary targets, so they combine with everything else.
- **Inheritance.** By default the new study inherits the parent's targets and adds the
  new ones. A design therefore doesn't quietly lose a safety factor it already met. You
  can remove inherited targets explicitly.
- The lineage (parent → child, with each run's study) is recorded, and the viewport can
  replay it.

**Where I'd push back:**
1. **"Evolutionary" here means iterative refinement, not a genetic algorithm.**
   - A design has 10⁵–10⁷ variables.
   - A genetic algorithm needs thousands of full analyses per generation and still
     converges far worse than gradient methods at that size.
   - The gradient-based generator re-runs from the parent, and you watch it evolve
     iteration by iteration, as ProjectGoal item 3 asks.
2. **Starting from an existing part finds the best design near that part, not
   necessarily the best design overall.**
   - Gradient methods are local.
   - That is what you want when keeping a design, but not when the targets change
     drastically, for example adding a vibration limit the old shape can't meet.
   - So a run can also start from the design space with the same targets. The report
     then compares the two certified results side by side, and you choose.

### Why this combination, and what was rejected

| Choice | Rejected | Why |
|---|---|---|
| Density with robust projection for topology | Level set for topology (Allaire 2004; Wang 2003) | Level sets can't create holes on their own. Every 3-D stress- and frequency-constrained result at the state of the art is density-based (Sigmund & Maute 2013 review) |
| Body-fitted P2 shape stage | Stopping at the voxel field | Voxel boundaries give unreliable boundary stress, and stress at the boundary is where parts fail |
| Body-fitted P2 shape stage | An unfitted cut-cell method (CutFEM / AgFEM) in the loop | Elegant, but the research describes 3-D boundary stress, contact and plasticity on cut cells as research-grade |
| Augmented Lagrangian + MMA | p-norm stress aggregation; OC | Aggregation loses the location of the maximum and gives a poor estimate of it. OC handles one constraint |
| GPU geometric multigrid | Sparse direct factorisation (PARDISO and similar) | At 10⁶ DOFs a factor takes 4–10 GB and must be redone every iteration. Multigrid is O(n), warm-startable and fits 8 GB of GPU memory to about 10⁷ DOFs |
| AMG for unstructured tets | — | Multigrid needs a hierarchy, and body-fitted tets don't have one. Smoothed aggregation with rigid-body modes is the standard for elasticity |

**What is original here.** No published work combines all of the following in one
generator:
- robust density topology optimization;
- local augmented-Lagrangian constraints for stress, fatigue and harmonic stress;
- anisotropic failure criteria with build or fibre orientation;
- eigenfrequency separation;
- transient impact with discrete adjoints;
- a body-fitted P2 shape stage;
- error-bounded certification.

The research survey found the pieces separately. The combination, and making it work
reliably on CAD parts, is our contribution.

---

## 2. Load cases and how each enters the optimizer

Each case works as an analysis on its own, and as a target or limit in both stages.

| Case | Analysis | Sensitivity | Sources |
|---|---|---|---|
| **Static** (forces, torques, wrenches, pressure, bolt loads) | Linear elasticity | One adjoint per constraint group | — |
| **Modal** | Kφ = λMφ, lowest J ≈ 4–6 modes, by LOBPCG | Simple: φᵀ(∂K − λ∂M)φ. Repeated: N×N sub-eigenproblem (directional derivatives) | Bound formulation, Du & Olhoff 2007: maximise β s.t. λⱼ ≥ β. Mode tracking by MAC. Localised-mode fix, Pedersen 2000 |
| **Harmonic / oscillating load** (motor torque or force at a frequency or over a band) | Structural damping K(1 + iη) − ω²M. Tonal lines are direct complex solves; sweeps use a second-order Krylov reduced model | Complex adjoint per frequency | Li, Sigmund, Jensen & Aage 2021; stress-based harmonic optimization, Long, Wang & Liu 2018. The limit is the peak harmonic stress amplitude, plus the drive frequency's separation from the nearest natural frequency |
| **Fatigue** | Stress amplitude per tonal line (harmonic case) or rainflow (transient) → Basquin S-N with mean-stress correction → Palmgren–Miner damage | Chain rule through the harmonic or transient adjoint | Oest & Lund 2017; Holmberg, Torstenfelt & Klarbring 2014. Dirlik for broadband (Mršnik et al. 2013) |
| **Transient impact** (drop or strike, with contact) | Implicit HHT-α (α ≈ −0.05) over a short window, node-to-surface penalty contact, attached rigid masses | **Discrete adjoint** backwards in time, with checkpointing | Jensen, Nakshatrala & Tortorelli 2014. Rejected: equivalent static loads, which are provably not optimal in general (Stolpe et al. 2018) |
| **Nonlinear: large deformation** | Neo-Hookean finite strain, Newton | One adjoint at the converged state. Energy interpolation keeps void regions stable | Wang, Lazarov, Sigmund & Jensen 2014 |
| **Nonlinear: plasticity** | Finite-strain J2 with hardening, return mapping | **Analysis and certification only**, not inside the loop | Path-dependent adjoints (Wallin et al. 2016) cost a full history per iteration, and the loop already keeps stress under yield. Certification shows how much reserve plasticity gives |

**Damping.** Harmonic cases use a structural loss factor η, which is frequency-independent
(η ≈ 2ζ). Transient cases use Rayleigh damping fitted to ζ at ω₁ and 3ω₁. Defaults come
from the material entry with its source. For example, printed ABS has measured ζ = 1.6–2.5 %
(He, Ning & Khan 2023), so the default is η = 0.03.

---

## 3. Materials: any type

A material is data with a recorded source. It has four parts:

| Part | Options |
|---|---|
| Elastic symmetry | isotropic (metals, cast or moulded plastics), transversely isotropic (FDM layers, unidirectional composite), orthotropic (woven composite, printed parts with raster direction) |
| Strength criterion | von Mises (ductile metals), Tresca, max principal / Rankine (brittle), Tsai-Wu (FDM layers and composites, unequal tension and compression, weak across layers) |
| Fatigue | S-N curve (Basquin) or endurance limit, with a Goodman, Gerber or Walker mean-stress correction |
| Inelastic | none, or J2 plasticity with hardening (metals, ductile polymers) |

- **Density** and the **damping loss factor** are scalar properties of every material.
- **Orientation.** Every material that is not isotropic has an orientation field: a
  printed part's build direction, or a composite's fibre direction.
  - The orientation can be fixed by you, or optimized: 2 angles for print orientation,
    started from a few candidates because the problem is non-convex.
  - An optimized fibre-direction field for continuous-fibre composites is planned as a
    later extension.
- **Manufacturing constraints** are separate from the material, so the same material can
  be made by different processes.
  - Additive: an overhang limit (Langelaar 2016) along the build direction, and a minimum
    feature size.
  - Planned later: milling access and casting draw direction (Gersborg & Andreasen 2011).

**Verification.**
- An orthotropic plate against classical laminate theory.
- Tsai-Wu against published FDM coupon failure envelopes.
- J2 against an analytic thick-walled cylinder under internal pressure.

---

## 4. Targets: any number, any combination

A target has four parts:
- a **quantity**;
- a **relation**: `≤`, `≥`, `=`, `minimize` or `maximize`;
- a **value**, where the relation needs one;
- the **cases** and **regions** it applies to.

```json
"targets": [
  {"quantity": "mass_change", "relation": "<=", "value": "-30%"},
  {"quantity": "safety_factor", "relation": ">=", "value": 2.0, "cases": "all"},
  {"quantity": "first_natural_frequency", "relation": ">=", "value": "1.3 * drive_frequency"},
  {"quantity": "fatigue_life", "relation": ">=", "value": 1e7, "cases": ["arming oscillation"]},
  {"quantity": "displacement", "relation": "<=", "value": 0.5, "at": "motor_pad", "cases": ["lift"]},
  {"quantity": "mass", "relation": "minimize"}
]
```

**Quantities:**
- mass, mass change (absolute or %), volume;
- safety factor (under the material's criterion);
- stress, displacement at a region, stiffness or compliance;
- natural frequency (the nth mode, or a gap around a band), frequency separation from a
  drive frequency;
- harmonic stress amplitude, fatigue life or damage;
- peak transient stress, peak acceleration at a receiver, peak contact force.

**How targets are handled:**
- Targets with a value become constraints, all met together.
- With one minimize/maximize target, it is the objective.
- With several, the generator traces the **Pareto front** between them by an ε-constraint
  sweep: each point is a full optimization, warm-started from its neighbour. It reports
  the front so you choose the balance. It never adds the targets together with fixed
  weights.
- If there are no minimize/maximize targets, the default objective is minimum mass.

**Mesh-independent stress for limits.**
- Linear-elastic peak stress at a sharp re-entrant edge or a rigid fixture edge grows
  without bound as the mesh is refined. The necked-link run showed this happening.
- Stress limits are therefore evaluated on stress averaged over the material's critical
  distance L. This is the Theory of Critical Distances (Taylor 2007), which Susmel has
  extended to printed PLA.
- The robust length scale gives real fillets, so the averaging only matters at CAD
  features and fixtures.
- Fixtures can be modelled as elastic supports with a stiffness, not only as rigid ones.

---

## 5. Software architecture

```
apps/gso (CLI) ──┐                    ┌── future viewport
                 ▼                    ▼
           gso::api        Engine, RunObserver, CancellationToken, results
   ┌─────────┼──────────┬───────────────┐
gso::study  gso::generate  gso::certify  gso::cad
   │         (stage 1, stage 2,  │        │ SOLIDWORKS COM, OBJ/STL
   │          targets, AL+MMA)   │        │
   │            ▼                ▼        │
   │         gso::analysis  static, modal, harmonic, fatigue, transient, nonlinear
   │            ▼                         │
   │         gso::discretization  voxel-grid elements, P2 tets, material models
   │            ▼                         │
   │         gso::solvers  block-CG, LOBPCG, geometric MG (GPU), AMG
   │            ▼                         │
   └──────►  gso::geometry  mesh, solids, level sets, SDF, meshing ◄──┘
                ▼
             gso::core   errors, logging, progress, cancellation, hashing, units
```

**Rules.**
- Dependencies point one way only. Each module is its own static CMake target, so a
  module that reaches "up" fails to link.
- Third-party types never appear in public headers: each library sits behind one
  wrapper.
- All data stays in memory between stages. Files are written only for checkpoints and
  results.

**Style.**
- C++20, warnings as errors.
- Functions short (about 60 lines at most). Clear names.
- Comments that explain why.
- Preconditions checked in Release builds.
- No recursion in numerical kernels, and bounded iteration loops (the JPL rules, applied
  as in revision 1).
- **The existing `src/`, `include/` and DLLs are not carried over.** Their ideas (front
  transport, H¹ surface motion, Newmark with contact, Hencky/J2) inform the new code, but
  the new tree starts clean.

---

## 6. Libraries (one per problem, all pinned by hash, fetched by CMake)

| Problem | Library | Licence | Notes |
|---|---|---|---|
| Dense linear algebra, sparse containers | Eigen 5.0.1 | MPL-2.0 | Built with `EIGEN_MPL2_ONLY` |
| Mesh ↔ signed distance, narrow-band grids, surface extraction | OpenVDB 14.0 + oneTBB | Apache-2.0 | Replaces MeshLib, which is non-commercial |
| Solid booleans (generated shape ∪ preserved CAD, − keep-outs), 2-D sections | Manifold 3.5.5 + Clipper2 | Apache-2.0, BSL-1.0 | — |
| Surface → tetrahedra | fTetWild (pinned commit) + Geogram + libigl core | MPL-2.0, BSD-3, MPL-2.0 | GMP's role (exact rationals in `Rational.h`) is taken by Boost.Multiprecision `cpp_rational` (BSL-1.0), in a one-file patch we publish as MPL requires. The 10-node tets come from our own code: midside nodes snapped to the level set, with a Jacobian check |
| AMG preconditioner (P2 tets) | AMGCL 1.5.1 | MIT | Smoothed aggregation with rigid-body modes |
| GPU kernels (multigrid, element operators) | CUDA runtime | NVIDIA EULA, redistributable | NVIDIA GPUs only; see question 1 |
| Nearest-neighbour queries | nanoflann 2.0.2 | BSD-2 | — |
| Study files, CLI, logging, tests | nlohmann/json 3.12, CLI11 2.7.2, spdlog 1.17, doctest 2.5.3 | MIT, BSD-3, MIT, MIT | — |
| Optimizer | **Written here**: GCMMA following Svanberg 2002, with the Aage–Lazarov dual solver, plus the augmented-Lagrangian loop | — | About 500 lines. It needs to work with the progress and cancellation hooks. Checked against Svanberg's published test problems |

**Dropped:**
- MeshLib (non-commercial);
- TetGen (AGPL);
- oneMKL PARDISO (direct factorisation is rejected above);
- Spectra (LOBPCG is written here);
- trimesh, pymeshlab, VTK, shapely, scipy and numpy (their functions are covered above).

Every licence above allows a closed-source commercial product. Nothing is GPL or AGPL.

---

## 7. Study files, errors, logging, progress, resume

These carry over from revision 1. The changes are the general targets, material and
orientation blocks above, and that the volume block is gone.

- **Study file.** JSON with an integer `version`, strict reading (unknown keys are
  errors), and in-memory upgrade functions between versions.
- **Errors.** Three kinds:
  - bad input, which is reported with the path to the field;
  - broken invariants, from checks that are always on;
  - expected search outcomes (for example a rejected candidate), which are returned as
    values, not thrown.

  The API boundary catches everything and returns a `RunResult`.
- **Logging.** Through a `gso::Log` interface, with spdlog sinks for the console and a
  log file. Each run writes a `run.json` manifest: versions, study hash, seed and
  settings.
- **Progress.**
  - A `RunObserver` gets stage, progress and snapshot callbacks. Snapshots carry the
    density field or mesh, so a viewport can watch the part evolve.
  - A `CancellationToken` is checked at every iteration and between solves.
  - Checkpoints are written atomically after each iteration. `--resume` reruns the same
    study; `--continue-from` reruns with an edited one.

---

## 8. Build order and checks

Every check below compares against something independent: an analytic answer, a
benchmark, a finite-difference gradient, or (later) SOLIDWORKS Simulation. There are
few tests and no self-checking piles.

| # | Phase | Checked against |
|---|---|---|
| 0 | **Foundation**: repository layout, CMake presets, all dependencies building on MSVC from a fresh clone, `core`, study file v1, OBJ I/O, SOLIDWORKS import rewritten cleanly | builds from a clean clone; imports the arm on the workstation |
| 1 | **Static analysis on curved P2 tets**: meshing, materials (isotropic, orthotropic), fixtures and loads from CAD faces, AMG-CG, stress recovery, critical-distance stress, ZZ error estimate | cantilever tip deflection; NAFEMS LE1 (elliptic membrane) and LE10 (thick plate); Lamé cylinder; laminate theory. Then certify the Python's original and generated necked-link parts: the first like-for-like comparison of the old generator's results |
| 2 | **Dynamics and nonlinear analysis**: modal, harmonic, fatigue, transient with contact, neo-Hookean, J2 | Euler–Bernoulli cantilever (βₙL = 1.8751, 4.6941, 7.8548) and Timoshenko for a stubby beam; NAFEMS FV32 (44.623, 130.03, 162.70 Hz…); damped SDOF amplitude and phase; elastic bar impact wave speed; Hertz contact; thick cylinder plasticity |
| 3 | **Stage 1 topology**: GPU multigrid, block CG, LOBPCG, filters and robust projection, AL + GCMMA, the targets framework, the sensitivities of every case, overhang filter, orientation | central finite differences for every sensitivity (static, repeated eigenvalue, harmonic near resonance, transient adjoint); a cantilever MBB beam against published results; the coupon and necked-link studies, with results certified by phase 1 |
| 4 | **Stage 2 shape**: level set from density, P2 remeshing per step, shape derivatives | finite differences of the shape derivatives; the boundary-stress improvement over stage 1 on necked-link |
| 5 | **Pareto front, certification report, resume** | an ε-constraint front on a two-objective cantilever |
| 6 | **Sunbeam study** (needs SOLIDWORKS, so on the workstation): static, motor-torque oscillation, braking, drop, strike, with your numbers | the original arm's failure reproduced as a violated fatigue or frequency limit; later, SOLIDWORKS Simulation |
| 7 | **Delete the Python**: `tools/`, the Python studies and tests, `requirements*.txt`, the Python parts of CMake, and `reference/` once phase 3 matches it | the C++ handles every study the Python did |

The Python reference data captured on 2026-11-25 (`reference/coupon`,
`reference/necked-link`) serves phase 1, which certifies the old parts, and phases 3–4,
where the new parts must be better under the same certification.

---

## 9. Results so far (unchanged from revision 1)

- **Build.** MSVC builds cleanly and ctest passes 3/3.
- **Reference runs.**
  - `design_inputs.py` passes.
  - necked-link completes. At equal volume the generated part is 21 % stiffer in tension
    and 57 % stiffer in the oblique case. It still fails its tension safety factor (1.91,
    against 2 required), because peak stress keeps rising with mesh refinement
    (utilisation 0.85 → 1.05). Section 4's critical-distance stress addresses this.
- **Not run.** Sunbeam, because there is no SOLIDWORKS on the laptop.

---

## 10. Open questions

1. **GPU.**
   - Stage 1's multigrid runs on NVIDIA GPUs through CUDA. This laptop has an RTX 5060
     Laptop GPU with 8 GB. Building needs the CUDA Toolkit (about 3 GB) installed here.
   - A machine without an NVIDIA GPU could not run stage 1.
   - Is NVIDIA-only acceptable? What GPU does your workstation have?
2. **Your numbers for the Sunbeam study:**
   - maximum thrust per motor;
   - motor model, mass and KV, and battery cell count;
   - prop diameter, pitch, mass and blade count;
   - frame mass;
   - material, print orientation, layer height, perimeters and infill;
   - the arming oscillation frequency, if known.
3. **GitHub repository URL**, for the branch.
