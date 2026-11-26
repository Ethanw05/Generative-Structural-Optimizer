# GSO study file format, version 1

A study is a JSON file the `gso` executable reads. It defines the part, the faces you
pick on it, where material may go, the materials, the load cases and the targets.
Working examples are in `studies/coupon/study.json` and `studies/necked-link/study.json`.
Check a file with `gso check study.json`.

**Strict reading.** Every key must be one GSO knows. A missing key, a key it doesn't
recognise, a wrong type, a reference to an undefined name or a physically impossible value
(for example a Poisson ratio of 0.6) stops with an error naming the field. A typo can
therefore never silently change a study. Relative paths are relative to the study file.

**Units are fixed:** mm, N, s, MPa, kg. Densities are in kg/m³.

## Top level

| Key | Required | Meaning |
|---|---|---|
| `format` | yes | Always `"gso-study"` |
| `version` | yes | Format version, currently `1`. Older versions are upgraded when read |
| `units` | yes | `{"length": "mm", "force": "N", "time": "s", "stress": "MPa", "mass": "kg"}`. Stated so every file is self-describing |
| `part` | yes | `{"source": "part.obj"}`, which may be OBJ, SLDPRT or SLDASM. An optional `"sha256"` of the file pins the exact part |
| `frame` | no | Local axes for loads: `{"origin": [x,y,z], "x_axis": [...], "up_axis": [...]}`. The two axes must be perpendicular; local y = up × x |
| `selections` | yes | Named faces (below) |
| `design_space` | yes | Where material may go (below) |
| `materials` | yes | Named materials (below) |
| `cases` | yes | Load cases (below) |
| `targets` | yes | What to achieve (below) |
| `start` | no | What generation starts from (below). Defaults to the design space with full topology freedom |

## Selections

```json
"selections": {
  "motor_pad": {"planar_face": {"point": [-155.37, 0, 165.89], "normal": [0, 1, 0], "tolerance_mm": 0.01}}
}
```

A planar face is the flat region of the part surface that contains `point` and whose
outward normal is `normal`. `tolerance_mm` is optional and defaults to 0.01.

## Design space

```json
"design_space": {
  "envelope": "bounding_box",
  "preserve": [{"selection": "motor_pad", "depth_mm": 1.0}],
  "exclude":  [{"solid": "propeller-clearance.obj"}]
}
```

- **`envelope`** is one of:
  - `"bounding_box"`: the part's bounding box.
  - `"part"`: the part itself, so material can only be removed.
  - `{"box": {"min_mm": [...], "max_mm": [...]}}`: an explicit box.
- **`preserve`** keeps a layer of material `depth_mm` thick behind a face, for fixtures,
  bolt pads and other interfaces.
- **`exclude`** lists closed solids that material may not enter.

## Materials

```json
"PLA (printed)": {
  "density_kg_m3": 1240,
  "elasticity": {"transversely_isotropic": {"in_plane_young_mpa": 2300, "in_plane_poisson": 0.35,
                 "axial_young_mpa": 1900, "in_plane_to_axial_poisson": 0.3, "axial_shear_mpa": 700}},
  "strength": {"tsai_wu": {"xt_mpa": 51, "xc_mpa": 60, "yt_mpa": 51, "yc_mpa": 60, "zt_mpa": 30,
               "zc_mpa": 55, "s12_mpa": 30, "s13_mpa": 20, "s23_mpa": 20}},
  "fatigue": {"basquin": {"coefficient_mpa": 80, "exponent": -0.19,
              "mean_stress_correction": "goodman", "ultimate_mpa": 55}},
  "damping_loss_factor": 0.02,
  "source": "Where these numbers come from"
}
```

(The values in this example are illustrative, not measured.)

| Key | Options |
|---|---|
| `elasticity` (required) | `isotropic` {`young_mpa`, `poisson`}; `transversely_isotropic` about axis 3 {`in_plane_young_mpa`, `in_plane_poisson`, `axial_young_mpa`, `in_plane_to_axial_poisson`, `axial_shear_mpa`}; `orthotropic` {`e1_mpa`, `e2_mpa`, `e3_mpa`, `nu12`, `nu13`, `nu23`, `g12_mpa`, `g13_mpa`, `g23_mpa`}. Non-isotropic constants must give a positive-definite compliance matrix |
| `strength` | `von_mises` {`yield_mpa`} for ductile metals and polymers; `tresca` {`yield_mpa`}; `max_principal` {`tensile_mpa`, `compressive_mpa`} for brittle materials; `tsai_wu` {9 strengths along the material axes} for composites and printed parts |
| `fatigue` | `basquin` {`coefficient_mpa`, `exponent` < 0, `mean_stress_correction`: `none`/`goodman`/`gerber`, `ultimate_mpa` (required by Goodman and Gerber)} |
| `plasticity` | `j2` {`yield_mpa`, `hardening_mpa`} |
| `damping_loss_factor` | Structural loss factor η, about twice the damping ratio |
| `source` (required) | Where the values come from |

## Load cases

```json
{"name": "lift", "type": "static", "material": "PLA (printed)",
 "fixtures": ["hub_top", "hub_bottom"],
 "loads": [{"on": "motor_pad", "force_n": [0, 33, 0], "frame": "local"},
           {"on": "motor_bolts", "torque_nmm": [0, 0, 1000]}]}
```

- A load has either `force_n` or `torque_nmm`: a resultant spread over the selection, with
  torques taken about its centroid.
- `frame` is `"global"` (the default) or `"local"`, which needs the study's `frame`.
- Case names must be unique.
- Version 1 has `static` cases only. `modal`, `harmonic`, `transient` and
  `nonlinear_static` arrive with their analyses and are rejected until then.

## Targets

```json
"targets": [
  {"quantity": "mass_change", "relation": "<=", "value": "-30%"},
  {"quantity": "safety_factor", "relation": ">=", "value": 2.0},
  {"quantity": "displacement", "relation": "<=", "value": 0.5, "at": "motor_pad", "cases": ["lift"]},
  {"quantity": "mass", "relation": "minimize"}
]
```

- **`relation`** is `<=`, `>=`, `=`, `minimize` or `maximize`.
  - The first three need a `value` and become constraints.
  - `minimize` and `maximize` take no value and become objectives. With several
    objectives the generator reports the trade-off between them (the Pareto front).
- **Quantities that belong to the part:** `mass`, `mass_change` and `volume`.
  - `mass_change` accepts a percentage of the starting part, such as `"-30%"`.
- **Quantities measured per case:**
  - `safety_factor` (under the material's strength criterion);
  - `stress`;
  - `displacement`, which needs `at`;
  - `compliance`.

  `cases` restricts one of these to named cases; it defaults to all.
- **Dynamic quantities** are accepted by the schema but need case types that aren't
  available yet: `natural_frequency`, `frequency_separation`, `harmonic_stress`,
  `fatigue_life`, `peak_transient_stress`, `peak_acceleration`, `peak_contact_force`.

## Start

```json
"start": {"from": "previous_run", "run": "../runs/arm-v1", "freedom": "shape"}
```

- **`from`** is `design_space` (the default), `part` or `previous_run`.
- **`freedom`** is `topology` (the default: members can appear and vanish) or `shape`
  (only the boundary moves).
- Shape freedom needs an existing part to start from.
- `mass_change` targets need a starting part.
