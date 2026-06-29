# Analysis

Analysis objects sample the system at a given frequency during a simulation
and optionally write results to output files.
They are defined in the `analysis` section of the YAML input:

```yaml
analysis:
  - !Energy
    file: energy.csv.gz
    frequency: !Every 100
  - !RadialDistribution
    selections: ["atomtype Na", "atomtype Cl"]
    file: rdf.dat
    dr: 0.1
    frequency: !End
```

### Sampling frequency

Each analysis requires a `frequency` field controlling when it is evaluated.

Frequency         | Description
----------------- | -------------------------------------------
`!Every N`        | Every N steps
`!Once N`         | Once at step N
`!End`            | Once after the last step
`!Probability P`  | Each step with probability P (0–1)

### Output file formats

Most analysis files are column data. For those files, the format is inferred
from the file extension:

Extension     | Format
------------- | -------------------------------------------
`.dat`        | Space-separated; header prefixed with `# `
`.csv`/`.tsv` | Comma/Tab-separated; plain header row (column labels)

Both formats support transparent gzip compression by appending `.gz`
(e.g. `energy.dat.gz`).

Grid-based analyses document their own file format.

Loading CSV output in Python:

```python
import numpy as np
data = np.loadtxt("data.csv.gz", delimiter=",", skiprows=1)

import pandas as pd
data = pd.read_csv("data.csv.gz").to_numpy()
```

---

## Collective Variable

Monitors a collective variable (CV) over the course of a simulation,
recording the instantaneous value and a running average at each sampled step.
This is useful for verifying that a CV remains within the expected range,
for convergence checking, and for post-processing of time/step series data.

If `file` is given, each sampled step writes a line with columns
`step`, `value`, and `running_average`.
If no file is given, only the mean and RMS are written to `output.yaml`.

### Example

```yaml
analysis:
  - !CollectiveVariable
    property: mass_center_position
    selection: "molecule protein"
    projection: z
    file: cv.dat
    frequency: !Every 100
```

### Options

Key          | Required | Default | Description
------------ | -------- | ------- | -------------------------------------------
`property`   | yes      |         | CV type (see table below)
`frequency`  | yes      |         | Sample frequency, e.g. `!Every 100`
`projection` | no       | `xyz`   | Axis projection (`x`, `y`, `z`, `xy`, …); alias: `dimension`
`selection`  | depends  |         | Selection expression for one atom or group
`selection2` | depends  |         | Second selection (for two-group properties)
`resolution` | no       |         | Bin width (only used by Penalty)
`file`       | no       |         | Output file path (see [Output file formats](#output-file-formats)); omit to only track the mean

### Supported properties

Property                 | Selection       | Description
------------------------ | --------------- | -------------------------------------------
`volume`                 | none            | Cell measure via `dimension`: volume (`xyz`), area (`xy`), or length (`z`). Note: `volume` uses `dimension`, not `projection`
`atom_position`          | one atom        | Signed component for single axis (`x`,`y`,`z`); Euclidean norm for multi-axis (`xy` etc.). Selection resolved live each evaluation — works with speciation/GCMC where the matching atom changes. Returns NaN if selection matches ≠ 1 atom. Use `atomtype` (not `name`) for atoms defined via explicit `atoms:` lists
`count`                  | atoms or groups | Molecule instances for Molecular groups; atom count for Atomic/Reservoir groups
`molarity`               | atoms or groups | Molar concentration (mol/L); molecule-based for Molecular groups, atom-based for Atomic/Reservoir
`charge`                 | atoms or groups | Sum of charges of active atoms matching selection
`size`                   | one group       | Number of active particles in a group
`end_to_end`             | one group       | End-to-end distance of a molecular group
`gyration_radius`        | one group       | Radius of gyration (default `xyz` = full Rg; single axis gives spread along it)
`dipole_moment`          | one group       | Electric dipole moment (default `xyz` = magnitude; single axis gives signed component)
`mass_center_position`   | one group       | Mass center position along `projection`
`mass_center_separation` | two groups      | Distance between two group mass centers
`dipole_product`         | two groups      | Normalized dipole dot product μ̂₁·μ̂₂ = cos(θ) (default `xyz` = full 3D; `projection` filters dipoles before comparing)

---

## Polymer Shape

Computes size and shape anisotropy descriptors from the mass-weighted gyration tensor
([doi:10/d6ff](https://doi.org/10/d6ff)).
For each matching group the 3×3 symmetric gyration tensor is built as
`S_ij = (1/M) Σ m_k r_i r_j`, where `r` is the PBC-aware displacement from the
centre of mass. Eigenvalue decomposition yields the principal moments
`λ₁ ≤ λ₂ ≤ λ₃` from which the following descriptors are derived.

Following [IUPAC 2014](https://doi.org/10.1515/pac-2013-0201) nomenclature,
the per-configuration squared radius of gyration is `s² = Tr(S)` (Def. 1.14),
and the reported `Rg = √⟨s²⟩` is the root-mean-square radius of gyration
(Def. 1.16). Note that `⟨s⟩ ≠ √⟨s²⟩` in general.
The reported ratio `⟨Re²⟩/⟨Rg²⟩` equals 6 for ideal Gaussian chains
([Wall & Erpenbeck 1959](https://doi.org/10.1063/1.1730022)).

| Descriptor                    | Formula                                                 | Reference |
| ----------------------------- | ------------------------------------------------------- | --------- |
| Radius of gyration squared    | `Rg² = λ₁ + λ₂ + λ₃`                                   | [IUPAC 2014](https://doi.org/10.1515/pac-2013-0201) Def. 1.14 |
| Asphericity                   | `b = λ₃ − (λ₁ + λ₂)/2`                                 | [Aronovitz & Nelson 1986](https://doi.org/10.1051/jphys:019860047090144500) |
| Acylindricity                 | `c = λ₂ − λ₁`                                           | [Aronovitz & Nelson 1986](https://doi.org/10.1051/jphys:019860047090144500) |
| Relative shape anisotropy     | `κ² = (b² + ¾c²) / Rg⁴`  ∈ [0, 1]                      | [Rudnick & Gaspari 1986](https://doi.org/10.1088/0305-4470/19/4/004) |
| Prolateness                   | `S = 27(λ₁−λ̄)(λ₂−λ̄)(λ₃−λ̄) / Rg⁶`  ∈ [−0.25, 2]      | [Theodorou & Suter 1985](https://doi.org/10.1021/ma00148a028) |
| Westin linear (rod-like)      | `Cl = (λ₃ − λ₂) / Rg²`                                 | [Westin 2002](https://doi.org/10.1016/S1361-8415(02)00053-1) |
| Westin planar (disc-like)     | `Cp = 2(λ₂ − λ₁) / Rg²`                                | [Westin 2002](https://doi.org/10.1016/S1361-8415(02)00053-1) |
| Westin spherical              | `Cs = 3λ₁ / Rg²`  (Cl + Cp + Cs = 1)                   | [Westin 2002](https://doi.org/10.1016/S1361-8415(02)00053-1) |

Mean gyration tensor components (`Sxx`, `Sxy`, `Sxz`, `Syy`, `Syz`, `Szz`) are
reported in the YAML output. The ratio `⟨Szz⟩ / ⟨Rg²⟩` indicates orientation
relative to the z-axis: ~0 for flat objects lying in the xy-plane, ~1/3 for
isotropic, ~1 for rod-like objects aligned with z.

If `file` is given (single-molecule selection only), each sampled step writes a line with
columns `step`, `Rg`, and the upper triangle of the gyration tensor
(`Sxx Sxy Sxz Syy Syz Szz`).

### Example

```yaml
analysis:
  - !PolymerShape
    selection: "molecule polymer"
    file: shape.csv.gz
    frequency: !Every 100
```

### Options

Key          | Required | Default | Description
------------ | -------- | ------- | -------------------------------------------
`selection`  | yes      |         | Selection expression for molecule group(s)
`file`       | no       |         | Streaming output file, single molecule only (see [Output file formats](#output-file-formats))
`frequency`  | yes      |         | Sample frequency, e.g. `!Every 100`

---

## Multipole

Per-group charge and dipole moment analysis, averaged over all groups
matching a selection. Useful for tracking titration state and charge
fluctuations.

Reports:
- **⟨Z⟩ ± σ** — mean net charge and standard deviation
- **capacitance** C = ⟨Z²⟩ − ⟨Z⟩² — charge variance
- **⟨|μ|⟩ ± σ** — mean dipole moment magnitude (eÅ)
- **per-atom ⟨q⟩ and ⟨q²⟩−⟨q⟩²** — for atoms with fluctuating charge (e.g. from titration or atom swaps)

The dipole moment is computed relative to each group's center of mass
with periodic boundary conditions applied.
Handles atom-type swaps (titration) and GCMC (only active groups contribute).

### Selection resolves group-wise

The selection picks **whole groups** that contain at least one active
matching atom — not just the matched atoms themselves. So an atom-level
selection like `atomtype CA` pulls in every group that has an active CA
atom and accumulates the full group's charge and dipole moment. This is
the same `resolve_groups_live` behavior used elsewhere in faunus, and it
makes it easy to address molecules whose name varies (e.g. across
protonation states) via a stable marker atom that they all share:

```yaml
- !Multipole
  selection: "atomtype INO"   # picks every phytate, regardless of protonation state
  frequency: !Every 10
```

By contrast, `!CollectiveVariable property: charge` sums the charge of
**only the matched atoms**, not their parent groups — see [Supported
properties](#supported-properties).

Per-atom charge statistics are reported in `output.yaml` as an `atoms` list
only when the selection resolves to a single molecular molecule kind.
Selections matching multiple molecule kinds are rejected, and atomic/reservoir
groups do not emit per-atom output. The list includes only atoms whose charge
variance is nonzero:

```yaml
multipole:
  selection: molecule MOL1
  num_samples: 2000
  charge: '4.7785 ± 2.2373'
  capacitance: 5.005
  dipole_moment: '141.6532 ± 23.8816'
  atoms:
    - {index: 1, name: NP, ⟨q⟩: -0.52, ⟨q²⟩-⟨q⟩²: 0.2496}
    - {index: 6, name: NP, ⟨q⟩: -0.48, ⟨q²⟩-⟨q⟩²: 0.2496}
```

### Example

```yaml
analysis:
  - !Multipole
    selection: "molecule peptide"
    frequency: !Every 10
```

### Options

Key          | Required | Default | Description
------------ | -------- | ------- | -------------------------------------------
`selection`  | yes      |         | Selection expression for molecule group(s)
`frequency`  | yes      |         | Sample frequency

---

## Energy

Streams energy values to a file at each sampled step.
Two modes are supported:

- **Total** (default): writes every Hamiltonian term plus the total.
  Output columns: `step term1 term2 ... total`.
- **Partial**: writes the nonbonded energy between two sets of atoms
  selected with VMD-like expressions.
  Output columns: `step energy running_average`.

### Examples

```yaml
analysis:
  # Total energy with per-term breakdown
  - !Energy
    file: energy.csv.gz
    frequency: !Every 100

  # Nonbonded energy between two molecules
  - !Energy
    file: mol1_mol2_energy.csv.gz
    frequency: !Every 100
    selections: ["molecule MOL1", "molecule MOL2"]

  # Nonbonded energy between hydrophobic atoms in two molecules
  - !Energy
    file: hydrophobic_energy.csv.gz
    frequency: !Every 100
    selections: ["hydrophobic and molecule MOL1", "hydrophobic and molecule MOL2"]
```

### Options

Key            | Required | Default | Description
-------------- | -------- | ------- | -------------------------------------------
`file`         | yes      |         | Output file path (see [Output file formats](#output-file-formats))
`frequency`    | yes      |         | Sample frequency, e.g. `!Every 100`
`selections`   | no       |         | Pair of selection expressions for partial nonbonded energy

When `selections` is omitted, total mode is used.
When given, only nonbonded energy terms contribute; other terms
(bonded, constraints, etc.) are skipped.
Selections resolve at the atom level, so atom-type filters
(e.g. `hydrophobic and molecule MOL1`) work correctly.
When both selections resolve to the same atoms, self-pairs and
duplicates are automatically excluded.

---

## Trajectory

Writes particle coordinates to a trajectory file at each sampled step.
The format is auto-detected from the file extension (`.xyz` or `.xtc`).

On finalization a companion PSF topology file and a VMD scene script are
written alongside the trajectory. The PSF contains atoms, bonds, angles,
and dihedrals; the Tcl script loads the PSF and trajectory, sets VDW radii
from sigma values, colors by charge, and draws the periodic box.

For speciation/GCMC/titration simulations, two additional companion files
enable per-frame visualization updates in VMD:
- **`.sizes.dat`** — per-frame group active counts (written when any group
  has inactive atoms). The VMD script hides inactive atoms (radius = 0).
- **`.charges.dat`** — per-frame atom charges (always written). The VMD
  script updates charges each frame so that atom-type swaps from titration
  and speciation are reflected in the charge coloring.

### Example

```yaml
analysis:
  - !Trajectory
    file: traj.xtc
    frequency: !Every 100
    save_frame_state: true
  - !Trajectory
    file: protein.xyz
    frequency: !Every 100
    selection: "protein"
```

After the simulation, visualize with:

```sh
vmd -e traj.tcl
```

### Options

Key                | Required | Default | Description
------------------ | -------- | ------- | -------------------------------------------
`file`             | yes      |         | Output file path (`.xyz`, `.xtc`, etc.)
`frequency`        | yes      |         | Sample frequency, e.g. `!Every 100` or `!End`
`selection`        | no       |         | VMD-like molecule selection, e.g. `"molecule protein"`. Writes only matching groups.
`save_frame_state` | no       | `false` | Write a binary `.aux` file alongside the trajectory (see [Rerun](#rerun)). Cannot be combined with `selection`.

### Output files

Given `file: traj.xtc`, the following files are produced:

File               | Description
------------------ | -------------------------------------------
`traj.xtc`         | Trajectory (coordinates per frame)
`traj.psf`         | X-PLOR PSF topology (atoms, bonds, angles, dihedrals, charges, masses)
`traj.tcl`         | VMD scene script (`vmd -e traj.tcl` loads everything)
`traj.charges.dat` | Per-frame atom charges (always written)
`traj.sizes.dat`   | Per-frame group active counts (only when groups have inactive atoms)
`traj.aux`         | Frame state file (only when `save_frame_state: true`)

### Frame state file (`.aux`)

XTC stores only atom positions. Rigid-body simulations also need
quaternions, mass centers, and group sizes (for GC moves); swap moves change
`atom_id`. When `save_frame_state: true`, a binary `.aux` file is written
alongside each XTC frame, storing this per-frame microstate data.

The `.aux` file is required by `faunus rerun` (see [Rerun](#rerun)) to fully
reconstruct the simulation state from each trajectory frame.

---

## Radial Distribution Function

Computes the radial distribution function g(r) for pairs of particles or
molecule centers of mass.
Distances use the minimum image convention for periodic boundary conditions.
The output file is updated at each sample with columns `r` and `g(r)`,
normalized using the average volume (NPT-compatible).

Two modes are supported:

- **Atom-atom** (default): pairwise distances between individual atoms.
  Intramolecular pairs (atoms in the same molecule) are excluded by default.
- **COM-COM** (`use_com: true`): pairwise distances between molecular
  centers of mass.

The `dimension` option controls which spatial components are used for the
pair distance and shell normalization:

- `xyz` (default): full 3D distance, normalized with spherical shells (4πr²).
- `xy`, `xz`, `yz`: projected distance in a plane, normalized with cylindrical shells (2πr).
- `x`, `y`, `z`: projected distance along an axis, normalized with linear shells.

This is useful for systems where particles are constrained to a plane or a line.

When both selections are identical, self-pairs and duplicates are
automatically excluded.

### Examples

```yaml
analysis:
  # Atom-atom RDF between Na and Cl
  - !RadialDistribution
    selections: ["atomtype Na", "atomtype Cl"]
    file: rdf_nacl.dat
    dr: 0.1
    frequency: !Every 100

  # COM-COM RDF for polymer molecules
  - !RadialDistribution
    selections: ["molecule polymer", "molecule polymer"]
    use_com: true
    file: rdf_com.dat
    dr: 0.5
    max_r: 30.0
    frequency: !Every 100
```

### Options

Key                        | Required | Default               | Description
-------------------------- | -------- | --------------------- | -------------------------------------------
`selections`               | yes      |                       | Pair of selection expressions, e.g. `["atomtype Na", "atomtype Cl"]`
`file`                     | yes      |                       | Output file path (see [Output file formats](#output-file-formats))
`dr`                       | yes      |                       | Bin width in distance units
`frequency`                | yes      |                       | Sample frequency, e.g. `!Every 100`
`max_r`                    | no       | half shortest box dim | Maximum distance for histogram
`use_com`                  | no       | `false`               | Use center-of-mass distances instead of atom-atom
`exclude_intramolecular`   | no       | `true` (atom-atom)    | Skip pairs within the same molecule (atom-atom only)
`dimension`                | no       | `xyz`                 | Axes for distance projection and normalization (`x`, `y`, `z`, `xy`, …)

---

## Spatial Distribution Function

Computes the spatial distribution function (SDF) of an atom selection around a
rigid molecular reference selection. The result is a scalar grid in the
reference molecule body frame, averaged over all matching reference molecules
and all sampled frames.

This is useful for mapping ions or solvent around rigid macromolecules in a
periodic box. Target positions are measured from each reference center using
minimum-image periodic displacements and then rotated by the inverse reference
quaternion before binning.

The reference selection must match rigid molecular groups. Flexible reference
molecules are not supported by this analysis.

### Example

```yaml
analysis:
  - !SpatialDistribution
    reference: "molecule Macro"
    selection: "atomtype Na"
    frequency: !Every 100
```

### Options

Key                 | Required | Default      | Description
------------------- | -------- | ------------ | -------------------------------------------
`reference`         | yes      |              | Molecular group selection defining the body-fixed frame
`selection`         | yes      |              | Atom selection accumulated on the grid
`frequency`         | yes      |              | Sample frequency, e.g. `!Every 100`
`file`              | no       | `spatial.dx` | OpenDX output grid
`resolution`        | no       | `1.0`        | Cubic grid spacing in Å
`padding`           | no       | `8.0`        | Extra grid extent around the reference molecule in Å
`bulk_normalize`    | no       | `true`       | Normalize by bulk density to produce dimensionless relative density
`exclude_reference` | no       | `true`       | Skip target atoms belonging to the current reference group

The grid bounds are determined from the body-frame coordinates of the active
reference molecule(s), rounded to the grid spacing and expanded by `padding`.
The output is written once at the end of the run.

### Normalization

With the default `bulk_normalize: true`, grid values are normalized by the
instantaneous bulk density of the target selection. A homogeneous ideal gas
therefore gives SDF ≈ 1. The normalization uses the instantaneous target count,
reference count, and cell volume at each sample, so it works with GCMC where
particle numbers fluctuate.

Set `bulk_normalize: false` to write molar concentration in mol/L instead of
relative density.

### Output

The output file is an OpenDX scalar grid suitable for visualization in VMD and
PyMOL. Grid coordinates are in the reference body frame, not the lab frame.

---

## Widom Insertion

Measures the excess chemical potential of a single ion species using the scaled
Widom method ([Svensson & Woodward, 1988](https://doi.org/10.1080/00268978800100203)).
A ghost particle is inserted at random positions; charge scaling maintains
electroneutrality in the finite periodic box, correcting the Coulombic size error
inherent in naive single-ion Widom insertion.

The excess chemical potential is decomposed into short-range and electrostatic
contributions. The electrostatic part is evaluated by numerical integration over
a charging parameter λ ∈ [0, 1]. Results are block-averaged with standard error
of the mean reported.

Pair interactions for the ghost particle are defined directly in the analysis
block using the same syntax as `energy.nonbonded.default`, allowing arbitrary
short-range potentials (WCA, LJ, HardSphere, etc.) beyond the primitive model.

### Example

```yaml
analysis:
  - !ScaledWidomInsertion
    atom: Na
    insertions: 20
    lambda_points: 11
    frequency: !Every 10
    default:
      - !Coulomb { cutoff: 1000.0 }
      - !WCA { mixing: arithmetic }
```

### Options

Key              | Required | Default | Description
---------------- | -------- | ------- | -------------------------------------------
`atom`           | yes      |         | Ghost atom type (must exist in topology)
`insertions`     | no       | `10`    | Number of ghost insertions per sample
`lambda_points`  | no       | `11`    | Quadrature points for charge scaling (odd recommended)
`default`        | yes      |         | Pair interactions (same syntax as `nonbonded.default`)
`frequency`      | yes      |         | Sample frequency, e.g. `!Every 10`

---

## Virtual Translate

Performs a virtual displacement of a single molecule and measures the
mean force by Widom perturbation
([Widom, 1963](https://doi.org/10.1063/1.1734110)):

$$f = \frac{k_BT \ln\langle e^{-\Delta U / k_BT}\rangle}{\delta L}$$

where $\Delta U$ is the energy change due to the displacement $\delta L$.
If `file` is given, each sampled step writes columns
`step`, `dL`, `dU/kT`, and `<force>/kT/Å`.

### Example

```yaml
analysis:
  - !VirtualTranslate
    selection: "molecule protein"
    dL: 0.01
    directions: !z
    file: force.dat
    frequency: !Every 10
```

### Options

Key           | Required | Default  | Description
------------- | -------- | -------- | -------------------------------------------
`selection`   | yes      |          | Selection matching exactly one molecule
`dL`          | yes      |          | Displacement magnitude (Å)
`directions`  | no       | `z`      | Displacement direction (`x`, `y`, `z`, `xy`, …)
`file`        | no       |          | Output file path (see [Output file formats](#output-file-formats))
`frequency`   | yes      |          | Sample frequency, e.g. `!Every 10`

---

## Virtual Volume Move

Performs a virtual volume perturbation and measures the excess pressure
by Widom perturbation ([doi:10.1063/1.472721](https://doi.org/10.1063/1.472721)):

$$P_{\text{ex}} = \frac{k_BT \ln\langle e^{-\Delta U / k_BT}\rangle}{\delta V}$$

where $\Delta U$ is the energy change due to the volume displacement $\delta V$.
All particle positions are scaled according to the chosen `method`.
If `file` is given, each sampled step writes columns
`step`, `dV/Å³`, `dU/kT`, and `<Pex>/kT/Å³`.

### Example

```yaml
analysis:
  - !VirtualVolumeMove
    dV: 0.2
    method: Isotropic
    file: pressure.csv
    frequency: !Every 10
```

### Options

Key           | Required | Default      | Description
------------- | -------- | ------------ | -------------------------------------------
`dV`          | yes      |              | Volume displacement (ų)
`method`      | no       | `Isotropic`  | Scaling policy: `Isotropic`, `ScaleZ`, `ScaleXY`
`file`        | no       |              | Output file path (see [Output file formats](#output-file-formats))
`frequency`   | yes      |              | Sample frequency, e.g. `!Every 10`

---

## Double Layer Pressure

Osmotic (disjoining) pressure between two uniformly charged planes with explicit
point-charge counterions, by the midplane method of Guldbrand et al.
([doi:10.1063/1.446912](https://doi.org/10.1063/1.446912)):

$$P_\text{osm} = k_BT\sum_i C_i(0) \;+\; \frac{F_z^{AB}}{\text{area}}$$

The first term is the entropic (repulsive) contribution from the ion concentration at the
midplane; the second is the cross-midplane electrostatic force. Its ion–ion correlation part
is the attractive, van-der-Waals-like force that dominates for divalent ions and that
mean-field Poisson–Boltzmann misses. The surface charge density $\sigma$ is set by
electroneutrality from the counterion charges — you do not specify it.

The analysis is **hardcoded to a `Slit` cell** (periodic in *xy*, walls at $z=\pm L_z/2$, so
the **midplane is $z=0$**) and to **electrostatics only**; it suits the primitive-model
double layer with point-charge counterions. Valency **mixtures need nothing special** —
select all the counterions (e.g. `"atomtype Na Ca"`) and report. Results are block-averaged
and reported as mean ± standard error.

### Example

```yaml
analysis:
  - !DoubleLayerPressure
    selection: "atomtype Na Ca"   # mono- + divalent counterions
    file: pressure.csv
    frequency: !Every 10
```

### Options

Key                  | Required | Default  | Description
-------------------- | -------- | -------- | -------------------------------------------
`selection`          | yes      |          | Mobile counterions; sets the electroneutral surface charge $\sigma$ (the Coulomb prefactor comes from the medium)
`midplane_halfwidth` | no       | `gap/12` | Half-width (Å) of the midplane density sampling window. The default scales with the plate separation and is usually fine; treat it as a convergence parameter
`density_bins`       | no       | `50`     | Resolution of the internal charge profile used by the long-range correction; increase and re-check if results drift
`file`               | no       |          | Output file path (see [Output file formats](#output-file-formats))
`frequency`          | yes      |          | Sample frequency, e.g. `!Every 10`

A **long-range correction** for the finite lateral box, reported as `F_iPB`, is computed and
added to the pressure automatically — it needs no configuration and handles valency
mixtures. For reliable numbers, **equilibrate first and start production from a state file**
(`-s`), and confirm the result is stable against `density_bins` and the lateral box size.

`output.yaml` reports `{mean, error}` for the midplane density `rho_mid/Å⁻³` and the
pressures `p_ideal/mM` (entropic), `p_corr/mM` (electrostatic), and `p_osm/mM` (total — the
number you usually want), plus `F_iPB/mM` and `p_osm/Pa`; the `p_corr` and `p_osm` means
include `F_iPB`. With `file` set, each sampled step writes `step`, `rho_mid/Å⁻³`,
`p_ideal/mM`, `p_corr/mM`, `p_osm/mM` — there the per-step `p_corr`/`p_osm` columns exclude
`F_iPB` (a run-averaged constant folded into the final means only).

---

## Electric Potential Profile

Mean electric potential $\varphi(z)$ along the $z$-axis of a slab, for an electrolyte with
implicit salt (Debye–Hückel / Yukawa screening). A uniformly charged plane in a screened
electrolyte produces a potential that decays exponentially away from it, so the
laterally-averaged potential is a screened sum over the charge in each thin slab:

$$\varphi(z) = \frac{2\pi\,l_B}{\kappa}\sum_{z'} \sigma(z')\,e^{-\kappa\,|z-z'|}$$

where $\sigma(z')$ is the average charge per unit area in the slab at $z'$, $l_B$ the Bjerrum
length, and $1/\kappa$ the Debye length — both taken from the medium. Because screening keeps
the sum convergent, no infinite-plane correction term is needed. The walls are assumed
neutral; only the explicit ions contribute.

**Without salt** (no Debye length) the kernel falls back automatically to **bare Coulomb**,
the unscreened limit $\kappa\to0$. The plane potential is then the one-dimensional Poisson
Green's function

$$\varphi(z) = -2\pi\,l_B\sum_{z'}\sigma(z')\,|z-z'|,$$

which is finite and physical only for an **electroneutral** slab (the linear background
cancels when $\sum_{z'}\sigma(z')=0$).

The cell geometry is detected automatically: a **cuboid or slit must have a square base**
($L_x = L_y$), and a **cylinder** uses its circular cross-section. The exponential treatment
assumes each charged plane is effectively infinite, which holds only when the lateral box
size is **much larger than the Debye length** — a warning is printed otherwise (screened
kernel only).

### Finite-box correction (optional)

For a box that is *not* much larger than the Debye length, set `finite_box_correction: true`.
The infinite-plane potential is then replaced by that of the *finite* minimum-image
cross-section, valid at any box size:

$$\varphi_\text{box}(z) = \varphi_\infty(z) - \varphi_\text{ext}(z),
\qquad \varphi_\infty(z) = \frac{2\pi\,l_B}{\kappa}\,e^{-\kappa|z|},$$

where $\varphi_\text{ext}$ is the contribution of the charge *outside* the cross-section. This
is the screened (Yukawa) analogue of the finite-box correction of Greberg et al.,
[doi:10/dhb9mj](https://doi.org/10/dhb9mj); screening makes every term finite, so the
divergent and linear pieces of the bare-Coulomb construction cancel before quadrature. For a
**square base** of half-width $a$ ($L_x = 2a$),

$$\varphi_\text{ext}(z) = \frac{8\,l_B}{\kappa}\int_0^{\pi/4}
   \exp\!\Big(\!-\kappa\sqrt{a^2/\cos^2\theta + z^2}\,\Big)\,\mathrm{d}\theta,$$

a smooth integral evaluated by quadrature; for a **disk** of radius $R$ (cylinder) it has the
closed form

$$\varphi_\text{ext}(z) = \frac{2\pi\,l_B}{\kappa}\,e^{-\kappa\sqrt{R^2 + z^2}}.$$

Both vanish as the cross-section grows ($\varphi_\text{ext}\to0$ for $\kappa a\gg1$), so the
correction matters only for thin boxes. Enable it only when the simulation itself does **not**
already apply such an external correction, otherwise the far field is subtracted twice.

For the **unscreened** (bare-Coulomb) kernel the correction reduces to Greberg's original
square-base form,

$$\varphi_\text{ext}(z) = -2\pi\,l_B\,|z| - l_B\,u_\text{box}(z),
\qquad
u_\text{box}(z) = 8a\,\ln\!\frac{\sqrt{2a^2+z^2}+a}{\sqrt{a^2+z^2}}
   - 2z\left(\frac{\pi}{2} + \arcsin\frac{a^4 - z^4 - 2a^2z^2}{(a^2+z^2)^2}\right),$$

recovered from the screened form as $\kappa\to0$ once the regularizing constant $2\pi l_B/\kappa$
is removed. Because Greberg's construction models a square minimum-image box, the unscreened
correction is **only defined for a square base**; enabling it for a cylinder is an error.

### Example

```yaml
analysis:
  - !ElectricPotentialProfile
    frequency: !Every 10
    # selection, resolution and file are optional:
    # selection: "all"      # atoms contributing charge (default: all)
    # resolution: 0.5       # slab thickness Δz in Å (default: 0.5)
    # file: potential.csv   # output file (default: potential.csv)
```

### Options

Key                    | Required | Default         | Description
---------------------- | -------- | --------------- | -------------------------------------------
`selection`            | no       | `all`           | Atoms whose charge contributes to the profile
`resolution`           | no       | `0.5`           | Slab thickness Δz (Å) along the $z$-axis
`finite_box_correction`| no       | `false`         | Report the finite-box (Greberg) potential instead of the infinite-plane one
`file`                 | no       | `potential.csv` | Output file path (see [Output file formats](#output-file-formats))
`frequency`            | yes      |                 | Sample frequency, e.g. `!Every 10`

The output file contains, per slab: the position `z/Å`, the slab charge density (per area
`e·Å⁻²` and per volume `e·Å⁻³`), the potential `potential/mV` with its statistical error
`potential_error/mV`, and the electric field `field/mV·Å⁻¹`. `output.yaml` additionally
reports the potential at each wall and at the midplane (with the wall-to-midplane drops) as
mean ± error. For reliable numbers, **equilibrate first and start production from a state
file** (`-s`).

---

## Mean Along Coordinate

Computes the average of one collective variable (CV1) binned along another (CV2).
CV2 is discretised into uniform bins of width `resolution`; no range is required
since bins are created on demand using a sorted map.
The output file contains columns for the bin center, running mean of CV1, and sample count per bin.
The file is rewritten at each sample so partial results survive crashes.

### Example

Average charge of GLU residue as a function of distance from a protein:

```yaml
analysis:
  - !MeanAlongCoordinate
    property: charge
    selection: "molecule GLU"
    coordinate:
      property: mass_center_separation
      selection: "molecule GLU"
      selection2: "molecule protein"
      resolution: 0.5
    file: charge_vs_dist.dat
    frequency: !Every 100
```

### Options

Key           | Required | Default | Description
------------- | -------- | ------- | -------------------------------------------
`property`    | yes      |         | CV1 type to average (see [Supported properties](#supported-properties))
`selection`   | depends  |         | Selection for CV1
`coordinate`  | yes      |         | CV2 block (must include `resolution`)
`file`        | yes      |         | Output file path (see [Output file formats](#output-file-formats))
`frequency`   | yes      |         | Sample frequency, e.g. `!Every 100`

The `coordinate` block accepts all [collective variable](#collective-variable) fields
(`property`, `selection`, `dimension`, etc.) plus a required `resolution` for the bin width.

---

## Rotational Diffusion

Estimates the anisotropic rotational diffusion tensor from the quaternion
covariance matrix Q̃(τ) of molecular orientations
([Favro 1960](https://doi.org/10.1103/PhysRev.119.53);
[Holtbrügge & Schäfer 2025](https://doi.org/10.1101/2025.05.27.656261)).

For each lag τ (in snapshot units), the body-frame reorientation quaternion
q(t,τ) = q(t)·q⁻¹(t+τ) is computed, and its vector components form the 3×3
covariance matrix Q̃_ij(τ) = ⟨q_i·q_j⟩. In the principal coordinate system,
Q̃ is diagonal and follows the Favro model:

Q_ii(τ) = ¼(1 + exp(−(D_j+D_k)τ) − exp(−(D_i+D_j)τ) − exp(−(D_i+D_k)τ))

where D_x, D_y, D_z are the principal rotational diffusion coefficients
(rad²/snapshot). Correlations are averaged over all molecules matching the
selection, exploiting ensemble averaging.

The selection must resolve to molecular groups (not atomic) that carry
rigid-body quaternion state.

### Example

```yaml
analysis:
  - !RotationalDiffusion
    selection: "molecule Water"
    file: rotdiff.dat.gz
    frequency: !Every 100
    max_lag: 1000
```

### Options

Key          | Required | Default | Description
------------ | -------- | ------- | -------------------------------------------
`selection`  | yes      |         | Selection for molecular group(s) to track
`file`       | no       |         | Streaming output file (see [Output file formats](#output-file-formats))
`frequency`  | yes      |         | Sample frequency, e.g. `!Every 100`
`max_lag`    | no       | `1000`  | Max lag in snapshots; actual time window = max_lag × frequency

The `max_lag` should be large enough for Q̃_ii(τ) to approach its plateau at ¼.
A warning is emitted if convergence is not reached.

### Output

The YAML output includes the covariance matrix at log-spaced lags,
time-dependent diffusion coefficients D_x(τ), D_y(τ), D_z(τ) from
eigenvalue decomposition, and an isotropic estimate from the trace.

---

## Rerun

The `rerun` subcommand replays a trajectory through a (possibly different) Hamiltonian,
running the analysis pipeline on each frame. This decouples analysis from propagation,
enabling e.g. comparison of explicit nonbonded energies against tabulated 6D potentials
for the same configurations.

The input YAML provides the Hamiltonian and analysis configuration;
the `propagate:` section is ignored. All analysis frequencies are overridden to
sample every frame.

### Reweighting biased trajectories

If the Hamiltonian contains a [`penalty`](energy.md#penalty-flat-histogram-bias) term
(from a converged Wang-Landau run), rerun automatically reweights all analyses by
$w = \exp(-\ln g(\text{bin}))$, recovering correct ensemble averages from the
biased trajectory. This is logged at startup:

```
Reweighting enabled: penalty bias detected (Δg=X.X kT)
```

No special configuration is needed — include the same `energy.penalty` section
used during the biased simulation.

### Usage

```sh
faunus rerun -i input.yaml --traj traj.xtc [--aux traj.aux]
```

Flag      | Required | Default                    | Description
--------- | -------- | -------------------------- | -------------------------------------------
`-i`      | yes      |                            | Input YAML with Hamiltonian + analysis config
`--traj`  | yes      |                            | XTC trajectory file
`--aux`   | no       | `traj.aux` (from `--traj`) | Frame state file

### Requirements

The trajectory must have been produced with `save_frame_state: true` on the
[Trajectory](#trajectory) analysis, so that a matching `.aux` file exists.
The `.aux` file header must match the topology in the rerun input (same number
of groups, particles, and molecule types).

### Example workflow

1. Run the original simulation with frame state output:

    ```yaml
    analysis:
      - !Trajectory
        file: traj.xtc
        frequency: !Every 100
        save_frame_state: true
      - !RadialDistribution
        selections: ["molecule A", "molecule B"]
        file: rdf_explicit.dat
        dr: 0.1
        frequency: !Every 100
    ```

2. Rerun with a different Hamiltonian (e.g. 6D tabulated potential):

    ```sh
    faunus rerun -i input_6dtable.yaml --traj traj.xtc -o output_rerun.yaml
    ```

    where `input_6dtable.yaml` uses the same topology but a different energy section,
    and defines the desired analysis objects (e.g. RDF, energy time series).

3. Compare the RDFs from the original and rerun outputs.
