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
    resolution: 0.1
    frequency: !End
```

### Sampling frequency

Each analysis requires a `frequency` field controlling when it is evaluated.

Frequency         | Description
----------------- | -------------------------------------------
`!Every N`        | Every N steps
`!Once N`         | Once at step N
`!End`            | Once after the last step

### Reported quantities

In `output.yaml` every analysis reports `num_samples`, the number of frames it sampled. Analyses
that loop over molecules also report how many molecules they accumulated: `num_groups_sampled` for
Polymer Shape, `num_blocks` for Widom Rotation. These counts give the raw observations behind the
reported means. Where an analysis uses automatic blocking, adjacent observations are combined into
blocks of increasing length. Longer blocks are less correlated with one another, so the estimated
standard error grows with block length and levels off once the blocks are effectively independent;
the error is read from that plateau
([Flyvbjerg & Petersen, 1989](https://doi.org/10.1063/1.457480)). A run whose blocking curve has
not levelled off is too short to resolve the correlation time, and the largest estimate is reported.
Levels holding fewer than 16 effective blocks are too noisy to estimate a variance and are skipped;
if no level qualifies, the uncorrelated standard error is reported. The mean always includes every
observation, and the blocking needs no input from the user — the one exception is Virtual Volume
Move, whose `block_size` is described with that analysis.

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
`selection`  | depends  |         | Selection expression for one atom or group (single-group properties)
`selections` | depends  |         | Pair `[a, b]` of selections (two-group properties, e.g. `mass_center_separation`)
`file`       | no       |         | Output file path (see [Output file formats](#output-file-formats)); omit to only track the mean

A collective variable specifies only *what* is measured. Where a value needs a
grid — the `coordinate` of [`mean_along_coordinate`](#mean-along-coordinate) and
of [Wang-Landau](wang_landau.md)/[penalty](energy.md#penalty-flat-histogram-bias) — the `resolution`
(and, for Wang-Landau, the `range`) is given alongside the CV there, not as a CV
option. Supplying `range`/`resolution` on a plain CV is an error.

### Supported properties

Property                 | Selection       | Description
------------------------ | --------------- | -------------------------------------------
`volume`                 | none            | Cell measure via `projection`: volume (`xyz`), area (`xy`), or length (`z`)
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
center of mass. Eigenvalue decomposition yields the principal moments
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

Per-group charge, dipole, and quadrupole analysis, averaged over all groups
matching a selection. Useful for tracking titration state and charge
fluctuations.

Reports:

- net charge $\langle Z\rangle \pm \sigma$ — mean and standard deviation
- capacitance $C = \langle Z^2\rangle - \langle Z\rangle^2$ — charge variance
- dipole magnitude $\langle|\boldsymbol{\mu}|\rangle \pm \sigma$ (eÅ)
- quadrupole magnitude $\langle\|\boldsymbol{\Theta}\|\rangle \pm \sigma$ — Frobenius norm (eÅ²)
- quadrupole tensor — mean and standard deviation $\sigma$ of the six independent components
- per-atom `⟨q⟩` and `⟨q²⟩-⟨q⟩²` — for atoms with fluctuating charge (e.g. from titration or atom swaps)

The dipole and quadrupole moments are computed relative to each group's
center of mass with periodic boundary conditions applied.
Handles atom-type swaps (titration) and GCMC (only active groups contribute).

The reported quadrupole is the traceless form (Buckingham convention)

$$\Theta_{\alpha\beta} = \tfrac{1}{2}\sum_i q_i\left(3 d_{i\alpha} d_{i\beta} - d_i^2\,\delta_{\alpha\beta}\right),$$

where $\boldsymbol{d}_i$ points from the group's center of mass to atom $i$. The isotropic
part dropped here does not affect the far-field potential, and removing it makes
the moment vanish for isotropic charge distributions, so the values compare
directly with experiment and literature.

### Selection resolves group-wise

The selection picks whole groups that contain at least one active
matching atom — not just the matched atoms themselves. So an atom-level
selection like `atomtype CA` pulls in every group that has an active CA
atom and accumulates the full group's charge and dipole moment. This is
useful when a molecule changes name during speciation, for example across
protonation states, but still contains a stable marker atom:

```yaml
- !Multipole
  selection: "atomtype INO"   # picks every phytate, regardless of protonation state
  frequency: !Every 10
```

By contrast, `!CollectiveVariable property: charge` sums the charge of
only the matched atoms, not their parent groups — see [Supported
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
  charge: {mean: 4.7785, error: 2.2373}
  capacitance: 5.005
  dipole_moment: {mean: 141.6532, error: 23.8816}
  quadrupole_moment: {mean: 312.4187, error: 45.1902}
  quadrupole_tensor:            # order [xx, xy, xz, yy, yz, zz], loads straight into NumPy
    order: [xx, xy, xz, yy, yz, zz]
    values: [180.21, 12.03, -4.11, -95.44, 8.90, -84.77]
    errors: [22.14, 3.02, 1.88, 15.30, 2.71, 11.05]
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

## Multipole Distribution

This analysis decomposes the electrostatic interaction between two molecular species into
ion–ion, ion–dipole, dipole–dipole, and ion–quadrupole terms, each averaged as a function of
the center-of-mass separation _R_. Comparing their sum with the exact Coulomb energy shows how
far a multipole expansion reproduces the true interaction, and how the molecules' mutual
orientation changes with distance. Each species may contain many molecules: every frame the
analysis averages over all molecule pairs — all pairs of distinct molecules when the two
selections name the same species, or all cross pairs when they name different species — so it
characterises a whole solution of _N_ macroions, not just an isolated pair.

Each molecule is reduced to a net charge, a dipole moment, and a traceless quadrupole moment
about its center of mass, with periodic boundaries applied. The pair energy at separation
$\boldsymbol{R} = \boldsymbol{r}_a - \boldsymbol{r}_b$ is

$$
u = \frac{q_a q_b}{R} + \frac{q_a(\boldsymbol{\mu}_b\cdot\boldsymbol{R}) - q_b(\boldsymbol{\mu}_a\cdot\boldsymbol{R})}{R^3} + \frac{\boldsymbol{\mu}_a\cdot\boldsymbol{\mu}_b}{R^3} - \frac{3(\boldsymbol{\mu}_a\cdot\boldsymbol{R})(\boldsymbol{\mu}_b\cdot\boldsymbol{R})}{R^5} + \frac{q_a\,\boldsymbol{R}^{\!\top}\boldsymbol{\Theta}_b\,\boldsymbol{R} + q_b\,\boldsymbol{R}^{\!\top}\boldsymbol{\Theta}_a\,\boldsymbol{R}}{R^5}
$$

Each term is thermally averaged over orientations and co-solute configurations. Energies are
reported in kJ/mol: the geometric factors above carry the Bjerrum length and _RT_ of the
[medium](topology.md). The exact energy is the explicit sum over atomic charges,
$u_\text{exact} = \sum_i^a\sum_j^b q_i q_j / |\boldsymbol{r}_i-\boldsymbol{r}_j|$.

Alongside the energies, five orientational measures are accumulated (column name in
parentheses):

- dipole correlation $\langle\hat{\boldsymbol{\mu}}_a\cdot\hat{\boldsymbol{\mu}}_b\rangle$ (`mucorr`), between $-1$ and $1$ — mean cosine between the dipole directions: $+1$ parallel, $-1$ antiparallel, $0$ uncorrelated.
- nematic order $\langle P_2\rangle = \langle(3\cos^2\theta - 1)/2\rangle$ (`p2`), between $-0.5$ and $1$ — alignment blind to dipole sign: $+1$ collinear (parallel or antiparallel), $0$ random, $-0.5$ mutually perpendicular.
- longitudinal projection $\langle(\hat{\boldsymbol{\mu}}_a\cdot\hat{\boldsymbol{R}})(\hat{\boldsymbol{\mu}}_b\cdot\hat{\boldsymbol{R}})\rangle$ (`long`), between $-1$ and $1$ — orientation relative to the separation axis: near $+1$ both point along $\boldsymbol{R}$ (head-to-tail, end-on), near $0$ both lie perpendicular to it (side-by-side, broadside).
- Kirkwood factor $g_K(R) = 1 + \frac{1}{N}\sum_{i \neq j,\, r_{ij} \leq R}\langle\hat{\boldsymbol{\mu}}_i\cdot\hat{\boldsymbol{\mu}}_j\rangle$ (`gK`) — the running dipole-correlation integral, where $N$ counts the reference molecules that carry a dipole (apolar molecules are excluded from the normalization): $g_K > 1$ signals net parallel ordering of the surrounding dipoles, $g_K < 1$ net antiparallel ordering. It applies when both selections are the same species; for two different species the bare cumulative sum is reported, without the $+1$.
- quadrupole correlation — the tensor overlap $\langle\boldsymbol{\Theta}_a : \boldsymbol{\Theta}_b\rangle$ (`quadcorr`) and its normalized form $\langle\hat{\boldsymbol{\Theta}}_a : \hat{\boldsymbol{\Theta}}_b\rangle$ (`quadcorr_norm`), between $-1$ and $1$: $+1$ identically oriented quadrupoles, $-1$ maximally anti-aligned, $0$ uncorrelated.

Both selections must resolve to molecules with a center of mass. Results are written to a CSV
file, one row per distance bin, with columns
`R,exact,tot,ii,id,dd,iq,mucorr,p2,long,gK,quadcorr,quadcorr_norm`.

### Example

```yaml
analysis:
  - !MultipoleDistribution
    selections: ["molecule protein", "molecule protein"]
    resolution: 2.0
    frequency: !Every 100
```

The output defaults to `multipole_dist.csv`; set `file` to override it.

### Options

Key           | Required | Default              | Description
------------- | -------- | -------------------- | ----------------------------------------
`selections`  | yes      |                      | Two selection expressions, _a_ and _b_
`file`        | no       | `multipole_dist.csv` | Output CSV file
`resolution`          | no       | 1.0                  | Distance resolution along _R_ (Å)
`max`         | no       | half box             | Maximum separation (Å)
`frequency`   | yes      |                      | Sample frequency

A medium is required for the Bjerrum length.

---

## Energy

Streams energy values to a file at each sampled step.
Two modes are supported:

- Total (default): writes every Hamiltonian term plus the total.
  Output columns: `step term1 term2 ... total`.
- Partial: writes the nonbonded energy between two sets of atoms
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

- `.sizes.dat` — per-frame group active counts (written when any group
  has inactive atoms). The VMD script hides inactive atoms (radius = 0).
- `.charges.dat` — per-frame atom charges (only when `save_charges: true`).
  The VMD script then updates charges each frame so that atom-type swaps from
  titration and speciation are reflected in the charge coloring. Without it,
  VMD colors by the static charges stored in the PSF.

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
`save_charges`     | no       | `false` | Write a per-frame `.charges.dat` file for VMD charge coloring of titration/speciation swaps. One float per atom per frame, so large for big systems.

### Output files

Given `file: traj.xtc`, the following files are produced:

File               | Description
------------------ | -------------------------------------------
`traj.xtc`         | Trajectory (coordinates per frame)
`traj.psf`         | X-PLOR PSF topology (atoms, bonds, angles, dihedrals, charges, masses)
`traj.tcl`         | VMD scene script (`vmd -e traj.tcl` loads everything)
`traj.charges.dat` | Per-frame atom charges (only when `save_charges: true`)
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

- Atom-atom (default): pairwise distances between individual atoms.
  Intramolecular pairs (atoms in the same molecule) are excluded by default.
- COM-COM (`use_com: true`): pairwise distances between molecular
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
    resolution: 0.1
    frequency: !Every 100

  # COM-COM RDF for polymer molecules
  - !RadialDistribution
    selections: ["molecule polymer", "molecule polymer"]
    use_com: true
    file: rdf_com.dat
    resolution: 0.5
    max: 30.0
    frequency: !Every 100
```

### Options

Key                        | Required | Default               | Description
-------------------------- | -------- | --------------------- | -------------------------------------------
`selections`               | yes      |                       | Pair of selection expressions, e.g. `["atomtype Na", "atomtype Cl"]`
`file`                     | yes      |                       | Output file path (see [Output file formats](#output-file-formats))
`resolution`                       | yes      |                       | Bin width in distance units
`frequency`                | yes      |                       | Sample frequency, e.g. `!Every 100`
`max`                      | no       | half shortest box dim | Maximum distance for histogram
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
`reference_file`    | no       |              | Optional XYZ structure of the reference molecule, for visualizing the density around it
`resolution`        | no       | `1.0`        | Cubic grid spacing in Å
`padding`           | no       | `8.0`        | Margin in Å added on every side of the reference molecule's bounding box
`bulk_normalize`    | no       | `true`       | Normalize by bulk density to produce dimensionless relative density
`exclude_reference` | no       | `true`       | Skip target atoms belonging to the current reference group

The grid is the axis-aligned bounding box of the active reference molecule(s) in
the body frame, rounded to the grid spacing and enlarged by `padding` on every
side. Set `reference_file` to also write the reference molecule itself (in the
same body frame) so it can be overlaid on the density. Output is written once at
the end of the run.

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
a charging parameter λ ∈ [0, 1]. Each sampled configuration contributes one
insertion average. Automatic blocking estimates the reported standard errors.

Pair interactions for the ghost particle are defined directly in the analysis
block using the same syntax as the `!Nonbonded` term's `default`, allowing arbitrary
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
    displacement: 0.01
    directions: !z
    file: force.dat
    frequency: !Every 10
```

### Options

Key           | Required | Default  | Description
------------- | -------- | -------- | -------------------------------------------
`selection`   | yes      |          | Selection matching exactly one molecule
`displacement`| yes      |          | Displacement magnitude (Å)
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
    volume_displacement: 0.2
    method: Isotropic
    file: pressure.csv
    frequency: !Every 10
```

### Options

Key           | Required | Default      | Description
------------- | -------- | ------------ | -------------------------------------------
`volume_displacement`          | yes      |              | Volume displacement (Å³)
`method`      | no       | `Isotropic`  | Scaling policy: `Isotropic`, `ScaleZ`, `ScaleXY`
`block_size`  | no       | `100`        | Samples per free-energy block (see below)
`file`        | no       |              | Output file path (see [Output file formats](#output-file-formats))
`frequency`   | yes      |              | Sample frequency, e.g. `!Every 10`

### Block size

Unlike other analyses, Virtual Volume Move averages an exponential, $\langle e^{-\Delta U/k_BT}\rangle$,
and $\Delta U$ enters non-linearly. Its uncertainty is therefore estimated from the scatter of
free energies computed over consecutive blocks of `block_size` samples, rather than from the
samples themselves. Each block must be long enough for its exponential average to converge, or
the reported error describes the scatter of $\Delta U$ instead of that of $P_\text{ex}$.

The blocks so formed feed the automatic blocking described under
[Reported quantities](#reported-quantities), which handles any residual correlation between
them. That hierarchy needs at least 16 blocks, so a run should collect at least
$16 \times$ `block_size` samples; below that the reported error ignores correlation between
blocks. Raising `block_size` therefore trades statistical resolution of the correlation time for
convergence of the exponential average within each block.

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

The analysis is hardcoded to a `Slit` cell (periodic in *xy*, walls at $z=\pm L_z/2$, so
the midplane is $z=0$) and to electrostatics only; it suits the primitive-model
double layer with point-charge counterions. Valency mixtures need nothing special —
select all the counterions (e.g. `"atomtype Na Ca"`) and report. Results are block-averaged
over sampled configurations and reported as mean ± standard error. Automatic blocking estimates
the errors.

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

A long-range correction for the finite lateral box, reported as `F_iPB`, is computed and
added to the pressure automatically — it needs no configuration and handles valency
mixtures. For reliable numbers, equilibrate first and start production from a state file
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

Without salt (no Debye length) the kernel falls back automatically to bare Coulomb,
the unscreened limit $\kappa\to0$. The plane potential is then the one-dimensional Poisson
Green's function

$$\varphi(z) = -2\pi\,l_B\sum_{z'}\sigma(z')\,|z-z'|,$$

which is finite and physical only for an electroneutral slab (the linear background
cancels when $\sum_{z'}\sigma(z')=0$).

The cell geometry is detected automatically: a cuboid or slit must have a square base
($L_x = L_y$), and a cylinder uses its circular cross-section. The exponential treatment
assumes each charged plane is effectively infinite, which holds only when the lateral box
size is much larger than the Debye length — a warning is printed otherwise (screened
kernel only).

The kernel treats the $z$-axis as open rather than periodic: the separation $|z-z'|$ is never
wrapped into the box, so the profile applies to a slab confined by walls or otherwise
electroneutral along $z$, and it becomes inaccurate near the $z$-boundaries of a cell that is
periodic in $z$. The slab grid is laid out once at start-up and the analysis therefore requires
a constant cell; it aborts if the box changes, as under volume moves (NPT).

### Finite-box correction (optional)

For a box that is *not* much larger than the Debye length, set `finite_box_correction: true`.
The infinite-plane potential is then replaced by that of the *finite* minimum-image
cross-section:

$$\varphi_\text{box}(z) = \varphi_\infty(z) - \varphi_\text{ext}(z),
\qquad \varphi_\infty(z) = \frac{2\pi\,l_B}{\kappa}\,e^{-\kappa|z|},$$

where $\varphi_\text{ext}$ is the contribution of the charge *outside* the cross-section. This
is the screened (Yukawa) analogue of the finite-box correction of Greberg et al.,
[doi:10/dhb9mj](https://doi.org/10/dhb9mj); screening makes every term finite, so the
divergent and linear pieces of the bare-Coulomb construction cancel before quadrature. For a
square base of half-width $a$ ($L_x = 2a$),

$$\varphi_\text{ext}(z) = \frac{8\,l_B}{\kappa}\int_0^{\pi/4}
   \exp\!\Big(\!-\kappa\sqrt{a^2/\cos^2\theta + z^2}\,\Big)\,\mathrm{d}\theta,$$

a smooth integral evaluated by quadrature; for a disk of radius $R$ (cylinder) it has the
closed form

$$\varphi_\text{ext}(z) = \frac{2\pi\,l_B}{\kappa}\,e^{-\kappa\sqrt{R^2 + z^2}}.$$

This disk expression is the potential *on the cylinder axis* — the on-axis potential of a
uniformly charged disk of radius $R$ — rather than a laterally-averaged cross-section, so for a
cylinder the correction is exact on-axis and approximate off it. Both forms vanish as the
cross-section grows ($\varphi_\text{ext}\to0$ for $\kappa a\gg1$), so the correction matters
only for thin boxes. Enable it only when the simulation itself does not
already apply such an external correction, otherwise the far field is subtracted twice.

For the unscreened (bare-Coulomb) kernel the correction reduces to Greberg's original
square-base form,

$$\varphi_\text{ext}(z) = -2\pi\,l_B\,|z| - l_B\,u_\text{box}(z),
\qquad
u_\text{box}(z) = 8a\,\ln\!\frac{\sqrt{2a^2+z^2}+a}{\sqrt{a^2+z^2}}
   - 2|z|\left(\frac{\pi}{2} + \arcsin\frac{a^4 - z^4 - 2a^2z^2}{(a^2+z^2)^2}\right),$$

recovered from the screened form as $\kappa\to0$ once the regularizing constant $2\pi l_B/\kappa$
is removed. Because Greberg's construction models a square minimum-image box, the unscreened
correction is only defined for a square base; enabling it for a cylinder is an error.

### ζ-potential (optional)

Setting `zeta_threshold` reports a ζ-potential — the potential at the hydrodynamic shear
plane — directly comparable to electrokinetic measurements. For a charged wall or brush the
shear plane sits at the outer edge of the fixed charge, where $\sigma(z)$ has decayed into the
diffuse layer. The edge is located as the slab nearest the solution where

$$|\sigma(z_\text{shear})| > \texttt{zeta\_threshold}\cdot\max_\text{wall}|\sigma(z)|,$$

where the $|\sigma(z)|$-weighted centroid first selects which wall the fixed charge sits on and
the peak $\max_\text{wall}|\sigma(z)|$ is taken over that wall's side, so an oppositely charged
far wall cannot mask a weaker surface charge on the measured side. Because $\sigma(z)$ is built
from the configured `selection`, which defaults to all charged atoms, mobile ions then enter it
and can shift the shear plane; restrict `selection` to the fixed wall or brush charge for a
well-defined ζ. The reported ζ is the mean potential $\varphi(z_\text{shear})$ with its
statistical error, alongside the slab position $z_\text{shear}$. A cutoff of `0.02` (2 % of the peak) is a
reasonable starting point, but the right value is system-dependent, so inspect $\sigma(z)$ in
`potential.csv` and adjust. The estimate is only meaningful where the fixed charge is localized
against a wall and decays into solution before the midplane; a slit charged symmetrically on
both walls reports the equivalent value at one of them, and it is skipped when no charge is
sampled. Note that the shear plane is here defined from the charge density, not the solvent
(matter) density that sets the true no-slip surface, so the magnitude depends on this choice
while the sign and trends are robust.

### Example

```yaml
analysis:
  - !ElectricPotentialProfile
    frequency: !Every 10
    # selection, resolution and file are optional:
    # selection: "all"      # atoms contributing charge (default: all)
    # resolution: 0.5       # slab thickness Δz in Å (default: 0.5)
    # zeta_threshold: 0.02  # also report a ζ-potential at the fixed-charge edge (off by default)
    # file: potential.csv   # output file (default: potential.csv)
```

### Options

Key                    | Required | Default         | Description
---------------------- | -------- | --------------- | -------------------------------------------
`selection`            | no       | `all`           | Atoms whose charge contributes to the profile
`resolution`           | no       | `0.5`           | Slab thickness Δz (Å) along the $z$-axis
`finite_box_correction`| no       | `false`         | Report the finite-box (Greberg) potential instead of the infinite-plane one
`zeta_threshold`       | no       | *(off)*         | Report a ζ-potential at the fixed-charge edge, as a fraction of peak $|\sigma|$ (e.g. `0.02`)
`file`                 | no       | `potential.csv` | Output file path (see [Output file formats](#output-file-formats))
`frequency`            | yes      |                 | Sample frequency, e.g. `!Every 10`

The output file contains, per slab: the position `z/Å`, the slab charge density (per area
`e·Å⁻²` and per volume `e·Å⁻³`), the potential `potential/mV` with its statistical error
`potential_error/mV`, and the electric field `field/mV·Å⁻¹`. `output.yaml` additionally reports
the potential at the two outermost slabs (`potential_edge_lower/mV`, `potential_edge_upper/mV`
— the outermost bin centres, half a slab $\Delta z/2$ inside the physical walls) and at the
midplane (`potential_midplane/mV`), together with the edge-to-midplane drops
(`potential_drop_lower/mV`, `potential_drop_upper/mV`), as mean ± error; the midplane and drop
errors are accumulated per configuration, so they reflect the correlation between the slabs they
combine. Automatic blocking estimates these errors across configurations. When `zeta_threshold`
is set it also reports the ζ-potential `zeta/mV` and the shear-plane position `z_shear/Å`. For reliable numbers, **equilibrate first and start production
from a state file** (`-s`).

---

## Density Profile

Average density of a chosen species along the $z$-axis. The cell is cut into slabs of equal
thickness $\Delta z$, and the density in the slab centred at $z$ is the mean number of
particles it holds divided by its volume,

$$\rho(z) = \frac{\langle N(z)\rangle}{A\,\Delta z},$$

where $A$ is the cross-sectional area of the cell. Use it to measure how a species distributes
itself between a wall and the bulk: ion layering at a charged surface, adsorption in a slit
pore, or the density of a fluid along a cylindrical channel.

By default each selected atom counts towards the slab holding it. Set `use_com: true` to
count each selected molecule in the slab holding its centre of mass instead, which profiles
whole molecules rather than their individual atoms. Molecules declared `atomic` have no centre
of mass, so selecting one with `use_com` is an error.

Every configuration contributes its own count, which is then averaged. The profile therefore
stays correct when particles are created and destroyed: $\langle N(z)\rangle$ is the mean
occupancy of the slab in the grand canonical ensemble. Particles absent from a configuration do
not count.

The cell must have the same cross-section at every height — a cuboid, slit, cylinder, or
hexagonal prism — since a slab volume is otherwise undefined. Spherical and unbounded cells are
rejected. The slabs are laid out once at the start of the run and keep their size, so the
profile is meaningful only at constant volume; rescaling the cell later triggers a warning.

### Example

Sodium and chloride profiles across a slit, plus the profile of whole water molecules:

```yaml
analysis:
  - !DensityProfile
    selection: "atomtype Na"
    file: density_na.csv
    frequency: !Every 10
  - !DensityProfile
    selection: "atomtype Cl"
    file: density_cl.csv
    frequency: !Every 10
  - !DensityProfile
    selection: "molecule water"
    file: density_water.csv
    use_com: true          # bin mass centres, not individual atoms
    resolution: 0.5        # slab thickness Δz in Å (default: 1.0)
    frequency: !Every 10
```

Each block profiles one species and writes one file; add a block per species.

### Options

Key          | Required | Default | Description
------------ | -------- | ------- | -----------------------------------------------------------
`selection`  | yes      |         | Atoms, or molecules when `use_com` is set, whose density is profiled
`file`       | yes      |         | Output file path (see [Output file formats](#output-file-formats))
`use_com`    | no       | `false` | Bin the centre of mass of each selected molecule instead of each atom
`resolution` | no       | `1.0`   | Slab thickness Δz (Å) along the $z$-axis
`frequency`  | yes      |         | Sample frequency, e.g. `!Every 10`

The output file gives, per slab, its centre `z/Å` followed by three equivalent measures of the
same density, each with its statistical error: the number density `density/Å⁻³`, the molar
concentration `molarity/mol·L⁻¹`, and the mass density `mass_density/g·mL⁻¹`. `output.yaml`
additionally reports the slab layout and the mean total number of selected particles,
`mean_count`, as mean ± error — a useful check, since summing $\rho(z)\,A\,\Delta z$ over all
slabs must return it. Automatic blocking estimates all these errors across sampled configurations.

---

## Preferential Interaction

Measures the preferential interaction of a ligand around one molecular substrate. A positive
$\Gamma$ means that the local solvent composition is enriched in ligand relative to bulk; a
negative value means that it is depleted. Both implicit- and explicit-solvent estimators are
available.

For implicit solvent, the analysis reports the cumulative ligand excess

$$\Gamma(\delta) = \left\langle N(\delta)
- c_{\mathrm{bulk}}\,\operatorname{Vol}\!\left[D(\delta)\right]\right\rangle,$$

where $N(\delta)$ is the number of ligands in the shell, $c_{\mathrm{bulk}}$ is the measured bulk
concentration, and $D(\delta)$ is the shell volume in the sampled conformation. The product is
formed for each frame before averaging, preserving correlations between concentration and shape.
The bulk concentration is measured in the remaining part of the simulation cell, so no
substrate-free simulation is required.

For explicit solvent, components 1 and 3 denote solvent and ligand. The finite-box estimator is

$$\Gamma_{23}(\delta) = \left\langle N_3(\delta)
- \frac{N_3^{\rm bulk}}{N_1^{\rm bulk}}N_1(\delta)\right\rangle.$$

The ratio and both local populations are evaluated in each frame before averaging. Ligand and
solvent are counted in the same $D(\delta)$; using different spatial regions would introduce a
spurious displaced-solvent contribution. The bulk populations are those outside the widest shell,
and that common bulk ratio is used throughout the profile. A sampled frame with no bulk solvent is
an error because its reference composition is undefined.

### Choosing the shell

Use the profile to select $\Gamma$: increase `shell.max` until $\Gamma(\delta)$ has reached a
plateau. Do not use a profile that still changes at the largest $\delta$; enlarge the cell or
extend the shell instead. In implicit solvent, a hard-core atomic ligand gives an excluded-volume
contribution at $\delta=0$. This interpretation does not apply to a soft or centre-of-mass ligand.

The substrate selection must cover exactly one complete molecular group. For a rigid substrate,
the surface geometry is calculated once and reused under translation and rotation. For a flexible
substrate, the surface, shell volumes, and residue partition are recalculated at every sampled
frame. A flexible substrate can therefore be analyzed during simulation or by
[`rerun`](#rerun), including trajectories whose cuboid changes between frames. Flexible analysis
is substantially more expensive; increase the sampling interval if tessellation dominates the
run time.

Leave sufficient bulk beyond the largest shell. Every sampled conformation must fit together with
its largest shell without overlapping its periodic image or reaching a spherical boundary. A
rigid substrate still requires a fixed cell because its geometric reference is reused.

### Ions and residue contributions

Analyze each ionic species separately. For a charged substrate, individual ion excesses include
the electroneutrality contribution and are not neutral-salt coefficients. Combine the ion
profiles with the salt stoichiometry and the activity convention appropriate to the system.

The residue file partitions the total excess among residues:

$$\Gamma = \sum_i \gamma_i.$$

The contributions sum to the profile at the same shell thickness and provide a spatial
description of ligand enrichment. In implicit solvent, the local-to-bulk partition coefficient is

$$K_{p,i} = \frac{\left\langle N_i(\delta)-N_i(0)\right\rangle}
{\left\langle c_{\mathrm{bulk}}\,v_i^{\rm acc}(\delta)\right\rangle},$$

where $v_i^{\rm acc}$ is the volume accessible to ligand centres around residue $i$. Values above
one indicate local enrichment and values below one indicate depletion. In explicit solvent the
corresponding denominator is

$$\left\langle \frac{N_3^{\rm bulk}}{N_1^{\rm bulk}}
\left[N_{1,i}(\delta)-N_{1,i}(0)\right]\right\rangle,$$

so $K_{p,i}=1$ when the accessible slab has the bulk ligand-to-solvent ratio. The implicit-solvent
hydration density

$$b_{1,i} = \frac{\left\langle v_i^{\rm w}\right\rangle}
{\bar v_{\rm w}\left\langle\mathrm{ASA}_i\right\rangle}$$

estimates the water population from the water-probe volume. With explicit solvent,
$v_i^{\rm w}/\bar v_{\rm w}$ is replaced by the sampled
$N_{1,i}(\delta)-N_{1,i}(0)$ in the ligand-accessible slab. `kp` and `b1` are local descriptors,
not thermodynamic coefficients; they depend on the selected shell thickness and are `nan` when the
required local population, volume, or surface area is unavailable.

### Example

One block is needed for each ligand species. Use `use_com: true` for a molecular ligand; otherwise,
the selected atoms are counted individually. To use the explicit-solvent estimator, add `solvent`
and state whether its selection is counted by molecular mass centre or by atom. Molecular water is
normally counted by mass centre.

```yaml
analysis:
  - !PreferentialInteraction
    substrate: "molecule protein"
    ligand: "atomtype Na"
    shell: {max: 15.0, resolution: 0.5}   # the δ ladder
    profile: gamma_na.csv                 # Γ(δ) — read this first
    file: residues_na.csv                 # one row per residue
    frequency: !Every 100
  - !PreferentialInteraction
    substrate: "molecule protein"
    ligand: "molecule polyphosphate"
    use_com: true                         # count molecules at their centre of mass
    solvent:                              # enables the explicit-solvent estimator
      selection: "molecule water"
      use_com: true                       # count one centre per water molecule
    shell: {max: 15.0, resolution: 0.5}
    profile: gamma_pp.csv
    file: residues_pp.csv
    frequency: !Every 100
```

### Options

Key             | Required | Default | Description
--------------- | -------- | ------- | --------------------------------------------------------
`substrate`     | yes      |         | The complete molecular group the ligand is counted around
`ligand`        | yes      |         | Atoms, or molecules when `use_com` is set, whose excess is measured
`shell`         | yes      |         | The δ ladder: `{max, resolution}` in Å, with `max` a multiple of `resolution`
`use_com`       | no       | `false` | Count the mass centre of each selected molecule instead of each atom
`solvent`       | no       |         | Explicit solvent as `{selection, use_com}`; its presence selects the finite-box estimator
`radius`        | no       | σ/2, or 0 with `use_com` | Ligand radius (Å), setting the exclusion boundary
`probe_radius`  | no       | `1.4`   | Water radius (Å), setting the accessible surface and implicit-solvent shell behind $b_1$
`profile`       | no       |         | Output file for Γ(δ) (see [Output file formats](#output-file-formats))
`file`          | no       |         | Output file for the per-residue table
`frequency`     | yes      |         | Sample frequency, e.g. `!Every 100`

The profile file contains `delta/Å`, `gamma`, and its statistical error. The residue file contains
the residue identifier, `asa/Å²`, `accessible_volume/Å³`, `b1/Å⁻²`, `kp`, and `gamma` with its
error at the widest shell. Geometric quantities are ensemble averages for a flexible substrate.
`output.yaml` reports `gamma` at that shell, the measured bulk `concentration/Å⁻³`, and the mean
`excluded_volume/Å³`. With explicit solvent it also reports `solvent_concentration/Å⁻³` and
`bulk_ligand_to_solvent_ratio`. Rerun weights are applied to all reported averages. Errors are
estimated automatically by hierarchical blocking: adjacent observations are combined into blocks
of increasing length, and the coarsest level with at least 16 blocks sets the standard error. For
short trajectories that never reach that threshold, the finest level is used instead. This reduces
bias from serial correlation without requiring a user-selected block size. The trajectory must
still be long enough to contain multiple correlation times ([Flyvbjerg & Petersen, 1989](https://doi.org/10.1063/1.457480)).

### References

- Record & Anderson, *Biophys. J.* **68**, 786 (1995).
  [10.1016/S0006-3495(95)80254-7](https://doi.org/10.1016/S0006-3495(95)80254-7)
- Courtenay, Capp & Record, *Protein Sci.* **10**, 2485 (2001).
  [10.1110/ps.ps.20801](https://doi.org/10.1110/ps.ps.20801)
- Pegram & Record, *J. Phys. Chem. B* **112**, 9428 (2008).
  [10.1021/jp800816a](https://doi.org/10.1021/jp800816a)
- Tang & Bloomfield, *Biophys. J.* **82**, 2876 (2002).
  [10.1016/S0006-3495(02)75629-4](https://doi.org/10.1016/S0006-3495(02)75629-4)
- Vagenende & Trout, *Biophys. J.* **103**, 1354 (2012).
  [10.1016/j.bpj.2012.08.011](https://doi.org/10.1016/j.bpj.2012.08.011)
- Flyvbjerg & Petersen, *J. Chem. Phys.* **91**, 461 (1989).
  [10.1063/1.457480](https://doi.org/10.1063/1.457480)

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
      selections: ["molecule GLU", "molecule protein"]
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
covariance matrix $\tilde{Q}(\tau)$ of molecular orientations
([Favro 1960](https://doi.org/10.1103/PhysRev.119.53);
[Holtbrügge & Schäfer 2025](https://doi.org/10.1101/2025.05.27.656261)).

For each lag $\tau$ (in snapshot units), the body-frame reorientation quaternion
$q(t,\tau) = q(t)\cdot q^{-1}(t+\tau)$ is computed, and its vector components form
the $3\times3$ covariance matrix $\tilde{Q}_{ij}(\tau) = \langle q_i\cdot q_j\rangle$.
In the principal coordinate system, $\tilde{Q}$ is diagonal and follows the Favro model

$$\tilde{Q}_{ii}(\tau) = \tfrac{1}{4}\left(1
  + e^{-(D_j+D_k)\tau}
  - e^{-(D_i+D_j)\tau}
  - e^{-(D_i+D_k)\tau}\right),$$

where $D_x$, $D_y$ and $D_z$ are the principal rotational diffusion coefficients
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
`max_lag`    | no       | `1000`  | Max lag in snapshots; actual time window = `max_lag` × `frequency`

The `max_lag` should be large enough for $\tilde{Q}_{ii}(\tau)$ to approach its plateau at $\tfrac{1}{4}$.
A warning is emitted if convergence is not reached.

### Output

The YAML output includes the covariance matrix at log-spaced lags,
time-dependent diffusion coefficients $D_x(\tau)$, $D_y(\tau)$, $D_z(\tau)$ from
eigenvalue decomposition, and an isotropic estimate from the trace.

---

## Widom Rotation

Measures how the surroundings restrict a molecule's orientation. For each
snapshot the analysis holds a molecule at its center of mass and rigidly turns it
to `M` trial orientations spread evenly over all rotations
([Alexa 2022](https://doi.org/10.1109/CVPR52688.2022.00811)), evaluating each
with the run's own energy function. Turning a molecule about its center of mass
leaves its internal geometry unchanged, so the internal energy cancels.

Each snapshot is referenced to its deepest accessible orientation, so a constant
that shifts every orientation equally cancels. This keeps the analysis correct
for a stateful energy term that returns a whole-system total for a single-molecule
change: Ewald reciprocal space returns the entire $k$-space energy, not the tagged
molecule's share. The reported free energy and interaction are therefore excess
quantities, measured relative to that deepest well rather than as absolute
interactions.

From the referenced trial energies $u_k$ the analysis forms, per snapshot, a cage
free energy, a mean energy and an entropy that satisfy $F_b = U_b - T S_b$
exactly, and averages each over snapshots. The cage free energy is

$$F_b = -k_BT \ln\left[\frac{1}{M}\sum_k e^{-(u_k - u_\text{min})/k_BT}\right],
\qquad W = \langle F_b\rangle,$$

which matches the per-vector cage partition function of
[Akke et al. 1993](https://doi.org/10.1021/ja00074a073). The mean excess
interaction is $U_b = \sum_k w_k (u_k - u_\text{min})$ with Boltzmann weights
$w_k$, and the effective number of accessible orientations is
$N_\text{eff} = 1/\sum_k w_k^2$. The orientational entropy relative to free
rotation,

$$S_\text{orient}/R = -\sum_k w_k \ln (M w_k),$$

follows directly from the sampled orientations and assumes nothing about the shape
of the well. It is dimensionless and never positive; zero means the molecule
rotates freely.

For each requested vector the analysis reports the ensemble Lipari–Szabo order
parameter ([Lipari & Szabo 1982](https://doi.org/10.1021/ja00381a009)):

$$S^2 = \frac{3}{2}\sum_{\alpha\beta}\langle v_\alpha v_\beta\rangle^2 - \frac{1}{2},$$

where the average runs over all snapshots, so $S^2 = 1$ marks a fully locked
vector and $S^2 = 0$ a free one. An axis that is locked within each snapshot but
wanders between them gives $S^2 \to 0$, as the ensemble average intends. A vector
is given as a pair of atoms (`!Pair [i, j]`), a gyration principal axis
(`!Axis 0|1|2`), or an explicit direction in the molecule's reference frame
(`!Body [x, y, z]`).

Two further diagnostics are optional. The root-mean-square torque
$\sqrt{\langle\tau^2\rangle}$ about each lab axis measures the strength of the
orientational restoring force; the mean torque of an equilibrium molecule is zero,
so its magnitude is the informative quantity. The local-harmonic stiffness
$RT/\sigma^2$ estimates the curvature of the well from the librational variance
$\sigma^2$ about each principal axis. It is a small-angle estimate, defined only
for a confined, near-harmonic cage: an axis whose libration approaches free
rotation is left out, so the reported value is conditioned on the confined
snapshots.

The analysis reports one set of numbers per run; for a spatial profile, run
separate [umbrella](umbrella.md) windows and combine them afterwards. With
implicit solvent the reported energies are free energies relative to pure solvent,
not mechanical energies. A `rerun` weights every snapshot by its trajectory
weight, so biased (penalty or Wang–Landau) trajectories are reweighted correctly.

The selection must resolve to molecular groups (not atomic) of a single kind.

### Example

```yaml
analysis:
  - !WidomRotation
    selection: "molecule phytate"
    orientations: 500
    vectors:
      - !Pair [0, 5]
      - !Axis 2
      - !Body [0.0, 0.0, 1.0]
    torque: true
    stiffness: true
    file: widomrot.csv.gz
    frequency: !Every 100
```

### Options

Key            | Required | Default | Description
-------------- | -------- | ------- | -------------------------------------------
`selection`    | yes      |         | Selection for molecular group(s) of one kind
`orientations` | yes      |         | Number of trial orientations `M` (typically 100–1000)
`vectors`      | no       | `[]`    | Vectors for `S²`: `!Pair [i,j]`, `!Axis 0\|1\|2`, or `!Body [x,y,z]`
`torque`       | no       | `false` | Also report the RMS torque about each axis
`dtheta`       | no       | `0.01`  | Finite-difference rotation step for the torque (rad)
`stiffness`    | no       | `false` | Also report the local-harmonic cage stiffness
`file`         | no       |         | Streaming output file (see [Output file formats](#output-file-formats))
`frequency`    | yes      |         | Sample frequency, e.g. `!Every 100`

### Output

The YAML output includes `W`, the mean excess interaction, the orientational
entropy, `N_eff`, and the per-vector and mean `S²`, each with a statistical error.
Snapshots in which the molecule clashes in every trial orientation are excluded
from these averages and counted under `num_inaccessible`. The RMS torque
(`rms_torque`) and stiffness (`local_harmonic_stiffness`) are added when enabled;
an axis with no resolved samples reports null. Errors use the same hierarchical
blocking fallback as above: the coarsest level with at least 16 blocks wins, and
short runs fall back to the finest level. If `file` is given, each sampled step
writes columns `step`, `W/kJ/mol`, `mean_S2`, and `N_eff`.

Errors for `W`, the mean interaction, entropy, `N_eff`, torque, and stiffness use
automatic hierarchical blocking with at least 16 blocks at the selected level. The
`S²` errors instead retain the joint covariance between tensor components and vectors.

`W`, the mean excess interaction, and the stiffness are in kJ/mol, the torque is
in kT per radian, and the entropy and `S²` are dimensionless. Each output key
names its own unit.

---

## Rerun

The `rerun` subcommand replays a trajectory through a (possibly different) Hamiltonian,
running the analysis pipeline on each frame. This decouples analysis from propagation,
enabling e.g. comparison of explicit nonbonded energies against tabulated 6D potentials
for the same configurations.

The input YAML provides the Hamiltonian and analysis configuration;
the `propagate:` section is ignored. All analysis frequencies are overridden to
sample every frame.

### Fluctuating cells

Each frame is evaluated in the cell it was generated in, read from the box stored in the trajectory,
so a constant-pressure trajectory reruns at its own volumes rather than at the volume declared in
the input. A trajectory records an orthorhombic box, which determines a `Cuboid`, a `Slit`, a
`Sphere` and a `Cylinder`. A `HexagonalPrism` is written to a trajectory as an expanded orthorhombic
supercell and cannot be rerun. An `Endless` cell has no box, so it is left as the input declares it.

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
        resolution: 0.1
        frequency: !Every 100
    ```

2. Rerun with a different Hamiltonian (e.g. 6D tabulated potential):

    ```sh
    faunus rerun -i input_6dtable.yaml --traj traj.xtc -o output_rerun.yaml
    ```

    where `input_6dtable.yaml` uses the same topology but a different energy section,
    and defines the desired analysis objects (e.g. RDF, energy time series).

3. Compare the RDFs from the original and rerun outputs.
