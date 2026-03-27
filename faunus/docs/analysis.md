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

The column-data format is inferred from the file extension:

Extension     | Format
------------- | -------------------------------------------
`.dat`        | Space-separated; header prefixed with `# `
`.csv`/`.tsv` | Comma/Tab-separated; plain header row (column labels)

Both formats support transparent gzip compression by appending `.gz`
(e.g. `energy.dat.gz`).

Loading CSV output in Python:

```python
import numpy as np
data = np.loadtxt("data.csv.gz", delimiter=",", skiprows=1)

import pandas as pd
data = pd.read_csv("data.csv.gz").to_numpy()
```

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
`atom_position`          | one atom        | Signed component for single axis (`x`,`y`,`z`); Euclidean norm for multi-axis (`xy` etc.)
`count`                  | atoms or groups | Number of active atoms matching selection
`molarity`               | atoms or groups | Molar concentration (mol/L) of matching atoms
`charge`                 | atoms or groups | Sum of charges of active atoms matching selection
`size`                   | one group       | Number of active particles in a group
`end_to_end`             | one group       | End-to-end distance of a molecular group
`gyration_radius`        | one group       | Radius of gyration (default `xyz` = full Rg; single axis gives spread along it)
`dipole_moment`          | one group       | Electric dipole moment (default `xyz` = magnitude; single axis gives signed component)
`mass_center_position`   | one group       | Mass center position along `projection`
`mass_center_separation` | two groups      | Distance between two group mass centers
`dipole_product`         | two groups      | Normalized dipole dot product μ̂₁·μ̂₂ = cos(θ) (default `xyz` = full 3D; `projection` filters dipoles before comparing)

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

## Trajectory

Writes particle coordinates to a trajectory file at each sampled step.
The format is auto-detected from the file extension (`.xyz` or `.xtc`).

On finalization a companion PSF topology file and a VMD scene script are
written alongside the trajectory. The PSF contains atoms, bonds, angles,
and dihedrals; the Tcl script loads the PSF and trajectory, sets VDW radii
from sigma values, and draws the periodic box.

### Example

```yaml
analysis:
  - !Trajectory
    file: traj.xtc
    frequency: !Every 100
    save_frame_state: true
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
`save_frame_state` | no       | `false` | Write a binary `.aux` file alongside the trajectory (see [Rerun](#rerun))

### Output files

Given `file: traj.xtc`, the following files are produced:

File        | Description
----------- | -------------------------------------------
`traj.xtc`  | Trajectory (coordinates per frame)
`traj.psf`  | X-PLOR PSF topology (atoms, bonds, angles, dihedrals, charges, masses)
`traj.tcl`  | VMD scene script (`vmd -e traj.tcl` loads everything)
`traj.aux`  | Frame state file (only when `save_frame_state: true`)

### Frame state file (`.aux`)

XTC stores only atom positions. Rigid-body simulations also need
quaternions, mass centers, and group sizes (for GC moves); swap moves change
`atom_id`. When `save_frame_state: true`, a binary `.aux` file is written
alongside each XTC frame, storing this per-frame microstate data.

The `.aux` file is required by `faunus rerun` (see [Rerun](#rerun)) to fully
reconstruct the simulation state from each trajectory frame.

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

## Virtual Volume Move

Performs a virtual volume perturbation and measures the excess pressure
by Widom perturbation ([doi:10.1063/1.472721](https://doi.org/10.1063/1.472721)):

$$P_{\text{ex}} = \frac{k_BT \ln\langle e^{-\Delta U / k_BT}\rangle}{\delta V}$$

where $\Delta U$ is the energy change due to the volume displacement $\delta V$.
All particle positions are scaled according to the chosen `method`.

### Example

```yaml
analysis:
  - !VirtualVolumeMove
    dV: 0.2
    method: Isotropic
    frequency: !Every 10
```

### Options

Key           | Required | Default      | Description
------------- | -------- | ------------ | -------------------------------------------
`dV`          | yes      |              | Volume displacement (ų)
`method`      | no       | `Isotropic`  | Scaling policy: `Isotropic`, `ScaleZ`, `ScaleXY`
`frequency`   | yes      |              | Sample frequency, e.g. `!Every 10`

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
