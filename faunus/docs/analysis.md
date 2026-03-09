# Analysis

Analysis objects sample the system at a given frequency during a simulation
and optionally write results to output files.
They are defined in the `analysis` section of the YAML input.

## Collective Variable

Monitors a collective variable (CV) over the course of a simulation,
recording the instantaneous value and a running average at each sampled step.
This is useful for verifying that a CV remains within the expected range,
for convergence checking, and for post-processing of time/step series data.

If `file` is given, each sampled step writes a line with columns
`step`, `value`, and `running_average`.
The file may be gzip-compressed by using a `.gz` extension.

### Example

```yaml
analysis:
  - !CollectiveVariable
    property: mass_center_position
    selection: "molecule protein"
    dimension: z
    range: [-50.0, 50.0]
    file: cv.dat
    frequency: !Every 100
```

### Options

Key          | Required | Default | Description
------------ | -------- | ------- | -------------------------------------------
`property`   | yes      |         | CV type (see table below)
`range`      | yes      |         | Allowed `[min, max]` interval
`frequency`  | yes      |         | Sample frequency, e.g. `!Every 100`
`dimension`  | no       | `xyz`   | Axis projection (`x`, `y`, `z`, `xy`, …)
`selection`  | depends  |         | Selection expression for one atom or group
`selection2` | depends  |         | Second selection (for two-group properties)
`resolution` | no       |         | Bin width (only used by Penalty)
`file`       | no       |         | Output file path; omit to only track the mean

### Supported properties

Property                 | Selection       | Description
------------------------ | --------------- | -------------------------------------------
`volume`                 | none            | Simulation cell volume
`box_length`             | none            | Cell side length along `dimension`
`atom_position`          | one atom        | Atom position projected onto `dimension`
`count`                  | atoms or groups | Number of active atoms matching selection
`charge`                 | atoms or groups | Sum of charges of active atoms matching selection
`size`                   | one group       | Number of active particles in a group
`end_to_end`             | one group       | End-to-end distance of a molecular group
`mass_center_position`   | one group       | Mass center position along `dimension`
`mass_center_separation` | two groups      | Distance between two group mass centers

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
The file may be gzip-compressed by using a `.gz` extension.

### Example

```yaml
analysis:
  - !PolymerShape
    selection: "molecule polymer"
    file: shape.dat.gz
    frequency: !Every 100
```

### Options

Key          | Required | Default | Description
------------ | -------- | ------- | -------------------------------------------
`selection`  | yes      |         | Selection expression for molecule group(s)
`file`       | no       |         | Streaming output file (single molecule only)
`frequency`  | yes      |         | Sample frequency, e.g. `!Every 100`

## Energy

Streams energy values to a file at each sampled step.
Two modes are supported:

- **Total** (default): writes every Hamiltonian term plus the total.
  Output columns: `step term1 term2 ... total`.
- **Partial**: writes the nonbonded energy between two sets of atoms
  selected with VMD-like expressions.
  Output columns: `step energy running_average`.

The file may be gzip-compressed by using a `.gz` extension.

### Examples

```yaml
analysis:
  # Total energy with per-term breakdown
  - !Energy
    file: energy.dat.gz
    frequency: !Every 100

  # Nonbonded energy between two molecules
  - !Energy
    file: mol1_mol2_energy.dat.gz
    frequency: !Every 100
    selections: ["molecule MOL1", "molecule MOL2"]

  # Nonbonded energy between hydrophobic atoms in two molecules
  - !Energy
    file: hydrophobic_energy.dat.gz
    frequency: !Every 100
    selections: ["hydrophobic and molecule MOL1", "hydrophobic and molecule MOL2"]
```

### Options

Key            | Required | Default | Description
-------------- | -------- | ------- | -------------------------------------------
`file`         | yes      |         | Output file path (`.gz` for gzip)
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
The format is auto-detected from the file extension (`.xyz`, `.xtc`, or
other formats via the `chemfiles` feature).

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
```

After the simulation, visualize with:

```sh
vmd -e traj.tcl
```

### Options

Key         | Required | Default | Description
----------- | -------- | ------- | -------------------------------------------
`file`      | yes      |         | Output file path (`.xyz`, `.xtc`, etc.)
`frequency` | yes      |         | Sample frequency, e.g. `!Every 100` or `!End`

### Output files

Given `file: traj.xtc`, the following files are produced:

File        | Description
----------- | -------------------------------------------
`traj.xtc`  | Trajectory (coordinates per frame)
`traj.psf`  | X-PLOR PSF topology (atoms, bonds, angles, dihedrals, charges, masses)
`traj.tcl`  | VMD scene script (`vmd -e traj.tcl` loads everything)

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
`file`                     | yes      |                       | Output file path (`.gz` for gzip)
`dr`                       | yes      |                       | Bin width in distance units
`frequency`                | yes      |                       | Sample frequency, e.g. `!Every 100`
`max_r`                    | no       | half shortest box dim | Maximum distance for histogram
`use_com`                  | no       | `false`               | Use center-of-mass distances instead of atom-atom
`exclude_intramolecular`   | no       | `true` (atom-atom)    | Skip pairs within the same molecule (atom-atom only)
`dimension`                | no       | `xyz`                 | Dimension for distance projection and normalization (`x`, `y`, `z`, `xy`, …)
