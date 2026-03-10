# Topology

The topology defines the static structure of the simulation system: atom types, molecule types,
bonds, and how molecules are placed in the simulation cell.
It does not include dynamic state such as positions or velocities.

## Atoms

Atom types define properties shared by all atoms of the same kind.
An atom need not be a chemical element — it can represent any interaction site.

```yaml
atoms:
  - name: OW
    mass: 16.0
    charge: -1.0
    element: O
    sigma: 3.4
    epsilon: 1.8
    hydrophobicity: !SurfaceTension 1.0
  - name: HW
    mass: 1.0
    sigma: 1.0
    eps: 0.5
```

| Key              | Required | Default | Description                              |
|------------------|----------|---------|------------------------------------------|
| `name`           | yes      |         | Unique atom type name                    |
| `mass`           | no       | 0       | Mass (g/mol)                             |
| `charge`         | no       | 0       | Charge (elementary charges)              |
| `element`        | no       |         | Chemical symbol (e.g., `O`, `C`, `He`)   |
| `sigma` / `σ`    | no       |         | Lennard-Jones diameter (Å)               |
| `epsilon` / `ε` / `eps` | no |      | Lennard-Jones well depth (kJ/mol)        |
| `hydrophobicity` | no       |         | See below                                |
| `activity`       | no       |         | Activity for implicit species (molar)    |
| `custom`         | no       | `{}`    | Arbitrary key-value properties           |

### Hydrophobicity

| Variant              | Example                      | Description                      |
|----------------------|------------------------------|----------------------------------|
| `Hydrophobic`        | `hydrophobicity: Hydrophobic`| Flag as hydrophobic              |
| `Hydrophilic`        | `hydrophobicity: Hydrophilic`| Flag as hydrophilic              |
| `!SurfaceTension`    | `!SurfaceTension 1.0`       | Surface tension (kJ/mol/Å²)     |
| `!Lambda` / `!λ`     | `!Lambda 0.5`               | Ashbaugh-Hatch scaling factor    |

## Molecules

A molecule is a collection of atoms, optionally connected by bonds, torsions, and dihedrals.

```yaml
molecules:
  - name: water
    atoms: [OW, HW, HW]
    bonds:
      - {index: [0, 1], kind: !Harmonic {k: 100.0, req: 1.0}}
      - {index: [0, 2], kind: !Harmonic {k: 100.0, req: 1.0}}
    torsions:
      - {index: [1, 0, 2], kind: !Harmonic {k: 50.0, aeq: 109.47}}
    excluded_neighbours: 1
    degrees_of_freedom: Free
```

| Key                   | Required | Default | Description                                      |
|-----------------------|----------|---------|--------------------------------------------------|
| `name`                | yes      |         | Unique molecule type name                        |
| `atoms`               | no       | `[]`    | List of atom type names                          |
| `from_structure`      | no       |         | Load atoms from a structure file (XYZ, PDB, etc.)|
| `fasta`               | no       |         | Build from FASTA sequence (see below)            |
| `bonds`               | no       | `[]`    | Intramolecular bonds                             |
| `torsions`            | no       | `[]`    | Three-body angle potentials                      |
| `dihedrals`           | no       | `[]`    | Four-body dihedral potentials                    |
| `excluded_neighbours` | no       | 0       | Exclude nonbonded interactions within _n_ bonds  |
| `exclusions`          | no       | `[]`    | Manual atom pair exclusions, e.g. `[[0, 4]]`     |
| `keep_excluded_coulomb` | no     | `false` | Add back Coulomb for excluded pairs (see [energy](energy.md#excluded-pair-coulomb-correction)) |
| `degrees_of_freedom`  | no       | `Free`  | See below                                        |
| `atom_names`          | no       | `[]`    | Per-atom names (use `null` to skip)              |
| `residues`            | no       | `[]`    | Protein residues                                 |
| `chains`              | no       | `[]`    | Protein chains                                   |
| `activity`            | no       |         | Activity for GCMC fugacity (molar)               |
| `has_com`             | no       | `true`  | Whether center-of-mass makes sense               |
| `atomic`              | no       | `false` | Pool all instances into a single group (see below)|
| `custom`              | no       | `{}`    | Arbitrary key-value properties                   |

### FASTA Sequence

Linear, flexible peptides can be built from a [FASTA](https://doi.org/10.1073/pnas.85.8.2444) sequence string.
Each letter is mapped to a three-letter atom type name following
[IUPAC-IUB conventions](https://doi.org/10.1111/j.1432-1033.1984.tb07877.x),
and consecutive residues are connected by harmonic bonds.
The atom types must be defined in the `atoms` section (or an included force field file).

```yaml
molecules:
  - name: peptide
    fasta:
      sequence: "nAGGKRc"
      k: 80.33       # harmonic bond force constant (kJ/mol/Å²)
      req: 3.8       # equilibrium bond distance (Å)
    excluded_neighbours: 1
```

Uppercase letters follow the standard
[IUPAC one-letter codes](https://doi.org/10.1111/j.1432-1033.1984.tb07877.x)
for the 20 amino acids.
In addition, the following Faunus-specific lowercase codes are supported:

| FASTA letter | Atom type | Description |
|:------------:|-----------|-------------|
| `n`          | NTR       | N-terminus  |
| `c`          | CTR       | C-terminus  |
| `a`          | ANK       | Ankyrin repeat |

Whitespace in the sequence is ignored and `*` terminates parsing.
The `fasta` field is mutually exclusive with `atoms` and `from_structure`.

If `sequence` ends with `.fasta`, it is read as a [FASTA file](https://doi.org/10.1073/pnas.85.8.2444)
where header lines (`>`) and comment lines (`;`) are skipped:

```yaml
molecules:
  - name: my_protein
    fasta: {sequence: "my_protein.fasta", k: 80.33, req: 3.8}
```

For the [CALVADOS](https://doi.org/10.1073/pnas.2111696119) coarse-grained model,
use `k: 80.33` and `req: 3.8` together with the `calvados3.yaml` force field:

```yaml
include: [calvados3.yaml]
molecules:
  - name: histatin5
    fasta: {sequence: "DSHAKRHHGYKRKFHEKHHSHRGY", k: 80.33, req: 3.8}
    excluded_neighbours: 1
system:
  cell: !Cuboid [200.0, 200.0, 200.0]
  blocks:
    - molecule: histatin5
      N: 1
      insert: !RandomWalk {bond_length: 3.8}
```

### Degrees of Freedom

| Value              | Description                                          |
|--------------------|------------------------------------------------------|
| `Free`             | All degrees of freedom are active (default)          |
| `Frozen`           | All degrees of freedom are frozen                    |
| `Rigid`            | Rigid body — only translations and rotations; all intra-molecular nonbonded pairs excluded |
| `RigidAlchemical`  | Like `Rigid` but with free alchemical degrees of freedom (charges, etc.); use with `keep_excluded_coulomb` for titration |

### Atomic Molecules

Setting `atomic: true` pools all instances of a single-atom molecule into one group
instead of creating one group per atom.
This reduces per-group overhead and enables O(1) group-pair iteration for the energy.
The molecule must have exactly one atom type and no bonds.

```yaml
molecules:
  - name: Na
    atoms: [Na]
    atomic: true
  - name: Cl
    atoms: [Cl]
    atomic: true
```

When `atomic: true`:
- `has_com` is forced to `false` (no center-of-mass).
- GCMC inserts/deletes individual atoms within the group via expand/shrink.
- `TranslateAtom` works as usual; `TranslateMolecule` is not allowed.
- The `N` field in `blocks` sets the group capacity (maximum atom count).

### Bonds

Bonds connect pairs of atoms within a molecule.

```yaml
bonds:
  - {index: [0, 1], kind: !Harmonic {k: 100.0, req: 1.0}, order: Single}
  - {index: [1, 2], kind: !FENE {k: 25.0, req: 1.5, rmax: 5.0}}
  - {index: [2, 3], kind: !Morse {k: 100.0, req: 1.0, d: 10.0}}
  - {index: [3, 4]}  # unspecified kind (topology only, no energy)
```

| Bond kind       | Parameters            | Description                           |
|-----------------|-----------------------|---------------------------------------|
| `!Harmonic`     | `k`, `req`            | Harmonic spring potential             |
| `!FENE`         | `k`, `req`, `rmax`    | Finitely extensible nonlinear elastic |
| `!Morse`        | `k`, `req`, `d`       | Anharmonic Morse potential            |
| `!UreyBradley`  | `k`, `req`            | Urey-Bradley potential                |

| Parameter | Unit        |
|-----------|-------------|
| `k`       | kJ/mol/Å²   |
| `req`     | Å           |
| `rmax`    | Å           |
| `d`       | kJ/mol      |

Bond order can optionally be specified: `Single`, `Double`, `Triple`, `Aromatic`, etc.

### Torsions (Three-Body Angles)

```yaml
torsions:
  - {index: [0, 1, 2], kind: !Harmonic {k: 50.0, aeq: 109.47}}
  - {index: [1, 2, 3], kind: !Cosine {k: 50.0, aeq: 45.0}}
```

| Torsion kind | Parameters  | Description                |
|--------------|-------------|----------------------------|
| `!Harmonic`  | `k`, `aeq`  | Harmonic angle bending     |
| `!Cosine`    | `k`, `aeq`  | Cosine form (GROMOS-96)    |

| Parameter | Unit        |
|-----------|-------------|
| `k`       | kJ/mol/deg² |
| `aeq`     | degrees     |

### Dihedrals (Four-Body Angles)

```yaml
dihedrals:
  - index: [0, 1, 2, 3]
    kind: !ProperHarmonic {k: 100.0, aeq: 180.0}
    electrostatic_scaling: 0.5
    lj_scaling: 0.5
  - {index: [0, 1, 2, 3], kind: !ProperPeriodic {k: 10.0, n: 3, phi: 0.0}}
  - {index: [0, 1, 2, 3], kind: !ImproperHarmonic {k: 100.0, aeq: 90.0}}
```

| Dihedral kind        | Parameters       | Description                      |
|----------------------|------------------|----------------------------------|
| `!ProperHarmonic`    | `k`, `aeq`       | Proper harmonic dihedral         |
| `!ProperPeriodic`    | `k`, `n`, `phi`  | Proper periodic dihedral         |
| `!ImproperHarmonic`  | `k`, `aeq`       | Improper harmonic dihedral       |
| `!ImproperPeriodic`  | `k`, `n`, `phi`  | Improper periodic dihedral       |

| Parameter | Unit                                    |
|-----------|-----------------------------------------|
| `k`       | kJ/mol/deg² (harmonic) or kJ/mol (periodic) |
| `aeq`     | degrees                                 |
| `n`       | integer                                 |
| `phi`     | degrees                                 |

Optional 1-4 scaling factors: `electrostatic_scaling` and `lj_scaling`.

### Residues and Chains

For protein structures, residues and chains describe contiguous atom ranges:

```yaml
residues:
  - {name: ALA, number: 2, range: [0, 3]}
  - {name: GLY, number: 3, range: [3, 6]}
chains:
  - {name: A, range: [0, 50]}
```

Ranges are half-open intervals `[start, end)`. Residues and chains must not overlap.

## System

The `system` section defines the simulation cell, molecule blocks, and optionally intermolecular bonded interactions.

```yaml
system:
  medium:
    permittivity: !Vacuum
    temperature: 298.15

  cell: !Cuboid [30.0, 30.0, 30.0]

  blocks:
    - {molecule: water, N: 256, insert: !RandomCOM {filename: water.xyz, rotate: true}}
    - {molecule: Na, N: 10, insert: !RandomAtomPos {}}
```

### Simulation Cell

The `cell` field selects the geometry and boundary conditions.

| Cell | PBC | Parameters | Description |
|------|-----|------------|-------------|
| `!Cuboid` | XYZ | `[x, y, z]` | Orthorhombic box with side lengths x, y, z |
| `!HexagonalPrism` | XYZ | `{side, height}` | Hexagonal cross-section prism |
| `!Slit` | XY | `[x, y, z]` | Cuboidal box, hard walls in Z |
| `!Cylinder` | Z | `{radius, height}` | Cylindrical cell, hard walls in XY |
| `!Sphere` | — | `{radius}` | Spherical cell, hard walls |
| `!Endless` | — | | Infinite, open cell |

Examples:

```yaml
cell: !Cuboid [30.0, 30.0, 30.0]
cell: !HexagonalPrism {side: 15.0, height: 30.0}
cell: !Cylinder {radius: 10.0, height: 50.0}
```

### Molecule Blocks

Blocks specify how many copies of each molecule to create and how to initialize their positions.

| Key        | Required | Default | Description                                       |
|------------|----------|---------|---------------------------------------------------|
| `molecule` | yes      |         | Name of the molecule type                         |
| `N`        | yes      |         | Number of molecules                               |
| `active`   | no       | all     | Number of initially active molecules              |
| `insert`   | no       |         | Insertion policy (see below)                      |

### Insertion Policies

| Policy           | Example                                              | Description                                    |
|------------------|------------------------------------------------------|------------------------------------------------|
| `!RandomCOM`     | `{filename: mol.xyz, rotate: true, min_distance: 2.0}` | Random center-of-mass placement              |
| `!RandomAtomPos` | `{directions: xy}`                                   | Random position per atom                       |
| `!FixedCOM`      | `{filename: mol.xyz, position: [0, 0, 0]}`           | Place at specific position                     |
| `!FromFile`      | `structure.xyz`                                      | Read all positions from file                   |
| `!Manual`        | `[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]`                | Explicit coordinates for all atoms             |
| `!RandomWalk`    | `{bond_length: 3.8}`                                 | Self-avoiding walk; respects cell boundaries   |
| `!GridCOM`       | `{filename: mol.xyz, rotate: true}`                  | Simple cubic grid; cuboidal cells only         |

The `directions` field controls which axes are randomized: `xyz` (default), `xy`, `xz`, `yz`, `x`, `y`, or `z`.
The optional `min_distance` field (Å) in `!RandomCOM` uses bounding sphere rejection to avoid molecular overlaps in dense systems.
`!GridCOM` places molecules on a simple cubic lattice with auto-calculated spacing; requires `!Cuboid` or `!Slit` cells.

### Intermolecular Bonded Interactions

Bonds, torsions, and dihedrals between atoms in _different_ molecules use global atom indices:

```yaml
system:
  intermolecular:
    bonds:
      - {index: [0, 220], kind: !Harmonic {k: 50.0, req: 3.0}}
    torsions:
      - {index: [1, 75, 128], kind: !Harmonic {k: 100.0, aeq: 120.0}}
    dihedrals:
      - {index: [1, 35, 75, 128], kind: !ProperHarmonic {k: 27.5, aeq: 105.0}}
```

## Including Files

Topology files can include other YAML files. Paths are relative to the including file.
Definitions in later files take precedence; definitions in the main file take precedence over all includes.

```yaml
include: [forcefield.yaml, overrides.yaml]
```

## Chemical Reactions

Chemical reactions are used for [speciation moves](moves.md#speciation-move-reaction-ensemble)
in the grand canonical and semi-grand canonical ensembles.
A participant is either an atom, a molecule, or an implicit participant.
When parsing a reaction, atoms are prefixed with a dot or an atom sign, e.g. _.Na_ or _⚛Na_.
Implicit participants are prefixed with a tilde or a ghost, e.g. _~H_ or _👻H_.
Molecules are not prefixed, e.g. _Cl_.

Participant | Example                |  Notes
------------|----------------------- | ------------------------------------
Molecular   | `A + A ⇌ D`            | Possible arrows: `=`, `⇌`, `⇄`, `→`
Implicit    | `RCOO- + 👻H+ ⇌ RCOOH` | Mark with `👻` or `~`
Atomic      | `⚛Pb ⇄ ⚛Au`            | Mark with `⚛` or `.`

When a reaction has one molecular reactant and one molecular product with equal
atom counts (e.g. `A = B + ~H+`), it is treated as a molecular swap:
the source group is deactivated and a target group is activated with aligned positions.
Unicode names are supported in reactions and molecule definitions.
