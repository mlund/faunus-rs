# Moves

Monte Carlo moves are defined in the `propagate` section of the YAML input file.
Moves are grouped into collections that are either stochastic (one move randomly selected per repeat)
or deterministic (all moves executed per repeat).
Within a stochastic collection, moves are selected with probability proportional to their `weight`.

```yaml
propagate:
  seed: Hardware
  criterion: Metropolis
  repeat: 1000
  collections:
    - !Stochastic
      repeat: 10
      moves:
        - !TranslateMolecule { molecule: Water, dp: 0.5, weight: 1.0 }
        - !RotateMolecule { molecule: Water, dp: 0.3, weight: 1.0 }
        - !VolumeMove { dV: 0.04, weight: 0.5 }
    - !Deterministic
      repeat: 1
      moves:
        - !TranslateAtom { molecule: Water, atom: O, dp: 0.1, weight: 1.0 }
```

## Translate Molecule

Picks a random molecule of the given type and translates it by a random displacement vector
with magnitude uniformly sampled in $[-\text{dp}, +\text{dp}]$.

```yaml
- !TranslateMolecule { molecule: Water, dp: 0.5, weight: 1.0 }
- !TranslateMolecule { molecule: Protein, dp: 0.2, weight: 2.0, repeat: 3, directions: xy }
```

Key          | Required | Default | Description
------------ | -------- | ------- | -------------------------------------------
`molecule`   | yes      |         | Name of the molecule type
`dp`         | yes      |         | Maximum displacement (Angstrom)
`weight`     | yes      |         | Selection weight
`repeat`     | no       | 1       | Repetitions per selection
`directions` | no       | `xyz`   | Active directions (`x`, `y`, `z`, `xy`, `xz`, `yz`, `xyz`)

## Translate Atom

Picks a random atom and translates it by a random displacement.
If `molecule` is specified, the atom is chosen from that molecule type only.
If `atom` is specified, only atoms of that type are selected.

```yaml
- !TranslateAtom { dp: 0.1, weight: 1.0 }
- !TranslateAtom { molecule: Water, atom: O, dp: 0.2, weight: 1.0, repeat: 5 }
```

Key        | Required | Default | Description
---------- | -------- | ------- | -------------------------------------------
`dp`       | yes      |         | Maximum displacement (Angstrom)
`weight`   | yes      |         | Selection weight
`molecule` | no       |         | Restrict to this molecule type
`atom`     | no       |         | Restrict to this atom type
`repeat`   | no       | 1       | Repetitions per selection

## Rotate Molecule

Picks a random molecule of the given type and rotates it around a random axis
by an angle uniformly sampled in $[-\text{dp}, +\text{dp}]$ (radians).

```yaml
- !RotateMolecule { molecule: Protein, dp: 0.3, weight: 1.0 }
```

Key        | Required | Default | Description
---------- | -------- | ------- | -------------------------------------------
`molecule` | yes      |         | Name of the molecule type
`dp`       | yes      |         | Maximum angular displacement (radians)
`weight`   | yes      |         | Selection weight
`repeat`   | no       | 1       | Repetitions per selection

## Pivot Move

Picks a random atom as pivot in a polymer chain, randomly selects a bonded direction,
and rotates the connected sub-tree around the pivot position.
Uses the bond graph from the molecule topology, so it works for arbitrary
topologies (linear, branched, star, dendrimer).
Molecules without bonds are skipped.

See [Madras & Sokal, _J. Stat. Phys._ 50, 109–186 (1988)](https://doi.org/10.1007/BF01022990).

> **Note:** The rotated sub-tree must fit within half the box length (L/2).
> For molecules spanning more than L/2, the minimum-image convention used
> during rotation can map atoms to incorrect periodic images.

```yaml
- !PivotMove { molecule: Polymer, dp: 1.5, weight: 1.0 }
```

Key        | Required | Default | Description
---------- | -------- | ------- | -------------------------------------------
`molecule` | yes      |         | Name of the molecule type
`dp`       | yes      |         | Maximum angular displacement (radians)
`weight`   | yes      |         | Selection weight
`repeat`   | no       | 1       | Repetitions per selection

## Crankshaft Move

Picks a random proper dihedral in the molecule and rotates the smaller sub-tree
around the middle bond vector (the dihedral axis) by an angle uniformly sampled
in $[-\text{dp}, +\text{dp}]$ (radians).
This preserves bond lengths and angles by construction and is well suited
for sampling internal degrees of freedom in peptides and other molecules with
defined dihedral angles.
Molecules without proper dihedrals are skipped.

```yaml
- !CrankshaftMove { molecule: Peptide, dp: 0.5, weight: 1.0 }
```

Key        | Required | Default | Description
---------- | -------- | ------- | -------------------------------------------
`molecule` | yes      |         | Name of the molecule type
`dp`       | yes      |         | Maximum angular displacement (radians)
`weight`   | yes      |         | Selection weight
`repeat`   | no       | 1       | Repetitions per selection

## Volume Move (NPT)

Proposes isotropic or anisotropic volume changes for the NPT ensemble.
The volume is sampled logarithmically:

$$
V_\text{new} = \exp\!\bigl(\ln V_\text{old} + (\xi - 0.5) \cdot \text{dV}\bigr)
$$

where $\xi$ is a uniform random number in $[0, 1)$.

No move-level bias is applied; the acceptance is handled by standard Metropolis
sampling together with the [`isobaric`](energy.md#external-pressure-isobaric) energy term
which contributes $PV - (N+1) k_BT \ln V$.

### Example

```yaml
energy:
  isobaric:
    P/atm: 1.0

propagate:
  repeat: 10000
  collections:
    - !Stochastic
      repeat: 100
      moves:
        - !TranslateMolecule { molecule: Water, dp: 0.5, weight: 1.0 }
        - !VolumeMove { dV: 0.04, weight: 0.5 }
```

### Options

Key      | Required | Default      | Description
-------- | -------- | ------------ | -------------------------------------------
`dV`     | yes      |              | Volume displacement parameter (log-scale)
`weight` | yes      |              | Selection weight
`method` | no       | `Isotropic`  | Scaling policy (see table below)
`repeat` | no       | 1            | Repetitions per selection

### Scaling policies

Policy       | Description
------------ | -------------------------------------------
`Isotropic`  | Equal scaling in all directions (default)
`ScaleZ`     | Scale along the z-axis only
`ScaleXY`    | Scale the xy-plane only
`IsochoricZ` | Scale z and xy at constant total volume

### Anisotropic example

```yaml
- !VolumeMove { dV: 0.05, method: ScaleZ, weight: 0.5, repeat: 2 }
```
