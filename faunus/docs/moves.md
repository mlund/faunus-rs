# Moves

Monte Carlo moves are defined in the `propagate` section of the YAML input file.

## Propagate settings

Key           | Required | Default              | Description
------------- | -------- | -------------------- | -------------------------------------------
`repeat`      | no       | 1                    | Number of propagate cycles (outer loop)
`collections` | no       | `[]`                 | Ordered list of move collections (see below)
`seed`        | no       | `Hardware`           | RNG seed: `Hardware` (random) or `!Fixed N`
`criterion`   | no       | `MetropolisHastings` | Acceptance criterion (`Metropolis` or `Minimize`)

Each propagate cycle executes every collection in order; after all collections
have run, one cycle is complete and the step counter advances.

## Collections

A collection groups moves together and controls how they are selected.
Two types are available:

- `!Stochastic` — each repeat draws *one* move at random, with probability
  proportional to `weight`. This is the normal Monte Carlo sampling mode.
- `!Deterministic` — each repeat executes *all* moves in order.
  Useful for e.g. hybrid MC/MD schemes. The `weight` field is unused here but
  still required by the move schema.

Key      | Required | Default | Description
-------- | -------- | ------- | -------------------------------------------
`repeat` | no       | 1       | How many times the collection runs per propagate cycle
`moves`  | no       | `[]`    | List of moves

## Common move options

Every move accepts at least:

Key      | Description
-------- | -------------------------------------------
`weight` | Selection weight (only meaningful inside `!Stochastic` collections)
`repeat` | How many trial moves to attempt *each time the move is selected* (default 1)
`max_displacement` | Maximum translational displacement (Å), for translational moves (alias `dp`)
`max_angle`        | Maximum rotational angle (radians), for rotational moves (alias `dprot`)

The two `repeat` levels nest: the collection's `repeat` controls how often
moves are drawn, and the move's `repeat` controls how many trials are
performed per draw. For example, a stochastic collection with `repeat: 100`
containing a move with `repeat: 5` will attempt that move up to 500 times
per propagate cycle (depending on how often it is drawn).

## Example

```yaml
propagate:
  seed: Hardware
  criterion: Metropolis
  repeat: 1000
  collections:
    - !Stochastic
      repeat: 10
      moves:
        - !TranslateMolecule { molecule: Water, max_displacement: 0.5, weight: 1.0 }
        - !RotateMolecule { molecule: Water, max_angle: 0.3, weight: 1.0 }
        - !VolumeMove { volume_displacement: 0.04, weight: 0.5 }
    - !Deterministic
      repeat: 1
      moves:
        - !TranslateAtom { molecule: Water, atom: O, max_displacement: 0.1, weight: 1.0 }
```

---

## Translate Molecule

Picks a random molecule of the given type and translates it by a random displacement vector
with magnitude uniformly sampled in $[-\Delta r, +\Delta r]$, set by `max_displacement`.

```yaml
- !TranslateMolecule { molecule: Water, max_displacement: 0.5, weight: 1.0 }
- !TranslateMolecule { molecule: Protein, max_displacement: 0.2, weight: 2.0, repeat: 3, directions: xy }
```

Key          | Required | Default | Description
------------ | -------- | ------- | -------------------------------------------
`molecule`   | yes      |         | Name of the molecule type (not allowed for `atomic` molecules)
`max_displacement` | yes |    | Maximum displacement (Å); alias `dp`
`weight`     | yes      |         | Selection weight
`repeat`     | no       | 1       | Repetitions per selection
`directions` | no       | `xyz`   | Active directions (`x`, `y`, `z`, `xy`, `xz`, `yz`, `xyz`)

---

## Translate Atom

Picks a random atom and translates it by a random displacement.
If `molecule` is specified, the atom is chosen from that molecule type only.
If `atom` is specified, only atoms of that type are selected.

```yaml
- !TranslateAtom { max_displacement: 0.1, weight: 1.0 }
- !TranslateAtom { molecule: Water, atom: O, max_displacement: 0.2, weight: 1.0, repeat: 5 }
```

Key            | Required | Default | Description
-------------- | -------- | ------- | -------------------------------------------
`max_displacement` | yes |    | Maximum displacement (Å); alias `dp`
`weight`       | yes      |         | Selection weight
`molecule`     | no       |         | Restrict to this molecule type
`atom`         | no       |         | Restrict to this atom type
`repeat`       | no       | 1       | Repetitions per selection
`preferential` | no       |         | Preferential sampling settings (see below)

### Preferential Sampling

Biases atom selection toward reference group(s) using distance-dependent weights,
with a corresponding acceptance correction to maintain detailed balance.
Useful in dilute solutions where standard MC wastes most trial moves on bulk
particles far from the solute
([Owicki & Scheraga, 1977](https://doi.org/10.1016/0009-2614(77)85051-3);
[Allen & Tildesley, 2017](https://doi.org/10.1093/oso/9780198803195.001.0001), §9.3.1).

Each candidate atom receives weight $W'(r) = (r + \text{offset})^{-\nu}$
where $r$ is the nearest bounding-sphere distance across all matching reference groups.
The acceptance criterion includes a correction $\ln(W_\text{new} / W_\text{old})$
where $W = \sum_j W'(r_j)$ is the normalization sum over all candidates.

Must be placed in a `!Deterministic` block so that reference groups move first
(updating their positions), followed by the biased atom moves:

```yaml
- !Deterministic
  moves:
    - !TranslateMolecule { molecule: Protein, max_displacement: 0.5 }
    - !RotateMolecule { molecule: Protein, max_angle: 0.5 }
    - !TranslateAtom
        molecule: Na
        max_displacement: 0.5
        repeat: 100
        preferential:
          reference: "molecule Protein"
          exponent: 2
          offset: 1.0
```

Multiple reference groups can be selected; the distance is always to the nearest one:

```yaml
        preferential:
          reference: "molecule Protein1 or molecule Protein2"
```

The `reference` field uses the [selection language](selection_language.md),
e.g. `"molecule Protein"`, `"protein"`, or boolean combinations.

Key         | Required | Default | Description
----------- | -------- | ------- | -------------------------------------------
`reference` | yes      |         | Selection expression for reference group(s)
`exponent`  | no       | 2       | Exponent $\nu$ in the weight function
`offset`    | no       | 1.0     | Offset (Angstrom) to avoid singularity at $r = 0$
`file`      | no       |         | Path to write selection-distance histogram (.dat/.csv)

---

## Rotate Molecule

Picks a random molecule of the given type and rotates it around a random axis
by an angle uniformly sampled in $[-\theta_\text{max}, +\theta_\text{max}]$, set by `max_angle`.
Atomic molecules have no orientation to sample and are rejected.

```yaml
- !RotateMolecule { molecule: Protein, max_angle: 0.3, weight: 1.0 }
```

Key        | Required | Default | Description
---------- | -------- | ------- | -------------------------------------------
`molecule` | yes      |         | Name of the molecule type
`max_angle` | yes |    | Maximum angular displacement, in $(0, \pi]$ radians; alias `dprot`
`weight`   | yes      |         | Selection weight
`repeat`   | no       | 1       | Repetitions per selection

---

## Pivot Move

Picks a random atom as pivot in a polymer chain, randomly selects a bonded direction,
and rotates the connected sub-tree around the pivot position.
Uses the bond graph from the molecule topology, so it works for arbitrary
topologies (linear, branched, star, dendrimer).
The sub-tree is unwrapped along its bonds before rotation, so chains reaching further
than half the box length turn correctly.

See [Madras & Sokal, _J. Stat. Phys._ 50, 109–186 (1988)](https://doi.org/10.1007/BF01022990).

```yaml
- !PivotMove { molecule: Polymer, max_angle: 1.5, weight: 1.0 }
```

Key        | Required | Default | Description
---------- | -------- | ------- | -------------------------------------------
`molecule` | yes      |         | Name of the molecule type
`max_angle` | yes |    | Maximum angular displacement, in $(0, \pi]$ radians; alias `dprot`
`weight`   | yes      |         | Selection weight
`repeat`   | no       | 1       | Repetitions per selection

The molecule must be bonded and free to change shape; declaring it rigid or frozen, or
giving it no bonds, is an input error. Partially active molecules, as encountered under
grand-canonical or speciation moves, are skipped.

---

## Crankshaft Move

Picks a random bond axis in the molecule and rotates the smaller sub-tree
around it by an angle uniformly sampled in $[-\theta_\text{max}, +\theta_\text{max}]$, set by `max_angle`.
When proper dihedrals are defined, only their middle bonds are used as axes;
otherwise all bonds serve as candidate axes (e.g. FASTA chains with only
harmonic bonds). Terminal bonds are excluded, as no torsion is defined about them.

Because rotation is constrained to a single bond axis (1-DOF), crankshaft
preserves bond lengths and bond angles by construction. This makes it
complementary to [`PivotMove`](#pivot-move), which applies a full 3D rotation
(3-DOF) and can change bond angles at the pivot. For models without angle
potentials, `PivotMove` alone is typically sufficient.

```yaml
- !CrankshaftMove { molecule: Peptide, max_angle: 0.5, weight: 1.0 }
```

Key        | Required | Default | Description
---------- | -------- | ------- | -------------------------------------------
`molecule` | yes      |         | Name of the molecule type
`max_angle` | yes |    | Maximum angular displacement, in $(0, \pi]$ radians; alias `dprot`
`weight`   | yes      |         | Selection weight
`repeat`   | no       | 1       | Repetitions per selection

The restrictions listed for [`PivotMove`](#pivot-move) apply here too.

---

## Cluster Move

Picks a random seed molecule, grows a cluster of nearby molecules of the same type, and
translates and rotates the whole cluster as one rigid body. In strongly associating systems —
for example proteins with short-ranged attractions at high concentration — single-molecule moves
leave a molecule trapped by its neighbours, so aggregates diffuse only slowly and structural
observables such as $S(q)$ converge poorly. Moving whole clusters restores that diffusion.

Two molecules join the same cluster when their separation is below `threshold`. The separation is
measured between mass centers (`use_com: true`, the default) or between the closest pair of beads
(`use_com: false`); the latter makes the threshold a surface separation that transfers across
molecule sizes. Membership is transitive: the cluster grows outward from the seed until no further
molecule lies within the threshold.

Detailed balance is preserved by rejecting any move after which the cluster membership would
change, and by drawing the translation and rotation from symmetric distributions, so the move
samples the correct Boltzmann distribution. The cluster grows in unwrapped coordinates, so a
cluster that spans the periodic boundary still rotates correctly.

See [Dress & Krauth, _J. Phys. A_ 28, L597 (1995)](https://doi.org/10.1088/0305-4470/28/23/001) and
[Whitelam & Geissler, _J. Chem. Phys._ 127, 154101 (2007)](https://doi.org/10.1063/1.2790421).

```yaml
- !ClusterMove { molecule: Protein, max_displacement: 10.0, max_angle: 0.5, threshold: 35.0 }
```

Key         | Required | Default | Description
----------- | -------- | ------- | -------------------------------------------
`molecule`  | yes      |         | Name of the molecule type
`max_displacement` | yes |    | Maximum translational displacement (Å); alias `dp`
`max_angle` | yes |    | Maximum angular displacement (radians); alias `dprot`
`threshold` | yes      |         | Clustering distance (Å); mass-center or bead separation
`use_com`   | no       | `true`  | Cluster on mass-center (`true`) or closest-bead (`false`) distance
`weight`    | no       | 1       | Selection weight in a `!Stochastic` collection
`repeat`    | no       | 1       | Repetitions per selection

Output reports the mean cluster size $\langle N\rangle$ with its standard error, the fraction of
trials rejected to preserve membership (`bias_rejection_rate`), and the root-mean-square
translational and rotational displacements. For an ideal gas, $\langle N\rangle = 1 + (N/V)\,\tfrac{4}{3}\pi R_c^3$
with threshold $R_c$, a useful check that the clustering is unbiased.

---

## Volume Move (_NPT_)

Proposes isotropic or anisotropic volume changes for sampling the _NPT_ ensemble.
The volume is sampled logarithmically:

$$
V_\text{new} = \exp\!\bigl(\ln V_\text{old} + (\xi - 0.5)\, \Delta_V \bigr)
$$

where $\xi$ is a uniform random number in $[0, 1)$.

No move-level bias is applied; the acceptance is handled by standard Metropolis
sampling together with the [`isobaric`](energy.md#external-pressure-isobaric) energy term
which contributes $PV - (N+1) k_BT \ln V$.

### Example

```yaml
system:
  energy:
    pressure: !atm 1.0

propagate:
  collections:
    - !Stochastic
      moves:
        - !TranslateMolecule { molecule: Water, max_displacement: 0.5, weight: 1.0 }
        - !VolumeMove { volume_displacement: 0.04, weight: 0.5 }
```

### Options

Key      | Required | Default      | Description
-------- | -------- | ------------ | -------------------------------------------
`volume_displacement` | yes |     | Volume displacement parameter, log-scale (alias `dV`)
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

---

## Speciation Move (Reaction Ensemble)

Performs molecular insertion/deletion and atom-type swaps according to
chemical reactions in the grand canonical or semi-grand canonical ensemble.
The acceptance criterion follows
[Smith & Triska, _J. Chem. Phys._ 100, 3019 (1994)](https://doi.org/10.1063/1.466443).

Each step randomly picks a reaction and direction (forward or backward),
then proposes the corresponding operations:

- Molecular insertion: activates an empty group at a random position in the cell.
- Molecular deletion: deactivates a randomly chosen active group.
- Molecular swap: replaces a molecule of one type with another of equal atom count,
  preserving spatial orientation via gyration tensor alignment.
- Atom swap: changes the type of a random atom (for reactions like $A \rightleftharpoons B$).

Reactions are written using the syntax described in [Chemical Reactions](topology.md#chemical-reactions).
The equilibrium constant $K$ is related to the excess chemical potential,
and the system must pre-allocate inactive molecule slots using `active < N`
in the `blocks` section.

### Molecular insertion and deletion

$K$ is dimensionless, referred to a standard state of 1 M. Concentrations
therefore enter the acceptance relative to the standard-state number density

$$
c_0 = N_A \times 10^{-27}\ \text{Å}^{-3} \approx 6.022 \times 10^{-4}\ \text{Å}^{-3},
$$

so it is $c_0 V$, not $V$, that pairs with $K$.

For a reaction that creates or destroys molecules (e.g. $\emptyset \rightleftharpoons M$
with equilibrium constant $K$), the acceptance follows:

$$
\operatorname{acc}_\text{insert} = \min\!\biggl(1,\;
K \cdot \frac{c_0 V}{N+1} \cdot e^{-\beta \Delta U}\biggr),
\qquad
\operatorname{acc}_\text{delete} = \min\!\biggl(1,\;
\frac{N}{K \cdot c_0 V} \cdot e^{-\beta \Delta U}\biggr)
$$

where $N$ is the number of molecules _before_ the move and $V$ is the cell volume.
At equilibrium for an ideal gas ($\Delta U = 0$) this yields

$$
\langle N \rangle = K c_0 V .
$$

To place on average ten molecules in a $10 \times 10 \times 10$ Å cell, set
$K = 10 / (c_0 V) = 16.6054$.

For a general reaction $\sum_i \nu_i A_i = 0$ involving multiple species,
the combinatorial bias generalises to

$$
\ln \Gamma = \sum_i \sum_{j=0}^{|\nu_i|-1}
\ln\!\Bigl(\frac{N_i^{(\text{old})} \pm (j+1)}{c_0 V}\Bigr)
$$

with the sign matching the direction of the stoichiometric change
([doi:10/fqcpg3](https://doi.org/10/fqcpg3)).

[Reservoir molecules](topology.md#implicit-reservoirs) participate as molecular
reaction species and do not need a `~` prefix.
The factor $c_0 V$ is replaced by unity for reservoir species,
since they exist outside the simulation cell.

### Atom-type swaps

For reactions that swap atom types within a molecule (e.g. $A \rightleftharpoons B$
with equilibrium constant $K$), the acceptance is:

$$
\operatorname{acc} = \min\!\biggl(1,\;
K \cdot \frac{N_\text{from}}{N_\text{to}+1} \cdot e^{-\beta \Delta U}\biggr)
$$

where $N_\text{from}$ and $N_\text{to}$ are the counts of the source and target
atom types _before_ the swap, summed over all molecules of the relevant type.
This $N_\text{from}/(N_\text{to}+1)$ factor ensures detailed balance and
yields a binomial equilibrium distribution with
$\langle N_B \rangle / \langle N_A \rangle = K$,
consistent with
[Faunus](https://doi.org/10.5281/zenodo.5235137) and
[ESPResSo](https://doi.org/10.1140/epjst/e2019-800186-9).

### Molecular swaps

When a reaction has exactly one molecular reactant and one molecular product
(each with multiplicity one) of the same atom count
(e.g. $A \rightleftharpoons B + \text{implicit}$),
the move is automatically detected as a molecular swap.
A full group of the source type is deactivated and an empty group of the
target type is activated with positions transferred via
[gyration tensor](https://doi.org/10.1002/jcc.21776) principal-axis alignment.

A species appearing with a coefficient greater than one (e.g. $2A \rightleftharpoons B$)
is not a swap: it is handled as insertion/deletion of the individual molecules,
conserving charge across the whole reaction. Charge-conserving exchanges such as
$2\,\mathrm{Na}^+ \rightleftharpoons \mathrm{Ca}^{2+}$ work with either one group per
ion or pooled `atomic: true` ions.

A one-to-one reaction between two species is always read as a molecular swap, so both
species must be non-atomic: every member of an `atomic: true` kind shares a single
group, which cannot move between full and empty on its own.

The acceptance is:

$$
\operatorname{acc} = \min\!\biggl(1,\;
K_\text{eff} \cdot \frac{N_\text{from}}{N_\text{to}+1} \cdot e^{-\beta \Delta U}\biggr)
$$

where $N_\text{from}$ and $N_\text{to}$ are the counts of source and target
molecule groups _before_ the move.
No volume factor appears because the total molecule count is conserved.
$\Delta U$ excludes intramolecular energy (bonded and nonbonded self-interactions)
since these are absorbed into the equilibrium constant $K$.
The total energy still counts it, so if the two states differ in intramolecular energy —
say their sites carry different charges — the incremental and the recomputed energy
disagree and the run reports an energy drift. Exclude every intramolecular pair of the
swapped molecules to remove the term from both sides; faunus warns at startup when a swap
between molecules of differing atom types leaves such pairs unexcluded.
Each species must have pre-allocated groups (`active < N`) to serve as a pool;
if the pool is exhausted, the move is silently rejected.

#### Phosphate titration example

Four charge states of orthophosphate with three $pK_a$ values
and implicit protons:

```yaml
atoms:
  - {name: P, mass: 31.0, sigma: 3.0}
  - {name: O, mass: 16.0, sigma: 2.8}
  - {name: H+, mass: 1.0, activity: 6.31e-8}  # pH 7.2

molecules:
  - name: H₃PO₄
    atoms: [P, O, O, O, O]
  - name: H₂PO₄⁻
    atoms: [P, O, O, O, O]
  - name: HPO₄²⁻
    atoms: [P, O, O, O, O]
  - name: PO₄³⁻
    atoms: [P, O, O, O, O]

system:
  blocks:
    - {molecule: H₃PO₄,  N: 20, active: 0,  insert: !RandomAtomPos {}}
    - {molecule: H₂PO₄⁻, N: 20, active: 20, insert: !RandomAtomPos {}}
    - {molecule: HPO₄²⁻,  N: 20, active: 0,  insert: !RandomAtomPos {}}
    - {molecule: PO₄³⁻,   N: 20, active: 0,  insert: !RandomAtomPos {}}

propagate:
  collections:
    - !Deterministic
      moves:
        - !SpeciationMove
          reactions:
            - ["H₃PO₄ = H₂PO₄⁻ + ~H+", !pK 2.15]
            - ["H₂PO₄⁻ = HPO₄²⁻ + ~H+", !pK 7.20]
            - ["HPO₄²⁻ = PO₄³⁻ + ~H+", !pK 12.35]
```

At $pH = pK_{a2} = 7.20$, the effective equilibrium constant for the
second reaction is unity, giving equal populations of H₂PO₄⁻ and HPO₄²⁻.

### Grand-canonical salt example

Insertion and deletion of a neutral ion pair from a salt reservoir at fixed activity.
For single-atom species, `atomic: true` pools all instances into one group, reducing
overhead from $N$ groups to one.

```yaml
atoms:
  - {name: Na, mass: 22.99, charge: 1.0, sigma: 3.3, epsilon: 0.01}
  - {name: Cl, mass: 35.45, charge: -1.0, sigma: 4.4, epsilon: 0.01}

molecules:
  - name: cation
    atoms: [Na]
    atomic: true
    activity: 0.05            # molar
  - name: anion
    atoms: [Cl]
    atomic: true
    activity: 0.05

system:
  cell: !Cuboid [40.0, 40.0, 40.0]
  medium:
    permittivity: !Water
    temperature: 298.15
  energy:
    nonbonded:
      default:
        - !Coulomb {cutoff: 20.0}
        - !WeeksChandlerAndersen {mixing: LB}
  blocks:
    - {molecule: cation, N: 120, active: 30, insert: !RandomAtomPos {}}
    - {molecule: anion,  N: 120, active: 30, insert: !RandomAtomPos {}}

propagate:
  repeat: 10000
  collections:
    - !Stochastic
      moves:
        - !TranslateAtom {atom: Na, molecule: cation, max_displacement: 5.0}
        - !TranslateAtom {atom: Cl, molecule: anion, max_displacement: 5.0}
    - !Deterministic
      moves:
        - !SpeciationMove
          repeat: 5
          reactions:
            # Both activities fold into K_eff; see Activity folding below.
            - ["= cation + anion", !dG 0.0]
```

The ion densities settle *above* the 0.05 M reservoir activity — about 0.061 M here.
The activity, not the concentration, is what the reservoir fixes; ion–ion attraction makes
the electrostatic excess chemical potential negative, so the mean activity coefficient
$\gamma_\pm \approx 0.82 < 1$ and $c = a / \gamma_\pm > a$.

### Options

Key           | Required | Default | Description
------------- | -------- | ------- | -------------------------------------------
`reactions`   | yes      |         | List of reactions (see below)
`weight`      | no       | 1       | Selection weight
`repeat`      | no       | 1       | Repetitions per selection
`temperature` | no       |         | Deprecated and ignored; see below

The system temperature (`system.medium.temperature`) sets the thermal energy $RT$ used
throughout. The `temperature` key is a leftover that duplicated it, and a value
disagreeing with the system temperature is an error — remove the key.

Each reaction is a two-element tuple `[reaction_string, equilibrium_constant]`:

- Reaction string: e.g. `"= NaCl"` or `"⚛A = ⚛B"`
- Equilibrium constant: `!K <value>`, `!lnK <value>`, `!pK <value>` ($K = 10^{-\text{pK}}$),
  or `!dG <kJ/mol>` ($K = e^{-\Delta G / RT}$, with $R$ the molar gas constant, matching
  the kJ/mol energy unit)

An equilibrium constant that overflows to infinity — from a large `!lnK`, or from a
`!dG` mistakenly given in J/mol — is rejected at startup rather than accepting every
trial.

### Activity folding

The user-specified $K$ (or $10^{-\text{pK}}$) is combined with species
activities to form an effective equilibrium constant used in acceptance:

$$\ln K_\text{eff} = \ln K
  + \sum_{\text{implicit reactants}} \ln a_i
  - \sum_{\text{implicit products}} \ln a_i
  - \sum_{\text{insert/delete reactants}} \ln z_i
  + \sum_{\text{insert/delete products}} \ln z_i$$

where:

- Implicit species (`~H+`): $a_i$ is the `activity` field on the matching atom type,
  or, if no atom type carries that name, on the matching molecule type — so an implicit
  molecular reservoir such as `~H2O` is legal. Consumed reactants increase
  $K_\text{eff}$; produced products decrease it.
- Molecular species involved in insertion/deletion (`Na+`, `Cl-`):
  $z_i = a_i c_0$ converts the molar `activity` on the molecule type to a number
  density. The sign is reversed relative to implicit species because the acceptance
  criterion already includes the combinatorial factor
  $(c_0 V)^{\Delta\nu} \cdot \prod [N!/(N+\nu)!]$, which uses $N/(c_0 V)$; the fugacity
  correction turns this into $N/(c_0 V a)$.
- Molecular swap participants are excluded from fugacity folding since
  no volume factor enters the acceptance (total molecule count is conserved).

For example, with `["= Na+ + Cl-", !K 1.0]` and molecular activities
of 0.030 M on both ions, the bare $K=1$ is modified by the two product
fugacities: $\ln K_\text{eff} = 0 + \ln z_\text{Na} + \ln z_\text{Cl}$.

### Notes

- Molecule blocks must have `active < N` to provide empty slots for insertion and swap targets.
- A titratable atom type may occur in several molecule types; all of them titrate, and
  $N_\text{from}$ and $N_\text{to}$ are counted over the whole population. If some molecule
  type lists *both* protonation states, only those types titrate — which keeps a species
  that merely reuses the same atom type (say a monatomic salt ion) out of the reaction.
- Molecular swaps are auto-detected when a reaction has one molecular reactant and one molecular
  product with equal atom counts. Pool exhaustion silently rejects the move — increase `N` if needed.
- Inserted molecules arrive with their reference geometry at a random position and orientation.
- The move targets the entire system, so energy is recomputed globally on each trial.

The following are rejected at startup rather than sampled as a different reaction:

- an atom or molecule name that matches no type;
- unbalanced atom stoichiometry, e.g. `"⚛A + ⚛A = ⚛B"`;
- an equilibrium constant that is not positive and finite;
- a reaction naming a molecule for which no `blocks:` entry allocates groups;
- a one-to-one reaction between `atomic: true` species.

---

## Gibbs Ensemble

The Gibbs ensemble method
([Panagiotopoulos, _Mol. Phys._ 61, 813 (1987)](https://doi.org/10.1080/00268978700101491))
simulates two coupled simulation boxes to study phase coexistence without
an explicit interface.
When `propagate.gibbs` is present, faunus clones the system into two boxes
and alternates between:

1. Intra-box MC — each box runs `intra_steps` propagation cycles in parallel
   (using the `collections` defined above).
2. Inter-box moves — volume exchange and particle transfer between the two boxes.

The total number of Gibbs sweeps is `repeat / intra_steps`.

Molecule blocks must pre-allocate inactive slots for transfer using `active < N`
in the `blocks` section, e.g. `{ molecule: LJ, N: 600, active: 300 }`.

### Configuration

```yaml
propagate:
  repeat: 5000
  collections:
    - !Stochastic
      moves:
        - !TranslateMolecule { molecule: LJ, max_displacement: 0.3, repeat: 100 }
  gibbs:
    intra_steps: 1
    moves:
      - !GibbsVolumeExchange { volume_displacement: 0.3 }
      - !GibbsParticleTransfer { molecule: LJ }
```

Key            | Required | Default | Description
-------------- | -------- | ------- | -------------------------------------------
`intra_steps`  | yes      |         | Intra-box propagation cycles between inter-box moves
`moves`        | yes      |         | List of inter-box moves

### Gibbs Volume Exchange

Exchanges volume between two boxes while conserving total volume $V_1 + V_2$.
Both boxes are isotropically rescaled and all particle positions scale with the box.
$N$ counts independently translatable mass centers: one per molecular group,
one per active atom in atomic groups.

Logarithmic (default) — displaces in $\ln(V_1/V_2)$ space
([Frenkel & Smit, Sec. 8.3.2](https://doi.org/10.1016/B978-012267351-1/50006-7)):

$$
\operatorname{acc} = \min\!\bigl(1,\;
  \exp\bigl[-\beta(\Delta U_1 + \Delta U_2)
  + (N_1{+}1) \ln(V_1'/V_1) + (N_2{+}1) \ln(V_2'/V_2)\bigr]\bigr)
$$

The $(N{+}1)$ factor comes from the Jacobian of the $\ln V$ transformation.

Linear (opt-in) — direct $\delta V$ transfer
([Allen & Tildesley, Eq. 9.75](https://doi.org/10.1093/oso/9780198803195.001.0001)):

$$
\operatorname{acc} = \min\!\bigl(1,\;
  \exp\bigl[-\beta(\Delta U_1 + \Delta U_2)
  + N_1 \ln(V_1'/V_1) + N_2 \ln(V_2'/V_2)\bigr]\bigr)
$$

```yaml
- !GibbsVolumeExchange { volume_displacement: 10 }                 # default: Logarithmic
- !GibbsVolumeExchange { volume_displacement: 10, method: Linear } # opt-in linear
```

Key      | Required | Default        | Description
-------- | -------- | -------------- | -------------------------------------------
`volume_displacement` | yes |       | Volume displacement parameter (alias `dV`)
`method` | no       | `Logarithmic`  | Displacement method: `Logarithmic` or `Linear`

### Gibbs Particle Transfer

Transfers a molecule from one box to the other.
A random direction is chosen (box 0 → 1 or box 1 → 0).
Both molecular and atomic molecule kinds are supported:
for molecular groups, a full group is deactivated/activated;
for atomic groups, a single atom is removed/inserted via the
speciation expand/shrink mechanism.
If no suitable slot exists in either box, the move is rejected.

Acceptance follows Panagiotopoulos Eq. 8:

$$
\operatorname{acc} = \min\!\bigl(1,\;
  \exp\bigl[-\beta(\Delta U_\text{src} + \Delta U_\text{tgt})
  + \ln\!\bigl(\tfrac{V_\text{tgt}\, N_\text{src}}{V_\text{src}\,(N_\text{tgt}+1)}\bigr)\bigr]\bigr)
$$

where $N$ counts are measured _before_ the move.

```yaml
- !GibbsParticleTransfer { molecule: LJ }
```

Key        | Required | Description
---------- | -------- | -------------------------------------------
`molecule` | yes      | Name of the molecule type to transfer

---

## Langevin Dynamics

Rigid-body Langevin dynamics using the BAOAB splitting scheme, accelerated via
[CubeCL](https://github.com/tracel-ai/cubecl) (wgpu, CUDA, or CPU backends).
Molecules are treated as rigid bodies with translational and rotational degrees of freedom,
integrated with per-body friction and stochastic forces at the target temperature.
Requires the `gpu` cargo feature (`cargo run --features gpu`).

Langevin dynamics is placed as a collection entry alongside Monte Carlo moves,
enabling hybrid MC/LD schemes where MC sweeps alternate with LD blocks within
each propagation cycle. On each MC→LD transition, atom positions, centers of mass,
and rigid-body orientations are uploaded to the compute device; after the LD block,
updated positions and orientations are downloaded back.

Pair interactions are evaluated on-device using cubic spline interpolation of the
tabulated pair potentials (see `energy.spline` in [Energy](energy.md)).
A cell list reduces pairwise force evaluation from $O(n^2)$ to $O(n \cdot k)$
where $k$ is the number of atoms in neighboring cells.
The cell list is built on the CPU from periodically downloaded positions
and uploaded as a CSR structure; the rebuild interval is configurable.
Intramolecular bonded forces (harmonic bonds, harmonic angles, and periodic/harmonic
dihedrals) are also computed on-device when present in the topology.

### Theory

The Langevin equation couples Newtonian dynamics to a heat bath through
friction and stochastic forces:

$$
m \ddot{\mathbf{r}} = \mathbf{F}(\mathbf{r}) - \gamma\, m\, \dot{\mathbf{r}} + \sqrt{2\gamma\, m\, k_BT}\;\mathbf{\xi}(t)
$$

where $m$ is the particle mass, $\gamma$ the friction coefficient (1/ps),
$\mathbf{F}$ the conservative force, and $\mathbf{\xi}(t)$ is Gaussian white noise
with $\langle \xi_i(t)\,\xi_j(t') \rangle = \delta_{ij}\,\delta(t - t')$.
The friction and noise terms satisfy the fluctuation–dissipation theorem,
ensuring that the system samples the canonical (NVT) ensemble at temperature $T$.

#### BAOAB integrator

Time integration uses the BAOAB splitting
([Leimkuhler & Matthews, _Appl. Math. Res. Express_ 2013, 34–56](https://doi.org/10.1093/amrx/abs010)),
which splits each timestep $\Delta t$ into five sub-steps:

| Step | Operation |
|------|-----------|
| B | Half-kick: $\mathbf{v} \leftarrow \mathbf{v} + \frac{\Delta t}{2m}\,\mathbf{F}$ |
| A | Half-drift: $\mathbf{r} \leftarrow \mathbf{r} + \frac{\Delta t}{2}\,\mathbf{v}$ |
| O | Ornstein–Uhlenbeck: $\mathbf{v} \leftarrow e^{-\gamma \Delta t}\,\mathbf{v} + \sqrt{\frac{k_BT}{m}(1 - e^{-2\gamma \Delta t})}\;\mathbf{R}$ |
| A | Half-drift: $\mathbf{r} \leftarrow \mathbf{r} + \frac{\Delta t}{2}\,\mathbf{v}$ |
| B | Half-kick: $\mathbf{v} \leftarrow \mathbf{v} + \frac{\Delta t}{2m}\,\mathbf{F}$ |

Here $\mathbf{R}$ is a vector of independent standard normal variates.
Placing the stochastic step (O) at the center gives superior configurational
sampling accuracy compared to other splittings, with the
configurational distribution correct to $\mathcal{O}(\Delta t^2)$.

#### Rigid-body rotational dynamics

Each molecule is represented by its center-of-mass position and a unit quaternion
$\mathbf{q}$ encoding orientation. Atom positions are reconstructed as

$$
\mathbf{r}_i = \mathbf{r}_\text{com} + \mathbf{q}\,\mathbf{r}_i^{\,\text{ref}}\,\mathbf{q}^{-1}
$$

where $\mathbf{r}_i^{\,\text{ref}}$ are time-independent reference coordinates in the body frame.
Rotational equations of motion follow the same BAOAB pattern, with
the body-frame angular velocity $\mathbf{\omega}$ and diagonal inertia tensor
$\mathbf{I} = \operatorname{diag}(I_{xx}, I_{yy}, I_{zz})$ replacing
$\mathbf{v}$ and $m$. The O step applies the Ornstein–Uhlenbeck
thermostat independently to each angular velocity component:

$$
\omega_\alpha \leftarrow e^{-\gamma \Delta t}\,\omega_\alpha + \sqrt{\frac{k_BT}{I_{\alpha\alpha}}(1 - e^{-2\gamma \Delta t})}\;R_\alpha
$$

The A drift steps update the quaternion by composing an incremental rotation:

$$
\mathbf{q} \leftarrow \Delta\mathbf{q}\;\mathbf{q}, \quad
\Delta\mathbf{q} = \exp\!\bigl(\tfrac{\Delta t}{4}\,\mathbf{\omega}\bigr)
$$

Torques are computed from the forces on each atom in the body frame:
$\mathbf{\tau} = \sum_i \mathbf{r}_i^{\,\text{ref}} \times \mathbf{f}_i^{\,\text{body}}$.

### Example

```yaml
propagate:
  seed: Hardware
  criterion: Metropolis
  repeat: 500
  collections:
    - !Stochastic
      repeat: 20
      moves:
        - !TranslateMolecule { molecule: Water, max_displacement: 0.5, repeat: 1 }
        - !RotateMolecule { molecule: Water, max_angle: 0.3, repeat: 1 }
    - !LangevinDynamics
      timestep: 0.1
      friction: 5.0
      steps: 20
      temperature: 298.15
```

### Options

Key                  | Required | Default | Description
-------------------- | -------- | ------- | -------------------------------------------
`timestep`           | yes      |         | Integration timestep (ps)
`friction`           | yes      |         | Friction coefficient (1/ps)
`steps`              | yes      |         | Number of LD steps per block
`temperature`        | yes      |         | Target temperature (K)
`cell_list_rebuild`  | no       | 20      | Cell list rebuild interval (steps; 0 = only at start)

### Output

After simulation, the YAML output reports measured temperatures
from the equipartition theorem:

```yaml
- !LangevinDynamics
  timestep: 0.1
  friction: 5.0
  steps: 20
  temperature: 298.15
  measured_temperature:
    translational: {mean: 298.3, error: 1.2}
    rotational: {mean: 297.8, error: 2.1}
```

### Notes

- Molecules must have `degrees_of_freedom: Rigid` and `has_com: true` in the topology.
- The simulation cell must be bounded (cuboid).
- Velocities are initialized from the Maxwell–Boltzmann distribution on the first call
  and persist on the compute device across subsequent LD blocks.
- MC rotation moves (`!RotateMolecule`) automatically track rigid-body orientation,
  which is transferred to the device at each LD block start.
- Energy splines must be configured (`energy.spline`) to provide on-device pair potentials.
