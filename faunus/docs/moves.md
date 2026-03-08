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

- **`!Stochastic`** — each repeat draws *one* move at random, with probability
  proportional to `weight`. This is the normal Monte Carlo sampling mode.
- **`!Deterministic`** — each repeat executes *all* moves in order.
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
`dp`     | Maximum displacement parameter (meaning depends on the move)

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
        - !TranslateMolecule { molecule: Water, dp: 0.5, weight: 1.0 }
        - !RotateMolecule { molecule: Water, dp: 0.3, weight: 1.0 }
        - !VolumeMove { dV: 0.04, weight: 0.5 }
    - !Deterministic
      repeat: 1
      moves:
        - !TranslateAtom { molecule: Water, atom: O, dp: 0.1, weight: 1.0 }
```

---

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

---

## Speciation Move (Reaction Ensemble)

Performs molecular insertion/deletion and atom-type swaps according to
chemical reactions in the grand canonical or semi-grand canonical ensemble.
The acceptance criterion follows
[Smith & Triska, _J. Chem. Phys._ 100, 3019 (1994)](https://doi.org/10.1063/1.466443).

Each step randomly picks a reaction and direction (forward or backward),
then proposes the corresponding operations:

- **Molecular insertion**: activates an empty group at a random position in the cell.
- **Molecular deletion**: deactivates a randomly chosen active group.
- **Molecular swap**: replaces a molecule of one type with another of equal atom count,
  preserving spatial orientation via gyration tensor alignment.
- **Atom swap**: changes the type of a random atom (for reactions like $A \rightleftharpoons B$).

Reactions are written using the syntax described in [Chemical Reactions](topology.md#chemical-reactions).
The equilibrium constant $K$ is related to the excess chemical potential,
and the system must pre-allocate inactive molecule slots using `active < N`
in the `blocks` section.

### Molecular insertion and deletion

For a reaction that creates or destroys molecules (e.g. $\emptyset \rightleftharpoons M$
with equilibrium constant $K$), the acceptance follows:

$$
\operatorname{acc}_\text{insert} = \min\!\biggl(1,\;
K \cdot \frac{V}{N+1} \cdot e^{-\beta \Delta U}\biggr),
\qquad
\operatorname{acc}_\text{delete} = \min\!\biggl(1,\;
\frac{N}{K \cdot V} \cdot e^{-\beta \Delta U}\biggr)
$$

where $N$ is the number of molecules _before_ the move and $V$ is the cell volume.
At equilibrium for an ideal gas ($\Delta U = 0$), this yields $\langle N \rangle = KV$.

For a general reaction $\sum_i \nu_i A_i = 0$ involving multiple species,
the combinatorial bias generalises to

$$
\ln \Gamma = \sum_i \sum_{j=0}^{|\nu_i|-1}
\ln\!\Bigl(\frac{N_i^{(\text{old})} \pm (j+1)}{V}\Bigr)
$$

with the sign matching the direction of the stoichiometric change
([doi:10/fqcpg3](https://doi.org/10/fqcpg3)).

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
with the same number of atoms (e.g. $A \rightleftharpoons B + \text{implicit}$),
the move is automatically detected as a molecular swap.
A full group of the source type is deactivated and an empty group of the
target type is activated with positions transferred via
[gyration tensor](https://doi.org/10.1002/jcc.21776) principal-axis alignment.

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
          temperature: 298.15
          reactions:
            - ["H₃PO₄ = H₂PO₄⁻ + ~H+", !pK 2.15]
            - ["H₂PO₄⁻ = HPO₄²⁻ + ~H+", !pK 7.20]
            - ["HPO₄²⁻ = PO₄³⁻ + ~H+", !pK 12.35]
```

At $pH = pK_{a2} = 7.20$, the effective equilibrium constant for the
second reaction is unity, giving equal populations of H₂PO₄⁻ and HPO₄²⁻.

### GCMC example

```yaml
molecules:
  - name: Na+
    atoms: [Na]
    activity: 0.030           # molar GCMC fugacity
  - name: Cl-
    atoms: [Cl]
    activity: 0.030

system:
  blocks:
    - molecule: Na+
      N: 30
      active: 10
      insert: !RandomAtomPos {}
    - molecule: Cl-
      N: 30
      active: 10
      insert: !RandomAtomPos {}

propagate:
  repeat: 10000
  collections:
    - !Deterministic
      moves:
        - !SpeciationMove
          temperature: 298.15
          reactions:
            # Coupled titration + salt exchange to maintain electroneutrality
            - ["⚛HGLU + Cl- = ⚛GLU + ~H+", !pK 4.24]
            # Grand canonical salt: activities folded into K_eff
            - ["= Na+ + Cl-", !K 1.0]
        - !TranslateAtom { atom: Na, molecule: Na+, dp: 50.0 }
        - !TranslateAtom { atom: Cl, molecule: Cl-, dp: 50.0 }
```

### Options

Key           | Required | Default | Description
------------- | -------- | ------- | -------------------------------------------
`temperature` | yes      |         | Temperature in Kelvin (used to compute $k_BT$)
`reactions`   | yes      |         | List of reactions (see below)
`weight`      | no       | 1       | Selection weight
`repeat`      | no       | 1       | Repetitions per selection

Each reaction is a two-element tuple `[reaction_string, equilibrium_constant]`:

- **Reaction string**: e.g. `"= NaCl"` or `"⚛A = ⚛B"`
- **Equilibrium constant**: `!K <value>`, `!lnK <value>`, `!pK <value>` ($K = 10^{-\text{pK}}$), or `!dG <kJ/mol>` ($K = e^{-\Delta G/k_BT}$)

### Activity folding

The user-specified $K$ (or $10^{-\text{pK}}$) is combined with species
activities to form an effective equilibrium constant used in acceptance:

$$\ln K_\text{eff} = \ln K
  + \sum_{\text{implicit reactants}} \ln a_i
  - \sum_{\text{implicit products}} \ln a_i
  - \sum_{\text{insert/delete reactants}} \ln z_i
  + \sum_{\text{insert/delete products}} \ln z_i$$

where:

- **Implicit species** (`~H+`): $a_i$ is the `activity` field on the
  matching atom type. Consumed reactants increase $K_\text{eff}$;
  produced products decrease it.
- **Molecular species** involved in insertion/deletion (`Na+`, `Cl-`):
  $z_i = a_i \times N_A / 10^{27}$
  converts the molar `activity` on the molecule type to number density.
  The sign is reversed relative to implicit species because the
  acceptance criterion already includes a $V$-dependent combinatorial
  factor $V^{\Delta\nu} \cdot \prod [N!/(N+\nu)!]$ that uses $N/V$;
  the fugacity correction replaces $N/V$ with $N/(V \cdot z)$.
- **Molecular swap** participants are excluded from fugacity folding since
  no volume factor enters the acceptance (total molecule count is conserved).

For example, with `["= Na+ + Cl-", !K 1.0]` and molecular activities
of 0.030 M on both ions, the bare $K=1$ is modified by the two product
fugacities: $\ln K_\text{eff} = 0 + \ln z_\text{Na} + \ln z_\text{Cl}$.

### Notes

- Molecule blocks must have `active < N` to provide empty slots for insertion and swap targets.
- Atom swap reactions require that at least one atom type belongs to the target molecule definition.
- Molecular swaps are auto-detected when a reaction has one molecular reactant and one molecular
  product with equal atom counts. Pool exhaustion silently rejects the move — increase `N` if needed.
- The move targets the entire system, so energy is recomputed globally on each trial.

---

## Gibbs Ensemble

The Gibbs ensemble method
([Panagiotopoulos, _Mol. Phys._ 61, 813 (1987)](https://doi.org/10.1080/00268978700101491))
simulates two coupled simulation boxes to study phase coexistence without
an explicit interface.
When `propagate.gibbs` is present, faunus clones the system into two boxes
and alternates between:

1. **Intra-box MC** — each box runs `intra_steps` propagation cycles in parallel
   (using the `collections` defined above).
2. **Inter-box moves** — volume exchange and particle transfer between the two boxes.

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
        - !TranslateMolecule { molecule: LJ, dp: 0.3, repeat: 100 }
  gibbs:
    intra_steps: 1
    moves:
      - !GibbsVolumeExchange { dV: 10 }
      - !GibbsParticleTransfer { molecule: LJ }
```

Key            | Required | Default | Description
-------------- | -------- | ------- | -------------------------------------------
`intra_steps`  | yes      |         | Intra-box propagation cycles between inter-box moves
`moves`        | yes      |         | List of inter-box moves

### Gibbs Volume Exchange

Proposes a linear volume transfer $\Delta V$ between the two boxes:
$V_1' = V_1 + \Delta V$, $V_2' = V_2 - \Delta V$,
where $\Delta V \in [-\text{dV}/2, +\text{dV}/2]$.
Both boxes are isotropically rescaled and all particle positions scale with the box.

Acceptance includes the ideal-gas entropy bias:

$$
\operatorname{acc} = \min\!\bigl(1,\;
  \exp\bigl[-\beta(\Delta U_1 + \Delta U_2)
  + N_1 \ln(V_1'/V_1) + N_2 \ln(V_2'/V_2)\bigr]\bigr)
$$

```yaml
- !GibbsVolumeExchange { dV: 10 }
```

Key  | Required | Description
---- | -------- | -------------------------------------------
`dV` | yes      | Maximum volume displacement (linear scale)

### Gibbs Particle Transfer

Transfers a molecule from one box to the other.
A random direction is chosen (box 0 → 1 or box 1 → 0).
A full group is deactivated in the source box, and an empty group
is activated at a random position in the target box.
If no suitable group exists in either box, the move is rejected.

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
| **B** | Half-kick: $\mathbf{v} \leftarrow \mathbf{v} + \frac{\Delta t}{2m}\,\mathbf{F}$ |
| **A** | Half-drift: $\mathbf{r} \leftarrow \mathbf{r} + \frac{\Delta t}{2}\,\mathbf{v}$ |
| **O** | Ornstein–Uhlenbeck: $\mathbf{v} \leftarrow e^{-\gamma \Delta t}\,\mathbf{v} + \sqrt{\frac{k_BT}{m}(1 - e^{-2\gamma \Delta t})}\;\mathbf{R}$ |
| **A** | Half-drift: $\mathbf{r} \leftarrow \mathbf{r} + \frac{\Delta t}{2}\,\mathbf{v}$ |
| **B** | Half-kick: $\mathbf{v} \leftarrow \mathbf{v} + \frac{\Delta t}{2m}\,\mathbf{F}$ |

Here $\mathbf{R}$ is a vector of independent standard normal variates.
Placing the stochastic step (**O**) at the center gives superior configurational
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
$\mathbf{v}$ and $m$. The **O** step applies the Ornstein–Uhlenbeck
thermostat independently to each angular velocity component:

$$
\omega_\alpha \leftarrow e^{-\gamma \Delta t}\,\omega_\alpha + \sqrt{\frac{k_BT}{I_{\alpha\alpha}}(1 - e^{-2\gamma \Delta t})}\;R_\alpha
$$

The **A** drift steps update the quaternion by composing an incremental rotation:

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
        - !TranslateMolecule { molecule: Water, dp: 0.5, repeat: 1 }
        - !RotateMolecule { molecule: Water, dp: 0.3, repeat: 1 }
    - !LangevinDynamics
      timestep: 0.1
      friction: 5.0
      steps: 20
      temperature: 298.15
```

### Options

Key           | Required | Default | Description
------------- | -------- | ------- | -------------------------------------------
`timestep`    | yes      |         | Integration timestep (ps)
`friction`    | yes      |         | Friction coefficient (1/ps)
`steps`       | yes      |         | Number of LD steps per block
`temperature` | yes      |         | Target temperature (K)

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
    translational: "298.3 ± 1.2"
    rotational: "297.8 ± 2.1"
```

### Notes

- Molecules must have `degrees_of_freedom: Rigid` and `has_com: true` in the topology.
- The simulation cell must be bounded (cuboid).
- Velocities are initialized from the Maxwell–Boltzmann distribution on the first call
  and persist on the compute device across subsequent LD blocks.
- MC rotation moves (`!RotateMolecule`) automatically track rigid-body orientation,
  which is transferred to the device at each LD block start.
- Energy splines must be configured (`energy.spline`) to provide on-device pair potentials.
