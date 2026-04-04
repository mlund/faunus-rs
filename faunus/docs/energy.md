# Energy

The Hamiltonian is the sum of all energy terms acting on the system.
Energy terms are defined in the `energy` section of the YAML input.

---

## External Pressure (Isobaric)

The `pressure` energy term adds an external pressure contribution for the _NPT_ ensemble:

$$
U = PV - (N + 1) k_BT \ln V
$$

where $P$ is the external pressure, $V$ is the volume, $k_BT$ is the thermal energy,
and $N$ is the number of independently translatable entities
(individual atoms for single-atom molecules, one per molecule for multi-atom molecules).

### YAML configuration

```yaml
energy:
  pressure: !atm 1.0
```

Supported pressure units (YAML tags):

| Tag    | Unit                          |
|--------|-------------------------------|
| `!atm` | atmospheres                   |
| `!bar` | bar                           |
| `!Pa`  | Pascal                        |
| `!kT`  | $k_BT/\text{Å}^3$            |
| `!mM`  | millimolar (ideal gas)        |

--- 

## Nonbonded Interactions

The `nonbonded` energy term handles pairwise particle interactions using a matrix
of pair potentials indexed by atom type for O(1) lookup.
Multiple potentials can be summed for each pair, and interactions for specific atom
pairs can override the default.

### YAML configuration

```yaml
energy:
  nonbonded:
    default:
      - !LennardJones {mixing: LorentzBerthelot}
      - !Coulomb {cutoff: 12.0}
    replace:
      [Na, Cl]:
        - !LennardJones {σ: 3.2, ε: 1.5}
        - !Coulomb {cutoff: 12.0}
```

Three sub-sections control how interactions are assigned to atom pairs:

| Key                      | Description                                          |
|--------------------------|------------------------------------------------------|
| `default`                | List of pair potentials applied to all atom pairs     |
| `replace: [a, b]: [...]` | Completely replaces `default` for that pair           |
| `append: [a, b]: [...]`  | Merges with `default` by interaction type             |

**`replace`** pairs get only what is listed — no default inheritance.

**`append`** pairs inherit `default`, but if both define the same interaction type
(e.g. AshbaughHatch), the append entry replaces that type; other default types
(e.g. Coulomb) are kept.

A pair may not appear in both `replace` and `append`.

### Loading nonbonded from include files

Nonbonded definitions can be provided in an included force field file.
The top-level `include` list is scanned for files with an `energy` section,
and any `nonbonded` entries are merged into the input.
`replace`/`append` entries in the input take precedence over includes.
`default` lists are concatenated; duplicate types from includes are skipped with a warning.

```yaml
# assets/forcefield.yaml
energy:
  nonbonded:
    default:
      - !AshbaughHatch {mixing: arithmetic, cutoff: 20.0}
    replace:
      [A, A]:
        - !KimHummer {sigma: 5.0, epsilon: -0.18}

# input.yaml — gets both AshbaughHatch (from include) and Coulomb as defaults
include: [assets/forcefield.yaml]
system:
  energy:
    nonbonded:
      default:
        - !Coulomb {cutoff: 40.0}
    spline: {cutoff: 40.0}
```

### Short-range potentials

Each potential is specified as a YAML tag. Parameters can be given directly
or generated from atom properties (`sigma`, `epsilon`) via a combination rule.

| Tag                        | Aliases | Parameters (direct)            |
|----------------------------|---------|--------------------------------|
| `!LennardJones`            |         | `sigma`/`σ`, `epsilon`/`eps`/`ε` |
| `!WeeksChandlerAndersen`   | `!WCA`  | `sigma`/`σ`, `epsilon`/`eps`/`ε` |
| `!HardSphere`              |         | (mixing only)                  |
| `!KimHummer`               | `!KH`   | `sigma`/`σ`, `epsilon`/`eps`/`ε` |
| `!AshbaughHatch`           | `!AH`   | (mixing only, requires `cutoff`); or `wca: true` for purely repulsive |
| `!CustomPotential`         |         | `function`, `cutoff`, `constants` |

When using a combination rule, specify `mixing` instead of explicit parameters:

```yaml
- !LennardJones {mixing: LorentzBerthelot}
- !AshbaughHatch {mixing: arithmetic, cutoff: 20.0}
```

#### Ashbaugh-Hatch WCA mode

Setting `wca: true` on `!AshbaughHatch` makes the potential purely repulsive
(Weeks-Chandler-Andersen) by setting λ=1 and cutoff=σ·2^(1/6),
ignoring any explicit `lambda` or `cutoff` values:

```yaml
# Purely repulsive pair via wca flag
[ARG, ALA]:
  - !AshbaughHatch {epsilon: 0.8368, sigma: 5.62, wca: true}
```

> **Note:** `lambda: 0` does NOT give WCA — it produces a flat step function
> of height ε. Use `wca: true` or `lambda: 1` with cutoff=σ·2^(1/6) for
> true repulsive excluded volume.

Available combination rules:

| Rule              | Aliases                            | $\varepsilon_{ij}$                                | $\sigma_{ij}$                            |
|-------------------|------------------------------------|---------------------------------------------------|------------------------------------------|
| `LorentzBerthelot`| `LB`, `lorentz-berthelot`          | $\sqrt{\varepsilon_i \varepsilon_j}$              | $(\sigma_i + \sigma_j)/2$                |
| `Arithmetic`      | `arithmetic`                       | $(\varepsilon_i + \varepsilon_j)/2$               | $(\sigma_i + \sigma_j)/2$                |
| `Geometric`       | `geometric`                        | $\sqrt{\varepsilon_i \varepsilon_j}$              | $\sqrt{\sigma_i \sigma_j}$               |
| `FenderHalsey`    | `FH`, `fender-halsey`              | $2\varepsilon_i\varepsilon_j/(\varepsilon_i + \varepsilon_j)$ | $(\sigma_i + \sigma_j)/2$ |

#### Custom pair potential

`!CustomPotential` lets you define any pair potential as a mathematical expression in `r` (the pair distance in Å).
Named constants can be passed via `constants`:

```yaml
- !CustomPotential
  function: "4*eps*((sigma/r)^12 - (sigma/r)^6)"
  cutoff: 14.0
  constants: { eps: 0.5, sigma: 3.4 }
```

| Key         | Required | Default | Description                                      |
|-------------|----------|---------|--------------------------------------------------|
| `function`  | yes      |         | Math expression in `r` (kJ/mol)                 |
| `cutoff`    | yes      |         | Cutoff distance (Å)                              |
| `constants` | no       | `{}`    | Named constants substituted before parsing       |

### Medium

The `medium` section (under `system`) defines the implicit solvent environment.
Electrostatic potentials read permittivity, temperature, and optional salt from here.

```yaml
system:
  medium:
    permittivity: !Water
    temperature: 298.15
    salt: [!NaCl, 0.005]  # optional: [salt type, molarity in mol/l]
```

| Key             | Required | Description                                      |
|-----------------|----------|--------------------------------------------------|
| `permittivity`  | yes      | Dielectric constant model (see table below)      |
| `temperature`   | yes      | Temperature (K)                                  |
| `salt`          | no       | Salt type and molarity as `[type, mol/l]`        |

#### Dielectric constant models

| Tag         | Description                                          |
|-------------|------------------------------------------------------|
| `!Vacuum`   | Free space, $\varepsilon_r = 1$                      |
| `!Water`    | Temperature-dependent water (empirical NR model)     |
| `!Ethanol`  | Temperature-dependent ethanol (empirical NR model)   |
| `!Methanol` | Temperature-dependent methanol (empirical NR model)  |
| `!Metal`    | Perfect conductor, $\varepsilon_r = \infty$          |
| `!Fixed`    | Custom constant, e.g. `!Fixed 80.0`                 |

#### Salt types

| Tag            | Formula                          |
|----------------|----------------------------------|
| `!NaCl`        | NaCl (1:1 electrolyte)           |
| `!CaCl₂`      | CaCl₂                           |
| `!CaSO₄`      | CaSO₄                           |
| `!Na₂SO₄`     | Na₂SO₄                          |
| `!KAl(SO₄)₂`  | KAl(SO₄)₂                       |
| `!LaCl₃`      | LaCl₃                           |
| `!Custom`      | Custom ion valencies, e.g. `!Custom [2, -1]` |

### Electrostatic potentials

Electrostatic potentials combine atom charges with a coulombic scheme.

| Aliases           | Parameters                                 |
|-------------------|--------------------------------------------|
| `!Coulomb`        | `cutoff`                                   |
| `!Ewald`          | `alpha`, `cutoff`                          |
| `!ReactionField`  | `epsr_in`, `epsr_out`, `cutoff`, `shift`   |
| `!Fanourgakis`    | `cutoff`                                   |

---

## Ewald Summation

The Ewald method splits long-range electrostatic (or Yukawa) interactions into
a short-range real-space sum and a long-range reciprocal-space sum,
plus a self-energy correction:

$$U = U_\text{real} + U_\text{recip} + U_\text{self} + U_\text{bg}$$

The reciprocal-space contribution is

$$U_\text{recip} = \frac{2\pi}{V} \sum_{\mathbf{k} \neq 0}
  \frac{\exp\bigl(-(k^2 + \kappa^2)/4\alpha^2\bigr)}{k^2 + \kappa^2}
  \, |S(\mathbf{k})|^2$$

where $S(\mathbf{k}) = \sum_j q_j e^{i \mathbf{k} \cdot \mathbf{r}_j}$ is the
structure factor, $\alpha$ is the Ewald splitting parameter, and $\kappa$ is the
Debye screening parameter ($\kappa = 0$ for unscreened Coulomb).
For screened electrostatics ($\kappa > 0$), the method generalises to Yukawa Ewald summation
([Salin & Caillol, 2000](https://doi.org/10.1063/1.1326477)).

### Self-energy and electroneutrality corrections

The self-energy correction removes spurious self-interaction from the real-space sum.
For non-neutral systems ($Q = \sum_j q_j \neq 0$), additional corrections are needed
because the $\mathbf{k}=0$ term is excluded from the reciprocal sum.
The treatment differs between Coulomb and Yukawa:

**Coulomb** ($\kappa = 0$): The divergent $\mathbf{k}=0$ term is cancelled by a
uniform neutralizing background charge, leaving a finite correction
([Frenkel & Smit](https://doi.org/10.1016/B978-0-12-267351-1.X5000-7), Eq. 12.1.25):

$$U_\text{self} = -\frac{\alpha}{\sqrt{\pi}} \sum_j q_j^2, \quad
  U_\text{bg} = -\frac{\pi Q^2}{2V\alpha^2}$$

**Yukawa** ($\kappa > 0$): The $\mathbf{k}=0$ term is finite due to screening
and is included explicitly instead of a neutralizing background:

$$U_\text{self} = \left(-\frac{\alpha}{\sqrt{\pi}} e^{-\kappa^2/4\alpha^2}
  + \frac{\kappa}{2}\operatorname{erfc}\!\frac{\kappa}{2\alpha}\right) \sum_j q_j^2, \quad
  U_{k=0} = \frac{2\pi}{V} \frac{e^{-\kappa^2/4\alpha^2}}{\kappa^2} Q^2$$

Both $U_\text{bg}$ and $U_{k=0}$ vanish for electroneutral systems.

### How accuracy controls parameters

The `accuracy` parameter $\varepsilon$ (typically $10^{-5}$) controls the
Ewald splitting parameter and the number of k-vectors via the
Kolafa–Perram error estimates:

$$\alpha = \frac{\sqrt{-\ln \varepsilon}}{r_c}, \quad
  n_\text{max} = \left\lceil \frac{\alpha \, L_\text{max}}{\pi}
  \sqrt{-\ln \varepsilon} \right\rceil$$

where $r_c$ is the real-space cutoff and $L_\text{max}$ is the largest box side length.
Higher accuracy (smaller $\varepsilon$) increases both $\alpha$ and the number of
k-vectors. For Yukawa ($\kappa > 0$), the tighter Pålsson–Tornberg bound
([arXiv:1911.04875](https://arxiv.org/abs/1911.04875)) is used instead,
yielding a smaller $\alpha$ and fewer k-vectors than the Coulomb estimate.
When $2\kappa r_c \geq -\ln\varepsilon$, the screening is strong enough that
the real-space sum alone achieves the target accuracy and the reciprocal sum
is skipped entirely.

### YAML configuration

The `ewald` section automatically sets up both the real-space pair potential
(injected into the nonbonded defaults before splining) and the reciprocal-space
energy term. No manual `!Ewald` entry is needed in the nonbonded list.

```yaml
energy:
  nonbonded:
    default:
      - !LennardJones {mixing: LB}
  spline:
    cutoff: 14.0
  ewald:
    cutoff: 9.0
    accuracy: 1e-5
    policy: PBC
```

| Key        | Required | Default | Description                                                 |
|------------|----------|---------|-------------------------------------------------------------|
| `cutoff`   | yes      |         | Real-space cutoff (Å)                                       |
| `accuracy` | no       | `1e-5`  | Target relative accuracy $\varepsilon$                      |
| `policy`   | no       | `PBC`   | `PBC` or `IPBC` ([Stenqvist & Lund, 2018](https://doi.org/10.1080/00268976.2018.1516231)) |
| `optimize` | no       | `false` | Jointly optimize $\alpha$ and $n_\text{max}$ to minimize k-vectors (Yukawa only) |

If the medium defines a salt concentration, the corresponding Debye screening
parameter $\kappa$ is automatically propagated to both the real-space and
reciprocal-space Ewald terms (Yukawa Ewald summation).

### MC move optimisation

For single-group Monte Carlo moves, the structure factors are updated
incrementally in $O(M \cdot N_k)$ time (where $M$ is the number of moved particles),
rather than the $O(N \cdot N_k)$ full rebuild required for volume changes or
multi-group moves.

### Bonded exclusions

Particles connected by bonds within the same molecule can have their nonbonded
interactions excluded. This is configured per molecule in the topology via the
`exclusions` key (list of intra-molecular index pairs).

### Spline tabulation

For performance, all nonbonded potentials can be tabulated using cubic Hermite splines.
Add a `spline` section alongside `nonbonded`:

```yaml
energy:
  nonbonded:
    default:
      - !WCA {mixing: LB}
      - !Coulomb {cutoff: 200}
  spline:
    cutoff: 200.0
    n_points: 2000
    grid_type: PowerLaw2
```

| Key                | Required | Default      | Description                              |
|--------------------|----------|--------------|------------------------------------------|
| `cutoff`           | yes      |              | Cutoff distance (Å)                      |
| `n_points`         | no       | `2000`       | Number of spline grid points             |
| `grid_type`        | no       | `PowerLaw2`  | Grid spacing strategy (see below)        |
| `shift_energy`     | no       | `true`       | Shift energy to zero at cutoff           |
| `shift_force`      | no       | `false`      | Shift force to zero at cutoff            |
| `cell_list`        | no       | `true`       | Use cell list for spatial acceleration   |
| `bounding_spheres` | no       | `true`       | Use bounding-sphere culling of distant group pairs |

Available grid types:

| Grid type      | Description                                              |
|----------------|----------------------------------------------------------|
| `UniformRsq`   | Uniform spacing in $r^2$ — sparse at short range        |
| `UniformR`     | Uniform spacing in $r$                                   |
| `PowerLaw2`    | Power-law with $p=2$ — dense at short range (default)   |
| `PowerLaw(p)`  | Power-law with custom exponent $p$                       |
| `InverseRsq`   | Uniform in $1/r^2$ — dense at short range               |

---

## Excluded-Pair Coulomb Correction

Excluded pairs (from `excluded_neighbours` or `exclusions`) skip all nonbonded
interactions, including Coulomb. For charge titration or alchemical MC moves
where charges change, the Coulomb contribution between excluded neighbors
must still be evaluated.

Setting `keep_excluded_coulomb: true` on a molecule adds a correction term that
evaluates Coulomb for all excluded pairs in that molecule:

$$E_\text{correction} = \sum_{\text{excluded } (i,j)} U_\text{Coulomb}(r_{ij})$$

The Coulomb scheme and parameters are taken from the nonbonded configuration.
The term is automatically added to the Hamiltonian when at least one molecule
opts in and a Coulomb interaction is configured.

For `Rigid` and `RigidAlchemical` molecules, all intra-molecular pairs are
excluded (distances are constant). The correction adds back the
charge-dependent Coulomb part that varies under alchemical moves.

### YAML configuration

```yaml
molecules:
  - name: peptide
    atoms: [A, B, C]
    excluded_neighbours: 1
    keep_excluded_coulomb: true
```

No additional `energy:` section is needed — the term is created automatically.

---

## Bonded Interactions

Bonded energy terms are automatically added to the Hamiltonian based on the
bonds, torsions, and dihedrals defined in the [topology](topology.md#molecules).
No explicit `energy:` configuration is needed.

- **Intramolecular** — bonds, torsions, and dihedrals within each molecule,
  using local atom indices defined in the molecule type.
  Always active.
- **Intermolecular** — bonds, torsions, and dihedrals between atoms in different
  molecules, using global atom indices defined in
  [`system.intermolecular`](topology.md#intermolecular-bonded-interactions).
  Only added when at least one intermolecular interaction is defined.

### Supported potentials

**Bonds** (two-body, function of distance $r$):

| Kind            | Energy                                                       |
|-----------------|--------------------------------------------------------------|
| `!Harmonic`     | $\frac{1}{2} k (r - r_\text{eq})^2$                         |
| `!FENE`         | $-\tfrac{1}{2}k R_0^2 \ln\bigl(1 - (r/R_0)^2\bigr)$        |
| `!Morse`        | $D_e\bigl(1 - e^{-a(r - r_e)}\bigr)^2$                      |
| `!UreyBradley`  | $\frac{1}{2} k (r - r_\text{eq})^2$                         |

**Torsions** (three-body, function of angle $\theta$ in degrees):

| Kind         | Energy                                         |
|--------------|------------------------------------------------|
| `!Harmonic`  | $\frac{1}{2} k (\theta - \theta_\text{eq})^2$  |
| `!Cosine`    | $\frac{1}{2} k (\cos\theta - \cos\theta_\text{eq})^2$ |

**Dihedrals** (four-body, function of dihedral angle $\phi$ in degrees):

| Kind               | Energy                                                  |
|--------------------|---------------------------------------------------------|
| `!ProperHarmonic`  | $\frac{1}{2} k (\phi - \phi_\text{eq})^2$               |
| `!ProperPeriodic`  | $k \bigl[1 + \cos(n\phi - \phi_0)\bigr]$                |
| `!ImproperHarmonic`| $\frac{1}{2} k (\phi - \phi_\text{eq})^2$               |
| `!ImproperPeriodic`| $k \bigl[1 + \cos(n\phi - \phi_0)\bigr]$                |

See [Topology — Bonds, Torsions, Dihedrals](topology.md#bonds) for YAML syntax, parameters, and units.

---

## Custom External Potential

The `customexternal` energy term applies a user-defined mathematical expression
as an external potential to selected atoms or molecular mass centers.
The energy is evaluated per particle (or per mass center) and summed over all matching groups.

### Variables

The expression can use any subset of:

| Variable | Description              |
|----------|--------------------------|
| `q`      | particle charge          |
| `x`      | x-coordinate (Å)        |
| `y`      | y-coordinate (Å)        |
| `z`      | z-coordinate (Å)        |

Standard math operators (`+`, `-`, `*`, `/`, `^`) and functions
(`sin`, `cos`, `exp`, `ln`, `sqrt`, `abs`, `signum`, `floor`, etc.) are supported
via [exmex](https://docs.rs/exmex). Built-in constants `PI`, `TAU` (=2π), and `π` are available.

Python-style conditionals are supported:
```
"1.0 if x > 0 else -1.0"
"(3 if x < 0.5 else 5) * sin(TAU * y)"
```

Named constants use word-boundary matching, so single-letter names like `c` won't
collide with function names like `cos`.

### Preset potentials

The `function` field can also name a built-in potential, bypassing the expression
parser for better performance:

| Preset              | Description |
|---------------------|-------------|
| `staircase-sincos`  | Piecewise 2D staircase × sinusoidal surface (Frenkel & Smit) |

### YAML configuration

```yaml
energy:
  customexternal:
    # Expression with named constants
    - selection: "molecule water"
      com: true
      constants: { radius: 15, k: 100 }
      function: "0.5 * k * (x^2 + y^2 + z^2 - radius^2)"
    # Expression with conditional
    - selection: "all"
      function: "10.0 if x > 0 else -5.0"
    # Built-in preset
    - selection: "all"
      function: staircase-sincos
```

| Key         | Required | Default | Description                                      |
|-------------|----------|---------|--------------------------------------------------|
| `selection` | yes      |         | Selection expression for atoms/molecules         |
| `function`  | yes      |         | Math expression, conditional, or preset name     |
| `com`       | no       | `false` | Evaluate at molecular mass center                |
| `constants` | no       | `{}`    | Named constants substituted before parsing       |

When `com` is `true`, the expression is evaluated once per matching group at the mass
center position, with `q` set to the net group charge.
When `false` (default), the expression is evaluated at each matching atom position.

---

## Penalty (Flat-Histogram Bias)

Applies a static bias potential loaded from a converged [Wang-Landau](wang_landau.md)
checkpoint. The bias energy is $\ln g(\text{bin}) \times k_BT$, which flattens the
free energy surface along the collective variable(s).

During biased sampling, analysis averages are incorrect.
Use [`faunus rerun`](analysis.md#rerun) to replay the trajectory:
when a penalty term is detected, rerun automatically reweights all analyses
by $w = \exp(-\ln g(\text{bin}))$, recovering the correct ensemble averages.

### YAML configuration

```yaml
energy:
  penalty:
    file: wl_states/histogram.yaml
    coordinate:
      property: atom_position
      selection: "atomtype A"
      dimension: x
      range: [-2.0, 2.0]
      resolution: 0.1
    coordinate2:  # optional, for 2D
      property: atom_position
      selection: "atomtype A"
      dimension: y
      range: [-2.0, 2.0]
      resolution: 0.1
```

| Key           | Required | Description                                                    |
|---------------|----------|----------------------------------------------------------------|
| `file`        | yes      | Path to `FlatHistogramState` checkpoint (e.g. from Wang-Landau)|
| `coordinate`  | yes      | Primary collective variable (see [analysis](analysis.md))      |
| `coordinate2` | no       | Second CV for 2D surfaces                                      |

The penalty is placed at the front of the Hamiltonian so that out-of-range
CV values return infinite energy, short-circuiting expensive downstream terms.

---

## Solvent Accessible Surface Area (SASA)

Computes an implicit solvation energy based on the solvent-accessible surface area
of each particle, using Voronoi tessellation (via `voronota-ltr`).
The energy is summed over all active particles:

$$U = \sum_i \gamma_i \, A_i$$

where $\gamma_i$ is the surface energy density (kJ/mol/Å²) and $A_i$ is the
solvent-accessible surface area of particle $i$.
An isolated particle has maximal SAS area, $4\pi (r_i + r_\text{probe})^2$.
Neighboring particles reduce the exposed area through mutual occlusion.
Positive $\gamma$ means exposed surface is energetically costly — the particle is
hydrophobic and burial is favorable.
Negative $\gamma$ means solvent exposure is favorable.

This differs from the [contact tessellation](#contact-tessellation-energy) energy
which sums over inter-body _contact_ areas rather than per-atom _exposed_ areas.

Surface energy densities are set per atom type via the `hydrophobicity` field
in the [topology](topology.md#atoms).
The tessellation is updated incrementally: when a single group moves,
only the moved atoms and their spatial neighbors are re-tessellated.
This works for both rigid-body and flexible (polymer) moves.
Periodic boundary conditions are supported for orthorhombic cells.

### YAML configuration

```yaml
atoms:
  - {name: A, sigma: 3.0, hydrophobicity: !Gamma 0.9}
  - {name: B, sigma: 4.0, hydrophobicity: !Gamma 1.5}

system:
  energy:
    sasa:
      probe_radius: 1.4
```

| Key                | Required | Default | Description                                           |
|--------------------|----------|---------|-------------------------------------------------------|
| `probe_radius`     | yes      |         | Probe radius (Å) for the tessellation                 |
| `energy_offset`    | no       |         | Constant energy shift (kJ/mol)                        |
| `offset_from_first`| no       | `false` | Set offset so that the first configuration has zero energy |

Particle radii are taken from `sigma / 2` of the atom type definition.
`!SurfaceTension` is accepted as an alias for `!Gamma`.

---

## Contact Tessellation Energy

Computes the contact energy between rigid bodies using radical (power) tessellation.
For each pair of nearby rigid bodies, the atoms of both bodies are tessellated together
and only inter-body contact areas are extracted. The energy is a sum over all
inter-body contacts:

$$U = s \sum_{\text{contacts}} \gamma_{ij} \; A_{ij}$$

where $A_{ij}$ is the radical tessellation contact area between atoms $a$ and $b$
belonging to different rigid bodies, $s$ is an optional scaling factor, and the
combining rule for the surface energy density is:

$$\gamma_{ij} = \begin{cases} \operatorname{sign}(\gamma_i) \sqrt{|\gamma_i \, \gamma_j|} & \text{if } \operatorname{sign}(\gamma_i) = \operatorname{sign}(\gamma_j) \\ 0 & \text{otherwise}\end{cases}$$

Negative $\gamma$ gives attractive contacts (favoring inter-body burial);
positive values give repulsive contacts. Atoms with opposite signs
(e.g. hydrophobic vs. hydrophilic) contribute zero energy.

This differs from the [SASA](#solvent-accessible-surface-area-sasa) energy where
$\gamma_i$ multiplies the _exposed_ area of each atom directly — here,
$\gamma_{ij}$ multiplies the _contact_ area between atom pairs from different bodies.

Body pairs beyond bounding-sphere contact range are skipped.
Only pairs involving the moved body are recomputed each Monte Carlo step,
giving $O(N)$ scaling in the number of bodies. Periodic boundary conditions
are supported for orthorhombic (cuboid) cells.

!!! note
    This energy term is currently designed for **rigid bodies** only — all atoms in
    each molecule move as a unit. Support for flexible molecules and polymers may be
    added in future versions.

### YAML configuration

```yaml
atoms:
  - {name: ALA, sigma: 5.0, hydrophobicity: !Gamma -0.5}
  - {name: ARG, sigma: 6.0, hydrophobicity: !Gamma  0.3}

system:
  energy:
    contact_tessellation: { probe_radius: 1.4 }
```

| Key              | Required | Default | Description                                         |
|------------------|----------|---------|-----------------------------------------------------|
| `probe_radius`   | yes      |         | Probe radius (Å) for the tessellation               |
| `scaling`        | no       | `1.0`   | Global multiplicative scaling of the contact energy |

Particle radii are taken from `sigma / 2` of the atom type.
Surface energy densities are set via `hydrophobicity: !Gamma <value>` (`!SurfaceTension` is accepted as alias).
The bounding sphere cutoff automatically includes the probe diameter to account
for the expanded tessellation radii.

---

## Constrain

Constrains a collective variable to a specified range.
Two modes are supported:

- **Hard constraint** (default): returns infinite energy if the CV value falls
  outside `[min, max]`, otherwise zero.
- **Harmonic constraint**: applies a quadratic penalty
  $\frac{1}{2} k (x_\text{eq} - x)^2$ around an equilibrium value.

### YAML configuration

```yaml
energy:
  constrain:
    - property: volume
      range: [1000.0, 5000.0]
    - property: mass_center_position
      selection: "molecule protein"
      projection: z
      range: [-50.0, 50.0]
      harmonic: # optional
        force_constant: 100.0
        equilibrium: 0.0
```

| Key              | Required | Default | Description                                    |
|------------------|----------|---------|------------------------------------------------|
| `property`       | yes      |         | CV type (see [collective variables](analysis.md#supported-properties)) |
| `range`          | no       | `[-∞, ∞]` | Allowed `[min, max]` interval               |
| `harmonic`       | no       |         | Harmonic restraint parameters (see below)      |
| `projection`     | no       | `xyz`   | Axis projection (`x`, `y`, `z`, `xy`, …); alias: `dimension` |
| `selection`      | depends  |         | Selection expression for one atom or group     |

#### Harmonic parameters

| Key              | Required | Description                      |
|------------------|----------|----------------------------------|
| `force_constant` | yes      | Spring constant $k$ (kJ/mol/unit²) |
| `equilibrium`    | yes      | Target value $x_\text{eq}$       |

## Polymer Depletion Many-Body Interaction

The `polymer_depletion` energy term implements the Forsman & Woodward many-body
Hamiltonian for colloids immersed in an ideal polymer fluid
([Forsman & Woodward, Soft Matter, 2012, 8, 2121](https://doi.org/10.1039/c2sm06737d)),
generalised with Robin boundary conditions for tunable polymer–surface affinity.

Rigid macromolecules of arbitrary shape are treated as neutral spheres using
their center of mass and bounding sphere radius. The polymers are modelled
implicitly via an effective potential that captures many-body depletion effects
through pairwise sums, at $O(N_c^2)$ computational cost.

The free energy change, $\Delta \omega$, due to inserting $N_c$ colloids into a polymer reservoir is:

$$\frac{\beta \Delta\omega}{4\pi\rho_P^*} \approx
  \frac{f \cdot N_c}{\kappa^{3/2}} \left(\sigma + \sigma^2 + \frac{\sigma^3}{3}\right)
  - \frac{f^2 \cdot \sigma^2 e^{2\sigma}}{\kappa^{3/2}}
    \sum_{i=1}^{N_c}
    \frac{\displaystyle\sum_{j \neq i} k_0(\lambda R_{ij})}
         {1 + \tfrac{1}{2}(e^{2\sigma} - 1)\displaystyle\sum_{j \neq i} k_0(\lambda R_{ij})}$$

where $R_{ij}$ is the center-to-center distance between colloids $i$ and $j$,
$\sigma = \sqrt{\kappa}\,R_c / R_g$, $\lambda = \sqrt{\kappa}/R_g$,
$k_0(x) = e^{-x}/x$, $\rho_P^* = \rho_P R_g^3$ is the reduced polymer
reservoir number density, $\kappa = n + 1$ with $n$ being the Schulz–Flory
distribution order ($n = 0$ for equilibrium polymers),
$R_c$ is the colloid (bounding sphere) radius, and
$R_g$ is the polymer radius of gyration.

### Robin boundary condition

The original Forsman–Woodward model assumes non-adsorbing colloid surfaces
(Dirichlet boundary condition, $\hat{g} = 0$ at the surface).
The Robin BC generalises this using the
[de Gennes extrapolation length](https://doi.org/10.1021/ma00115a001)
$b = 1/h$, replacing the Dirichlet monopole amplitude $A_D = \sigma e^\sigma$
with $A_R = A_D \cdot f$, where

$$f(\sigma, \tilde{h}) = \frac{\tilde{h}}{(1 + \sigma) + \tilde{h}}$$

and $\tilde{h} = R_c / b$ is the dimensionless Robin parameter.
The single-particle insertion term scales as $f$ and the many-body pairwise
term as $f^2$; this asymmetry is not reproducible by an effective radius.

| $\tilde{h}$ | Boundary condition | Physical interpretation |
|---|---|---|
| omitted | Dirichlet ($f=1$) | Full depletion; original model |
| large positive | near-Dirichlet | Weakly reduced depletion |
| $0$ | Neumann ($f=0$) | Neutral surface; no polymer-mediated interaction |
| $> -(1+\sigma)$ and $< 0$ | Adsorption ($f<0$) | Polymer accumulates at surface; interactions sign-inverted |

### Self-consistent steric adsorption

The constant $\tilde{h}$ Robin BC diverges when polymer adsorption is strong enough
to drive $\tilde{h} \to -(1+\sigma)$.
The `steric_adsorption` option replaces the fixed $\tilde{h}$ with a per-colloid,
configuration-dependent $\tilde{h}_\text{eff}(i)$ obtained by self-consistent
iteration.
The steric free energy cost of crowding adsorbed chains limits the surface
density to a finite saturation value $g_0$, preventing the divergence.

The effective inverse extrapolation length $\varepsilon_\text{eff}(i)$ is
determined from the surface polymer density $\hat{g}_S(i)$ via

$$\varepsilon_\text{eff} = \varepsilon_0'
  + \ln\!\left(1 - \frac{\hat{g}_S^2}{g_0^2}\right)
  - \frac{\hat{g}_S^2}{g_0^2 - \hat{g}_S^2}$$

and feeds into the Robin amplitude factor as
$\tilde{h}_\text{eff}(i) = -\varepsilon_\text{eff}(i) \cdot R_c$.
The surface density $\hat{g}_S$ is itself a function of $\varepsilon_\text{eff}$
through the modified Helmholtz Green's function, closing the self-consistency
loop solved by Picard iteration.

| Parameter          | Physical meaning |
|--------------------|------------------|
| `epsilon0_prime`   | Bare polymer–surface adsorption strength $\varepsilon_0'$ (dimensionless). Larger values mean stronger bare attraction between polymer segments and the colloid surface. Positive values are required; the self-consistent scheme handles the transition to adsorption internally. |
| `g0`               | Saturation surface density $g_0$. The maximum polymer density that can pack onto the colloid surface before steric repulsion between adsorbed chains halts further accumulation. Must be $> 1$; typical values 5–20. Smaller $g_0$ means surface saturates sooner. |

| `picard_mixing`    | Picard iteration mixing parameter $\alpha \in (0, 1]$. Controls how aggressively the surface density is updated each iteration: $\hat{g}_S^\text{new} = \alpha\,\hat{g}_S^\text{calc} + (1-\alpha)\,\hat{g}_S^\text{old}$. Smaller values improve stability for strongly adsorbing systems at the cost of more iterations. |
| `max_iterations`   | Maximum number of Picard iterations before accepting the current solution. |
| `tolerance`        | Convergence threshold: iteration stops when the largest change in $\hat{g}_S$ across all colloids falls below this value. |

> **Note:** `steric_adsorption` is mutually exclusive with `h_tilde`.
> Analytical forces are not yet implemented for this mode.

### Applicability

- Ideal (non-interacting) polymers under theta conditions
- Colloid radius $R_c \gtrsim 10$ bond lengths (to avoid curvature artefacts)
- Size ratio $q = R_g/R_c$ arbitrary; best tested for $q \sim 0.25$–$2$
- $\kappa = 1$ gives equilibrium (living) polymers; $\kappa \gtrsim 5$ approaches
  monodisperse limit

### YAML configuration

```yaml
energy:
  polymer_depletion:
    polymer_rg: 10.0
    polymer_density: 0.5
    kappa: 1.0
    molecules: [Colloid]
    h_tilde: 5.0  # optional; omit for full depletion (Dirichlet)
```

Or with self-consistent steric adsorption (mutually exclusive with `h_tilde`):

```yaml
energy:
  polymer_depletion:
    polymer_rg: 100.0
    polymer_density: 1.0
    kappa: 1.0
    molecules: [Colloid]
    colloid_radius: 5.0
    steric_adsorption:
      epsilon0_prime: 0.02
      g0: 10.0
```

| Key                | Required | Default | Description                                           |
|--------------------|----------|---------|-------------------------------------------------------|
| `polymer_rg`       | yes      |         | Polymer radius of gyration $R_g$ (Å)                  |
| `polymer_density`  | yes      |         | Reduced reservoir density $\rho_P^*$ (dimensionless)  |
| `kappa`            | no       | `1.0`   | Schulz–Flory order $\kappa = n + 1$                   |
| `molecules`        | yes      |         | Molecule types treated as colloids                    |
| `colloid_radius`   | no       |         | Fixed $R_c$ (Å); default: bounding sphere radius      |
| `colloid_radius_scaling` | no | `1.0`  | Scaling factor for the effective colloid radius        |
| `h_tilde` / `h̃`  | no       |         | Robin BC parameter $\tilde{h} = R_c/b$; omit for Dirichlet |
| `steric_adsorption` | no     |         | Self-consistent steric adsorption block (see below)   |

**`steric_adsorption` sub-keys:**

| Key                | Required | Default | Description                                           |
|--------------------|----------|---------|-------------------------------------------------------|
| `epsilon0_prime`   | yes      |         | Bare adsorption parameter $\varepsilon_0'$ (dimensionless, $> 0$) |
| `g0`               | yes      |         | Saturation surface density $g_0$ (must be $> 1$)      |

| `picard_mixing`    | no       | `0.3`   | Picard mixing parameter $\alpha \in (0, 1]$           |
| `max_iterations`   | no       | `50`    | Maximum self-consistency iterations                   |
| `tolerance`        | no       | `1e-8`  | Convergence threshold for $\hat{g}_S$                 |

---

## Tabulated 6D Rigid-Body Energy

The `tabulated6d` energy term loads pre-computed energy tables for pairs of rigid
molecules, providing O(1) energy lookups during rigid-body Monte Carlo moves.

Tables are generated by [Duello](https://github.com/mlund/duello), which scans all
six degrees of freedom $(R, \omega, \theta_1\varphi_1, \theta_2\varphi_2)$ of two
rigid molecules and stores the pairwise energy (kJ/mol) on an icosphere mesh.
Table energies are converted to simulation units using the medium temperature.

At runtime, the current separation and orientations of each molecule pair
are mapped to 6D table coordinates and the energy is
obtained by Boltzmann-weighted barycentric interpolation on the icosphere faces.
This interpolates $\exp(-\beta U)$ rather than $U$ directly, avoiding Jensen's
inequality bias that would otherwise overestimate repulsive energies at contact.
A pairwise group energy cache gives O(1) lookups for single rigid-body MC moves.

### Table formats

Two binary table formats are supported and auto-detected at load time:

- **Flat** (legacy): uniform angular resolution across all separations.
  Uses `f16` storage.
- **Adaptive**: per-slab angular resolution that adjusts with separation.
  At short range where molecules overlap, slabs with negligible Boltzmann
  weights (exp(−βU) ≈ 0) store no angular data. At long range where the
  energy surface is smooth, slabs are collapsed to a single scalar value
  or use nearest-vertex lookup instead of full interpolation.
  This reduces both file size and lookup cost.
  Adaptive tables are temperature-dependent: the repulsive slab classification
  uses the temperature from table generation. A warning is emitted if the
  simulation temperature is significantly lower than the generation temperature.

### YAML configuration

```yaml
energy:
  tabulated6d:
    - molecules: [ProteinA, ProteinA]
      file: table_AA.bin.gz
    - molecules: [ProteinA, ProteinB]
      file: table_AB.bin
```

| Key             | Required | Default | Description                                     |
|-----------------|----------|---------|-------------------------------------------------|
| `molecules`     | yes      |         | Ordered pair of molecule type names              |
| `file`          | yes      |         | Path to binary table file (`.gz` enables gzip)   |
| `single_lookup` | no       | `false` | Skip swap averaging for homo-dimers (~2× faster) |

For homo-dimer entries (identical molecules), the energy is by default averaged
over both A↔B perspectives to eliminate interpolation asymmetry.
Setting `single_lookup: true` uses only one perspective, halving the lookup
cost at the expense of a small energy drift. The drift decreases with angular
resolution and is negligible at `n_div ≥ 3`.

Molecule pair order in the table must match the order used when generating the
table with Duello. Reversed pairs are detected automatically.
Pairs not covered by any table entry contribute zero energy.
Separations below the table minimum return infinite energy (hard wall).

### Tail correction

Beyond the table's maximum distance $R_\text{max}$, the energy is extrapolated
using a sum of screened multipole terms stored in the table metadata:

$$u_\text{tail}(R) = \frac{e^2}{4\pi\varepsilon_0\varepsilon_r} \sum_i \frac{C_i \, e^{-\kappa_i R}}{R^{p_i}}$$

where $C_i$ is a dimensionless coefficient (charge product $z_1 z_2$ for ion-ion,
fitted for higher-order terms), $\kappa_i$ is the screening parameter, and $p_i$
is the power of $R$ in the denominator ($p=1$ for ion-ion, $p=4$ for ion-dipole).
The Coulomb prefactor $e^2/(4\pi\varepsilon_0\varepsilon_r)$ is stored once in
the table metadata.
If no tail correction metadata is present, separations beyond the table range
return zero.

If the table covers a sufficiently large range (so that the energy is
negligible at $R_\text{max}$), the tail correction is not needed and
can be omitted.

A `medium` with `temperature` is required so that the inverse thermal energy
$\beta = 1/k_BT$ can be computed for the Boltzmann-weighted interpolation.

### Automatic nonbonded exclusion

When `tabulated6d` or `tabulated3d` is active, molecule-type pairs covered by table entries are
automatically excluded from the `nonbonded` energy term.
This prevents double-counting when mixing tabulated rigid-body interactions with
atom-level nonbonded potentials for other molecule types.

---

## Tabulated 3D Molecule-Atom Energy

The `tabulated3d` energy term loads pre-computed energy tables for interactions
between a rigid molecule and an atomic group (e.g. ion-protein),
providing O(1) energy lookups during Monte Carlo moves.

Tables are generated by [Duello](https://github.com/mlund/duello) `atom-scan`, which
scans three degrees of freedom $(R, \theta, \varphi)$ — the separation distance
and the direction from molecule to atom in the molecule's body frame — and stores
the pairwise energy (kJ/mol) on an icosphere mesh.

At runtime, the separation vector is transformed into the rigid molecule's
body frame using its quaternion, and the energy is obtained by Boltzmann-weighted
barycentric interpolation, identical to the 6D term.
The same pairwise group energy cache gives O(1) lookups for rigid-body MC moves.

### Table format

Only the **adaptive** format is supported, using per-slab angular resolution
that adjusts with separation (same scheme as 6D adaptive tables).
No swap averaging is performed since the interaction is inherently asymmetric.

### YAML configuration

```yaml
energy:
  tabulated3d:
    - molecules: [Protein, Sodium]
      file: protein_na.bin.gz
    - molecules: [Protein, Chloride]
      file: protein_cl.bin.gz
```

| Key         | Required | Description                                        |
|-------------|----------|----------------------------------------------------|
| `molecules` | yes      | Pair of molecule type names `[rigid, atomic]`      |
| `file`      | yes      | Path to binary table file (`.gz` enables gzip)     |

The first molecule must be the rigid body whose orientation is used for the
lookup; the second is the atomic group (single atom, no orientation).
Reversed pairs are detected automatically.

### Tail correction and nonbonded exclusion

Beyond $R_\text{max}$, the tail correction from table metadata is used (if present);
otherwise zero is returned. A `medium` with `temperature` is required for
the Boltzmann-weighted interpolation.

Molecule-type pairs covered by `tabulated3d` entries are automatically excluded
from the `nonbonded` energy term, same as for `tabulated6d`.
