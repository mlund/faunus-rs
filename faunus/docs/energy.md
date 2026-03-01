# Energy

The Hamiltonian is the sum of all energy terms acting on the system.
Energy terms are defined in the `energy` section of the YAML input.

## External Pressure (Isobaric)

The `pressure` energy term adds an external pressure contribution for the NPT ensemble:

$$
E = PV - (N + 1) k_BT \ln V
$$

where $P$ is the external pressure, $V$ is the volume, $k_BT$ is the thermal energy,
and $N$ is the number of independently translatable entities.
For single-atom molecule kinds, each atom counts independently ($N$ += group size);
for multi-atom molecule kinds, each non-empty group contributes 1.

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
    [Na, Cl]:
      - !LennardJones {σ: 3.2, ε: 1.5}
      - !Coulomb {cutoff: 12.0}
```

| Key                | Required | Description                                           |
|--------------------|----------|-------------------------------------------------------|
| `default`          | yes      | List of pair potentials applied to all atom pairs      |
| `[atom1, atom2]`   | no       | Override for a specific pair (order does not matter)   |

### Short-range potentials

Each potential is specified as a YAML tag. Parameters can be given directly
or generated from atom properties (`sigma`, `epsilon`) via a combination rule.

| Tag                        | Aliases | Parameters (direct)            |
|----------------------------|---------|--------------------------------|
| `!LennardJones`            |         | `sigma`/`σ`, `epsilon`/`eps`/`ε` |
| `!WeeksChandlerAndersen`   | `!WCA`  | `sigma`/`σ`, `epsilon`/`eps`/`ε` |
| `!HardSphere`              |         | (mixing only)                  |
| `!KimHummer`               | `!KH`   | `sigma`/`σ`, `epsilon`/`eps`/`ε` |
| `!AshbaughHatch`           |         | (mixing only, requires `cutoff`) |
| `!CustomPotential`         |         | `function`, `cutoff`, `constants` |

When using a combination rule, specify `mixing` instead of explicit parameters:

```yaml
- !LennardJones {mixing: LorentzBerthelot}
- !AshbaughHatch {mixing: arithmetic, cutoff: 20.0}
```

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

| Key            | Required | Default      | Description                              |
|----------------|----------|--------------|------------------------------------------|
| `cutoff`       | yes      |              | Cutoff distance (Å)                      |
| `n_points`     | no       | `2000`       | Number of spline grid points             |
| `grid_type`    | no       | `PowerLaw2`  | Grid spacing strategy (see below)        |
| `shift_energy` | no       | `true`       | Shift energy to zero at cutoff           |

Available grid types:

| Grid type      | Description                                              |
|----------------|----------------------------------------------------------|
| `UniformRsq`   | Uniform spacing in $r^2$ — sparse at short range        |
| `UniformR`     | Uniform spacing in $r$                                   |
| `PowerLaw2`    | Power-law with $p=2$ — dense at short range (default)   |
| `PowerLaw(p)`  | Power-law with custom exponent $p$                       |
| `InverseRsq`   | Uniform in $1/r^2$ — dense at short range               |

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
| `!FENE`         | not yet implemented                                          |
| `!Morse`        | not yet implemented                                          |
| `!UreyBradley`  | $\frac{1}{2} k (r - r_\text{eq})^2$                         |

**Torsions** (three-body, function of angle $\theta$ in degrees):

| Kind         | Energy                                         |
|--------------|------------------------------------------------|
| `!Harmonic`  | $\frac{1}{2} k (\theta - \theta_\text{eq})^2$  |
| `!Cosine`    | not yet implemented                             |

**Dihedrals** (four-body, function of dihedral angle $\phi$ in degrees):

| Kind               | Energy                                                  |
|--------------------|---------------------------------------------------------|
| `!ProperHarmonic`  | $\frac{1}{2} k (\phi - \phi_\text{eq})^2$               |
| `!ProperPeriodic`  | $k \bigl[1 + \cos(n\phi - \phi_0)\bigr]$                |
| `!ImproperHarmonic`| $\frac{1}{2} k (\phi - \phi_\text{eq})^2$               |
| `!ImproperPeriodic`| $k \bigl[1 + \cos(n\phi - \phi_0)\bigr]$                |

See [Topology — Bonds, Torsions, Dihedrals](topology.md#bonds) for YAML syntax, parameters, and units.

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
(`sin`, `cos`, `exp`, `ln`, `sqrt`, `abs`, etc.) are supported via [exmex](https://docs.rs/exmex).

### YAML configuration

```yaml
energy:
  customexternal:
    - selection: "molecule water"
      com: true
      constants: { radius: 15, k: 100 }
      function: "0.5 * k * (x^2 + y^2 + z^2 - radius^2)"
    - selection: "atomtype Na"
      function: "q * 0.1 * z"
```

| Key         | Required | Default | Description                                      |
|-------------|----------|---------|--------------------------------------------------|
| `selection` | yes      |         | Selection expression for atoms/molecules         |
| `function`  | yes      |         | Math expression for the potential (kJ/mol)       |
| `com`       | no       | `false` | Evaluate at molecular mass center                |
| `constants` | no       | `{}`    | Named constants substituted before parsing       |

When `com` is `true`, the expression is evaluated once per group at the mass center
position with the net charge (sum of atom charges).
When `false` (default), the expression is evaluated and summed over all active particles
in each matching group.
