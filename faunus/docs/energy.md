# Energy

The Hamiltonian is the sum of all energy terms acting on the system.
Energy terms are defined in the `energy` section of the YAML input.

## External Pressure (Isobaric)

The `pressure` energy term adds an external pressure contribution for the NPT ensemble:

$$
E = PV - (N + 1) k_BT \ln V
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

| Key                      | Required | Default | Description                                          |
|--------------------------|----------|---------|------------------------------------------------------|
| `default`                | no       |         | List of pair potentials applied to all atom pairs     |
| `[atom1, atom2]`         | no       |         | Override for a specific pair (order does not matter)  |
| `combine_with_default`   | no       | `false` | Pair-specific entries extend `default` instead of replacing it |

By default, a pair-specific entry replaces the `default` entirely.
When `combine_with_default` is `true`, pair-specific potentials are _added_ to the
default interactions, which is useful for layering per-pair short-range potentials
on top of a shared Coulomb term.

### Loading nonbonded from include files

Nonbonded pair definitions can be provided in an included force field file
instead of being inlined in the input. The top-level `include` list is scanned
for files containing an `energy` section, and any `nonbonded` pairs found there
are merged into the input.
Pair-specific entries in the input take precedence over includes.
`default` lists are concatenated — e.g. an include providing `!AshbaughHatch`
and the input providing `!Coulomb` yields both as defaults.
Duplicate potential types (same variant) from includes are skipped with a warning.

```yaml
# assets/forcefield.yaml
energy:
  nonbonded:
    default:
      - !AshbaughHatch {mixing: arithmetic, cutoff: 20.0}
    [A, A]:
      - !KimHummer {sigma: 5.0, epsilon: -0.18}

# input.yaml — gets both AshbaughHatch (from include) and Coulomb as defaults
include: [assets/forcefield.yaml]
system:
  energy:
    combine_with_default: true
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
  combine_with_default: true
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

## Polymer Depletion Many-Body Interaction

The `polymer_depletion` energy term implements the Forsman & Woodward many-body
Hamiltonian for colloids immersed in an ideal polymer fluid
([Forsman & Woodward, Soft Matter, 2012, 8, 2121](https://doi.org/10.1039/c2sm06737d)).

Rigid macromolecules of arbitrary shape are treated as neutral spheres using
their center of mass and bounding sphere radius. The polymers are modelled
implicitly via an effective potential that captures many-body depletion effects
through pairwise sums, at $O(N_c^2)$ computational cost.

The free energy change $\beta\Delta\omega$ ($\beta = 1/k_BT$) due to inserting $N_c$ colloids into a polymer reservoir is:

$$\frac{\beta\,\Delta\omega}{4\pi\rho_P^*} \approx
  \frac{N_c}{\kappa^{3/2}} \left(\sigma + \sigma^2 + \frac{\sigma^3}{3}\right)
  - \frac{\sigma^2 e^{2\sigma}}{\kappa^{3/2}}
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
```

| Key                | Required | Default | Description                                           |
|--------------------|----------|---------|-------------------------------------------------------|
| `polymer_rg`       | yes      |         | Polymer radius of gyration $R_g$ (Å)                  |
| `polymer_density`  | yes      |         | Reduced reservoir density $\rho_P^*$ (dimensionless)  |
| `kappa`            | no       | `1.0`   | Schulz–Flory order $\kappa = n + 1$                   |
| `molecules`        | yes      |         | Molecule types treated as colloids                    |
| `colloid_radius`   | no       |         | Fixed $R_c$ (Å); default: bounding sphere radius      |
| `colloid_radius_scaling` | no | `1.0`  | Scaling factor for the effective colloid radius        |
