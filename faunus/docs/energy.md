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
