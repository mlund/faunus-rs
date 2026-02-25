# Energy

The Hamiltonian is the sum of all energy terms acting on the system.
Energy terms are defined in the `energy` section of the YAML input.

## External Pressure (Isobaric)

The `isobaric` energy term adds an external pressure contribution for the NPT ensemble:

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
  isobaric:
    P/atm: 1.0
```

Supported pressure units:

| Key      | Unit                          |
|----------|-------------------------------|
| `P/atm`  | atmospheres                  |
| `P/bar`  | bar                          |
| `P/Pa`   | Pascal                       |
| `P/kT`   | $k_BT/\text{Å}^3$           |
| `P/mM`   | millimolar (ideal gas)       |

## Example

$$
\int_a^{\infty} \frac{1}{r^6} \, dx
$$
