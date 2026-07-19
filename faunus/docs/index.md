# Faunus User Guide

Faunus is a flexible molecular simulation framework written in Rust.
This guide covers the YAML input format and methodology.

## Installation

Install Faunus on macOS, Linux, or Windows with:

```sh
pip install faunus
```

## Getting started

Faunus reads a YAML input file that defines the system topology, energy terms, and simulation protocol.
A minimal example:

```yaml
atoms:
  - {name: A, mass: 1.0}

molecules:
  - {name: particle, atoms: [A], atomic: true}

system:
  medium:
    permittivity: !Vacuum
    temperature: 298.15
  cell: !Cuboid [100.0, 100.0, 100.0]
  blocks:
    - {molecule: particle, N: 1, insert: !RandomAtomPos {}}
  energy: {}

propagate:
  repeat: 100
  collections:
    - !Stochastic
      moves:
        - !TranslateAtom {molecule: particle, max_displacement: 1.0, weight: 1.0}

analysis: []
```

## Running simulations

Run a simulation from an input file:

```sh
faunus run -i input.yaml
```

Faunus writes the main YAML summary to `output.yaml` by default. Use the
top-level `-o` option to choose another file:

```sh
faunus -o output.yaml run -i input.yaml
```

To save and restart a simulation state, pass a state file with `-s`:

```sh
faunus -o output.yaml run -i input.yaml -s state.yaml
```

If `state.yaml` exists, Faunus starts from it. At the end of the run, the same
file is updated with the final state.

The main output file contains the simulation summary, final energy terms, move
statistics, and analysis results that do not use a separate file. Analyses with
a `file:` field write their own data files, such as `energy.csv.gz`, `rdf.dat`,
or `traj.xtc`. A trajectory that will later be used with `faunus rerun` must be
written with `save_frame_state: true`; see [Rerun](analysis.md#rerun).

Tracked examples:

| Example | Use case |
|---------|----------|
| [CALVADOS protein model](https://github.com/mlund/faunus-rs/tree/main/examples/calvados3) | Coarse-grained protein simulation |
| [Phosphate titration](https://github.com/mlund/faunus-rs/blob/main/examples/phosphate-titration/input.yaml) | Acid-base reactions and implicit protons |
| [Double layer](https://github.com/mlund/faunus-rs/blob/main/examples/double_layer/input.yaml) | Charged slit geometry and electrostatic profiles |
| [2D Wang-Landau](https://github.com/mlund/faunus-rs/blob/main/examples/wang_landau_2d/input.yaml) | Flat-histogram free-energy sampling |
| [Gibbs ensemble LJ fluid](https://github.com/mlund/faunus-rs/blob/main/tests/files/gibbs_ensemble/input.yaml) | Two-box phase coexistence reference input from the test suite |

## Subcommands

- `faunus run -i input.yaml` — run a simulation
- `faunus rerun -i input.yaml --traj traj.xtc` — replay a trajectory through a different Hamiltonian (see [Rerun](analysis.md#rerun))
- `faunus umbrella -i input.yaml` — multi-walker umbrella sampling (see [Umbrella Sampling](umbrella.md))
- `faunus wang-landau -i input.yaml` — flat-histogram free energy estimation (see [Wang-Landau](wang_landau.md))

## Sections

- [Topology](topology.md) — atoms, molecules, and chemical reactions
- [Units and Conventions](units.md) — default units, naming, and input conventions
- [Energy](energy.md) — Hamiltonian and energy terms
- [Moves](moves.md) — Monte Carlo moves and propagation
- [Analysis](analysis.md) — runtime analysis, output, and trajectory rerun
- [Wang-Landau](wang_landau.md) — flat-histogram free energy sampling
- [Umbrella Sampling](umbrella.md) — windowed free-energy calculations
- [Selection Language](selection_language.md) — VMD-like atom selection expressions

---

## Template Support

YAML input files can use [MiniJinja](https://docs.rs/minijinja) (Jinja2-compatible)
templates for variables, loops, and expressions.

### Variables and expressions

```yaml
{% set Lz = 200.0 %}
{% set pH = 7.0 %}
{% set n_chains = 50 %}
{% set area_per_chain = 340.0 %}
{% set Lx = (n_chains * area_per_chain) ** 0.5 %}

system:
  cell: !Slit [{{ Lx }}, {{ Lx }}, {{ Lz }}]
  medium:
    permittivity: !Water
    temperature: 298.15
    salt: [!NaCl, 0.15]

atoms:
  - {name: H+, mass: 1.0, activity: {{ 10.0 ** (-pH) }}}
```

### Loops

Generate repetitive sections from lists:

```yaml
{% set names = ["H3PO4", "H2PO4-", "HPO4--", "PO4---"] %}
{% set pKa = [2.15, 7.20, 12.35] %}

molecules:
  {% for name in names %}
  - name: "{{ name }}"
    atoms: [P, O, O, O, O]
  {% endfor %}

reactions:
  {% for i in range(pKa | length) %}
  - ["{{ names[i] }} = {{ names[i+1] }} + ~H+", !pK {{ pKa[i] }}]
  {% endfor %}
```

### Comments

Use normal YAML comments (`#`) to disable individual lines or blocks:

```yaml
# analysis:
#   - !Energy
#     file: energy.csv
#     frequency: !Every 100
```

Faunus also ignores top-level keys whose name starts with `_`. This is useful
for temporarily disabling whole sections while keeping them as valid YAML:

```yaml
_umbrella:
  coordinate: ...
  windows: ...
```

For template input files, MiniJinja block comments can hide entire YAML
sections without per-line `#`:

```yaml
{# Disabled section:
umbrella:
  coordinate: ...
  windows: ...
#}
```

### Tips

- YAML tags (`!Slit`, `!Lambda`, `!pK`) are preserved — MiniJinja does not interpret them
- Use `| round(4)` to control decimal precision: `{{ (a + b) / 2 | round(4) }}`
- Variables are file-scoped; `include:` files are rendered independently
- See `examples/grafted_phosphate/input.yaml` for a full example
