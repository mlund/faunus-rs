# Faunus User Guide

Faunus is a flexible molecular simulation framework written in Rust.
This guide covers the YAML input format and methodology.

!!! note
    The text is partly extracted from the code using LLM assistance, and may contain inaccuracies.
    Please report issues on [Github](https://github.com/mlund/faunus-rs/issues).

## Getting started

Faunus reads a YAML input file that defines the system topology, energy terms, and simulation protocol.
A minimal example:

```yaml
temperature: 298.15
geometry:
  type: cuboid
  length: [100.0, 100.0, 100.0]
```

## Subcommands

- `faunus run -i input.yaml` — run a simulation
- `faunus rerun -i input.yaml --traj traj.xtc` — replay a trajectory through a different Hamiltonian (see [Rerun](analysis.md#rerun))
- `faunus umbrella -i input.yaml` — multi-walker umbrella sampling (see [Umbrella Sampling](umbrella.md))
- `faunus wang-landau -i input.yaml` — flat-histogram free energy estimation (see [Wang-Landau](wang_landau.md))

## Sections

- [Topology](topology.md) — atoms, molecules, and chemical reactions
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

Block comments hide entire YAML sections without per-line `#`:

```yaml
{# Disabled section:
umbrella:
  cv: ...
  windows: ...
#}
```

### Tips

- YAML tags (`!Slit`, `!Lambda`, `!pK`) are preserved — MiniJinja does not interpret them
- Use `| round(4)` to control decimal precision: `{{ (a + b) / 2 | round(4) }}`
- Variables are file-scoped; `include:` files are rendered independently
- See `examples/grafted_phosphate/input.yaml` for a full example
