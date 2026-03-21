# Faunus User Guide

Faunus is a molecular simulation framework written in Rust.
This guide covers the YAML input format and the methodology behind the simulations.

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
