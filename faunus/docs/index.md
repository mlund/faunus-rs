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

## Sections

- [Topology](topology.md) — atoms, molecules, and chemical reactions
- [Energy](energy.md) — Hamiltonian and energy terms
- [Moves](moves.md) — Monte Carlo moves and propagation
- [Analysis](analysis.md) — runtime analysis and output
