---
name: faunus-yaml
description: Create, validate, and explain Faunus YAML input files for molecular simulation. Use when setting up systems, configuring energy terms, MC moves, or analysis.
---

Help the user create or modify Faunus YAML input files for molecular simulation.
Use the reference material below and the valid example files listed here.
Always read relevant example files when generating new configs to ensure accuracy.

**Valid example input files** (read these for reference):

Examples — full simulation setups:
- `examples/calvados3/input.yaml` — coarse-grained protein with include files
- `examples/kimhummer/input.yaml` — Kim-Hummer coarse-grained model
- `examples/langevin/langevin.yaml` — Langevin dynamics
- `examples/langevin/equilibrate.yaml` — equilibration workflow
- `examples/sticks/input.yaml` — rigid stick molecules
- `examples/twobody/twobody.yaml` — two-body interaction analysis
- `examples/protein_ions/input.yaml` — protein with ions in spherical cell

Regression tests — compact, well-tested inputs:
- `tests/files/gcmc_ideal_gas/input.yaml` — ideal gas GCMC
- `tests/files/gcmc_swap/input.yaml` — atom swap + GCMC
- `tests/files/gibbs_ensemble/input.yaml` — Gibbs ensemble
- `tests/files/npt_polymers/input.yaml` — NPT polymer simulation
- `tests/files/npt_water/input.yaml` — NPT water
- `tests/files/npt_water_ewald/input.yaml` — NPT water with Ewald summation
- `tests/files/titration/input.yaml` — protein titration with salt
- `tests/files/titration_implicit/input.yaml` — implicit proton titration

Unit test topologies — partial configs for specific features:
- `tests/files/speciation_test.yaml` — speciation move setup
- `tests/files/topology_pass.yaml` — valid topology with includes
- `tests/files/translate_molecules_simulation.yaml` — molecule translation
- `tests/files/bonded_interactions.yaml` — bonded energy terms
- `tests/files/nonbonded_interactions.yaml` — nonbonded energy terms
- `tests/files/nonbonded_kimhummer.yaml` — Kim-Hummer potential
- `tests/files/nonbonded_custom.yaml` — custom pair potential
- `tests/files/sasa_interactions.yaml` — SASA energy terms
- `tests/files/cell_sphere.yaml` — spherical cell geometry

Include file fragments (not standalone):
- `examples/calvados3/calvados3.yaml` — CALVADOS3 forcefield
- `examples/sticks/duello-topology.yaml` — stick molecule topology
- `tests/files/top2.yaml`, `tests/files/top3.yaml` — partial topologies

## YAML Input Structure

A Faunus input file has this top-level structure:

```yaml
include: [forcefield.yaml]      # optional: merge external YAML files
version: 0.2.0                  # optional: semantic version of include files

atoms: [...]                    # required: define atom/bead types
molecules: [...]                # required: define molecular topologies

system:
  medium: {...}                 # required: temperature, dielectric, salt
  cell: ...                     # required: simulation box geometry
  blocks: [...]                 # required: place molecules in box
  energy: {...}                 # required: interaction potentials
  intermolecular: {...}         # optional: cross-molecule bonded terms

propagate: {...}                # required: MC moves or Langevin dynamics
analysis: [...]                 # optional: trajectory, RDF, energy output
```

See `reference.md` in this skill directory for the full specification of all sections, options, and recipes.

## Key Tips

- Use `spline` tabulation for performance; add `bounding_spheres: true` for rigid molecules
- Use `combine_with_default: true` when include files provide default nonbonded terms
- `!Stochastic` collections for mixed molecule types; `!Deterministic` with `repeat` for sweeps
