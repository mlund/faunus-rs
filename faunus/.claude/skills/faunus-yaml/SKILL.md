---
name: faunus-yaml
description: Create, validate, and explain Faunus YAML input files for molecular simulation. Use when setting up systems, configuring energy terms, MC moves, or analysis.
---

Help the user create or modify Faunus YAML input files for molecular simulation.
Use the reference material below and existing examples in `examples/` and `tests/files/`.
Always read relevant example files when generating new configs to ensure accuracy.

**Trusted YAML sources:**
- `examples/` — all `.yaml` files are valid working examples
- `tests/files/*/input.yaml` — subdirectory inputs are valid integration tests
- `tests/files/*.yaml` — valid **unless** the filename describes an error condition

**Skip these files:**
- **Invalid test inputs:** Many files under `tests/files/` are deliberately malformed
  for validation testing. Avoid any file whose name contains error-describing words like
  `nonunique`, `undefined`, `unknown_field`, `invalid`, `missing`, `nonexistent`,
  `too_few`, `too_many`, `overlap`, or `duplicate`.
- **Output files:** `*output.yaml`, `*state.yaml`, and `*reference_output.yaml` are
  simulation output, not input. Do not use them as input file examples.

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
