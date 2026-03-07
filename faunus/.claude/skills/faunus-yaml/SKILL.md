---
name: faunus-yaml
description: Create, validate, and explain Faunus YAML input files for molecular simulation. Use when setting up systems, configuring energy terms, MC moves, analysis, running simulations, equilibrating, or working with state files.
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

## Running Simulations

```bash
# Build
cargo build --release

# Build with native SIMD (recommended for production)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Basic run
faunus run -i input.yaml

# With state file for checkpoint/restart
faunus run -i input.yaml -s state.yaml

# Custom output path
faunus run -i input.yaml -o results.yaml

# Verbose / debug logging
faunus run -i input.yaml -v
RUST_LOG=Debug faunus run -i input.yaml

# GPU Langevin dynamics
cargo run --release --features gpu -- run -i input.yaml
```

## Equilibration Workflow

1. **Two-phase approach**: Run a short equilibration with `analysis: []`, saving state with `-s state.yaml`. Then run production loading the same state file.
2. **Energy minimization**: Use `criterion: Minimize` to accept only downhill moves, then switch to `Metropolis`.
3. **Gradual displacement**: Start with small `dp` values to resolve overlaps, increase for production.

Always check `output.yaml` for move acceptance ratios (target ~30-50% for translations).

## State Files

State files (`-s` flag) store particle positions for checkpoint/restart:
```yaml
particles:
  - {atom_id: 0, index: 0, pos: [1.23, 4.56, 7.89]}
  - {atom_id: 1, index: 1, pos: [2.34, 5.67, 8.90]}
```

- Positions loaded on startup if the file exists; written after simulation completes
- Does NOT store cell dimensions or topology (those come from input YAML)
- Gibbs ensemble generates per-box files: `box0_state.yaml`, `box1_state.yaml`

## Key Tips

- Use `spline` tabulation for performance; add `bounding_spheres: true` for rigid molecules
- Use `combine_with_default: true` when include files provide default nonbonded terms
- Use `!Fixed <seed>` for reproducible runs during development
- Energy drift in `output.yaml` should be ~0; large drift indicates a bug
- `!Stochastic` collections for mixed molecule types; `!Deterministic` with `repeat` for sweeps
