---
name: faunus
description: Create and validate Faunus YAML input files, and build, run, profile, and manage molecular simulations. Use when setting up systems, configuring energy terms, MC moves or analysis, compiling, running, working with state files, equilibrating, or debugging simulation output.
---

Help the user set up Faunus input files and build, run, and manage simulations. Detail
lives in the two reference files below — read the relevant one before generating a config
or a command, and read real example inputs before writing new ones.

- **Writing or modifying input YAML** → read `input-reference.md` in this skill directory.
- **Building, running, reruns, state files, profiling, testing** → read `running.md`.

Example inputs to copy from (regression-tested ones are the most reliable):
`examples/*/input.yaml` and `tests/files/*/input.yaml`.

## Input file structure

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

## Cross-cutting gotchas

- **Global CLI flags go before the subcommand.** `-o/--output` and `-v/--verbose` belong
  to the top-level command: `faunus -o out.yaml run -i input.yaml`, not `faunus run … -o`.
- **Unknown YAML keys are rejected** (`deny_unknown_fields`): a mistyped key is an error,
  not silently ignored — a fast way to catch typos is simply to run the input.
- **Type tags are PascalCase** (`!Cuboid`, `!Harmonic`); unit tags keep scientific casing
  (`!atm`, `!Pa`, `!kT`, `!mM`) and `Axes` are lowercase values (`projection: z`).
