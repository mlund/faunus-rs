# Building and Running Faunus

`faunus --help` and `faunus <subcommand> --help` are the authoritative source for flags;
the recipes below are the common paths.

**Global flags come before the subcommand.** `-o/--output` (default `output.yaml`) and
`-v/--verbose` belong to the top-level command, so they must precede `run`/`rerun`/etc.
`faunus run -i in.yaml -o out.yaml` fails with `unexpected argument '-o'`; write
`faunus -o out.yaml run -i in.yaml`.

## Building

```bash
# Build
cargo build --release

# Build with native SIMD (recommended for production)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# GPU Langevin dynamics
cargo run --release --features gpu -- run -i input.yaml
```

## Subcommands

- `run` — Monte Carlo / Langevin simulation (`-i input`, `-s state`)
- `rerun` — replay a trajectory through a (possibly different) Hamiltonian
- `umbrella` — multi-walker umbrella sampling with free-energy stitching
- `wang-landau` — Wang–Landau flat-histogram sampling

`umbrella` and `wang-landau` take `-s <state_dir>` and `-j <threads>` and write their own
output (`-o pmf.csv` / `-o free_energy.csv`) as *subcommand* flags, i.e. after the
subcommand: `faunus umbrella -i input.yaml -o pmf.csv -j 4`.

## Running Simulations

```bash
# Basic run (writes output.yaml)
faunus run -i input.yaml

# With state file for checkpoint/restart
faunus run -i input.yaml -s state.yaml

# Custom output path (global -o, before the subcommand)
faunus -o results.yaml run -i input.yaml

# Verbose / debug logging
faunus -v run -i input.yaml
RUST_LOG=Debug faunus run -i input.yaml
```

## Rerunning Trajectories

Replay a trajectory through a different Hamiltonian (e.g. compare explicit vs 6D tabulated energies):

```bash
# Rerun with a different energy configuration
faunus rerun -i input_6dtable.yaml --traj traj.xtc

# Explicit aux path (default: traj.aux, derived from --traj) and custom output
faunus -o rerun_output.yaml rerun -i input.yaml --traj traj.xtc --aux traj.aux
```

The original simulation must write a `.aux` frame state file alongside the XTC:
```yaml
analysis:
  - !Trajectory
    file: traj.xtc
    frequency: !Every 100
    save_frame_state: true   # writes traj.aux with quaternions, group sizes, atom_ids
```

`save_frame_state` cannot be combined with a `selection` (the aux frame must cover every
particle). The rerun input YAML provides the Hamiltonian and analysis config; `propagate:`
is ignored and all analysis frequencies are overridden to sample every frame.

## Equilibration Workflow

1. **Two-phase approach**: Run a short equilibration with `analysis: []`, saving state with `-s state.yaml`. Then run production loading the same state file.
2. **Energy minimization**: Use `criterion: Minimize` to accept only downhill moves, then switch to `Metropolis`.
3. **Gradual displacement**: Start with small `dp` values to resolve overlaps, increase for production.

Always check `output.yaml` for move acceptance ratios (target ~30-50% for translations).

## State Files

State files (`-s` flag) checkpoint the runtime-mutable state. Top-level keys:

```yaml
particles:                        # one per particle, in index order
  - {atom_id: 0, pos: [1.23, 4.56, 7.89]}
cell: !Cuboid [30.0, 30.0, 30.0]  # the box geometry IS stored
groups:                           # one per molecule group
  - {molecule: 0, capacity: 1, size: Full, quaternion: [0.0, 0.0, 0.0, 1.0]}
step: 5000
```

- Stores particle types + positions, the cell, per-group size and rigid-body `quaternion`
  (needed by LD and 6D-tabulated energies), and the step count.
- Does NOT store topology, Hamiltonian, moves, or analyses — those are rebuilt from the
  input YAML on resume, which keeps the file small and avoids serializing trait objects.
- `size` is `Full`, `Empty`, or a partial count for grand-canonical / speciation groups.
- Positions loaded on startup if the file exists; written after the run completes.
- Gibbs ensemble generates per-box files: `box0_state.yaml`, `box1_state.yaml`.

## Profiling (macOS)

Use the built-in `sample` command to profile a running simulation:

```bash
# Sample a running process for 10 seconds at 1ms intervals
sample faunus 10 -f profile.txt

# Or by PID
sample <pid> 10 -f profile.txt
```

The output shows a call tree with hit counts, useful for identifying hot functions.

## Remote Execution via SSH

With password-free SSH login, sync and run on remote servers:

```bash
# Sync source to remote
rsync -az --exclude target/ ./ user@host:faunus/

# Build and run remotely (Rust installed in user space)
ssh user@host 'source ~/.cargo/env && cd faunus && RUSTFLAGS="-C target-cpu=native" cargo build --release && ./target/release/faunus run -i input.yaml'

# Fetch results back
rsync -az user@host:faunus/output.yaml .
```

## Testing

```bash
# Unit and integration tests
cargo test
```

Regression tests are ignored by default and validate against committed fixtures. Run them
through the sanctioned script, which knows how to run the checks without regenerating
fixtures:

```bash
scripts/regression_tests.sh
```

**Ask before running the regression suite** — it is slow. Never invoke the ignored tests
as a bare `cargo test --test regression -- --include-ignored`: that also runs the fixture
*generators* and can silently overwrite committed fixtures. When running one by hand,
filter with `--exact` (e.g. `cargo test --release --test regression <name>::regression --
--exact --ignored`) so a generator does not run alongside it.

## Key Tips

- Energy drift in `output.yaml` should be ~0; large drift indicates a bug
- Use `!Fixed <seed>` for reproducible runs during development
