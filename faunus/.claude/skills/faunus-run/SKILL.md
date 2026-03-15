---
name: faunus-run
description: Build, run, profile, and manage Faunus simulations. Use when compiling, running simulations, working with state files, equilibrating systems, or debugging simulation output.
---

Help the user build, run, and manage Faunus molecular simulations.

## Building

```bash
# Build
cargo build --release

# Build with native SIMD (recommended for production)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# GPU Langevin dynamics
cargo run --release --features gpu -- run -i input.yaml
```

## Running Simulations

```bash
# Basic run
faunus run -i input.yaml

# With state file for checkpoint/restart
faunus run -i input.yaml -s state.yaml

# Custom output path
faunus run -i input.yaml -o results.yaml

# Verbose / debug logging
faunus run -i input.yaml -v
RUST_LOG=Debug faunus run -i input.yaml
```

## Rerunning Trajectories

Replay a trajectory through a different Hamiltonian (e.g. compare explicit vs 6D tabulated energies):

```bash
# Rerun with a different energy configuration
faunus rerun -i input_6dtable.yaml --traj traj.xtc

# Explicit aux file path (default: traj.aux derived from --traj)
faunus rerun -i input.yaml --traj traj.xtc --aux traj.aux -o rerun_output.yaml
```

The original simulation must write a `.aux` frame state file alongside the XTC:
```yaml
analysis:
  - !Trajectory
    file: traj.xtc
    frequency: !Every 100
    save_frame_state: true   # writes traj.aux with quaternions, group sizes, atom_ids
```

The rerun input YAML provides the Hamiltonian and analysis config; `propagate:` is ignored. All analysis frequencies are overridden to sample every frame.

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

# Regression tests (always use --release for performance)
cargo test --test regression --release -- --include-ignored --test-threads=1
```

## Key Tips

- Energy drift in `output.yaml` should be ~0; large drift indicates a bug
- Use `!Fixed <seed>` for reproducible runs during development
