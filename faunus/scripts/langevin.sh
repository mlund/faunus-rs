#!/bin/sh
# Run GPU Langevin dynamics example from an MC-equilibrated state.

set -e
cd examples/langevin
cargo run --release --features gpu -- run --input equilibrate.yaml --state state.yaml
cargo run --release --features gpu -- run --input input.yaml --state state.yaml
