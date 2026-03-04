#!/bin/sh
# Run GPU Langevin dynamics example from an equilibrated state.

cd examples/langevin
cp eq_state.yaml state.yaml
cargo run --release --features gpu -- run --input langevin.yaml --state state.yaml
