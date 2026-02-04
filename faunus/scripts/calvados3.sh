#!/bin/sh

example=$(basename "$0" .sh)

cd examples/${example}
RUST_LOG="Trace" cargo run --release -- run --input input.yaml
