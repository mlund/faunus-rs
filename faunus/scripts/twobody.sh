#!/bin/sh

example=$(basename "$0" .sh)

cd examples/${example}
cargo run --release -- run --input ${example}.yaml
