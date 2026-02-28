#!/bin/sh
# Run ignored regression tests (not fixture generators)
set -e
cargo test --release regression -- --ignored --test-threads=1
