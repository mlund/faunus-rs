#!/bin/sh
# Run ignored regression tests (not fixture generators)
#
# `regression` is a name filter, so the `<name>::fixtures` generators never run.
# Each test simulates in its own temporary directory from a fixed seed, so the
# threads share nothing; four keeps the box responsive without oversubscribing.
set -e
cargo test --release regression -- --ignored --test-threads=4
