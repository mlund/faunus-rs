#!/bin/sh
# Regenerate reference fixtures for regression tests
set -e
cargo test --release fixtures -- --ignored --test-threads=1
