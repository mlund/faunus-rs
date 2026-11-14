#!/bin/sh
# Regenerate reference fixtures for regression tests
set -e
FAUNUS_REGENERATE=1 cargo test --release fixtures -- --ignored --test-threads=1
