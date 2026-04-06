#!/bin/bash
# Scan pH and ionic strength parameter space using umbrella sampling.
# Each (pH, I) combination runs in its own directory under runs/.
#
# Usage: ./run_scan.sh [--rerun]
#
# Options:
#   --rerun   Re-run even if pmf.csv already exists
#
# Prerequisites:
#   - cargo build --release
#   - An equilibrated state.yaml

set -euo pipefail

FAUNUS="${FAUNUS:-cargo run --release --bin faunus --}"
INPUT="input.yaml"
STATE="state.yaml"
RERUN=false

for arg in "$@"; do
    [[ "$arg" == "--rerun" ]] && RERUN=true
done

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$BASE_DIR/runs"

if [[ ! -f "$BASE_DIR/$STATE" ]]; then
    echo "Error: $BASE_DIR/$STATE not found. Run an equilibration first."
    exit 1
fi

for pH in 4 7 10; do
    for I in 0.030 0.15 1000; do
        dir="$RUNS_DIR/pH${pH}_I${I}"
        mkdir -p "$dir"

        if [[ -f "$dir/pmf.csv" && "$RERUN" == "false" ]]; then
            echo "=== pH=$pH, I=$I M === SKIP (pmf.csv exists; use --rerun to force)"
            continue
        fi

        # Substitute template variables for this (pH, I) point
        sed \
            -e "s/{% set pH = .* %}/{% set pH = ${pH}.0 %}/" \
            -e "s/{% set salt = .* %}/{% set salt = ${I} %}/" \
            "$BASE_DIR/$INPUT" > "$dir/$INPUT"

        # Symlink structure file into the run directory
        ln -sf "$BASE_DIR/phytate.xyz" "$dir/phytate.xyz"

        # Clear old window states on rerun
        [[ "$RERUN" == "true" ]] && rm -rf "$dir/umbrella_states"

        echo "=== pH=$pH, I=$I M === ($dir)"
        (cd "$dir" && $FAUNUS umbrella -i "$INPUT" --state "$BASE_DIR/$STATE" -o pmf.csv) || {
            echo "FAILED: pH=$pH I=$I"
            continue
        }
    done
done

echo ""
echo "Results in:"
find "$RUNS_DIR" -name "pmf.csv" | sort
