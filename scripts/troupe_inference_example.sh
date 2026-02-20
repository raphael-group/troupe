#!/bin/bash

# Runs TROUPE inference on the example dataset.
# Run from the repository root: bash scripts/troupe_inference_example.sh

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$BASE_DIR/.venv/bin/activate"

python scripts/run_troupe.py \
    -i "$BASE_DIR/example/trees.pkl" \
    -o "$BASE_DIR/example/results" \
    --regularizations 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30
