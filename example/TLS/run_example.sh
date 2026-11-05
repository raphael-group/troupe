#!/usr/bin/env bash
# Runs TROUPE (scripts/run_classe_troupe.py) on the TLS lineage trees.
#
# Usage (from anywhere, with the virtual environment activated):
#   bash example/TLS/run_example.sh 10 3 1 0.3       # custom regularization strengths
#
# Outputs are written to example/TLS/results/.

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$EXAMPLE_DIR/../.." && pwd)"
DATA_DIR="$EXAMPLE_DIR/data"
OUT_DIR="$EXAMPLE_DIR/results"

# Leaf sampling probability: fraction of cells in each TLS that were sequenced.
SAMPLING_PROBABILITY=0.05

if [ "$#" -gt 0 ]; then
    REGULARIZATIONS=("$@")
else
    REGULARIZATIONS=(10 1 0.1)
fi

mkdir -p "$OUT_DIR"

# Phase 1 (sparse MLE per regularization), Phase 2 (potency extraction +
# debiased re-fit), and model selection (knee of neg-log-likelihood vs. number
# of states).
python "$REPO_DIR/scripts/run_classe_troupe.py" \
    -i "$DATA_DIR/trees.pkl" \
    -o "$OUT_DIR" \
    --terminal_labels "$DATA_DIR/terminal_labels.txt" \
    --observed_potencies "$DATA_DIR/observed_potencies.txt" \
    --sampling_probability "$SAMPLING_PROBABILITY" \
    --regularizations "${REGULARIZATIONS[@]}"

# Differentiation map + growth-rate plots for every regularization value.
python "$REPO_DIR/scripts/evaluate_results.py" plot-differentiation-maps \
    -i "$OUT_DIR"

echo
echo "Done. Model selection summary: $OUT_DIR/classe_troupe_summary.txt"
cat "$OUT_DIR/classe_troupe_summary.txt"
