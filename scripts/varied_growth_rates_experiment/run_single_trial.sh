#!/bin/bash

trees_file=$1
outdir=$2
VENV_PATH=$3

source "$VENV_PATH"

python scripts/run_troupe.py \
        -i $trees_file \
        -o $outdir \
        --regularizations 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100