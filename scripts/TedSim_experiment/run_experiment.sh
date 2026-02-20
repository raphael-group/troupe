#!/bin/bash

experiment_dir=$BASE_DIR/experiments/TedSim
trees_file=$experiment_dir/batch_trees_ncells=2048_p=0.9/trees.pkl
terminal_labels=$experiment_dir/terminal_labels.txt
observed_potency=$experiment_dir/observed_potency.txt
num_terminals=3
starting_state=''

source "$(dirname "$0")/../env.conf"
source "$VENV_PATH"


for reg in 0.01 0.03 0.1 0.3 1 3 10 30 100
do
    outdir=$experiment_dir/results/reg=$reg
    mkdir -p $outdir

    python $BASE_DIR/scripts/save_potency_sets.py \
        -i $trees_file \
        -o $outdir \
        -l $terminal_labels \
        -k $observed_potency \
        -n $num_terminals
        
    bash $BASE_DIR/scripts/single_run_inference.sh \
        $trees_file \
        $outdir \
        $reg \
        $terminal_labels \
        $observed_potency \
        $starting_state
done