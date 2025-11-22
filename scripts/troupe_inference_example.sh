#!/bin/bash

# Runs TROUPE inference on example dataset
#
# Example usage:
#   `bash scripts/troupe_inference_example.sh`

# NOTE: Must change this to be the absolute directory to your troupe directory
working_dir=/n/fs/ragr-research/users/wh8114/projects/troupe
source $working_dir/.venv/bin/activate

data_dir=$working_dir/example/data
out_dir=$working_dir/example/results

rate_matrix_path=$data_dir/ground_truth_parameters.json
trees_path=$data_dir/trees.pkl

for regularization in 0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10
# for regularization in 1
do
    echo Solving with regularization=$regularization

    reg_out_dir=$out_dir/reg=$regularization
    mkdir -p $reg_out_dir

    python scripts/get_observed_potencies.py -i $trees_path \
                                             -o $reg_out_dir \
                                             -r $rate_matrix_path

    num_terminals=5
    max_num_hidden_states=1000
    python scripts/save_potency_sets.py \
            -p $max_num_hidden_states \
            -i $trees_path \
            -o $reg_out_dir \
            -l $reg_out_dir/terminal_labels.txt \
            -k $reg_out_dir/observed_potencies.txt \
            -n $num_terminals
    
    subsampling_rate=1.0
    scripts/single_run_inference.sh $max_num_hidden_states \
                                    $trees_path \
                                    $reg_out_dir \
                                    $regularization \
                                    $reg_out_dir/terminal_labels.txt \
                                    $reg_out_dir/observed_potencies.txt \
                                    $subsampling_rate \
                                    $working_dir
done