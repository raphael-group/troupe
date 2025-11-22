#!/bin/bash

# Example usage:
#   `bash scripts/TLS_experiment/cluster_run_sse_inference.sh`

subsampling_rate=1.0
regularization=0


max_num_hidden_states=3
num_terminals=5
experiment_name=TLS

# max_num_hidden_states=3
# num_terminals=4
# experiment_name=TLSC

# num_terminals=4
# experiment_name="test_rates=10_num_trees=32_time=1.75_trial=0"
# experiment_name="test_rates=10_num_trees=128_time=1.75_trial=0"

num_cores=1

working_dir=/n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml
source /n/fs/ragr-research/users/wh8114/projects/.venv/bin/activate

trees_file=$working_dir/experiments/$experiment_name/processed_data/trees.pkl
outdir=$working_dir/experiments/$experiment_name/results_SSE_unconstrained

echo $trees_file

mkdir -p $outdir

python scripts/save_potency_sets.py \
    -p $max_num_hidden_states \
    -i $trees_file \
    -o $outdir \
    -l $working_dir/experiments/$experiment_name/terminal_labels.txt \
    -k $working_dir/experiments/$experiment_name/observed_potencies_unconstrained.txt \
    -n $num_terminals \
    --unconstrained_unobserved \
    --unconstrained_observed

# logfile=$outdir/inference
# sbatch -t 1-0 -c $num_cores -J "$regularization"_"$experiment_name" -o $logfile.log -e $logfile.err \
#     scripts/single_run_inference.sh $max_num_hidden_states \
#                                     $trees_file \
#                                     $outdir \
#                                     $regularization \
#                                     $working_dir/experiments/$experiment_name/terminal_labels.txt \
#                                     $working_dir/experiments/$experiment_name/observed_potencies_unconstrained.txt \
#                                     $subsampling_rate
bash scripts/single_run_inference.sh $max_num_hidden_states \
                                    $trees_file \
                                    $outdir \
                                    $regularization \
                                    $working_dir/experiments/$experiment_name/terminal_labels.txt \
                                    $working_dir/experiments/$experiment_name/observed_potencies_unconstrained.txt \
                                    $subsampling_rate