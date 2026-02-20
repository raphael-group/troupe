#!/bin/bash

# Example usage:
#   `bash scripts/TLS_experiment/cluster_run_inference.sh`

subsampling_rate=1.0

max_num_hidden_states=1000

num_terminals=5
experiment_name=TLS

# num_terminals=4
# experiment_name=TLSC

# num_terminals=4
# experiment_name="test_rates=10_num_trees=32_time=1.75_trial=0"
# experiment_name="test_rates=10_num_trees=128_time=1.75_trial=0"

num_cores=1

working_dir=/n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml
source /n/fs/ragr-research/users/wh8114/projects/.venv/bin/activate


# for regularization in 0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30
for regularization in 0.71 0.73 0.75 0.77 0.79
do
    echo regularization strength = $regularization
    
    # TODO: Remove vanilla_likelihood
    trees_file=$working_dir/experiments/$experiment_name/processed_data/trees.pkl
    outdir=$working_dir/experiments/$experiment_name/results_vanilla_likelihood/reg=$regularization

    echo $trees_file

    mkdir -p $outdir

    python scripts/save_potency_sets.py \
        -p $max_num_hidden_states \
        -i $trees_file \
        -o $outdir \
        -l $working_dir/experiments/$experiment_name/terminal_labels.txt \
        -k $working_dir/experiments/$experiment_name/observed_potencies.txt \
        -n $num_terminals

    logfile=$outdir/inference
    sbatch -t 1-0 -c $num_cores -J "$regularization"_"$experiment_name" -o $logfile.log -e $logfile.err \
        scripts/single_run_inference.sh $max_num_hidden_states \
                                        $trees_file \
                                        $outdir \
                                        $regularization \
                                        $working_dir/experiments/$experiment_name/terminal_labels.txt \
                                        $working_dir/experiments/$experiment_name/observed_potencies.txt \
                                        $subsampling_rate
done