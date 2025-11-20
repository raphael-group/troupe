#!/bin/bash

# Example usage:
#   `bash scripts/TLS_experiment/cluster_run_equal_NMP_inference.sh`

subsampling_rate=1.0
max_num_hidden_states=1000

num_terminals=4
experiment_name=TLS
used_reg=0.3

num_cores=1

working_dir=/n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml
source /n/fs/ragr-research/users/wh8114/projects/.venv/bin/activate


for regularization in 0 10 100 1000 10000 100000
# for regularization in 1000
do
    echo regularization strength = $regularization
    
    trees_file=$working_dir/experiments/$experiment_name/processed_data/trees.pkl
    indir=$working_dir/experiments/$experiment_name/results_vanilla_likelihood/reg=$used_reg
    outdir=$working_dir"/experiments/"$experiment_name"_constrained_transitions/results/reg="$regularization

    echo $trees_file

    mkdir -p $outdir

    # python scripts/save_potency_sets.py \
    #     -p $max_num_hidden_states \
    #     -i $trees_file \
    #     -o $indir \
    #     -l $working_dir/experiments/$experiment_name/terminal_labels.txt \
    #     -k $working_dir/experiments/$experiment_name/observed_potencies.txt \
    #     -n $num_terminals

    logfile=$outdir/inference
    sbatch -t 1-0 -c $num_cores -J "$regularization"_"$experiment_name" -o $logfile.log -e $logfile.err \
        scripts/TLS_experiment/inference_equal_NMP.sh $indir \
                                                      $trees_file \
                                                      $outdir \
                                                      $regularization \
                                                      $working_dir/experiments/$experiment_name/terminal_labels.txt \
                                                      $working_dir/experiments/$experiment_name/observed_potencies.txt \
                                                      $subsampling_rate
done