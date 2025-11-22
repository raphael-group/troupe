#!/bin/bash

# ....
# Example usage:
#   `bash scripts/sample_efficiency_experiment/cluster_run_troupe_inference.sh`

num_terminals=5
process_time=2.35
rate_matrix=12

max_num_hidden_states=1000
num_cores=1
subsampling_rate=1.0
working_dir=/n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml
experiment_name=sample_efficiency_experiment/rate_matrix_$rate_matrix/troupe
rate_matrix_path=$working_dir/scripts/branching_process_experiment/model_params/rate_matrix_$rate_matrix.json
stopping_text=time_$process_time

trial_start=0
trial_stop=4

source /n/fs/ragr-research/users/wh8114/projects/.venv/bin/activate


for num_trees in 16 32 64 128 256
do
    for trial in $(seq $trial_start $trial_stop)
    do
        trees_dir=$working_dir/simulated_data/branching_process_experiment/$rate_matrix/trees_$num_trees/$stopping_text/trial_$trial
        trees_file=$trees_dir/trees.pkl
        echo $trees_file

        trial_outdir=$working_dir/results/$experiment_name/trees_$num_trees/$stopping_text/trial_$trial

        mkdir -p $trial_outdir

        for regularization in 0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10
        do
            echo trial=$trial num_trees=$num_trees regularization=$regularization

            outdir=$trial_outdir/reg=$regularization

            python scripts/get_observed_potencies.py -i $trees_file \
                                             -o $trial_outdir \
                                             -r $rate_matrix_path

            python scripts/save_potency_sets.py \
                    -p $max_num_hidden_states \
                    -i $trees_file \
                    -o $outdir \
                    -l $trial_outdir/terminal_labels.txt \
                    -k $trial_outdir/observed_potencies.txt \
                    -n $num_terminals

            logfile=$outdir/inference
            sbatch -t 12:00:00 -c $num_cores -J "$trial|$num_trees|$regularization" -o $logfile.log -e $logfile.err \
                    scripts/single_run_inference.sh $max_num_hidden_states \
                                                    $trees_file \
                                                    $outdir \
                                                    $regularization \
                                                    $trial_outdir/terminal_labels.txt \
                                                    $trial_outdir/observed_potencies.txt \
                                                    $subsampling_rate
        done
    done
done
