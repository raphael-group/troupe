#!/bin/bash

# ....
# Example usage:
#   `bash scripts/sample_efficiency_experiment/cluster_run_constrained_observed_inference.sh`

# Change these depending on the simulation
# num_terminals=5
# max_num_hidden_states=4
# process_time=2.5
# rate_matrix=12

# num_terminals=4
# max_num_hidden_states=3
# process_time=1.75
# rate_matrix=2

# num_terminals=4
# max_num_hidden_states=3
# process_time=1.8
# rate_matrix=13

num_terminals=5
max_num_hidden_states=4
process_time=2.35
rate_matrix=12

num_cores=1
subsampling_rate=1.0
regularization=0.0
working_dir=/n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml
experiment_name=sample_efficiency_experiment/rate_matrix_$rate_matrix/constrained_observed
rate_matrix_path=$working_dir/scripts/branching_process_experiment/model_params/rate_matrix_$rate_matrix.json
stopping_text=time_$process_time

trial_start=0
trial_stop=4

source /n/fs/ragr-research/users/wh8114/projects/.venv/bin/activate


for num_trees in 8 16 32 64 128 256 512 1024
do
    for trial in $(seq $trial_start $trial_stop)
    do
        trees_dir=$working_dir/simulated_data/branching_process_experiment/$rate_matrix/trees_$num_trees/$stopping_text/trial_$trial
        trees_file=$trees_dir/trees.pkl
        echo $trees_file

        outdir=$working_dir/results/$experiment_name/trees_$num_trees/$stopping_text/trial_$trial
        mkdir -p $outdir
        echo trial=$trial num_trees=$num_trees

        python scripts/get_observed_potencies.py -i $trees_file \
                                                -o $outdir \
                                                -r $rate_matrix_path

        python scripts/save_potency_sets.py \
                -p $max_num_hidden_states \
                -i $trees_file \
                -o $outdir \
                -l $outdir/terminal_labels.txt \
                -k $outdir/observed_potencies.txt \
                -n $num_terminals \
                --unconstrained_unobserved

        logfile=$outdir/inference
        sbatch -t 2:00:00 -c $num_cores -J "$trial|$num_trees|$regularization" -o $logfile.log -e $logfile.err \
                scripts/single_run_inference.sh $max_num_hidden_states \
                                                $trees_file \
                                                $outdir \
                                                $regularization \
                                                $outdir/terminal_labels.txt \
                                                $outdir/observed_potencies.txt \
                                                $subsampling_rate
    done
done