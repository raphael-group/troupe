#!/bin/bash
source "$(dirname "$0")/../env.conf"

experiment_dir=$BASE_DIR/experiments/varied_growth_rates
trees_dir=$experiment_dir/processed_data

for trial in 0 1 2 3 4
do
    outdir=$experiment_dir/results/trial_$trial
    mkdir -p $outdir
    trees_file=$trees_dir/trial_$trial/trees.pkl

    sbatch -t 6:00:00 -c 2 -J $trial'_varied_growth_rates' -o $outdir/logfile.log -e $outdir/logfile.err \
    /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/scripts/varied_growth_rates_experiment/run_single_trial.sh \
    $trees_file $outdir $VENV_PATH
done