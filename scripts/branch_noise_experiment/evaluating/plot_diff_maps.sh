
source /n/fs/ragr-research/users/wh8114/projects/.venv/bin/activate

for num_trees in 16 32 64
do
    for sample_prob in 0.05 0.1 0.2
    do
        for trial in 0 1 2 3 4 5 6 7 8 9
        do

            python scripts/evaluate_results.py plot-differentiation-maps \
                -i $PWD/results/subsampled_leaves_4_terminals/classe/fundamental/trees_$num_trees/time_5.0/sample_$sample_prob/trial_$trial

        done
    done
done