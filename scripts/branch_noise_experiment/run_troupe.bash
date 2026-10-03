
source $PWD/.venv/bin/activate
SAMPLE_P=0.05

for trial in 1 2 3 4 5 6 7 8 9
do
    python scripts/run_classe_troupe.py \
        -i $PWD/experiments/branch_noise_experiment/trees_24/time_1.75/sample_0.05/trial_$trial/trees_zeroed_branches.pkl \
        -o $PWD/results/branch_noise_experiment/trees_24/time_1.75/sample_0.05/zeroed_edges/trial_$trial \
        --regularizations 3 1 0.9 0.7 0.5 0.3 0.1 \
        --sampling_probability $SAMPLE_P

    python scripts/run_classe_troupe.py \
        -i $PWD/experiments/branch_noise_experiment/trees_24/time_1.75/sample_0.05/trial_$trial/trees.pkl \
        -o $PWD/results/branch_noise_experiment/trees_24/time_1.75/sample_0.05/full_edges/trial_$trial \
        --regularizations 3 1 0.9 0.7 0.5 0.3 0.1 \
        --sampling_probability $SAMPLE_P 
done