#!/bin/bash
#SBATCH --job-name=troupe_cardiac
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=troupe_cardiac_%j.out

source $PWD/../.venv/bin/activate
SAMPLE_P=0.5
python scripts/run_classe_troupe.py \
    -i $PWD/experiments/cardiac/processed_data/trees.pkl \
    -o $PWD/results/cardiac/sample_$SAMPLE_P \
    --regularizations 0.0001 0.001 0.01 0.1 1 10 100 \
    --sampling_probability $SAMPLE_P \
    --terminal_labels $PWD/experiments/cardiac/processed_data/terminal_labels.txt \
    --observed_potencies $PWD/experiments/cardiac/processed_data/observed_potencies.txt \
    --reachability_threshold 1e-4
