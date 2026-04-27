
source $PWD/.venv/bin/activate
SAMPLE_P=0.5
python scripts/run_classe_troupe.py \
    -i $PWD/experiments/cardiac/processed_data/trees.pkl \
    -o $PWD/results/cardiac/sample_$SAMPLE_P \
    --regularizations 0.0001 0.001 0.01 0.1 1 10 100 \
    --sampling_probability $SAMPLE_P \
    --terminal_labels $PWD/experiments/cardiac/processed_data/terminal_labels.txt \
    --observed_potencies $PWD/experiments/cardiac/processed_data/observed_potencies.txt \
    --reachability_threshold 1e-4

# # Uncomment later...
# # New as of 4/2 12:00: Group LASSO
# python scripts/run_classe_troupe.py \
#   -i "$PWD/experiments/$TLS_TYPE/processed_data/trees.pkl" \
#   -o "$PWD/results/"$TLS_TYPE"_group_lasso/sample_$SAMPLE_P" \
#   --regularizations 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100 \
#   --sampling_probability $SAMPLE_P \
#   --terminal_labels "$PWD/experiments/$TLS_TYPE/terminal_labels.txt" \
#   --observed_potencies "$PWD/experiments/$TLS_TYPE/observed_potencies.txt" \
#   --phase1_penalty column_group_lasso
