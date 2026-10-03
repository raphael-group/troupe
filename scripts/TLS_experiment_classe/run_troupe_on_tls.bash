
source $PWD/.venv/bin/activate
TLS_TYPE=$1
SAMPLE_P=0.05
python scripts/run_classe_troupe.py \
    -i $PWD/experiments/$TLS_TYPE/processed_data/trees.pkl \
    -o $PWD/results/$TLS_TYPE/sample_$SAMPLE_P \
    --regularizations 10 3 1 0.3 0.1 0.03 \
    --sampling_probability $SAMPLE_P \
    --terminal_labels $PWD/experiments/$TLS_TYPE/terminal_labels.txt \
    --observed_potencies $PWD/experiments/$TLS_TYPE/observed_potencies.txt

# --phase2_num_restarts 5
# --regularizations 0.01 0.03 0.1 0.3 1 1.3 1.9 2.3 2.4 2.5 2.6 2.7 2.8 2.9 0.3 1 3 10 30 100 \
# 10 3 1 0.3 0.1 0.03

# NOTE: This is for group lasso version of TROUPE
# python scripts/run_classe_troupe.py \
#   -i "$PWD/experiments/$TLS_TYPE/processed_data/trees.pkl" \
#   -o "$PWD/results/"$TLS_TYPE"_group_lasso/sample_$SAMPLE_P" \
#   --regularizations 0.1 0.3 1 3 10 30 100 \
#   --sampling_probability $SAMPLE_P \
#   --terminal_labels "$PWD/experiments/$TLS_TYPE/terminal_labels.txt" \
#   --observed_potencies "$PWD/experiments/$TLS_TYPE/observed_potencies.txt" \
#   --phase1_penalty column_group_lasso \
#   --phase2_penalty l1
