
# TODO: combine tls and tlsc into one dataset

source $PWD/.venv/bin/activate
# TLS_TYPE=tls_and_tlsc
# python scripts/run_classe_troupe.py \
#     -i $PWD/experiments/$TLS_TYPE/processed_data/trees.pkl \
#     -o $PWD/results/$TLS_TYPE \
#     --regularizations 0.3 1 3 10 30 \
#     --sampling_probability 0.05 \
#     --terminal_labels $PWD/experiments/$TLS_TYPE/terminal_labels.txt \
#     --observed_potencies $PWD/experiments/$TLS_TYPE/observed_potencies.txt \
#     --phase2_num_restarts 4 

# TLS_TYPE=tls_no_endothelial
# python scripts/run_classe_troupe.py \
#     -i $PWD/experiments/$TLS_TYPE/processed_data/trees.pkl \
#     -o $PWD/results/tls_and_tlsc/$TLS_TYPE \
#     --regularizations 0.3 \
#     --sampling_probability 0.05 \
#     --terminal_labels $PWD/experiments/$TLS_TYPE/terminal_labels.txt \
#     --observed_potencies $PWD/experiments/$TLS_TYPE/observed_potencies.txt \
#     --phase2_num_restarts 4 \
#     --skip_phase_1

TLS_TYPE=tlsc
python scripts/run_classe_troupe.py \
    -i $PWD/experiments/$TLS_TYPE/processed_data/trees.pkl \
    -o $PWD/results/tls_and_tlsc/$TLS_TYPE \
    --regularizations 0.3 \
    --sampling_probability 0.05 \
    --terminal_labels $PWD/experiments/$TLS_TYPE/terminal_labels.txt \
    --observed_potencies $PWD/experiments/$TLS_TYPE/observed_potencies.txt \
    --phase2_num_restarts 4 \
    --skip_phase_1