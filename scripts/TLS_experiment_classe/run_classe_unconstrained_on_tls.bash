
source $PWD/.venv/bin/activate
TLS_TYPE=TLS_no_endothelial_new #$1
SAMPLE_P=0.05
python scripts/run_classe_unconstrained.py \
    -i $PWD/experiments/$TLS_TYPE/processed_data/trees.pkl \
    -o $PWD/results/${TLS_TYPE}_classe/sample_$SAMPLE_P \
    --sampling_probability $SAMPLE_P \
    --terminal_states NeuralTube Endoderm Somite PCGLC \
    --num_hidden 2 \
    --num_restarts 20
