#!/bin/bash
max_num_hidden_states=$1
trees_file=$2
outdir=$3
regularization=$4
terminal_labels=$5
observed_potency=$6
subsampling_rate=$7

source /n/fs/ragr-research/users/wh8114/projects/.venv/bin/activate

# python scripts/infer_model.py \
#     -i $trees_file \
#     -o $outdir \
#     --l1_regularization $regularization \
#     -p $outdir/potency.pkl \
#     --num_hidden_states $max_num_hidden_states \
#     --model_type PureBirthLikelihoodModel \
#     -l $terminal_labels \
#     --observed_potencies_path $observed_potency \
#     -s $subsampling_rate

if (( "$regularization" > 0 )); then
    echo ""
    echo "Extracting potency sets..."
    # Extracts the learned potency set from the model
    python scripts/extract_potencies.py -i $outdir --terminal_label_path $terminal_labels

    echo ""
    echo "Running MLE on smaller potency-constrained paramaters..."
    # Infers the model given the new potency set on a restricted set of states
    mkdir $outdir/no_reg
    python scripts/infer_model.py \
        -i $trees_file \
        -o $outdir/no_reg \
        --l1_regularization 0 \
        --potency_path $outdir/inferred_potency.pkl \
        --model_type PureBirthLikelihoodModel \
        --observed_potencies_path $observed_potency \
        -s $subsampling_rate \
        --initialized_model_info_path $outdir/model_info_inferred_init.pkl
