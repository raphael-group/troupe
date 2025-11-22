#!/bin/bash

max_num_hidden_states=$1
trees_file=$2
outdir=$3
regularization=$4
terminal_labels=$5
observed_potency=$6
subsampling_rate=$7
working_dir=$8

# echo "single_run_inference command line args 1-7"
# echo $max_num_hidden_states
# echo $trees_file
# echo $outdir
# echo $regularization
# echo $terminal_labels
# echo $observed_potency
# echo $subsampling_rate
# echo ""
# echo ""

source $working_dir/.venv/bin/activate

python scripts/infer_model.py \
    -i $trees_file \
    -o $outdir \
    --l1_regularization $regularization \
    -p $outdir/potency.pkl \
    --num_hidden_states $max_num_hidden_states \
    --model_type PureBirthLikelihoodModel \
    -l $terminal_labels \
    -k $observed_potency \
    --subsampling_rate $subsampling_rate

echo ""
echo "Extracting potency sets..."
python scripts/extract_potencies.py \
    -i $outdir \
    --terminal_label_path $terminal_labels \
    --observed_potency_path $observed_potency

echo ""
echo "Running MLE on smaller potency-constrained paramaters..."
mkdir -p $outdir/select_potencies
python scripts/infer_model.py \
    -i $trees_file \
    -o $outdir/select_potencies \
    --l1_regularization 0.0001 \
    --potency_path $outdir/inferred_potency.pkl \
    --model_type PureBirthLikelihoodModel \
    --observed_potencies_path $observed_potency \
    --terminal_label_path $terminal_labels \
    -s $subsampling_rate \
    --initialized_model_info_path $outdir/inferred_model_info_init.pkl