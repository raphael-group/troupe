#!/bin/bash

indir=$1
trees_file=$2
outdir=$3
regularization=$4
terminal_labels=$5
observed_potency=$6
subsampling_rate=$7

echo "single_run_inference command line args 1-7"
echo $indir
echo $trees_file
echo $outdir
echo $regularization
echo $terminal_labels
echo $observed_potency
echo $subsampling_rate
echo ""
echo ""

source /n/fs/ragr-research/users/wh8114/projects/.venv/bin/activate

working_dir=/n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml
inferred_model_info_path=$indir/inferred_model_info_init.pkl
inferred_potency_path=$indir/inferred_potency.pkl

# echo ""
# echo "Extracting potency sets..."
# # Extracts the learned potency set from the model
# python scripts/extract_potencies.py \
#     -i $outdir \
#     --terminal_label_path $terminal_labels \
#     --observed_potency_path $observed_potency

echo ""
echo "Running MLE on smaller potency-constrained paramaters..."
# Infers the model given the new potency set on a restricted set of states
mkdir -p $outdir/select_potencies
python scripts/infer_model.py \
    -i $trees_file \
    -o $outdir/select_potencies \
    --nmp_transition_strength $regularization \
    --potency_path $inferred_potency_path \
    --model_type PureBirthLikelihoodModel \
    --observed_potencies_path $observed_potency \
    --terminal_label_path $terminal_labels \
    -s $subsampling_rate \
    --initialized_model_info_path $inferred_model_info_path