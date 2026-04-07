
echo "Running script from root dir $PWD..."

for num_trees in 8
# for num_trees in 16 32 64 128 256 512
do
    for sample_prob in 0.05 0.1 0.2
    # for sample_prob in 0.1 0.2 0.4 0.8 1.0
    do    
        python scripts/simulate_data.py \
            -b 10 \
            -t 5.0 \
            -n $num_trees \
            -s $sample_prob \
            -r \
            -o $PWD/experiments/subsampled_leaves_4_terminals
    done
done