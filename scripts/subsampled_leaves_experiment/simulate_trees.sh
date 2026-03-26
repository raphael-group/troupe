
for num_trees in 64 128
# for num_trees in 16 32 64 128 256 512
do
    for sample_prob in 0.1 0.2
    # for sample_prob in 0.1 0.2 0.4 0.8 1.0
    do    
        python scripts/simulate_data.py \
            -b 10 \
            -t 5.0 \
            -n $num_trees \
            -s $sample_prob \
            -r \
            -o /Users/william_hs/Desktop/Projects/troupe/experiments/subsampled_leaves_4_terminals
    done
done