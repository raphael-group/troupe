"""
Simulates cell tree according to a pure-birth process.

Example usage:
    python scripts/simulate_data.py \
    -b 5 \
    -t 2.35 \
    -v 12 \
    -n 512

Each simulated tree is saved to the following file:
    simulated_data/branching_process_experiment/<rate-matrix-idx>/<num-trees>/<num-leaves>/<trials-num>/trees.pkl
"""

import argparse
import os
import pickle
import copy
from collections import Counter, OrderedDict
import json
from scipy.linalg import expm
import warnings

import numpy as np
# from branching_simulation import simulate_tree
from classe_branching_simulation import simulate_tree

def main():
    parser = argparse.ArgumentParser(
        description="CLI wrapper for simulate_tree."
    )
    parser.add_argument(
        "-n", "--num_trees",
        type=int,
        default=5,
        help="Number of trees to simulate (default: 5)."
    )
    parser.add_argument(
        "-t", "--time",
        type=float,
        default=1.0,
        help="Time cutoff for simulation (default: 1.0)."
    )
    parser.add_argument(
        "-s", "--sample_probability",
        type=float,
        default=1.0,
        help="Probability of sampling each leaf (default: 1.0)."
    )
    parser.add_argument(
        "-b", "--num_trials",
        type=int,
        default=5,
        help="Number of trials to simulate independently (default: 5)."
    )
    parser.add_argument(
        "-o", "--out_dir",
        type=str,
        help="Directory to save simulated data in. Assumes model JSON file is located in out_dir/model.json"
    )
    parser.add_argument(
        "-r", "--remove_progenitors",
        action="store_true",
        help="Determines whether to prune progenitor types (after subsampling)."
    )
    args = parser.parse_args()

    input_path = f"{args.out_dir}/model.json"

    rng = np.random.default_rng(seed=1234)

    with open(input_path, 'r') as f:
        data_dict = json.load(f)
    birth_kernel = np.array(data_dict["birth_kernel"])
    growth_rates = np.array(data_dict["growth_rates"])
    init_distribution = np.array(data_dict["init_distribution"])


    terminals = [str(i) for i in range(len(birth_kernel)) if birth_kernel[i,i] == 1.0]

    # Each simulation needs a different seed
    num_seeds = args.num_trees * args.num_trials
    seeds = [rng.choice(num_seeds, replace=False) for _ in range(num_seeds)]

    for trial in range(args.num_trials):
        output_dir = f"{args.out_dir}/trees_{args.num_trees}/time_{args.time}/sample_{args.sample_probability}/trial_{trial}"
        os.makedirs(output_dir, exist_ok=True)
        tree_list = []
        num_terminal = 0
        total_leaves = 0
        total_type_counts = Counter()
        for j in range(args.num_trees):
            seed = seeds[j + trial * args.num_trees]

            tree = simulate_tree(birth_kernel,
                                growth_rates,
                                np.argmax(init_distribution),
                                T=args.time,
                                seed=seed,
                                sample_probability=args.sample_probability)

            if tree is None:
                continue
                
            if args.remove_progenitors:
                leaves = tree.get_leaves()
                to_keep = [leaf for leaf in leaves if leaf.state in terminals]
                if len(to_keep) == 0:
                    continue
                tree.prune(to_keep)


            num_leaves = len(tree.get_leaves())
            total_leaves += num_leaves

            node_state_counter = Counter()
            leaf_state_counter = Counter()
            terminal_state_nodes = []
            for node in tree.traverse():
                node_state_counter[node.state] += 1
                if node.is_leaf():
                    leaf_state_counter[node.state] += 1
                    total_type_counts[node.state] += 1
                if node.is_leaf() and node.state in terminals:
                    terminal_state_nodes.append(node)
            num_terminal += len(terminal_state_nodes)
            tree_list.append(tree)

        out_filename = "trees"
        with open(f"{output_dir}/{out_filename}.pkl", "wb") as fp:
            pickle.dump(tree_list, fp)

        clone_size = []
        clone_depth = []
        leaf_composition = Counter()
        for tree in tree_list:
            clone_size.append(len(tree.get_leaves()))
            clone_depth.append(tree.get_farthest_leaf(topology_only=True)[1])

            leaf_states = list(set([leaf.state for leaf in tree.get_leaves()]))
            leaf_states.sort()
            leaf_types = tuple(leaf_states)
            leaf_composition[leaf_types] += 1

        # Print out clone stats
        print()
        print(f"{trial} Clone stats:")
        print(f"\tprop terminal:  {num_terminal / total_leaves}")
        print(f"\tavg size:       {sum(clone_size) / len(clone_size)}")
        print(f"\tmax size:       {max(clone_size)}")
        print(f"\tavg depth:      {sum(clone_depth) / len(clone_depth)}")
        print(f"\tmax depth:      {max(clone_depth)}")
        print(f"\ttype counts:    {OrderedDict(total_type_counts)}")
        print(f"\ttype comp:      {leaf_composition}")
        print()




if __name__ == "__main__":
    main()