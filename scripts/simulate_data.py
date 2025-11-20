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
from branching_simulation import simulate_tree

seed_val = 123
rng = np.random.default_rng(seed=seed_val)

def main():
    parser = argparse.ArgumentParser(
        description="CLI wrapper for simulate_tree_state in src/simulation.py."
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
        default=9999,
        help="Time cutoff for simulation (default: 9999)."
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
        default="/n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/simulated_data/branching_process_experiment",
        help="Directory to save simulated data in."
    )
    parser.add_argument(
        "-v", "--rate_matrix",
        type=int
    )
    args = parser.parse_args()

    base_dir = "/n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml"
    input_dir = f"{base_dir}/scripts/branching_process_experiment/model_params"


    matrix_idxs = [int(args.rate_matrix)]

    rate_matrices = {}
    growth_rates = {}
    init_distributions = {}
    for i in matrix_idxs:
        filename = f"{input_dir}/rate_matrix_{i}.json"
        with open(filename, 'r') as f:
            data_dict = json.load(f)
        rate_matrices[i] = np.array(data_dict["rate_matrix"])
        growth_rates[i] = np.array(data_dict["growth_rates"])
        if "init_distribution" in data_dict:
            init_distributions[i] = np.array(data_dict["init_distribution"])


    # Each simulation needs a different seed
    # TODO: Change this to be samples without replacement and choose exactly args.num_trees * args.num_trials * len(matrix_idxs) seeds
    num_seeds = args.num_trees * args.num_trials * len(matrix_idxs)
    seeds = [rng.choice(num_seeds, replace=False) for _ in range(num_seeds)]

    for i in range(len(matrix_idxs)):
        rate_matrix_idx = matrix_idxs[i]
        Q = np.array(rate_matrices[rate_matrix_idx])
        lam = np.array(growth_rates[rate_matrix_idx])
        
        terminals = []
        for l in range(len(Q)):
            if abs(Q[l, l]) <= len(Q) * 1e-2:    # TODO: This threshold needs to adapt
                terminals.append(l)

        long_term_transition_mat = expm(100 * Q)
        potency_sets = []
        num_progenitors = len(Q) - len(terminals)
        for l in range(num_progenitors):
            potency_idx = l + len(terminals)
            potency = []
            for k in terminals:
                if long_term_transition_mat[potency_idx, k] > 1e-6:
                    potency.append(k)
            if len(potency) > 0:
                potency_sets.append(potency)

        # print()
        # print("Rate matrix:")
        # print(Q)
        # print("Long-term TPM:")
        # print(long_term_transition_mat)
        # print("terminal states:", terminals)
        # print("potency sets:   ", potency_sets)

        rate_dir = f"{args.out_dir}/{rate_matrix_idx}"

        os.makedirs(rate_dir, exist_ok=True)
        simulation_info = {
            "Q": Q,
            "lam": lam,
            "terminals": terminals,
            "potency_sets": potency_sets
        }
        with open(f"{rate_dir}/simulation_info.pkl", "wb") as fp:
            pickle.dump(simulation_info, fp)
        with open(f"{rate_dir}/Q.pkl", "wb") as fp:
            pickle.dump(Q, fp)

        if matrix_idxs[i] not in init_distributions:
            # TODO: Deprecate this
            starting_state = len(terminals) 
            warnings.warn(f"Automatically set starting state to be idx {starting_state}. " + \
                           "If this was not intentional, please add an initial distribution to the rate matrix")
        else:
            starting_state = int(np.argmax(init_distributions[matrix_idxs[i]]))    # NOTE: Currently assuming that this is a [1, 0, ... 0] distribution
            print("Starting state is", starting_state)

        
        for trial in range(args.num_trials):
            temp_str = f"time_{args.time}"
            output_dir = f"{rate_dir}/trees_{args.num_trees}/{temp_str}/trial_{trial}"
            # output_dir = f"{rate_dir}/{temp_str}/trees_{args.num_trees}/trial_{trial}"
            os.makedirs(output_dir, exist_ok=True)
            tree_list = []
            num_terminal = 0
            total_leaves = 0
            total_type_counts = Counter()
            for j in range(args.num_trees):
                seed = seeds[j + trial * args.num_trees + i * args.num_trees * trial]

                tree = simulate_tree(Q,
                                     lam,
                                     starting_state,
                                     T=args.time,
                                     seed=seed)

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
            for tree in tree_list:
                clone_size.append(len(tree.get_leaves()))
                clone_depth.append(tree.get_farthest_leaf(topology_only=True)[1])

            # Print out clone stats
            print()
            print(f"{trial} Clone stats:")
            print(f"\tprop terminal:  {num_terminal / total_leaves}")
            print(f"\tavg size:       {sum(clone_size) / len(clone_size)}")
            print(f"\tmax size:       {max(clone_size)}")
            print(f"\tavg depth:      {sum(clone_depth) / len(clone_depth)}")
            print(f"\tmax depth:      {max(clone_depth)}")
            print(f"\ttype counts:    {OrderedDict(total_type_counts)}")
            print()




if __name__ == "__main__":
    main()