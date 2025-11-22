"""
A script for saving potency sets to pickle files. Exits with error code 

E.g., example usage
python scripts/potency_experiment/save_potency_sets.py \
    -p 11 \
    -i /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/simulated_data/branching_process_experiment/2/trees_50/time_2.5/trial_0/trees_only_terminals.pkl \
    -o /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/results/branching_process_experiment/infer_overparametrized/2/trees_50/time_2.5/trial_0

# TODO: Add this to utils_cli.py
"""

import pickle
import sys
import argparse
import os
import warnings
from collections import Counter
from itertools import combinations
from utils import get_terminal_labels, get_observed_potencies


def main():
    parser = argparse.ArgumentParser(
        description="Saves pickled potency sets to `output_dir/potency.pkl`."
    )
    parser.add_argument(
        "-p", "--max_num_potencies",
        type=int,
        help="Number of potencies to add to potency set. Will always include the totipotent progenitor."
    )
    parser.add_argument(
        "-i", "--input_trees_path",
        type=str,
        help="Path to pickled tree list."
    )
    parser.add_argument(
        "-o", "--trial_output_dir",
        type=str,
        help="Output directory of trial."
    )
    parser.add_argument(
        "-n", "--num_terminals",
        type=int,
        default=-1,
        help="The size of the largest potency set."
    )
    parser.add_argument(
        "-l", "--terminal_label_path",
        type=str,
        help="The path to the set of terminal labels."
    )
    parser.add_argument(
        "-k", "--observed_potencies_path",
        type=str,
        help="The path to the set of known potencies."
    )
    parser.add_argument(
        "-u", "--unconstrained_unobserved",
        action="store_true",
        help="Flag that, if set, makes it so that all unobserved states are totipotent."
    )
    parser.add_argument(
        "-d", "--unconstrained_observed",
        action="store_true",
        help="Flag that, if set, makes it so that all observed states are totipotent."
    )
    parser.add_argument(
        "-e", "--exact",
        action="store_true",
        help="Flag that, if set, makes it so that potencies are saved to outdir/i/potency.pkl."
    )
    args = parser.parse_args()

    with open(args.input_trees_path, "rb") as fp:
        trees = pickle.load(fp)
        is_int_state = isinstance(list(trees[0].get_leaves())[0].state, int)
        print("is_int_state:", is_int_state)

    observed_potencies = get_observed_potencies(args.observed_potencies_path, is_int_state)
    terminal_labels = get_terminal_labels(args.terminal_label_path, is_int_state)
    max_potency = tuple(sorted(list(terminal_labels)))

    if args.unconstrained_unobserved:

        if args.unconstrained_observed:
            num_hidden_progenitors = len(terminal_labels) + args.max_num_potencies
            potency_list = []
        else:
            potency_list = list(observed_potencies.values())
            num_observed_progenitors = 0
            for potency in potency_list:
                if len(potency) > 1:
                    num_observed_progenitors += 1
            num_hidden_progenitors = args.max_num_potencies - num_observed_progenitors

        for _ in range(num_hidden_progenitors):
            potency_list.append(max_potency)
        
        os.makedirs(args.trial_output_dir, exist_ok=True)
        with open(f"{args.trial_output_dir}/potency.pkl", "wb") as fp:
            sorted_potency_list = sorted(potency_list, key=len, reverse=True)
            pickle.dump(sorted_potency_list, fp)

    else:
        induced_potencies = set()
        for tree in trees:
            counter = Counter()
            for node in tree.traverse("postorder"):
                if node.is_leaf():
                    if node.state in terminal_labels:
                        clade = set([node.state])
                        counter[node.state]+=1
                    else:
                        clade = set()
                    node.add_feature("clade", clade)
                else:
                    clade = set()
                    for child in node.get_children():
                        clade = clade.union(child.clade)
                    node.add_feature("clade", clade)

                if len(clade) > 0:
                    list_clade = list(clade)
                    list_clade.sort()
                    induced_potencies.add(tuple(list_clade))

        potencies_to_add = set()
        for potency in induced_potencies:
            assert len(potency) > 0
            if len(potency) > 1:
                for state in potency:
                    potency_list = list(potency)
                    potency_list.remove(state)
                    potencies_to_add.add(tuple(potency_list))
            for state in terminal_labels:
                if state in potency:
                    continue
                else:
                    potency_list = list(potency)
                    potency_list.append(state)
                    potency_list.sort()
                    potencies_to_add.add(tuple(potency_list))
        
        potencies = induced_potencies.union(potencies_to_add)
        potency_list = list(potencies)
        potency_list.sort(key=len)
        largest_potency = potency_list[-1]

        # Remove known potencies (i.e., totipotent and each observed potency)
        removed_potencies = []
        if len(largest_potency) == len(terminal_labels):
            potencies.remove(largest_potency)
            removed_potencies.append(largest_potency)
        to_remove = sorted(list(observed_potencies.values()), key=len)

        if largest_potency in to_remove:
            to_remove = to_remove[:-1]
        for potency in to_remove:
            if potency == largest_potency:
                continue
            assert potency in potencies
            potencies.remove(potency)
            removed_potencies.append(potency)

        num_unobserved_potencies = args.max_num_potencies - (len(removed_potencies) - len(terminal_labels))
        k = min(num_unobserved_potencies, len(potencies))
        assert k >= 0
        size_k_subsets = list(combinations(potencies, k))
        for i, subset in enumerate(size_k_subsets):
            subset_list = list(subset)

            # Add known potencies back in
            for potency in removed_potencies:
                subset_list.append(potency)
            # print(f"{i} \t {subset_list}")

            if len(size_k_subsets) == 1 and not args.exact:
                output_dir = args.trial_output_dir
            else:
                output_dir = f"{args.trial_output_dir}/{i}"

            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/potency.pkl", "wb") as fp:
                sorted_subset_list = sorted(subset_list, key=len, reverse=True)
                pickle.dump(sorted_subset_list, fp)

                    


if __name__ == "__main__":
    main()