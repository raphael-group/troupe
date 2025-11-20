"""
...
"""

import pickle
import json
import sys
import argparse
import os
import warnings
import numpy as np

from utils import get_idx2potency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path")
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-r", "--rate_path")
    parser.add_argument("-u", "--unconstrained", action="store_true",
                        help="Set this flag to make all observed types have maximum potency.")
    
    args = parser.parse_args()

    with open(args.input_path, "rb") as fp:
        tree_list = pickle.load(fp)
        tree_length = 1.75

    with open(args.rate_path, "r") as fp:
        info_dict = json.load(fp)
        ground_truth_np = np.array(info_dict["rate_matrix"])

    potency_mapping = get_idx2potency(ground_truth_np, tree_length=tree_length)
    terminal_states = [state for state, potency in potency_mapping.items() if len(potency) == 1]

    if args.unconstrained:
        max_potency = tuple(sorted(terminal_states))
        potency_mapping = {state: max_potency for state in potency_mapping.keys()}

    print("potency mapping:", potency_mapping)

    observed_states = set()
    for tree in tree_list:
        for leaf in tree.get_leaves():
            observed_states.add(leaf.state)

    print("observed states:", observed_states)
    
    with open(f"{args.output_path}/observed_potencies.txt", "w") as fp:
        for state in observed_states:
            line = f"{state}\t"
            assert len(potency_mapping[state]) > 0
            for terminal in potency_mapping[state]:
                line+= f"{terminal},"
            line = line[:-1] + "\n"
            fp.write(line)

    with open(f"{args.output_path}/terminal_labels.txt", "w") as fp:
        for state in terminal_states:
            line = f"{state}\n"
            fp.write(line)
                    


if __name__ == "__main__":
    main()