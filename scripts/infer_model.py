"""
This is the main script for inferring the model

"""

import torch
from ete3 import TreeNode

from utils import get_terminal_labels, get_observed_potencies
from optimizer import compute_mle

import copy
import pickle
import argparse
import os
import warnings


def main():
    parser = argparse.ArgumentParser(
        description="Infer the rate matrix and progenitor distribution via MLE."
    )
    parser.add_argument(
        "-o", "--out_dir",
        type=str,
        help="Directory to put results in."
    )
    parser.add_argument(
        "-i", "--input_trees_path",
        type=str,
        help="Pickle file with list of simulated ete trees."
    )
    parser.add_argument(
        "-r", "--l1_regularization",
        type=float,
        default=0.0,
        help="The amount of l1 regularization to use in loss function."
    )
    parser.add_argument(
        "-g", "--group_lasso_regularization",
        type=float,
        default=0.0,
        help="The amount of group lasso regularization to use in loss function."
    )
    parser.add_argument(
        "-t", "--nmp_transition_strength",
        type=float,
        default=0.0,
        help="Used to constrain both NMP transitions to have equal rates."
    )
    parser.add_argument(
        "-n", "--num_hidden_states",
        type=int,
        default=0,
        help="The number of extra states to add to the rate matrix."
    )
    parser.add_argument(
        "-s", "--subsampling_rate",
        type=float,
        default=1.0,
        help="The proportion of observed leaves (out of the total number of leaves)."
    )
    parser.add_argument(
        "-a", "--start_state",
        type=int,
        default=None,
        help="The state that the process starts in."
    )
    parser.add_argument(
        "-m", "--model_type",
        type=str,
        default="PotencyLikelihoodModel",
        help="Name of model class to use. See `src/models.py`."
    )
    parser.add_argument(
        "-p", "--potency_path",
        type=str,
        default="",
        help="Path to pickled potency list to constrain inference w.r.t.. If 'unconstrained' will assume that there are no potency sets."
    )
    parser.add_argument(
        "-l", "--terminal_label_path",
        type=str,
        default=None,
        help="Path to text file with terminal labels."
    )
    parser.add_argument(
        "-k", "--observed_potencies_path",
        type=str,
        default=None,
        help="The path to the set of known potencies."
    )
    parser.add_argument(
        "-b", "--initialized_model_info_path",
        type=str,
        default=None,
        help="Optionally can initialize model info with various parameters (e.g., Q, lambda, idx2state, etc)"
    )
    args = parser.parse_args()
    assert args.input_trees_path is not None and args.out_dir is not None

    model_info = None
    if args.initialized_model_info_path is not None:
        with open(args.initialized_model_info_path, "rb") as fp:
            model_info = pickle.load(fp)
            model_info["subsampling_rate"] = args.subsampling_rate
            print("model info at initialization:")
            for k,v in model_info.items():
                print(k, v)

    with open(args.input_trees_path, "rb") as fp:
        trees = pickle.load(fp)
        is_int_state = isinstance(list(trees[0].get_leaves())[0].state, int)
    states = set()
    for tree in trees:
        for leaf in tree.get_leaves():
            states.add(leaf.state)
    num_obs_states = len(states)

    observed_potencies = get_observed_potencies(args.observed_potencies_path, is_int_state)
    terminal_labels = get_terminal_labels(args.terminal_label_path, is_int_state)
    
    print(f"=> Using provided potency sets at {args.potency_path}")
    with open(args.potency_path, "rb") as fp:
        potency_sets = pickle.load(fp)
        num_hidden_states = len(potency_sets) - len(observed_potencies)
    print("Potency sets:")
    for potency in potency_sets:
        print(potency)
    print("Number of hidden states:  ", num_hidden_states)
    print("Number of observed states:", num_obs_states)

    if model_info is None:

        num_states = num_obs_states + num_hidden_states
        
        # Index the observed leaf states
        state_list = list(states)
        state_list.sort()
        # Index observed states
        state2idx = {state: i for i, state in enumerate(state_list)}
        # Index hidden states as well
        observed_idxs = set(state2idx.values())
        all_idxs = set(range(num_states))
        unobserved_idxs = all_idxs.difference(observed_idxs)
        for idx in unobserved_idxs:
            state_name = f"U{idx}"
            state2idx[state_name] = idx
        idx2state = {i: state for state, i in state2idx.items()}

        if potency_sets is not None:
            idx2potency = {}
            unobserved_potencies = copy.deepcopy(potency_sets)
            # Remove all observed potencies
            for potency in observed_potencies.values():
                i = unobserved_potencies.index(potency)
                assert i != -1
                unobserved_potencies.pop(i)
            
            for idx, state in idx2state.items():
                if state in observed_potencies.keys():
                    potency = observed_potencies[state]
                else:
                    potency = unobserved_potencies.pop(-1)
                idx2potency[idx] = potency
                
            potency_list = list(potency_sets)
            potency_list.sort(key=len)
            largest_potency = potency_list[-1]
            
            is_unconstrained = all([len(potency)==len(terminal_labels) for potency in observed_potencies.values()])
            start_state = args.start_state
            if not is_unconstrained:
                for idx, potency in idx2potency.items():
                    if potency == largest_potency:
                        start_state = idx
                        break
                assert start_state is not None
        else:
            start_state = args.start_state
        
        model_info = {
            "idx2potency": idx2potency,
            "idx2state": idx2state,
            "start_state": start_state, # NOTE: This needs to be the index of the starting state or None
            "terminal_states": [state2idx[state] for state in terminal_labels],
            "subsampling_rate": args.subsampling_rate
        }
    
    state2idx = {state: idx for idx, state in model_info["idx2state"].items()}

    print("idx -> state")
    for idx, state in model_info["idx2state"].items():
        print(idx, "\t", state)
    print("idx -> potency")
    for idx, potency in model_info["idx2potency"].items():
        print(idx, "\t", potency)

    # Relabel the leaf states of the tree
    for tree in trees:
        for leaf in tree.get_leaves():
            leaf.state = state2idx[leaf.state]
            
    device = torch.device('cpu')

    llh, loss = compute_mle(trees,
                            (num_obs_states, num_hidden_states),
                            device,
                            args.out_dir,
                            l1_regularization_strength=args.l1_regularization,
                            group_lasso_strength=args.group_lasso_regularization,
                            nmp_transition_strength=args.nmp_transition_strength,
                            model_type=args.model_type,
                            model_info=model_info)
            
    with open(f"{args.out_dir}/loss.txt", "w") as fp:
        fp.write(f"{loss}")
    

if __name__ == "__main__":
    main()