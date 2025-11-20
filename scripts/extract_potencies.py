"""
Given the path to an input_dir that has a model_dict.pkl for an overparameterized model run,
extract its potency sets and save results at input_dir/inferred_potency.pkl and save
the pruned matrix (with just reachable idxs) as model_dict_init.pkl.

Example usage:
python scripts/extract_potencies.py \
    -i /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/experiments/TLS/results_vanilla_likelihood/reg=1 \
    -t /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/experiments/TLS/terminal_labels.txt

python scripts/infer_model.py \
    -i /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/experiments/TLS/processed_data/trees.pkl \
    -o /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/experiments/TLS/results_vanilla_likelihood/reg=1/no_reg \
    --l1_regularization 0 \
    -p /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/experiments/TLS/results_vanilla_likelihood/reg=1/inferred_potency.pkl \
    --model_type PureBirthLikelihoodModel \
    -s 1.0 \
    --observed_potencies_path /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/experiments/TLS/observed_potencies.txt \
    --initialized_model_info_path /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/experiments/TLS/results_vanilla_likelihood/reg=1/inferred_model_info_init.pkl


python scripts/extract_potencies.py \
    -i /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/results/sample_efficiency_experiment/rate_matrix_12/relaxed/trees_256/time_2.35/trial_0/reg=1 \
    -t /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/results/sample_efficiency_experiment/rate_matrix_12/relaxed/trees_256/time_2.35/trial_0/terminal_labels.txt \
    -p /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/results/sample_efficiency_experiment/rate_matrix_12/relaxed/trees_256/time_2.35/trial_0/observed_potencies.txt

python scripts/extract_potencies.py \
    -i /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/results/sample_efficiency_experiment/rate_matrix_12/relaxed/trees_512/time_2.35/trial_4/reg=10 \
    -t /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/results/sample_efficiency_experiment/rate_matrix_12/relaxed/trees_512/time_2.35/trial_4/terminal_labels.txt \
    -p /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/results/sample_efficiency_experiment/rate_matrix_12/relaxed/trees_512/time_2.35/trial_4/observed_potencies.txt
"""

import pickle
import json
import argparse
import numpy as np
import torch

from utils import get_terminal_labels, get_idx2potency, get_reachable_idxs, get_observed_potencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir")
    parser.add_argument("-t", "--terminal_label_path")
    parser.add_argument("-p", "--observed_potency_path")
    args = parser.parse_args()

    working_dir = args.input_dir

    with open(f"{working_dir}/model_dict.pkl", "rb") as fp:
        model_dict = pickle.load(fp)
        inferred_rate_matrix = model_dict["rate_matrix"]
        inferred_growth_rates = model_dict["growth_rates"]
        inferred_rate_matrix_np = inferred_rate_matrix.detach().numpy()
        idx2state = model_dict["idx2state"]
        is_int_state = any([isinstance(state, int) for state in idx2state.values()])
        starting_idx = torch.argmax(model_dict["root_distribution"]).detach().item()
    reachable_idxs = get_reachable_idxs(inferred_rate_matrix_np, starting_idx, threshold=0.00001)
    terminal_labels = get_terminal_labels(args.terminal_label_path, is_int_state)
    observed_potency = get_observed_potencies(args.observed_potency_path, is_int_state)

    print('terminal_labels')
    print(terminal_labels)
    print('observed_potency')
    print(observed_potency)
    print("reachable_idxs")
    print(reachable_idxs)

    old2newidx = {old_idx: i for i, old_idx in enumerate(reachable_idxs)}
    newidx2state = {old2newidx[idx]: idx2state[idx] for idx in reachable_idxs}
    state2newidx = {state: newidx for newidx, state in newidx2state.items()}

    print('old2newidx')
    print(old2newidx)
    print('newidx2state')
    print(newidx2state)

    # This may include non-terminal states
    idx2potency_ = get_idx2potency(inferred_rate_matrix_np, tree_length=10)
    idx2potency = {}
    # Remove non-terminals from potencies
    for idx in reachable_idxs:
        potency_ = idx2potency_[idx]
        potency = [idx2state[s] for s in potency_ if idx2state[s] in terminal_labels]
        print(idx, potency)
        if len(potency) == 0:
            print("\tSkipping")
            continue
        potency.sort()
        idx2potency[old2newidx[idx]] = tuple(potency)
    
    # If idx2potency doesn't contain an observed potency, then add it.
    for state, state_potency in observed_potency.items():
        if state2newidx[state] not in idx2potency:
            potency = [s for s in state_potency]
            potency.sort()
            idx2potency[state2newidx[state]] = tuple(potency)

    print('idx2potency')
    print(idx2potency)
    
    potency_list = list(idx2potency.values())
    with open(f"{working_dir}/inferred_potency.pkl", "wb") as fp:
        pickle.dump(potency_list, fp)

    lam = inferred_growth_rates[reachable_idxs]
    Q = inferred_rate_matrix[reachable_idxs, :][:, reachable_idxs]
    initial_idx = old2newidx[starting_idx]

    # invert softplus
    Q = Q.fill_diagonal_(0).clamp_min(1e-20)
    Q_params_init = torch.log(torch.exp(Q) - 1)
    Q_params_init = Q_params_init.fill_diagonal_(0)
    growth_params_init = torch.log(torch.exp(lam.clamp_min(1e-2)) - 1)

    model_dict_inferred = {
        "Q_params_init": Q_params_init,
        "growth_params_init": growth_params_init,
        "start_state": initial_idx,
        "idx2state": newidx2state,
        "idx2potency": idx2potency
    }
    for k, v in model_dict_inferred.items():
        print(k)
        print(v)
    with open(f"{working_dir}/inferred_model_info_init.pkl", "wb") as fp:
        pickle.dump(model_dict_inferred, fp)
    
                    


if __name__ == "__main__":
    main()