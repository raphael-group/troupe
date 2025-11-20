from likelihood import log_vec_likelihood, _prep_log_tree, log_vec_likelihood_numerical
from branching_simulation import simulate_tree
from models import PureBirthLikelihoodModel

import numpy as np
import torch
import ete3
import pytest
import pickle

# device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device =", device)

all_methods = [
    "log_vec_lik",
    "pure_birth_model",
    "log_vec_lik_num",
    "pure_birth_model_num"
]

analytic_methods = [
    "log_vec_lik",
    "pure_birth_model"
]

numerical_methods = [
    "log_vec_lik_num",
    "pure_birth_model_num"
]


def test_subsampling_big():
    Q = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 4.5, -9.0, 4.5, 0.0],
        [0.0, 0.0, 4.0, 0.0, 0.0, -8.0, 4.0],
        [4.0, 4.0, 0.0, 0.0, 0.0, 0.0, -8.0]
    ], dtype= torch.float64, device=device)
    growth_rates = torch.tensor([1, 1, 1, 1, 6, 4, 4], dtype= torch.float64, device=device)
    pi_params = torch.tensor([-1e20, -1e20, -1e20, -1e20, 1e20, -1e20, -1e20], dtype= torch.float64, device=device)
    num_trees = 50
    rho = 0.4

    # Simulate
    trees = []
    for seed in range(num_trees):
        tree = simulate_tree(Q, growth_rates, starting_type=0, T=2.0, seed=seed*17)
        tree = _prep_log_tree(tree, len(Q))
        trees.append(tree)

    import time

    # Reverse the softplus
    Q_params = torch.log(torch.exp(Q)-1)
    Q_params_no_diag = Q_params.fill_diagonal_(0)
    Q_from_params = torch.log(torch.exp(Q_params_no_diag) + 1)
    Q_from_params = Q_from_params - torch.diag(Q_from_params.sum(dim=1))
    assert torch.isclose(Q_from_params, Q).all()
    growth_params = torch.log(torch.exp(growth_rates)-1)
    model = PureBirthLikelihoodModel([tree.copy("deepcopy") for tree in trees],
                                      len(Q), Q_params, pi_params, growth_params,
                                      device=device, num_hidden=0, subsampling_rate=rho)
    model = model.to(device)
    Q_model = model.rate_builder.forward(model.get_Q_params())
    lam_model = model.get_growth_rates()
    model.prepare_numerical_pruner(lam_model, Q_model, rho=rho)

    start = time.time()
    lik_model_numerical = sum([model(i) for i in range(len(trees))])
    elapsed_model_numerical = time.time() - start

    start = time.time()
    lik_numerical = log_vec_likelihood_numerical(trees, Q, pi_params, growth_rates=growth_rates, rho=rho)
    elapsed_numerical = time.time() - start

    print(lik_numerical, elapsed_numerical)
    print(lik_model_numerical, elapsed_model_numerical)
    assert are_all_within_k_sig_figs([lik_numerical.item(), lik_model_numerical.item()])


def test_subsampling():

    Q = torch.tensor([[-1.0, 1.0],
                      [0.0, 0.0]], dtype= torch.float64, device=device)
    lam = torch.tensor([1.0, 1.0])
    growth_rates = torch.tensor([2.0, 1.0], dtype= torch.float64, device=device)
    pi_params = torch.tensor([1e20, -1e20], dtype= torch.float64, device=device)
    num_trees = 20

    # Simulate
    trees = []
    for seed in range(num_trees):
        tree = simulate_tree(Q, lam, starting_type=0, T=1, seed=seed*17)
        tree = _prep_log_tree(tree, len(Q))
        trees.append(tree)

    lik_numerical = log_vec_likelihood_numerical(trees, Q, pi_params, growth_rates=growth_rates, rho=0.999)
    lik = log_vec_likelihood(trees, Q, pi_params, growth_rates=growth_rates)

    print(lik_numerical)
    print(lik)
    assert are_all_within_k_sig_figs([lik_numerical.item(), lik.item()])


def test_non_reversible():
    Q = np.array([[-1.0, 1.0],
                  [0.0, 0.0]])
    lam = np.array([1.0, 1.0])
    pi = torch.tensor([1e20, -1e20])
    num_trees = 5

    # Simulate
    trees = []
    for seed in range(num_trees):
        tree = simulate_tree(Q, lam, starting_type=0, T=1, seed=seed*17)
        tree = _prep_log_tree(tree, len(Q))
        trees.append(tree)
    
    return compare_methods(trees, Q, pi, lam, all_methods)


def test_three_state_non_reversible():
    Q = np.array([[-1.0, 0.5, 0.5],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
    lam = np.array([2.0, 1.0, 1.0])
    pi = torch.tensor([1e20, -1e20, -1e20])
    num_trees = 10

    # Simulate
    trees = []
    for seed in range(num_trees):
        tree = simulate_tree(Q, lam, starting_type=0, T=1, seed=seed*37)
        tree = _prep_log_tree(tree, len(Q))
        trees.append(tree)
    
    return compare_methods(trees, Q, pi, lam, analytic_methods) # all_methods)

def test_big_non_reversible():
    Q = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 4.0, -8.0, 4.0, 0.0],
        [0.0, 0.0, 4.0, 0.0, 0.0, -8.0, 4.0],
        [4.0, 4.0, 0.0, 0.0, 0.0, 0.0, -8.0]
    ])
    lam = np.array([1, 1, 1, 1, 6, 4, 4])
    pi = torch.tensor([-1e20, -1e20, -1e20, -1e20, 1e20, -1e20, -1e20])
    num_trees = 32

    # Simulate
    states = set()
    trees = []
    for seed in range(num_trees):
        tree = simulate_tree(Q, lam, starting_type=4, T=1.75, seed=seed*37)
        tree = _prep_log_tree(tree, len(Q))
        tree_states = {leaf.state for leaf in tree.get_leaves()}
        states = states.union(tree_states)
        trees.append(tree)
    
    print("Observed states")
    print(states)

    
    return compare_methods(trees, Q, pi, lam, analytic_methods) # all_methods)



def are_all_within_k_sig_figs(numbers, k=4):
    """
    Check if all elements in the list are within k significant figures of each other.
    This is determined by rounding each element to k significant figures
    and verifying that all rounded values are equal.

    Parameters:
        numbers (list of float): List of numerical values.
        k (int)                : Number of sig figs.

    Returns:
        bool: True if all values rounded to k significant figures are equal, otherwise False.
    """
    if len(numbers) == 0:
        return False
    for num in numbers:
        assert isinstance(num, float)
    # Format each number with k significant digits.
    rounded_values = {f"{num:.{k}}" for num in numbers if num}
    print(rounded_values)
    return len(rounded_values) == 1


def compare_methods(trees, Q, pi_params, growth_rates, methods):
    scores = {name: [] for name in methods}

    Q_tensor = torch.tensor(Q, device=device)
    pi_params = pi_params.to(device)
    growth_rates = torch.tensor(growth_rates, device=device)

    # Reverse the softplus
    Q_params = torch.log(torch.exp(Q_tensor)-1)
    Q_params.fill_diagonal_(0)
    growth_params = torch.log(torch.exp(growth_rates)-1)

    
    if "pure_birth_model" in methods:
        pure_birth_model = PureBirthLikelihoodModel([tree.copy("deepcopy") for tree in trees],
                                                    len(Q), Q_params, pi_params, growth_params,
                                                    device=device, num_hidden=0)
        pure_birth_model = pure_birth_model.to(device)
    
    if "pure_birth_model_num" in methods:
        pure_birth_model_num = PureBirthLikelihoodModel([tree.copy("deepcopy") for tree in trees],
                                                    len(Q), Q_params, pi_params, growth_params,
                                                    device=device, num_hidden=0, subsampling_rate=0.999)
        pure_birth_model_num = pure_birth_model_num.to(device)
        Q_model = pure_birth_model_num.rate_builder.forward(pure_birth_model_num.get_Q_params())
        lam_model = pure_birth_model_num.get_growth_rates()
        pure_birth_model_num.prepare_numerical_pruner(lam_model, Q_model, rho=pure_birth_model_num.rho)


    for method_name in methods:
        if method_name == "log_vec_lik":
            lik = log_vec_likelihood(trees, Q_tensor, pi_params, growth_rates=growth_rates)
        elif method_name == "log_vec_lik_num":
            lik = log_vec_likelihood_numerical(trees, Q_tensor, pi_params, growth_rates=growth_rates)
        elif method_name == "pure_birth_model":
            lik = sum([pure_birth_model(i) for i in range(len(trees))])
        elif method_name == "pure_birth_model_num":
            lik = sum([pure_birth_model_num(i) for i in range(len(trees))])
        scores[method_name].append(lik.item())

    print()
    overall_log_lik = {}
    for method, liks in scores.items():
        if len(liks) == 0:
            continue
        overall_log_lik[method] = sum(liks)
        print(f"{method}\t{overall_log_lik[method]}")

    assert are_all_within_k_sig_figs(list(overall_log_lik.values()))
    assert False
