from simulation import simulate_tree_state
from optimizer import _run_lbfgs
from OLD_models import LogLikelihoodModel
from models import CTMCLikelihoodModel, GeneralEmissionModel

import numpy as np
import torch
import ete3
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device =", device)


all_methods = [
    "CTMCLikelihoodModel",
    "LogLikelihoodModel",
    "GeneralEmissionModel",
    # "PotencyLikelihoodModel"
]


def test_simple():
    nwk = "((A:0.5, B:0.5):0.4, (C:0.5, D:0.5):0.4):0.1;"   # TODO: Add unifurcating root branch?
    tree = ete3.Tree(nwk)
    for leaf in tree.get_leaves():
        leaf.state = 1

    print(tree.get_ascii(attributes=["state", "name", "dist"]))

    compare_methods([tree], all_methods, n_states=2)


def compare_methods(trees, methods, n_states):
    assert isinstance(trees, list)

    method2model = {}

    for method_name in methods:
        print()
        print(method_name)
        llh = compute_model_mle(trees, (n_states, 0), device, method_name)
        method2model[method_name] = (llh.get_rate_matrix().float(), llh.get_root_distribution().float(),
                                     sum([llh(i).float() for i in range(len(trees))]))

    base_Q, base_pi, base_log_lik = method2model[methods[0]]
    for method, (rate_matrix, root_distribution, log_lik) in method2model.items():
        print()
        print(f"{method}")
        print("Log likelihood", log_lik)
        print("Rate matrix")
        print(rate_matrix)
        print("Root distribution")
        print(root_distribution)

        relative_error = torch.sum(torch.abs(base_Q - rate_matrix))/torch.sum(torch.abs(base_Q))
        print("relative_error:", relative_error)

        assert torch.isclose(base_log_lik, log_lik, atol=1e-3).all()
        assert relative_error < 0.5
        assert torch.isclose(base_pi, root_distribution, atol=1e-2).all()


def compute_model_mle(trees, num_states, device, model_type):
    is_testing = model_type == "LogLikelihoodModel"
    
    output_dir = "/n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/tests/tmp"
    os.makedirs(output_dir, exist_ok=True)
    regularization_strength = 0
    do_logging = False
    return _run_lbfgs(trees, num_states, device, output_dir, regularization_strength,
                      do_logging, model_type=model_type, is_testing=is_testing, loss_thresh=1e-8)