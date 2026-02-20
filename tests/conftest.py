"""Shared fixtures for the troupe test suite."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from branching_simulation import simulate_tree
from likelihood import _prep_log_tree

dtype = torch.float64
torch.set_default_dtype(dtype)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# CTMC systems — (Q, lam, pi_params)
# ---------------------------------------------------------------------------
@pytest.fixture
def two_state_system():
    """2-state non-reversible CTMC: 0 → 1 only."""
    Q = np.array([[-1.0, 1.0],
                  [0.0, 0.0]])
    lam = np.array([1.0, 1.0])
    pi_params = torch.tensor([1e20, -1e20], dtype=dtype)
    return Q, lam, pi_params


@pytest.fixture
def three_state_system():
    """3-state non-reversible CTMC: 0 → 1, 0 → 2, 1 → 2."""
    Q = np.array([[-1.0, 0.5, 0.5],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
    lam = np.array([2.0, 1.0, 1.0])
    pi_params = torch.tensor([1e20, -1e20, -1e20], dtype=dtype)
    return Q, lam, pi_params


@pytest.fixture
def seven_state_system():
    """7-state system with 4 observed + 3 hidden progenitor states."""
    Q = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 4.0, -8.0, 4.0, 0.0],
        [0.0, 0.0, 4.0, 0.0, 0.0, -8.0, 4.0],
        [4.0, 4.0, 0.0, 0.0, 0.0, 0.0, -8.0],
    ])
    lam = np.array([1, 1, 1, 1, 6, 4, 4], dtype=float)
    pi_params = torch.tensor([-1e20, -1e20, -1e20, -1e20, 1e20, -1e20, -1e20], dtype=dtype)
    return Q, lam, pi_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_Q_params(Q, device=None):
    """Reverse the softplus: params = log(exp(Q) - 1), diagonal zeroed."""
    Q_tensor = torch.tensor(Q, dtype=dtype, device=device)
    Q_params = torch.log(torch.exp(Q_tensor) - 1)
    Q_params.fill_diagonal_(0)
    return Q_params


def make_growth_params(lam, device=None):
    """Reverse the softplus for growth rates."""
    lam_tensor = torch.tensor(lam, dtype=dtype, device=device)
    return torch.log(torch.exp(lam_tensor) - 1)


def simulate_and_prep_trees(Q, lam, num_trees, T=1.0, starting_type=0, seed_multiplier=17):
    """Simulate trees and prepare them with _prep_log_tree."""
    trees = []
    for seed in range(num_trees):
        tree = simulate_tree(Q, lam, starting_type=starting_type, T=T, seed=seed * seed_multiplier)
        tree = _prep_log_tree(tree, len(Q))
        trees.append(tree)
    return trees


def are_all_within_k_sig_figs(numbers, k=4):
    """Check if all floats agree to k significant figures."""
    if len(numbers) == 0:
        return False
    for num in numbers:
        assert isinstance(num, float)
    rounded_values = {f"{num:.{k}}" for num in numbers if num}
    return len(rounded_values) == 1
