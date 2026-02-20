"""Tests for likelihood.py: log_vec_likelihood, _prep_log_tree, compute_transition_tensor."""

import math
import numpy as np
import torch
import torch.nn.functional as F
import pytest

from likelihood import (
    log_vec_likelihood,
    log_vectorized_felsenstein_pruning,
    _prep_log_tree,
    compute_transition_tensor,
)
from branching_simulation import simulate_tree

dtype = torch.float64
torch.set_default_dtype(dtype)


class TestPrepLogTree:

    def test_adds_log_partials_to_leaves(self):
        Q = np.array([[-1.0, 1.0], [0.0, 0.0]])
        lam = np.array([1.0, 1.0])
        tree = simulate_tree(Q, lam, starting_type=0, T=1.0, seed=42)
        prepped = _prep_log_tree(tree, 2)
        for leaf in prepped.get_leaves():
            assert hasattr(leaf, "log_partials")
            assert leaf.log_partials.shape == (2,)

    def test_adds_unifurcating_root(self):
        Q = np.array([[-1.0, 1.0], [0.0, 0.0]])
        lam = np.array([1.0, 1.0])
        tree = simulate_tree(Q, lam, starting_type=0, T=1.0, seed=42)
        prepped = _prep_log_tree(tree, 2)
        # Root should have exactly 1 child (unifurcating)
        assert len(prepped.children) == 1

    def test_leaf_partials_correct(self):
        Q = np.array([[-1.0, 0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        lam = np.array([1.0, 1.0, 1.0])
        tree = simulate_tree(Q, lam, starting_type=0, T=1.0, seed=42)
        prepped = _prep_log_tree(tree, 3)
        for leaf in prepped.get_leaves():
            state = leaf.state
            # log_partial[state] should be 0 (log(1)), others -inf
            assert leaf.log_partials[state].item() == 0.0
            for s in range(3):
                if s != state:
                    assert leaf.log_partials[s].item() == -float('inf')

    def test_branch_lens_attached(self):
        Q = np.array([[-1.0, 1.0], [0.0, 0.0]])
        lam = np.array([1.0, 1.0])
        tree = simulate_tree(Q, lam, starting_type=0, T=1.0, seed=42)
        prepped = _prep_log_tree(tree, 2)
        assert hasattr(prepped, "branch_lens")
        assert prepped.branch_lens.ndim == 1
        num_nodes = len(list(prepped.traverse()))
        assert prepped.branch_lens.shape[0] == num_nodes


class TestComputeTransitionTensor:

    def test_identity_at_zero(self):
        Q = torch.tensor([[-1.0, 1.0], [1.0, -1.0]], dtype=dtype)
        branch_lens = torch.tensor([0.0], dtype=dtype)
        P = compute_transition_tensor(branch_lens, Q)
        assert torch.allclose(P[0], torch.eye(2, dtype=dtype), atol=1e-10)

    def test_rows_sum_to_one(self):
        Q = torch.tensor([[-2.0, 1.0, 1.0],
                          [0.5, -1.0, 0.5],
                          [0.5, 0.5, -1.0]], dtype=dtype)
        branch_lens = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=dtype)
        P = compute_transition_tensor(branch_lens, Q)
        row_sums = P.sum(dim=2)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-10)

    def test_with_growth_rates(self):
        Q = torch.tensor([[-1.0, 1.0], [0.0, 0.0]], dtype=dtype)
        growth_rates = torch.tensor([2.0, 1.0], dtype=dtype)
        branch_lens = torch.tensor([0.5], dtype=dtype)
        P_no_growth = compute_transition_tensor(branch_lens, Q)
        P_growth = compute_transition_tensor(branch_lens, Q, growth_rates=growth_rates)
        # Growth rates shift the Q matrix, so results should differ
        assert not torch.allclose(P_no_growth, P_growth)


class TestLogVecLikelihood:

    def test_returns_finite(self):
        Q_np = np.array([[-1.0, 1.0], [0.0, 0.0]])
        lam_np = np.array([1.0, 1.0])
        trees = []
        for seed in range(5):
            tree = simulate_tree(Q_np, lam_np, starting_type=0, T=1.0, seed=seed)
            tree = _prep_log_tree(tree, 2)
            trees.append(tree)
        Q = torch.tensor(Q_np, dtype=dtype)
        pi = torch.tensor([1e20, -1e20], dtype=dtype)
        growth_rates = torch.tensor(lam_np, dtype=dtype)
        result = log_vec_likelihood(trees, Q, pi, growth_rates=growth_rates)
        assert torch.isfinite(result)
        assert result.item() < 0  # Log-likelihood is negative

    def test_likelihood_higher_at_truth(self):
        """Likelihood at true Q should be higher than at a random Q."""
        Q_true_np = np.array([[-1.0, 1.0], [0.0, 0.0]])
        lam_np = np.array([1.0, 1.0])
        trees = []
        for seed in range(20):
            tree = simulate_tree(Q_true_np, lam_np, starting_type=0, T=1.0, seed=seed * 7)
            tree = _prep_log_tree(tree, 2)
            trees.append(tree)

        pi = torch.tensor([1e20, -1e20], dtype=dtype)
        growth_rates = torch.tensor(lam_np, dtype=dtype)

        Q_true = torch.tensor(Q_true_np, dtype=dtype)
        lik_true = log_vec_likelihood(trees, Q_true, pi, growth_rates=growth_rates)

        # Perturbed Q: much smaller transition rate
        Q_wrong = torch.tensor([[-0.01, 0.01], [0.0, 0.0]], dtype=dtype)
        lik_wrong = log_vec_likelihood(trees, Q_wrong, pi, growth_rates=growth_rates)

        assert lik_true.item() > lik_wrong.item()

    def test_single_tree_matches_batch(self):
        Q_np = np.array([[-1.0, 1.0], [0.0, 0.0]])
        lam_np = np.array([1.0, 1.0])
        tree = simulate_tree(Q_np, lam_np, starting_type=0, T=1.0, seed=42)
        tree = _prep_log_tree(tree, 2)

        Q = torch.tensor(Q_np, dtype=dtype)
        pi = torch.tensor([1e20, -1e20], dtype=dtype)
        growth_rates = torch.tensor(lam_np, dtype=dtype)

        lik_batch = log_vec_likelihood([tree], Q, pi, growth_rates=growth_rates)

        # Manual single-tree computation
        branch_lens = tree.branch_lens
        transition_tensor = compute_transition_tensor(branch_lens, Q, growth_rates=growth_rates)
        lik_single = log_vectorized_felsenstein_pruning(
            tree, transition_tensor, pi, growth_rates=growth_rates
        )

        assert torch.allclose(lik_batch, lik_single, atol=1e-10)
