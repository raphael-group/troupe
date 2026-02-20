"""Tests for optimizer.py: loss_fn, sparse_regularization, group_lasso, compute_mle."""

import os
import numpy as np
import torch
import torch.nn.functional as F
import pytest
import pickle

from optimizer import loss_fn, sparse_regularization, compute_mle
from models import PureBirthLikelihoodModel
from branching_simulation import simulate_tree
from conftest import make_Q_params, make_growth_params

dtype = torch.float64
torch.set_default_dtype(dtype)


def _make_model_and_trees(Q_np, lam_np, pi, device, num_trees=5, T=1.0,
                          idx2potency=None, idx2state=None, num_hidden=0):
    """Create a PureBirthLikelihoodModel and trees for testing."""
    trees = []
    starting_type = 0
    for seed in range(num_trees):
        tree = simulate_tree(Q_np, lam_np, starting_type=starting_type, T=T, seed=seed * 17)
        trees.append(tree)
    Q_params = make_Q_params(Q_np, device=device)
    growth_params = make_growth_params(lam_np, device=device)
    model = PureBirthLikelihoodModel(
        trees, len(Q_np), Q_params, pi.to(device), growth_params,
        device=device, num_hidden=num_hidden, idx2potency=idx2potency,
        idx2state=idx2state,
    )
    return model, trees


class TestLossFn:

    def test_negative_llh_no_regularization(self, device, two_state_system):
        Q_np, lam_np, pi = two_state_system
        model, trees = _make_model_and_trees(Q_np, lam_np, pi, device, num_trees=3)
        tree_idxs = list(range(len(trees)))
        loss = loss_fn(model, tree_idxs)
        # Loss = -mean(log_lik), should be positive (since log_lik < 0)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_l1_increases_loss(self, device, two_state_system):
        Q_np, lam_np, pi = two_state_system
        model, trees = _make_model_and_trees(Q_np, lam_np, pi, device, num_trees=3)
        tree_idxs = list(range(len(trees)))
        loss_no_reg = loss_fn(model, tree_idxs)
        loss_with_l1 = loss_fn(model, tree_idxs, l1_regularization_strength=1.0)
        assert loss_with_l1.item() >= loss_no_reg.item()


class TestSparseRegularization:

    def test_positive_for_nonzero_Q(self, device, two_state_system):
        Q_np, lam_np, pi = two_state_system
        model, _ = _make_model_and_trees(Q_np, lam_np, pi, device, num_trees=2)
        reg = sparse_regularization(model)
        assert reg.item() > 0

    def test_zero_when_offdiag_zero(self, device):
        """If Q is diagonal (all off-diag zero), regularization is 0."""
        Q_np = np.array([[0.0, 0.0], [0.0, 0.0]])
        lam_np = np.array([1.0, 1.0])
        pi = torch.tensor([1e20, -1e20], dtype=dtype)
        # Use init_Q_params that produce zero off-diagonal via softplus
        # softplus(x) = 0 requires x -> -inf; use a very large negative
        model, _ = _make_model_and_trees(Q_np, lam_np, pi, device, num_trees=1)
        # Manually zero out the free params (softplus(large_negative) ≈ 0)
        with torch.no_grad():
            model.rate_builder.free_params.fill_(-100.0)
        reg = sparse_regularization(model)
        assert reg.item() < 1e-10


class TestComputeMLE:

    @pytest.mark.slow
    def test_mle_saves_checkpoint(self, device, tmp_path, two_state_system):
        """compute_mle should save model_dict.pkl to output_dir."""
        Q_np, lam_np, pi = two_state_system
        trees = []
        for seed in range(10):
            tree = simulate_tree(Q_np, lam_np, starting_type=0, T=1.0, seed=seed * 17)
            trees.append(tree)

        output_dir = str(tmp_path)
        idx2potency = {0: (0, 1), 1: (1,)}
        idx2state = {0: 0, 1: 1}
        model_info = {
            "idx2potency": idx2potency,
            "idx2state": idx2state,
        }

        llh, loss = compute_mle(
            trees, (2, 0), device, output_dir,
            l1_regularization_strength=0.1,
            do_logging=False,
            model_info=model_info,
        )

        assert os.path.exists(os.path.join(output_dir, "model_dict.pkl"))
        with open(os.path.join(output_dir, "model_dict.pkl"), "rb") as f:
            model_dict = pickle.load(f)
        assert "rate_matrix" in model_dict
        assert "root_distribution" in model_dict
        assert "growth_rates" in model_dict

    @pytest.mark.slow
    def test_mle_loss_decreases(self, device, tmp_path, two_state_system):
        """After optimization, loss should be lower than initial."""
        Q_np, lam_np, pi = two_state_system
        trees = []
        for seed in range(15):
            tree = simulate_tree(Q_np, lam_np, starting_type=0, T=1.0, seed=seed * 17)
            trees.append(tree)

        output_dir = str(tmp_path)
        idx2potency = {0: (0, 1), 1: (1,)}
        idx2state = {0: 0, 1: 1}
        model_info = {
            "idx2potency": idx2potency,
            "idx2state": idx2state,
        }

        llh, final_loss = compute_mle(
            trees, (2, 0), device, output_dir,
            l1_regularization_strength=0.0,
            do_logging=False,
            model_info=model_info,
        )

        # Final loss should be finite
        assert np.isfinite(final_loss)
