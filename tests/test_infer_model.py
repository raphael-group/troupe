"""Integration tests for the infer_model.py workflow.

Exercises compute_mle with group lasso (which requires terminal_idx)
and verifies the full optimization pipeline succeeds end-to-end.
"""

import os
import numpy as np
import torch
import pytest
import pickle

from optimizer import compute_mle
from branching_simulation import simulate_tree

dtype = torch.float64
torch.set_default_dtype(dtype)


def _simulate_trees(Q, lam, num_trees=10, T=1.0, starting_type=0):
    trees = []
    for seed in range(num_trees):
        tree = simulate_tree(Q, lam, starting_type=starting_type, T=T, seed=seed * 17)
        trees.append(tree)
    return trees


class TestInferModelWorkflow:

    @pytest.mark.slow
    def test_two_state_with_group_lasso(self, device, tmp_path, two_state_system):
        """Group lasso path exercises terminal_idx on the model."""
        Q_np, lam_np, pi = two_state_system
        trees = _simulate_trees(Q_np, lam_np, num_trees=10)

        idx2potency = {0: (0, 1), 1: (1,)}
        idx2state = {0: 0, 1: 1}
        model_info = {
            "idx2potency": idx2potency,
            "idx2state": idx2state,
        }

        llh, loss = compute_mle(
            trees, (2, 0), device, str(tmp_path),
            group_lasso_strength=0.1,
            do_logging=False,
            model_info=model_info,
        )

        assert np.isfinite(loss)
        assert os.path.exists(os.path.join(str(tmp_path), "model_dict.pkl"))

    @pytest.mark.slow
    def test_three_state_with_group_lasso(self, device, tmp_path, three_state_system):
        """Three-state system with group lasso to verify terminal_idx works for multiple terminals."""
        Q_np, lam_np, pi = three_state_system
        trees = _simulate_trees(Q_np, lam_np, num_trees=10)

        idx2potency = {0: (0, 1, 2), 1: (1,), 2: (2,)}
        idx2state = {0: 0, 1: 1, 2: 2}
        model_info = {
            "idx2potency": idx2potency,
            "idx2state": idx2state,
        }

        llh, loss = compute_mle(
            trees, (3, 0), device, str(tmp_path),
            group_lasso_strength=0.1,
            do_logging=False,
            model_info=model_info,
        )

        assert np.isfinite(loss)

    @pytest.mark.slow
    def test_group_lasso_with_l1(self, device, tmp_path, two_state_system):
        """Both L1 and group lasso active simultaneously."""
        Q_np, lam_np, pi = two_state_system
        trees = _simulate_trees(Q_np, lam_np, num_trees=10)

        idx2potency = {0: (0, 1), 1: (1,)}
        idx2state = {0: 0, 1: 1}
        model_info = {
            "idx2potency": idx2potency,
            "idx2state": idx2state,
        }

        llh, loss = compute_mle(
            trees, (2, 0), device, str(tmp_path),
            l1_regularization_strength=0.1,
            group_lasso_strength=0.05,
            do_logging=False,
            model_info=model_info,
        )

        assert np.isfinite(loss)
        assert os.path.exists(os.path.join(str(tmp_path), "model_dict.pkl"))
