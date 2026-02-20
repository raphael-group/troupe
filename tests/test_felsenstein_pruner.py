"""Unit tests for FelsensteinPruner."""

import math
import numpy as np
import torch
import torch.nn.functional as F
import pytest

from models import FelsensteinPruner, TreeTensorizer, RateMatrixBuilder
from branching_simulation import simulate_tree

dtype = torch.float64
torch.set_default_dtype(dtype)


def _tensorize_one_tree(tree, num_states, device):
    """Tensorize a single tree and return components."""
    tt = TreeTensorizer([tree], num_states, device)
    return (
        tt.postorders[0],
        tt.children[0],
        tt.branch_lens[0],
        tt.partials[0],
        tt.levels[0],
    )


class TestFelsensteinPruner:

    def test_log_likelihood_finite(self, device, two_state_system):
        Q_np, lam_np, _ = two_state_system
        tree = simulate_tree(Q_np, lam_np, starting_type=0, T=1.0, seed=42)
        post, children, blens, partials, levels = \
            _tensorize_one_tree(tree, 2, device)

        Q = torch.tensor(Q_np, dtype=dtype, device=device)
        growth_rates = torch.tensor(lam_np, dtype=dtype, device=device)
        pruner = FelsensteinPruner(2)
        root_partial = pruner.log_prune(
            post, children, blens, partials, Q, levels,
            growth_rates=growth_rates
        )
        assert torch.isfinite(root_partial).all()

    def test_gradient_flows(self, device, two_state_system):
        Q_np, lam_np, pi = two_state_system
        tree = simulate_tree(Q_np, lam_np, starting_type=0, T=1.0, seed=42)
        post, children, blens, partials, levels = \
            _tensorize_one_tree(tree, 2, device)

        builder = RateMatrixBuilder(2, device=device)
        Q = builder.forward()
        growth_rates = F.softplus(torch.zeros(2, dtype=dtype, device=device, requires_grad=True))
        pruner = FelsensteinPruner(2)
        root_partial = pruner.log_prune(
            post, children, blens, partials, Q, levels,
            growth_rates=growth_rates
        )
        loss = root_partial.sum()
        loss.backward()
        assert builder.free_params.grad is not None

    def test_growth_rates_affect_likelihood(self, device, two_state_system):
        Q_np, lam_np, _ = two_state_system
        tree = simulate_tree(Q_np, lam_np, starting_type=0, T=1.0, seed=42)
        post, children, blens, partials, levels = \
            _tensorize_one_tree(tree, 2, device)

        Q = torch.tensor(Q_np, dtype=dtype, device=device)
        pruner = FelsensteinPruner(2)

        gr1 = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
        result1 = pruner.log_prune(
            post, children, blens, partials.clone(), Q, levels,
            growth_rates=gr1
        )

        gr2 = torch.tensor([5.0, 5.0], dtype=dtype, device=device)
        result2 = pruner.log_prune(
            post, children, blens, partials.clone(), Q, levels,
            growth_rates=gr2
        )

        assert not torch.allclose(result1, result2)

    def test_two_leaf_cherry(self, device):
        """A cherry tree with known structure should give a finite result."""
        from ete3 import TreeNode

        # Build: root -> internal -> (leaf0 state=0, leaf1 state=1)
        root = TreeNode()
        root.dist = 0
        internal = TreeNode()
        internal.dist = 0  # zero-length unifurcating root edge
        root.add_child(internal)

        leaf0 = TreeNode()
        leaf0.dist = 0.5
        leaf0.add_feature("state", 0)
        internal.add_child(leaf0)

        leaf1 = TreeNode()
        leaf1.dist = 0.5
        leaf1.add_feature("state", 1)
        internal.add_child(leaf1)

        Q = torch.tensor([[-1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=device)
        growth_rates = torch.tensor([1.0, 1.0], dtype=dtype, device=device)

        post, children, blens, partials, levels = \
            _tensorize_one_tree(root, 2, device)
        pruner = FelsensteinPruner(2)
        result = pruner.log_prune(
            post, children, blens, partials, Q, levels,
            growth_rates=growth_rates
        )
        assert torch.isfinite(result).all()
        # Result should be negative (log-probability)
        assert (result < 0).any()
