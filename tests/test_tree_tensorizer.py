"""Unit tests for TreeTensorizer."""

import numpy as np
import torch
import pytest

from models import TreeTensorizer
from branching_simulation import simulate_tree

dtype = torch.float64
torch.set_default_dtype(dtype)


def _make_simple_trees(num_states=2, num_trees=3, T=1.0):
    """Helper: simulate trees for tensorizer tests."""
    Q = np.zeros((num_states, num_states))
    if num_states >= 2:
        Q[0, 1] = 1.0
        Q[0, 0] = -1.0
    lam = np.ones(num_states)
    trees = []
    for seed in range(num_trees):
        tree = simulate_tree(Q, lam, starting_type=0, T=T, seed=seed * 17)
        trees.append(tree)
    return trees, num_states


class TestTreeTensorizer:

    def test_postorder_length(self, device):
        trees, ns = _make_simple_trees(num_trees=1)
        tt = TreeTensorizer(trees, ns, device)
        # Postorder should have one entry per node
        tree = trees[0]
        # Count nodes (TreeTensorizer deep-copies and may add a unifurcating root)
        n_nodes = len(tt.postorders[0])
        assert n_nodes >= len(list(tree.traverse()))

    def test_leaf_partials_one_hot(self, device):
        trees, ns = _make_simple_trees(num_states=3, num_trees=1)
        tt = TreeTensorizer(trees, ns, device)
        partials = tt.partials[0]
        leaf_idxs = tt.leaf_idxs[0]
        for li in leaf_idxs:
            row = partials[li]
            # Exactly one entry should be 0.0 (log(1)), rest should be -inf
            assert (row == 0.0).sum().item() == 1
            assert (row == -float('inf')).sum().item() == ns - 1

    def test_internal_partials_neg_inf(self, device):
        trees, ns = _make_simple_trees(num_trees=1)
        tt = TreeTensorizer(trees, ns, device)
        partials = tt.partials[0]
        leaf_idxs_set = set(tt.leaf_idxs[0].tolist())
        for idx in range(len(partials)):
            if idx not in leaf_idxs_set:
                row = partials[idx]
                # Internal nodes should be initialized to all -inf
                assert (row == -float('inf')).all()

    def test_branch_lengths_nonneg(self, device):
        trees, ns = _make_simple_trees(num_trees=2)
        tt = TreeTensorizer(trees, ns, device)
        for blens in tt.branch_lens:
            assert (blens >= 0).all()

    def test_levels_monotonic(self, device):
        trees, ns = _make_simple_trees(num_trees=1)
        tt = TreeTensorizer(trees, ns, device)
        levels = tt.levels[0]
        # Leaves should be level 0
        leaf_idxs = tt.leaf_idxs[0]
        for li in leaf_idxs:
            assert levels[li].item() == 0
        # Max level should be > 0 (at least root is above leaves)
        assert levels.max().item() > 0

    def test_multiple_trees(self, device):
        trees, ns = _make_simple_trees(num_trees=5)
        tt = TreeTensorizer(trees, ns, device)
        assert len(tt.postorders) == 5
        assert len(tt.children) == 5
        assert len(tt.branch_lens) == 5
        # Global time grid should be populated
        assert len(tt.global_time_grid) > 0

    def test_ultrametric_abs_times(self, device):
        Q = np.array([[-1.0, 1.0],
                      [0.0, 0.0]])
        lam = np.array([1.0, 1.0])
        T = 2.0
        tree = simulate_tree(Q, lam, starting_type=0, T=T, seed=42)
        tt = TreeTensorizer([tree], 2, device)
        abs_times = tt.abs_times[0]
        leaf_idxs = tt.leaf_idxs[0]
        leaf_times = abs_times[leaf_idxs]
        # All leaf absolute times should be 0 (leaves are at present)
        assert torch.allclose(leaf_times, torch.zeros_like(leaf_times), atol=1e-10)
