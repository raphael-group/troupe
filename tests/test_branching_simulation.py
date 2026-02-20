"""Tests for simulate_tree, edge_process, and branching_process."""

import math
import numpy as np
import pytest

from branching_simulation import simulate_tree, edge_process
from utils import is_ultrametric


class TestSimulateTree:

    def test_tree_is_ultrametric(self):
        Q = np.array([[-1.0, 1.0],
                      [0.0, 0.0]])
        lam = np.array([1.0, 1.0])
        for seed in range(10):
            tree = simulate_tree(Q, lam, starting_type=0, T=1.0, seed=seed)
            assert is_ultrametric(tree), f"Tree with seed={seed} is not ultrametric"

    def test_tree_has_leaves_with_states(self):
        Q = np.array([[-1.0, 1.0],
                      [0.0, 0.0]])
        lam = np.array([1.0, 1.0])
        tree = simulate_tree(Q, lam, starting_type=0, T=1.0, seed=42)
        for leaf in tree.get_leaves():
            assert hasattr(leaf, "state")
            assert leaf.state in [0, 1]

    def test_seed_reproducibility(self):
        Q = np.array([[-2.0, 1.0, 1.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])
        lam = np.array([2.0, 1.0, 1.0])
        t1 = simulate_tree(Q, lam, starting_type=0, T=1.0, seed=123)
        t2 = simulate_tree(Q, lam, starting_type=0, T=1.0, seed=123)
        # Same seed → same newick
        assert t1.write() == t2.write()

    def test_different_seeds_differ(self):
        Q = np.array([[-2.0, 1.0, 1.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])
        lam = np.array([2.0, 1.0, 1.0])
        t1 = simulate_tree(Q, lam, starting_type=0, T=1.0, seed=0)
        t2 = simulate_tree(Q, lam, starting_type=0, T=1.0, seed=999)
        # Very unlikely to be identical
        assert t1.write() != t2.write()

    def test_expected_leaf_count(self):
        """Mean leaf count should be approximately exp(growth_rate * T)."""
        Q = np.array([[0.0]])  # 1 state, no transitions
        lam = np.array([2.0])
        T = 1.5
        num_trials = 200
        leaf_counts = []
        for seed in range(num_trials):
            tree = simulate_tree(Q, lam, starting_type=0, T=T, seed=seed)
            leaf_counts.append(len(tree.get_leaves()))
        mean_leaves = np.mean(leaf_counts)
        expected = math.exp(lam[0] * T)
        # Allow generous tolerance (3x standard error)
        std_err = np.std(leaf_counts) / math.sqrt(num_trials)
        assert abs(mean_leaves - expected) < 3 * std_err + 1, \
            f"mean={mean_leaves:.1f}, expected≈{expected:.1f}"

    def test_terminal_states_absorbing(self):
        """States with all-zero Q rows should never produce children of different type."""
        Q = np.array([[-1.0, 1.0],
                      [0.0, 0.0]])  # State 1 is absorbing
        lam = np.array([1.0, 1.0])
        for seed in range(20):
            tree = simulate_tree(Q, lam, starting_type=0, T=2.0, seed=seed)
            for node in tree.traverse():
                if hasattr(node, "state") and node.state == 1:
                    # All descendants of a state-1 node should be state 1
                    for desc in node.iter_descendants():
                        if hasattr(desc, "state"):
                            assert desc.state == 1


class TestEdgeProcess:

    def test_stays_in_state_with_no_transitions(self):
        """With Q=0 (no transitions), edge_process returns (start_type, T)."""
        Q = np.array([[0.0]])
        lam = np.array([1.0])
        rng = np.random.default_rng(42)
        # The only event possible is birth, so edge ends at birth time or T
        end_type, branch_len = edge_process(Q, lam, 0, 10.0, rng)
        assert end_type == 0
        assert branch_len <= 10.0

    def test_type_changes_with_high_rate(self):
        """With very high transition rate, type eventually changes."""
        Q = np.array([[-100.0, 100.0],
                      [0.0, 0.0]])
        lam = np.array([0.01, 0.01])  # Very slow birth
        rng = np.random.default_rng(42)
        changed = False
        for _ in range(50):
            end_type, _ = edge_process(Q, lam, 0, 100.0, rng)
            if end_type == 1:
                changed = True
                break
        assert changed, "Type never changed despite high transition rate"
