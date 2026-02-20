"""Tests for utility functions in utils.py."""

import math
import os
import numpy as np
import pytest

from ete3 import TreeNode
from branching_simulation import simulate_tree
from utils import is_ultrametric, get_idx2potency, binarize_tree, get_terminal_labels, get_observed_potencies


class TestIsUltrametric:

    def test_simulated_tree_is_ultrametric(self):
        Q = np.array([[-1.0, 1.0], [0.0, 0.0]])
        lam = np.array([1.0, 1.0])
        tree = simulate_tree(Q, lam, starting_type=0, T=1.0, seed=42)
        assert is_ultrametric(tree)

    def test_non_ultrametric(self):
        # Build a tree where leaves have different distances to root
        root = TreeNode()
        root.dist = 0
        child = TreeNode()
        child.dist = 1.0
        root.add_child(child)

        leaf1 = TreeNode()
        leaf1.dist = 0.5
        child.add_child(leaf1)

        leaf2 = TreeNode()
        leaf2.dist = 1.0  # Different from leaf1
        child.add_child(leaf2)

        assert not is_ultrametric(root)


class TestGetIdx2Potency:

    def test_diagonal_Q_self_potency(self):
        """Diagonal Q (no transitions) → each state can only reach itself."""
        Q = np.array([[0.0, 0.0], [0.0, 0.0]])
        result = get_idx2potency(Q)
        assert result[0] == (0,)
        assert result[1] == (1,)

    def test_transitive_closure(self):
        """If 0→1 and 1→2, state 0's potency should include state 2."""
        Q = np.array([[-1.0, 1.0, 0.0],
                      [0.0, -1.0, 1.0],
                      [0.0, 0.0, 0.0]])
        result = get_idx2potency(Q)
        # State 0 can eventually reach state 2
        assert 2 in result[0]
        # State 1 can reach state 2
        assert 2 in result[1]
        # State 2 is absorbing — only reaches itself
        assert result[2] == (2,)


class TestBinarizeTree:

    def test_all_binary_after_binarization(self):
        # Build a multifurcating tree
        root = TreeNode()
        root.dist = 0
        for i in range(4):
            child = TreeNode()
            child.dist = 1.0
            child.add_feature("state", i % 2)
            root.add_child(child)

        binarize_tree(root)
        for node in root.traverse():
            n_children = len(node.children)
            assert n_children in [0, 1, 2], f"Node has {n_children} children after binarization"

    def test_preserves_leaves(self):
        root = TreeNode()
        root.dist = 0
        leaf_states = [0, 1, 0, 1, 1]
        for s in leaf_states:
            child = TreeNode()
            child.dist = 1.0
            child.add_feature("state", s)
            root.add_child(child)

        original_leaf_count = len(root.get_leaves())
        binarize_tree(root)
        # Leaf count should be the same
        assert len(root.get_leaves()) == original_leaf_count


class TestGetTerminalLabels:

    def test_reads_labels(self, tmp_path):
        label_file = tmp_path / "labels.txt"
        label_file.write_text("alpha\nbeta\ngamma\n")
        result = get_terminal_labels(str(label_file))
        assert result == ["alpha", "beta", "gamma"]

    def test_int_labels(self, tmp_path):
        label_file = tmp_path / "labels.txt"
        label_file.write_text("0\n1\n2\n")
        result = get_terminal_labels(str(label_file), is_int_state=True)
        assert result == [0, 1, 2]

    def test_none_path(self):
        result = get_terminal_labels(None)
        assert result == []


class TestGetObservedPotencies:

    def test_reads_tsv(self, tmp_path):
        pot_file = tmp_path / "potencies.txt"
        pot_file.write_text("A\tB,C\nB\tB\nC\tC\n")
        result = get_observed_potencies(str(pot_file))
        assert result["A"] == ("B", "C")
        assert result["B"] == ("B",)
        assert result["C"] == ("C",)

    def test_int_labels(self, tmp_path):
        pot_file = tmp_path / "potencies.txt"
        pot_file.write_text("0\t1,2\n1\t1\n2\t2\n")
        result = get_observed_potencies(str(pot_file), is_int_state=True)
        assert result[0] == (1, 2)
        assert result[1] == (1,)

    def test_none_path(self):
        result = get_observed_potencies(None)
        assert result == {}
