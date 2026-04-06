import os
import sys

import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from eval_utils import _augment_edges_for_required_nodes


def test_augment_edges_for_required_nodes_keeps_terminal_with_strongest_incoming():
    adj = np.array([
        [0.0, 0.02, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.005, 0.0],
    ])
    edges = []
    required_nodes = {1}
    active_nodes = {0, 1, 2}

    augmented_edges, nodes = _augment_edges_for_required_nodes(
        adj, edges, required_nodes, active_nodes=active_nodes
    )

    assert nodes == {0, 1, 2}
    assert augmented_edges == [(0, 1, 0.02)]


def test_augment_edges_for_required_nodes_preserves_existing_incoming_edge():
    adj = np.array([
        [0.0, 0.02],
        [0.0, 0.0],
    ])
    edges = [(0, 1, 0.02)]

    augmented_edges, nodes = _augment_edges_for_required_nodes(
        adj, edges, {1}, active_nodes={0, 1}
    )

    assert nodes == {0, 1}
    assert augmented_edges == [(0, 1, 0.02)]
