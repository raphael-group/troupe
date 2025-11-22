import ete3 as ete
import torch
import warnings
from likelihood import EPS
import math
from scipy.linalg import expm
from ete3 import TreeNode

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="networkx backend defined more than once: nx-loopback")
    import networkx as nx

import networkx as nx



def get_terminal_labels(terminal_label_path, is_int_state=False):
    terminal_labels = []
    if terminal_label_path is not None:
        with open(terminal_label_path, "r") as fp:
            for line in fp.readlines():
                if len(line) == 0:
                    continue
                state = line.strip()
                if is_int_state:
                    state = int(state)
                terminal_labels.append(state)
    else:
        warnings.warn("terminal_label_path is None")

    return terminal_labels

def get_observed_potencies(observed_potencies_path, is_int_state=False):
    observed_potencies = {}
    if observed_potencies_path is not None:
        with open(observed_potencies_path, "r") as fp:
            observed_potencies = {}
            for line in fp.readlines():
                if len(line) == 0:
                    continue
                entries = line.strip().split("\t")
                state = entries[0]
                potency = entries[1].split(",")
                if is_int_state:
                    state = int(state)
                    new_potency = []
                    for terminal_state in potency:
                        new_potency.append(int(terminal_state))
                    potency = new_potency
                potency.sort()
                observed_potencies[state] = tuple(potency)    
    else:
        warnings.warn("observed_potencies_path is None")

    return observed_potencies

def is_ultrametric(tree):
    """
    Returns true if the tree is (roughly) ultrametric. A tree
    """
    leaf_dists = [tree.get_distance(leaf) for leaf in tree.get_leaves()]
    return all([math.isclose(leaf_dists[i-1], leaf_dists[i], rel_tol=1e-2) for i in range(1, len(leaf_dists))])

def get_idx2potency(rate_matrix, eps=1e-4, tree_length=1.0):
    """
    Given a numpy array that represents a rate matrix, determine the potency set for each node.
    """
    transition_matrix = expm(100 * tree_length * rate_matrix)
    
    n = len(transition_matrix)
    idx2list = {}
    for row in range(n):
        idx2list[row] = []
        for col in range(n):
            if transition_matrix[row, col] > eps:
                idx2list[row].append(col)
    
    potency_map = {idx: tuple(sorted(potency)) for idx, potency in idx2list.items()}
    return potency_map

def get_reachable_idxs(adj_matrix, starting_state, threshold):
    G = nx.DiGraph()
    num_nodes = len(adj_matrix)
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = adj_matrix[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)

    assert starting_state in G
    
    reachable_nodes = list(nx.descendants(G, starting_state))
    reachable_nodes.append(starting_state)
    return reachable_nodes

def binarize_tree(t):
    to_fix = [n for n in t.traverse() if len(n.children) > 2]
    # Traverse over a snapshot of nodes to avoid issues while modifying children
    for node in to_fix:
        orig_children = list(node.children)
        for ch in orig_children[1:]:
            ch.detach()
        current_parent = node
        for ch in orig_children[1:]:
            splitter = TreeNode()
            splitter.dist = 0.0
            current_parent.add_child(splitter)
            old_dist = ch.dist
            splitter.add_child(ch)
            ch.dist = old_dist
            current_parent = splitter
    return t
