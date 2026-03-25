"""
Source code for simulating a multi-type ClaSSE branching process.

In this model cells do not change type during their lifetime.  Instead, type
changes occur only at cell division: a cell of type i divides at rate lam[i]
and each daughter independently inherits type j with probability B[i, j].
"""

import random
import numpy as np
from ete3 import TreeNode

INF = float('inf')


def simulate_tree(birth_kernel, growth_rates, starting_type, T, sample_probability=1.0, seed=123):
    """Simulates a multi-type ClaSSE pure-birth branching tree.

    Starting from a single cell of the given type, simulates the branching
    process forward in time until time T.  A unifurcating root with branch
    length 0 is prepended for computational convenience.

    Args:
        birth_kernel: A 2D numpy array of shape (K, K).  Row i is the
            probability distribution over daughter types when a type-i cell
            divides (must be non-negative and sum to 1 along each row).
        growth_rates: A 1D numpy array of birth rates per cell type.
        starting_type: The integer index of the starting cell type.
        T: Total simulation time.
        sample_probability: The prob with which each leaf is sampled.
        seed: Random seed for reproducibility.

    Returns:
        An ete3 TreeNode representing the root of the simulated tree.
    """
    rng = np.random.default_rng(seed=seed)
    start_node = TreeNode()
    end_type, branch_len = edge_process(growth_rates, starting_type, T, rng)
    start_node.add_feature("state", str(end_type))
    start_node.dist = branch_len
    if T - branch_len > 0:
        branching_process(birth_kernel, growth_rates, start_node, T - branch_len, rng)

    # Add 0-length root node for ease of computation
    root = TreeNode()
    root.add_feature("state", str(starting_type))
    root.dist = 0
    root.add_child(start_node)

    if sample_probability < 1.0:
        leaves = root.get_leaves()
        ps = rng.random(size = len(leaves))
        to_keep = [leaf for leaf, p in zip(leaves, ps) if p < sample_probability]
        if len(to_keep) >= 1:
            root.prune(nodes=to_keep, preserve_branch_length=True)
        else:
            # print("==> Warning: no leaves found in tree!")
            # root = simulate_tree(birth_kernel, growth_rates, starting_type, T, sample_probability, seed=seed+1)
            return None

    return root


def branching_process(birth_kernel, growth_rates, current_node, T, rng):
    """Recursively simulates the ClaSSE branching process from a given node.

    At each division event two daughter cells are created.  Each daughter's
    type is drawn independently from the row of birth_kernel corresponding to
    the parent type, and each daughter then independently undergoes its own
    edge process.

    Args:
        birth_kernel: A 2D numpy array of shape (K, K).  Row i is the
            daughter-type distribution for a type-i parent.
        growth_rates: A 1D numpy array of birth rates per cell type.
        current_node: The ete3 TreeNode at which division has just occurred.
        T: Remaining simulation time available to the daughters.
        rng: A numpy random Generator instance.

    Returns:
        The current_node with two children attached.
    """
    assert T > 0
    parent_type = current_node.state
    n_states = len(growth_rates)

    for _ in range(2):
        daughter_type = rng.choice(n_states, p=birth_kernel[int(parent_type)])
        end_type, branch_len = edge_process(growth_rates, daughter_type, T, rng)
        child = TreeNode()
        child.dist = branch_len
        child.add_feature("state", str(end_type))
        current_node.add_child(child)
        if T - branch_len > 0:
            branching_process(birth_kernel, growth_rates, child, T - branch_len, rng)

    return current_node


def edge_process(growth_rates, start_type, T, rng):
    """Simulates a single edge of the ClaSSE process.

    A cell of the given type lives for an exponentially distributed time with
    rate growth_rates[start_type].  If the waiting time exceeds T the edge is
    truncated at T (cell is observed as a leaf).  The cell type is constant
    throughout the edge; daughter types are drawn at division inside
    branching_process.

    Args:
        growth_rates: A 1D numpy array of birth rates per cell type.
        start_type: The integer index of the current cell type.
        T: Maximum time for this edge.
        rng: A numpy random Generator instance.

    Returns:
        A tuple (end_type, branch_length) where end_type equals start_type
        (cells do not change type mid-edge) and branch_length is the edge
        duration (capped at T).
    """
    rate = growth_rates[int(start_type)]
    if rate <= 0:
        return start_type, T
    birth_time = rng.exponential(1.0 / rate)
    if birth_time >= T:
        return start_type, T
    return start_type, birth_time
