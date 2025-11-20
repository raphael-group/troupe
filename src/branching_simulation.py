"""
Source code for simulating a multi-type pure-birth branching process. 

"""


import numpy as np
from ete3 import TreeNode

INF = float('inf')

def simulate_tree(transition_rates, growth_rates, starting_type, T, seed=123):
    rng = np.random.default_rng(seed=seed)
    start_node = TreeNode()
    end_type, branch_len = edge_process(transition_rates, growth_rates, starting_type, T, rng)
    start_node.add_feature("state", end_type)
    start_node.dist = branch_len
    if T-branch_len > 0:
        branching_process(transition_rates, growth_rates, start_node, T-branch_len, rng)

    # Add 0 height root node for ease of computation
    root = TreeNode()
    root.add_feature("state", starting_type)
    root.dist = 0
    root.add_child(start_node)
    
    return root

def branching_process(transition_rates, growth_rates, current_node, T, rng):
    """
    Simulate from this node until time T (time remaining).
    We simulate the parent's next event: either a birth (split) or nothing
    before T. If birth occurs at t < T, create two daughters born at the same t.
    """
    assert T > 0    # TODO: What happens if T <= 0
    starting_type = current_node.state

    # Left child
    end_type_left, branch_len_left = edge_process(transition_rates, growth_rates, starting_type, T, rng)
    left_child = TreeNode()
    left_child.dist = branch_len_left
    current_node.add_child(left_child)
    # left_child.name = f"node_{idx}"
    # idx += 1
    left_child.add_feature("state", end_type_left)
    if T - branch_len_left > 0:         # If time remains, populate the left clade's children
        branching_process(transition_rates, growth_rates, left_child, T-branch_len_left, rng)

    # Right child
    end_type_right, branch_len_right = edge_process(transition_rates, growth_rates, starting_type, T, rng)
    right_child = TreeNode()
    right_child.dist = branch_len_right
    current_node.add_child(right_child)
    # right_child.name = f"node_{idx}"
    # idx += 1
    right_child.add_feature("state", end_type_right)
    if T - branch_len_right > 0:         # If time remains, populate the right clade's children
        branching_process(transition_rates, growth_rates, right_child, T-branch_len_right, rng)

    return current_node


def edge_process(Q, lam, start_type, T, rng):
    """
    Multi-type pure-birth branching process for a single edge
    """
    curr_type = start_type
    time_to_birth = 0
    while True:
        # Draw waiting time from an exponential distribution with rate 'rate' for all but curr type
        waiting_times = [rng.exponential(1.0 / rate) if i != curr_type and rate!=0 else INF \
                          for i, rate in enumerate(Q[curr_type, :])]
        birth_time = rng.exponential(1.0 / lam[curr_type])
        time_to_next_event = min(min(waiting_times), birth_time)
        time_to_birth += time_to_next_event
        if time_to_birth > T:                   # truncate due to reached time limit
            time_to_birth = T
            break
        if birth_time == time_to_next_event:    # branching occurs
            break
        else:                                   # type-change occurs
            curr_type = waiting_times.index(time_to_next_event)

    return curr_type, time_to_birth



    