import torch
import torch.nn.functional as F
from ete3 import TreeNode
import math

EPS = 1e-300  # Numerical stability parameter (rate matrix entries must be geq to avoid NaNs)


def _default_state2idx(num_states):
    """Returns the identity mapping {i: i for i in range(num_states)}.

    Args:
        num_states: Number of states.

    Returns:
        A dict mapping each state index to itself.
    """
    return {i: i for i in range(num_states)}


def compute_transition_tensor(branch_lens, Q, growth_rates=None):
    """Computes transition matrices for each branch via batched matrix exponential.

    Args:
        branch_lens: A 1D tensor of branch lengths.
        Q: The rate matrix (shape: [num_states, num_states]).
        growth_rates: Optional 1D tensor of birth rates. If provided, the
            growth rates are subtracted from the diagonal of Q before
            exponentiation.

    Returns:
        A tensor of transition matrices (shape: [num_branches, num_states,
        num_states]).
    """
    if growth_rates is not None:
        Q = Q - torch.diag(growth_rates)
    scaled_Q_tensor = Q.unsqueeze(0) * branch_lens.view(-1, 1, 1)
    return torch.matrix_exp(scaled_Q_tensor).to(Q.dtype)

def log_vec_likelihood(trees, Q, root_logits, state2idx=None, growth_rates=None):
    """Computes the total log-likelihood over a list of prepped trees.

    Args:
        trees: A single tree or list of trees, each prepped via
            ``_prep_log_tree``.
        Q: The rate matrix (shape: [num_states, num_states]).
        root_logits: Unnormalized log-probabilities for the root state
            distribution (shape: [num_states]).
        state2idx: Optional mapping from state labels to indices.
        growth_rates: Optional 1D tensor of birth rates per type.

    Returns:
        The summed log-likelihood (a scalar tensor).
    """
    # NOTE: Assumes that trees is the output of _prep_log_tree
    if not isinstance(trees, list):
        trees = [trees]
    log_liks = 0
    for i, tree in enumerate(trees):
        branch_lens = tree.branch_lens
        transition_tensor = compute_transition_tensor(branch_lens, Q, growth_rates=growth_rates)
        log_lik = log_vectorized_felsenstein_pruning(tree, transition_tensor, root_logits, state2idx=state2idx, growth_rates=growth_rates)
        log_liks = log_liks + log_lik

    return log_liks

def log_vectorized_felsenstein_pruning(tree,
                                       transition_tensor,
                                       root_logits,
                                       state2idx=None,
                                       growth_rates=None):
    """Computes the log-likelihood via Felsenstein's pruning algorithm.

    Performs a post-order traversal of the tree, computing log partial
    likelihoods at each node using log-space arithmetic to avoid underflow.

    Args:
        tree: An ete3 tree prepped via ``_prep_log_tree``, with ``branch_lens``
            tensor attached and integer ``name`` fields for indexing.
        transition_tensor: Precomputed transition matrices (shape:
            [num_nodes, num_states, num_states]).
        root_logits: Unnormalized log-probabilities for the root state
            distribution (shape: [num_states]).
        state2idx: Optional mapping from state labels to indices. Defaults
            to the identity mapping.
        growth_rates: Optional 1D tensor of birth rates. When present, a
            log(growth_rates) term is added at internal bifurcations.

    Returns:
        The log-likelihood of the tree (a scalar tensor).
    """

    num_states = len(root_logits)
    if state2idx is None:
        state2idx = _default_state2idx(num_states)

    # Process nodes in post-order. Leaves come first.
    for i, node in enumerate(tree.traverse("postorder")):

        if node.is_leaf():
            # Initialize leaf log partials: 0 for the observed state, -inf for others.
            observed_state = node.state
            log_partial = -float('inf') * torch.ones(num_states, dtype=transition_tensor.dtype, device=transition_tensor.device)
            log_partial[state2idx[observed_state]] = 0  # log(1) = 0
            node.log_partials = log_partial
        else:
            # For an internal node, accumulate contributions from each child.
            # We will sum (in log-space) the contributions from each child.
            child_list = node.get_children()
            child = child_list[0]
            P = transition_tensor[child.name, :, :]
            logP = torch.log(P)     # NOTE: This requires that P is strictly positive.
            child_term = logP + child.log_partials.unsqueeze(0)
            log_sum = torch.logsumexp(child_term, dim=1)
            if growth_rates is not None and len(child_list) >= 2:
                # Log the growth rates, expand them, and add to left_contrib
                log_gr = torch.log(growth_rates)
                log_sum = log_sum + log_gr
            for child in child_list[1:]:
                P = transition_tensor[child.name, :, :]                 # shape: (num_states, num_states)
                logP = torch.log(P)                                     # shape: (num_states, num_states)
                child_term = logP + child.log_partials.unsqueeze(0)     # shape: (num_states, num_states)
                contribution = torch.logsumexp(child_term, dim=1)       # shape: (num_states,)
                log_sum = log_sum + contribution
            node.log_partials = log_sum

    # At the root, combine the computed log partials with the root state distribution.
    L_root = tree.log_partials  # shape: (num_states,)
    # Compute progenitor probabilities via softmax (differentiable) and take log.
    root_probs = F.softmax(root_logits, dim=0)
    root_log_probs = torch.log(root_probs)
    # The overall log likelihood is logsumexp over the states at the root.
    log_lik = torch.logsumexp(L_root + root_log_probs, dim=0)

    return log_lik


def _prep_log_tree(tree, num_states, state2idx=None, device=None):
    """Prepares an ete3 tree for use with log-space Felsenstein pruning.

    Deep-copies the tree, assigns integer node IDs in preorder, initializes
    leaf log-partials, and collects branch lengths into a single tensor.
    A unifurcating root is added if one is not already present.

    Args:
        tree: An ete3 TreeNode with ``.state`` attributes on leaves.
        num_states: The total number of states.
        state2idx: Optional mapping from state labels to indices. Defaults
            to the identity mapping.
        device: Torch device for tensors (default: CPU).

    Returns:
        The prepped ete3 tree with ``branch_lens`` tensor and
        ``log_partials`` set on each leaf.
    """
    if state2idx is None:
        state2idx = _default_state2idx(num_states)

    tree = tree.copy("deepcopy")
    if len(tree.children) != 1:
        root = TreeNode()
        root.name = "root"
        root.dist = 0
        root.add_child(tree)
        tree = root

    curr_id = 0
    node_list = list(tree.traverse("preorder"))

    branch_lens = torch.zeros(len(node_list))

    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node.time = 0
        else:
            node.time = max([child.dist + child.time for child in node.children])

    for node in node_list:
        node.name = curr_id
        if not node.is_root():
            branch_lens[curr_id] = node.dist

        if node.is_leaf():
            observed_state = node.state
            log_partials = -float('inf') * torch.ones(num_states)
            log_partials[state2idx[observed_state]] = 0.0
            node.log_partials = log_partials
            node.log_partials = node.log_partials.float()
            if device is not None:
                node.log_partials = node.log_partials.to(device)

        curr_id = curr_id + 1
    branch_lens = branch_lens.to(device)
    tree.branch_lens = branch_lens.float()

    return tree
