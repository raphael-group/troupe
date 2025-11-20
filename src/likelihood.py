import torch
import torch.nn.functional as F
# NOTE: If you use adjoint, you need to set adjoint_params = (Q, lam, E_interp.Es)
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
from ete3 import TreeNode
import math

EPS = 1e-300  # Numerical stability parameter (rate matrix entries must be geq to avoid NaNs)
num_atol = 1e-12
num_rtol = 1e-8


def compute_progenitor_liklihood(tree, progenitor_logits):    
    sm = F.softmax(progenitor_logits, dim=0)
    return sm

def compute_transition_tensor(branch_lens, Q, growth_rates=None):
    """
    Computes tansition matrix for each branch in a parallelized GPU-friendly manner.
    """
    if growth_rates is not None:
        Q = Q - torch.diag(growth_rates)
    scaled_Q_tensor = Q.unsqueeze(0) * branch_lens.view(-1, 1, 1)
    return torch.matrix_exp(scaled_Q_tensor).to(Q.dtype)

def log_vec_likelihood(trees, Q, root_logits, state2idx=None, growth_rates=None, rho=1.0):
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
                                       growth_rates=None,
                                       rho=1.0):
    """
    Compute the likelihood under a substitution model using
    an iterative (vectorized) implementation of Felsenstein's pruning algorithm.
    The only difference between this method and vanilla vectorized_felsenstein_pruning is that
    this method performs operations in log-space and returns the log likelihood.
    
    Args:
        tree (ete tree): Assumes that each node has a.... `name` field, a partial
                         likelihood vector in the `obs` field, and a num_states x num_children
                         empty tensor at each non-leaf node in the `propogated` field.
        Q (torch.Tensor): Rate matrix (shape: [num_states, num_states]).
        root_probs (torch.Tensor): Distribution over states at time 0 (shape: [num_states]).  
        
    Returns:
        torch.Tensor: The likelihood of the tree given the parameters.
    """

    num_states = len(root_logits)
    if state2idx is None:
        state2idx = {i: i for i in range(num_states)}

    # Process nodes in post-order. Leaves come first.
    for i, node in enumerate(tree.traverse("postorder")):

        if node.is_leaf():
            # Initialize leaf log partials: 0 for the observed state, -inf for others.
            observed_state = node.state
            log_partial = -float('inf') * torch.ones(num_states, dtype=transition_tensor.dtype, device=transition_tensor.device)
            log_partial[state2idx[observed_state]] = math.log(rho)  # log(1) = 0
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
    root_probs = compute_progenitor_liklihood(tree, root_logits)
    root_log_probs = torch.log(root_probs)
    # The overall log likelihood is logsumexp over the states at the root.
    log_lik = torch.logsumexp(L_root + root_log_probs, dim=0)
    
    return log_lik


def _transition_matrix(branch_length, Q):
    """
    Compute the transition probability matrix for a given branch length via matrix exponentiation.
    Q is a torch tensor of shape (num_states, num_states).
    """
    return torch.matrix_exp(Q * branch_length).float()


def _prep_log_tree(tree, num_states, state2idx=None, device=None):
    """
    Companion method to vectorized_felsenstein_pruning that initializes the tree to contain the
    correct tensor information. For now, this includes adding a unique node id and the `partials`
    field which contains the conditional likelihood of the subtree.

    Args:
        tree (ete tree): Assumes that the leaves have state information indexed 0, ..., num_states-1.

    Returns:
        tree (ete tree): 
    """
    if state2idx is None:
        state2idx = {i: i for i in range(num_states)}

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


### Subsampling rate < 1.0 ###

class Logit_E_ODE(torch.nn.Module):
    """
    u'(t) = -lambda + (Q@E)/(E*(1-E)), with E = sigmoid(u)

    E(t)_i := the probability of not observing any descendants of the subsequent lineage given
              that it starts in state i (i.e., P(extinction|i) )
    """
    def __init__(self, lam, Q, eps=1e-12):
        super().__init__()
        self.lam = lam
        self.Q = Q
        self.eps = eps

    def forward(self, t, u):
        E = torch.sigmoid(u)
        QE = self.Q @ E
        return -self.lam + QE / (E * (1.0 - E) + self.eps)

def solve_E_timeseries(lam, Q, rho, T, n_grid=1024):
    k = len(lam)
    one_minus_rho = max(1e-12, float(1.0 - rho))
    u0_scalar = math.log(one_minus_rho) - math.log(1.0 - one_minus_rho)
    u0 = torch.full((k,), u0_scalar, device=lam.device, dtype=lam.dtype)
    rhs = Logit_E_ODE(lam, Q)
    ts = torch.linspace(0.0, float(T), steps=n_grid, device=lam.device, dtype=lam.dtype)
    us = odeint(rhs, u0, ts, rtol=num_rtol, atol=num_atol)  # (m,k)
    Es = torch.sigmoid(us)                                                   # (m,k)
    return ts, Es

class EInterp(torch.nn.Module):
    """Differentiable linear interpolator E(t)."""
    def __init__(self, ts, Es):
        super().__init__()
        self.register_buffer("ts", ts)   # (m,)
        self.register_buffer("Es", Es)   # (m,k)
        self.m = ts.shape[0]
        self.k = Es.shape[1]

    def forward(self, t):
        t = t.clamp(min=self.ts[0], max=self.ts[-1])
        idx = torch.searchsorted(self.ts, t).clamp(1, self.m - 1)
        t0 = self.ts[idx-1]; t1 = self.ts[idx]
        w = (t - t0) / (t1 - t0 + 1e-12)
        E0 = self.Es[idx-1]; E1 = self.Es[idx]
        return (1.0 - w) * E0 + w * E1  # (k,)

class FundMat_ODE(torch.nn.Module):
    """
    F'(t) = A(t) F(t),  with  A(t) = Q + diag(-lambda + 2*lambda*E(t)).
    State F is (k,k); odeint supports tensor states directly.
    """
    def __init__(self, lam, Q, E_interp):
        super().__init__()
        self.lam = lam
        self.Q = Q
        self.E_interp = E_interp

    def forward(self, t, F):
        E_t = self.E_interp(t)                           # (k,)
        A_diag = -self.lam + 2.0 * self.lam * E_t        # (k,)
        return (self.Q @ F) + A_diag.unsqueeze(1) * F

def _transition_matrix_timevarying(t_child, t_parent, Q, lam, E_interp, rtol=num_rtol, atol=num_atol):
    """
    Numerical P(t_parent, t_child) via F'(t)=A(t)F, F(t_child)=I.

    Args:
        t_child (float): absolute time of child (0 for tips).
        t_parent (float): absolute time of parent.
        Q (k,k), lam (k,), E_interp: from Pass 1.

    Returns:
        P (k,k) tensor (nonnegative); clamp small entries before log() downstream.
    """
    if abs(float(t_parent - t_child)) <= 1e-15:
        k = len(lam)
        return torch.eye(k, dtype=lam.dtype, device=lam.device)
    F0 = torch.eye(len(lam), dtype=lam.dtype, device=lam.device)
    rhs = FundMat_ODE(lam, Q, E_interp)
    tspan = torch.tensor([float(t_child), float(t_parent)], device=lam.device, dtype=lam.dtype)
    F = odeint(rhs, F0, tspan, rtol=rtol, atol=atol)[-1]  # (k,k)
    return F

def log_vec_likelihood_numerical(trees, Q, root_logits, state2idx=None, growth_rates=None, rho=1.0):
    """
    Driver for log_vectorized_felsenstein_pruning. Assumes trees is the output of _prep_log_tree
    """
    if not isinstance(trees, list):
        trees = [trees]
    log_liks = 0

    transition_tensor_list = build_transition_tensor_numerical(trees, growth_rates, Q, rho)
    for i, tree in enumerate(trees):
        branch_lens = tree.branch_lens
        log_lik = log_vectorized_felsenstein_pruning(tree, transition_tensor_list[i], root_logits,
                                                     state2idx=state2idx, growth_rates=growth_rates, rho=rho)
        log_liks = log_liks + log_lik

    return log_liks

def build_transition_tensor_numerical(trees, lam, Q, rho, E_grid=1024):
    device, dtype = lam.device, lam.dtype
    k = len(lam)

    # solve E(t) once for all trees
    T = max([tree.time for tree in trees])
    ts, Es = solve_E_timeseries(lam, Q, rho, T, n_grid=E_grid)
    E_interp = EInterp(ts, Es)

    out = []
    for tree in trees:
        N = len(list(tree.traverse()))
        Ps = {}
        I = torch.eye(k, dtype=dtype, device=device)
        for node in tree.traverse("postorder"):
            if node.is_root():
                continue
            t_c  = float(node.time)
            t_p  = float(node.up.time)
            P = _transition_matrix_timevarying(t_c, t_p, Q, lam, E_interp)  # keep graph
            Ps[int(node.name)] = P.clamp_min(EPS)
        rows = [Ps.get(i, I) for i in range(N)]     # include identity row(s) for root/missing
        out.append(torch.stack(rows, dim=0))        # (N,k,k), differentiable
    return out
