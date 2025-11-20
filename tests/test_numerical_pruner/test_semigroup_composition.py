import torch
import pytest
from ete3 import Tree
from models import TreeTensorizer, FelsensteinPruner

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_chain_tree():
    # 3-node chain: a <- b <- c (c is a leaf with state 0)
    # Newick: ((0:0.3):0.4):0.5;
    t = Tree("((0:0.3):0.4):0.5;", format=1)
    for lf in t:
        lf.add_features(state=int(lf.name))
    return t

def test_semigroup_composition():
    k = 3
    lam = torch.tensor([0.4, 0.7, 1.1], device=device)
    # Build a valid generator Q with positive off-diagonals and zero row sums
    Qoff = torch.tensor([[0.0, 0.6, 0.2],
                         [0.1, 0.0, 0.3],
                         [0.2, 0.5, 0.0]], device=device)
    Q = Qoff - torch.diag(Qoff.sum(dim=1))
    rho = 0.6

    t = make_chain_tree()
    tens = TreeTensorizer([t], num_states=k, device=device)
    pruner = FelsensteinPruner(k)

    # One-shot precompute of Phi on the global time grid
    time_grid = tens.global_time_grid.to(device=device, dtype=torch.float64)
    pruner.prepare_global_cache(time_grid, lam, Q, rho)

    # Build transitions for the (only) tree
    abs_times = tens.abs_times[0]
    parents   = tens.parents[0]
    blens     = tens.branch_lens[0]
    trans = pruner.build_transitions_from_cache(abs_times, parents, blens)  # (N,k,k)

    # find c (leaf), b = parent[c], a = parent[b]
    # there is exactly one leaf in this chain
    c = int((abs_times == 0).nonzero(as_tuple=True)[0][0].item())
    b = int(parents[c].item())
    a = int(parents[b].item())

    # P(a->b) == trans[b], P(b->c) == trans[c]
    Pab = trans[b]
    Pbc = trans[c]

    # Direct P(a->c) from cached Phi
    Phi = pruner._cache["Phi"]
    tg  = pruner._cache["time_grid"]
    idx_a = torch.searchsorted(tg, abs_times[a])
    idx_c = torch.searchsorted(tg, abs_times[c])
    # Solve instead of explicit inverse: (Phi_c^T) X^T = Phi_a^T
    X_T = torch.linalg.solve(Phi[idx_c].T, Phi[idx_a].T)
    Pac_direct = X_T.T

    Pac_via_b = Pab @ Pbc
    assert torch.allclose(Pac_direct, Pac_via_b, rtol=1e-7, atol=1e-9)
