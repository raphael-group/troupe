# tests/test_tree_likelihood_Q0_analytic.py
import torch
from likelihood import _prep_log_tree, log_vectorized_felsenstein_pruning
from likelihood import build_transition_tensor_numerical
import math
from ete3 import Tree

def build_transition_tensor_diag_analytic(tree, lam, rho):
    # Build analytic diagonal P for Q=0 using the closed form in `test_edge_matrix_Q0_closed_form`.
    k = len(lam)
    N = len(list(tree.traverse()))
    TT = torch.zeros((N, k, k), dtype=lam.dtype, device=lam.device)

    def P_diag(t_c, t_p):
        dt = t_p - t_c
        out = torch.empty(k, dtype=lam.dtype, device=lam.device)
        for i in range(k):
            li = lam[i].item()
            num = rho + (1 - rho) * math.exp(-li * t_c)
            den = rho + (1 - rho) * math.exp(-li * t_p)
            out[i] = math.exp(-li * dt) * (num / den) ** 2
        return torch.diag(out)

    for node in tree.traverse("postorder"):
        if node.is_root(): 
            continue
        TT[node.name, :, :] = P_diag(node.time, node.up.time).clamp_min(1e-300)
    return TT

def test_tree_likelihood_Q0_matches_analytic():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a small random-ish binary tree with known times and leaf states
    newick = "((0:0.7,1:0.7):0.5,(2:0.4,0:0.4):0.8);"
    t = Tree(newick, format=1)
    for node in t.traverse():
        if node.is_leaf():
            node.state = int(node.name)
    k = 3
    tree = _prep_log_tree(t, k)

    Q = torch.zeros((k, k), device=device)
    lam = torch.tensor([0.3, 0.9, 1.5], device=device)
    rho = 0.4
    pi_logits = torch.tensor([0.0, 0.0, 0.0], device=device)

    # numerical transition tensor (ODE)
    TT_num = build_transition_tensor_numerical([tree], lam, Q, rho)[0]
    # analytic diagonal transition tensor
    TT_ana = build_transition_tensor_diag_analytic(tree, lam, rho)

    logL_num = log_vectorized_felsenstein_pruning(tree, TT_num, pi_logits, rho=rho)
    logL_ana = log_vectorized_felsenstein_pruning(tree, TT_ana, pi_logits, rho=rho)

    print(logL_num)
    print(logL_ana)
    print(logL_num-logL_ana)

    assert torch.allclose(logL_num, logL_ana, atol=1e-4)
