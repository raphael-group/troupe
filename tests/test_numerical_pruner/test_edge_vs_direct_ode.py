import torch
import pytest
from ete3 import Tree
from models import TreeTensorizer, FelsensteinPruner
from torchdiffeq import odeint

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_small_binary_tree():
    # ((0:0.4,1:0.4):0.3,2:0.7);
    t = Tree("((0:0.4,1:0.4):0.3,2:0.7);", format=1)
    for lf in t:
        lf.add_features(state=int(lf.name))
    return t

def test_edge_transition_matches_direct_ode():
    k = 3
    lam = torch.tensor([0.5, 0.9, 1.3], device=device)
    Qoff = torch.tensor([[0.0, 0.4, 0.2],
                         [0.3, 0.0, 0.1],
                         [0.2, 0.5, 0.0]], device=device)
    Q = Qoff - torch.diag(Qoff.sum(dim=1))
    rho = 0.45

    t = make_small_binary_tree()
    tens = TreeTensorizer([t], num_states=k, device=device)
    pruner = FelsensteinPruner(k)

    # precompute Phi on shared grid
    tg = tens.global_time_grid.to(device=device, dtype=torch.float64)
    pruner.prepare_global_cache(tg, lam, Q, rho)

    abs_times = tens.abs_times[0]
    parents   = tens.parents[0]
    blens     = tens.branch_lens[0]
    trans = pruner.build_transitions_from_cache(abs_times, parents, blens)  # (N,k,k)

    # Rebuild E(t) interp exactly as the pruner did (uses cached dense grid)
    Phi_cache = pruner._cache  # just for access to E-grid

    # Dense E grid:
    # Grab the max time and rebuild E_interp
    T_max = float(tg[-1].item())
    ts_dense = torch.linspace(0.0, T_max, steps=1024, device=device, dtype=torch.float64)
    u0_scalar = torch.log(torch.tensor(1 - rho, device=device, dtype=torch.float64).clamp_min(1e-12)) \
                - torch.log(torch.tensor(rho, device=device, dtype=torch.float64) + 1e-12*0)
    u0 = u0_scalar.repeat(k)
    us = odeint(pruner.Logit_E_ODE(lam, Q), u0, ts_dense, rtol=1e-6, atol=1e-9)
    Es = torch.sigmoid(us)
    E_interp = pruner._EInterp(ts_dense, Es)

    class EdgeODE(torch.nn.Module):
        def __init__(self, lam, Q, E_interp):
            super().__init__()
            self.lam = lam; self.Q = Q; self.E_interp = E_interp
        def forward(self, t, F):
            E_t = self.E_interp(t)
            A_diag = -self.lam + 2.0*self.lam*E_t
            return (self.Q @ F) + A_diag.unsqueeze(1) * F

    rhs = EdgeODE(lam, Q, E_interp)
    I = torch.eye(k, device=device, dtype=torch.float64)

    # Check every non-root, non-zero edge
    for i in range(abs_times.numel()):
        p = int(parents[i].item())
        if p < 0 or blens[i] <= 1e-14:
            continue
        t_c = float(abs_times[i].item())
        t_p = float(abs_times[p].item())
        F = odeint(rhs, I, torch.tensor([t_c, t_p], device=device, dtype=torch.float64),
                   rtol=1e-6, atol=1e-9)[-1]
        P_direct = F
        P_cache  = trans[i]

        print(P_cache)
        print(P_direct)
        print(P_cache - P_direct)
        
        assert torch.allclose(P_cache, P_direct, atol=1e-4)
