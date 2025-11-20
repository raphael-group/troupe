import torch
import pytest
from ete3 import Tree
from models import TreeTensorizer, FelsensteinPruner

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_mixed_tree():
    t = Tree("((0:0.3,1:0.3):0.5,(2:0.2,0:0.2):0.6);", format=1)
    for lf in t:
        lf.add_features(state=int(lf.name))
    return t

def test_lambda_zero_reduces_to_expQ():
    k = 3
    lam = torch.zeros(k, device=device)
    Qoff = torch.tensor([[0.0, 0.5, 0.1],
                         [0.2, 0.0, 0.3],
                         [0.1, 0.4, 0.0]], device=device)
    Q = Qoff - torch.diag(Qoff.sum(dim=1))
    rho = 0.35  # arbitrary. Shouldn't matter since lam = 0

    t = make_mixed_tree()
    tens = TreeTensorizer([t], num_states=k, device=device)
    pruner = FelsensteinPruner(k)

    tg = tens.global_time_grid.to(device=device, dtype=torch.float64)
    pruner.prepare_global_cache(tg, lam, Q, rho)

    abs_times = tens.abs_times[0]
    parents   = tens.parents[0]
    blens     = tens.branch_lens[0]
    trans     = pruner.build_transitions_from_cache(abs_times, parents, blens)

    print("Unique times")
    print(abs_times)
    print("Branch lens")
    print(blens)

    A = Q - torch.diag(lam)  # constant
    for i in range(abs_times.numel()):
        p = int(parents[i].item())
        dt = float(blens[i].item())
        P_ref = torch.matrix_exp(A * dt)

        print(i)
        print(trans[i])
        print(P_ref)
        print(trans[i] - P_ref)

        # NOTE: Off by ~0.03 for time scale of 0.3

        assert torch.allclose(trans[i], P_ref, atol=1e-6)

def test_Q_zero_closed_form_diagonal():
    k = 4
    lam = torch.tensor([0.3, 0.7, 1.2, 1.8], device=device)
    Q = torch.zeros((k, k), device=device)
    rho = 0.4

    t = make_mixed_tree()
    tens = TreeTensorizer([t], num_states=k, device=device)
    pruner = FelsensteinPruner(k)

    tg = tens.global_time_grid.to(device=device, dtype=torch.float64)
    pruner.prepare_global_cache(tg, lam, Q, rho)

    abs_times = tens.abs_times[0]
    parents   = tens.parents[0]
    blens     = tens.branch_lens[0]
    trans = pruner.build_transitions_from_cache(abs_times, parents, blens)

    def closed_form_diag(li, tc, tp, rho):
        dt = tp - tc
        num = rho + (1.0 - rho) * torch.exp(-li * tc)
        den = rho + (1.0 - rho) * torch.exp(-li * tp)
        return torch.exp(-li * dt) * (num / den) ** 2

    I = torch.eye(k, device=device, dtype=torch.float64)
    for i in range(abs_times.numel()):
        p = int(parents[i].item())
        if p < 0 or blens[i] <= 1e-14:
            assert torch.allclose(trans[i], I)
            continue
        tc = abs_times[i]; tp = abs_times[p]
        diag_vals = torch.stack([closed_form_diag(lam[j], tc, tp, torch.tensor(rho, device=device))
                                 for j in range(k)])
        P_ref = torch.diag(diag_vals)

        print(i)
        print(trans[i])
        print(P_ref)
        print(trans[i] - P_ref)

        assert torch.allclose(trans[i], P_ref, atol=1e-8)
