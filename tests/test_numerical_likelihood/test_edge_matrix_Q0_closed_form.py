# tests/test_edge_matrix_Q0_closed_form.py
#   When there is no transitions the edge matrix has a closed form

import torch
from likelihood import solve_E_timeseries, EInterp, _transition_matrix_timevarying
import math

def closed_form_P_diag(lam, rho, t_c, t_p):
    # lam: (k,)
    dt = t_p - t_c
    k = len(lam)
    out = torch.empty(k, dtype=lam.dtype, device=lam.device)
    for i in range(k):
        li = lam[i]
        num = rho + (1.0 - rho) * math.exp(-li * t_c)
        den = rho + (1.0 - rho) * math.exp(-li * t_p)
        out[i] = math.exp(-li * dt) * (num / den) ** 2
    return out

def test_Q0_closed_form_diagonal():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = 4
    Q = torch.zeros((k, k), device=device)
    lam = torch.tensor([0.2, 0.5, 1.1, 2.0], device=device)
    rho = 0.3
    t_child, t_parent = 0.0, 3.7
    T = t_parent

    ts, Es = solve_E_timeseries(lam, Q, rho, T, n_grid=1024)
    E_interp = EInterp(ts, Es)

    P_num = _transition_matrix_timevarying(t_child, t_parent, Q, lam, E_interp, atol=1e-12)
    P_closed_form_diag = closed_form_P_diag(lam, rho, t_child, t_parent)

    print(P_num)
    print(torch.diag(P_closed_form_diag))
    print(P_num - torch.diag(P_closed_form_diag))

    # off-diagonals should be 0 and diagonals should match closed form
    assert torch.allclose(P_num, torch.diag(P_closed_form_diag), atol=1e-5)