# tests/test_edge_matrix_lambda0.py

import torch
from likelihood import solve_E_timeseries, EInterp, _transition_matrix_timevarying

def test_lambda0_reduces_to_expmQ():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = 3
    Q = torch.tensor([[-2.0, 2.0, 0.0],
                      [ 0.0,-1.0, 1.0],
                      [ 0.0, 0.0, 0.0]], device=device)
    lam = torch.zeros(k, device=device)
    rho = 0.2
    t_child, t_parent = 0.4, 3.1
    T = t_parent

    ts, Es = solve_E_timeseries(lam, Q, rho, T, n_grid=16)  # trivial E
    E_interp = EInterp(ts, Es)
    P_num = _transition_matrix_timevarying(t_child, t_parent, Q, lam, E_interp)
    P_ref = torch.matrix_exp(Q * (t_parent - t_child))

    print(P_num)
    print(P_ref)
    print(P_num - P_ref)

    assert torch.allclose(P_num, P_ref, atol=1e-8)