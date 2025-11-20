# tests/test_edge_matrix_rho1.py
#   When rho=1, E(t) = 0 and it should reduce exactly to the vanilla likelihood

import torch
from likelihood import solve_E_timeseries, EInterp, _transition_matrix_timevarying

def test_edge_matrix_matches_expm_when_rho1():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = torch.tensor([[-1.0,  1.0,  0.0],
                      [ 0.5, -1.0,  0.5],
                      [ 0.0,  1.0, -1.0]], device=device)
    lam = torch.tensor([0.3, 0.6, 0.9], device=device)
    rho = 1.0
    t_child, t_parent = 0.0, 2.5
    T = t_parent

    # E(t) (should be ~zero for rho=1)
    ts, Es = solve_E_timeseries(lam, Q, rho, T, n_grid=32)
    goal = torch.zeros(Es.shape)
    assert torch.allclose(Es, torch.zeros(Es.shape), atol=1e-10)
    E_interp = EInterp(ts, Es)

    # Numerical fundamental matrix
    P_num = _transition_matrix_timevarying(t_child, t_parent, Q, lam, E_interp)

    # Analytic constant-coefficient exponential
    A = Q - torch.diag(lam)
    P_ref = torch.matrix_exp(A * (t_parent - t_child))

    assert torch.allclose(P_num, P_ref, atol=1e-8)