"""Unit tests for RateMatrixBuilder."""

import torch
import torch.nn.functional as F
import pytest

from models import RateMatrixBuilder

dtype = torch.float64
torch.set_default_dtype(dtype)


class TestRateMatrixBuilder:

    def test_rows_sum_to_zero(self, device):
        builder = RateMatrixBuilder(4, device=device)
        Q = builder.forward()
        row_sums = Q.sum(dim=1)
        assert torch.allclose(row_sums, torch.zeros(4, device=device, dtype=dtype), atol=1e-12)

    def test_off_diagonal_nonneg(self, device):
        builder = RateMatrixBuilder(3, device=device)
        Q = builder.forward()
        mask = ~torch.eye(3, dtype=torch.bool, device=device)
        assert (Q[mask] >= 0).all()

    def test_output_shape(self, device):
        n = 5
        builder = RateMatrixBuilder(n, device=device)
        Q = builder.forward()
        assert Q.shape == (n, n)

    def test_potency_constraints_mask_zeros(self, device):
        # State 0 can reach {0, 1}, state 1 can only reach {1}
        idx2potency = {0: (0, 1), 1: (1,)}
        builder = RateMatrixBuilder(2, potency_constraints=idx2potency, device=device)
        # Initialize free params to positive values to ensure nonzero where allowed
        with torch.no_grad():
            builder.free_params.fill_(1.0)
        Q = builder.forward()
        # State 1 cannot transition to state 0 (potency of 0 is {0,1} which is NOT contained in {1})
        assert Q[1, 0].item() == 0.0

    def test_potency_constraints_allow(self, device):
        # State 0 can reach {0, 1}, state 1 can only reach {1}
        idx2potency = {0: (0, 1), 1: (1,)}
        builder = RateMatrixBuilder(2, potency_constraints=idx2potency, device=device)
        with torch.no_grad():
            builder.free_params.fill_(1.0)
        Q = builder.forward()
        # State 0 → 1 should be allowed (potency {1} is subset of {0,1})
        assert Q[0, 1].item() > 0

    def test_hidden_states(self, device):
        n_obs = 3
        n_hidden = 2
        n_total = n_obs + n_hidden
        builder = RateMatrixBuilder(n_total, num_hidden_states=n_hidden, device=device)
        Q = builder.forward()
        assert Q.shape == (n_total, n_total)

    def test_gradient_flows(self, device):
        builder = RateMatrixBuilder(3, device=device)
        Q = builder.forward()
        Q.sum().backward()
        assert builder.free_params.grad is not None
        assert not torch.isnan(builder.free_params.grad).any()

    def test_init_Q_params_matrix(self, device):
        n = 3
        init = torch.ones((n, n), dtype=dtype, device=device)
        builder = RateMatrixBuilder(n, device=device, init_Q_params=init)
        # Without potency constraints, mask is all-True so free_params = n*n
        # (diagonal entries are free params but diagonal of Q is computed from row sums)
        num_free = int(builder.potency_constraint_mask.sum().item())
        assert len(builder.free_params) == num_free

    def test_init_Q_params_vector(self, device):
        n = 3
        builder_ref = RateMatrixBuilder(n, device=device)
        num_free = len(builder_ref.free_params)
        init_vec = torch.full((num_free,), 2.0, dtype=dtype, device=device)
        builder = RateMatrixBuilder(n, device=device, init_Q_params=init_vec)
        assert torch.allclose(builder.free_params, init_vec)
