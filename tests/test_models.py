"""Integration tests for CTMCLikelihoodModel and PureBirthLikelihoodModel."""

import numpy as np
import torch
import torch.nn.functional as F
import pytest

from models import CTMCLikelihoodModel, PureBirthLikelihoodModel
from branching_simulation import simulate_tree
from conftest import make_Q_params, make_growth_params

dtype = torch.float64
torch.set_default_dtype(dtype)


def _make_trees(Q, lam, num_trees=3, T=1.0, starting_type=0):
    trees = []
    for seed in range(num_trees):
        tree = simulate_tree(Q, lam, starting_type=starting_type, T=T, seed=seed * 17)
        trees.append(tree)
    return trees


class TestCTMCLikelihoodModel:

    def test_forward_finite(self, device, two_state_system):
        Q_np, lam_np, pi = two_state_system
        trees = _make_trees(Q_np, lam_np, num_trees=3)
        Q_params = make_Q_params(Q_np, device=device)
        growth_params = make_growth_params(lam_np, device=device)
        model = CTMCLikelihoodModel(
            trees, 2, Q_params=Q_params, pi_params=pi.to(device),
            device=device, growth_params=growth_params
        )
        for i in range(3):
            result = model(i)
            assert torch.isfinite(result)
            assert result.item() < 0  # Log-likelihood

    def test_get_rate_matrix_valid(self, device, two_state_system):
        Q_np, lam_np, pi = two_state_system
        trees = _make_trees(Q_np, lam_np, num_trees=1)
        Q_params = make_Q_params(Q_np, device=device)
        growth_params = make_growth_params(lam_np, device=device)
        model = CTMCLikelihoodModel(
            trees, 2, Q_params=Q_params, pi_params=pi.to(device),
            device=device, growth_params=growth_params
        )
        Q = model.get_rate_matrix()
        # Rows sum to zero
        assert torch.allclose(Q.sum(dim=1), torch.zeros(2, device=device, dtype=dtype), atol=1e-10)
        # Off-diagonal non-negative
        mask = ~torch.eye(2, dtype=torch.bool, device=device)
        assert (Q[mask] >= 0).all()

    def test_get_root_distribution_sums_to_one(self, device, two_state_system):
        Q_np, lam_np, pi = two_state_system
        trees = _make_trees(Q_np, lam_np, num_trees=1)
        Q_params = make_Q_params(Q_np, device=device)
        growth_params = make_growth_params(lam_np, device=device)
        model = CTMCLikelihoodModel(
            trees, 2, Q_params=Q_params, pi_params=pi.to(device),
            device=device, growth_params=growth_params
        )
        root_dist = model.get_root_distribution()
        assert torch.allclose(root_dist.sum(), torch.tensor(1.0, dtype=dtype, device=device), atol=1e-10)

    def test_parameters_optimizable(self, device, two_state_system):
        Q_np, lam_np, pi = two_state_system
        trees = _make_trees(Q_np, lam_np, num_trees=2)
        Q_params = make_Q_params(Q_np, device=device)
        growth_params = make_growth_params(lam_np, device=device)
        model = CTMCLikelihoodModel(
            trees, 2, Q_params=Q_params, pi_params=pi.to(device),
            device=device, growth_params=growth_params
        )
        loss = -model(0)
        loss.backward()
        # At least Q free_params should have gradients
        assert model.rate_builder.free_params.grad is not None


class TestPureBirthLikelihoodModel:

    def test_growth_rates_positive(self, device, two_state_system):
        Q_np, lam_np, pi = two_state_system
        trees = _make_trees(Q_np, lam_np, num_trees=2)
        Q_params = make_Q_params(Q_np, device=device)
        growth_params = make_growth_params(lam_np, device=device)
        model = PureBirthLikelihoodModel(
            trees, 2, Q_params, pi.to(device), growth_params,
            device=device, num_hidden=0
        )
        gr = model.get_growth_rates()
        assert (gr > 0).all()

    def test_start_state_fixing(self, device, two_state_system):
        Q_np, lam_np, pi = two_state_system
        trees = _make_trees(Q_np, lam_np, num_trees=2)
        Q_params = make_Q_params(Q_np, device=device)
        growth_params = make_growth_params(lam_np, device=device)
        idx2state = {0: 0, 1: 1}
        model = PureBirthLikelihoodModel(
            trees, 2, Q_params, pi.to(device), growth_params,
            device=device, num_hidden=0, start_state=0, idx2state=idx2state
        )
        root_dist = model.get_root_distribution()
        # Almost all mass should be on state 0
        assert root_dist[0].item() > 0.99
        # pi_params should not require grad when start_state is set
        assert model.pi_params.requires_grad is False

    def test_potency_constraints_applied(self, device):
        # 3-state: state 0 can reach {0,1,2}, state 1 can reach {1}, state 2 can reach {2}
        Q_np = np.array([[-2.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])
        lam_np = np.array([2.0, 1.0, 1.0])
        pi = torch.tensor([1e20, -1e20, -1e20], dtype=dtype)
        idx2potency = {0: (0, 1, 2), 1: (1,), 2: (2,)}
        idx2state = {0: 0, 1: 1, 2: 2}
        trees = _make_trees(Q_np, lam_np, num_trees=2)
        Q_params = make_Q_params(Q_np, device=device)
        growth_params = make_growth_params(lam_np, device=device)
        model = PureBirthLikelihoodModel(
            trees, 3, Q_params, pi.to(device), growth_params,
            device=device, num_hidden=0, idx2potency=idx2potency, idx2state=idx2state
        )
        Q = model.get_rate_matrix()
        # State 1 and 2 cannot transition anywhere (terminal states)
        assert Q[1, 0].item() == 0.0
        assert Q[1, 2].item() == 0.0
        assert Q[2, 0].item() == 0.0
        assert Q[2, 1].item() == 0.0

    def test_hidden_states_expand_matrix(self, device):
        n_obs = 3
        n_hidden = 2
        n_total = n_obs + n_hidden
        Q_np = np.zeros((n_total, n_total))
        Q_np[3, 0] = 1.0
        Q_np[3, 1] = 1.0
        Q_np[3, 3] = -2.0
        Q_np[4, 1] = 1.0
        Q_np[4, 2] = 1.0
        Q_np[4, 4] = -2.0
        lam_np = np.ones(n_total)
        pi = torch.tensor([1e20] + [-1e20] * (n_total - 1), dtype=dtype)
        trees = _make_trees(Q_np, lam_np, num_trees=2, starting_type=3)
        Q_params = make_Q_params(Q_np, device=device)
        growth_params = make_growth_params(lam_np, device=device)
        model = PureBirthLikelihoodModel(
            trees, n_total, Q_params, pi.to(device), growth_params,
            device=device, num_hidden=n_hidden
        )
        Q = model.get_rate_matrix()
        assert Q.shape == (n_total, n_total)
