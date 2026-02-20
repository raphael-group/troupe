"""Cross-validation between log_vec_likelihood (likelihood.py) and PureBirthLikelihoodModel (models.py).

This is the most critical test file: it verifies that the two independent
Felsenstein pruning implementations produce the same log-likelihood values.
"""

import numpy as np
import torch
import torch.nn.functional as F
import pytest

from likelihood import log_vec_likelihood, _prep_log_tree
from branching_simulation import simulate_tree
from models import PureBirthLikelihoodModel

from conftest import make_Q_params, make_growth_params, simulate_and_prep_trees, are_all_within_k_sig_figs

dtype = torch.float64
torch.set_default_dtype(dtype)


def _compare_implementations(trees, Q_np, pi_params, lam_np, device, num_hidden=0, idx2potency=None):
    """Compute log-likelihood from both implementations and return both values."""
    Q_tensor = torch.tensor(Q_np, dtype=dtype, device=device)
    pi = pi_params.to(device)
    growth_rates = torch.tensor(lam_np, dtype=dtype, device=device)

    Q_params = make_Q_params(Q_np, device=device)
    growth_params = make_growth_params(lam_np, device=device)

    # Method 1: log_vec_likelihood (likelihood.py)
    lik_vec = log_vec_likelihood(trees, Q_tensor, pi, growth_rates=growth_rates)

    # Method 2: PureBirthLikelihoodModel (models.py)
    model = PureBirthLikelihoodModel(
        [tree.copy("deepcopy") for tree in trees],
        len(Q_np),
        Q_params,
        pi,
        growth_params,
        device=device,
        num_hidden=num_hidden,
        idx2potency=idx2potency,
    )
    model = model.to(device)
    lik_model = sum(model(i) for i in range(len(trees)))

    return lik_vec.item(), lik_model.item()


class TestLikelihoodAgreement:

    def test_two_state_agreement(self, device, two_state_system):
        Q, lam, pi = two_state_system
        trees = simulate_and_prep_trees(Q, lam, num_trees=5, T=1.0, starting_type=0)
        lik_vec, lik_model = _compare_implementations(trees, Q, pi, lam, device)
        assert are_all_within_k_sig_figs([lik_vec, lik_model], k=4)

    def test_three_state_agreement(self, device, three_state_system):
        Q, lam, pi = three_state_system
        trees = simulate_and_prep_trees(Q, lam, num_trees=10, T=1.0, starting_type=0, seed_multiplier=37)
        lik_vec, lik_model = _compare_implementations(trees, Q, pi, lam, device)
        assert are_all_within_k_sig_figs([lik_vec, lik_model], k=4)

    def test_seven_state_with_hidden_agreement(self, device, seven_state_system):
        Q, lam, pi = seven_state_system
        trees = simulate_and_prep_trees(Q, lam, num_trees=32, T=1.75, starting_type=4, seed_multiplier=37)
        lik_vec, lik_model = _compare_implementations(trees, Q, pi, lam, device, num_hidden=0)
        assert are_all_within_k_sig_figs([lik_vec, lik_model], k=4)

    def test_agreement_varying_growth_rates(self, device):
        """Test with asymmetric growth rates."""
        Q = np.array([[-1.0, 1.0],
                      [0.0, 0.0]])
        lam = np.array([3.0, 0.5])
        pi = torch.tensor([1e20, -1e20], dtype=dtype)
        trees = simulate_and_prep_trees(Q, lam, num_trees=10, T=1.0, starting_type=0)
        lik_vec, lik_model = _compare_implementations(trees, Q, pi, lam, device)
        assert are_all_within_k_sig_figs([lik_vec, lik_model], k=4)

    def test_agreement_large_trees(self, device, two_state_system):
        """Test with larger trees (longer simulation time)."""
        Q, lam, pi = two_state_system
        trees = simulate_and_prep_trees(Q, lam, num_trees=3, T=3.0, starting_type=0)
        lik_vec, lik_model = _compare_implementations(trees, Q, pi, lam, device)
        assert are_all_within_k_sig_figs([lik_vec, lik_model], k=4)
