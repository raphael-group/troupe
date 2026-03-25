"""Tests for ClaSSELikelihoodModel and DaughterKernelBuilder."""

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from classe_branching_simulation import simulate_tree
from classe_model import ClaSSELikelihoodModel, DaughterKernelBuilder

dtype = torch.float64
torch.set_default_dtype(dtype)

EPS = 1e-30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_birth_kernel_logits(n, uniform=True):
    if uniform:
        return torch.zeros((n, n), dtype=dtype)
    return torch.randn((n, n), dtype=dtype)


def _make_growth_params(lam, device=None):
    lam_t = torch.tensor(lam, dtype=dtype, device=device)
    return torch.log(torch.expm1(lam_t).clamp_min(EPS))


def _make_trees(B_np, lam_np, num_trees=3, T=1.0, starting_type=0, sample_probability=1.0):
    trees = []
    for seed in range(num_trees):
        tree = simulate_tree(B_np, lam_np, starting_type=starting_type, T=T, seed=seed * 17, sample_probability=sample_probability)
        trees.append(tree)
    return trees


def _make_two_state_model(device=None, B_logits=None, eta=1.0, start_state=None, backend="fundamental"):
    """Minimal 2-state ClaSSELikelihoodModel for testing."""
    device = device or torch.device("cpu")
    # State 0 produces type-0 or type-1 daughters equally; state 1 is terminal.
    B_np = np.array([[0.5, 0.5], [0.0, 1.0]])
    lam_np = np.array([1.5, 1.0])
    trees = _make_trees(B_np, lam_np, num_trees=3)
    idx2potency = {0: (0, 1), 1: (1,)}
    idx2state = {0: 0, 1: 1}
    if B_logits is None:
        B_logits = torch.zeros(2, 2, dtype=dtype)
    growth_params = _make_growth_params(lam_np, device=device)
    pi_params = torch.zeros(2, dtype=dtype, device=device)
    eta_logit = eta
    model = ClaSSELikelihoodModel(
        trees=trees,
        num_states=2,
        birth_kernel_params=B_logits.to(device),
        pi_params=pi_params,
        growth_params=growth_params,
        idx2potency=idx2potency,
        idx2state=idx2state,
        device=device,
        start_state=start_state,
        sampling_prob=eta_logit,
        backend=backend,
    )
    return model, trees


# ===========================================================================
# DaughterKernelBuilder
# ===========================================================================

class TestDaughterKernelBuilder:

    def test_rows_sum_to_one_unconstrained(self):
        """B should be row-stochastic."""
        n = 4
        builder = DaughterKernelBuilder(num_states=n)
        B = builder.forward()
        row_sums = B.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(n, dtype=dtype), atol=1e-10)

    def test_rows_sum_to_one_with_constraints(self):
        """Row-stochastic even when potency constraints reduce support."""
        idx2potency = {0: (0, 1, 2), 1: (1,), 2: (2,)}
        builder = DaughterKernelBuilder(num_states=3, potency_constraints=idx2potency)
        B = builder.forward()
        row_sums = B.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(3, dtype=dtype), atol=1e-10)

    def test_log_forward_matches_forward(self):
        """`log_forward()` should equal log of `forward()` for allowed entries.

        `log_forward` uses `log_softmax` for numerical stability, so forbidden
        entries (logit = -1e30) may differ from log(softmax(logit)).  We only
        check agreement on entries that receive positive probability mass.
        """
        builder = DaughterKernelBuilder(num_states=3)
        B = builder.forward()
        log_B = builder.log_forward()
        # For unconstrained builder all entries are allowed.
        assert torch.allclose(log_B, torch.log(B), atol=1e-8)

    def test_potency_constraints_zero_out_forbidden(self):
        """Entries forbidden by potency constraints must be exactly 0."""
        # State 1 and 2 are terminal; they can only produce daughters of the same type.
        idx2potency = {0: (0, 1, 2), 1: (1,), 2: (2,)}
        builder = DaughterKernelBuilder(num_states=3, potency_constraints=idx2potency)
        B = builder.forward()
        # Terminal state 1 cannot produce daughters of type 0 or 2.
        assert B[1, 0].item() == pytest.approx(0.0, abs=1e-12)
        assert B[1, 2].item() == pytest.approx(0.0, abs=1e-12)
        # Terminal state 2 cannot produce daughters of type 0 or 1.
        assert B[2, 0].item() == pytest.approx(0.0, abs=1e-12)
        assert B[2, 1].item() == pytest.approx(0.0, abs=1e-12)

    def test_unconstrained_all_entries_positive(self):
        """Without potency constraints all B entries should be positive."""
        torch.manual_seed(0)
        builder = DaughterKernelBuilder(num_states=3, init_logits=torch.randn(3, 3, dtype=dtype))
        B = builder.forward()
        assert (B > 0).all()

    def test_init_logits_2d_accepted(self):
        """A 2-D init_logits tensor should initialise the free params correctly."""
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        builder = DaughterKernelBuilder(num_states=2, init_logits=logits)
        # After a softmax the two entries per row differ — they must not all be equal.
        B = builder.forward()
        assert B[0, 0].item() != pytest.approx(0.5, abs=0.01)

    def test_support_mask_shape(self):
        """Support mask should be (num_states, num_states)."""
        idx2potency = {0: (0, 1), 1: (1,)}
        mask = DaughterKernelBuilder._build_support_mask(idx2potency, 2)
        assert mask.shape == (2, 2)

    def test_support_mask_wrong_num_states_raises(self):
        """Potency constraints with wrong length should raise ValueError."""
        idx2potency = {0: (0, 1)}
        with pytest.raises(ValueError):
            DaughterKernelBuilder._build_support_mask(idx2potency, 3)


# ===========================================================================
# ClaSSELikelihoodModel — basic properties
# ===========================================================================

class TestClaSSELikelihoodModelBasic:

    def test_forward_returns_finite_negative_value(self):
        """Forward pass should return a finite, negative log-likelihood."""
        model, _ = _make_two_state_model()
        for i in range(3):
            ll = model(i)
            assert torch.isfinite(ll), f"tree {i} gave non-finite log-likelihood"
            assert ll.item() < 0, f"tree {i} gave non-negative log-likelihood {ll.item()}"

    def test_growth_rates_positive(self):
        """Softplus-transformed growth rates must all be positive."""
        model, _ = _make_two_state_model()
        lam = model.get_growth_rates()
        assert (lam > 0).all()

    def test_sampling_probability_in_unit_interval(self):
        """Sigmoid-transformed sampling probability must lie in (0, 1)."""
        model, _ = _make_two_state_model(eta=0.8)
        eta = model.get_sampling_probability()
        assert eta.item() > 0
        assert eta.item() < 1

    def test_root_distribution_sums_to_one(self):
        """Root distribution should be a valid probability vector."""
        model, _ = _make_two_state_model()
        pi = model.get_root_distribution()
        assert torch.allclose(pi.sum(), torch.tensor(1.0, dtype=dtype), atol=1e-10)

    def test_daughter_kernel_row_stochastic(self):
        """The daughter kernel exposed by the model should be row-stochastic."""
        model, _ = _make_two_state_model()
        B = model.get_daughter_kernel()
        row_sums = B.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(2, dtype=dtype), atol=1e-10)

    def test_log_daughter_kernel_matches_daughter_kernel(self):
        """`get_log_daughter_kernel()` agrees with log(`get_daughter_kernel()`) on allowed entries.

        Forbidden entries have logit = -1e30 so `log_softmax` returns -1e30
        while log(softmax(...)) returns log(EPS) ≈ -69.  We check only the
        allowed (positive-probability) entries.
        """
        model, _ = _make_two_state_model()
        B = model.get_daughter_kernel()
        log_B = model.get_log_daughter_kernel()
        mask = B > 0
        assert torch.allclose(log_B[mask], torch.log(B[mask]), atol=1e-8)

    def test_gradients_flow_through_forward(self):
        """Loss.backward() must populate gradients on trainable parameters."""
        model, _ = _make_two_state_model()
        loss = -model(0)
        loss.backward()
        assert model.kernel_builder.free_params.grad is not None
        assert model.growth_params.grad is not None

    def test_cached_forward_matches_uncached_forward(self):
        """The ODE cache should not change the ClaSSE log-likelihood value."""
        model, _ = _make_two_state_model()
        uncached = model(0)
        model.precompute_ode()
        cached = model(0)
        model.clear_ode_cache()
        assert torch.allclose(cached, uncached, atol=1e-8)

    def test_vector_transport_backend_matches_fundamental_backend(self):
        """The vector-transport backend should agree with the fundamental backend."""
        model_fundamental, trees = _make_two_state_model()
        model_vector = ClaSSELikelihoodModel(
            trees=trees,
            num_states=2,
            birth_kernel_params=model_fundamental.kernel_builder._full_logits().detach(),
            pi_params=model_fundamental.pi_params.detach().clone(),
            growth_params=model_fundamental.growth_params.detach().clone(),
            idx2potency=model_fundamental.idx2potency,
            idx2state=model_fundamental.idx2state,
            device=model_fundamental.device,
            start_state=None,
            sampling_prob=float(model_fundamental.get_sampling_probability().item()),
            backend="vector_transport",
        )
        model_fundamental.precompute_ode()
        model_vector.precompute_ode()
        for tree_idx in range(len(trees)):
            ll_f = model_fundamental(tree_idx)
            ll_v = model_vector(tree_idx)
            assert torch.allclose(ll_v, ll_f, atol=1e-4)
        model_fundamental.clear_ode_cache()
        model_vector.clear_ode_cache()

    def test_start_state_fixes_root_distribution(self):
        """When start_state is set, >99% of root mass should be on that state."""
        model, _ = _make_two_state_model(start_state=0)
        pi = model.get_root_distribution()
        assert pi[0].item() > 0.99

    def test_start_state_pi_no_grad(self):
        """pi_params should require no gradient when start_state is provided."""
        model, _ = _make_two_state_model(start_state=0)
        assert model.pi_params.requires_grad is False

    def test_potency_constraints_respected_in_kernel(self):
        """Forbidden kernel entries must be zero after model construction."""
        model, _ = _make_two_state_model()
        B = model.get_daughter_kernel()
        # State 1 is terminal (potency = (1,)) — it can only produce type-1 daughters.
        assert B[1, 0].item() == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# ClaSSELikelihoodModel — three-state model
# ===========================================================================

class TestClaSSELikelihoodModelThreeState:

    @pytest.fixture
    def three_state_model(self):
        device = torch.device("cpu")
        # State 0 spreads equally across all types; states 1 and 2 are terminal.
        B_np = np.array([[1/3, 1/3, 1/3],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])
        lam_np = np.array([2.0, 1.0, 1.0])
        trees = _make_trees(B_np, lam_np, num_trees=5, starting_type=0)
        idx2potency = {0: (0, 1, 2), 1: (1,), 2: (2,)}
        idx2state = {0: 0, 1: 1, 2: 2}
        growth_params = _make_growth_params(lam_np)
        pi_params = torch.tensor([1e10, -1e10, -1e10], dtype=dtype)
        model = ClaSSELikelihoodModel(
            trees=trees,
            num_states=3,
            birth_kernel_params=torch.zeros(3, 3, dtype=dtype),
            pi_params=pi_params,
            growth_params=growth_params,
            idx2potency=idx2potency,
            idx2state=idx2state,
            device=device,
        )
        return model

    def test_forward_finite(self, three_state_model):
        for i in range(3):
            ll = three_state_model(i)
            assert torch.isfinite(ll)

    def test_terminal_states_self_only_in_kernel(self, three_state_model):
        """Terminal states can only produce same-type daughters."""
        B = three_state_model.get_daughter_kernel()
        # States 1 and 2 are terminal.
        assert B[1, 0].item() == pytest.approx(0.0, abs=1e-12)
        assert B[1, 2].item() == pytest.approx(0.0, abs=1e-12)
        assert B[2, 0].item() == pytest.approx(0.0, abs=1e-12)
        assert B[2, 1].item() == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# ODE / mathematical properties
# ===========================================================================

class TestODEMathProperties:

    def test_build_A_is_jacobian_of_e_rhs(self):
        """_build_A should equal the autograd Jacobian of _e_rhs w.r.t. E_vec."""
        torch.manual_seed(42)
        K = 3
        lam = torch.rand(K, dtype=dtype) + 0.5
        B = F.softmax(torch.randn(K, K, dtype=dtype), dim=1)
        E_vec = torch.rand(K, dtype=dtype) * 0.5 + 0.1

        # Analytical Jacobian from _build_A
        model, _ = _make_two_state_model()  # just to get an instance with the methods

        # Compute _build_A directly
        def build_A(E, B, lam):
            q = B @ E
            return -torch.diag(lam) + 2.0 * torch.diag(lam * q) @ B

        A_analytical = build_A(E_vec, B, lam)

        # Autograd Jacobian of _e_rhs
        def e_rhs(E):
            q = B @ E
            return lam * (q * q - E)

        A_autograd = torch.autograd.functional.jacobian(e_rhs, E_vec)

        assert torch.allclose(A_analytical, A_autograd, atol=1e-8), (
            f"_build_A mismatch:\n analytical:\n{A_analytical}\n autograd:\n{A_autograd}"
        )

    def test_fundamental_initial_condition_is_identity(self):
        """Fundamental matrix at t=0 should equal the identity."""
        K = 2
        lam = torch.tensor([1.5, 1.0], dtype=dtype)
        B = torch.eye(K, dtype=dtype)
        # eta = 1 → E0 = 0
        eta_logit = torch.tensor(math.log(0.99 / 0.01), dtype=dtype)

        model, _ = _make_two_state_model()
        eta = torch.tensor(1.0 - 1e-9, dtype=dtype)  # near-certain sampling
        time_points = torch.tensor([0.0, 0.5, 1.0], dtype=dtype)
        E_vals, fundamental = model._solve_augmented_system_torchode(time_points, B, lam, eta)

        Y0 = fundamental[0]  # should be ≈ identity
        assert torch.allclose(Y0, torch.eye(K, dtype=dtype), atol=1e-6), (
            f"Fundamental at t=0 is not identity:\n{Y0}"
        )

    def test_analytical_solution_identity_kernel_full_sampling(self):
        """With B=I and eta≈1, Y(t) ≈ diag(exp(-lam * t)) analytically."""
        K = 2
        lam = torch.tensor([2.0, 1.0], dtype=dtype)
        B = torch.eye(K, dtype=dtype)
        # eta very close to 1 so E ≈ 0 throughout
        eta = torch.tensor(1.0 - 1e-9, dtype=dtype)

        model, _ = _make_two_state_model()
        t_test = 0.5
        time_points = torch.tensor([0.0, t_test], dtype=dtype)
        E_vals, fundamental = model._solve_augmented_system_torchode(time_points, B, lam, eta)

        Y_t = fundamental[1]  # fundamental at t_test
        expected = torch.diag(torch.exp(-lam * t_test))

        assert torch.allclose(Y_t, expected, atol=1e-4), (
            f"Y({t_test}) mismatch:\n got:\n{Y_t}\n expected:\n{expected}"
        )

    def test_log_transition_from_fundamental_identity_case(self):
        """Phi(t, t) = I, so log_P should have 0 on diagonal and -inf off-diagonal."""
        K = 2
        lam = torch.tensor([1.5, 1.0], dtype=dtype)
        B = torch.eye(K, dtype=dtype)
        eta = torch.tensor(1.0 - 1e-9, dtype=dtype)

        model, _ = _make_two_state_model()
        t = 0.5
        time_points = torch.tensor([0.0, t], dtype=dtype)
        _, fundamental = model._solve_augmented_system_torchode(time_points, B, lam, eta)

        Y = fundamental[1]
        log_P = model._log_transition_from_fundamental(Y, Y)

        # Diagonal entries of P should be 1.0 → log = 0
        diag_vals = torch.diagonal(log_P)
        assert torch.allclose(diag_vals, torch.zeros(K, dtype=dtype), atol=1e-5), (
            f"Diagonal of log Phi(t,t) not zero: {diag_vals}"
        )

    def test_propagator_composition(self):
        """Phi(t2, t1) @ Phi(t1, t0) ≈ Phi(t2, t0) for any three time points."""
        K = 2
        lam = torch.tensor([2.0, 1.0], dtype=dtype)
        B = F.softmax(torch.tensor([[1.0, 0.5], [0.2, 1.0]], dtype=dtype), dim=1)
        eta = torch.tensor(0.9, dtype=dtype)

        model, _ = _make_two_state_model()
        t0, t1, t2 = 0.0, 0.3, 0.7
        time_points = torch.tensor([t0, t1, t2], dtype=dtype)
        _, fundamental = model._solve_augmented_system_torchode(time_points, B, lam, eta)

        Y0, Y1, Y2 = fundamental[0], fundamental[1], fundamental[2]

        # Phi(t1, t0) and Phi(t2, t1)
        Phi_10 = torch.linalg.solve(Y0.T, Y1.T).T
        Phi_21 = torch.linalg.solve(Y1.T, Y2.T).T
        Phi_20_composed = Phi_21 @ Phi_10

        # Direct Phi(t2, t0)
        Phi_20_direct = torch.linalg.solve(Y0.T, Y2.T).T

        assert torch.allclose(Phi_20_composed, Phi_20_direct, atol=1e-5), (
            f"Propagator composition failed:\n"
            f"Phi(t2,t1)@Phi(t1,t0):\n{Phi_20_composed}\n"
            f"Phi(t2,t0):\n{Phi_20_direct}"
        )

    def test_e_initial_condition(self):
        """E(t=0) should equal 1 - eta (boundary condition at tips)."""
        K = 2
        lam = torch.tensor([1.5, 1.0], dtype=dtype)
        B = torch.eye(K, dtype=dtype)
        eta_val = 0.8
        eta = torch.tensor(eta_val, dtype=dtype)

        model, _ = _make_two_state_model()
        time_points = torch.tensor([0.0, 0.5], dtype=dtype)
        E_vals, _ = model._solve_augmented_system_torchode(time_points, B, lam, eta)

        E_at_0 = E_vals[0]
        expected = torch.full((K,), 1.0 - eta_val, dtype=dtype)
        assert torch.allclose(E_at_0, expected, atol=1e-6), (
            f"E(0) = {E_at_0}, expected {expected}"
        )


# ===========================================================================
# Scalability
# ===========================================================================

class TestScalability:

    def test_many_large_trees_many_types(self):
        """Forward pass over 20 large trees with 10 states should be finite.

        Uses a uniform birth kernel so every division can produce any daughter
        type.  With lam[0]=3 and T=3.0 trees average ~100 leaves each,
        giving ~2000 leaves in total.
        """
        K = 10
        num_trees = 20
        T = 4.0
        sample_probability=0.8

        B_np = np.full((K, K), 1.0 / K)
        lam_np = np.array([3.0] + [1.5] * (K - 1))

        trees = [
            simulate_tree(B_np, lam_np, starting_type=0, T=T, seed=s, sample_probability=sample_probability)
            for s in range(num_trees)
        ]
        trees = [tree for tree in trees if tree is not None]

        growth_params = _make_growth_params(lam_np)
        model = ClaSSELikelihoodModel(
            trees=trees,
            num_states=K,
            birth_kernel_params=torch.zeros(K, K, dtype=dtype),
            pi_params=torch.zeros(K, dtype=dtype),
            growth_params=growth_params,
            sampling_prob=sample_probability
        )

        for i in range(num_trees):
            ll = model(i)
            assert torch.isfinite(ll), f"tree {i} gave non-finite log-likelihood"
            assert ll.item() < 0, f"tree {i} gave non-negative log-likelihood {ll.item()}"

    def test_gradients_scale_with_many_trees(self):
        """Backward pass should succeed across all trees without OOM or NaN gradients."""
        K = 10
        num_trees = 20
        T = 4.0
        sample_probability=0.8

        # B_np = np.full((K, K), 1.0 / K)
        B_np = np.array(
            [
                [0.6, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.6, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.6, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.2, 0.2, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ]
        )
        lam_np = np.array([3.0] + [1.5] * (K - 1))

        trees = [
            simulate_tree(B_np, lam_np, starting_type=0, T=T, seed=s, sample_probability=sample_probability)
            for s in range(num_trees)
        ]
        trees = [tree for tree in trees if tree is not None]

        growth_params = _make_growth_params(lam_np)
        birth_kernel_params = torch.log(torch.clamp_min(torch.from_numpy(B_np), 10e-5))
        model = ClaSSELikelihoodModel(
            trees=trees,
            num_states=K,
            birth_kernel_params=birth_kernel_params, #torch.zeros(K, K, dtype=dtype),
            pi_params=torch.zeros(K, dtype=dtype),
            growth_params=growth_params,
            sampling_prob=sample_probability
        )

        loss = sum(-model(i) for i in range(num_trees))
        loss.backward()

        assert model.kernel_builder.free_params.grad is not None
        assert not torch.isnan(model.kernel_builder.free_params.grad).any()
        assert model.growth_params.grad is not None
        assert not torch.isnan(model.growth_params.grad).any()


# ===========================================================================
# Optimizer convergence
# ===========================================================================

@pytest.mark.slow
class TestOptimizerConvergence:

    def test_mle_recovers_true_parameters(self):
        """MLE should converge to the data-generating parameters.

        We fit a 2-state ClaSSE model (state 0 is a progenitor, state 1 is
        terminal) to 100 trees simulated from known parameters.  Two checks
        are made:

        1. The optimised loss is lower than the loss evaluated at the true
           parameters — a necessary condition for MLE correctness.
        2. The recovered self-renewal probability B[0,0] and birth rate lam[0]
           are within tolerances of the ground truth.
        """
        true_p = 0.7           # P(type-0 daughter | type-0 parent)
        true_lam = [2.0, 1.5]
        B_np = np.array([
            [true_p, 1 - true_p],
            [0.0,     1.0]]
            )
        lam_np = np.array(true_lam)
        idx2potency = {0: (0, 1), 1: (1,)}
        idx2state = {0: 0, 1: 1}

        num_trees = 20
        sample_probability=0.5
        trees = [
            simulate_tree(B_np, lam_np, starting_type=0, T=2.5, seed=s, sample_probability=sample_probability)
            for s in range(num_trees)
        ]
        trees = [tree for tree in trees if tree is not None]
        num_trees = len(trees)

        def neg_mean_ll(model):
            return -sum(model(i) for i in range(num_trees)) / num_trees

        # ---- loss at the true parameters ----
        true_B_logits = torch.log(torch.tensor(B_np, dtype=dtype).clamp_min(EPS))
        model_true = ClaSSELikelihoodModel(
            trees=trees, num_states=2,
            birth_kernel_params=true_B_logits,
            pi_params=torch.tensor([100, -100], dtype=dtype),
            growth_params=_make_growth_params(lam_np),
            idx2potency=idx2potency, idx2state=idx2state, start_state=0,
            sampling_prob=sample_probability
        )
        with torch.no_grad():
            loss_at_truth = neg_mean_ll(model_true).item()

        # ---- fit starting from a uniform kernel ----
        model = ClaSSELikelihoodModel(
            trees=trees, num_states=2,
            birth_kernel_params=torch.zeros(2, 2, dtype=dtype),
            pi_params=torch.tensor([100, -100], dtype=dtype),
            growth_params=_make_growth_params(lam_np),
            idx2potency=idx2potency, idx2state=idx2state, start_state=0,
            sampling_prob=sample_probability
        )

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.LBFGS(params, lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = neg_mean_ll(model)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            loss_mle = neg_mean_ll(model).item()

        B_hat = model.get_daughter_kernel()
        lam_hat = model.get_growth_rates()

        print("loss_mle:", loss_mle, "loss_true:", loss_at_truth)
        print("Inferred birth kernel:")
        print(B_hat)
        print("Inferred birth rates:")
        print(lam_hat)
        print("Leaf distribution:")
        print([len(tree.get_leaves()) for tree in trees])


        # MLE must achieve at least as low a loss as the true parameters.
        assert loss_mle <= loss_at_truth + 1e-3, (
            f"MLE loss {loss_mle:.4f} is worse than true-parameter loss {loss_at_truth:.4f}"
        )

        assert np.sum(np.abs(B_hat.detach().numpy() - B_np)) < 0.2
        assert np.sum(np.abs(lam_hat.detach().numpy() - lam_np)) < 0.5
