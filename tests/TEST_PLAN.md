# Test Suite Rework Plan

## Overview

The current test suite is largely broken. Of 7 test files, only `test_branching_likelihood.py` passes; the rest import modules that were removed during a codebase refactor (`simulation`, `OLD_models`, `gillespie`, `OLD_branching_simulation`, `GeneralEmissionModel`, `PotencyLikelihoodModel`). Several also depend on hardcoded cluster paths. This plan replaces the broken suite with a portable, self-contained test suite organized around the current module API.

---

## Current State

| File | Status | Issues |
|------|--------|--------|
| `test_branching_likelihood.py` | Working | Tests agreement between `log_vec_likelihood` and `PureBirthLikelihoodModel` |
| `test_overparameterized_likelihood.py` | Broken | Hardcoded path to cluster-specific pickled experiment data |
| `test_likelihood.py` | Broken | Imports `simulation.simulate_tree_state`, `OLD_models.LogLikelihoodModel` |
| `test_branching_simulation.py` | Broken | Imports `gillespie.simulate_tree_gillespie`, `OLD_branching_simulation`; uses `main()` instead of pytest |
| `test_optimization.py` | Broken | Imports `simulation`, `OLD_models`; hardcoded `/n/fs/ragr-research/...` paths |
| `test_potency_model.py` | Broken | Imports `PotencyLikelihoodModel` (removed class); hardcoded paths |
| `test_training.py` | Broken | Exact duplicate of `test_optimization.py` |
| `likelihood_dojo.py` | N/A | Benchmarking script, not a test file |

**Files to delete:** `test_overparameterized_likelihood.py`, `test_potency_model.py`, `test_training.py`, `likelihood_dojo.py`.

**Files to replace:** `test_likelihood.py`, `test_branching_simulation.py`, `test_optimization.py`.

**File to keep as-is:** `test_branching_likelihood.py` (working; eventually superseded by `test_likelihood_agreement.py`).

---

## Proposed Test Files

### 1. `conftest.py` — Shared Fixtures

Centralizes reusable test infrastructure so individual test files stay focused.

**Fixtures:**

- `device` — Returns `torch.device("cuda" if available else "cpu")`.
- `two_state_system` — Returns `(Q, lam, pi_params)` for a 2-state non-reversible CTMC:
  ```python
  Q = np.array([[-1.0, 1.0],
                [0.0,  0.0]])
  lam = np.array([1.0, 1.0])
  pi_params = torch.tensor([1e20, -1e20])  # Deterministic start in state 0
  ```
- `three_state_system` — Returns `(Q, lam, pi_params)` for a 3-state non-reversible CTMC:
  ```python
  Q = np.array([[-2.0, 1.0, 1.0],
                [ 0.0,-1.0, 1.0],
                [ 0.0, 0.0, 0.0]])
  lam = np.array([1.0, 1.0, 1.0])
  pi_params = torch.tensor([1e20, -1e20, -1e20])
  ```
- `seven_state_system` — Returns `(Q, lam, pi_params)` for the 7-state system with hidden states (matches the existing `test_big_non_reversible` from `test_branching_likelihood.py`).
- `simulate_trees(Q, lam, num_trees, T, num_states)` — Factory fixture that simulates trees and preps them with `_prep_log_tree`.
- `make_Q_params(Q)` — Helper: reverse softplus `torch.log(torch.exp(Q_tensor) - 1)` with diagonal zeroed.
- `make_growth_params(lam)` — Helper: reverse softplus for growth rates.
- `tmp_output_dir(tmp_path)` — Returns a temporary directory for optimizer output.

**Why a shared conftest?** Every test file needs simulated trees and CTMC parameters. Duplicating this setup (as the current `test_branching_likelihood.py` does inline) is fragile and verbose.

---

### 2. `test_rate_matrix_builder.py` — `RateMatrixBuilder` Unit Tests

**Source under test:** `src/models.py :: RateMatrixBuilder`

**Tests:**

| Test | What it verifies |
|------|-----------------|
| `test_rows_sum_to_zero` | Each row of `Q = builder.forward()` sums to zero (within float64 tolerance). |
| `test_off_diagonal_nonneg` | All off-diagonal entries of Q are ≥ 0. |
| `test_output_shape` | Q is `(num_states, num_states)`. |
| `test_potency_constraints_mask_zeros` | When `idx2potency` forbids a transition, the corresponding Q entry is exactly 0. |
| `test_potency_constraints_allow` | When `idx2potency` allows a transition, the corresponding Q entry is > 0 (given positive params). |
| `test_hidden_states` | With `num_hidden_states > 0`, Q has shape `(num_states + num_hidden, ...)` and hidden→terminal transitions are allowed. |
| `test_gradient_flows` | `Q.sum().backward()` succeeds and `builder.free_params.grad` is not None. |
| `test_init_Q_params` | Passing `init_Q_params` sets the initial free parameters correctly. |

---

### 3. `test_tree_tensorizer.py` — `TreeTensorizer` Unit Tests

**Source under test:** `src/models.py :: TreeTensorizer`

**Tests:**

| Test | What it verifies |
|------|-----------------|
| `test_postorder_length` | Postorder tensor has one entry per node in the tree. |
| `test_leaf_partials_one_hot` | Leaf partial vectors are one-hot for the leaf's state. |
| `test_internal_partials_zero` | Internal node partial vectors are all zeros (to be filled by pruning). |
| `test_branch_lengths_positive` | All branch lengths ≥ 0; root branch length is 0. |
| `test_levels_monotonic` | Level indices are valid and levels proceed from leaves toward root. |
| `test_multiple_trees` | Tensorizer correctly handles a list of multiple trees (separate postorders, shared `global_time_grid`). |
| `test_ultrametric_abs_times` | For ultrametric trees, `abs_times` at leaves are all equal (≈ T). |

---

### 4. `test_felsenstein_pruner.py` — `FelsensteinPruner` Unit Tests

**Source under test:** `src/models.py :: FelsensteinPruner`

**Tests:**

| Test | What it verifies |
|------|-----------------|
| `test_single_leaf_tree` | A single-leaf tree returns log P(root = leaf_state). |
| `test_two_leaf_cherry` | A cherry (two leaves, one internal node) gives the correct hand-computable likelihood. |
| `test_log_likelihood_finite` | Output is a finite scalar for a simulated tree. |
| `test_gradient_flows` | Backprop through the pruner produces gradients on Q params. |
| `test_growth_rates_affect_likelihood` | Changing growth rates changes the returned likelihood. |
| `test_subsampling_rho` | Setting `rho < 1` changes the likelihood compared to `rho = 1`. |

---

### 5. `test_branching_simulation.py` — Simulation Tests (replaces broken file)

**Source under test:** `src/branching_simulation.py :: simulate_tree, edge_process, branching_process`

**Tests:**

| Test | What it verifies |
|------|-----------------|
| `test_tree_is_ultrametric` | `is_ultrametric(tree)` returns True for every simulated tree. |
| `test_tree_has_leaves_with_states` | Every leaf has a `.state` attribute that is a valid state index. |
| `test_seed_reproducibility` | Same seed produces identical trees. |
| `test_different_seeds_differ` | Different seeds produce different trees (with high probability). |
| `test_expected_leaf_count` | Over many trials, mean leaf count ≈ `exp(growth_rate * T)` (within 3σ). |
| `test_edge_process_stays_in_state` | With Q = 0 (no transitions), `edge_process` returns `(start_type, T)`. |
| `test_edge_process_type_changes` | With high transition rate, `edge_process` eventually returns a different type. |
| `test_terminal_states_absorbing` | States with all-zero transition rows never produce children of a different type. |
| `test_three_state_leaf_proportions` | For a 3-state system over many trees, leaf type proportions are approximately consistent with the CTMC rates (statistical test, generous tolerance). |

---

### 6. `test_likelihood.py` — Likelihood Function Tests (replaces broken file)

**Source under test:** `src/likelihood.py :: log_vec_likelihood, log_vectorized_felsenstein_pruning, _prep_log_tree, compute_transition_tensor`

**Tests:**

| Test | What it verifies |
|------|-----------------|
| `test_prep_log_tree_adds_partials` | `_prep_log_tree` adds `log_partials` to every node. |
| `test_prep_log_tree_adds_root` | If the root is not unifurcating, a unifurcating root is added. |
| `test_prep_log_tree_leaf_partials` | Leaf log-partials are 0 for the correct state and -inf elsewhere. |
| `test_transition_tensor_identity_at_zero` | `compute_transition_tensor([0.0], Q)` ≈ identity matrix. |
| `test_transition_tensor_rows_sum_to_one` | Each row of `exp(Qt)` sums to 1. |
| `test_transition_tensor_with_growth_rates` | Growth rates shift Q by `Q - diag(lam)` before exponentiation. |
| `test_log_vec_likelihood_finite` | Returns a finite negative scalar for simulated data. |
| `test_log_vec_likelihood_increases_at_truth` | Likelihood at the true Q is higher than at a perturbed Q. |
| `test_single_tree_matches_batch` | `log_vec_likelihood([tree], Q, pi)` equals `log_vectorized_felsenstein_pruning(tree, ...)`. |

---

### 7. `test_likelihood_agreement.py` — Cross-Validation (MOST CRITICAL)

**Source under test:** `src/likelihood.py` vs `src/models.py :: PureBirthLikelihoodModel`

This is the most important test file. It verifies that the two independent Felsenstein implementations (`log_vec_likelihood` in `likelihood.py` and `FelsensteinPruner` inside `PureBirthLikelihoodModel` in `models.py`) produce the same log-likelihood values. Disagreement here indicates a bug in one or both implementations.

**Tests:**

| Test | What it verifies |
|------|-----------------|
| `test_two_state_agreement` | 2-state system, 5 trees: both methods agree to 6 significant figures. |
| `test_three_state_agreement` | 3-state system, 10 trees: both methods agree. |
| `test_seven_state_with_hidden_agreement` | 7-state system with hidden states, 32 trees: both methods agree. |
| `test_agreement_with_subsampling` | Both methods agree when `rho < 1`. |
| `test_agreement_with_potency_constraints` | Both methods agree when `idx2potency` is set on `PureBirthLikelihoodModel`. |
| `test_gradient_agreement` | Gradients of log-likelihood w.r.t. Q params agree between methods (within tolerance). |

**Implementation note:** This file absorbs the role of `test_branching_likelihood.py`. The `compare_methods` helper and `are_all_within_k_sig_figs` utility move here. Once this file is in place, `test_branching_likelihood.py` can be deleted.

---

### 8. `test_models.py` — Model Integration Tests

**Source under test:** `src/models.py :: CTMCLikelihoodModel, PureBirthLikelihoodModel`

**Tests:**

| Test | What it verifies |
|------|-----------------|
| `test_ctmc_model_forward_finite` | `model(tree_idx)` returns a finite scalar. |
| `test_ctmc_model_get_rate_matrix` | `get_rate_matrix()` returns a valid Q matrix (rows sum to 0, off-diag ≥ 0). |
| `test_ctmc_model_get_root_distribution` | `get_root_distribution()` sums to 1. |
| `test_pure_birth_requires_growth` | `PureBirthLikelihoodModel` raises an error or handles gracefully when growth params are missing. |
| `test_pure_birth_growth_rates_positive` | `get_growth_rates()` returns all-positive values (softplus parameterization). |
| `test_start_state_fixing` | When `start_state` is set, root distribution puts all mass on that state. |
| `test_potency_constraints_applied` | When `idx2potency` is passed, rate matrix zeros match the constraint mask. |
| `test_hidden_states_expand_matrix` | With `num_hidden > 0`, rate matrix has shape `(num_states + num_hidden, ...)`. |
| `test_model_parameters_optimizable` | All model parameters have `requires_grad=True` and gradients flow through `forward()`. |

---

### 9. `test_optimizer.py` — Optimization Tests (replaces broken file)

**Source under test:** `src/optimizer.py :: compute_mle, loss_fn, sparse_regularization, group_lasso`

**Tests:**

| Test | What it verifies |
|------|-----------------|
| `test_loss_fn_negative_llh` | `loss_fn(-llh)` equals `llh` when regularization is 0. |
| `test_sparse_regularization_zero_at_zero` | `sparse_regularization` returns 0 when off-diagonal Q entries are 0. |
| `test_sparse_regularization_positive` | `sparse_regularization` returns a positive value for non-zero Q. |
| `test_group_lasso_positive` | `group_lasso` returns a positive value for non-zero Q. |
| `test_mle_recovers_two_state` | `compute_mle` on 50+ trees from a 2-state system recovers Q within tolerance (slow, mark `@pytest.mark.slow`). |
| `test_mle_with_l1_produces_sparse_Q` | L1 regularization drives some off-diagonal entries toward 0. |
| `test_mle_saves_checkpoint` | `compute_mle` writes `model_dict.pkl` to the output directory. |

---

### 10. `test_utils.py` — Utility Function Tests

**Source under test:** `src/utils.py :: is_ultrametric, get_idx2potency, binarize_tree, get_terminal_labels, get_observed_potencies`

**Tests:**

| Test | What it verifies |
|------|-----------------|
| `test_is_ultrametric_true` | Returns True for a simulated ultrametric tree. |
| `test_is_ultrametric_false` | Returns False for a manually constructed non-ultrametric tree. |
| `test_binarize_tree_all_binary` | After binarization, every internal node has exactly 2 children. |
| `test_binarize_tree_preserves_leaves` | Leaf count and leaf states are unchanged after binarization. |
| `test_get_idx2potency_identity` | For a diagonal Q (no transitions), each state's potency set is `{state}`. |
| `test_get_idx2potency_transitive` | If 0→1 and 1→2, state 0's potency includes {0, 1, 2}. |
| `test_get_terminal_labels` | Reads a file with one label per line, returns the correct list. |
| `test_get_observed_potencies` | Reads a TSV file, returns the correct dict. |

---

## Implementation Order

The test files should be implemented in dependency order:

1. **`conftest.py`** — Everything else depends on shared fixtures.
2. **`test_utils.py`** — Pure functions, no model dependencies.
3. **`test_branching_simulation.py`** — Simulation is upstream of all likelihood tests.
4. **`test_rate_matrix_builder.py`** — Isolated unit, foundation for model tests.
5. **`test_tree_tensorizer.py`** — Isolated unit, foundation for model tests.
6. **`test_likelihood.py`** — Tests the `likelihood.py` functions.
7. **`test_felsenstein_pruner.py`** — Tests the `models.py` pruner.
8. **`test_likelihood_agreement.py`** — Cross-validates the two implementations.
9. **`test_models.py`** — Integration tests for full model classes.
10. **`test_optimizer.py`** — End-to-end optimization (slowest tests, run last).

## Cleanup

After the new suite is in place, delete:

- `test_branching_likelihood.py` (superseded by `test_likelihood_agreement.py`)
- `test_overparameterized_likelihood.py` (cluster-dependent, not portable)
- `test_potency_model.py` (tests a removed class)
- `test_training.py` (duplicate of `test_optimization.py`)
- `likelihood_dojo.py` (benchmark script, not a test)

## Conventions

- **Float64 everywhere:** All tensors use `torch.float64` to match the codebase.
- **Tolerance:** Use `torch.allclose` with `atol=1e-8` for numerical comparisons; use `are_all_within_k_sig_figs(values, k=6)` for likelihood agreement.
- **Deterministic seeds:** All simulations use fixed seeds for reproducibility.
- **No external data:** Every test generates its own data via `simulate_tree`. No pickled files, no hardcoded paths.
- **Pytest markers:** Tag slow tests (MLE convergence) with `@pytest.mark.slow` so `pytest` runs fast by default and `pytest -m slow` includes them.
