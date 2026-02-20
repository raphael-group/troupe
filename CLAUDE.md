# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The package (`troupe`) is installed in editable mode from `src/` via `setup.py`.

## Running Tests

```bash
# Run all fast tests
pytest tests/

# Skip slow tests (MLE convergence)
pytest -m "not slow" tests/

# Run slow tests only
pytest -m slow tests/

# Run a single test
pytest tests/test_models.py::test_ctmc_model_forward_finite
```

All tests must be self-contained — no pickled files or hardcoded paths. Use `simulate_tree` from `src/branching_simulation.py` to generate data programmatically. Tag MLE convergence tests with `@pytest.mark.slow`.

## Running the Full Pipeline

```bash
# Unified pipeline (preferred — handles both phases + model selection)
python scripts/run_troupe.py \
    -i example/data/trees.pkl \
    -o example/results \
    --regularizations 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10

# Legacy multi-script pipeline (see troupe_inference_example.sh)
# Note: working_dir in troupe_inference_example.sh must be updated to your local path
```

## Architecture

### Core Algorithm (TROUPE)

TROUPE infers cell differentiation dynamics from lineage trees by fitting a Continuous Time Markov Chain (CTMC) model with per-state birth rates and potency constraints.

**Two-phase inference:**
1. **Phase 1** — Overparameterized MLE with L1 regularization. Runs at multiple regularization strengths to produce a range of sparse solutions.
2. **Phase 2** — Potency extraction from Phase 1 model → reduced state space → debiased re-fit with warm-start initialization.
3. **Model selection** — Knee detection on neg-log-likelihood vs. number of reachable states (via `kneed.KneeLocator`).

The unified pipeline is in `scripts/run_troupe.py`. The legacy pipeline uses individual scripts (`scripts/save_potency_sets.py`, `scripts/infer_model.py`, etc.).

### Source Modules (`src/`)

**`models.py`** — PyTorch model classes (primary implementation):
- `RateMatrixBuilder`: Learns a CTMC rate matrix Q via softplus-parameterized free parameters. Potency constraints are enforced via a boolean mask so forbidden transitions are exactly zero.
- `TreeTensorizer`: Converts ete3 trees into tensor representations (postorder indices, children, branch lengths, leaf partials, levels, absolute times) for batched GPU-compatible computation.
- `FelsensteinPruner`: Felsenstein's pruning algorithm in log-space, batched by tree level. Uses `torch.matrix_exp` for transition matrices.
- `CTMCLikelihoodModel` / `PureBirthLikelihoodModel`: Compose `RateMatrixBuilder` + `FelsensteinPruner` into end-to-end differentiable likelihood models. `PureBirthLikelihoodModel` adds per-type growth rates and potency constraints.

**`likelihood.py`** — Legacy Felsenstein implementation (traversal-based, node attributes on ete3 objects). Kept for cross-validation against `FelsensteinPruner`. Both implementations must agree — this is tested in `tests/test_likelihood_agreement.py`.

**`optimizer.py`** — LBFGS optimization loop (`compute_mle`). Initializes rates via constant-rate MLE, trains `PureBirthLikelihoodModel`, saves best checkpoint (`model_dict.pkl`) when loss improves.

**`branching_simulation.py`** — Simulates multi-type pure-birth branching processes (Gillespie-style). Used to generate test data.

**`utils.py`** — File I/O helpers (`get_terminal_labels`, `get_observed_potencies`), tree utilities (`binarize_tree`, `is_ultrametric`), and potency inference from rate matrices (`get_idx2potency`, `get_reachable_idxs`).

### Key Data Conventions

- **Trees**: ete3 `TreeNode` objects. Leaves must have a `.state` attribute (integer index). Trees must be binary (use `binarize_tree`) and ultrametric with a unifurcating root (added automatically if missing).
- **Potency sets**: A dict `idx2potency` mapping each state index to a tuple of terminal state indices the state can eventually differentiate into. Observed (terminal) states map to `(self,)`. Hidden states map to multi-element tuples.
- **Saved models**: `model_dict.pkl` contains `rate_matrix`, `root_distribution`, `growth_rates`, `idx2state`, `idx2potency`.
- **Float64**: All tensors use `torch.float64`. Set globally via `torch.set_default_dtype(torch.float64)`.
- **Numerical stability**: `EPS = 1e-30` used as log-space zero; softplus with `threshold=10` for parameter transformations.
