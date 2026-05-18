"""Constrained MLE for ClaSSE via quadratic penalty method.

Provides:
  compute_unconstrained_mle  — standard ClaSSE LBFGS fit
  compute_constrained_mle    — fit with extra linear inequality constraints on B
  compute_log_likelihood     — evaluate total log-likelihood for a fitted model
"""

import logging
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from classe_model import ClaSSELikelihoodModel
from constraints import KernelConstraint
from optimizer import constant_rate_mle

logger = logging.getLogger(__name__)

EPS = 1e-30
dtype = torch.float64
torch.set_default_dtype(dtype)


# ---------------------------------------------------------------------------
# Helpers shared by both optimizers
# ---------------------------------------------------------------------------

def _safe_softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=EPS)
    return torch.log(torch.expm1(x))


def _uniform_kernel_logits(idx2potency, n_states, device):
    logits = torch.full((n_states, n_states), -1e30, device=device, dtype=dtype)
    if idx2potency is None:
        logits[:] = 0.0
        return logits
    for i, potency in idx2potency.items():
        allowed = [
            j for j, child_potency in idx2potency.items()
            if all(t in potency for t in child_potency)
        ]
        logits[i, allowed] = 0.0
    return logits


def _build_llh(trees, n_states, model_info, sampling_prob, device,
               bk_params_init, growth_params_init):
    """Construct a ClaSSELikelihoodModel with given initialization.

    Note: sampling_prob must be strictly less than 1. When sampling_prob == 1
    the E ODE has a zero initial RHS (E_0 = 0) which causes the torchode
    step-size controller to divide by zero during the backward pass.
    """
    if float(sampling_prob) >= 1.0:
        raise ValueError(
            "sampling_prob must be strictly less than 1 for ClaSSE (got "
            f"{sampling_prob}). The ClaSSE ODE requires E_0 = 1 - eta > 0 "
            "for well-defined gradients."
        )
    idx2potency = model_info["idx2potency"]
    idx2state = model_info["idx2state"]
    start_state = model_info.get("start_state")
    optimize_growth = model_info.get("optimize_growth", True)
    integration_max_step = model_info.get("integration_max_step", 0.05)
    ode_method = model_info.get("ode_method", "Dopri5")
    ode_atol = model_info.get("ode_atol", 1e-8)
    ode_rtol = model_info.get("ode_rtol", 1e-6)
    backend = model_info.get("backend", "fundamental")

    n_hidden = sum(1 for pot in idx2potency.values() if len(pot) > 1)

    pi_params_init = torch.zeros(n_states, device=device, dtype=dtype)

    return ClaSSELikelihoodModel(
        trees,
        n_states,
        bk_params_init,
        pi_params_init,
        growth_params_init,
        optimize_growth=optimize_growth,
        num_hidden=n_hidden,
        idx2potency=idx2potency,
        device=device,
        idx2state=idx2state,
        start_state=start_state,
        sampling_prob=float(sampling_prob),
        integration_max_step=integration_max_step,
        ode_method=ode_method,
        ode_atol=ode_atol,
        ode_rtol=ode_rtol,
        backend=backend,
    )


def compute_log_likelihood(llh: ClaSSELikelihoodModel, trees: list) -> float:
    """Compute total log-likelihood (sum over trees) for a fitted model.

    Args:
        llh: Fitted ClaSSELikelihoodModel.
        trees: List of trees (must match those the model was built on).

    Returns:
        Total log-likelihood as a float.
    """
    llh.precompute_ode()
    with torch.no_grad():
        ll = sum(llh(j).item() for j in range(len(trees)))
    llh.clear_ode_cache()
    return ll


# ---------------------------------------------------------------------------
# Constraint penalty
# ---------------------------------------------------------------------------

def constraint_penalty(llh: ClaSSELikelihoodModel,
                        constraints: list,
                        mu: float) -> torch.Tensor:
    """Quadratic penalty for violated birth kernel constraints.

    Operates on the softmax output B (not the logits), so gradients flow
    through softmax → free_params automatically via autograd.

    Args:
        llh: ClaSSELikelihoodModel instance.
        constraints: List of KernelConstraint objects.
        mu: Penalty weight (mu/2 * sum_c max(0, g_c)^2).

    Returns:
        Scalar tensor (differentiable w.r.t. kernel free_params).
    """
    B = llh.get_daughter_kernel()   # softmax output — fully differentiable
    penalty = B.new_zeros(())
    for c in constraints:
        lhs = c.lhs_coeff * B[c.lhs_i, c.lhs_j]
        if c.rhs_i >= 0:
            rhs = c.rhs_coeff * B[c.rhs_i, c.rhs_j]
        else:
            rhs = B.new_zeros(())
        violation = rhs + c.offset - lhs   # positive when violated
        penalty = penalty + torch.clamp(violation, min=0.0) ** 2
    return (mu / 2.0) * penalty


def constraints_satisfied(llh: ClaSSELikelihoodModel,
                           constraints: list,
                           tol: float = 1e-5) -> bool:
    """Return True if all constraints hold to within tol."""
    with torch.no_grad():
        B = llh.get_daughter_kernel()
        return all(c.is_satisfied(B, tol=tol) for c in constraints)


# ---------------------------------------------------------------------------
# Unconstrained MLE
# ---------------------------------------------------------------------------

def compute_unconstrained_mle(
    trees: list,
    n_states: int,
    model_info: dict,
    sampling_prob: float,
    device,
    output_dir: str = None,
    num_iter: int = 100,
    do_logging: bool = False,
) -> tuple:
    """Fit ClaSSELikelihoodModel via LBFGS without additional constraints.

    Args:
        trees: Binarized ete3 trees with int-castable leaf states.
        n_states: Total number of states.
        model_info: Dict with idx2potency, idx2state, and optional
            B_params_init, growth_params_init, start_state, etc.
        sampling_prob: Leaf sampling probability eta in (0, 1].
        device: Torch device.
        output_dir: If provided, save model_dict.pkl and state_dict.pth here.
        num_iter: Maximum LBFGS outer iterations.
        do_logging: Whether to log progress.

    Returns:
        (llh, log_likelihood) where log_likelihood is the total LL summed
        over all trees (not the averaged NLL used during training).
    """
    idx2potency = model_info["idx2potency"]

    if "B_params_init" in model_info:
        bk_params_init = model_info["B_params_init"].to(device=device, dtype=dtype)
    else:
        bk_params_init = _uniform_kernel_logits(idx2potency, n_states, device)

    if "growth_params_init" in model_info:
        growth_params_init = model_info["growth_params_init"].to(device=device, dtype=dtype)
    else:
        lam0 = constant_rate_mle(trees)
        growth_params_init = _safe_softplus_inverse(
            torch.ones(n_states, device=device, dtype=dtype) * lam0
        )

    llh = _build_llh(trees, n_states, model_info, sampling_prob, device,
                     bk_params_init, growth_params_init)

    llh = _run_lbfgs(
        trees, llh, constraints=[], mu=0.0,
        num_iter=num_iter, do_logging=do_logging,
        output_dir=output_dir, model_info=model_info, sampling_prob=sampling_prob,
    )

    ll = compute_log_likelihood(llh, trees)
    return llh, ll


# ---------------------------------------------------------------------------
# Constrained MLE
# ---------------------------------------------------------------------------

def compute_constrained_mle(
    trees: list,
    n_states: int,
    model_info: dict,
    sampling_prob: float,
    constraints: list,
    device,
    warm_start_llh: ClaSSELikelihoodModel = None,
    penalty_weight_schedule: tuple = (1.0, 10.0, 100.0, 1000.0),
    constraint_tol: float = 1e-5,
    num_iter_per_phase: int = 50,
    output_dir: str = None,
    do_logging: bool = False,
) -> tuple:
    """Fit ClaSSE under additional linear inequality constraints via quadratic penalty.

    The potency support mask is always respected (it is encoded in the model's
    DaughterKernelBuilder). The ``constraints`` list adds *extra* linear
    inequalities on top.

    Args:
        trees: Binarized ete3 trees.
        n_states: Total number of states.
        model_info: Same dict used for the unconstrained model.
        sampling_prob: Leaf sampling probability.
        constraints: List of KernelConstraint objects.
        device: Torch device.
        warm_start_llh: If provided, use its parameters as initialization.
            Passing the unconstrained MLE here is strongly recommended.
        penalty_weight_schedule: Sequence of mu values for the outer penalty loop.
        constraint_tol: Tolerance for constraint satisfaction check.
        num_iter_per_phase: LBFGS iterations per penalty phase.
        output_dir: If provided, save final model_dict.pkl here.
        do_logging: Whether to log progress.

    Returns:
        (llh, log_likelihood) where log_likelihood is the total LL summed
        over all trees under the fitted null model.
    """
    idx2potency = model_info["idx2potency"]

    # Validate constraints against potency mask before running anything.
    if warm_start_llh is not None:
        mask = warm_start_llh.kernel_builder.support_mask
        for c in constraints:
            c.validate(mask)

    # Initialize from warm_start_llh if provided, else use model_info defaults.
    if warm_start_llh is not None:
        bk_params_init = warm_start_llh.kernel_builder.free_params.detach().clone()
        growth_params_init = warm_start_llh.growth_params.detach().clone()
    else:
        if "B_params_init" in model_info:
            bk_params_init = model_info["B_params_init"].to(device=device, dtype=dtype)
        else:
            bk_params_init = _uniform_kernel_logits(idx2potency, n_states, device)
        if "growth_params_init" in model_info:
            growth_params_init = model_info["growth_params_init"].to(device=device, dtype=dtype)
        else:
            lam0 = constant_rate_mle(trees)
            growth_params_init = _safe_softplus_inverse(
                torch.ones(n_states, device=device, dtype=dtype) * lam0
            )

    llh = _build_llh(trees, n_states, model_info, sampling_prob, device,
                     bk_params_init, growth_params_init)

    # Outer penalty loop: increase mu until all constraints are satisfied.
    for phase_idx, mu in enumerate(penalty_weight_schedule):
        if do_logging:
            logger.info("Constrained MLE phase %d / %d  (mu=%.1f)",
                        phase_idx + 1, len(penalty_weight_schedule), mu)

        llh = _run_lbfgs(
            trees, llh, constraints=constraints, mu=mu,
            num_iter=num_iter_per_phase, do_logging=do_logging,
            output_dir=None, model_info=None, sampling_prob=sampling_prob,
        )

        if constraints_satisfied(llh, constraints, tol=constraint_tol):
            if do_logging:
                logger.info("All constraints satisfied after phase %d (mu=%.1f).",
                            phase_idx + 1, mu)
            break

        # Warm-start next phase from current solution.
        bk_params_next = llh.kernel_builder.free_params.detach().clone()
        growth_params_next = llh.growth_params.detach().clone()
        llh = _build_llh(trees, n_states, model_info, sampling_prob, device,
                         bk_params_next, growth_params_next)
    else:
        if not constraints_satisfied(llh, constraints, tol=constraint_tol):
            logger.warning(
                "Constrained MLE did not fully satisfy all constraints after the "
                "final penalty phase. Consider increasing penalty_weight_schedule."
            )

    if output_dir is not None:
        _save_model_dict(llh, model_info, sampling_prob, output_dir)

    ll = compute_log_likelihood(llh, trees)
    return llh, ll


# ---------------------------------------------------------------------------
# Internal LBFGS loop (shared by constrained and unconstrained)
# ---------------------------------------------------------------------------

def _run_lbfgs(
    trees: list,
    llh: ClaSSELikelihoodModel,
    constraints: list,
    mu: float,
    num_iter: int,
    do_logging: bool,
    output_dir: str,
    model_info: dict,
    sampling_prob: float,
) -> ClaSSELikelihoodModel:
    """Run one LBFGS phase on a pre-built ClaSSELikelihoodModel.

    Args:
        trees: Training trees.
        llh: Model to optimize (modified in place via LBFGS).
        constraints: List of KernelConstraint (empty list = unconstrained).
        mu: Quadratic penalty weight for violated constraints.
        num_iter: Maximum LBFGS outer iterations.
        do_logging: Log progress every iteration.
        output_dir: If not None, save best checkpoint when loss improves.
        model_info: Needed for saving model_dict; can be None if output_dir is None.
        sampling_prob: Needed for saving model_dict.

    Returns:
        The llh with best parameters (loaded from checkpoint if output_dir given).
    """
    tree_idxs = list(range(len(trees)))
    rel_loss_thresh = 1e-5 / len(trees)
    EPS_conv = 1e-30

    changeable_params = [p for p in llh.parameters() if p.requires_grad]
    optimizer = optim.LBFGS(changeable_params, lr=0.1, line_search_fn="strong_wolfe")

    losses = []
    closure_state = {"last_loss": None}

    _nan_sentinel = torch.tensor(1e8, dtype=dtype, device=llh.device)

    def closure():
        optimizer.zero_grad()
        try:
            llh.precompute_ode()
            nll = sum(-llh(j) for j in tree_idxs) / len(tree_idxs)
            if constraints and mu > 0:
                objective = nll + constraint_penalty(llh, constraints, mu)
            else:
                objective = nll
            if not torch.isfinite(objective):
                llh.clear_ode_cache()
                # Return a high finite value with zero gradients so LBFGS backtracks.
                (_nan_sentinel * 0 + sum(p.sum() * 0 for p in changeable_params) + 1e8).backward()
                return _nan_sentinel
            objective.backward()
            closure_state["last_loss"] = float(objective.detach().item())
            llh.clear_ode_cache()
            return objective
        except Exception:
            llh.clear_ode_cache()
            (_nan_sentinel * 0 + sum(p.sum() * 0 for p in changeable_params) + 1e8).backward()
            return _nan_sentinel

    start = time.time()
    for i in range(num_iter):
        optimizer.step(closure)
        if closure_state["last_loss"] is None:
            raise RuntimeError("LBFGS step did not evaluate the objective.")
        loss_value = closure_state["last_loss"]

        with torch.no_grad():
            losses.append(loss_value)
            if torch.isnan(llh.get_daughter_kernel()).any():
                raise ValueError(f"NaN in daughter kernel at iteration {i}.")

        if output_dir is not None and losses[-1] <= min(losses):
            os.makedirs(output_dir, exist_ok=True)
            torch.save(llh.state_dict(), f"{output_dir}/state_dict.pth")
            if model_info is not None:
                _save_model_dict(llh, model_info, sampling_prob, output_dir)

        if do_logging:
            logger.info("Iter %d  loss=%.6f", i, loss_value)
            logger.info("  B diag: %s", llh.get_daughter_kernel().diag().detach().tolist())
            logger.info("  lam:    %s", llh.get_growth_rates().detach().tolist())
            if constraints:
                with torch.no_grad():
                    B = llh.get_daughter_kernel()
                for c in constraints:
                    lhs_val = c.lhs_coeff * float(B[c.lhs_i, c.lhs_j].item())
                    rhs_val = (c.rhs_coeff * float(B[c.rhs_i, c.rhs_j].item())
                               if c.rhs_i >= 0 else 0.0)
                    logger.info("  %s: lhs=%.4f rhs=%.4f satisfied=%s",
                                c.label, lhs_val, rhs_val + c.offset,
                                lhs_val >= rhs_val + c.offset - 1e-5)
            logger.info("  %.4f s/iter", (time.time() - start))
            start = time.time()

        if len(losses) > 2:
            rel = abs(losses[-1] - losses[-2]) / (abs(losses[-2]) + EPS_conv)
            if rel <= rel_loss_thresh:
                if do_logging:
                    logger.info("Converged at iteration %d (rel_loss=%.2e).", i, rel)
                break

    # Restore best checkpoint if one was saved.
    if output_dir is not None:
        best_path = f"{output_dir}/state_dict.pth"
        if os.path.isfile(best_path):
            llh.load_state_dict(torch.load(best_path, map_location=llh.device))

    return llh


def _save_model_dict(llh, model_info, sampling_prob, output_dir):
    import pickle
    B = llh.get_daughter_kernel()
    model_dict = {
        "rate_matrix": B,
        "daughter_kernel": B,
        "growth_rates": llh.get_growth_rates(),
        "root_distribution": llh.get_root_distribution(),
        "sampling_probability": llh.get_sampling_probability(),
        "sampling_prob_float": float(sampling_prob),
        "idx2state": model_info["idx2state"],
        "idx2potency": model_info["idx2potency"],
        "n_states": llh.num_states,
        "start_state": model_info.get("start_state"),
        "backend": model_info.get("backend", "fundamental"),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/model_dict.pkl", "wb") as fp:
        pickle.dump(model_dict, fp)
