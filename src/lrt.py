"""Parametric bootstrap likelihood ratio test for ClaSSE birth kernel constraints.

Main entry points:
  compute_lrt_statistic     — fit both models on a given tree dataset
  parametric_bootstrap_lrt  — full bootstrap LRT with p-value
"""

import json
import logging
import os

import numpy as np

from bootstrap import sample_bootstrap_trees, estimate_simulation_time
from constrained_optimizer import (
    compute_unconstrained_mle,
    compute_constrained_mle,
)

logger = logging.getLogger(__name__)


def compute_lrt_statistic(
    trees: list,
    n_states: int,
    model_info: dict,
    sampling_prob: float,
    constraints: list,
    device,
    output_dir: str = None,
    num_iter_uncon: int = 100,
    num_iter_con: int = 50,
    penalty_weight_schedule: tuple = (1.0, 10.0, 100.0, 1000.0),
    constraint_tol: float = 1e-5,
    do_logging: bool = False,
) -> tuple:
    """Fit unconstrained and null (constrained) ClaSSE models and return the LRT stat.

    Both models always respect the potency support mask encoded in model_info.
    The null model adds the extra linear inequalities in ``constraints``.

    Args:
        trees: Binarized ete3 trees with int-castable leaf states.
        n_states: Total number of states (observed + hidden).
        model_info: Dict with idx2potency, idx2state, and optional init params.
        sampling_prob: Leaf sampling probability eta in (0, 1].
        constraints: List of KernelConstraint — extra inequalities for H0.
        device: Torch device.
        output_dir: If given, save unconstrained and null model dicts to
            ``output_dir/unconstrained/`` and ``output_dir/null/``.
        num_iter_uncon: LBFGS iterations for unconstrained fit.
        num_iter_con: LBFGS iterations per penalty phase for constrained fit.
        penalty_weight_schedule: Mu values for the outer penalty loop.
        constraint_tol: Tolerance for constraint satisfaction check.
        do_logging: Log optimization progress.

    Returns:
        (log_lik_unconstrained, log_lik_null, lrt_stat)
        lrt_stat = 2 * (ll_uncon - ll_null) >= 0.
    """
    uncon_dir = os.path.join(output_dir, "unconstrained") if output_dir else None
    null_dir = os.path.join(output_dir, "null") if output_dir else None

    # Step 1: Fit unconstrained model.
    logger.info("Fitting unconstrained model (%d trees, %d states).",
                len(trees), n_states)
    uncon_llh, ll_uncon = compute_unconstrained_mle(
        trees, n_states, model_info, sampling_prob, device,
        output_dir=uncon_dir, num_iter=num_iter_uncon, do_logging=do_logging,
    )
    logger.info("Unconstrained log-likelihood: %.4f", ll_uncon)

    # Step 2: Fit null (constrained) model, warm-started from unconstrained MLE.
    logger.info("Fitting null (constrained) model.")
    null_llh, ll_null = compute_constrained_mle(
        trees, n_states, model_info, sampling_prob, constraints, device,
        warm_start_llh=uncon_llh,
        penalty_weight_schedule=penalty_weight_schedule,
        constraint_tol=constraint_tol,
        num_iter_per_phase=num_iter_con,
        output_dir=null_dir,
        do_logging=do_logging,
    )
    logger.info("Null log-likelihood: %.4f", ll_null)

    lrt_stat = 2.0 * (ll_uncon - ll_null)
    # Clamp at 0 to avoid tiny negative values from numerical noise.
    lrt_stat = max(0.0, lrt_stat)
    logger.info("LRT statistic: %.4f", lrt_stat)

    return ll_uncon, ll_null, lrt_stat


def parametric_bootstrap_lrt(
    observed_trees: list,
    null_model_dict: dict,
    n_states: int,
    model_info: dict,
    sampling_prob: float,
    constraints: list,
    device,
    T: float = None,
    B: int = 99,
    alpha: float = 0.05,
    seed: int = 0,
    output_dir: str = "lrt_results",
    num_iter_uncon: int = 100,
    num_iter_con: int = 50,
    penalty_weight_schedule: tuple = (1.0, 10.0, 100.0, 1000.0),
    constraint_tol: float = 1e-5,
    do_logging: bool = False,
) -> dict:
    """Full parametric bootstrap likelihood ratio test.

    Workflow:
      1. Compute observed LRT statistic Lambda_obs on the observed trees.
      2. Simulate B datasets from the null (constrained) model.
      3. For each bootstrap replicate, compute Lambda_b by re-fitting both models.
      4. Empirical p-value = #{Lambda_b >= Lambda_obs} / B.
      5. Reject H0 if p-value < alpha.

    Args:
        observed_trees: Observed dataset (list of binarized trees).
        null_model_dict: model_dict from the constrained MLE (contains
            ``daughter_kernel`` and ``growth_rates`` for simulation).
        n_states: Total number of states.
        model_info: Dict with idx2potency, idx2state, etc.
        sampling_prob: Leaf sampling probability.
        constraints: List of KernelConstraint objects defining H0.
        device: Torch device.
        T: Simulation time horizon. If None, estimated from observed_trees
            via estimate_simulation_time.
        B: Number of bootstrap replicates.
        alpha: Significance level.
        seed: Base random seed for simulation.
        output_dir: Directory for saving results JSON and bootstrap plot.
        num_iter_uncon: LBFGS iterations for unconstrained fits.
        num_iter_con: LBFGS iterations per penalty phase for constrained fits.
        penalty_weight_schedule: Mu schedule for constrained optimizer.
        constraint_tol: Tolerance for constraint satisfaction.
        do_logging: Log optimization progress for each replicate.

    Returns:
        Dict with keys:
          lrt_stat_observed, log_lik_unconstrained, log_lik_null,
          bootstrap_stats, p_value, critical_value, reject, alpha, B,
          constraints, T.
    """
    if T is None:
        T = estimate_simulation_time(observed_trees)
        logger.info("Estimated simulation time T = %.4f", T)

    start_state = model_info.get("start_state")
    if start_state is None:
        # Infer from idx2potency: most potent state is the root.
        idx2potency = model_info["idx2potency"]
        start_state = max(idx2potency, key=lambda k: len(idx2potency[k]))
        logger.info("Inferred start_state = %d", start_state)

    # -----------------------------------------------------------------------
    # Step 1: Observed LRT statistic.
    # -----------------------------------------------------------------------
    obs_out = os.path.join(output_dir, "observed") if output_dir else None
    ll_uncon, ll_null, lrt_obs = compute_lrt_statistic(
        observed_trees, n_states, model_info, sampling_prob, constraints, device,
        output_dir=obs_out,
        num_iter_uncon=num_iter_uncon,
        num_iter_con=num_iter_con,
        penalty_weight_schedule=penalty_weight_schedule,
        constraint_tol=constraint_tol,
        do_logging=do_logging,
    )
    logger.info("Observed LRT statistic: %.4f", lrt_obs)

    # -----------------------------------------------------------------------
    # Step 2: Simulate B bootstrap datasets from the null model.
    # -----------------------------------------------------------------------
    logger.info("Simulating %d bootstrap datasets (T=%.4f, n_trees=%d).",
                B, T, len(observed_trees))
    boot_datasets = sample_bootstrap_trees(
        null_model_dict=null_model_dict,
        n_trees=len(observed_trees),
        T=T,
        B=B,
        start_state=start_state,
        sampling_prob=float(sampling_prob),
        seed=seed,
    )

    # -----------------------------------------------------------------------
    # Step 3: Bootstrap LRT statistics.
    # -----------------------------------------------------------------------
    bootstrap_stats = []
    for b, boot_trees in enumerate(boot_datasets):
        if len(boot_trees) == 0:
            logger.warning("Bootstrap replicate %d has 0 trees — skipping.", b)
            bootstrap_stats.append(0.0)
            continue

        boot_out = os.path.join(output_dir, f"bootstrap_{b:04d}") if False else None
        try:
            _, _, lrt_b = compute_lrt_statistic(
                boot_trees, n_states, model_info, sampling_prob, constraints, device,
                output_dir=boot_out,
                num_iter_uncon=num_iter_uncon,
                num_iter_con=num_iter_con,
                penalty_weight_schedule=penalty_weight_schedule,
                constraint_tol=constraint_tol,
                do_logging=False,
            )
        except RuntimeError as e:
            logger.warning("Bootstrap replicate %d failed (%s). Using LRT=0.", b, e)
            lrt_b = 0.0

        bootstrap_stats.append(lrt_b)
        logger.info("Bootstrap replicate %d / %d: LRT=%.4f", b + 1, B, lrt_b)

    # -----------------------------------------------------------------------
    # Step 4: p-value.
    # -----------------------------------------------------------------------
    boot_arr = np.array(bootstrap_stats)
    p_value = float(np.mean(boot_arr >= lrt_obs))
    critical_value = float(np.quantile(boot_arr, 1.0 - alpha)) if len(boot_arr) > 0 else float("nan")
    reject = p_value < alpha

    results = {
        "lrt_stat_observed": lrt_obs,
        "log_lik_unconstrained": ll_uncon,
        "log_lik_null": ll_null,
        "bootstrap_stats": boot_arr.tolist(),
        "p_value": p_value,
        "critical_value": critical_value,
        "reject": reject,
        "alpha": alpha,
        "B": B,
        "T": T,
        "constraints": [c.to_dict() for c in constraints],
    }

    logger.info(
        "LRT result: Λ_obs=%.4f  p=%.4f  critical(%.0f%%)=%.4f  reject=%s",
        lrt_obs, p_value, (1 - alpha) * 100, critical_value, reject,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "lrt_results.json")
        with open(results_path, "w") as fp:
            json.dump(results, fp, indent=2)
        logger.info("Results saved to %s", results_path)

        _save_bootstrap_plot(lrt_obs, boot_arr, alpha, critical_value, output_dir)

    return results


def _save_bootstrap_plot(lrt_obs, boot_arr, alpha, critical_value, output_dir):
    """Save a histogram of bootstrap statistics with Lambda_obs marked."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(boot_arr, bins=max(10, len(boot_arr) // 5), color="steelblue",
                edgecolor="white", alpha=0.8, label="Bootstrap Λ_b")
        ax.axvline(lrt_obs, color="crimson", linewidth=2, label=f"Observed Λ = {lrt_obs:.3f}")
        ax.axvline(critical_value, color="darkorange", linewidth=1.5, linestyle="--",
                   label=f"{int((1-alpha)*100)}% critical value = {critical_value:.3f}")
        ax.set_xlabel("LRT statistic Λ")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        fig.tight_layout()
        plot_path = os.path.join(output_dir, "bootstrap_distribution.pdf")
        fig.savefig(plot_path)
        plt.close(fig)
        logger.info("Bootstrap plot saved to %s", plot_path)
    except Exception as e:
        logger.warning("Could not save bootstrap plot: %s", e)
