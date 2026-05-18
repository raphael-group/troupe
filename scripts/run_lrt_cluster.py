#!/usr/bin/env python
"""Cluster-parallel parametric bootstrap LRT for ClaSSE birth kernel constraints.

Distributes the B bootstrap inference jobs as a SLURM job array, cutting
wall-clock time from O(B) to O(1) (one job per replicate runs in parallel).

Three-step workflow:

  Step 1 — prepare (run locally, ~minutes):
    Computes the observed LRT statistic, simulates all B bootstrap datasets,
    saves them to disk, and writes a ready-to-submit SLURM job-array script.

    python scripts/run_lrt_cluster.py prepare \\
        -i experiments/TLSC/processed_data/trees.pkl \\
        --model results/TLSC/sample_0.05/best_model_dict.pkl \\
        --constraint "B[NMPs,NeuralTube] >= 2.5 * B[NMPs,Somite]" \\
        --sampling_prob 0.05 --B 99 \\
        --output experiments/TLSC/lrt/nmp_neuraltube_somite

  Step 2 — submit (one command):
    sbatch experiments/TLSC/lrt/nmp_neuraltube_somite/slurm_submit.sh
    (or pass --submit to prepare to do this automatically)

  Step 3 — aggregate (run locally after jobs finish, seconds):
    python scripts/run_lrt_cluster.py aggregate \\
        --output experiments/TLSC/lrt/nmp_neuraltube_somite

Output directory layout:
  <output>/
    work/
      meta.pkl               model params and optimizer settings for workers
      observed_lrt.json      LRT stat on the real data (computed in prepare)
      bootstrap_trees/       one .pkl per replicate
      bootstrap_results/     one .json per replicate (written by workers)
    slurm_submit.sh          generated SLURM job-array script
    slurm_logs/              per-task stdout / stderr
    lrt_results.json         final p-value and decision (written by aggregate)
    bootstrap_distribution.pdf
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_lrt import parse_constraint_str, load_trees
from bootstrap import sample_bootstrap_trees, estimate_simulation_time
from lrt import compute_lrt_statistic, _save_bootstrap_plot

sys.setrecursionlimit(5000)
torch.set_default_dtype(torch.float64)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SLURM script builder
# ---------------------------------------------------------------------------

def _make_slurm_script(
    B: int,
    work_dir: str,
    repo_root: str,
    venv_activate: str,
    log_dir: str,
    time: str,
    mem: str,
    cpus: int,
    partition: str,
    job_name: str = "lrt_bootstrap",
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{B - 1}",
        f"#SBATCH --time={time}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    lines += [
        f"#SBATCH --output={log_dir}/%A_%a.out",
        f"#SBATCH --error={log_dir}/%A_%a.err",
        "",
        f"source {venv_activate}",
        f"cd {repo_root}",
        "python scripts/lrt_worker.py \\",
        f"    --work_dir {work_dir} \\",
        "    --replicate $SLURM_ARRAY_TASK_ID",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------

def cmd_prepare(args):
    device = torch.device(args.device)
    output_dir = os.path.abspath(args.output)
    work_dir = os.path.join(output_dir, "work")
    trees_dir = os.path.join(work_dir, "bootstrap_trees")
    results_dir = os.path.join(work_dir, "bootstrap_results")
    log_dir = os.path.join(output_dir, "slurm_logs")
    for d in [work_dir, trees_dir, results_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    # Load model dict.
    logger.info("Loading model from %s", args.model)
    with open(args.model, "rb") as fp:
        model_dict = pickle.load(fp)

    idx2state = model_dict["idx2state"]
    idx2potency = model_dict["idx2potency"]
    state2idx = {s: i for i, s in idx2state.items()}
    n_states = model_dict.get("n_states", len(idx2state))
    start_state = model_dict.get("start_state")

    sampling_prob = args.sampling_prob
    if sampling_prob is None:
        sampling_prob = float(model_dict.get("sampling_prob_float",
                                             model_dict.get("sampling_probability", 1.0)))
    logger.info("Sampling probability: %.4f", sampling_prob)

    model_info = {
        "idx2potency": idx2potency,
        "idx2state": idx2state,
        "start_state": start_state,
        "backend": model_dict.get("backend", "fundamental"),
    }
    if "daughter_kernel" in model_dict:
        B_loaded = model_dict["daughter_kernel"]
        if hasattr(B_loaded, "detach"):
            model_info["B_params_init"] = torch.log(B_loaded.clamp(min=1e-10))
    if "growth_rates" in model_dict:
        lam = model_dict["growth_rates"]
        if hasattr(lam, "detach"):
            model_info["growth_params_init"] = torch.log(torch.expm1(lam.clamp(min=1e-10)))

    # Parse constraints.
    constraints = []
    for cstr in args.constraints:
        try:
            c = parse_constraint_str(cstr, state2idx)
        except ValueError as e:
            logger.error("Failed to parse constraint %r: %s", cstr, e)
            sys.exit(1)
        logger.info("Constraint: %s", c.label)
        constraints.append(c)

    # Load trees.
    logger.info("Loading trees from %s", args.input)
    trees = load_trees(args.input, state2idx=state2idx)
    logger.info("Loaded %d trees.", len(trees))

    # Estimate simulation time T.
    T = args.T
    if T is None:
        T = estimate_simulation_time(trees)
        logger.info("Estimated simulation time T = %.4f", T)

    if start_state is None:
        start_state = max(idx2potency, key=lambda k: len(idx2potency[k]))
        model_info["start_state"] = start_state
        logger.info("Inferred start_state = %d", start_state)

    # Compute observed LRT statistic (runs locally).
    logger.info("Computing observed LRT statistic...")
    obs_dir = os.path.join(output_dir, "observed")
    ll_uncon, ll_null, lrt_obs = compute_lrt_statistic(
        trees=trees,
        n_states=n_states,
        model_info=model_info,
        sampling_prob=sampling_prob,
        constraints=constraints,
        device=device,
        output_dir=obs_dir,
        num_iter_uncon=args.num_iter_uncon,
        num_iter_con=args.num_iter_con,
        penalty_weight_schedule=tuple(args.penalty_schedule),
        constraint_tol=args.constraint_tol,
        do_logging=args.verbose,
    )
    logger.info("Observed LRT: Lambda=%.4f (ll_uncon=%.4f, ll_null=%.4f)",
                lrt_obs, ll_uncon, ll_null)

    with open(os.path.join(work_dir, "observed_lrt.json"), "w") as fp:
        json.dump({
            "lrt_stat_observed": lrt_obs,
            "log_lik_unconstrained": ll_uncon,
            "log_lik_null": ll_null,
            "T": T,
            "B": args.B,
            "alpha": args.alpha,
            "constraints": [c.to_dict() for c in constraints],
        }, fp, indent=2)

    # Simulate B bootstrap datasets.
    logger.info("Simulating %d bootstrap datasets (T=%.4f)...", args.B, T)
    boot_datasets = sample_bootstrap_trees(
        null_model_dict=model_dict,
        n_trees=len(trees),
        T=T,
        B=args.B,
        start_state=start_state,
        sampling_prob=float(sampling_prob),
        seed=args.seed,
    )
    for b, dataset in enumerate(boot_datasets):
        with open(os.path.join(trees_dir, f"{b:04d}.pkl"), "wb") as fp:
            pickle.dump(dataset, fp)
    logger.info("Saved %d bootstrap datasets.", args.B)

    # Save metadata for workers.
    meta = {
        "n_states": n_states,
        "model_info": model_info,
        "sampling_prob": sampling_prob,
        "constraints": constraints,
        "num_iter_uncon": args.num_iter_uncon,
        "num_iter_con": args.num_iter_con,
        "penalty_weight_schedule": tuple(args.penalty_schedule),
        "constraint_tol": args.constraint_tol,
    }
    with open(os.path.join(work_dir, "meta.pkl"), "wb") as fp:
        pickle.dump(meta, fp)

    # Write SLURM job-array script.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    venv = args.venv or os.path.join(repo_root, ".venv")
    venv_activate = os.path.join(venv, "bin", "activate")

    slurm_script = _make_slurm_script(
        B=args.B,
        work_dir=output_dir,
        repo_root=repo_root,
        venv_activate=venv_activate,
        log_dir=log_dir,
        time=args.slurm_time,
        mem=args.slurm_mem,
        cpus=args.slurm_cpus,
        partition=args.slurm_partition,
    )
    slurm_path = os.path.join(output_dir, "slurm_submit.sh")
    with open(slurm_path, "w") as fp:
        fp.write(slurm_script)
    os.chmod(slurm_path, 0o755)
    logger.info("SLURM script written to %s", slurm_path)

    print(f"\nPrepare complete. Observed Lambda = {lrt_obs:.4f}")
    if args.submit:
        logger.info("Submitting SLURM job array...")
        result = subprocess.run(["sbatch", slurm_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            logger.error("sbatch failed:\n%s", result.stderr)
            sys.exit(1)
    else:
        print(f"  Submit: sbatch {slurm_path}")
        print(f"  Then aggregate: python scripts/run_lrt_cluster.py aggregate --output {output_dir}")


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def cmd_aggregate(args):
    output_dir = os.path.abspath(args.output)
    work_dir = os.path.join(output_dir, "work")
    results_dir = os.path.join(work_dir, "bootstrap_results")

    obs_path = os.path.join(work_dir, "observed_lrt.json")
    if not os.path.exists(obs_path):
        logger.error("observed_lrt.json not found in %s — run 'prepare' first.", work_dir)
        sys.exit(1)
    with open(obs_path) as fp:
        obs = json.load(fp)

    lrt_obs = obs["lrt_stat_observed"]
    B = obs["B"]
    alpha = args.alpha if args.alpha is not None else obs["alpha"]

    # Collect per-replicate results.
    bootstrap_stats = []
    missing = []
    for b in range(B):
        path = os.path.join(results_dir, f"{b:04d}.json")
        if not os.path.exists(path):
            missing.append(b)
            continue
        with open(path) as fp:
            bootstrap_stats.append(json.load(fp)["lrt_stat"])

    n_complete = len(bootstrap_stats)
    if missing:
        logger.warning(
            "%d / %d replicates missing: %s%s",
            len(missing), B,
            str(missing[:10]),
            " ..." if len(missing) > 10 else "",
        )
    if n_complete == 0:
        logger.error("No bootstrap results found in %s.", results_dir)
        sys.exit(1)

    logger.info("Aggregating %d / %d replicates.", n_complete, B)

    boot_arr = np.array(bootstrap_stats)
    p_value = float(np.mean(boot_arr >= lrt_obs))
    critical_value = float(np.quantile(boot_arr, 1.0 - alpha))
    reject = p_value < alpha

    results = {
        **obs,
        "bootstrap_stats": boot_arr.tolist(),
        "n_bootstrap_completed": n_complete,
        "p_value": p_value,
        "critical_value": critical_value,
        "reject": reject,
        "alpha": alpha,
    }
    results_path = os.path.join(output_dir, "lrt_results.json")
    with open(results_path, "w") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Results saved to %s", results_path)

    _save_bootstrap_plot(lrt_obs, boot_arr, alpha, critical_value, output_dir)

    print("\n" + "=" * 60)
    print("  LIKELIHOOD RATIO TEST RESULT")
    print("=" * 60)
    for c in obs.get("constraints", []):
        print(f"  H0: {c['label']}")
    print(f"\n  Observed LRT statistic:  {lrt_obs:.4f}")
    print(f"  Log-lik unconstrained:   {obs['log_lik_unconstrained']:.4f}")
    print(f"  Log-lik null:            {obs['log_lik_null']:.4f}")
    print(f"\n  Bootstrap replicates:    {n_complete} / {B}")
    print(f"  p-value:                 {p_value:.4f}")
    print(f"  Critical value (alpha={alpha}): {critical_value:.4f}")
    print(f"  Decision:                {'REJECT H0' if reject else 'FAIL TO REJECT H0'}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cluster-parallel parametric bootstrap LRT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- prepare ---
    p = sub.add_parser("prepare",
                       help="Simulate bootstrap datasets and generate SLURM job array.")
    p.add_argument("-i", "--input", required=True,
                   help="Path to trees (.pkl or .nwk).")
    p.add_argument("--model", required=True,
                   help="Path to model_dict.pkl from a fitted ClaSSE-TROUPE run.")
    p.add_argument("--constraint", action="append", dest="constraints",
                   metavar="EXPR", required=True,
                   help=("Constraint expression, e.g. 'B[NMPs,NeuralTube] >= 2.5 * B[NMPs,Somite]'. "
                         "Repeatable for joint hypotheses."))
    p.add_argument("-o", "--output", default="lrt_results",
                   help="Output directory (default: lrt_results).")
    p.add_argument("--B", type=int, default=99,
                   help="Number of bootstrap replicates (default: 99).")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level (default: 0.05).")
    p.add_argument("--T", type=float, default=None,
                   help="Simulation time horizon. If omitted, estimated from trees.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for bootstrap simulation (default: 0).")
    p.add_argument("--sampling_prob", type=float, default=None,
                   help="Leaf sampling probability. Overrides value in model_dict.")
    p.add_argument("--num_iter_uncon", type=int, default=100)
    p.add_argument("--num_iter_con", type=int, default=50)
    p.add_argument("--penalty_schedule", type=float, nargs="+",
                   default=[1.0, 10.0, 100.0, 1000.0])
    p.add_argument("--constraint_tol", type=float, default=1e-5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", action="store_true")
    # SLURM options
    p.add_argument("--slurm_time", default="2:00:00",
                   help="Wall-clock limit per job (default: 2:00:00).")
    p.add_argument("--slurm_mem", default="8G",
                   help="Memory per job (default: 8G).")
    p.add_argument("--slurm_cpus", type=int, default=1,
                   help="CPUs per task (default: 1).")
    p.add_argument("--slurm_partition", default=None,
                   help="SLURM partition (optional).")
    p.add_argument("--venv", default=None,
                   help="Path to virtual environment directory "
                        "(default: <repo_root>/.venv).")
    p.add_argument("--submit", action="store_true",
                   help="Submit the SLURM script automatically via sbatch.")

    # --- aggregate ---
    a = sub.add_parser("aggregate",
                       help="Collect per-replicate results and compute p-value.")
    a.add_argument("-o", "--output", required=True,
                   help="Output directory used in the prepare step.")
    a.add_argument("--alpha", type=float, default=None,
                   help="Significance level (default: inherited from prepare).")

    args = parser.parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    else:
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
