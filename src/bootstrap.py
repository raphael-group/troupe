"""Parametric bootstrap sampling for ClaSSE likelihood ratio testing.

Simulates synthetic tree datasets from a fitted null (constrained) ClaSSE
model. Each bootstrap replicate has the same number of trees as the observed
dataset, simulated at an estimated time T.
"""

import logging

import numpy as np
from ete3 import TreeNode

from classe_branching_simulation import simulate_tree as _simulate_one_tree
from utils import binarize_tree

logger = logging.getLogger(__name__)


def estimate_simulation_time(trees: list) -> float:
    """Estimate the simulation horizon T as the mean root-to-tip distance.

    ClaSSE trees are ultrametric, so any leaf gives the same root-to-tip
    sum within a tree. We average this across all trees.

    Args:
        trees: List of ete3 TreeNode objects (from the observed dataset).

    Returns:
        Estimated T as a float.
    """
    times = []
    for t in trees:
        leaf = t.get_leaves()[0]
        total = leaf.dist + sum(n.dist for n in leaf.iter_ancestors())
        times.append(total)
    return float(np.mean(times))


def _simulate_one_dataset(
    birth_kernel: np.ndarray,
    growth_rates: np.ndarray,
    start_state: int,
    T: float,
    n_trees: int,
    sampling_prob: float,
    seed: int,
) -> list:
    """Simulate one bootstrap dataset of binarized trees.

    Args:
        birth_kernel: (K, K) numpy array — row-stochastic daughter kernel.
        growth_rates: (K,) numpy array — per-type birth rates.
        start_state: Integer index of the root cell type.
        T: Simulation time horizon.
        n_trees: Number of trees to simulate (some may be None if sampling
            prunes all leaves; these are skipped).
        sampling_prob: Leaf sampling probability passed to simulate_tree.
        seed: Base random seed; individual trees use seed + tree_index.

    Returns:
        List of binarized ete3 TreeNode objects (length <= n_trees).
    """
    trees = []
    for k in range(n_trees):
        tree = _simulate_one_tree(
            birth_kernel=birth_kernel,
            growth_rates=growth_rates,
            starting_type=start_state,
            T=T,
            sample_probability=sampling_prob,
            seed=seed + k,
        )
        if tree is None:
            continue
        trees.append(binarize_tree(tree))
    return trees


def sample_bootstrap_trees(
    null_model_dict: dict,
    n_trees: int,
    T: float,
    B: int,
    start_state: int,
    sampling_prob: float = 1.0,
    seed: int = 0,
) -> list:
    """Generate B bootstrap datasets from the fitted null ClaSSE model.

    Each dataset is a list of binarized trees simulated using the null model's
    daughter kernel and growth rates.

    Args:
        null_model_dict: model_dict.pkl loaded from the constrained MLE output.
            Must contain ``daughter_kernel`` and ``growth_rates`` tensors.
        n_trees: Number of trees per bootstrap replicate (matches observed data).
        T: Simulation time horizon (use estimate_simulation_time on observed trees).
        B: Number of bootstrap replicates.
        start_state: Integer index of the starting (root) cell type.
        sampling_prob: Leaf sampling probability for each simulated tree.
        seed: Base random seed; replicate b uses seed + b * n_trees as its offset.

    Returns:
        List of B elements, each a list of binarized TreeNode objects.
        Replicates with fewer than n_trees // 2 surviving trees trigger a warning.
    """
    B_kernel = null_model_dict["daughter_kernel"]
    lam = null_model_dict["growth_rates"]

    if hasattr(B_kernel, "detach"):
        birth_kernel_np = B_kernel.detach().cpu().numpy()
    else:
        birth_kernel_np = np.asarray(B_kernel)

    if hasattr(lam, "detach"):
        growth_rates_np = lam.detach().cpu().numpy()
    else:
        growth_rates_np = np.asarray(lam)

    datasets = []
    for b in range(B):
        replicate_seed = seed + b * n_trees
        dataset = _simulate_one_dataset(
            birth_kernel=birth_kernel_np,
            growth_rates=growth_rates_np,
            start_state=start_state,
            T=T,
            n_trees=n_trees,
            sampling_prob=sampling_prob,
            seed=replicate_seed,
        )
        if len(dataset) < max(1, n_trees // 2):
            logger.warning(
                "Bootstrap replicate %d produced only %d / %d trees "
                "(many leaves were pruned by sampling). "
                "Consider increasing sampling_prob or T.",
                b, len(dataset), n_trees,
            )
        datasets.append(dataset)
        if (b + 1) % 10 == 0:
            logger.info("Simulated %d / %d bootstrap replicates.", b + 1, B)

    return datasets
