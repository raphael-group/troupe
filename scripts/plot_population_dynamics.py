#!/usr/bin/env python
"""Plot expected population proportion over time for a ClaSSE model_dict.

Given a fitted model_dict.pkl (birth kernel B and per-type growth rates lam),
this computes the expected number of cells of each type over time and plots
each type's share of the total population.

Model dynamics (ClaSSE pure-birth):
  A type-i cell divides at rate lam[i].  At division it is replaced by two
  daughters, each independently of type j with probability B[i, j].  Letting
  n_j(t) be the expected number of type-j cells, the expected counts evolve as

      d n_j / dt = sum_i n_i * lam_i * (2 * B[i, j] - delta_{ij})

  i.e. dn/dt = M^T n with M[i, j] = 2 * lam_i * B[i, j] - lam_i * delta_{ij},
  so n(t) = expm(t * M^T) @ n(0).  The plotted proportion is n_j(t) / sum_k n_k(t).

The initial condition n(0) is the model's root_distribution (the distribution
over the starting cell's type).

Usage:
    python scripts/plot_population_dynamics.py \
        -i /Users/william_hs/Desktop/Projects/troupe/results/TLSC_new/sample_0.05/reg=1.0/select_potencies/model_dict.pkl \
        -o /Users/william_hs/Desktop/Projects/troupe/results/TLSC_new/sample_0.05/reg=1.0/select_potencies/population_comp.pdf \
        --trees /Users/william_hs/Desktop/Projects/troupe/experiments/TLSC_new/processed_data/trees.pkl \
        -T 1.0 \
        --n_bars 200

    python scripts/plot_population_dynamics.py \
        -i /Users/william_hs/Desktop/Projects/troupe/results/TLS_no_endothelial_new/sample_0.05/reg=3.0/select_potencies/model_dict.pkl \
        -o /Users/william_hs/Desktop/Projects/troupe/results/TLS_no_endothelial_new/sample_0.05/reg=3.0/select_potencies/population_comp.pdf \
        --trees /Users/william_hs/Desktop/Projects/troupe/experiments/TLS_no_endothelial_new/processed_data/trees.pkl \
        -T 1.0 \
        --n_bars 200
"""

import argparse
import collections
import os
import pickle
import sys

import numpy as np
import torch
from scipy.linalg import expm

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_theme()
sns.set_style("white")
sns.set_palette("Dark2")

mpl.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 17,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'font.family': 'sans-serif',
    'font.weight': 600,
    'axes.labelweight': 600,
    'axes.titleweight': 600,
    'figure.autolayout': True
    })
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# Reuse the TLS palette / fallback colors from the birth-kernel plotter.
TLS_COLORS = {
    'NeuralTube':  '#1b9e77',
    'Somite':      '#d95f02',
    'Endoderm':    '#7570b3',
    'PCGLC':       '#e7298a',
    'NMP':         '#e6ab02'
,
}
FALLBACK_COLORS = [
    '#66a61e', '#999999',
]


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def population_counts(B, growth_rates, n0, times):
    """Expected number of cells of each type over a grid of times.

    Args:
        B: (K, K) birth kernel. B[i, j] is the probability a type-i parent
            produces a daughter of type j (rows sum to 1).
        growth_rates: (K,) per-type division rates lam.
        n0: (K,) initial expected counts (e.g. the root distribution).
        times: 1D array of times at which to evaluate.

    Returns:
        (len(times), K) array of expected counts n_j(t), unnormalized.
    """
    lam = np.asarray(growth_rates, dtype=float)
    K = B.shape[0]
    # M[i, j] = 2 * lam_i * B[i, j] - lam_i * delta_{ij}
    M = 2.0 * lam[:, None] * B - np.diag(lam)
    A = M.T  # dn/dt = A n
    counts = np.empty((len(times), K))
    for t_idx, t in enumerate(times):
        counts[t_idx] = expm(A * t) @ n0
    return counts


def _tls_color(name):
    """Return the TLS palette color for a state name, or None if not in palette.

    Matches exact names and tolerates a trailing plural 's' (e.g. the observed
    state "NMPs" maps to the "NMP" palette entry).
    """
    key = str(name)
    if key in TLS_COLORS:
        return TLS_COLORS[key]
    if key.endswith("s") and key[:-1] in TLS_COLORS:
        return TLS_COLORS[key[:-1]]
    return None


def assign_colors(state_names):
    """Color each state: TLS palette for observed states, fallbacks otherwise.

    Fallback colors are handed out in order to the states not in the palette so
    that distinct states never collide.
    """
    colors = []
    fallback_rank = 0
    for name in state_names:
        tls = _tls_color(name)
        if tls is not None:
            colors.append(tls)
        else:
            colors.append(FALLBACK_COLORS[fallback_rank % len(FALLBACK_COLORS)])
            fallback_rank += 1
    return colors


def observed_counts(trees_path, state_names):
    """Number of leaves in each state, aligned to state_names.

    Leaves carry their terminal label in the ``.state`` attribute (a name that
    matches the model's idx2state values for observed states).  Hidden states
    that never appear at a leaf get count 0.

    Args:
        trees_path: Path to trees as a pickle (.pkl) or newick (.nwk) file.
        state_names: Ordered list of model state names to align the output to.

    Returns:
        A 1D array of length len(state_names) of leaf counts, one entry per
        state in state_names order.
    """
    if trees_path.endswith(".pkl"):
        with open(trees_path, "rb") as fp:
            trees = pickle.load(fp)
    elif trees_path.endswith(".nwk"):
        import ete3
        trees = []
        with open(trees_path) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                t = ete3.Tree(line, format=1)
                for leaf in t.get_leaves():
                    leaf.add_feature("state", leaf.name)
                trees.append(t)
    else:
        raise ValueError(f"Unsupported trees format: {trees_path}. Use .pkl or .nwk")

    counts = collections.Counter()
    for t in trees:
        for leaf in t.get_leaves():
            counts[str(leaf.state)] += 1

    unmatched = sorted(set(counts) - set(state_names))
    if unmatched:
        print(f"Warning: leaf states not present in the model, ignored: {unmatched}")

    return np.array([counts.get(name, 0) for name in state_names], dtype=float)


def main():
    parser = argparse.ArgumentParser(
        description="Plot expected population proportion over time for a ClaSSE model_dict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to model_dict.pkl (or a directory containing it).")
    parser.add_argument("-o", "--output", default=None,
                        help="Output figure path (.pdf/.png/.svg). "
                             "Default: figures/population_dynamics.pdf next to model_dict.pkl.")
    parser.add_argument("-T", "--time", type=float, default=1.0,
                        help="Total time horizon to simulate (default: 1.0, the unit "
                             "tree height used elsewhere in the pipeline).")
    parser.add_argument("--n_bars", type=int, default=20,
                        help="Number of time bins (bars) to draw (default: 20).")
    parser.add_argument("--trees", default=None,
                        help="Optional path to trees (.pkl or .nwk). Adds a "
                             "white-hatched bar of observed leaf-state proportions "
                             "on the right for comparison.")
    args = parser.parse_args()

    model_path = args.input
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model_dict.pkl")
    if not os.path.isfile(model_path):
        sys.exit(f"ERROR: model_dict.pkl not found at {model_path}")

    with open(model_path, "rb") as fp:
        model_dict = pickle.load(fp)

    B = _to_numpy(model_dict["daughter_kernel"]).astype(float)
    growth_rates = _to_numpy(model_dict["growth_rates"]).astype(float).ravel()
    idx2state = model_dict["idx2state"]
    K = B.shape[0]

    # Initial condition: the root (starting-cell) distribution.
    n0 = _to_numpy(model_dict["root_distribution"]).astype(float).ravel()
    if n0.sum() > 0:
        n0 = n0 / n0.sum()

    times = np.linspace(0.0, args.time, args.n_bars)
    counts = population_counts(B, growth_rates, n0, times)

    state_names = [str(idx2state[i]) for i in range(K)]
    colors = assign_colors(state_names)

    # With trees, rescale expected counts so the final expected total equals the
    # observed leaf total; this makes the Expected/Observed bars directly
    # comparable (both stacks reach the same height).
    obs = None
    if args.trees is not None:
        obs = observed_counts(args.trees, state_names)
        exp_total_T = counts[-1].sum()
        if exp_total_T > 0 and obs.sum() > 0:
            counts = counts * (obs.sum() / exp_total_T)

    # Stacked area on a log y-axis: each type is a filled polygon between the
    # cumulative counts of the types stacked below and including it.
    fig, ax = plt.subplots(figsize=(8, 5))
    cum = np.concatenate([np.zeros((len(times), 1)), np.cumsum(counts, axis=1)], axis=1)
    for i in range(K):
        ax.fill_between(times, cum[:, i], cum[:, i + 1],
                        label=state_names[i], color=colors[i], linewidth=0.0)

    # Comparison bars on the right: expected counts at the final time (solid)
    # next to the observed leaf counts (white-hatched).  Totals match by
    # construction, so the stacks are the same height and easy to compare.
    if obs is not None:
        exp_counts = counts[-1]

        gap = args.time * 0.05        # polygon -> first bar
        bar_w = args.time * 0.09
        bar_gap = args.time * 0.04    # between the two bars
        exp_center = args.time + gap + bar_w / 2
        obs_center = exp_center + bar_w + bar_gap

        def _stack(center, values, hatch=None):
            bottom = 0.0
            for i in range(K):
                ax.bar(center, values[i], width=bar_w, bottom=bottom,
                       color=colors[i], edgecolor="white", linewidth=0.0,
                       hatch=hatch)
                bottom += values[i]

        _stack(exp_center, exp_counts)
        _stack(obs_center, obs, hatch="////")

        # Drop the final time tick so it doesn't crowd the bar labels.
        time_ticks = np.linspace(0.0, args.time, 6)[:-1]
        ax.set_xticks(list(time_ticks) + [exp_center, obs_center])
        labels = ax.set_xticklabels(
            [f"{t:g}" for t in time_ticks] + ["Expected", "Observed"])
        for lab in labels[-2:]:  # angle the two bar labels so they don't collide
            lab.set_rotation(30)
            lab.set_ha("right")
        ax.set_xlim(times[0], obs_center + bar_w / 2)
    else:
        ax.set_xlim(times[0], times[-1])

    total_curve = counts.sum(axis=1)
    ax.set_ylim(0, total_curve.max() * 1.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("Expected population count")
    ax.set_title("Expected population count over time")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="State")
    sns.despine(ax=ax)  # drop the top and right spines
    fig.tight_layout()

    if args.output is None:
        fig_dir = os.path.join(os.path.dirname(os.path.abspath(model_path)), "figures")
        os.makedirs(fig_dir, exist_ok=True)
        outfile = os.path.join(fig_dir, "population_dynamics.pdf")
    else:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        outfile = args.output

    fig.savefig(outfile, bbox_inches="tight")
    print(f"Saved to {outfile}")

    # Report the final-time composition for convenience.
    print(f"\nExpected counts at T={args.time:g}:")
    for i in range(K):
        print(f"  {state_names[i]:>12}: {counts[-1, i]:.4g}")


if __name__ == "__main__":
    main()
