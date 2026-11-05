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
  so n(t) = expm(t * M^T) @ n(0).  By default n_j(t) itself is plotted; with
  --proportion the plotted value is the composition n_j(t) / sum_k n_k(t).

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

Mela: the trees are ultrametric at height 8.0 with the root at t=0, so time is
embryonic days.  Extrapolating to E8.5 puts the horizon past the leaves, which
is why the comparison bars are pinned to --observed_time:

    python scripts/plot_population_dynamics.py \
        -i /Users/william_hs/Desktop/Projects/troupe/results/mela/sample_0.5_E8.0_phylogmm_lineage_R1/reg=50.0/select_potencies/model_dict.pkl \
        -o /Users/william_hs/Desktop/Projects/troupe/results/mela/sample_0.5_E8.0_phylogmm_lineage_R1/reg=50.0/select_potencies/figures/population_frac_E8.0_R1.pdf \
        --trees /Users/william_hs/Desktop/Projects/troupe/experiments/mela/processed_data/trees_E8.0_phylogmm_lineage_R1.pkl \
        -T 8.5 --observed_time 8.0 --mark_times 7.5 8.5 \
        --proportion --lump_hidden --n_bars 200
"""

import argparse
import collections
import os
import pickle
import re
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
# Coarse lineage labels used by the mela (chimeric mouse embryo) experiment.
MELA_COLORS = {
    'Mesoderm':               '#d95f02',
    'Neural':                 '#1b9e77',
    'SurfaceEctoderm':        '#66a61e',
    'Endoderm':               '#7570b3',
    'ExtraembryonicEctoderm': '#e7298a',
    'AxialProgenitor':        '#e6ab02',
    'EctodermProgenitor':     '#a6761d',
}
# run_classe_troupe names every state introduced during potency selection
# "U<idx>" (see newidx2state there).  These are exactly the inferred progenitors
# that carry no observed leaf label, and --lump_hidden merges them into one band.
HIDDEN_RE = re.compile(r"U\d+")
HIDDEN_LABEL = "Unobserved Progenitor"
EXTRA_COLORS = {HIDDEN_LABEL: '#999999'}

# Handed out in order to states with no palette entry (the hidden U* states when
# they are drawn separately).  Needs enough entries that no two states collide.
FALLBACK_COLORS = [
    '#999999', '#4d4d4d', '#cccccc', '#6a3d9a', '#1f78b4',
    '#b15928', '#fb9a99', '#33a02c', '#ff7f00', '#000000',
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
    """Return the palette color for a state name, or None if not in a palette.

    Matches exact names and tolerates a trailing plural 's' (e.g. the observed
    state "NMPs" maps to the "NMP" palette entry).
    """
    key = str(name)
    for palette in (TLS_COLORS, MELA_COLORS, EXTRA_COLORS):
        if key in palette:
            return palette[key]
        if key.endswith("s") and key[:-1] in palette:
            return palette[key[:-1]]
    return None


def assign_colors(state_names):
    """Color each state: named palette for observed states, fallbacks otherwise.

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


def lump_hidden(state_names, arrays):
    """Merge the pipeline's unobserved states into a single trailing column.

    The hidden states are the ones named "U<idx>" by run_classe_troupe, i.e.
    the inferred progenitors that no leaf label is attached to.  Drawn one per
    state they are a stack of slivers; merged they are one readable band.

    Args:
        state_names: Ordered list of state names.
        arrays: Iterable of arrays whose last axis is indexed by state. A None
            entry passes through as None.

    Returns:
        (new_names, new_arrays). If no state name is hidden, both are returned
        unchanged.
    """
    hidden = [i for i, n in enumerate(state_names) if HIDDEN_RE.fullmatch(n)]
    if not hidden:
        return list(state_names), list(arrays)

    keep = [i for i in range(len(state_names)) if i not in set(hidden)]
    new_names = [state_names[i] for i in keep] + [HIDDEN_LABEL]
    new_arrays = []
    for a in arrays:
        if a is None:
            new_arrays.append(None)
            continue
        a = np.asarray(a)
        new_arrays.append(np.concatenate(
            [a[..., keep], a[..., hidden].sum(axis=-1, keepdims=True)], axis=-1))
    return new_names, new_arrays


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
    parser.add_argument("--proportion", action="store_true",
                        help="Plot each state's fraction of the cells alive at "
                             "that time (the stack sums to 1 at every time) "
                             "instead of expected counts.")
    parser.add_argument("--mark_times", type=float, nargs="*", default=None,
                        help="Times to mark with a dashed vertical line and a "
                             "bold axis tick (e.g. --mark_times 7.5 8.5).")
    parser.add_argument("--lump_hidden", action="store_true",
                        help="Merge the inferred unobserved states (named U<idx> "
                             "by the pipeline) into one band labelled "
                             f"'{HIDDEN_LABEL}'.")
    parser.add_argument("--xlabel", default="Time",
                        help="x-axis label (default: 'Time').")
    parser.add_argument("--observed_time", type=float, default=None,
                        help="Time at which to take the Expected comparison bar, "
                             "i.e. the sampling time of --trees. Default: -T. Set "
                             "this when the horizon runs past the trees' leaf time.")
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

    obs = None
    if args.trees is not None:
        obs = observed_counts(args.trees, state_names)

    # The Expected comparison bar is taken at the trees' sampling time, which is
    # not the end of the horizon when the horizon extrapolates past the leaves.
    obs_time = args.time if args.observed_time is None else args.observed_time
    exp_at_obs_time = population_counts(B, growth_rates, n0, [obs_time])[0]

    # Merge the hidden states before coloring, so the merged band gets one color
    # and every downstream array is indexed by the same state list.
    if args.lump_hidden:
        state_names, (counts, exp_at_obs_time, obs) = lump_hidden(
            state_names, (counts, exp_at_obs_time, obs))
        K = len(state_names)

    colors = assign_colors(state_names)

    if args.proportion:
        # Normalize each time slice, so the stack is the composition of the
        # cells alive at that time and sums to 1 everywhere.
        series = counts / counts.sum(axis=1, keepdims=True)
        exp_bar = exp_at_obs_time / exp_at_obs_time.sum()
        obs_bar = None if obs is None else obs / obs.sum()
        y_max = 1.0
        y_label = "Fraction of cells"
        y_title = "Expected composition"
    else:
        # Rescale expected counts so the expected total at the trees' sampling
        # time equals the observed leaf total; this makes the Expected/Observed
        # bars directly comparable (both stacks reach the same height).
        series, exp_bar, obs_bar = counts, exp_at_obs_time, obs
        if obs is not None and exp_at_obs_time.sum() > 0 and obs.sum() > 0:
            scale = obs.sum() / exp_at_obs_time.sum()
            series = series * scale
            exp_bar = exp_bar * scale
        y_max = series.sum(axis=1).max() * 1.05
        y_label = "Expected population count"
        y_title = "Expected counts"

    # Stacked area: each type is a filled polygon between the cumulative values
    # of the types stacked below and including it.
    fig, ax = plt.subplots(figsize=(8, 5))
    cum = np.concatenate([np.zeros((len(times), 1)), np.cumsum(series, axis=1)], axis=1)
    for i in range(K):
        ax.fill_between(times, cum[:, i], cum[:, i + 1],
                        label=state_names[i], color=colors[i], linewidth=0.0)

    span = times[-1] - times[0]
    marks = sorted({t for t in (args.mark_times or []) if times[0] <= t <= times[-1]})
    for t in marks:
        ax.axvline(t, color="black", linestyle="--", linewidth=1.3, zorder=5)

    # Round tick values (0, 2, 4, ...) rather than an even split of the horizon,
    # minus any that would land on a marked time (the marks always get their own
    # tick) or crowd the comparison bars.
    locator = mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
    base_ticks = [t for t in locator.tick_values(times[0], times[-1])
                  if times[0] - 1e-9 <= t <= times[-1] + 1e-9]
    if obs_bar is not None:
        base_ticks = [t for t in base_ticks if t < times[-1] - 1e-9]
    base_ticks = [t for t in base_ticks
                  if all(abs(t - m) > 0.1 * span for m in marks)]
    time_ticks = sorted(base_ticks + marks)

    # Comparison bars on the right: expected composition at the trees' sampling
    # time (solid) next to the observed leaf composition (white-hatched).
    if obs_bar is not None:
        gap = span * 0.11        # polygon -> first bar
        bar_w = span * 0.09
        bar_gap = span * 0.06    # between the two bars
        exp_center = times[-1] + gap + bar_w / 2
        obs_center = exp_center + bar_w + bar_gap

        def _stack(center, values, hatch=None):
            bottom = 0.0
            for i in range(K):
                ax.bar(center, values[i], width=bar_w, bottom=bottom,
                       color=colors[i], edgecolor="white", linewidth=0.0,
                       hatch=hatch)
                bottom += values[i]

        _stack(exp_center, exp_bar)
        _stack(obs_center, obs_bar, hatch="////")

        ax.text((exp_center + obs_center) / 2, y_max * 1.02, f"t = {obs_time:g}",
                ha="center", va="bottom", fontsize=13)

        ax.set_xticks(list(time_ticks) + [exp_center, obs_center])
        labels = ax.set_xticklabels(
            [f"{t:g}" for t in time_ticks] + ["Expected", "Observed"])
        for lab in labels[-2:]:  # angle the two bar labels so they don't collide
            lab.set_rotation(30)
            lab.set_ha("right")
        ax.set_xlim(times[0], obs_center + bar_w / 2)
    else:
        ax.set_xticks(time_ticks)
        labels = ax.set_xticklabels([f"{t:g}" for t in time_ticks])
        ax.set_xlim(times[0], times[-1])

    # Bold the marked ticks so they read as the times of interest.
    for lab, t in zip(labels, time_ticks):
        if any(abs(t - m) < 1e-9 for m in marks):
            lab.set_fontweight("bold")

    ax.set_ylim(0, y_max)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(y_label)
    ax.set_title(y_title, loc="left")
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

    # Report the composition at the marked times and at the horizon, which is
    # what the figure is read for.
    report_times = sorted({*marks, times[-1]})
    report = population_counts(B, growth_rates, n0, report_times)
    if args.lump_hidden:
        _, (report,) = lump_hidden([str(idx2state[i]) for i in range(len(idx2state))],
                                   (report,))
    fracs = report / report.sum(axis=1, keepdims=True)
    width = max(len(n) for n in state_names)
    header = "".join(f"{t:>12g}" for t in report_times)
    print(f"\nComposition (fraction of cells alive at each time):")
    print(f"{'state':<{width}}{header}")
    for i in range(K):
        row = "".join(f"{fracs[t_idx, i]:>12.4f}"
                      for t_idx in range(len(report_times)))
        print(f"{state_names[i]:<{width}}{row}")
    if obs_bar is not None:
        obs_frac = obs / obs.sum()
        print(f"\nObserved leaf composition at t={obs_time:g} "
              f"({int(obs.sum())} leaves):")
        for i in range(K):
            print(f"  {state_names[i]:<{width}} {obs_frac[i]:.4f}")


if __name__ == "__main__":
    main()
