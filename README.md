# Tree-informed Rate Optimization Using Potency Enforcement (TROUPE)


This repository is a part of an in-review paper titled *Inferring Cell Differentiation Dynamics with Unobserved Progenitors* by William Howard-Snyder, Richard Zhang, Henri Schmidt, Michelle Chan, and Ben Raphael.
It contains our implementation for Tree-informed Rate Optimization Using Potency Enforcement (TROUPE) as well as code to simulate data and run experiments.


# Installation

Clone this repository, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Plotting differentiation maps also requires the [Graphviz](https://graphviz.org/download/) system package (`brew install graphviz` on macOS, `sudo apt-get install graphviz` on Ubuntu/Debian, or `conda install graphviz`).

> **GPU support:** the above installs the CPU build of PyTorch. For CUDA, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the appropriate build before running `pip install -r requirements.txt`.


# Example: running TROUPE on the TLS data

The `example/TLS/` folder contains everything needed to reproduce the TROUPE analysis of the trunk-like structure (TLSC) lineage trees from the paper:

| File | Description |
|------|-------------|
| `example/TLS/data/trees.pkl` | TLSC lineage trees (pickled list of ete3 trees; each leaf has a `.state` cell-type label) |
| `example/TLS/data/terminal_labels.txt` | The terminal (fully differentiated) cell types: NeuralTube, Somite, Endoderm, PCGLC |
| `example/TLS/data/observed_potencies.txt` | Potency of each observed cell type (e.g. the observed NMPs can produce NeuralTube and Somite) |
| `example/TLS/run_example.sh` | Runs TROUPE (`scripts/run_classe_troupe.py`) over several regularization strengths and plots the results |

After installing TROUPE, activate the virtual environment and run the example from the repository root:

```bash
source .venv/bin/activate
bash example/TLS/run_example.sh
```

This runs TROUPE at the regularization strengths 10, 1, and 0.1, with a leaf sampling probability of 0.05.
For each regularization strength, TROUPE
1. fits an overparameterized model with every candidate progenitor state, using an L1 penalty to make the differentiation map sparse (Phase 1);
2. extracts the progenitor states that remain in the sparse model and re-fits the reduced model without the strong penalty (Phase 2).

TROUPE then picks the regularization strength at the knee of the negative log-likelihood vs. number-of-states curve (model selection).
To try a subset of regularization strengths, pass them as arguments, e.g. `bash example/TLS/run_example.sh 10 3 1 0.3`.

## Output

Results are written to `example/TLS/results/`:

| Path | Description |
|------|-------------|
| `classe_troupe_summary.txt` | Selected regularization strength, number of states, and negative log-likelihood for every regularization strength |
| `best_model_dict.pkl` | Parameters of the selected model |
| `pareto_front.pdf` | Negative log-likelihood vs. number of states, with the selected model marked |
| `reg=<value>/` | Phase 1 model for that regularization strength |
| `reg=<value>/select_potencies/model_dict.pkl` | Phase 2 (re-fit) model for that regularization strength |
| `reg=<value>/select_potencies/figures/differentiation_map.pdf` | Inferred differentiation map |
| `reg=<value>/select_potencies/figures/growth_rate_bars.pdf` | Inferred per-state growth rates |

Each `model_dict.pkl` is a pickled dict with the following entries:

| Key | Description |
|-----|-------------|
| `daughter_kernel` | Row-stochastic birth kernel B: `B[i, j]` is the probability that a division of a type-i cell produces a type-j daughter |
| `growth_rates` | Division rate of each state |
| `root_distribution` | Distribution of the root cell's state |
| `idx2state` | State index to state name; inferred progenitor states are given new names |
| `idx2potency` | State index to the set of terminal types the state can produce |

`classe_troupe_summary.txt` should look like this (exact values may vary slightly across platforms and PyTorch versions):

```
best_reg	3.0
best_model_dir	/path/to/troupe/example/TLS/results/reg=3.0/select_potencies
knee_num_states	10
knee_loss	901.6692406255844
sampling_probability	0.05
num_regularizations_tested	6

All results:
reg	num_states	neg_llh
10.0	7	1282.4999852950345
3.0	10	901.6692406255844
1.0	12	923.9256324947544
0.3	19	1005.8348146577059
0.1	20	1007.2728760573672
0.03	15	1086.2475528889943
```

To load the selected model in Python:

```python
import pickle
with open("example/TLS/results/best_model_dict.pkl", "rb") as f:
    model = pickle.load(f)
print(model["idx2state"])
print(model["idx2potency"])
print(model["daughter_kernel"])
```

## Running on your own data

`scripts/run_classe_troupe.py` takes the same inputs for any dataset.
Your trees must be ultrametric, with each leaf's cell type stored in its `.state` attribute.
Pass them as a pickled list of ete3 trees or as a Newick file with one tree per line.
Run `python scripts/run_classe_troupe.py --help` for all options.
For example:

```bash
python scripts/run_classe_troupe.py \
    -i path/to/trees.pkl \
    -o path/to/results \
    --terminal_labels path/to/terminal_labels.txt \
    --observed_potencies path/to/observed_potencies.txt \
    --sampling_probability 0.05 \
    --regularizations 10 1 0.1
```

`--terminal_labels` and `--observed_potencies` are optional.
If you omit them, every observed cell type is treated as terminal.
`terminal_labels.txt` lists one terminal cell type per line.
`observed_potencies.txt` has one line per observed cell type: the type name, a tab, then a comma-separated list of the terminal types it can produce.
To plot differentiation maps for every regularization strength, run

```bash
python scripts/evaluate_results.py plot-differentiation-maps -i path/to/results
```

# Simulation experiments

Software for simulating data and benchmarking other methods (e.g., ICE-FASE and Carta) can be found [at this repo](https://github.com/raphael-group/troupe-experiments).
Please navigate to that repository and follow the instructions there to replicate our results.