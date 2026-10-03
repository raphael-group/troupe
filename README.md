# Tree-informed Rate Optimization Using Potency Enforcement (TROUPE)


This reposiotry is a part of an in-review paper titled *Inferring Cell Differentiation Dynamics with Unobserved Progenitors* by William Howard-Snyder, Richard Zhang, Henri Schmidt, Michelle Chan, and Ben Raphael.
It contains our implementation for Tree-informed Rate Optimization Using Potency Enforcement (TROUPE) as well as code to simulate data and run experiments.


# Installation

Clone this repository, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> **GPU support:** the above installs the CPU build of PyTorch. For CUDA, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the appropriate build before running `pip install -r requirements.txt`.


# Example method usage

To run our method on one trial of the simulated data (with 32 trees and only 5 states observed out of 9 total states) run the following command:

    bash scripts/troupe_inference_example.sh

This produces a file at `troupe/example/results/reg=0/select_potencies/model_dict.pkl` for each regularization value among 0, 0.001, 0.003, 0.01, 0.03, ..., 10, 30.
Each of these files stores the inferred transition and growth rates at the given regularization value.
The script also generates a summary of the inferred results at `troupe/example/results/troupe_summary.txt`, which should look something like

    best_reg	0.1
    best_model_dir	path/to/inferred/params
    knee_num_states	9
    knee_loss	1240.465220608524
    num_regularizations_tested	10

Note that TROUPE correcly infers that the total number of states in the process is 9.

To plot the differentiation maps for each regularization value run

    BASE_DIR=path/to/troupe;
    python scripts/evaluate_results.py plot-differentiation-maps \
        -i $BASE_DIR/example/results

you can view the differentiation map at `troupe/example/results/reg=0.1/select_potencies/figures/inferred_graph.pdf`.
You can compare these results to the ground truth parameters (see `troupe/example/data/ground_truth_diff_map.png`) and notice that they are close to the inferred rates.


# Likelihood Ratio Testing

TROUPE includes a parametric bootstrap likelihood ratio test (LRT) for evaluating whether specific edges in the inferred differentiation map are statistically significant. The test compares an unconstrained model against a null model that enforces a user-specified constraint on the birth kernel, then calibrates the rejection threshold via simulation from the null -- avoiding the standard chi-squared approximation, which is invalid when the MLE lies on a parameter boundary.

## Example: NMPs preferentially produce NeuralTube over Somite (TLSC data)

This test asks whether the NMP to NeuralTube birth probability is at least 2.5x larger than NMP to Somite.

**Prerequisites:** fit the TLSC ClaSSE-TROUPE model first (from the repo root):

```bash
bash scripts/TLS_experiment_classe/run_troupe_on_tls.bash TLSC
```

This writes the best model to `results/TLSC_group_lasso/sample_0.05/best_model_dict.pkl`. Then run the LRT:

```bash
python scripts/run_lrt.py \
    -i experiments/TLSC/processed_data/trees.pkl \
    --model results/TLSC_group_lasso/sample_0.05/best_model_dict.pkl \
    --constraint "B[NMPs,NeuralTube] >= 5.5 * B[NMPs,Somite]" \
    --sampling_prob 0.05 \
    --B 99 \
    --output experiments/TLSC_group_lasso/lrt/nmp_neuraltube_somite
```

Results are written to `experiments/TLSC_group_lasso/lrt/nmp_neuraltube_somite/`:

| File | Description |
|------|-------------|
| `lrt_results.json` | Observed Gamma, bootstrap statistics, p-value, decision |
| `bootstrap_distribution.pdf` | Histogram of bootstrap Gamma values with observed Gamma marked |
| `observed/` | Unconstrained and null model fits on the real data |

The console will print a summary:

```
============================================================
  LIKELIHOOD RATIO TEST RESULT
============================================================
  H0: B[NMPs,NeuralTube] >= 2.5 * B[NMPs,Somite]

  Observed LRT statistic:  <value>
  Log-lik unconstrained:   <value>
  Log-lik null:            <value>

  Bootstrap replicates:    99
  p-value:                 <value>
  Critical value (alpha=0.05): <value>
  Decision:                REJECT H0 / FAIL TO REJECT H0
============================================================
```

**Interpreting the result:** rejecting H0 means the data are inconsistent with the null constraint — i.e., the observed asymmetry between NeuralTube and Somite production exceeds what would be expected by chance under a model where B[NMPs,NeuralTube] >= 2.5 x B[NMPs,Somite].

## Running on a cluster (SLURM)

With B=99 replicates, each requiring two full model fits, the sequential version can take hours. `run_lrt_cluster.py` distributes the bootstrap inference jobs as a SLURM job array so all B replicates run in parallel.

**Step 1 — prepare** (run locally, ~minutes): computes the observed LRT statistic, simulates all B bootstrap datasets, and writes a ready-to-submit SLURM job-array script.

```bash
python scripts/run_lrt_cluster.py prepare \
    -i experiments/TLSC/processed_data/trees.pkl \
    --model results/TLSC/sample_0.05/best_model_dict.pkl \
    --constraint "B[NMPs,NeuralTube] >= 2.5 * B[NMPs,Somite]" \
    --sampling_prob 0.05 \
    --B 99 \
    --output experiments/TLSC/lrt/nmp_neuraltube_somite \
    --slurm_time 2:00:00 \
    --slurm_mem 8G
```

**Step 2 — submit:**

```bash
sbatch experiments/TLSC/lrt/nmp_neuraltube_somite/slurm_submit.sh
```

Pass `--submit` to the prepare step to combine steps 1 and 2. Each array task runs `lrt_worker.py` for one replicate and writes its result to `work/bootstrap_results/XXXX.json`.

**Step 3 — aggregate** (run locally after all jobs finish, seconds):

```bash
python scripts/run_lrt_cluster.py aggregate \
    --output experiments/TLSC/lrt/nmp_neuraltube_somite
```

This reads all per-replicate results, computes the empirical p-value, prints the same summary table as the sequential script, and writes `lrt_results.json` and `bootstrap_distribution.pdf`. If some jobs failed, a warning is printed and the p-value is computed from the completed replicates.

SLURM options (all have sensible defaults):

| Flag | Default | Description |
|------|---------|-------------|
| `--slurm_time` | `2:00:00` | Wall-clock limit per job |
| `--slurm_mem` | `8G` | Memory per job |
| `--slurm_cpus` | `1` | CPUs per task |
| `--slurm_partition` | *(none)* | SLURM partition |
| `--venv` | `<repo_root>/.venv` | Path to virtual environment |


## General constraint syntax

Two constraint forms are supported via `--constraint`:

```
# Ratio: one kernel entry must be at least k times another
--constraint "B[StateA,StateB] >= k * B[StateC,StateD]"

# Lower bound: one kernel entry must exceed a fixed value
--constraint "B[StateA,StateB] >= c"
```

Multiple `--constraint` flags can be combined to test joint hypotheses. State names must match those in the fitted model's `idx2state` mapping (visible in the terminal labels file for each experiment).


# Experiments

We also provide the code we used to run TROUPE and SSE for the exeriments (simulated and TLS data) from our paper.
Note that these scripts are mainly for reference for how we conducted our experiments and are not intended to be maintained.

## Simulated data experiments (Section 4.1)

Scripts for running TROUPE and SSE are located in `troupe/scripts/TLS_experiment`.

## TLS experiments (Section 4.2)

Scripts for running TROUPE and SSE are located in `troupe/scripts/sample_efficiency_experiment`.
