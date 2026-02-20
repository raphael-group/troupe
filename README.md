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


# Experiments

We also provide the code we used to run TROUPE and SSE for the exeriments (simulated and TLS data) from our paper.
Note that these scripts are mainly for reference for how we conducted our experiments and are not intended to be maintained.

## Simulated data experiments (Section 4.1)

Scripts for running TROUPE and SSE are located in `troupe/scripts/TLS_experiment`.

## TLS experiments (Section 4.2)

Scripts for running TROUPE and SSE are located in `troupe/scripts/sample_efficiency_experiment`.
