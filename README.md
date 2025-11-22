# Tree-informed Rate Optimization Using Potency Enforcement (TROUPE)

This reposiotry is a part of an in-review paper titled *Inferring Cell Differentiation Dynamics with Unobserved Progenitors* by William Howard-Snyder, Richard Zhang, Henri Schmidt, Michelle Chan, and Ben Raphael.
It contains our implementation for Tree-informed Rate Optimization Using Potency Enforcement (TROUPE) as well as code to simulate data and run experiments.


# Dependencies
To run our code please clone this repository and `cd` into it.
Install dependencies via

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -e .


# Example method usage

To run our method on one trial of the simulated data (with 32 trees and only 5 out of 9 states) run the following command:

    bash scripts/troupe_inference_example.sh

This produces a file at `troupe/example/results/reg=0/select_potencies/model_dict.pkl` for each regularization value in 0, 0.001, 0.003, 0.01, 0.03, ..., 10, 30.
Each of these files stores the debiased transition and growth rates at the given regularization value.

To plot the pareto front you can run

    python scripts/evaluate_results.py plot-knees \
        -r troupe/example/results \
        -t troupe/example/data/trees.pkl

TROUPE identifies that the number of states in the process is 9 (see the red star in `troupe/example/results/figures/loss_vs_num_states.pdf`), which corresponds to a regularization strength of 1.

To visualize the infered rates inferred at this pareto front, you can run

    python scripts/evaluate_results.py draw-graph \
        -r troupe/example/results/reg=1/select_potencies \
        -p troupe/example/results/reg=1

which saves the differentiation map to `troupe/example/results/reg=0.1/select_potencies/figures/inferred_graph.pdf`.
You can compare these results to the ground truth parameters (see `troupe/example/data/ground_truth_diff_map.png`) and notice that they are close to the inferred rates.


# Experiments

We also provide our code used to run TROUPE and SSE for the exeriments (simulated and TLS data) from our paper.
Note that these scripts assume access to a slurm cluster.

## Simulated data experiments (Section 4.1)

Scripts for running TROUPE and SSE are located in `troupe/scripts/TLS_experiment`.

## TLS experiments (Section 4.2)

Scripts for running TROUPE and SSE are located in `troupe/scripts/sample_efficiency_experiment`.
