#!/usr/bin/env python
"""Print a model_dict.pkl as human-readable JSON.

Usage:
    python scripts/modeldict2json.py results/TLS_no_endothelial/sample_0.05/reg=2.5/select_potencies
    python scripts/modeldict2json.py results/TLSC/sample_0.05/best_model_dict.pkl
    python scripts/modeldict2json.py results/TLSC/sample_0.05/reg=10.0/select_potencies/ -o model.json
"""

import argparse
import json
import os
import pickle
import sys


def load_model_dict(path: str) -> dict:
    if path.endswith(".pkl"):
        pkl_path = path
    else:
        pkl_path = os.path.join(path, "model_dict.pkl")
    if not os.path.exists(pkl_path):
        sys.exit(f"Error: {pkl_path} not found.")
    with open(pkl_path, "rb") as fp:
        return pickle.load(fp)


def model_dict_to_json(model_dict: dict, precision: int = 6) -> dict:
    idx2state = model_dict["idx2state"]
    idx2potency = model_dict["idx2potency"]
    n = model_dict.get("n_states", len(idx2state))
    start_idx = model_dict.get("start_state")

    states = [idx2state[i] for i in range(n)]

    # Birth kernel: labeled rows and columns.
    B = model_dict["daughter_kernel"].detach().cpu()
    birth_kernel = {
        idx2state[i]: {
            idx2state[j]: round(float(B[i, j]), precision)
            for j in range(n)
        }
        for i in range(n)
    }

    # Growth rates: labeled by state.
    lam = model_dict["growth_rates"].detach().cpu()
    growth_rates = {idx2state[i]: round(float(lam[i]), precision) for i in range(n)}

    # Initial (root) distribution: labeled by state.
    pi = model_dict["root_distribution"].detach().cpu()
    initial_distribution = {idx2state[i]: round(float(pi[i]), precision) for i in range(n)}

    # Potency sets: map state name -> sorted list of terminal state names.
    potency = {
        idx2state[i]: sorted(pot)
        for i, pot in idx2potency.items()
    }

    return {
        "states": states,
        "birth_kernel": birth_kernel,
        "growth_rates": growth_rates,
        "initial_distribution": initial_distribution,
        "potency": potency,
        "start_state": idx2state[start_idx] if start_idx is not None else None,
        "sampling_probability": float(model_dict.get(
            "sampling_prob_float",
            model_dict.get("sampling_probability", None)
        )),
        "n_states": n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert a model_dict.pkl to human-readable JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("path", help="Directory containing model_dict.pkl, or path to .pkl directly.")
    # parser.add_argument("-o", "--output", default=None,
    #                     help="Write JSON to this file instead of stdout.")
    parser.add_argument("--precision", type=int, default=6,
                        help="Decimal places for floating-point values (default: 6).")
    args = parser.parse_args()

    model_dict = load_model_dict(args.path)
    out = model_dict_to_json(model_dict, precision=args.precision)
    text = json.dumps(out, indent=2)

    # if args.output:
    #     with open(args.output, "w") as fp:
    #         fp.write(text + "\n")
    #     print(f"Written to {args.output}")
    # else:
    #     print(text)
    with open(f"{args.path}/model.json", "w") as fp:
        fp.write(text + "\n")
    print(f"Written to {args.path}/model.json")


if __name__ == "__main__":
    main()
