import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from ete3 import Tree

from run_classe_troupe import (
    _birth_kernel_regularization,
    _build_phase2_label_maps,
    _collect_phase2_kept_states,
    _discover_regularizations_from_output_dir,
    _infer_support_graph_potencies_classe,
    _write_model_selection_outputs,
)


def test_phase2_preserves_observed_nonterminal_label_aliases():
    ordered_potencies = [
        ("Endoderm",),
        ("NeuralTube",),
        ("Somite",),
        ("NeuralTube", "Somite"),
    ]
    terminal_labels = ["NeuralTube", "Somite", "Endoderm", "PCGLC"]
    observed_potencies = {
        "NeuralTube": ("NeuralTube",),
        "Somite": ("Somite",),
        "Endoderm": ("Endoderm",),
        "PCGLC": ("PCGLC",),
        "NMPs": ("NeuralTube", "Somite"),
    }

    idx2potency, newidx2state, state2newidx = _build_phase2_label_maps(
        ordered_potencies, terminal_labels, observed_potencies
    )

    assert idx2potency[3] == ("NeuralTube", "Somite")
    assert newidx2state[3] == "NMPs"
    assert state2newidx["NMPs"] == 3
    assert state2newidx["NeuralTube"] == 1
    assert state2newidx["Somite"] == 2


def test_phase2_force_keeps_observed_states_present_in_trees():
    tree = Tree("(a,b);", format=1)
    leaves = tree.get_leaves()
    leaves[0].add_feature("state", "PCGLC")
    leaves[1].add_feature("state", "Somite")

    idx2state = {
        0: "Somite",
        1: "PCGLC",
        2: "U2",
    }
    reachable_idxs = [0, 2]
    inferred_idx2potency = {
        0: (0,),
        1: (),
        2: (0,),
    }
    terminal_labels = ["Somite", "PCGLC"]
    observed_potencies = {
        "Somite": ("Somite",),
        "PCGLC": ("PCGLC",),
    }

    kept_old_idxs, dropped_old_idxs, old_idx2potency, mandatory_old_idxs = _collect_phase2_kept_states(
        [tree],
        idx2state,
        reachable_idxs,
        inferred_idx2potency,
        terminal_labels,
        observed_potencies,
    )

    assert mandatory_old_idxs == [0, 1]
    assert kept_old_idxs == [0, 1, 2]
    assert dropped_old_idxs == []
    assert old_idx2potency[1] == ("PCGLC",)


def test_support_graph_potencies_keep_rare_observed_state():
    idx2state = {
        0: "NeuralTube",
        1: "Somite",
        2: "PCGLC",
        3: "U3",
    }
    observed_potencies = {
        "NeuralTube": ("NeuralTube",),
        "Somite": ("Somite",),
        "PCGLC": ("PCGLC",),
    }
    B = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.7, 0.2999999999, 1e-10, 0.0],
    ])

    potency_map, support, forced_edges = _infer_support_graph_potencies_classe(
        B,
        idx2state,
        observed_potencies,
        abs_eps=1e-8,
        rel_eps=1e-3,
        top_k=2,
    )

    assert potency_map[2] == ("PCGLC",)
    assert potency_map[3] == ("NeuralTube", "PCGLC", "Somite")
    assert support[3, 2]
    assert forced_edges == [(3, 2, 1e-10)]


def test_support_graph_potencies_respect_observed_nonterminal_anchor():
    idx2state = {
        0: "NeuralTube",
        1: "Somite",
        2: "NMPs",
        3: "U3",
    }
    observed_potencies = {
        "NeuralTube": ("NeuralTube",),
        "Somite": ("Somite",),
        "NMPs": ("NeuralTube", "Somite"),
    }
    B = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.4, 0.6, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    potency_map, support, forced_edges = _infer_support_graph_potencies_classe(
        B,
        idx2state,
        observed_potencies,
        abs_eps=1e-8,
        rel_eps=1e-3,
        top_k=2,
    )

    assert potency_map[2] == ("NeuralTube", "Somite")
    assert potency_map[3] == ("NeuralTube", "Somite")
    assert support[3, 2]
    assert forced_edges == []


def test_birth_kernel_regularization_supports_column_group_lasso():
    B = torch.tensor(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.8, 0.0],
            [0.0, 0.4, 0.6],
        ],
        dtype=torch.float64,
    )
    support_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [False, True, True],
        ],
        dtype=torch.bool,
    )

    l1_penalty = _birth_kernel_regularization(B, regularization_type="l1")
    group_penalty = _birth_kernel_regularization(
        B,
        regularization_type="column_group_lasso",
        support_mask=support_mask,
    )

    expected_l1 = torch.tensor(1.0, dtype=torch.float64)
    expected_group = torch.tensor(
        0.2 + np.sqrt(2.0) * 0.5 + 0.1,
        dtype=torch.float64,
    )

    assert torch.isclose(l1_penalty, expected_l1)
    assert torch.isclose(group_penalty, expected_group)


def test_discover_regularizations_from_output_dir(tmp_path):
    (tmp_path / "reg=0.001").mkdir()
    (tmp_path / "reg=0.03").mkdir()
    (tmp_path / "reg=10").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "reg=abc").mkdir()

    regs = _discover_regularizations_from_output_dir(str(tmp_path))

    assert regs == [0.001, 0.03, 10.0]


def test_write_model_selection_outputs_writes_pareto_front(tmp_path):
    best_model_dir = tmp_path / "reg=0.1" / "select_potencies"
    best_model_dir.mkdir(parents=True)
    (best_model_dir / "model_dict.pkl").write_bytes(b"dummy")

    selection = {
        "knee_num_states": 6,
        "knee_loss": 12.5,
        "best_reg": 0.1,
        "best_model_dir": str(best_model_dir),
        "all_results": [
            {"reg": 1.0, "num_states": 4, "neg_llh": 14.0, "model_dir": "a"},
            {"reg": 0.3, "num_states": 6, "neg_llh": 12.5, "model_dir": "b"},
            {"reg": 0.1, "num_states": 6, "neg_llh": 12.0, "model_dir": str(best_model_dir)},
        ],
    }

    _write_model_selection_outputs(str(tmp_path), selection, sampling_probability=0.1)

    assert (tmp_path / "best_model_dict.pkl").is_file()
    assert (tmp_path / "classe_troupe_summary.txt").is_file()
    assert (tmp_path / "pareto_front.pdf").is_file()
