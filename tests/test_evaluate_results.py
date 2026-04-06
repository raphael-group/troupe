import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from evaluate_results import _build_plot_state2potency


def test_build_plot_state2potency_uses_saved_label_potencies():
    idx2state = {
        0: "NeuralTube",
        1: "Somite",
        2: "Endoderm",
        3: "PCGLC",
        4: "NMPs",
        5: "U5",
    }
    idx2potency = {
        0: ("NeuralTube",),
        1: ("Somite",),
        2: ("Endoderm",),
        3: ("PCGLC",),
        4: ("NeuralTube", "Somite"),
        5: ("Endoderm", "NeuralTube", "PCGLC", "Somite"),
    }
    terminal_idxs = [0, 1, 2, 3]

    result = _build_plot_state2potency(idx2state, idx2potency, terminal_idxs)

    assert result[3] == (3,)
    assert result[4] == (0, 1)
    assert result[5] == (0, 1, 2, 3)


def test_build_plot_state2potency_accepts_integer_potencies():
    idx2state = {0: "A", 1: "B", 2: "U2"}
    idx2potency = {
        0: (0,),
        1: (1,),
        2: (0, 1),
    }
    terminal_idxs = [0, 1]

    result = _build_plot_state2potency(idx2state, idx2potency, terminal_idxs)

    assert result == {0: (0,), 1: (1,), 2: (0, 1)}
