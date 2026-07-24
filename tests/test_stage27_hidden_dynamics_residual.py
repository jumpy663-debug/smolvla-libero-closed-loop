from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_stage27():
    path = Path(__file__).parents[1] / "scripts" / "wm" / "stage27_hidden_dynamics_residual.py"
    scripts_path = str(path.parent)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location(
        "stage27_hidden_dynamics_residual_for_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE27 = load_stage27()
ROOT = Path(__file__).parents[1]


def test_influenced_windows_have_fixed_causal_semantics() -> None:
    assert tuple(range(10, 41)) == STAGE27.PRE_STARTS
    assert tuple(range(41, 51)) == STAGE27.INFLUENCED_STARTS
    assert STAGE27.INFLUENCED_STARTS[0] + STAGE27.HORIZON == 51
    assert STAGE27.INFLUENCED_STARTS[-1] + STAGE27.HORIZON == 60


def test_standardization_uses_only_pre_values() -> None:
    pre = np.linspace(1.0, 2.0, len(STAGE27.PRE_STARTS))
    first = np.concatenate([pre, np.full(10, 4.0)])
    second = np.concatenate([pre, np.full(10, 400.0)])
    first_anomaly, first_calibration = STAGE27.standardize_anomaly(first)
    second_anomaly, second_calibration = STAGE27.standardize_anomaly(second)
    assert first_calibration == second_calibration
    assert np.array_equal(
        first_anomaly[: len(STAGE27.PRE_STARTS)],
        second_anomaly[: len(STAGE27.PRE_STARTS)],
    )
    assert second_anomaly[-1] > first_anomaly[-1]


def test_gate_and_dominance_require_complete_pair_evidence() -> None:
    primary = {
        "mean": 2.0,
        "positive_pairs": 5,
        "exact_sign_flip_p": 0.03125,
    }
    weak = {
        "mean": 1.0,
        "positive_pairs": 5,
        "exact_sign_flip_p": 0.03125,
    }
    dominant = {
        "mean": 3.0,
        "positive_pairs": 6,
        "exact_sign_flip_p": 0.015625,
    }
    assert STAGE27.passes_pair_gate(primary)
    assert not STAGE27.baseline_dominates(primary, weak)
    assert STAGE27.baseline_dominates(primary, dominant)


def test_holm_adjustment_controls_exploratory_family() -> None:
    adjusted = STAGE27.holm_adjust(
        {
            "best": 0.015625,
            "second": 0.078125,
            "third": 0.15625,
            "fourth": 0.25,
            "fifth": 0.28125,
        }
    )
    assert adjusted["best"] == 0.078125
    assert all(0 <= value <= 1 for value in adjusted.values())


def test_public_stage27_result_preserves_discovery_boundary() -> None:
    result_dir = ROOT / "results" / "wm" / "stage27_hidden_dynamics_residual"
    report = json.loads((result_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["scope"]["pairs"] == 6
    assert report["scope"]["episodes"] == 12
    assert not report["conclusion"]["primary_pair_gate_passed"]
    assert report["statistics"]["wm_relative_residual"]["positive_pairs"] == 4
    assert report["statistics"]["no_action_residual"]["positive_pairs"] == 6
    assert report["conclusion"]["exploratory_best_metric"] == ("no_action_residual")
    assert report["conclusion"]["exploratory_best_metric_holm_p"] == (0.078125)
    assert not report["conclusion"]["exploratory_best_survives_holm_0_05"]
    assert not report["negative_controls"]["stage25_held_out_test_states_accessed"]
    assert report["negative_controls"]["action_interface_mismatch_steps_total"] == 0

    with (result_dir / "window_scores.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 492

    for key in (
        "protocol",
        "feature_manifest",
        "calibration",
        "window_scores",
        "pair_effects",
        "statistics",
    ):
        path = result_dir / report["artifacts"][key]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == report["artifacts"][f"{key}_sha256"]
