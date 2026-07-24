from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def load_stage23():
    path = Path(__file__).parents[1] / "scripts" / "wm" / "stage23_confirmatory_wm_scoring.py"
    scripts_path = str(path.parent)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location(
        "stage23_confirmatory_wm_scoring",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE23 = load_stage23()


def test_registered_windows_and_primary_unit_are_fixed() -> None:
    assert tuple(range(10, 40)) == STAGE23.PRE_STARTS
    assert tuple(range(50, 80)) == STAGE23.POST_STARTS
    valid = torch.ones(89, dtype=torch.bool)
    selected = STAGE23.validate_window_starts(valid, STAGE23.POST_STARTS)
    assert torch.equal(selected, torch.arange(50, 80))


def test_one_sided_exact_sign_flip_has_six_pair_resolution() -> None:
    result = STAGE23.exact_sign_flip(
        np.ones(6),
        alternative="greater",
    )
    assert result["sign_assignments"] == 64
    assert result["p_value"] == pytest.approx(1 / 64)
    two_sided = STAGE23.exact_sign_flip(
        np.ones(6),
        alternative="two-sided",
    )
    assert two_sided["p_value"] == pytest.approx(2 / 64)


def test_primary_averages_interventions_before_inference() -> None:
    period_rows = []
    for condition in STAGE23.CONDITIONS:
        for period in ("pre", "post"):
            commanded = 1.0
            executed = 1.0
            if condition == "motion_attenuation_0p5" and period == "post":
                commanded = 1.4
            if condition == "action_delay_3" and period == "post":
                commanded = 1.2
            period_rows.append(
                {
                    "pair_id": "task04-init07",
                    "condition": condition,
                    "period": period,
                    "commanded_h10_raw_mse": commanded,
                    "executed_h10_raw_mse": executed,
                }
            )
    stage22_pairs = [
        {
            "pair_id": "task04-init07",
            "benchmark_task_id": "4",
            "init_state_index": "7",
            "motion_attenuation_0p5_post_pixel_mae_mean": "2",
            "motion_attenuation_0p5_post_eef_motion_l2_mean": "0.1",
            "action_delay_3_post_pixel_mae_mean": "4",
            "action_delay_3_post_eef_motion_l2_mean": "0.2",
        }
    ]
    result = STAGE23.build_pair_confirmation_rows(
        period_rows,
        stage22_pairs,
    )
    assert len(result) == 1
    assert result[0]["motion_attenuation_0p5_post_commanded_minus_executed"] == pytest.approx(0.4)
    assert result[0]["action_delay_3_post_commanded_minus_executed"] == pytest.approx(0.2)
    assert result[0]["primary_mean_across_interventions"] == pytest.approx(0.3)


def test_bootstrap_is_deterministic() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    first = STAGE23.bootstrap_ci(values, seed=2323)
    second = STAGE23.bootstrap_ci(values, seed=2323)
    assert first == second
