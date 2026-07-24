from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def load_stage21():
    path = Path(__file__).parents[1] / "scripts" / "wm" / "stage21_paired_wm_scoring.py"
    scripts_path = str(path.parent)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location("stage21_paired_wm_scoring", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE21 = load_stage21()


def test_registered_windows_do_not_cross_intervention() -> None:
    assert tuple(range(10, 40)) == STAGE21.PRE_STARTS
    assert tuple(range(50, 80)) == STAGE21.POST_STARTS
    assert max(STAGE21.PRE_STARTS) + STAGE21.HORIZON == 49
    assert min(STAGE21.POST_STARTS) == 50
    valid = torch.ones(90, dtype=torch.bool)
    assert torch.equal(
        STAGE21.validate_window_starts(valid, STAGE21.POST_STARTS),
        torch.arange(50, 80),
    )


def test_invalid_transition_in_registered_window_is_rejected() -> None:
    valid = torch.ones(90, dtype=torch.bool)
    valid[58] = False
    with pytest.raises(AssertionError, match="Invalid transition"):
        STAGE21.validate_window_starts(valid, STAGE21.POST_STARTS)


def test_exact_sign_flip_uses_six_pairs_not_windows() -> None:
    result = STAGE21.exact_sign_flip(np.ones(6))
    assert result["sign_assignments"] == 64
    assert result["two_sided_p_value"] == pytest.approx(2 / 64)


def test_holm_adjustment_controls_diagnostic_family() -> None:
    adjusted = STAGE21.holm_adjust(
        {
            "a": 0.03125,
            "b": 0.03125,
            "c": 0.03125,
            "d": 0.1875,
        }
    )
    assert adjusted == {
        "a": pytest.approx(0.125),
        "b": pytest.approx(0.125),
        "c": pytest.approx(0.125),
        "d": pytest.approx(0.1875),
    }


def test_difference_in_differences_is_computed_at_pair_level() -> None:
    rows = []
    for condition, pre, post in (
        ("nominal", 1.0, 2.0),
        ("persistent_motion_dropout", 1.0, 5.0),
    ):
        for period, value in (("pre", pre), ("post", post)):
            rows.append(
                {
                    "pair_id": "task04-init08",
                    "condition": condition,
                    "period": period,
                    "benchmark_task_id": 4,
                    "init_state_index": 8,
                    **{name: value for name in STAGE21.SCORE_NAMES},
                }
            )
    result = STAGE21.pair_effect_rows(rows)
    assert len(result) == 1
    for name in STAGE21.SCORE_NAMES:
        assert result[0][f"{name}_difference_in_differences"] == pytest.approx(3.0)


def test_source_action_contract_for_nominal_and_fault() -> None:
    def arrays(condition: str) -> dict[str, np.ndarray]:
        steps = 90
        commanded = np.ones((steps, 7), dtype=np.float32)
        executed = commanded.copy()
        mask = np.zeros(steps, dtype=bool)
        if condition != "nominal":
            executed[50:, :6] = 0
            mask[50:] = True
        return {
            "camera1": np.zeros((steps + 1, 360, 360, 3), dtype=np.uint8),
            "camera2": np.zeros((steps + 1, 360, 360, 3), dtype=np.uint8),
            "eef_pos": np.zeros((steps + 1, 3), dtype=np.float32),
            "eef_quat": np.zeros((steps + 1, 4), dtype=np.float32),
            "gripper_qpos": np.zeros((steps + 1, 2), dtype=np.float32),
            "commanded_action": commanded,
            "executed_action": executed,
            "transition_valid": np.ones(steps, dtype=bool),
            "intervention_mask": mask,
        }

    for condition in STAGE21.CONDITIONS:
        STAGE21.validate_source_arrays(
            arrays(condition),
            {
                "steps": "90",
                "valid_transitions": "90",
                "condition": condition,
            },
        )
