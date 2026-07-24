from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def load_stage22():
    path = Path(__file__).parents[1] / "scripts" / "wm" / "stage22_collect_confirmatory_interventions.py"
    scripts_path = str(path.parent)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location(
        "stage22_collect_confirmatory_interventions",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE22 = load_stage22()


def scout_rows() -> list[dict]:
    rows = []
    for task_id, indices in STAGE22.EXPECTED_CANDIDATE_INDICES.items():
        for init_state_index in indices:
            rows.append(
                {
                    "benchmark_task_id": task_id,
                    "init_state_index": init_state_index,
                    "episode_seed": STAGE22.STAGE18.episode_seed(
                        task_id,
                        init_state_index,
                    ),
                    "task_name": f"task-{task_id}",
                    "success": True,
                    "steps": 100,
                    "initial_observation_sha256": f"obs-{task_id}-{init_state_index}",
                }
            )
    return rows


def test_candidate_selection_is_score_blind_and_pinned() -> None:
    rows = scout_rows()
    first = STAGE22.select_screen_candidates(rows, set())
    for row in rows:
        row["h10_raw_mse"] = -float(row["init_state_index"])
    second = STAGE22.select_screen_candidates(list(reversed(rows)), set())
    assert first == second
    assert len(first) == 12


def test_stage20_screen_states_are_excluded() -> None:
    rows = scout_rows()
    rows.append(
        {
            **rows[0],
            "init_state_index": 99,
            "episode_seed": 99,
            "initial_observation_sha256": "unused",
        }
    )
    with pytest.raises(AssertionError, match="Candidate cohort changed"):
        STAGE22.select_screen_candidates(
            rows,
            {
                (
                    int(rows[0]["benchmark_task_id"]),
                    int(rows[0]["init_state_index"]),
                )
            },
        )


def test_attenuation_and_delay_transform_only_arm_dimensions() -> None:
    history = [np.full((1, 7), value, dtype=np.float32) for value in range(50)]
    command = np.full((1, 7), 100, dtype=np.float32)
    attenuated, active = STAGE22.apply_intervention(
        command,
        history,
        step=50,
        condition="motion_attenuation_0p5",
    )
    assert active
    assert np.array_equal(attenuated[:, :6], np.full((1, 6), 50))
    assert attenuated[0, 6] == 100
    delayed, active = STAGE22.apply_intervention(
        command,
        history,
        step=50,
        condition="action_delay_3",
    )
    assert active
    assert np.array_equal(delayed[:, :6], np.full((1, 6), 47))
    assert delayed[0, 6] == 100


def test_intervention_never_changes_preperiod() -> None:
    command = np.ones((1, 7), dtype=np.float32)
    for condition in STAGE22.CONDITIONS:
        executed, active = STAGE22.apply_intervention(
            command,
            [],
            step=49,
            condition=condition,
        )
        assert not active
        assert np.array_equal(executed, command)


def test_freeze_requires_scoreable_fresh_successes() -> None:
    candidates = STAGE22.select_screen_candidates(scout_rows(), set())
    records = []
    for candidate in candidates:
        records.append(
            {
                "benchmark_task_id": candidate["benchmark_task_id"],
                "init_state_index": candidate["init_state_index"],
                "success": candidate["screen_rank_within_task"] < 2,
                "steps": 90,
                "initial_observation_sha256": "a" * 64,
                "action_sha256": "b" * 64,
            }
        )
    selected = STAGE22.freeze_pairs(candidates, records)
    assert len(selected) == 6
    records[1]["steps"] = 89
    with pytest.raises(AssertionError, match="scoreable fresh successes"):
        STAGE22.freeze_pairs(candidates, records)
