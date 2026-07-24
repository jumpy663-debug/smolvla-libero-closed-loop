from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_stage25():
    path = Path(__file__).parents[1] / "scripts" / "wm" / "stage25_detector_calibration.py"
    scripts_path = str(path.parent)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location(
        "stage25_detector_calibration_for_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE25 = load_stage25()
ROOT = Path(__file__).parents[1]


def test_selection_excludes_all_prior_screened_states_and_is_pinned() -> None:
    scout = STAGE25.read_rows(ROOT / "results" / "wm" / "stage18_initial_state_scout" / "scout_episodes.csv")
    scout.extend(STAGE25.read_rows(ROOT / "results" / "wm" / "stage19_balanced_cohort" / "task8_extension.csv"))
    excluded = set()
    for path in (
        ROOT / "results" / "wm" / "stage20_paired_interventions" / "fresh_reset_screen.csv",
        ROOT / "results" / "wm" / "stage22_confirmatory_interventions" / "fresh_reset_screen.csv",
    ):
        excluded.update(
            (
                int(row["benchmark_task_id"]),
                int(row["init_state_index"]),
            )
            for row in STAGE25.read_rows(path)
        )
    selected = STAGE25.select_pairs(scout, excluded)
    assert len(selected) == 12
    assert all(
        (
            int(row["benchmark_task_id"]),
            int(row["init_state_index"]),
        )
        not in excluded
        for row in selected
    )
    assert sum(row["split"] == "calibration" for row in selected) == 6
    assert sum(row["split"] == "test" for row in selected) == 6


def synthetic_episode(
    *,
    pair_id: str,
    condition: str,
    positive_starts: set[int],
    value: float = 0.01,
) -> dict:
    timeline = []
    for start in range(81):
        timeline.append(
            {
                "start_index": start,
                "target_index": start + 10,
                "available_after_step": start + 9,
                "commanded_minus_executed": (value if start in positive_starts else 0.0),
            }
        )
    return {
        "pair_id": pair_id,
        "split": "calibration",
        "condition": condition,
        "timeline": timeline,
    }


def test_alarm_is_causal_and_requires_three_consecutive_windows() -> None:
    episode = synthetic_episode(
        pair_id="fault",
        condition="motion_attenuation_0p5",
        positive_starts={41, 42, 43},
    )
    alarm = STAGE25.first_consecutive_alarm(
        episode,
        threshold=0.005,
        starts=STAGE25.INFLUENCED_STARTS,
    )
    assert alarm is not None
    assert alarm["start_index"] == 43
    assert alarm["available_after_step"] == 52
    assert int(alarm["available_after_step"]) - STAGE25.INTERVENTION_START == 2
    broken = synthetic_episode(
        pair_id="broken",
        condition="motion_attenuation_0p5",
        positive_starts={41, 43, 44},
    )
    assert (
        STAGE25.first_consecutive_alarm(
            broken,
            threshold=0.005,
            starts=STAGE25.INFLUENCED_STARTS,
        )
        is None
    )


def test_threshold_uses_only_calibration_episodes_and_frozen_rule() -> None:
    episodes = []
    for index in range(6):
        episodes.append(
            synthetic_episode(
                pair_id=f"pair-{index}",
                condition="nominal",
                positive_starts=set(),
            )
        )
        episodes.append(
            synthetic_episode(
                pair_id=f"pair-{index}",
                condition="motion_attenuation_0p5",
                positive_starts=set(range(41, 81)),
            )
        )
    threshold = STAGE25.freeze_threshold(episodes)
    assert threshold["threshold"] == 0.005
    assert threshold["selected_calibration_counts"]["nominal_false_alarms"] == 0
    assert threshold["selected_calibration_counts"]["fault_episodes_detected"] == 6
    assert threshold["test_scores_accessed"] is False
