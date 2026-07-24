from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results" / "wm" / "stage22_confirmatory_interventions"
STAGE20 = ROOT / "results" / "wm" / "stage20_paired_interventions"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_confirmation_cohort_is_new_complete_and_score_blind() -> None:
    stage20_keys = {
        (int(row["benchmark_task_id"]), int(row["init_state_index"]))
        for row in read_csv(STAGE20 / "fresh_reset_screen.csv")
    }
    screen = read_csv(RESULTS / "fresh_reset_screen.csv")
    stage22_keys = {(int(row["benchmark_task_id"]), int(row["init_state_index"])) for row in screen}
    assert len(screen) == 12
    assert stage20_keys.isdisjoint(stage22_keys)
    assert sum(row["success"] == "True" for row in screen) == 11
    assert all(row["world_model_scores_computed"] == "False" for row in screen)

    episodes = read_csv(RESULTS / "episodes.csv")
    assert len(episodes) == 18
    assert Counter(row["condition"] for row in episodes) == {
        "nominal": 6,
        "motion_attenuation_0p5": 6,
        "action_delay_3": 6,
    }
    assert all(row["success"] == "True" for row in episodes)
    assert all(int(row["valid_transitions"]) >= 89 for row in episodes)
    assert all(row["world_model_scores_computed"] == "False" for row in episodes)


def test_motion_and_step_aggregates_recompute_from_pairs() -> None:
    rows = read_csv(RESULTS / "pairs.csv")
    assert len(rows) == 6
    assert all(row["ready_for_confirmatory_wm_scoring"] == "True" for row in rows)
    assert sum(int(row["nominal_steps"]) for row in rows) == 692
    assert sum(int(row["motion_attenuation_0p5_steps"]) for row in rows) == 914
    assert sum(int(row["action_delay_3_steps"]) for row in rows) == 1002
    for condition, eef_expected, pixel_expected in (
        ("motion_attenuation_0p5", 0.5347917310162443, 0.8413143636742528),
        ("action_delay_3", 0.9313750979433408, 0.9454382038693894),
    ):
        mismatch = np.array([float(row[f"{condition}_post_action_mismatch_l2_mean"]) for row in rows])
        assert bool((mismatch > 0).all())
        eef = np.array([float(row[f"{condition}_eef_motion_ratio_vs_nominal"]) for row in rows])
        pixels = np.array([float(row[f"{condition}_pixel_motion_ratio_vs_nominal"]) for row in rows])
        assert np.median(eef) == pytest.approx(eef_expected)
        assert np.median(pixels) == pytest.approx(pixel_expected)


def test_report_hash_contract_and_stage23_gate() -> None:
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["scope"]["world_model_scores_computed"] == 0
    assert report["stage23_readiness"]["frozen_wm_scoring_can_start"] is True
    for key, filename in (
        ("fresh_reset_screen_csv_sha256", "fresh_reset_screen.csv"),
        ("episodes_csv_sha256", "episodes.csv"),
        ("pairs_csv_sha256", "pairs.csv"),
    ):
        assert report["artifacts"][key] == sha256_file(RESULTS / filename)
