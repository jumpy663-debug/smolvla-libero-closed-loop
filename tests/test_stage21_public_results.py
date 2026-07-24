from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results" / "wm" / "stage21_paired_wm_scoring"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_pair_statistics_recompute_independently() -> None:
    with (RESULTS / "pair_effects.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert sorted(int(row["benchmark_task_id"]) for row in rows) == [4, 4, 7, 7, 8, 8]
    values = np.array([float(row["commanded_h10_raw_mse_difference_in_differences"]) for row in rows])
    assert values.mean() == pytest.approx(-0.040002922217051186)
    assert int((values > 0).sum()) == 3

    assignments = np.array(list(itertools.product((-1.0, 1.0), repeat=6)))
    null = (assignments * values).mean(axis=1)
    exact_p = np.mean(np.abs(null) >= abs(values.mean()) - 1e-15)
    assert len(null) == 64
    assert exact_p == pytest.approx(0.65625)

    gap = np.array([float(row["fault_post_commanded_minus_executed"]) for row in rows])
    assert bool((gap > 0).all())
    assert gap.mean() == pytest.approx(0.16190887490908304)


def test_window_counts_and_report_hash_contract() -> None:
    with (RESULTS / "window_scores.csv").open(newline="", encoding="utf-8") as handle:
        windows = list(csv.DictReader(handle))
    assert len(windows) == 720
    assert {int(row["start_index"]) for row in windows if row["period"] == "pre"} == set(range(10, 40))
    assert {int(row["start_index"]) for row in windows if row["period"] == "post"} == set(range(50, 80))

    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert report["conclusion"]["decision_rule_passed"] is False
    assert report["scope"]["model_parameters_updated"] == 0
    assert report["scope"]["pair_is_statistical_unit"] is True
    for key, filename in (
        ("feature_manifest_sha256", "feature_manifest.csv"),
        ("window_scores_sha256", "window_scores.csv"),
        ("episode_period_scores_sha256", "episode_period_scores.csv"),
        ("pair_effects_sha256", "pair_effects.csv"),
        ("statistics_sha256", "statistics.json"),
    ):
        assert report["artifacts"][key] == file_sha256(RESULTS / filename)
