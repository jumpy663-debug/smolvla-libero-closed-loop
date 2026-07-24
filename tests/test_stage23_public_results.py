from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results" / "wm" / "stage23_confirmatory_wm_scoring"
DISCOVERY = ROOT / "results" / "wm" / "stage21_paired_wm_scoring"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_primary_confirmation_recomputes_from_six_pairs() -> None:
    rows = read_csv(RESULTS / "pair_confirmation.csv")
    assert len(rows) == 6
    values = np.array([float(row["primary_mean_across_interventions"]) for row in rows])
    assert values.mean() == pytest.approx(0.015322749813397726)
    assert int((values > 0).sum()) == 5
    assignments = np.array(list(itertools.product((-1.0, 1.0), repeat=6)))
    null = (assignments * values).mean(axis=1)
    exact_p = np.mean(null >= values.mean() - 1e-15)
    assert len(null) == 64
    assert exact_p == pytest.approx(0.03125)

    discovery_ids = {row["pair_id"] for row in read_csv(DISCOVERY / "pair_effects.csv")}
    confirmation_ids = {row["pair_id"] for row in rows}
    assert discovery_ids.isdisjoint(confirmation_ids)


def test_intervention_specific_results_recompute() -> None:
    rows = read_csv(RESULTS / "pair_confirmation.csv")
    expected = {
        "motion_attenuation_0p5": (0.02241802215576172, 5),
        "action_delay_3": (0.008227477471033732, 4),
    }
    for condition, (mean, positives) in expected.items():
        values = np.array([float(row[f"{condition}_post_commanded_minus_executed"]) for row in rows])
        assert values.mean() == pytest.approx(mean)
        assert int((values > 0).sum()) == positives


def test_window_counts_negative_controls_and_hash_contract() -> None:
    windows = read_csv(RESULTS / "window_scores.csv")
    assert len(windows) == 1080
    assert {int(row["start_index"]) for row in windows if row["period"] == "pre"} == set(range(10, 40))
    assert {int(row["start_index"]) for row in windows if row["period"] == "post"} == set(range(50, 80))
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert report["conclusion"]["independent_confirmation_passed"] is True
    assert report["scope"]["model_parameters_updated"] == 0
    assert report["scope"]["thresholds_fitted"] == 0
    assert all(report["negative_controls"].values())
    for key, filename in (
        ("feature_manifest_sha256", "feature_manifest.csv"),
        ("window_scores_sha256", "window_scores.csv"),
        ("episode_period_scores_sha256", "episode_period_scores.csv"),
        ("pair_confirmation_sha256", "pair_confirmation.csv"),
        ("statistics_sha256", "statistics.json"),
    ):
        assert report["artifacts"][key] == sha256_file(RESULTS / filename)
