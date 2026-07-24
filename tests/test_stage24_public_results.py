from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results" / "wm" / "stage24_online_sidecar"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_online_replay_and_live_control_contracts() -> None:
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["replay_equivalence"]["registered_windows"] == 1080
    assert report["replay_equivalence"]["maximum_absolute_score_difference"] <= 2e-5
    assert report["conclusion"]["wm_runs_inside_smolvla_libero_loop"] is True
    assert report["conclusion"]["sidecar_preserves_control_outputs_bit_exact"] is True
    assert report["scope"]["actions_modified_by_monitor"] == 0
    assert report["scope"]["thresholds_fitted"] == 0
    assert report["scope"]["shield_claimed"] is False


def test_live_timeline_recomputes_registered_gap_means() -> None:
    rows = read_csv(RESULTS / "live_timeline.csv")
    assert len(rows) == 162
    assert {row["condition"] for row in rows} == {
        "nominal",
        "motion_attenuation_0p5",
    }
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    summaries = {row["condition"]: row for row in report["live_pair"]}
    for condition in summaries:
        selected = [
            float(row["commanded_minus_executed"])
            for row in rows
            if row["condition"] == condition and 50 <= int(row["start_index"]) < 80
        ]
        assert len(selected) == 30
        assert np.mean(selected) == pytest.approx(
            summaries[condition]["post_h10_gap_mean"],
        )
    assert summaries["nominal"]["post_h10_gap_mean"] == 0.0
    assert summaries["motion_attenuation_0p5"]["post_h10_gap_mean"] == pytest.approx(
        0.040184650818506876,
    )


def test_public_hashes_and_videos_are_auditable() -> None:
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    for key, filename in (
        ("replay_comparison_sha256", "replay_comparison.csv"),
        ("live_timeline_sha256", "live_timeline.csv"),
        ("live_summary_sha256", "live_summary.json"),
    ):
        assert report["artifacts"][key] == sha256_file(RESULTS / filename)
    for artifact in report["artifacts"]["videos"]:
        path = ROOT / artifact["local_path"]
        assert path.is_file()
        assert artifact["sha256"] == sha256_file(path)
