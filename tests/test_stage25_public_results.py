from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results" / "wm" / "stage25_detector_calibration"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_threshold_is_calibration_only_and_test_is_disjoint() -> None:
    threshold = json.loads((RESULTS / "frozen_threshold.json").read_text(encoding="utf-8"))
    assert threshold["threshold"] == pytest.approx(0.024414002895355225)
    assert threshold["test_scores_accessed"] is False
    assert threshold["consecutive_windows"] == 3
    pairs = read_csv(RESULTS / "pairs.csv")
    calibration = {
        (row["benchmark_task_id"], row["init_state_index"]) for row in pairs if row["split"] == "calibration"
    }
    test = {(row["benchmark_task_id"], row["init_state_index"]) for row in pairs if row["split"] == "test"}
    assert len(calibration) == len(test) == 6
    assert calibration.isdisjoint(test)


def test_test_metrics_recompute_from_event_units() -> None:
    rows = [row for row in read_csv(RESULTS / "detector_events.csv") if row["split"] == "test"]
    assert len(rows) == 18
    normal = [row for row in rows if row["condition"] == "nominal"]
    attenuation = [row for row in rows if row["condition"] == "motion_attenuation_0p5"]
    delay = [row for row in rows if row["condition"] == "action_delay_3"]
    assert sum(row["wm_alarm"] == "True" for row in normal) == 0
    assert sum(row["wm_alarm"] == "True" for row in attenuation) == 5
    assert sum(row["wm_alarm"] == "True" for row in delay) == 4
    assert sum(row["direct_alarm"] == "True" for row in normal) == 0
    assert sum(row["direct_alarm"] == "True" for row in attenuation) == 6
    assert sum(row["direct_alarm"] == "True" for row in delay) == 6
    attenuation_latency = [int(row["wm_latency_steps"]) for row in attenuation if row["wm_latency_steps"]]
    delay_latency = [int(row["wm_latency_steps"]) for row in delay if row["wm_latency_steps"]]
    assert sum(attenuation_latency) / len(attenuation_latency) == pytest.approx(12.8)
    assert sum(delay_latency) / len(delay_latency) == pytest.approx(10.0)


def test_pairing_and_public_hash_contract() -> None:
    pairs = read_csv(RESULTS / "pairs.csv")
    assert len(pairs) == 12
    assert all(row["preperiod_pairing_passed"] == "True" for row in pairs)
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert report["scope"]["actions_modified_by_detector"] == 0
    assert report["scope"]["shield_claimed"] is False
    assert report["conclusion"]["direct_mismatch_baseline_is_stronger_or_equal"]
    for key, filename in (
        ("protocol_sha256", "protocol.json"),
        ("pairs_sha256", "pairs.csv"),
        ("episodes_sha256", "episodes.csv"),
        ("window_scores_sha256", "window_scores.csv"),
        ("detector_events_sha256", "detector_events.csv"),
        ("frozen_threshold_sha256", "frozen_threshold.json"),
    ):
        assert report["artifacts"][key] == sha256_file(RESULTS / filename)
