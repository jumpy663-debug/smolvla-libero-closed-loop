from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from statistical_evaluation_common import (  # noqa: E402
    EpisodeJob,
    build_jobs,
    load_frozen_protocol,
    validate_resume_record,
)


def load_analysis_module():
    path = SCRIPTS_DIR / "stage30_analyze_statistical_evaluation.py"
    spec = importlib.util.spec_from_file_location("stage30_analysis_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = load_analysis_module()


def load_failure_review_module():
    path = SCRIPTS_DIR / "stage31_prepare_failure_review.py"
    spec = importlib.util.spec_from_file_location("stage31_failure_review_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FAILURE_REVIEW = load_failure_review_module()


def load_public_artifact_module():
    path = SCRIPTS_DIR / "stage32_build_statistical_public_artifacts.py"
    spec = importlib.util.spec_from_file_location("stage32_public_artifacts_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PUBLIC_ARTIFACTS = load_public_artifact_module()


@pytest.fixture(scope="module")
def frozen_protocol() -> tuple[dict, str]:
    return load_frozen_protocol(REPOSITORY_ROOT / "protocols/statistical_evaluation_v1.json")


def test_frozen_manifest_counts_and_scope(frozen_protocol: tuple[dict, str]) -> None:
    protocol, _ = frozen_protocol
    formal = build_jobs(protocol, "formal")
    randomness = build_jobs(protocol, "randomness")
    pilot = build_jobs(protocol, "pilot")

    assert len(formal) == 182
    assert len(randomness) == 52
    assert len(pilot) == 1
    assert len({job.episode_id for job in [*formal, *randomness, *pilot]}) == 235
    assert all(job.init_state_index in range(3, 10) for job in formal)
    assert all(job.maximum_control_steps == 280 for job in formal)
    assert pilot[0].init_state_index == 49
    assert pilot[0].maximum_control_steps == 30


def test_formal_condition_counts(frozen_protocol: tuple[dict, str]) -> None:
    protocol, _ = frozen_protocol
    formal = build_jobs(protocol, "formal")
    counts = {}
    for job in formal:
        counts[(job.policy, job.action_horizon)] = counts.get((job.policy, job.action_horizon), 0) + 1
    assert counts == {
        ("pretrained", 50): 70,
        ("pretrained", 25): 28,
        ("pretrained", 10): 28,
        ("expert_only", 10): 28,
        ("lora", 10): 28,
    }


def test_randomness_jobs_preserve_environment_and_change_only_policy_seed(
    frozen_protocol: tuple[dict, str],
) -> None:
    protocol, _ = frozen_protocol
    jobs = build_jobs(protocol, "randomness")
    assert all(job.policy == "pretrained" and job.action_horizon == 10 for job in jobs)
    assert all(job.environment_seed == 1000 + job.init_state_index for job in jobs)
    base_jobs = [job for job in jobs if job.policy_seed < 2000]
    assert len(base_jobs) == 12
    assert {job.init_state_index for job in base_jobs} == {0, 1, 2}


def test_resume_record_rejects_protocol_or_job_drift(frozen_protocol: tuple[dict, str]) -> None:
    _, protocol_sha256 = frozen_protocol
    job = EpisodeJob("formal", "pretrained", 0, 3, 1003, 1003, 50, 280)
    record = {
        "protocol_sha256": protocol_sha256,
        "episode_id": job.episode_id,
        "mode": job.mode,
        "policy": job.policy,
        "task_id": job.task_id,
        "init_state_index": job.init_state_index,
        "environment_seed": job.environment_seed,
        "policy_seed": job.policy_seed,
        "action_horizon": job.action_horizon,
        "maximum_control_steps": job.maximum_control_steps,
        "control_steps": 51,
        "logical_vlm_forward_count": 2,
        "actual_vlm_forward_count": 2,
        "initial_observation_sha256": "a" * 64,
        "action_sha256": "b" * 64,
        "video_sha256": "c" * 64,
        "trajectory_sha256": "d" * 64,
    }
    validate_resume_record(record, job, protocol_sha256)
    record["action_horizon"] = 25
    with pytest.raises(AssertionError, match="action_horizon"):
        validate_resume_record(record, job, protocol_sha256)


def test_exact_mcnemar_and_holm() -> None:
    assert ANALYSIS.exact_mcnemar_p_value(0, 0) == 1.0
    assert ANALYSIS.exact_mcnemar_p_value(0, 5) == pytest.approx(0.0625)
    assert ANALYSIS.exact_mcnemar_p_value(1, 4) == pytest.approx(0.375)
    assert ANALYSIS.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_wilson_and_pair_bootstrap() -> None:
    lower, upper = ANALYSIS.wilson_interval(5, 10)
    assert lower == pytest.approx(0.236593, abs=1e-6)
    assert upper == pytest.approx(0.763407, abs=1e-6)
    result = ANALYSIS.paired_bootstrap([2.0, 2.0, 2.0])
    assert result["pairs"] == 3
    assert result["mean_difference"] == 2.0
    assert result["confidence_interval_95"] == [2.0, 2.0]


def test_failure_taxonomy_is_frozen_and_complete() -> None:
    taxonomy = json.loads((REPOSITORY_ROOT / "protocols/failure_taxonomy_v1.json").read_text(encoding="utf-8"))
    assert taxonomy["status"] == "frozen"
    assert taxonomy["frozen_before_new_results"] is True
    codes = [item["code"] for item in taxonomy["categories"]]
    assert len(codes) == 10
    assert len(set(codes)) == 10
    assert "simulation_numeric_or_environment_error" in codes
    assert "other" in codes


def test_contact_sheet_sampling_includes_endpoints() -> None:
    indices = FAILURE_REVIEW.sample_indices(frame_count=281, sample_count=12)
    assert indices[0] == 0
    assert indices[-1] == 280
    assert len(indices) == len(set(indices)) == 12


def test_indexed_failure_labels_are_locked_to_episode_sequence(tmp_path: Path) -> None:
    rows = [
        {
            "episode_id": "episode-a",
            "primary_failure_code": "",
            "label_source": "",
            "review_status": "pending",
            "reviewer": "",
            "notes_zh": "",
        },
        {
            "episode_id": "episode-b",
            "primary_failure_code": "",
            "label_source": "",
            "review_status": "pending",
            "reviewer": "",
            "notes_zh": "",
        },
    ]
    episode_sequence = "episode-a\nepisode-b\n"
    labels = {
        "reviewer": "test-reviewer",
        "label_source": "human_contact_sheet",
        "episode_id_sequence_sha256": hashlib.sha256(episode_sequence.encode()).hexdigest(),
        "codes_by_annotation_index": ["approach_but_grasp_failed", "other"],
        "notes_by_annotation_index": {"1": "测试备注"},
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    FAILURE_REVIEW.apply_labels(
        rows,
        labels_path=labels_path,
        allowed_codes={"approach_but_grasp_failed", "other"},
    )
    assert [row["review_status"] for row in rows] == ["confirmed", "confirmed"]
    assert rows[1]["notes_zh"] == "测试备注"

    rows.reverse()
    with pytest.raises(AssertionError, match="sequence drifted"):
        FAILURE_REVIEW.apply_labels(
            rows,
            labels_path=labels_path,
            allowed_codes={"approach_but_grasp_failed", "other"},
        )


def test_failure_summary_reports_complete_condition_distributions() -> None:
    rows = [
        {
            "policy": "pretrained",
            "action_horizon": 50,
            "task_id": 4,
            "primary_failure_code": "approach_but_grasp_failed",
            "review_status": "confirmed",
        },
        {
            "policy": "pretrained",
            "action_horizon": 50,
            "task_id": 0,
            "primary_failure_code": "dropped_after_grasp",
            "review_status": "confirmed",
        },
        {
            "policy": "lora",
            "action_horizon": 10,
            "task_id": 5,
            "primary_failure_code": "dropped_after_grasp",
            "review_status": "confirmed",
        },
    ]
    report = ANALYSIS.summarize_failure_annotations(
        rows,
        taxonomy_codes=["approach_but_grasp_failed", "dropped_after_grasp"],
    )
    assert report["status"] == "passed"
    assert report["overall"]["failures"] == 3
    assert report["by_condition"]["pretrained_h50_all_tasks"]["failures"] == 2
    assert report["by_condition"]["pretrained_h50_core_tasks"]["failures"] == 1
    assert report["by_condition"]["lora_h10_core_tasks"]["counts"]["dropped_after_grasp"] == 1


def test_public_overview_is_derived_from_checked_in_statistics() -> None:
    statistics = json.loads(
        (REPOSITORY_ROOT / "results/statistical_evaluation/statistics.json").read_text(encoding="utf-8")
    )
    svg = PUBLIC_ARTIFACTS.build_svg(statistics)
    assert "58/100 = 58.0%" in svg
    assert "H25↔H10：p=0.453" in svg
    assert "LoRA 与预训练成功标签 40/40 完全一致" in svg
    assert "48/114（42.1%）" in svg


def test_checked_in_results_docs_and_media_are_consistent() -> None:
    results_dir = REPOSITORY_ROOT / "results/statistical_evaluation"
    statistics = json.loads((results_dir / "statistics.json").read_text(encoding="utf-8"))
    with (results_dir / "episodes.csv").open(encoding="utf-8", newline="") as file:
        episodes = list(csv.DictReader(file))
    with (results_dir / "failure_annotations.csv").open(encoding="utf-8", newline="") as file:
        failures = list(csv.DictReader(file))
    assert len(episodes) == len({row["episode_id"] for row in episodes}) == 260
    assert len(failures) == 114
    assert all(row["review_status"] == "confirmed" and row["primary_failure_code"] for row in failures)
    assert statistics["status"] == "passed"
    assert statistics["failure_taxonomy"]["status"] == "passed"
    assert statistics["randomness_audit"]["status"] == "passed"

    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    report = (REPOSITORY_ROOT / "docs/STATISTICAL_EVALUATION.md").read_text(encoding="utf-8")
    for required in (
        "58/100",
        "12/40",
        "20/40",
        "23/40",
        "0.015625",
        "0.010254",
        "0.453125",
        "114/114",
    ):
        assert required in readme
        assert required in report

    manifest = json.loads((results_dir / "media_manifest.json").read_text(encoding="utf-8"))
    paths_and_hashes = (
        (manifest["overview"]["svg"], manifest["overview"]["svg_sha256"]),
        (manifest["overview"]["png"], manifest["overview"]["png_sha256"]),
        (
            manifest["horizon_paired_video"]["public_video"],
            manifest["horizon_paired_video"]["public_video_sha256"],
        ),
        (
            manifest["horizon_paired_video"]["public_preview"],
            manifest["horizon_paired_video"]["public_preview_sha256"],
        ),
        (
            manifest["failure_examples"]["public_contact_sheet"],
            manifest["failure_examples"]["public_contact_sheet_sha256"],
        ),
    )
    for relative_path, expected_sha256 in paths_and_hashes:
        path = REPOSITORY_ROOT / relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256


def test_legacy_public_paths_are_portable_and_resolvable(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    asset_root = tmp_path / "legacy-assets"
    source = asset_root / "stage6" / "episode.mp4"
    public = ANALYSIS.relative_public_path(str(source), repository_root, asset_root)
    assert public == "external_asset_root/stage6/episode.mp4"
    resolved = FAILURE_REVIEW.resolve_public_artifact_path(
        public,
        repository_root=repository_root,
        asset_root=asset_root,
    )
    assert resolved == source
