from __future__ import annotations

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
    taxonomy = json.loads(
        (REPOSITORY_ROOT / "protocols/failure_taxonomy_v1.json").read_text(encoding="utf-8")
    )
    assert taxonomy["status"] == "frozen"
    assert taxonomy["frozen_before_new_results"] is True
    codes = [item["code"] for item in taxonomy["categories"]]
    assert len(codes) == 10
    assert len(set(codes)) == 10
    assert "simulation_numeric_or_environment_error" in codes
    assert "other" in codes
