#!/usr/bin/env python

"""Shared protocol and manifest helpers for the statistical evaluation stages."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

FORMAL_CORE_TASK_IDS = (4, 5, 7, 8)
FORMAL_INITIAL_STATE_INDICES = tuple(range(10))
NEW_INITIAL_STATE_INDICES = tuple(range(3, 10))
RANDOMNESS_INITIAL_STATE_INDICES = tuple(range(5))


@dataclass(frozen=True, slots=True)
class EpisodeJob:
    mode: str
    policy: str
    task_id: int
    init_state_index: int
    environment_seed: int
    policy_seed: int
    action_horizon: int
    maximum_control_steps: int

    @property
    def episode_id(self) -> str:
        return (
            f"{self.mode}-{self.policy}-h{self.action_horizon:02d}"
            f"-task{self.task_id:02d}-init{self.init_state_index:02d}"
            f"-env{self.environment_seed}-policy{self.policy_seed}"
        )

    @property
    def condition_id(self) -> str:
        return f"{self.policy}-h{self.action_horizon:02d}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("status") != "frozen":
        raise AssertionError("Statistical evaluation protocol is not frozen")
    if protocol.get("frozen_before_new_results") is not True:
        raise AssertionError("Protocol does not assert pre-result freezing")
    return protocol, sha256_bytes(raw)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main_seed(init_state_index: int) -> int:
    return 1000 + init_state_index


def build_new_formal_jobs(protocol: dict[str, Any]) -> list[EpisodeJob]:
    jobs: list[EpisodeJob] = []

    # Experiment A supplies every pretrained H=50 row. States 0--2 are frozen
    # legacy rows, so only 3--9 are new runs.
    for task_id in range(10):
        for init_state_index in NEW_INITIAL_STATE_INDICES:
            seed = main_seed(init_state_index)
            jobs.append(
                EpisodeJob(
                    mode="formal",
                    policy="pretrained",
                    task_id=task_id,
                    init_state_index=init_state_index,
                    environment_seed=seed,
                    policy_seed=seed,
                    action_horizon=50,
                    maximum_control_steps=280,
                )
            )

    # Experiment B reuses A's H=50 rows and adds H=25/H=10 on the four
    # preregistered tasks.
    for horizon in (25, 10):
        for task_id in FORMAL_CORE_TASK_IDS:
            for init_state_index in NEW_INITIAL_STATE_INDICES:
                seed = main_seed(init_state_index)
                jobs.append(
                    EpisodeJob(
                        mode="formal",
                        policy="pretrained",
                        task_id=task_id,
                        init_state_index=init_state_index,
                        environment_seed=seed,
                        policy_seed=seed,
                        action_horizon=horizon,
                        maximum_control_steps=280,
                    )
                )

    # Experiment C reuses B's pretrained H=10 rows and adds the two frozen
    # adaptation artifacts.
    for policy in ("expert_only", "lora"):
        for task_id in FORMAL_CORE_TASK_IDS:
            for init_state_index in NEW_INITIAL_STATE_INDICES:
                seed = main_seed(init_state_index)
                jobs.append(
                    EpisodeJob(
                        mode="formal",
                        policy=policy,
                        task_id=task_id,
                        init_state_index=init_state_index,
                        environment_seed=seed,
                        policy_seed=seed,
                        action_horizon=10,
                        maximum_control_steps=280,
                    )
                )

    expected = int(protocol["experiments"]["new_unique_episode_count"])
    if len(jobs) != expected or len({job.episode_id for job in jobs}) != expected:
        raise AssertionError(f"Expected {expected} unique formal jobs, built {len(jobs)}")
    return jobs


def build_randomness_jobs(protocol: dict[str, Any]) -> list[EpisodeJob]:
    audit = protocol["randomness"]["audit"]
    jobs = []
    # Legacy states 0--2 used a batched policy RNG stream. Re-run only those
    # three states sequentially at the primary policy seed; formal states 3--4
    # already use the required sequential seed and are reused by analysis.
    for task_id in FORMAL_CORE_TASK_IDS:
        for init_state_index in range(3):
            jobs.append(
                EpisodeJob(
                    mode="randomness",
                    policy="pretrained",
                    task_id=task_id,
                    init_state_index=init_state_index,
                    environment_seed=main_seed(init_state_index),
                    policy_seed=main_seed(init_state_index),
                    action_horizon=10,
                    maximum_control_steps=280,
                )
            )
    for policy_seed_base in (2000, 3000):
        for task_id in FORMAL_CORE_TASK_IDS:
            for init_state_index in RANDOMNESS_INITIAL_STATE_INDICES:
                jobs.append(
                    EpisodeJob(
                        mode="randomness",
                        policy="pretrained",
                        task_id=task_id,
                        init_state_index=init_state_index,
                        environment_seed=main_seed(init_state_index),
                        policy_seed=policy_seed_base + init_state_index,
                        action_horizon=10,
                        maximum_control_steps=280,
                    )
                )
    expected = int(audit["maximum_additional_episode_runs_after_main_experiment_reuse"])
    if len(jobs) != expected or len({job.episode_id for job in jobs}) != expected:
        raise AssertionError(f"Expected {expected} randomness jobs, built {len(jobs)}")
    return jobs


def build_pilot_job(protocol: dict[str, Any]) -> EpisodeJob:
    pilot = protocol["pipeline_validation"]
    return EpisodeJob(
        mode="pilot",
        policy=str(pilot["policy"]),
        task_id=int(pilot["task_id"]),
        init_state_index=int(pilot["init_state_index"]),
        environment_seed=int(pilot["environment_seed"]),
        policy_seed=int(pilot["policy_seed"]),
        action_horizon=int(pilot["action_horizon"]),
        maximum_control_steps=int(pilot["maximum_control_steps"]),
    )


def build_jobs(protocol: dict[str, Any], mode: str) -> list[EpisodeJob]:
    if mode == "formal":
        return build_new_formal_jobs(protocol)
    if mode == "randomness":
        return build_randomness_jobs(protocol)
    if mode == "pilot":
        return [build_pilot_job(protocol)]
    raise ValueError(f"Unsupported mode: {mode}")


def record_path(output_dir: Path, job: EpisodeJob) -> Path:
    return output_dir / "episodes" / job.mode / job.condition_id / f"{job.episode_id}.json"


def video_path(output_dir: Path, job: EpisodeJob) -> Path:
    return output_dir / "videos" / job.mode / job.condition_id / f"{job.episode_id}.mp4"


def trajectory_path(output_dir: Path, job: EpisodeJob) -> Path:
    return output_dir / "trajectories" / job.mode / job.condition_id / f"{job.episode_id}.npz"


def validate_resume_record(record: dict[str, Any], job: EpisodeJob, protocol_sha256: str) -> None:
    expected = {
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
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"Resume record mismatch for {key}: {record.get(key)!r} != {value!r}")
    steps = int(record["control_steps"])
    if steps <= 0 or steps > job.maximum_control_steps:
        raise AssertionError(f"Invalid control_steps in resume record: {steps}")
    if int(record["logical_vlm_forward_count"]) != (steps + job.action_horizon - 1) // job.action_horizon:
        raise AssertionError("Logical VLM forward count is inconsistent with steps and horizon")
    if int(record["actual_vlm_forward_count"]) != int(record["logical_vlm_forward_count"]):
        raise AssertionError("Sequential episode actual/logical VLM forward counts differ")
    for key in (
        "initial_observation_sha256",
        "action_sha256",
        "video_sha256",
        "trajectory_sha256",
    ):
        value = str(record[key])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise AssertionError(f"Invalid {key} in resume record")


def select_jobs(
    jobs: list[EpisodeJob],
    *,
    policies: set[str] | None,
    horizons: set[int] | None,
    task_ids: set[int] | None,
    init_state_indices: set[int] | None,
) -> list[EpisodeJob]:
    selected = []
    for job in jobs:
        if policies is not None and job.policy not in policies:
            continue
        if horizons is not None and job.action_horizon not in horizons:
            continue
        if task_ids is not None and job.task_id not in task_ids:
            continue
        if init_state_indices is not None and job.init_state_index not in init_state_indices:
            continue
        selected.append(job)
    return selected
