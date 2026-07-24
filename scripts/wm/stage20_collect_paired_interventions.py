#!/usr/bin/env python

"""Collect paired nominal and persistent motion-dropout SmolVLA rollouts.

For two score-blind selected successful initial states on each of Task 4/7/8,
run the same policy and seed under two conditions. The fault condition starts
at control step 50, zeros the six arm-motion dimensions passed to LIBERO, and
preserves the commanded gripper action. Both commanded and executed actions,
dual cameras, robot state, rewards, and labels are saved for later frozen-WM
action-consequence mismatch evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.envs import close_envs, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION

TASK_IDS = (4, 7, 8)
SCREEN_CANDIDATES_PER_TASK = 4
PAIRS_PER_TASK = 2
CONDITIONS = ("nominal", "persistent_motion_dropout")
INTERVENTION_START = 50
SELECTION_SALT = "stage20-paired-motion-dropout-v1"
MIN_FAULT_FAILURES = 4


def load_module(filename: str, name: str) -> ModuleType:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE18 = load_module("stage18_initial_state_scout.py", "stage18_for_stage20")
STAGE16 = STAGE18.STAGE16
STAGE3 = STAGE18.STAGE3
STAGE5 = STAGE18.STAGE5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/reproduction/smolvla_libero/stage2_checkpoint/checkpoint"),
    )
    parser.add_argument(
        "--stage2-manifest",
        type=Path,
        default=Path("outputs/reproduction/smolvla_libero/stage2_checkpoint/manifest.json"),
    )
    parser.add_argument(
        "--backbone-dir",
        type=Path,
        default=Path("outputs/reproduction/smolvla_libero/stage3_single_inference/backbone"),
    )
    parser.add_argument(
        "--stage17-report",
        type=Path,
        default=Path("results/wm/stage17_failure_signal/report.json"),
    )
    parser.add_argument(
        "--stage18-report",
        type=Path,
        default=Path("results/wm/stage18_initial_state_scout/report.json"),
    )
    parser.add_argument(
        "--stage18-episodes",
        type=Path,
        default=Path("results/wm/stage18_initial_state_scout/scout_episodes.csv"),
    )
    parser.add_argument(
        "--stage19-report",
        type=Path,
        default=Path("results/wm/stage19_balanced_cohort/report.json"),
    )
    parser.add_argument(
        "--stage19-extension",
        type=Path,
        default=Path("results/wm/stage19_balanced_cohort/task8_extension.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wm/stage20_paired_interventions"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage20_paired_interventions"),
    )
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(TASK_IDS))
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--overwrite-public", action="store_true")
    return parser.parse_args()


def canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot serialize empty rows")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def read_scout_rows(path: Path, *, source_stage: int) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = dict(raw)
        for key in ("benchmark_task_id", "init_state_index", "episode_seed", "steps"):
            row[key] = int(row[key])
        row["success"] = row["success"] == "True"
        row["source_stage"] = source_stage
        rows.append(row)
    return rows


def pair_selection_key(row: dict[str, Any]) -> str:
    payload = f"{SELECTION_SALT}|task={int(row['benchmark_task_id'])}|init={int(row['init_state_index'])}"
    return hashlib.sha256(payload.encode()).hexdigest()


def select_screen_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [(int(row["benchmark_task_id"]), int(row["init_state_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("Scout pool contains duplicate task/initial-state keys")
    selected: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        candidates = sorted(
            (
                row
                for row in rows
                if int(row["benchmark_task_id"]) == task_id
                and bool(row["success"])
                and int(row["steps"]) > INTERVENTION_START
            ),
            key=pair_selection_key,
        )
        if len(candidates) < SCREEN_CANDIDATES_PER_TASK:
            raise AssertionError(f"Task {task_id} has too few eligible successful scout episodes")
        for task_rank, row in enumerate(candidates[:SCREEN_CANDIDATES_PER_TASK]):
            selected.append(
                {
                    "candidate_id": f"task{task_id:02d}-init{int(row['init_state_index']):02d}",
                    "benchmark_task_id": task_id,
                    "task_name": row["task_name"],
                    "init_state_index": int(row["init_state_index"]),
                    "episode_seed": int(row["episode_seed"]),
                    "screen_rank_within_task": task_rank,
                    "selection_key": pair_selection_key(row),
                    "scout_source_stage": int(row["source_stage"]),
                    "scout_steps": int(row["steps"]),
                    "scout_initial_observation_sha256": row["initial_observation_sha256"],
                    "scout_action_sha256": row["action_sha256"],
                    "selection_uses_world_model_score": False,
                }
            )
    selected.sort(
        key=lambda row: (
            int(row["benchmark_task_id"]),
            int(row["screen_rank_within_task"]),
        )
    )
    if len(selected) != len(TASK_IDS) * SCREEN_CANDIDATES_PER_TASK:
        raise AssertionError("Fresh-reset screen selection has the wrong size")
    return selected


def freeze_pairs(
    screen_candidates: list[dict[str, Any]],
    screen_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    record_lookup = {(int(row["benchmark_task_id"]), int(row["init_state_index"])): row for row in screen_records}
    if len(record_lookup) != len(screen_records):
        raise AssertionError("Fresh-reset screen contains duplicate keys")
    selected: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        eligible = []
        for candidate in screen_candidates:
            if int(candidate["benchmark_task_id"]) != task_id:
                continue
            key = (task_id, int(candidate["init_state_index"]))
            if key not in record_lookup:
                raise AssertionError(f"Missing fresh-reset screen record: {key}")
            screen = record_lookup[key]
            if bool(screen["success"]) and int(screen["steps"]) > INTERVENTION_START:
                eligible.append((candidate, screen))
        if len(eligible) < PAIRS_PER_TASK:
            raise AssertionError(f"Task {task_id} has only {len(eligible)} stable fresh-reset successes")
        for candidate, screen in eligible[:PAIRS_PER_TASK]:
            selected.append(
                {
                    "pair_id": candidate["candidate_id"],
                    "benchmark_task_id": task_id,
                    "task_name": candidate["task_name"],
                    "init_state_index": int(candidate["init_state_index"]),
                    "episode_seed": int(candidate["episode_seed"]),
                    "screen_rank_within_task": int(candidate["screen_rank_within_task"]),
                    "selection_key": candidate["selection_key"],
                    "scout_source_stage": int(candidate["scout_source_stage"]),
                    "historical_scout_steps": int(candidate["scout_steps"]),
                    "historical_scout_initial_observation_sha256": candidate["scout_initial_observation_sha256"],
                    "historical_scout_action_sha256": candidate["scout_action_sha256"],
                    "fresh_screen_steps": int(screen["steps"]),
                    "fresh_screen_initial_observation_sha256": screen["initial_observation_sha256"],
                    "fresh_screen_action_sha256": screen["action_sha256"],
                    "selection_uses_world_model_score": False,
                }
            )
    selected.sort(
        key=lambda row: (
            int(row["benchmark_task_id"]),
            int(row["screen_rank_within_task"]),
        )
    )
    if len(selected) != len(TASK_IDS) * PAIRS_PER_TASK:
        raise AssertionError("Frozen pair selection has the wrong size")
    return selected


def protocol_payload(
    *,
    screen_candidates: list[dict[str, Any]],
    policy_sha256: str,
    backbone_sha256: str,
    stage17_sha256: str,
    stage18_report_sha256: str,
    stage18_episodes_sha256: str,
    stage19_report_sha256: str,
    stage19_extension_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite": "libero_spatial",
        "task_ids": list(TASK_IDS),
        "fresh_reset_screen_candidates_per_task": SCREEN_CANDIDATES_PER_TASK,
        "fresh_environment_instance_per_screen_episode": True,
        "fresh_screen_must_finish_full_block": True,
        "pairs_per_task": PAIRS_PER_TASK,
        "conditions": list(CONDITIONS),
        "fresh_reset_screen_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "benchmark_task_id": row["benchmark_task_id"],
                "init_state_index": row["init_state_index"],
                "episode_seed": row["episode_seed"],
            }
            for row in screen_candidates
        ],
        "pair_freeze_rule": (
            "Within each task, take the first two candidates in the pre-registered "
            "hash order that succeed after control step 50 in the complete fresh-reset screen."
        ),
        "fresh_environment_instance_per_full_episode": True,
        "selection_salt": SELECTION_SALT,
        "selection_uses_world_model_score": False,
        "n_action_steps": STAGE18.ACTION_HORIZON,
        "resolution": STAGE18.RESOLUTION,
        "episode_batch_size": 1,
        "intervention": {
            "name": "persistent_motion_dropout",
            "start_control_step_zero_based": INTERVENTION_START,
            "duration": "until episode termination or time limit",
            "commanded_action_saved": True,
            "executed_action_saved": True,
            "executed_arm_dimensions_0_to_5": 0.0,
            "executed_gripper_dimension_6": "preserve commanded value",
        },
        "minimum_fault_failures_for_viability": MIN_FAULT_FAILURES,
        "policy_sha256": policy_sha256,
        "backbone_sha256": backbone_sha256,
        "stage17_report_sha256": stage17_sha256,
        "stage18_report_sha256": stage18_report_sha256,
        "stage18_episodes_sha256": stage18_episodes_sha256,
        "stage19_report_sha256": stage19_report_sha256,
        "stage19_extension_sha256": stage19_extension_sha256,
    }


def episode_paths(
    output_dir: Path,
    pair_id: str,
    condition: str,
) -> tuple[Path, Path]:
    base = output_dir / "episodes" / f"{pair_id}_{condition}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def screen_record_path(output_dir: Path, candidate_id: str) -> Path:
    return output_dir / "fresh_reset_screen" / f"{candidate_id}.json"


def validate_screen_record(
    *,
    output_dir: Path,
    candidate: dict[str, Any],
    protocol_sha256: str,
) -> dict[str, Any] | None:
    path = screen_record_path(output_dir, candidate["candidate_id"])
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "candidate_id": candidate["candidate_id"],
        "benchmark_task_id": int(candidate["benchmark_task_id"]),
        "init_state_index": int(candidate["init_state_index"]),
        "episode_seed": int(candidate["episode_seed"]),
        "protocol_sha256": protocol_sha256,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"Fresh-reset screen mismatch for {key}")
    if int(record["steps"]) <= 0 or int(record["steps"]) > 280:
        raise AssertionError("Fresh-reset screen has an invalid episode length")
    for key in ("initial_observation_sha256", "action_sha256"):
        if len(str(record[key])) != 64:
            raise AssertionError(f"Fresh-reset screen has an invalid {key}")
    return record


def validate_episode_arrays(
    arrays: dict[str, np.ndarray],
    *,
    condition: str,
    resolution: int,
) -> None:
    required = {
        *STAGE16.CAMERA_KEYS,
        *STAGE16.STATE_PATHS,
        "commanded_action",
        "executed_action",
        "reward",
        "success",
        "done",
        "transition_valid",
        "intervention_mask",
    }
    if set(arrays) != required:
        raise AssertionError(f"Episode keys differ: {sorted(set(arrays) ^ required)}")
    steps = len(arrays["commanded_action"])
    observations = steps + 1
    for key in ("commanded_action", "executed_action"):
        if arrays[key].shape != (steps, 7) or arrays[key].dtype != np.float32:
            raise AssertionError(f"Unexpected {key}: {arrays[key].shape} {arrays[key].dtype}")
        if not np.isfinite(arrays[key]).all():
            raise AssertionError(f"Non-finite {key}")
    for key in ("reward", "success", "done", "transition_valid", "intervention_mask"):
        if arrays[key].shape != (steps,):
            raise AssertionError(f"Unexpected {key}: {arrays[key].shape}")
    for camera in STAGE16.CAMERA_KEYS:
        expected = (observations, resolution, resolution, 3)
        if arrays[camera].shape != expected or arrays[camera].dtype != np.uint8:
            raise AssertionError(f"Unexpected {camera}: {arrays[camera].shape}")
    for state_name in STAGE16.STATE_PATHS:
        if len(arrays[state_name]) != observations or arrays[state_name].dtype != np.float32:
            raise AssertionError(f"Invalid state array {state_name}")
        if not np.isfinite(arrays[state_name]).all():
            raise AssertionError(f"Non-finite state array {state_name}")
    if not bool(arrays["done"][-1]):
        raise AssertionError("Final retained step must be done")
    episode_success = bool(arrays["success"].any())
    expected_valid = STAGE16.transition_valid_mask(steps, episode_success)
    if not np.array_equal(arrays["transition_valid"], expected_valid):
        raise AssertionError("transition_valid differs from the reset-crossing rule")

    mask = arrays["intervention_mask"]
    if condition == "nominal":
        if bool(mask.any()) or not np.array_equal(arrays["commanded_action"], arrays["executed_action"]):
            raise AssertionError("Nominal episode unexpectedly modifies actions")
    elif condition == "persistent_motion_dropout":
        expected_mask = np.arange(steps) >= INTERVENTION_START
        if not np.array_equal(mask, expected_mask):
            raise AssertionError("Intervention mask differs from the pinned schedule")
        if not np.array_equal(
            arrays["commanded_action"][:INTERVENTION_START],
            arrays["executed_action"][:INTERVENTION_START],
        ):
            raise AssertionError("Fault episode modifies pre-intervention actions")
        if not np.array_equal(
            arrays["executed_action"][INTERVENTION_START:, :6],
            np.zeros_like(arrays["executed_action"][INTERVENTION_START:, :6]),
        ):
            raise AssertionError("Fault arm-motion dimensions are not all zero")
        if not np.array_equal(
            arrays["commanded_action"][INTERVENTION_START:, 6],
            arrays["executed_action"][INTERVENTION_START:, 6],
        ):
            raise AssertionError("Fault episode changed the gripper command")
    else:
        raise ValueError(f"Unknown condition: {condition}")


def load_episode(
    path: Path,
    *,
    condition: str,
    resolution: int,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_episode_arrays(arrays, condition=condition, resolution=resolution)
    return arrays


def observation_prefix_sha256(
    arrays: dict[str, np.ndarray],
    *,
    observations: int,
) -> str:
    digest = hashlib.sha256()
    for key in (*STAGE16.CAMERA_KEYS, *STAGE16.STATE_PATHS):
        value = arrays[key][:observations]
        STAGE18.array_digest_update(digest, key, value)
    return digest.hexdigest()


def collect_condition(
    *,
    env: Any,
    policy: torch.nn.Module,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    task_id: int,
    init_state_index: int,
    condition: str,
) -> tuple[dict[str, list[np.ndarray]], dict[str, torch.Tensor], str]:
    seed = STAGE18.episode_seed(task_id, init_state_index)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    policy.reset()
    env.set_attr("init_state_id", [init_state_index])
    observation, _ = env.reset(seed=[seed])
    initial_hash = STAGE18.initial_observation_sha256(observation)
    captured: dict[str, list[np.ndarray]] = {key: [] for key in (*STAGE16.CAMERA_KEYS, *STAGE16.STATE_PATHS)}
    STAGE16.capture_observation(observation, captured)
    commanded_actions: list[torch.Tensor] = []
    executed_actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    successes: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    intervention_masks: list[torch.Tensor] = []
    done = np.zeros(1, dtype=np.bool_)
    max_steps = int(env.call("_max_episode_steps")[0])

    for step in range(max_steps):
        policy_observation = preprocess_observation(observation)
        try:
            policy_observation["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            policy_observation["task"] = list(env.call("task"))
        policy_observation = env_preprocessor(policy_observation)
        policy_observation = preprocessor(policy_observation)
        with torch.inference_mode():
            commanded = policy.select_action(policy_observation)
        commanded = postprocessor(commanded)
        commanded = env_postprocessor({ACTION: commanded})[ACTION]
        commanded_numpy = commanded.detach().float().cpu().numpy()
        if commanded_numpy.shape != (1, 7) or not np.isfinite(commanded_numpy).all():
            raise AssertionError(f"Invalid commanded action at step {step}")

        intervene = condition == "persistent_motion_dropout" and step >= INTERVENTION_START
        executed_numpy = commanded_numpy.copy()
        if intervene:
            executed_numpy[:, :6] = 0.0
        observation, reward, terminated, truncated, info = env.step(executed_numpy)
        STAGE16.capture_observation(observation, captured)
        step_successes = STAGE16.extract_successes(info, 1)
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=np.bool_)

        commanded_actions.append(torch.from_numpy(commanded_numpy.copy()))
        executed_actions.append(torch.from_numpy(executed_numpy.copy()))
        rewards.append(torch.from_numpy(np.asarray(reward).copy()))
        successes.append(torch.tensor(step_successes, dtype=torch.bool))
        dones.append(torch.from_numpy(done.copy()))
        intervention_masks.append(torch.tensor([intervene], dtype=torch.bool))
        if bool(done[0]):
            break

    return (
        captured,
        {
            "commanded_action": torch.cat(commanded_actions, dim=0),
            "executed_action": torch.cat(executed_actions, dim=0),
            "reward": torch.cat(rewards, dim=0),
            "success": torch.cat(successes, dim=0),
            "done": torch.cat(dones, dim=0),
            "intervention_mask": torch.cat(intervention_masks, dim=0),
        },
        initial_hash,
    )


def save_episode(
    *,
    output_dir: Path,
    pair: dict[str, Any],
    condition: str,
    captured: dict[str, list[np.ndarray]],
    rollout: dict[str, torch.Tensor],
    initial_hash: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    archive_path, metadata_path = episode_paths(output_dir, pair["pair_id"], condition)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    steps = len(rollout["commanded_action"])
    episode_success = bool(rollout["success"].any().item())
    arrays = {key: np.stack([batch[0] for batch in values], axis=0) for key, values in captured.items()}
    arrays.update(
        {
            "commanded_action": rollout["commanded_action"].float().numpy(),
            "executed_action": rollout["executed_action"].float().numpy(),
            "reward": rollout["reward"].float().numpy(),
            "success": rollout["success"].numpy().astype(np.bool_),
            "done": rollout["done"].numpy().astype(np.bool_),
            "transition_valid": STAGE16.transition_valid_mask(steps, episode_success),
            "intervention_mask": rollout["intervention_mask"].numpy().astype(np.bool_),
        }
    )
    validate_episode_arrays(arrays, condition=condition, resolution=STAGE18.RESOLUTION)
    content_hash = STAGE16.episode_content_sha256(arrays)
    temporary_path = archive_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    round_trip = load_episode(
        temporary_path,
        condition=condition,
        resolution=STAGE18.RESOLUTION,
    )
    if STAGE16.episode_content_sha256(round_trip) != content_hash:
        raise AssertionError("NPZ round-trip changed episode content")
    temporary_path.replace(archive_path)
    prefix_observations = min(INTERVENTION_START + 1, steps + 1)
    record = {
        "pair_id": pair["pair_id"],
        "rollout_id": f"{pair['pair_id']}-{condition}",
        "condition": condition,
        "benchmark_task_id": int(pair["benchmark_task_id"]),
        "task_name": pair["task_name"],
        "init_state_index": int(pair["init_state_index"]),
        "episode_seed": int(pair["episode_seed"]),
        "success": episode_success,
        "steps": steps,
        "valid_transitions": int(arrays["transition_valid"].sum()),
        "intervention_start": INTERVENTION_START if condition != "nominal" else None,
        "intervention_steps": int(arrays["intervention_mask"].sum()),
        "initial_observation_sha256": initial_hash,
        "preintervention_observation_sha256": observation_prefix_sha256(
            arrays,
            observations=prefix_observations,
        ),
        "commanded_action_sha256": STAGE18.action_trace_sha256(list(arrays["commanded_action"])),
        "executed_action_sha256": STAGE18.action_trace_sha256(list(arrays["executed_action"])),
        "preintervention_commanded_action_sha256": STAGE18.action_trace_sha256(
            list(arrays["commanded_action"][:INTERVENTION_START])
        ),
        "archive_file": archive_path.as_posix(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "content_sha256": content_hash,
        "protocol_sha256": protocol_sha256,
    }
    STAGE18.write_deterministic(metadata_path, canonical_json(record))
    return record


def validate_resume(
    *,
    output_dir: Path,
    pair: dict[str, Any],
    condition: str,
    protocol_sha256: str,
) -> dict[str, Any] | None:
    archive_path, metadata_path = episode_paths(output_dir, pair["pair_id"], condition)
    if not archive_path.exists() and not metadata_path.exists():
        return None
    if not archive_path.exists() or not metadata_path.exists():
        raise RuntimeError(f"Partial Stage 20 output for {pair['pair_id']} {condition}")
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "pair_id": pair["pair_id"],
        "condition": condition,
        "benchmark_task_id": int(pair["benchmark_task_id"]),
        "init_state_index": int(pair["init_state_index"]),
        "episode_seed": int(pair["episode_seed"]),
        "protocol_sha256": protocol_sha256,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"Resume metadata mismatch for {key}")
    arrays = load_episode(
        archive_path,
        condition=condition,
        resolution=STAGE18.RESOLUTION,
    )
    if sha256_file(archive_path) != record["archive_sha256"]:
        raise AssertionError("Resume archive file hash mismatch")
    if STAGE16.episode_content_sha256(arrays) != record["content_sha256"]:
        raise AssertionError("Resume archive content hash mismatch")
    return record


def sanitized_episode_row(record: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    archive = Path(record["archive_file"])
    try:
        relative = Path("outputs") / archive.relative_to(output_dir.parent.parent)
    except ValueError:
        relative = Path("outputs/wm/stage20_paired_interventions/episodes") / archive.name
    return {
        **{key: value for key, value in record.items() if key != "archive_file"},
        "local_artifact": relative.as_posix(),
        "has_dual_camera_observations": True,
        "has_complete_robot_state": True,
        "has_commanded_and_executed_actions": True,
        "world_model_scores_computed": False,
    }


def build_pair_row(
    pair: dict[str, Any],
    nominal: dict[str, Any],
    fault: dict[str, Any],
) -> dict[str, Any]:
    initial_match = (
        nominal["initial_observation_sha256"]
        == fault["initial_observation_sha256"]
        == pair["fresh_screen_initial_observation_sha256"]
    )
    pre_action_match = (
        nominal["preintervention_commanded_action_sha256"] == fault["preintervention_commanded_action_sha256"]
    )
    pre_observation_match = nominal["preintervention_observation_sha256"] == fault["preintervention_observation_sha256"]
    nominal_screen_match = (
        bool(nominal["success"])
        and int(nominal["steps"]) == int(pair["fresh_screen_steps"])
        and nominal["initial_observation_sha256"] == pair["fresh_screen_initial_observation_sha256"]
        and nominal["commanded_action_sha256"] == pair["fresh_screen_action_sha256"]
    )
    historical_scout_initial_matches = (
        nominal["initial_observation_sha256"] == pair["historical_scout_initial_observation_sha256"]
    )
    return {
        **pair,
        "nominal_success": bool(nominal["success"]),
        "nominal_steps": int(nominal["steps"]),
        "fault_success": bool(fault["success"]),
        "fault_steps": int(fault["steps"]),
        "fault_caused_failure": bool(nominal["success"]) and not bool(fault["success"]),
        "step_delta_fault_minus_nominal": int(fault["steps"]) - int(nominal["steps"]),
        "fault_intervention_steps": int(fault["intervention_steps"]),
        "initial_observation_matches_fresh_screen_and_pair": initial_match,
        "preintervention_commanded_actions_match": pre_action_match,
        "preintervention_observations_match": pre_observation_match,
        "nominal_reproduces_fresh_screen": nominal_screen_match,
        "historical_scout_initial_observation_matches_fresh_reset": (historical_scout_initial_matches),
        "nominal_content_sha256": nominal["content_sha256"],
        "fault_content_sha256": fault["content_sha256"],
        "complete_observations_available": True,
        "ready_for_frozen_wm_scoring": (
            initial_match and pre_action_match and pre_observation_match and nominal_screen_match
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 20 requires a CUDA GPU")
    if len(set(args.task_ids)) != len(args.task_ids) or any(task_id not in TASK_IDS for task_id in args.task_ids):
        raise ValueError(f"--task-ids must be a duplicate-free subset of {TASK_IDS}")

    stage17 = json.loads(args.stage17_report.read_text(encoding="utf-8"))
    stage18_report = json.loads(args.stage18_report.read_text(encoding="utf-8"))
    stage19_report = json.loads(args.stage19_report.read_text(encoding="utf-8"))
    if stage17.get("status") != "passed":
        raise AssertionError("Stage 17 must have passed")
    if stage18_report.get("status") != "insufficient_balance":
        raise AssertionError("Stage 18 input differs from the audited result")
    if stage19_report.get("status") != "insufficient_balance":
        raise AssertionError("Stage 19 input differs from the audited result")
    if stage19_report["stage20_readiness"]["paired_action_intervention_pilot_recommended"] is not True:
        raise AssertionError("Stage 19 did not recommend the paired intervention pilot")
    if stage17["primary_result"]["score"] != "h10_raw_mse":
        raise AssertionError("The frozen Stage 17 primary score changed")

    scout_rows = read_scout_rows(args.stage18_episodes, source_stage=18)
    scout_rows.extend(read_scout_rows(args.stage19_extension, source_stage=19))
    screen_candidates = select_screen_candidates(scout_rows)
    stage2_manifest = STAGE3.verify_stage2_manifest(args.stage2_manifest)
    backbone_manifest = json.loads((args.backbone_dir.parent / "backbone_manifest.json").read_text(encoding="utf-8"))
    if stage2_manifest["resolved_revision"] != STAGE3.EXPECTED_POLICY_REVISION:
        raise AssertionError("Policy revision differs from the pinned revision")
    if backbone_manifest["resolved_revision"] != STAGE3.BACKBONE_REVISION:
        raise AssertionError("Backbone revision differs from the pinned revision")

    protocol = protocol_payload(
        screen_candidates=screen_candidates,
        policy_sha256=sha256_file(args.checkpoint_dir / "model.safetensors"),
        backbone_sha256=sha256_file(args.backbone_dir / "model.safetensors"),
        stage17_sha256=sha256_file(args.stage17_report),
        stage18_report_sha256=sha256_file(args.stage18_report),
        stage18_episodes_sha256=sha256_file(args.stage18_episodes),
        stage19_report_sha256=sha256_file(args.stage19_report),
        stage19_extension_sha256=sha256_file(args.stage19_extension),
    )
    protocol_text = canonical_json(protocol)
    protocol_sha256 = sha256_text(protocol_text)
    STAGE18.write_deterministic(args.output_dir / "protocol.json", protocol_text)

    config = PreTrainedConfig.from_pretrained(args.checkpoint_dir, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    config.device = "cuda"
    config.vlm_model_name = str(args.backbone_dir.resolve())
    config.load_vlm_weights = True
    config.empty_cameras = 1
    config.use_amp = False
    config.n_action_steps = STAGE18.ACTION_HORIZON
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint_dir),
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": {}},
            "tokenizer_processor": {"tokenizer_name": str(args.backbone_dir.resolve())},
            "device_processor": {"device": "cuda"},
        },
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint_dir,
        config=config,
        local_files_only=True,
    )
    policy.eval()
    model_loaded = time.perf_counter()

    requested = set(args.task_ids)
    for candidate in screen_candidates:
        task_id = int(candidate["benchmark_task_id"])
        if task_id not in requested:
            continue
        resumed_screen = validate_screen_record(
            output_dir=args.output_dir,
            candidate=candidate,
            protocol_sha256=protocol_sha256,
        )
        if resumed_screen is not None:
            print(
                f"Stage 20 screen resume: {candidate['candidate_id']} success={resumed_screen['success']}",
                flush=True,
            )
            continue
        env_cfg, env, envs = STAGE5.make_task_env(
            suite="libero_spatial",
            task_id=task_id,
            n_envs=1,
            resolution=STAGE18.RESOLUTION,
        )
        try:
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, config)
            episode_started = time.perf_counter()
            screen = STAGE18.run_episode(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task_id=task_id,
                init_state_index=int(candidate["init_state_index"]),
            )
            screen.update(
                {
                    "candidate_id": candidate["candidate_id"],
                    "screen_rank_within_task": int(candidate["screen_rank_within_task"]),
                    "protocol_sha256": protocol_sha256,
                }
            )
            STAGE18.write_deterministic(
                screen_record_path(args.output_dir, candidate["candidate_id"]),
                canonical_json(screen),
            )
            print(
                f"Stage 20 screen {candidate['candidate_id']}: "
                f"success={screen['success']} steps={screen['steps']} "
                f"seconds={time.perf_counter() - episode_started:.1f}",
                flush=True,
            )
        finally:
            close_envs(envs)

    completed_screen_records = []
    for candidate in screen_candidates:
        screen = validate_screen_record(
            output_dir=args.output_dir,
            candidate=candidate,
            protocol_sha256=protocol_sha256,
        )
        if screen is not None:
            completed_screen_records.append(screen)
    if args.local_only and set(args.task_ids) != set(TASK_IDS):
        finished = time.perf_counter()
        runtime = {
            "schema_version": 1,
            "model_load_seconds": model_loaded - started,
            "total_seconds_this_invocation": finished - started,
            "peak_pytorch_gpu_allocation_bytes": torch.cuda.max_memory_allocated(),
            "task_ids_requested": args.task_ids,
            "screen_only": True,
        }
        (args.output_dir / "runtime.json").write_text(canonical_json(runtime), encoding="utf-8")
        print(f"Stage 20 local-only screen passed: tasks={args.task_ids}", flush=True)
        return
    if len(completed_screen_records) != len(screen_candidates):
        raise AssertionError("Formal Stage 20 fresh-reset screen is incomplete")
    selected = freeze_pairs(screen_candidates, completed_screen_records)

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in selected:
        for condition in CONDITIONS:
            resumed = validate_resume(
                output_dir=args.output_dir,
                pair=pair,
                condition=condition,
                protocol_sha256=protocol_sha256,
            )
            if resumed is not None:
                print(
                    f"Stage 20 resume: {pair['pair_id']} {condition}",
                    flush=True,
                )
                continue
            task_id = int(pair["benchmark_task_id"])
            env_cfg, env, envs = STAGE5.make_task_env(
                suite="libero_spatial",
                task_id=task_id,
                n_envs=1,
                resolution=STAGE18.RESOLUTION,
            )
            try:
                env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, config)
                episode_started = time.perf_counter()
                captured, rollout, initial_hash = collect_condition(
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task_id=task_id,
                    init_state_index=int(pair["init_state_index"]),
                    condition=condition,
                )
                record = save_episode(
                    output_dir=args.output_dir,
                    pair=pair,
                    condition=condition,
                    captured=captured,
                    rollout=rollout,
                    initial_hash=initial_hash,
                    protocol_sha256=protocol_sha256,
                )
                print(
                    f"Stage 20 {pair['pair_id']} {condition}: "
                    f"success={record['success']} steps={record['steps']} "
                    f"seconds={time.perf_counter() - episode_started:.1f}",
                    flush=True,
                )
            finally:
                close_envs(envs)

    finished = time.perf_counter()
    runtime = {
        "schema_version": 1,
        "model_load_seconds": model_loaded - started,
        "total_seconds_this_invocation": finished - started,
        "peak_pytorch_gpu_allocation_bytes": torch.cuda.max_memory_allocated(),
        "task_ids_requested": args.task_ids,
    }
    (args.output_dir / "runtime.json").write_text(canonical_json(runtime), encoding="utf-8")
    if args.local_only:
        print(f"Stage 20 local-only passed: tasks={args.task_ids}", flush=True)
        return
    if set(args.task_ids) != set(TASK_IDS):
        raise ValueError("Public finalization requires all pinned tasks")

    for pair in selected:
        for condition in CONDITIONS:
            record = validate_resume(
                output_dir=args.output_dir,
                pair=pair,
                condition=condition,
                protocol_sha256=protocol_sha256,
            )
            if record is None:
                raise AssertionError(f"Missing formal episode: {pair['pair_id']} {condition}")
            records[(pair["pair_id"], condition)] = record

    pair_rows = [
        build_pair_row(
            pair,
            records[(pair["pair_id"], "nominal")],
            records[(pair["pair_id"], "persistent_motion_dropout")],
        )
        for pair in selected
    ]
    if not all(row["ready_for_frozen_wm_scoring"] for row in pair_rows):
        raise AssertionError("At least one pair failed fresh-reset/pre-intervention alignment")
    fault_failures = sum(bool(row["fault_caused_failure"]) for row in pair_rows)
    intervention_viable = fault_failures >= MIN_FAULT_FAILURES
    candidate_lookup = {candidate["candidate_id"]: candidate for candidate in screen_candidates}
    screen_public_rows = []
    for screen in completed_screen_records:
        candidate = candidate_lookup[screen["candidate_id"]]
        screen_public_rows.append(
            {
                "candidate_id": screen["candidate_id"],
                "benchmark_task_id": int(screen["benchmark_task_id"]),
                "task_name": screen["task_name"],
                "init_state_index": int(screen["init_state_index"]),
                "episode_seed": int(screen["episode_seed"]),
                "screen_rank_within_task": int(screen["screen_rank_within_task"]),
                "success": bool(screen["success"]),
                "steps": int(screen["steps"]),
                "initial_observation_sha256": screen["initial_observation_sha256"],
                "action_sha256": screen["action_sha256"],
                "historical_scout_success": True,
                "historical_scout_steps": int(candidate["scout_steps"]),
                "historical_scout_initial_observation_sha256": candidate["scout_initial_observation_sha256"],
                "initial_observation_matches_historical_scout": (
                    screen["initial_observation_sha256"] == candidate["scout_initial_observation_sha256"]
                ),
                "selected_for_full_pair": any(pair["pair_id"] == screen["candidate_id"] for pair in selected),
                "world_model_scores_computed": False,
            }
        )
    screen_public_rows.sort(
        key=lambda row: (
            int(row["benchmark_task_id"]),
            int(row["screen_rank_within_task"]),
        )
    )
    public_episode_rows = [
        sanitized_episode_row(records[(pair["pair_id"], condition)], args.output_dir)
        for pair in selected
        for condition in CONDITIONS
    ]
    episodes_csv = csv_text(public_episode_rows)
    pairs_csv = csv_text(pair_rows)
    screen_csv = csv_text(screen_public_rows)
    total_bytes = sum(int(row["archive_bytes"]) for row in public_episode_rows)
    report = {
        "schema_version": 1,
        "stage": "WM Stage 20 paired persistent motion-dropout recollection",
        "status": "passed" if intervention_viable else "intervention_not_viable",
        "scope": {
            "task_ids": list(TASK_IDS),
            "fresh_reset_screen_episodes": len(screen_public_rows),
            "fresh_reset_screen_successes": sum(bool(row["success"]) for row in screen_public_rows),
            "pairs": len(pair_rows),
            "episodes": len(public_episode_rows),
            "nominal_episodes": len(pair_rows),
            "fault_episodes": len(pair_rows),
            "fault_failures": fault_failures,
            "raw_archive_bytes": total_bytes,
            "test_demonstration_episodes_used": 0,
        },
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "alignment": {
            "all_nominal_episodes_reproduce_fresh_screen": all(
                row["nominal_reproduces_fresh_screen"] for row in pair_rows
            ),
            "all_initial_observations_match_fresh_screen_and_pair": all(
                row["initial_observation_matches_fresh_screen_and_pair"] for row in pair_rows
            ),
            "all_preintervention_commanded_actions_match": all(
                row["preintervention_commanded_actions_match"] for row in pair_rows
            ),
            "all_preintervention_observations_match": all(
                row["preintervention_observations_match"] for row in pair_rows
            ),
            "action_t_maps_observation_t_to_t_plus_1": True,
            "successful_final_transition_valid": False,
        },
        "reset_history_audit": {
            "historical_scout_initial_observations_matching_fresh_reset": sum(
                bool(row["initial_observation_matches_historical_scout"]) for row in screen_public_rows
            ),
            "screen_episodes": len(screen_public_rows),
            "fresh_environment_instance_used_for_every_screen_and_full_episode": True,
            "reason": (
                "The first attempted direct reuse of a later Stage 18 state did not "
                "reproduce from a fresh simulator instance; formal Stage 20 therefore "
                "uses a complete fresh-reset eligibility screen."
            ),
        },
        "intervention_viability": {
            "minimum_fault_failures": MIN_FAULT_FAILURES,
            "observed_fault_failures": fault_failures,
            "passes": intervention_viable,
        },
        "modalities": {
            "camera1": "agentview uint8 RGB",
            "camera2": "eye-in-hand uint8 RGB",
            "robot_state": list(STAGE16.STATE_PATHS),
            "commanded_action": "7-D postprocessed SmolVLA command",
            "executed_action": "7-D action actually passed to LIBERO",
            "intervention_mask": "true from step 50 onward in fault condition",
        },
        "frozen_world_model": {
            "stage17_primary_score": stage17["primary_result"]["score"],
            "world_model_scores_computed_in_stage20": 0,
            "thresholds_fitted_in_stage20": 0,
            "model_parameters_updated": 0,
        },
        "artifacts": {
            "fresh_reset_screen_csv": "fresh_reset_screen.csv",
            "fresh_reset_screen_csv_sha256": sha256_text(screen_csv),
            "episodes_csv": "episodes.csv",
            "episodes_csv_sha256": sha256_text(episodes_csv),
            "pairs_csv": "pairs.csv",
            "pairs_csv_sha256": sha256_text(pairs_csv),
            "raw_episode_archives": "local-only under outputs/wm/stage20_paired_interventions",
            "raw_archives_committed_to_git": False,
        },
        "stage21_readiness": {
            "paired_complete_observations_available": True,
            "intervention_viable": intervention_viable,
            "frozen_wm_paired_scoring_can_start": intervention_viable,
        },
        "limitations": [
            "Persistent motion dropout is a synthetic actuator fault, not a natural policy failure.",
            "Only six episode pairs across three tasks are collected in this pilot.",
            "The policy observes the diverged post-fault trajectory and replans normally.",
            "No world-model score or detector result is reported in this collection stage.",
        ],
    }
    STAGE18.write_deterministic(
        args.public_results_dir / "fresh_reset_screen.csv",
        screen_csv,
        overwrite=args.overwrite_public,
    )
    STAGE18.write_deterministic(
        args.public_results_dir / "episodes.csv",
        episodes_csv,
        overwrite=args.overwrite_public,
    )
    STAGE18.write_deterministic(
        args.public_results_dir / "pairs.csv",
        pairs_csv,
        overwrite=args.overwrite_public,
    )
    STAGE18.write_deterministic(
        args.public_results_dir / "report.json",
        canonical_json(report),
        overwrite=args.overwrite_public,
    )
    print(
        f"Stage 20 {report['status']}: pairs={len(pair_rows)}, "
        f"fault_failures={fault_failures}/{len(pair_rows)}, "
        f"raw_bytes={total_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
