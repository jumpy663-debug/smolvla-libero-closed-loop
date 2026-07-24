#!/usr/bin/env python

"""Collect a score-blind confirmation cohort with two mild actuator faults.

No world-model feature or score is loaded. Twelve initial states never screened
in Stage 20 are fresh-reset evaluated; the first two eligible successes per
task are frozen. Each selected state is then run under nominal control, 0.5x
arm-motion attenuation, and a three-step arm-action delay.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
CONDITIONS = ("nominal", "motion_attenuation_0p5", "action_delay_3")
INTERVENTION_START = 50
CONFIRMATORY_POST_END_EXCLUSIVE = 89
MINIMUM_SCREEN_STEPS = 90
ATTENUATION_SCALE = 0.5
ACTION_DELAY_STEPS = 3
SELECTION_SALT = "stage22-confirmatory-mild-interventions-v1"
EXPECTED_CANDIDATE_INDICES = {
    4: (7, 4, 12, 11),
    7: (5, 9, 3, 12),
    8: (11, 26, 18, 13),
}


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


STAGE20 = load_module(
    "stage20_collect_paired_interventions.py",
    "stage20_for_stage22",
)
STAGE18 = STAGE20.STAGE18
STAGE16 = STAGE20.STAGE16
STAGE5 = STAGE20.STAGE5
STAGE3 = STAGE20.STAGE3


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
        "--stage18-episodes",
        type=Path,
        default=Path("results/wm/stage18_initial_state_scout/scout_episodes.csv"),
    )
    parser.add_argument(
        "--stage19-extension",
        type=Path,
        default=Path("results/wm/stage19_balanced_cohort/task8_extension.csv"),
    )
    parser.add_argument(
        "--stage20-screen",
        type=Path,
        default=Path("results/wm/stage20_paired_interventions/fresh_reset_screen.csv"),
    )
    parser.add_argument(
        "--stage20-report",
        type=Path,
        default=Path("results/wm/stage20_paired_interventions/report.json"),
    )
    parser.add_argument(
        "--stage21-report",
        type=Path,
        default=Path("results/wm/stage21_paired_wm_scoring/report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wm/stage22_confirmatory_interventions"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage22_confirmatory_interventions"),
    )
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(TASK_IDS))
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--overwrite-public", action="store_true")
    return parser.parse_args()


def canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    return STAGE20.sha256_file(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def pair_selection_key(row: dict[str, Any]) -> str:
    payload = f"{SELECTION_SALT}|task={int(row['benchmark_task_id'])}|init={int(row['init_state_index'])}"
    return hashlib.sha256(payload.encode()).hexdigest()


def select_screen_candidates(
    rows: list[dict[str, Any]],
    excluded_keys: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    keys = [(int(row["benchmark_task_id"]), int(row["init_state_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("Scout pool contains duplicate task/initial-state keys")
    selected = []
    for task_id in TASK_IDS:
        candidates = sorted(
            (
                row
                for row in rows
                if int(row["benchmark_task_id"]) == task_id
                and bool(row["success"])
                and int(row["steps"]) >= MINIMUM_SCREEN_STEPS
                and (task_id, int(row["init_state_index"])) not in excluded_keys
            ),
            key=pair_selection_key,
        )
        if len(candidates) < SCREEN_CANDIDATES_PER_TASK:
            raise AssertionError(f"Task {task_id} has too few unused eligible states")
        for rank, row in enumerate(candidates[:SCREEN_CANDIDATES_PER_TASK]):
            selected.append(
                {
                    "candidate_id": (f"task{task_id:02d}-init{int(row['init_state_index']):02d}"),
                    "benchmark_task_id": task_id,
                    "task_name": row["task_name"],
                    "init_state_index": int(row["init_state_index"]),
                    "episode_seed": int(row["episode_seed"]),
                    "screen_rank_within_task": rank,
                    "selection_key": pair_selection_key(row),
                    "historical_scout_steps": int(row["steps"]),
                    "historical_scout_initial_observation_sha256": row["initial_observation_sha256"],
                    "selection_uses_world_model_score": False,
                }
            )
    selected.sort(
        key=lambda row: (
            int(row["benchmark_task_id"]),
            int(row["screen_rank_within_task"]),
        )
    )
    observed = {
        task_id: tuple(int(row["init_state_index"]) for row in selected if int(row["benchmark_task_id"]) == task_id)
        for task_id in TASK_IDS
    }
    if observed != EXPECTED_CANDIDATE_INDICES:
        raise AssertionError(f"Candidate cohort changed: {observed}")
    return selected


def freeze_pairs(
    candidates: list[dict[str, Any]],
    screen_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {(int(row["benchmark_task_id"]), int(row["init_state_index"])): row for row in screen_records}
    selected = []
    for task_id in TASK_IDS:
        eligible = []
        for candidate in candidates:
            if int(candidate["benchmark_task_id"]) != task_id:
                continue
            screen = lookup[(task_id, int(candidate["init_state_index"]))]
            if bool(screen["success"]) and int(screen["steps"]) >= MINIMUM_SCREEN_STEPS:
                eligible.append((candidate, screen))
        if len(eligible) < PAIRS_PER_TASK:
            raise AssertionError(f"Task {task_id} has only {len(eligible)} scoreable fresh successes")
        for candidate, screen in eligible[:PAIRS_PER_TASK]:
            selected.append(
                {
                    **candidate,
                    "pair_id": candidate["candidate_id"],
                    "fresh_screen_steps": int(screen["steps"]),
                    "fresh_screen_initial_observation_sha256": screen["initial_observation_sha256"],
                    "fresh_screen_action_sha256": screen["action_sha256"],
                }
            )
    if len(selected) != len(TASK_IDS) * PAIRS_PER_TASK:
        raise AssertionError("Frozen confirmation cohort has the wrong size")
    return selected


def protocol_payload(
    *,
    candidates: list[dict[str, Any]],
    policy_sha256: str,
    backbone_sha256: str,
    stage20_report_sha256: str,
    stage21_report_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite": "libero_spatial",
        "task_ids": list(TASK_IDS),
        "fresh_reset_screen_candidates_per_task": SCREEN_CANDIDATES_PER_TASK,
        "fresh_reset_screen_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "benchmark_task_id": row["benchmark_task_id"],
                "init_state_index": row["init_state_index"],
                "episode_seed": row["episode_seed"],
            }
            for row in candidates
        ],
        "stage20_screen_states_excluded": True,
        "pair_freeze_rule": (
            "Within each task, take the first two candidates in the pinned hash "
            "order that fresh-reset succeed with at least 90 control steps."
        ),
        "selection_salt": SELECTION_SALT,
        "selection_uses_world_model_score": False,
        "world_model_loaded_during_collection": False,
        "conditions": list(CONDITIONS),
        "fresh_environment_instance_per_episode": True,
        "intervention_start_zero_based": INTERVENTION_START,
        "confirmatory_h10_post_starts": [50, 79],
        "required_valid_transitions_through_index": (CONFIRMATORY_POST_END_EXCLUSIVE - 1),
        "interventions": {
            "motion_attenuation_0p5": {
                "arm_dimensions_0_to_5": "0.5 * current commanded action",
                "gripper_dimension_6": "current commanded action",
                "duration": "from step 50 until termination",
            },
            "action_delay_3": {
                "arm_dimensions_0_to_5": "commanded arm action from t-3",
                "gripper_dimension_6": "current commanded action",
                "duration": "from step 50 until termination",
            },
        },
        "stage23_candidate_score": ("fault-post commanded_h10_raw_mse minus executed_h10_raw_mse"),
        "stage23_candidate_direction": "positive",
        "stage23_discovery_data_reused": False,
        "n_action_steps": STAGE18.ACTION_HORIZON,
        "resolution": STAGE18.RESOLUTION,
        "episode_batch_size": 1,
        "policy_sha256": policy_sha256,
        "backbone_sha256": backbone_sha256,
        "stage20_report_sha256": stage20_report_sha256,
        "stage21_report_sha256": stage21_report_sha256,
    }


def screen_record_path(output_dir: Path, candidate_id: str) -> Path:
    return output_dir / "fresh_reset_screen" / f"{candidate_id}.json"


def validate_screen_record(
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
            raise AssertionError(f"Screen resume mismatch: {key}")
    return record


def episode_paths(
    output_dir: Path,
    pair_id: str,
    condition: str,
) -> tuple[Path, Path]:
    base = output_dir / "episodes" / f"{pair_id}_{condition}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def apply_intervention(
    commanded: np.ndarray,
    command_history: list[np.ndarray],
    *,
    step: int,
    condition: str,
) -> tuple[np.ndarray, bool]:
    if commanded.shape != (1, 7):
        raise ValueError(f"Unexpected commanded action shape: {commanded.shape}")
    executed = commanded.copy()
    intervene = condition != "nominal" and step >= INTERVENTION_START
    if not intervene:
        return executed, False
    if condition == "motion_attenuation_0p5":
        executed[:, :6] = commanded[:, :6] * ATTENUATION_SCALE
    elif condition == "action_delay_3":
        if len(command_history) < ACTION_DELAY_STEPS:
            raise AssertionError("Insufficient command history for delayed action")
        executed[:, :6] = command_history[-ACTION_DELAY_STEPS][:, :6]
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return executed, True


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
    captured = {key: [] for key in (*STAGE16.CAMERA_KEYS, *STAGE16.STATE_PATHS)}
    STAGE16.capture_observation(observation, captured)
    commanded_actions = []
    executed_actions = []
    rewards = []
    successes = []
    dones = []
    intervention_masks = []
    command_history: list[np.ndarray] = []
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
        executed_numpy, intervene = apply_intervention(
            commanded_numpy,
            command_history,
            step=step,
            condition=condition,
        )
        command_history.append(commanded_numpy.copy())
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
            "commanded_action": torch.cat(commanded_actions),
            "executed_action": torch.cat(executed_actions),
            "reward": torch.cat(rewards),
            "success": torch.cat(successes),
            "done": torch.cat(dones),
            "intervention_mask": torch.cat(intervention_masks),
        },
        initial_hash,
    )


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
            raise AssertionError(f"Unexpected {key}: {arrays[key].shape}")
        if not np.isfinite(arrays[key]).all():
            raise AssertionError(f"Non-finite {key}")
    for camera in STAGE16.CAMERA_KEYS:
        if arrays[camera].shape != (
            observations,
            resolution,
            resolution,
            3,
        ):
            raise AssertionError(f"Unexpected camera array: {camera}")
    for state_name in STAGE16.STATE_PATHS:
        if len(arrays[state_name]) != observations:
            raise AssertionError(f"Unexpected state array: {state_name}")
    episode_success = bool(arrays["success"].any())
    expected_valid = STAGE16.transition_valid_mask(steps, episode_success)
    if not np.array_equal(arrays["transition_valid"], expected_valid):
        raise AssertionError("transition_valid differs")
    commanded = arrays["commanded_action"]
    executed = arrays["executed_action"]
    mask = arrays["intervention_mask"]
    expected_mask = (
        np.zeros(steps, dtype=np.bool_) if condition == "nominal" else np.arange(steps) >= INTERVENTION_START
    )
    if not np.array_equal(mask, expected_mask):
        raise AssertionError("Intervention mask differs")
    if not np.array_equal(
        commanded[:INTERVENTION_START],
        executed[:INTERVENTION_START],
    ):
        raise AssertionError("Pre-intervention actions differ")
    if condition == "nominal":
        if not np.array_equal(commanded, executed):
            raise AssertionError("Nominal action changed")
    elif condition == "motion_attenuation_0p5":
        if not np.array_equal(
            executed[INTERVENTION_START:, :6],
            commanded[INTERVENTION_START:, :6] * ATTENUATION_SCALE,
        ):
            raise AssertionError("Attenuated arm actions differ")
        if not np.array_equal(
            executed[INTERVENTION_START:, 6],
            commanded[INTERVENTION_START:, 6],
        ):
            raise AssertionError("Attenuation changed gripper actions")
    elif condition == "action_delay_3":
        if not np.array_equal(
            executed[INTERVENTION_START:, :6],
            commanded[
                INTERVENTION_START - ACTION_DELAY_STEPS : -ACTION_DELAY_STEPS,
                :6,
            ],
        ):
            raise AssertionError("Delayed arm actions differ")
        if not np.array_equal(
            executed[INTERVENTION_START:, 6],
            commanded[INTERVENTION_START:, 6],
        ):
            raise AssertionError("Delay changed gripper actions")
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
    return STAGE20.observation_prefix_sha256(
        arrays,
        observations=observations,
    )


def post_diagnostics(arrays: dict[str, np.ndarray]) -> dict[str, float | int]:
    if int(arrays["transition_valid"].sum()) < CONFIRMATORY_POST_END_EXCLUSIVE:
        raise AssertionError("Episode does not cover the confirmatory post window")
    begin = INTERVENTION_START
    end = CONFIRMATORY_POST_END_EXCLUSIVE
    action_delta = arrays["commanded_action"][begin:end, :6] - arrays["executed_action"][begin:end, :6]
    action_l2 = np.linalg.norm(action_delta, axis=1)
    eef_delta = np.diff(arrays["eef_pos"][begin : end + 1], axis=0)
    eef_motion = np.linalg.norm(eef_delta, axis=1)
    pixel_motion = []
    for camera in STAGE16.CAMERA_KEYS:
        frames = arrays[camera][begin : end + 1].astype(np.float32)
        pixel_motion.append(np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3)))
    return {
        "diagnostic_transitions": end - begin,
        "post_arm_action_mismatch_l2_mean": float(action_l2.mean()),
        "post_arm_action_mismatch_l2_max": float(action_l2.max()),
        "post_eef_motion_l2_mean": float(eef_motion.mean()),
        "post_dual_camera_pixel_mae_mean": float(np.concatenate(pixel_motion).mean()),
    }


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
    archive_path, metadata_path = episode_paths(
        output_dir,
        pair["pair_id"],
        condition,
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    steps = len(rollout["commanded_action"])
    success = bool(rollout["success"].any())
    arrays = {key: np.stack([batch[0] for batch in values]) for key, values in captured.items()}
    arrays.update(
        {
            "commanded_action": rollout["commanded_action"].float().numpy(),
            "executed_action": rollout["executed_action"].float().numpy(),
            "reward": rollout["reward"].float().numpy(),
            "success": rollout["success"].numpy().astype(np.bool_),
            "done": rollout["done"].numpy().astype(np.bool_),
            "transition_valid": STAGE16.transition_valid_mask(steps, success),
            "intervention_mask": rollout["intervention_mask"].numpy().astype(np.bool_),
        }
    )
    validate_episode_arrays(
        arrays,
        condition=condition,
        resolution=STAGE18.RESOLUTION,
    )
    diagnostics = post_diagnostics(arrays)
    content_hash = STAGE16.episode_content_sha256(arrays)
    temporary = archive_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    round_trip = load_episode(
        temporary,
        condition=condition,
        resolution=STAGE18.RESOLUTION,
    )
    if STAGE16.episode_content_sha256(round_trip) != content_hash:
        raise AssertionError("NPZ round trip changed content")
    temporary.replace(archive_path)
    record = {
        "pair_id": pair["pair_id"],
        "rollout_id": f"{pair['pair_id']}-{condition}",
        "condition": condition,
        "benchmark_task_id": int(pair["benchmark_task_id"]),
        "task_name": pair["task_name"],
        "init_state_index": int(pair["init_state_index"]),
        "episode_seed": int(pair["episode_seed"]),
        "success": success,
        "steps": steps,
        "valid_transitions": int(arrays["transition_valid"].sum()),
        "intervention_start": (INTERVENTION_START if condition != "nominal" else None),
        "intervention_steps": int(arrays["intervention_mask"].sum()),
        "initial_observation_sha256": initial_hash,
        "preintervention_observation_sha256": observation_prefix_sha256(
            arrays,
            observations=INTERVENTION_START + 1,
        ),
        "commanded_action_sha256": STAGE18.action_trace_sha256(list(arrays["commanded_action"])),
        "executed_action_sha256": STAGE18.action_trace_sha256(list(arrays["executed_action"])),
        "preintervention_commanded_action_sha256": (
            STAGE18.action_trace_sha256(list(arrays["commanded_action"][:INTERVENTION_START]))
        ),
        **diagnostics,
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
    archive_path, metadata_path = episode_paths(
        output_dir,
        pair["pair_id"],
        condition,
    )
    if not archive_path.exists() and not metadata_path.exists():
        return None
    if not archive_path.exists() or not metadata_path.exists():
        raise RuntimeError(f"Partial output: {pair['pair_id']} {condition}")
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "pair_id": pair["pair_id"],
        "condition": condition,
        "benchmark_task_id": int(pair["benchmark_task_id"]),
        "init_state_index": int(pair["init_state_index"]),
        "protocol_sha256": protocol_sha256,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"Resume mismatch: {key}")
    arrays = load_episode(
        archive_path,
        condition=condition,
        resolution=STAGE18.RESOLUTION,
    )
    if sha256_file(archive_path) != record["archive_sha256"]:
        raise AssertionError("Resume file hash differs")
    if STAGE16.episode_content_sha256(arrays) != record["content_sha256"]:
        raise AssertionError("Resume content hash differs")
    return record


def sanitized_episode_row(
    record: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    archive = Path(record["archive_file"])
    try:
        relative = Path("outputs") / archive.relative_to(output_dir.parent.parent)
    except ValueError:
        relative = Path("outputs/wm/stage22_confirmatory_interventions/episodes") / archive.name
    return {
        **{key: value for key, value in record.items() if key != "archive_file"},
        "local_artifact": relative.as_posix(),
        "has_complete_observations": True,
        "world_model_scores_computed": False,
    }


def build_pair_row(
    pair: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    nominal = records["nominal"]
    initial_match = all(
        row["initial_observation_sha256"] == pair["fresh_screen_initial_observation_sha256"] for row in records.values()
    )
    pre_action_match = len({row["preintervention_commanded_action_sha256"] for row in records.values()}) == 1
    pre_observation_match = len({row["preintervention_observation_sha256"] for row in records.values()}) == 1
    nominal_screen_match = (
        bool(nominal["success"])
        and int(nominal["steps"]) == int(pair["fresh_screen_steps"])
        and nominal["commanded_action_sha256"] == pair["fresh_screen_action_sha256"]
    )
    result: dict[str, Any] = {
        **pair,
        "all_initial_observations_match": initial_match,
        "all_preintervention_commanded_actions_match": pre_action_match,
        "all_preintervention_observations_match": pre_observation_match,
        "nominal_reproduces_fresh_screen": nominal_screen_match,
    }
    for condition in CONDITIONS:
        row = records[condition]
        prefix = condition
        result[f"{prefix}_success"] = bool(row["success"])
        result[f"{prefix}_steps"] = int(row["steps"])
        result[f"{prefix}_valid_transitions"] = int(row["valid_transitions"])
        result[f"{prefix}_post_action_mismatch_l2_mean"] = float(row["post_arm_action_mismatch_l2_mean"])
        result[f"{prefix}_post_eef_motion_l2_mean"] = float(row["post_eef_motion_l2_mean"])
        result[f"{prefix}_post_pixel_mae_mean"] = float(row["post_dual_camera_pixel_mae_mean"])
        result[f"{prefix}_content_sha256"] = row["content_sha256"]
    nominal_eef = float(nominal["post_eef_motion_l2_mean"])
    nominal_pixels = float(nominal["post_dual_camera_pixel_mae_mean"])
    for condition in CONDITIONS[1:]:
        result[f"{condition}_eef_motion_ratio_vs_nominal"] = float(
            records[condition]["post_eef_motion_l2_mean"] / nominal_eef
        )
        result[f"{condition}_pixel_motion_ratio_vs_nominal"] = float(
            records[condition]["post_dual_camera_pixel_mae_mean"] / nominal_pixels
        )
    result["ready_for_confirmatory_wm_scoring"] = (
        initial_match
        and pre_action_match
        and pre_observation_match
        and nominal_screen_match
        and all(int(row["valid_transitions"]) >= CONFIRMATORY_POST_END_EXCLUSIVE for row in records.values())
        and all(float(records[condition]["post_arm_action_mismatch_l2_mean"]) > 0 for condition in CONDITIONS[1:])
    )
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 22 requires CUDA")
    if len(set(args.task_ids)) != len(args.task_ids) or any(task_id not in TASK_IDS for task_id in args.task_ids):
        raise ValueError(f"--task-ids must be a subset of {TASK_IDS}")
    stage20_report = json.loads(args.stage20_report.read_text())
    stage21_report = json.loads(args.stage21_report.read_text())
    if stage20_report["status"] != "passed":
        raise AssertionError("Stage 20 is not ready")
    if stage21_report["conclusion"]["decision_rule_passed"]:
        raise AssertionError("Stage 21 unexpectedly passed the original score")
    if stage21_report["stage22_readiness"]["online_shield_prototype_justified"]:
        raise AssertionError("Stage 21 unexpectedly authorized a shield")

    scout_rows = STAGE20.read_scout_rows(args.stage18_episodes, source_stage=18)
    scout_rows.extend(STAGE20.read_scout_rows(args.stage19_extension, source_stage=19))
    with args.stage20_screen.open(newline="", encoding="utf-8") as handle:
        import csv

        excluded = {(int(row["benchmark_task_id"]), int(row["init_state_index"])) for row in csv.DictReader(handle)}
    candidates = select_screen_candidates(scout_rows, excluded)

    stage2_manifest = STAGE3.verify_stage2_manifest(args.stage2_manifest)
    backbone_manifest = json.loads((args.backbone_dir.parent / "backbone_manifest.json").read_text())
    if stage2_manifest["resolved_revision"] != STAGE3.EXPECTED_POLICY_REVISION:
        raise AssertionError("Policy revision changed")
    if backbone_manifest["resolved_revision"] != STAGE3.BACKBONE_REVISION:
        raise AssertionError("Backbone revision changed")
    protocol = protocol_payload(
        candidates=candidates,
        policy_sha256=sha256_file(args.checkpoint_dir / "model.safetensors"),
        backbone_sha256=sha256_file(args.backbone_dir / "model.safetensors"),
        stage20_report_sha256=sha256_file(args.stage20_report),
        stage21_report_sha256=sha256_file(args.stage21_report),
    )
    protocol_text = canonical_json(protocol)
    protocol_sha256 = sha256_text(protocol_text)
    STAGE18.write_deterministic(args.output_dir / "protocol.json", protocol_text)

    config = PreTrainedConfig.from_pretrained(
        args.checkpoint_dir,
        local_files_only=True,
    )
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
    for candidate in candidates:
        task_id = int(candidate["benchmark_task_id"])
        if task_id not in requested:
            continue
        resumed = validate_screen_record(
            args.output_dir,
            candidate,
            protocol_sha256,
        )
        if resumed is not None:
            print(
                f"Stage 22 screen resume {candidate['candidate_id']}: "
                f"success={resumed['success']} steps={resumed['steps']}",
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
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg,
                config,
            )
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
                screen_record_path(
                    args.output_dir,
                    candidate["candidate_id"],
                ),
                canonical_json(screen),
            )
            print(
                f"Stage 22 screen {candidate['candidate_id']}: "
                f"success={screen['success']} steps={screen['steps']} "
                f"seconds={time.perf_counter() - episode_started:.1f}",
                flush=True,
            )
        finally:
            close_envs(envs)

    completed_screen = []
    for candidate in candidates:
        record = validate_screen_record(
            args.output_dir,
            candidate,
            protocol_sha256,
        )
        if record is not None:
            completed_screen.append(record)
    if args.local_only and requested != set(TASK_IDS):
        finished = time.perf_counter()
        (args.output_dir / "runtime.json").write_text(
            canonical_json(
                {
                    "model_load_seconds": model_loaded - started,
                    "total_seconds_this_invocation": finished - started,
                    "task_ids_requested": args.task_ids,
                    "screen_only": True,
                }
            )
        )
        print(f"Stage 22 partial screen complete: {args.task_ids}", flush=True)
        return
    if len(completed_screen) != len(candidates):
        raise AssertionError("Fresh-reset screen is incomplete")
    selected = freeze_pairs(candidates, completed_screen)

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
                    f"Stage 22 resume {pair['pair_id']} {condition}",
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
                    f"Stage 22 {pair['pair_id']} {condition}: "
                    f"success={record['success']} steps={record['steps']} "
                    f"mismatch={record['post_arm_action_mismatch_l2_mean']:.4f} "
                    f"seconds={time.perf_counter() - episode_started:.1f}",
                    flush=True,
                )
            finally:
                close_envs(envs)

    finished = time.perf_counter()
    (args.output_dir / "runtime.json").write_text(
        canonical_json(
            {
                "schema_version": 1,
                "model_load_seconds": model_loaded - started,
                "total_seconds_this_invocation": finished - started,
                "peak_pytorch_gpu_allocation_bytes": (torch.cuda.max_memory_allocated()),
                "task_ids_requested": args.task_ids,
            }
        )
    )
    if args.local_only:
        print(f"Stage 22 local-only complete: {args.task_ids}", flush=True)
        return
    if requested != set(TASK_IDS):
        raise ValueError("Public finalization requires all pinned tasks")

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in selected:
        for condition in CONDITIONS:
            record = validate_resume(
                output_dir=args.output_dir,
                pair=pair,
                condition=condition,
                protocol_sha256=protocol_sha256,
            )
            if record is None:
                raise AssertionError(f"Missing {pair['pair_id']} {condition}")
            records[(pair["pair_id"], condition)] = record
    pair_rows = [
        build_pair_row(
            pair,
            {condition: records[(pair["pair_id"], condition)] for condition in CONDITIONS},
        )
        for pair in selected
    ]
    readiness = all(row["ready_for_confirmatory_wm_scoring"] for row in pair_rows)
    candidate_lookup = {row["candidate_id"]: row for row in candidates}
    selected_ids = {row["pair_id"] for row in selected}
    screen_rows = []
    for screen in completed_screen:
        candidate = candidate_lookup[screen["candidate_id"]]
        screen_rows.append(
            {
                "candidate_id": screen["candidate_id"],
                "benchmark_task_id": int(screen["benchmark_task_id"]),
                "task_name": screen["task_name"],
                "init_state_index": int(screen["init_state_index"]),
                "episode_seed": int(screen["episode_seed"]),
                "screen_rank_within_task": int(screen["screen_rank_within_task"]),
                "success": bool(screen["success"]),
                "steps": int(screen["steps"]),
                "scoreable_through_h10_post": (
                    bool(screen["success"]) and int(screen["steps"]) >= MINIMUM_SCREEN_STEPS
                ),
                "initial_observation_sha256": screen["initial_observation_sha256"],
                "action_sha256": screen["action_sha256"],
                "historical_scout_steps": candidate["historical_scout_steps"],
                "selected_for_confirmation": (screen["candidate_id"] in selected_ids),
                "world_model_scores_computed": False,
            }
        )
    screen_rows.sort(
        key=lambda row: (
            int(row["benchmark_task_id"]),
            int(row["screen_rank_within_task"]),
        )
    )
    episode_rows = [
        sanitized_episode_row(
            records[(pair["pair_id"], condition)],
            args.output_dir,
        )
        for pair in selected
        for condition in CONDITIONS
    ]
    screen_text = STAGE20.csv_text(screen_rows)
    episodes_text = STAGE20.csv_text(episode_rows)
    pairs_text = STAGE20.csv_text(pair_rows)
    total_bytes = sum(int(row["archive_bytes"]) for row in episode_rows)
    report = {
        "schema_version": 1,
        "stage": "WM Stage 22 score-blind mild-intervention confirmation collection",
        "status": "passed" if readiness else "not_scoreable",
        "scope": {
            "task_ids": list(TASK_IDS),
            "fresh_reset_screen_episodes": len(screen_rows),
            "fresh_reset_screen_successes": sum(bool(row["success"]) for row in screen_rows),
            "pairs": len(pair_rows),
            "conditions_per_pair": len(CONDITIONS),
            "episodes": len(episode_rows),
            "raw_archive_bytes": total_bytes,
            "world_model_scores_computed": 0,
            "model_parameters_updated": 0,
            "test_demonstration_episodes_used": 0,
        },
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "alignment": {
            "all_initial_observations_match": all(row["all_initial_observations_match"] for row in pair_rows),
            "all_preintervention_commanded_actions_match": all(
                row["all_preintervention_commanded_actions_match"] for row in pair_rows
            ),
            "all_preintervention_observations_match": all(
                row["all_preintervention_observations_match"] for row in pair_rows
            ),
            "all_nominal_episodes_reproduce_screen": all(row["nominal_reproduces_fresh_screen"] for row in pair_rows),
            "all_episodes_cover_confirmatory_windows": all(
                int(row["valid_transitions"]) >= CONFIRMATORY_POST_END_EXCLUSIVE for row in episode_rows
            ),
        },
        "intervention_diagnostics": {
            condition: {
                "episodes": len(pair_rows),
                "action_mismatch_positive_episodes": sum(
                    float(row[f"{condition}_post_action_mismatch_l2_mean"]) > 0 for row in pair_rows
                ),
                "eef_motion_ratio_vs_nominal_median": float(
                    np.median([row[f"{condition}_eef_motion_ratio_vs_nominal"] for row in pair_rows])
                ),
                "pixel_motion_ratio_vs_nominal_median": float(
                    np.median([row[f"{condition}_pixel_motion_ratio_vs_nominal"] for row in pair_rows])
                ),
                "successes": sum(bool(row[f"{condition}_success"]) for row in pair_rows),
            }
            for condition in CONDITIONS[1:]
        },
        "artifacts": {
            "fresh_reset_screen_csv": "fresh_reset_screen.csv",
            "fresh_reset_screen_csv_sha256": sha256_text(screen_text),
            "episodes_csv": "episodes.csv",
            "episodes_csv_sha256": sha256_text(episodes_text),
            "pairs_csv": "pairs.csv",
            "pairs_csv_sha256": sha256_text(pairs_text),
            "raw_episode_archives": ("local-only under outputs/wm/stage22_confirmatory_interventions"),
        },
        "stage23_readiness": {
            "independent_confirmation_cohort_complete": readiness,
            "candidate_score_frozen_before_encoding": True,
            "frozen_wm_scoring_can_start": readiness,
        },
        "limitations": [
            "The cohort contains only six pairs across three tasks.",
            "Both interventions are synthetic actuator faults.",
            "Executed-action conditioning assumes access to actuator feedback.",
            "No world-model score is inspected in this collection stage.",
        ],
    }
    STAGE18.write_deterministic(
        args.public_results_dir / "fresh_reset_screen.csv",
        screen_text,
        overwrite=args.overwrite_public,
    )
    STAGE18.write_deterministic(
        args.public_results_dir / "episodes.csv",
        episodes_text,
        overwrite=args.overwrite_public,
    )
    STAGE18.write_deterministic(
        args.public_results_dir / "pairs.csv",
        pairs_text,
        overwrite=args.overwrite_public,
    )
    STAGE18.write_deterministic(
        args.public_results_dir / "report.json",
        canonical_json(report),
        overwrite=args.overwrite_public,
    )
    print(
        f"Stage 22 {report['status']}: pairs={len(pair_rows)} episodes={len(episode_rows)} bytes={total_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
