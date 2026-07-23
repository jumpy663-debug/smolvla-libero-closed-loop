#!/usr/bin/env python

"""Collect complete-observation SmolVLA closed-loop trajectories for WM analysis.

The policy path matches the official LeRobot rollout loop, but observations are
captured before policy preprocessing so dual-camera uint8 frames and the nested
robot state can be persisted without the large float32 observation cache used by
``rollout(return_observations=True)``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
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

TASK_IDS = (4, 5, 7, 8)
EPISODES_PER_TASK = 3
SEED = 1000
ACTION_HORIZON = 10
RESOLUTION = 360
CAMERA_KEYS = ("camera1", "camera2")
STATE_PATHS = {
    "eef_pos": ("eef", "pos"),
    "eef_quat": ("eef", "quat"),
    "eef_mat": ("eef", "mat"),
    "gripper_qpos": ("gripper", "qpos"),
    "gripper_qvel": ("gripper", "qvel"),
    "joint_pos": ("joints", "pos"),
    "joint_vel": ("joints", "vel"),
}


def load_script_module(name: str) -> ModuleType:
    path = Path(__file__).parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE3 = load_script_module("stage3_single_inference")
STAGE5 = load_script_module("stage5_small_benchmark")


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
        "--stage7-report",
        type=Path,
        default=Path("outputs/reproduction/smolvla_libero/stage7_action_horizon_ablation/report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wm/stage16_closed_loop_recollection"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage16_closed_loop_recollection"),
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(TASK_IDS))
    parser.add_argument("--episodes-per-task", type=int, default=EPISODES_PER_TASK)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-action-steps", type=int, default=ACTION_HORIZON)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Collect/validate local task shards without finalizing public results.",
    )
    return parser.parse_args()


def canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_text(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_deterministic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return
        raise FileExistsError(f"Refusing to overwrite non-identical output: {path}")
    path.write_text(content, encoding="utf-8")


def nested_array(mapping: dict[str, Any], path: tuple[str, ...]) -> np.ndarray:
    value: Any = mapping
    for key in path:
        value = value[key]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Expected ndarray at robot_state.{'.'.join(path)}")
    return value


def capture_observation(
    observation: dict[str, Any],
    destination: dict[str, list[np.ndarray]],
) -> None:
    pixels = observation["pixels"]
    robot_state = observation["robot_state"]
    for camera in CAMERA_KEYS:
        image = pixels[camera]
        if image.dtype != np.uint8 or image.ndim != 4 or image.shape[-1] != 3:
            raise AssertionError(f"Unexpected {camera} batch: {image.shape} {image.dtype}")
        destination[camera].append(np.array(image, copy=True))
    for name, path in STATE_PATHS.items():
        value = nested_array(robot_state, path)
        if value.ndim < 2 or not np.isfinite(value).all():
            raise AssertionError(f"Invalid state batch {name}: {value.shape}")
        destination[name].append(np.asarray(value, dtype=np.float32).copy())


def episode_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(value.dtype.str.encode())
        digest.update(b"\0")
        digest.update(canonical_json(list(value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def transition_valid_mask(steps: int, success: bool) -> np.ndarray:
    if steps <= 0:
        raise ValueError("steps must be positive")
    mask = np.ones(steps, dtype=np.bool_)
    if success:
        # LiberoEnv.step() resets immediately after success, so the returned
        # observation is the next initial state rather than the terminal state.
        mask[-1] = False
    return mask


def validate_episode_arrays(
    arrays: dict[str, np.ndarray],
    *,
    resolution: int,
) -> None:
    required = {
        *CAMERA_KEYS,
        *STATE_PATHS,
        "action",
        "reward",
        "success",
        "done",
        "transition_valid",
    }
    if set(arrays) != required:
        raise AssertionError(f"Episode keys differ: {sorted(set(arrays) ^ required)}")
    steps = len(arrays["action"])
    observations = steps + 1
    if arrays["action"].shape != (steps, 7):
        raise AssertionError(f"Unexpected action shape: {arrays['action'].shape}")
    for key in ("reward", "success", "done", "transition_valid"):
        if arrays[key].shape != (steps,):
            raise AssertionError(f"Unexpected {key} shape: {arrays[key].shape}")
    for camera in CAMERA_KEYS:
        expected = (observations, resolution, resolution, 3)
        if arrays[camera].shape != expected or arrays[camera].dtype != np.uint8:
            raise AssertionError(f"Unexpected {camera}: {arrays[camera].shape} {arrays[camera].dtype}")
    for state_name in STATE_PATHS:
        if len(arrays[state_name]) != observations:
            raise AssertionError(f"State/observation misalignment for {state_name}")
        if arrays[state_name].dtype != np.float32 or not np.isfinite(arrays[state_name]).all():
            raise AssertionError(f"Invalid state values for {state_name}")
    if not np.isfinite(arrays["action"]).all() or not np.isfinite(arrays["reward"]).all():
        raise AssertionError("Action or reward contains non-finite values")
    if not bool(arrays["done"][-1]):
        raise AssertionError("Final retained step must be done")
    episode_success = bool(arrays["success"].any())
    if episode_success and bool(arrays["transition_valid"][-1]):
        raise AssertionError("Successful final transition crosses an automatic reset")
    if not episode_success and not bool(arrays["transition_valid"].all()):
        raise AssertionError("A failed time-limit episode should retain every transition")


def load_episode(path: Path, *, resolution: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_episode_arrays(arrays, resolution=resolution)
    return arrays


def extract_successes(info: dict[str, Any], n_envs: int) -> list[bool]:
    if "final_info" in info:
        final_info = info["final_info"]
        if not isinstance(final_info, dict):
            raise RuntimeError("Unsupported final_info format")
        return [bool(value) for value in final_info["is_success"].tolist()]
    if "is_success" in info:
        values = info["is_success"]
        if hasattr(values, "tolist"):
            return [bool(value) for value in values.tolist()]
        return [bool(values)] * n_envs
    return [False] * n_envs


def collect_task(
    *,
    env: Any,
    policy: torch.nn.Module,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    seeds: list[int],
) -> tuple[dict[str, list[np.ndarray]], dict[str, torch.Tensor]]:
    policy.reset()
    observation, _ = env.reset(seed=seeds)
    captured: dict[str, list[np.ndarray]] = {key: [] for key in (*CAMERA_KEYS, *STATE_PATHS)}
    capture_observation(observation, captured)
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    successes: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    done = np.zeros(env.num_envs, dtype=np.bool_)
    max_steps = int(env.call("_max_episode_steps")[0])

    step = 0
    while not np.all(done) and step < max_steps:
        policy_observation = preprocess_observation(observation)
        try:
            policy_observation["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            policy_observation["task"] = list(env.call("task"))
        policy_observation = env_preprocessor(policy_observation)
        policy_observation = preprocessor(policy_observation)
        with torch.inference_mode():
            action = policy.select_action(policy_observation)
        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
        action_numpy = action.detach().cpu().numpy()
        if action_numpy.shape != (env.num_envs, 7):
            raise AssertionError(f"Unexpected action batch shape: {action_numpy.shape}")

        observation, reward, terminated, truncated, info = env.step(action_numpy)
        capture_observation(observation, captured)
        step_successes = extract_successes(info, env.num_envs)
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=np.bool_)

        actions.append(torch.from_numpy(action_numpy.copy()))
        rewards.append(torch.from_numpy(np.asarray(reward).copy()))
        successes.append(torch.tensor(step_successes, dtype=torch.bool))
        dones.append(torch.from_numpy(done.copy()))
        step += 1

    return captured, {
        "action": torch.stack(actions, dim=1),
        "reward": torch.stack(rewards, dim=1),
        "success": torch.stack(successes, dim=1),
        "done": torch.stack(dones, dim=1),
    }


def task_paths(output_dir: Path, task_id: int, episodes: int) -> tuple[Path, list[Path]]:
    summary = output_dir / "task_reports" / f"task_{task_id:02d}.json"
    shards = [output_dir / "episodes" / f"task_{task_id:02d}_init_{episode:02d}.npz" for episode in range(episodes)]
    return summary, shards


def validate_task_resume(
    output_dir: Path,
    task_id: int,
    episodes: int,
    resolution: int,
) -> dict[str, Any] | None:
    summary_path, shard_paths = task_paths(output_dir, task_id, episodes)
    exists = [path.exists() for path in shard_paths]
    if summary_path.exists() and all(exists):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for record, path in zip(summary["episodes"], shard_paths, strict=True):
            arrays = load_episode(path, resolution=resolution)
            if sha256_file(path) != record["file_sha256"]:
                raise AssertionError(f"File hash mismatch: {path}")
            if episode_content_sha256(arrays) != record["content_sha256"]:
                raise AssertionError(f"Content hash mismatch: {path}")
        return summary
    if summary_path.exists() or any(exists):
        raise RuntimeError(f"Partial task output requires manual inspection: task {task_id}")
    return None


def save_task(
    *,
    output_dir: Path,
    task_id: int,
    task_name: str,
    seeds: list[int],
    captured: dict[str, list[np.ndarray]],
    rollout_data: dict[str, torch.Tensor],
    resolution: int,
    episode_limit: int,
    old_lookup: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    summary_path, shard_paths = task_paths(output_dir, task_id, len(seeds))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    shard_paths[0].parent.mkdir(parents=True, exist_ok=True)
    done_indices = torch.argmax(rollout_data["done"].to(torch.int64), dim=1)
    records: list[dict[str, Any]] = []

    for episode_index, (seed, shard_path) in enumerate(zip(seeds, shard_paths, strict=True)):
        steps = int(done_indices[episode_index].item()) + 1
        episode_success = bool(rollout_data["success"][episode_index, :steps].any().item())
        arrays = {
            key: np.stack(
                [batch[episode_index] for batch in values[: steps + 1]],
                axis=0,
            )
            for key, values in captured.items()
        }
        arrays.update(
            {
                "action": rollout_data["action"][episode_index, :steps].float().numpy(),
                "reward": rollout_data["reward"][episode_index, :steps].float().numpy(),
                "success": rollout_data["success"][episode_index, :steps].numpy().astype(np.bool_),
                "done": rollout_data["done"][episode_index, :steps].numpy().astype(np.bool_),
                "transition_valid": transition_valid_mask(steps, episode_success),
            }
        )
        validate_episode_arrays(arrays, resolution=resolution)
        content_hash = episode_content_sha256(arrays)
        temporary_path = shard_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_path, **arrays)
        loaded = load_episode(temporary_path, resolution=resolution)
        if episode_content_sha256(loaded) != content_hash:
            raise AssertionError(f"Round-trip content mismatch: {temporary_path}")
        temporary_path.replace(shard_path)
        old = old_lookup[(task_id, episode_index)]
        records.append(
            {
                "rollout_id": f"pretrained-h10-task{task_id:02d}-init{episode_index:02d}",
                "benchmark_task_id": task_id,
                "task_name": task_name,
                "episode_index": episode_index,
                "init_state_index": episode_index,
                "seed": seed,
                "success": episode_success,
                "steps": steps,
                "valid_transitions": int(arrays["transition_valid"].sum()),
                "terminal_observation_kind": (
                    "automatic_reset_observation" if episode_success else "time_limit_observation"
                ),
                "old_success": bool(old["success"]),
                "success_matches_stage7": episode_success == bool(old["success"]),
                "old_steps": int(old["steps"]),
                "step_delta": steps - int(old["steps"]),
                "file": str(shard_path.as_posix()),
                "file_bytes": shard_path.stat().st_size,
                "file_sha256": sha256_file(shard_path),
                "content_sha256": content_hash,
            }
        )

    if not all(record["success_matches_stage7"] for record in records):
        raise AssertionError(f"Success labels did not reproduce Stage 7 for task {task_id}")
    summary = {
        "schema_version": 1,
        "task_id": task_id,
        "task_name": task_name,
        "resolution": resolution,
        "episode_limit": episode_limit,
        "episodes": records,
    }
    write_deterministic(summary_path, canonical_json(summary))
    return summary


def sanitized_episode_row(record: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    path = Path(record["file"])
    try:
        relative = Path("outputs") / path.relative_to(output_dir.parent.parent)
    except ValueError:
        relative = Path("outputs/wm/stage16_closed_loop_recollection/episodes") / path.name
    return {
        **{key: value for key, value in record.items() if key != "file"},
        "local_artifact": relative.as_posix(),
        "has_dual_camera_observations": True,
        "has_complete_robot_state": True,
        "has_actions_rewards_labels": True,
        "wm_transition_eligible": True,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 16 requires a CUDA GPU")
    if args.suite != "libero_spatial":
        raise ValueError("Stage 16 protocol is pinned to libero_spatial")
    if len(set(args.task_ids)) != len(args.task_ids):
        raise ValueError("--task-ids must not contain duplicates")
    if any(task not in TASK_IDS for task in args.task_ids):
        raise ValueError(f"Stage 16 task IDs must be a subset of {TASK_IDS}")
    if args.episodes_per_task != EPISODES_PER_TASK:
        raise ValueError(f"Stage 16 requires {EPISODES_PER_TASK} episodes per task")
    if args.seed != SEED or args.n_action_steps != ACTION_HORIZON or args.resolution != RESOLUTION:
        raise ValueError("Stage 16 seed, action horizon, and resolution are protocol-pinned")

    stage2_manifest = STAGE3.verify_stage2_manifest(args.stage2_manifest)
    local_backbone_manifest = args.backbone_dir.parent / "backbone_manifest.json"
    backbone_report = json.loads(local_backbone_manifest.read_text(encoding="utf-8"))
    if backbone_report["resolved_revision"] != STAGE3.BACKBONE_REVISION:
        raise AssertionError("Backbone revision differs from the pinned revision")
    if stage2_manifest["resolved_revision"] != STAGE3.EXPECTED_POLICY_REVISION:
        raise AssertionError("Policy revision differs from the pinned revision")

    stage7 = json.loads(args.stage7_report.read_text(encoding="utf-8"))
    if stage7.get("status") != "passed":
        raise AssertionError("Stage 7 report did not pass")
    old_rows = [
        row
        for row in stage7["per_episode"]
        if int(row["horizon"]) == ACTION_HORIZON and int(row["task_id"]) in TASK_IDS
    ]
    old_lookup = {(int(row["task_id"]), int(row["episode_index"])): row for row in old_rows}
    expected_keys = {(task, episode) for task in TASK_IDS for episode in range(EPISODES_PER_TASK)}
    if set(old_lookup) != expected_keys:
        raise AssertionError("Stage 7 paired recollection index is incomplete")

    config = PreTrainedConfig.from_pretrained(args.checkpoint_dir, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    config.device = "cuda"
    config.vlm_model_name = str(args.backbone_dir.resolve())
    config.load_vlm_weights = True
    config.empty_cameras = 1
    config.use_amp = False
    config.n_action_steps = ACTION_HORIZON
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint_dir),
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": {}},
            "tokenizer_processor": {"tokenizer_name": str(args.backbone_dir.resolve())},
            "device_processor": {"device": "cuda"},
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
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

    task_summaries: list[dict[str, Any]] = []
    for task_id in args.task_ids:
        resumed = validate_task_resume(
            args.output_dir,
            task_id,
            EPISODES_PER_TASK,
            RESOLUTION,
        )
        if resumed is not None:
            print(f"Stage 16 resume: task {task_id} already validated")
            task_summaries.append(resumed)
            continue

        env_cfg, env, envs = STAGE5.make_task_env(
            suite=args.suite,
            task_id=task_id,
            n_envs=EPISODES_PER_TASK,
            resolution=RESOLUTION,
        )
        try:
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, config)
            task_name = env.call("task")[0]
            episode_limit = int(env.call("_max_episode_steps")[0])
            seeds = [SEED + index for index in range(EPISODES_PER_TASK)]
            torch.manual_seed(SEED + task_id)
            torch.cuda.manual_seed_all(SEED + task_id)
            captured, rollout_data = collect_task(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                seeds=seeds,
            )
            summary = save_task(
                output_dir=args.output_dir,
                task_id=task_id,
                task_name=task_name,
                seeds=seeds,
                captured=captured,
                rollout_data=rollout_data,
                resolution=RESOLUTION,
                episode_limit=episode_limit,
                old_lookup=old_lookup,
            )
            task_summaries.append(summary)
            print(
                f"Stage 16 task {task_id} passed: {sum(record['success'] for record in summary['episodes'])}/3 success"
            )
        finally:
            close_envs(envs)

    torch.cuda.synchronize()
    finished = time.perf_counter()
    local_runtime = {
        "schema_version": 1,
        "model_load_seconds": model_loaded - started,
        "total_seconds_this_invocation": finished - started,
        "peak_pytorch_gpu_allocation_bytes": torch.cuda.max_memory_allocated(),
        "task_ids_requested": args.task_ids,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }
    runtime_path = args.output_dir / "runtime.json"
    runtime_path.write_text(canonical_json(local_runtime), encoding="utf-8")

    if args.local_only:
        print(f"Stage 16 local-only passed: tasks={args.task_ids}")
        return
    if set(args.task_ids) != set(TASK_IDS):
        raise ValueError("Public finalization requires the complete pinned task set")

    # Include resumed task summaries that were not requested only when a caller
    # explicitly requests the complete set above.
    summaries_by_task = {int(summary["task_id"]): summary for summary in task_summaries}
    if set(summaries_by_task) != set(TASK_IDS):
        raise AssertionError("Formal Stage 16 output is missing a task")
    records = [record for task_id in TASK_IDS for record in summaries_by_task[task_id]["episodes"]]
    public_rows = [sanitized_episode_row(record, args.output_dir) for record in records]
    episodes_csv = csv_text(public_rows)
    successes = sum(bool(row["success"]) for row in public_rows)
    valid_transitions = sum(int(row["valid_transitions"]) for row in public_rows)
    total_steps = sum(int(row["steps"]) for row in public_rows)
    report = {
        "schema_version": 1,
        "stage": "Stage 16 complete-observation closed-loop recollection",
        "status": "passed",
        "scope": {
            "suite": args.suite,
            "policy": "pretrained SmolVLA",
            "task_ids": list(TASK_IDS),
            "episodes_per_task": EPISODES_PER_TASK,
            "initial_state_indices": list(range(EPISODES_PER_TASK)),
            "seeds": [SEED + index for index in range(EPISODES_PER_TASK)],
            "n_action_steps": ACTION_HORIZON,
            "resolution": RESOLUTION,
            "episodes": len(public_rows),
            "successes": successes,
            "failures": len(public_rows) - successes,
            "total_control_steps": total_steps,
            "valid_wm_transitions": valid_transitions,
            "test_demonstration_episodes_used": 0,
        },
        "modalities": {
            "camera1": "agentview uint8 RGB, synchronized before every policy action",
            "camera2": "eye-in-hand uint8 RGB, synchronized before every policy action",
            "robot_state": list(STATE_PATHS),
            "action": "7-D postprocessed action actually passed to LIBERO",
            "labels": ["reward", "success", "cumulative done", "transition_valid"],
        },
        "alignment": {
            "observation_count_per_episode": "steps + 1",
            "action_t_maps_observation_t_to_t_plus_1": True,
            "successful_final_transition_valid": False,
            "reason": (
                "LiberoEnv returns an automatic-reset observation on success; "
                "the crossing transition is retained for audit but excluded from WM scoring."
            ),
        },
        "reproducibility": {
            "stage7_report_sha256": sha256_file(args.stage7_report),
            "policy_checkpoint_sha256": sha256_file(args.checkpoint_dir / "model.safetensors"),
            "backbone_checkpoint_sha256": sha256_file(args.backbone_dir / "model.safetensors"),
            "success_labels_matching_stage7": sum(bool(row["success_matches_stage7"]) for row in public_rows),
            "expected_success_labels": len(public_rows),
            "content_hash_is_container_independent": True,
        },
        "artifacts": {
            "episodes_csv": "episodes.csv",
            "episodes_csv_sha256": hashlib.sha256(episodes_csv.encode()).hexdigest(),
            "raw_episode_archives": "local-only under outputs/wm/stage16_closed_loop_recollection",
            "raw_archives_committed_to_git": False,
        },
        "stage17_readiness": {
            "complete_observation_rollouts_available": True,
            "success_and_failure_labels_available": successes > 0 and successes < len(public_rows),
            "world_model_failure_scoring_started": False,
        },
        "limitations": [
            "The cohort contains only 12 paired episodes on four tasks.",
            "Success episodes do not expose the true terminal observation after the final action.",
            "The same cohort is exploratory data, not a held-out failure-detection benchmark.",
            "No causal claim links latent rollout error to policy failure in this stage.",
        ],
    }
    write_deterministic(args.public_results_dir / "episodes.csv", episodes_csv)
    write_deterministic(args.public_results_dir / "report.json", canonical_json(report))
    print(
        f"Stage 16 passed: episodes={len(public_rows)}, success={successes}, "
        f"failure={len(public_rows) - successes}, valid_transitions={valid_transitions}"
    )


if __name__ == "__main__":
    main()
