#!/usr/bin/env python

"""Ablate SmolVLA action execution horizons on difficult LIBERO Spatial tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import lerobot
import mujoco
import numpy as np
import robosuite
import torch
import transformers
from lerobot.configs import PreTrainedConfig
from lerobot.envs import close_envs, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video
from stage3_single_inference import (
    BACKBONE_REVISION,
    EXPECTED_POLICY_REVISION,
    gpu_memory,
    prepare_backbone,
    verify_stage2_manifest,
)
from stage5_small_benchmark import make_task_env, sha256_file

BASELINE_HORIZON = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/stage2_checkpoint/checkpoint"),
    )
    parser.add_argument(
        "--stage2-manifest",
        type=Path,
        default=Path("outputs/stage2_checkpoint/manifest.json"),
    )
    parser.add_argument(
        "--backbone-dir",
        type=Path,
        default=Path("outputs/stage3_single_inference/backbone"),
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("outputs/stage6_full_spatial_benchmark/report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage7_action_horizon_ablation"),
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[4, 5, 7, 8])
    parser.add_argument("--horizons", type=int, nargs="+", default=[25, 10])
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=360)
    return parser.parse_args()


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "total": float(array.sum()),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [float("nan"), float("nan")]
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
    margin /= denominator
    return [center - margin, center + margin]


def load_baseline(
    report_path: Path,
    suite: str,
    task_ids: list[int],
    episodes_per_task: int,
    resolution: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, torch.Tensor]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scope = report.get("scope", {})
    if report.get("status") != "passed":
        raise AssertionError("Stage 6 baseline report did not pass")
    if scope.get("suite") != suite or scope.get("resolution") != resolution:
        raise AssertionError("Stage 6 baseline environment does not match this ablation")
    if scope.get("episodes_per_task") != episodes_per_task:
        raise AssertionError("Stage 6 baseline episode count does not match this ablation")
    if report.get("policy", {}).get("revision") != EXPECTED_POLICY_REVISION:
        raise AssertionError("Stage 6 policy revision does not match the pinned policy")
    if report.get("backbone", {}).get("revision") != BACKBONE_REVISION:
        raise AssertionError("Stage 6 backbone revision does not match the pinned backbone")

    task_set = set(task_ids)
    baseline_source_rows = [row for row in report.get("per_episode", []) if int(row["task_id"]) in task_set]
    if len(baseline_source_rows) != len(task_ids) * episodes_per_task:
        raise AssertionError(
            f"Expected {len(task_ids) * episodes_per_task} baseline rows, found {len(baseline_source_rows)}"
        )

    baseline_rows: list[dict[str, Any]] = []
    for row in baseline_source_rows:
        baseline_rows.append(
            {
                "horizon": BASELINE_HORIZON,
                "source": "stage6_baseline",
                "suite": row["suite"],
                "task_id": int(row["task_id"]),
                "task_name": row["task_name"],
                "episode_index": int(row["episode_index"]),
                "init_state_index": int(row["init_state_index"]),
                "seed": int(row["seed"]),
                "success": bool(row["success"]),
                "steps": int(row["steps"]),
                "sum_reward": float(row["sum_reward"]),
                "max_reward": float(row["max_reward"]),
                "terminated_before_limit": bool(row["terminated_before_limit"]),
                "video": row["video"],
            }
        )

    baseline_valid_actions: dict[int, torch.Tensor] = {}
    per_task = {int(item["task_id"]): item for item in report.get("per_task", [])}
    for task_id in task_ids:
        task_report = per_task.get(task_id)
        if task_report is None:
            raise KeyError(f"Task {task_id} is missing from the stage 6 report")
        trajectory_path = Path(task_report["trajectory"]["path"])
        trajectory = np.load(trajectory_path)
        actions = torch.from_numpy(trajectory["actions"]).float()
        done_indices = trajectory["done_indices"]
        valid = [actions[index, : int(done_indices[index]) + 1] for index in range(episodes_per_task)]
        baseline_valid_actions[task_id] = torch.cat(valid, dim=0)

    return report, baseline_rows, baseline_valid_actions


def aggregate_inference_from_baseline(baseline_report: dict[str, Any], task_ids: list[int]) -> dict[str, Any]:
    task_set = set(task_ids)
    summaries = [item["model_inference"] for item in baseline_report["per_task"] if int(item["task_id"]) in task_set]
    count = sum(int(item["count"]) for item in summaries)
    total = sum(float(item["total"]) for item in summaries)
    return {
        "count": count,
        "min": min(float(item["min"]) for item in summaries),
        "max": max(float(item["max"]) for item in summaries),
        "mean": total / count,
        "median": None,
        "total": total,
        "note": "Median unavailable because stage 6 stores per-task aggregate latency only.",
    }


def action_bounds_audit(actions: torch.Tensor) -> dict[str, Any]:
    low = torch.full((7,), -1.0)
    high = torch.full((7,), 1.0)
    outside = (actions < low) | (actions > high)
    return {
        "outside_value_count": int(outside.sum().item()),
        "total_value_count": actions.numel(),
        "outside_count_by_dimension": outside.sum(dim=0).tolist(),
        "maximum_underflow": float(torch.clamp(low - actions, min=0).max().item()),
        "maximum_overflow": float(torch.clamp(actions - high, min=0).max().item()),
        "actions_were_clipped": False,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 7 requires a CUDA GPU, but torch.cuda.is_available() is false")
    if args.episodes_per_task <= 0 or args.resolution <= 0:
        raise ValueError("Episode count and resolution must be positive")
    if len(set(args.task_ids)) != len(args.task_ids):
        raise ValueError("--task-ids must not contain duplicates")
    if not args.horizons or len(set(args.horizons)) != len(args.horizons):
        raise ValueError("--horizons must be non-empty and unique")
    if any(horizon <= 0 or horizon >= BASELINE_HORIZON for horizon in args.horizons):
        raise ValueError("New horizons must be in the range [1, 49]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage2_manifest = verify_stage2_manifest(args.stage2_manifest)
    backbone_report = prepare_backbone(args.backbone_dir)
    baseline_report, baseline_rows, baseline_actions_by_task = load_baseline(
        args.baseline_report,
        suite=args.suite,
        task_ids=args.task_ids,
        episodes_per_task=args.episodes_per_task,
        resolution=args.resolution,
    )

    config = PreTrainedConfig.from_pretrained(args.checkpoint_dir, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    config.device = "cuda"
    config.vlm_model_name = str(args.backbone_dir.resolve())
    config.load_vlm_weights = True
    config.empty_cameras = 1
    config.use_amp = False

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint_dir),
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": {}},
            "tokenizer_processor": {"tokenizer_name": str(args.backbone_dir.resolve())},
            "device_processor": {"device": "cuda"},
        },
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_load_started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint_dir,
        config=config,
        local_files_only=True,
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    memory_after_load = gpu_memory()

    current_horizon: int | None = None
    current_task_id: int | None = None
    inference_events: list[dict[str, Any]] = []
    queued_events: list[dict[str, Any]] = []
    original_select_action = policy.select_action

    def instrumented_select_action(batch, *call_args, **call_kwargs):
        performs_model_inference = len(policy._queues[ACTION]) == 0
        call_started = time.perf_counter()
        selected_action = original_select_action(batch, *call_args, **call_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - call_started
        event = {
            "horizon": current_horizon,
            "task_id": current_task_id,
            "seconds": elapsed,
        }
        if performs_model_inference:
            inference_events.append(event)
        else:
            queued_events.append(event)
        return selected_action

    policy.select_action = instrumented_select_action

    benchmark_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    new_rows: list[dict[str, Any]] = []
    new_task_reports: list[dict[str, Any]] = []
    new_video_artifacts: list[dict[str, Any]] = []
    valid_actions_by_horizon: dict[int, list[torch.Tensor]] = {
        BASELINE_HORIZON: list(baseline_actions_by_task.values())
    }
    environments_closed_cleanly = True

    for horizon in args.horizons:
        current_horizon = horizon
        config.n_action_steps = horizon
        valid_actions_by_horizon[horizon] = []
        for task_id in args.task_ids:
            current_task_id = task_id
            env_cfg, env, envs = make_task_env(
                suite=args.suite,
                task_id=task_id,
                n_envs=args.episodes_per_task,
                resolution=args.resolution,
            )
            frames: list[np.ndarray] = []
            task_started = time.perf_counter()
            inference_start_index = len(inference_events)
            try:
                env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, config)
                task_name = env.call("task")[0]
                task_description = env.call("task_description")[0]
                episode_limit = int(env.call("_max_episode_steps")[0])
                seeds = [args.seed + index for index in range(args.episodes_per_task)]

                def render_frame(
                    vector_env: gym.vector.VectorEnv,
                    frame_buffer: list[np.ndarray] = frames,
                ) -> None:
                    frame_buffer.append(
                        np.stack(
                            [
                                np.ascontiguousarray(vector_env.envs[index].render())
                                for index in range(args.episodes_per_task)
                            ]
                        )
                    )

                torch.manual_seed(args.seed + task_id)
                torch.cuda.manual_seed_all(args.seed + task_id)
                rollout_data = rollout(
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    seeds=seeds,
                    return_observations=False,
                    render_callback=render_frame,
                )
                torch.cuda.synchronize()

                actions = rollout_data[ACTION].detach().float().cpu()
                rewards = rollout_data["reward"].detach().float().cpu()
                successes = rollout_data["success"].detach().cpu()
                dones = rollout_data["done"].detach().cpu()
                if actions.shape[0] != args.episodes_per_task or actions.shape[2] != 7:
                    raise AssertionError(f"Unexpected action shape: {tuple(actions.shape)}")
                if not torch.isfinite(actions).all() or not torch.isfinite(rewards).all():
                    raise AssertionError(f"Non-finite rollout values for H{horizon} task {task_id}")

                done_indices = torch.argmax(dones.to(torch.int64), dim=1)
                task_successes = 0
                task_steps: list[int] = []
                task_video_artifacts: list[dict[str, Any]] = []
                for episode_index in range(args.episodes_per_task):
                    steps = int(done_indices[episode_index].item()) + 1
                    task_steps.append(steps)
                    episode_success = bool(successes[episode_index, :steps].any().item())
                    task_successes += int(episode_success)
                    valid_actions = actions[episode_index, :steps]
                    valid_actions_by_horizon[horizon].append(valid_actions)

                    episode_frames = np.stack([frame[episode_index] for frame in frames[: steps + 1]], axis=0)
                    video_dir = args.output_dir / "videos" / f"horizon_{horizon}" / f"task_{task_id:02d}"
                    video_dir.mkdir(parents=True, exist_ok=True)
                    video_path = video_dir / f"episode_{episode_index:02d}.mp4"
                    write_video(video_path, episode_frames, fps=env.unwrapped.metadata["render_fps"])
                    video_artifact = {
                        "horizon": horizon,
                        "task_id": task_id,
                        "episode_index": episode_index,
                        "path": str(video_path.resolve()),
                        "sha256": sha256_file(video_path),
                        "frame_count": int(episode_frames.shape[0]),
                    }
                    task_video_artifacts.append(video_artifact)
                    new_video_artifacts.append(video_artifact)

                    new_rows.append(
                        {
                            "horizon": horizon,
                            "source": "stage7_ablation",
                            "suite": args.suite,
                            "task_id": task_id,
                            "task_name": task_name,
                            "episode_index": episode_index,
                            "init_state_index": episode_index,
                            "seed": seeds[episode_index],
                            "success": episode_success,
                            "steps": steps,
                            "sum_reward": float(rewards[episode_index, :steps].sum().item()),
                            "max_reward": float(rewards[episode_index, :steps].max().item()),
                            "terminated_before_limit": steps < episode_limit,
                            "video": str(video_path.resolve()),
                        }
                    )

                trajectory_dir = args.output_dir / "trajectories" / f"horizon_{horizon}"
                trajectory_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = trajectory_dir / f"task_{task_id:02d}.npz"
                np.savez_compressed(
                    trajectory_path,
                    actions=actions.numpy(),
                    rewards=rewards.numpy(),
                    successes=successes.numpy(),
                    dones=dones.numpy(),
                    done_indices=done_indices.numpy(),
                )
                task_inference_times = [event["seconds"] for event in inference_events[inference_start_index:]]
                expected_calls = math.ceil(actions.shape[1] / horizon)
                if len(task_inference_times) != expected_calls:
                    raise AssertionError(
                        f"Expected {expected_calls} model calls for H{horizon}/task {task_id}, "
                        f"measured {len(task_inference_times)}"
                    )
                new_task_reports.append(
                    {
                        "horizon": horizon,
                        "task_id": task_id,
                        "task_name": task_name,
                        "task_description": task_description,
                        "successes": task_successes,
                        "episodes": args.episodes_per_task,
                        "success_rate": task_successes / args.episodes_per_task,
                        "steps": task_steps,
                        "episode_limit": episode_limit,
                        "model_inference": distribution(task_inference_times),
                        "wall_seconds": time.perf_counter() - task_started,
                        "trajectory": {
                            "path": str(trajectory_path.resolve()),
                            "sha256": sha256_file(trajectory_path),
                        },
                        "videos": task_video_artifacts,
                    }
                )
            finally:
                try:
                    close_envs(envs)
                except Exception:
                    environments_closed_cleanly = False
                    raise
                del frames

    benchmark_seconds = time.perf_counter() - benchmark_started
    benchmark_memory = gpu_memory()
    all_rows = baseline_rows + new_rows
    all_horizons = [BASELINE_HORIZON, *args.horizons]

    baseline_task_reports = {
        int(item["task_id"]): item for item in baseline_report["per_task"] if int(item["task_id"]) in set(args.task_ids)
    }
    horizon_reports: list[dict[str, Any]] = []
    for horizon in all_horizons:
        rows = [row for row in all_rows if int(row["horizon"]) == horizon]
        successes_count = sum(int(row["success"]) for row in rows)
        actions = torch.cat(valid_actions_by_horizon[horizon], dim=0)
        if horizon == BASELINE_HORIZON:
            inference_summary = aggregate_inference_from_baseline(baseline_report, args.task_ids)
            wall_seconds = sum(float(baseline_task_reports[task_id]["wall_seconds"]) for task_id in args.task_ids)
        else:
            times = [event["seconds"] for event in inference_events if event["horizon"] == horizon]
            inference_summary = distribution(times)
            wall_seconds = sum(
                float(item["wall_seconds"]) for item in new_task_reports if int(item["horizon"]) == horizon
            )
        per_task = []
        for task_id in args.task_ids:
            task_rows = [row for row in rows if int(row["task_id"]) == task_id]
            task_successes = sum(int(row["success"]) for row in task_rows)
            per_task.append(
                {
                    "task_id": task_id,
                    "successes": task_successes,
                    "episodes": len(task_rows),
                    "success_rate": task_successes / len(task_rows),
                    "steps": [int(row["steps"]) for row in task_rows],
                }
            )
        horizon_reports.append(
            {
                "horizon": horizon,
                "successes": successes_count,
                "episodes": len(rows),
                "success_rate": successes_count / len(rows),
                "success_rate_percent": 100 * successes_count / len(rows),
                "wilson_95_interval": wilson_interval(successes_count, len(rows)),
                "mean_episode_steps": float(np.mean([row["steps"] for row in rows])),
                "median_episode_steps": float(np.median([row["steps"] for row in rows])),
                "valid_control_steps": int(actions.shape[0]),
                "model_inference": inference_summary,
                "wall_seconds": wall_seconds,
                "action_bounds_audit": action_bounds_audit(actions),
                "per_task": per_task,
            }
        )

    baseline_lookup = {(int(row["task_id"]), int(row["episode_index"])): row for row in baseline_rows}
    paired_reports: list[dict[str, Any]] = []
    for horizon in args.horizons:
        horizon_rows = [row for row in new_rows if int(row["horizon"]) == horizon]
        transitions = {
            "failure_to_success": 0,
            "success_to_failure": 0,
            "success_to_success": 0,
            "failure_to_failure": 0,
        }
        pairs: list[dict[str, Any]] = []
        for row in horizon_rows:
            key = (int(row["task_id"]), int(row["episode_index"]))
            baseline = baseline_lookup[key]
            baseline_success = bool(baseline["success"])
            horizon_success = bool(row["success"])
            if not baseline_success and horizon_success:
                transition = "failure_to_success"
            elif baseline_success and not horizon_success:
                transition = "success_to_failure"
            elif baseline_success and horizon_success:
                transition = "success_to_success"
            else:
                transition = "failure_to_failure"
            transitions[transition] += 1
            pairs.append(
                {
                    "task_id": key[0],
                    "episode_index": key[1],
                    "baseline_success": baseline_success,
                    "horizon_success": horizon_success,
                    "transition": transition,
                    "baseline_steps": int(baseline["steps"]),
                    "horizon_steps": int(row["steps"]),
                    "step_delta": int(row["steps"]) - int(baseline["steps"]),
                }
            )
        paired_reports.append(
            {
                "horizon": horizon,
                "compared_with": BASELINE_HORIZON,
                "transitions": transitions,
                "pairs": pairs,
            }
        )

    maximum_successes = max(item["successes"] for item in horizon_reports)
    recommended_horizon = max(item["horizon"] for item in horizon_reports if item["successes"] == maximum_successes)
    tied_horizons = [item["horizon"] for item in horizon_reports if item["successes"] == maximum_successes]
    if len(tied_horizons) == 1:
        recommendation_reason = "Selected the only horizon with the highest number of successful paired episodes."
    else:
        recommendation_reason = (
            "Selected the largest (least compute-intensive) horizon among conditions tied for "
            "the highest number of successful paired episodes."
        )

    csv_path = args.output_dir / "episodes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: (-int(row["horizon"]), row["task_id"], row["episode_index"])))

    report = {
        "status": "passed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "stage": "pretrained SmolVLA action execution horizon ablation",
        "scope": {
            "suite": args.suite,
            "task_ids": args.task_ids,
            "episodes_per_task": args.episodes_per_task,
            "initial_state_indices": list(range(args.episodes_per_task)),
            "horizons": all_horizons,
            "baseline_horizon": BASELINE_HORIZON,
            "new_episode_count": len(new_rows),
            "total_compared_episode_count": len(all_rows),
            "resolution": args.resolution,
            "controlled_variable": "n_action_steps",
            "chunk_size_held_fixed": config.chunk_size,
        },
        "policy": {
            "repo_id": stage2_manifest["repo_id"],
            "revision": stage2_manifest["resolved_revision"],
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "weights_modified": False,
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "num_diffusion_steps": config.num_steps,
            "empty_cameras": config.empty_cameras,
        },
        "backbone": {
            "repo_id": backbone_report["repo_id"],
            "revision": backbone_report["resolved_revision"],
            "model_sha256": backbone_report["files"]["model.safetensors"]["sha256"],
        },
        "per_horizon": horizon_reports,
        "paired_comparisons": paired_reports,
        "recommendation": {
            "horizon": recommended_horizon,
            "reason": recommendation_reason,
            "maximum_successes": maximum_successes,
            "episodes": len(baseline_rows),
        },
        "per_episode": all_rows,
        "runtime": {
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "cuda_device": torch.cuda.get_device_name(),
            "cuda_capability": list(torch.cuda.get_device_capability()),
            "model_load_seconds": model_load_seconds,
            "new_conditions_wall_seconds": benchmark_seconds,
            "memory_after_model_load": memory_after_load,
            "benchmark_memory": benchmark_memory,
            "queued_action_selection": distribution([event["seconds"] for event in queued_events]),
            "environments_closed_cleanly": environments_closed_cleanly,
        },
        "versions": {
            "python": platform.python_version(),
            "lerobot": lerobot.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "mujoco": mujoco.__version__,
            "robosuite": robosuite.__version__,
        },
        "artifacts": {
            "report": str((args.output_dir / "report.json").resolve()),
            "episodes_csv": str(csv_path.resolve()),
            "baseline_report": str(args.baseline_report.resolve()),
            "new_videos": new_video_artifacts,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Stage 7 action-horizon ablation completed. Report: {report_path}")


if __name__ == "__main__":
    main()
