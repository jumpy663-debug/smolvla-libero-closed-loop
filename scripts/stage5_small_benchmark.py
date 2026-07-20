#!/usr/bin/env python

"""Evaluate pinned SmolVLA on three LIBERO tasks and three initial states per task."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
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
        "--output-dir",
        type=Path,
        default=Path("outputs/stage5_small_benchmark"),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Optional PEFT adapter loaded over --checkpoint-dir without merging.",
    )
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--fine-tune-report", type=Path, default=None)
    parser.add_argument("--comparison-report", type=Path, default=None)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=360)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def make_task_env(
    suite: str,
    task_id: int,
    n_envs: int,
    resolution: int,
) -> tuple[LiberoEnvConfig, gym.vector.SyncVectorEnv, dict]:
    env_cfg = LiberoEnvConfig(
        task=suite,
        task_ids=[task_id],
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        camera_name_mapping={
            "agentview_image": "camera1",
            "robot0_eye_in_hand_image": "camera2",
        },
        init_states=True,
        observation_height=resolution,
        observation_width=resolution,
        control_mode="relative",
    )
    envs = make_env(env_cfg, n_envs=n_envs, use_async_envs=False)
    env = envs[suite][task_id]
    if not isinstance(env, gym.vector.SyncVectorEnv):
        close_envs(envs)
        raise TypeError(f"Expected SyncVectorEnv, got {type(env).__name__}")
    return env_cfg, env, envs


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5 requires a CUDA GPU, but torch.cuda.is_available() is false")
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    if len(set(args.task_ids)) != len(args.task_ids):
        raise ValueError("--task-ids must not contain duplicates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.output_dir / "videos"
    trajectories_dir = args.output_dir / "trajectories"
    videos_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    stage2_manifest = verify_stage2_manifest(args.stage2_manifest)
    local_backbone_manifest = args.backbone_dir.parent / "backbone_manifest.json"
    if local_backbone_manifest.is_file():
        backbone_report = json.loads(local_backbone_manifest.read_text(encoding="utf-8"))
    else:
        backbone_report = prepare_backbone(args.backbone_dir)
    if backbone_report["resolved_revision"] != BACKBONE_REVISION:
        raise AssertionError("Backbone revision is not the stage 3 pinned revision")
    if stage2_manifest["resolved_revision"] != EXPECTED_POLICY_REVISION:
        raise AssertionError("Policy revision is not the stage 2 pinned revision")

    config = PreTrainedConfig.from_pretrained(args.checkpoint_dir, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    config.device = "cuda"
    config.vlm_model_name = str(args.backbone_dir.resolve())
    config.load_vlm_weights = True
    config.empty_cameras = 1
    config.use_amp = False
    if args.n_action_steps is not None:
        if args.n_action_steps <= 0 or args.n_action_steps > config.chunk_size:
            raise ValueError("--n-action-steps must be in [1, chunk_size]")
        config.n_action_steps = args.n_action_steps

    fine_tune_report = None
    if args.fine_tune_report is not None:
        fine_tune_report = json.loads(args.fine_tune_report.read_text(encoding="utf-8"))
        if fine_tune_report.get("status") != "passed":
            raise AssertionError("Fine-tune report did not pass")

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
    adapter_config = None
    if args.adapter_dir is not None:
        from peft import PeftConfig, PeftModel

        adapter_config = PeftConfig.from_pretrained(args.adapter_dir, local_files_only=True)
        if Path(adapter_config.base_model_name_or_path).resolve() != args.checkpoint_dir.resolve():
            raise AssertionError("PEFT adapter base does not match --checkpoint-dir")
        policy = PeftModel.from_pretrained(
            policy,
            args.adapter_dir,
            config=adapter_config,
            is_trainable=False,
            local_files_only=True,
        )
        policy.to("cuda")
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    memory_after_load = gpu_memory()

    inference_events: list[dict[str, Any]] = []
    queued_action_seconds: list[float] = []
    current_task_id: int | None = None
    original_select_action = policy.select_action

    def instrumented_select_action(batch, *call_args, **call_kwargs):
        performs_model_inference = len(policy._queues[ACTION]) == 0
        call_started = time.perf_counter()
        selected_action = original_select_action(batch, *call_args, **call_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - call_started
        if performs_model_inference:
            inference_events.append({"task_id": current_task_id, "seconds": elapsed})
        else:
            queued_action_seconds.append(elapsed)
        return selected_action

    policy.select_action = instrumented_select_action

    benchmark_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    episode_rows: list[dict[str, Any]] = []
    task_reports: list[dict[str, Any]] = []
    all_valid_actions: list[torch.Tensor] = []
    all_video_artifacts: list[dict[str, Any]] = []
    environments_closed_cleanly = True

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
            seeds = [args.seed + episode_index for episode_index in range(args.episodes_per_task)]

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
                raise AssertionError(f"Unexpected action shape for task {task_id}: {tuple(actions.shape)}")
            if not torch.isfinite(actions).all() or not torch.isfinite(rewards).all():
                raise AssertionError(f"Non-finite rollout values for task {task_id}")

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
                all_valid_actions.append(valid_actions)

                episode_frames = np.stack([frame[episode_index] for frame in frames[: steps + 1]], axis=0)
                task_video_dir = videos_dir / f"task_{task_id:02d}"
                task_video_dir.mkdir(parents=True, exist_ok=True)
                video_path = task_video_dir / f"episode_{episode_index:02d}.mp4"
                write_video(video_path, episode_frames, fps=env.unwrapped.metadata["render_fps"])
                video_artifact = {
                    "episode_index": episode_index,
                    "path": str(video_path.resolve()),
                    "sha256": sha256_file(video_path),
                    "frame_count": int(episode_frames.shape[0]),
                }
                task_video_artifacts.append(video_artifact)
                all_video_artifacts.append(video_artifact)

                episode_rows.append(
                    {
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

            trajectory_path = trajectories_dir / f"task_{task_id:02d}.npz"
            np.savez_compressed(
                trajectory_path,
                actions=actions.numpy(),
                rewards=rewards.numpy(),
                successes=successes.numpy(),
                dones=dones.numpy(),
                done_indices=done_indices.numpy(),
            )
            task_inference_times = [event["seconds"] for event in inference_events[inference_start_index:]]
            task_reports.append(
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "task_description": task_description,
                    "successes": task_successes,
                    "episodes": args.episodes_per_task,
                    "success_rate": task_successes / args.episodes_per_task,
                    "steps": task_steps,
                    "mean_steps": float(np.mean(task_steps)),
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
    valid_actions = torch.cat(all_valid_actions, dim=0)
    action_low = torch.full((7,), -1.0)
    action_high = torch.full((7,), 1.0)
    outside_bounds = (valid_actions < action_low) | (valid_actions > action_high)
    successes_total = sum(int(row["success"]) for row in episode_rows)
    episodes_total = len(episode_rows)
    success_rate = successes_total / episodes_total
    confidence_interval = wilson_interval(successes_total, episodes_total)

    paired_comparison = None
    if args.comparison_report is not None:
        comparison_report = json.loads(args.comparison_report.read_text(encoding="utf-8"))
        if comparison_report.get("status") != "passed":
            raise AssertionError("Comparison report did not pass")
        baseline_rows = [
            row
            for row in comparison_report.get("per_episode", [])
            if int(row.get("horizon", -1)) == config.n_action_steps and int(row["task_id"]) in set(args.task_ids)
        ]
        baseline_lookup = {(int(row["task_id"]), int(row["episode_index"])): row for row in baseline_rows}
        if len(baseline_lookup) != episodes_total:
            raise AssertionError(f"Expected {episodes_total} paired baseline rows, found {len(baseline_lookup)}")
        transitions = {
            "failure_to_success": 0,
            "success_to_failure": 0,
            "success_to_success": 0,
            "failure_to_failure": 0,
        }
        pairs = []
        for row in episode_rows:
            key = (int(row["task_id"]), int(row["episode_index"]))
            baseline = baseline_lookup[key]
            baseline_success = bool(baseline["success"])
            fine_tuned_success = bool(row["success"])
            if not baseline_success and fine_tuned_success:
                transition = "failure_to_success"
            elif baseline_success and not fine_tuned_success:
                transition = "success_to_failure"
            elif baseline_success and fine_tuned_success:
                transition = "success_to_success"
            else:
                transition = "failure_to_failure"
            transitions[transition] += 1
            pairs.append(
                {
                    "task_id": key[0],
                    "episode_index": key[1],
                    "baseline_success": baseline_success,
                    "fine_tuned_success": fine_tuned_success,
                    "transition": transition,
                    "baseline_steps": int(baseline["steps"]),
                    "fine_tuned_steps": int(row["steps"]),
                    "step_delta": int(row["steps"]) - int(baseline["steps"]),
                }
            )
        paired_comparison = {
            "report": str(args.comparison_report.resolve()),
            "horizon": config.n_action_steps,
            "baseline_successes": sum(int(row["success"]) for row in baseline_rows),
            "fine_tuned_successes": successes_total,
            "transitions": transitions,
            "pairs": pairs,
        }

    csv_path = args.output_dir / "episodes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    is_full_spatial_benchmark = args.suite == "libero_spatial" and sorted(args.task_ids) == list(range(10))
    report = {
        "status": "passed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "stage": (
            "fine-tuned SmolVLA paired closed-loop evaluation"
            if fine_tune_report is not None
            else "full-task-coverage SmolVLA LIBERO Spatial evaluation (3 episodes per task)"
            if is_full_spatial_benchmark
            else "small multi-task SmolVLA LIBERO benchmark"
        ),
        "scope": {
            "suite": args.suite,
            "task_ids": args.task_ids,
            "episodes_per_task": args.episodes_per_task,
            "initial_state_indices": list(range(args.episodes_per_task)),
            "total_episodes": episodes_total,
            "resolution": args.resolution,
            "vector_env": "SyncVectorEnv",
        },
        "policy": {
            "repo_id": stage2_manifest["repo_id"],
            "revision": stage2_manifest["resolved_revision"],
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "adapter_dir": str(args.adapter_dir.resolve()) if args.adapter_dir is not None else None,
            "adapter_type": (str(adapter_config.peft_type.value) if adapter_config is not None else None),
            "weights_modified": fine_tune_report is not None,
            "fine_tune_report": (str(args.fine_tune_report.resolve()) if args.fine_tune_report is not None else None),
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "n_action_steps": config.n_action_steps,
            "chunk_size": config.chunk_size,
            "num_diffusion_steps": config.num_steps,
            "empty_cameras": config.empty_cameras,
        },
        "backbone": {
            "repo_id": backbone_report["repo_id"],
            "revision": backbone_report["resolved_revision"],
            "directory": str(args.backbone_dir.resolve()),
            "model_sha256": backbone_report["files"]["model.safetensors"]["sha256"],
        },
        "overall": {
            "successes": successes_total,
            "episodes": episodes_total,
            "success_rate": success_rate,
            "success_rate_percent": 100 * success_rate,
            "wilson_95_interval": confidence_interval,
            "wilson_95_interval_percent": [100 * value for value in confidence_interval],
            "mean_episode_steps": float(np.mean([row["steps"] for row in episode_rows])),
            "median_episode_steps": float(np.median([row["steps"] for row in episode_rows])),
            "total_valid_control_steps": int(valid_actions.shape[0]),
            "action_bounds_audit": {
                "outside_value_count": int(outside_bounds.sum().item()),
                "total_value_count": valid_actions.numel(),
                "outside_count_by_dimension": outside_bounds.sum(dim=0).tolist(),
                "maximum_underflow": float(torch.clamp(action_low - valid_actions, min=0).max().item()),
                "maximum_overflow": float(torch.clamp(valid_actions - action_high, min=0).max().item()),
                "actions_were_clipped": False,
            },
        },
        "per_task": task_reports,
        "per_episode": episode_rows,
        "paired_comparison": paired_comparison,
        "runtime": {
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "cuda_device": torch.cuda.get_device_name(),
            "cuda_capability": list(torch.cuda.get_device_capability()),
            "model_load_seconds": model_load_seconds,
            "benchmark_seconds": benchmark_seconds,
            "seconds_per_episode": benchmark_seconds / episodes_total,
            "effective_control_steps_per_second": valid_actions.shape[0] / benchmark_seconds,
            "model_inference": distribution([event["seconds"] for event in inference_events]),
            "queued_action_selection": distribution(queued_action_seconds),
            "memory_after_model_load": memory_after_load,
            "benchmark_memory": benchmark_memory,
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
            "videos": all_video_artifacts,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    benchmark_label = "full-task-coverage LIBERO Spatial evaluation" if is_full_spatial_benchmark else "small benchmark"
    print(f"SmolVLA {benchmark_label} completed. Report: {report_path}")


if __name__ == "__main__":
    main()
