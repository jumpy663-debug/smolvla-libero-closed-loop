#!/usr/bin/env python

"""Run one official-path SmolVLA closed-loop episode on a pinned LIBERO task."""

from __future__ import annotations

import argparse
import hashlib
import json
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
        default=Path("outputs/stage4_closed_loop_episode"),
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 4 requires a CUDA GPU, but torch.cuda.is_available() is false")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage2_manifest = verify_stage2_manifest(args.stage2_manifest)
    backbone_report = prepare_backbone(args.backbone_dir)
    if backbone_report["resolved_revision"] != BACKBONE_REVISION:
        raise AssertionError("Backbone revision is not the stage 3 pinned revision")
    if stage2_manifest["resolved_revision"] != EXPECTED_POLICY_REVISION:
        raise AssertionError("Policy revision is not the stage 2 pinned revision")

    env_cfg = LiberoEnvConfig(
        task=args.suite,
        task_ids=[args.task_id],
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        camera_name_mapping={
            "agentview_image": "camera1",
            "robot0_eye_in_hand_image": "camera2",
        },
        init_states=True,
        observation_height=args.resolution,
        observation_width=args.resolution,
        control_mode="relative",
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    if args.suite not in envs or args.task_id not in envs[args.suite]:
        close_envs(envs)
        raise KeyError(f"Requested environment {args.suite}/{args.task_id} was not created")
    env = envs[args.suite][args.task_id]
    if not isinstance(env, gym.vector.SyncVectorEnv):
        close_envs(envs)
        raise TypeError(f"Expected SyncVectorEnv, got {type(env).__name__}")

    closed_cleanly = False
    report: dict[str, Any] | None = None
    try:
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
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, config)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        policy = SmolVLAPolicy.from_pretrained(
            args.checkpoint_dir,
            config=config,
            local_files_only=True,
        )
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started
        memory_after_load = gpu_memory()

        model_inference_seconds: list[float] = []
        queued_action_seconds: list[float] = []
        original_select_action = policy.select_action

        def instrumented_select_action(batch, *call_args, **call_kwargs):
            performs_model_inference = len(policy._queues[ACTION]) == 0
            call_started = time.perf_counter()
            selected_action = original_select_action(batch, *call_args, **call_kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - call_started
            if performs_model_inference:
                model_inference_seconds.append(elapsed)
            else:
                queued_action_seconds.append(elapsed)
            return selected_action

        policy.select_action = instrumented_select_action

        frames: list[np.ndarray] = []

        def render_frame(vector_env: gym.vector.VectorEnv) -> None:
            frame = vector_env.envs[0].render()
            frames.append(np.ascontiguousarray(frame))

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        rollout_started = time.perf_counter()
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=[args.seed],
            return_observations=False,
            render_callback=render_frame,
        )
        torch.cuda.synchronize()
        rollout_seconds = time.perf_counter() - rollout_started
        rollout_memory = gpu_memory()

        actions = rollout_data[ACTION].detach().float().cpu()
        rewards = rollout_data["reward"].detach().float().cpu()
        successes = rollout_data["success"].detach().cpu()
        dones = rollout_data["done"].detach().cpu()
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 7:
            raise AssertionError(f"Unexpected rollout action shape: {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise AssertionError("Closed-loop actions contain non-finite values")
        if not torch.isfinite(rewards).all():
            raise AssertionError("Closed-loop rewards contain non-finite values")

        episode_steps = int(actions.shape[1])
        episode_success = bool(successes.any().item())
        terminated_before_limit = episode_steps < int(env.call("_max_episode_steps")[0])
        if len(frames) != episode_steps + 1:
            raise AssertionError(
                f"Expected one initial frame plus one per action; got {len(frames)} frames for {episode_steps} actions"
            )
        expected_model_calls = (episode_steps + config.n_action_steps - 1) // config.n_action_steps
        if len(model_inference_seconds) != expected_model_calls:
            raise AssertionError(
                f"Expected {expected_model_calls} model calls, measured {len(model_inference_seconds)}"
            )

        action_low = torch.from_numpy(env.single_action_space.low).view(1, 1, -1)
        action_high = torch.from_numpy(env.single_action_space.high).view(1, 1, -1)
        below_bounds = actions < action_low
        above_bounds = actions > action_high
        outside_bounds = below_bounds | above_bounds

        video_path = args.output_dir / "closed_loop_episode.mp4"
        write_video(video_path, frames, fps=env.unwrapped.metadata["render_fps"])
        trajectory_path = args.output_dir / "trajectory.npz"
        np.savez_compressed(
            trajectory_path,
            actions=actions.numpy(),
            rewards=rewards.numpy(),
            successes=successes.numpy(),
            dones=dones.numpy(),
        )

        task_name = env.call("task")[0]
        task_description = env.call("task_description")[0]
        report = {
            "status": "passed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "stage": "one official-path SmolVLA closed-loop LIBERO episode",
            "policy": {
                "repo_id": stage2_manifest["repo_id"],
                "revision": stage2_manifest["resolved_revision"],
                "checkpoint_dir": str(args.checkpoint_dir.resolve()),
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
            "environment": {
                "suite": args.suite,
                "task_id": args.task_id,
                "task_name": task_name,
                "task_description": task_description,
                "seed": args.seed,
                "resolution": args.resolution,
                "vector_env": type(env).__name__,
                "control_mode": env_cfg.control_mode,
                "episode_step_limit": int(env.call("_max_episode_steps")[0]),
                "render_fps": env.unwrapped.metadata["render_fps"],
                "closed_cleanly": True,
            },
            "episode": {
                "success": episode_success,
                "steps": episode_steps,
                "terminated_before_limit": terminated_before_limit,
                "sum_reward": float(rewards.sum().item()),
                "max_reward": float(rewards.max().item()),
                "success_value_count": int(successes.sum().item()),
                "final_done": bool(dones[0, -1].item()),
                "frame_count": len(frames),
                "action_shape": list(actions.shape),
                "action_min": float(actions.min().item()),
                "action_max": float(actions.max().item()),
                "action_mean": float(actions.mean().item()),
                "action_std": float(actions.std(unbiased=False).item()),
                "bounds_audit": {
                    "low": env.single_action_space.low.tolist(),
                    "high": env.single_action_space.high.tolist(),
                    "outside_value_count": int(outside_bounds.sum().item()),
                    "total_value_count": actions.numel(),
                    "outside_count_by_dimension": outside_bounds.sum(dim=(0, 1)).tolist(),
                    "maximum_underflow": float(torch.clamp(action_low - actions, min=0).max().item()),
                    "maximum_overflow": float(torch.clamp(actions - action_high, min=0).max().item()),
                    "actions_were_clipped": False,
                },
            },
            "runtime": {
                "mujoco_gl": os.environ.get("MUJOCO_GL"),
                "cuda_device": torch.cuda.get_device_name(),
                "cuda_capability": list(torch.cuda.get_device_capability()),
                "model_load_seconds": round(model_load_seconds, 4),
                "rollout_seconds": round(rollout_seconds, 4),
                "effective_control_steps_per_second": round(episode_steps / rollout_seconds, 4),
                "model_inference_calls": len(model_inference_seconds),
                "model_inference_seconds": distribution(model_inference_seconds),
                "queued_action_selection_seconds": distribution(queued_action_seconds),
                "memory_after_model_load": memory_after_load,
                "rollout_memory": rollout_memory,
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
                "video": str(video_path.resolve()),
                "video_sha256": sha256_file(video_path),
                "trajectory": str(trajectory_path.resolve()),
                "trajectory_sha256": sha256_file(trajectory_path),
            },
        }
    finally:
        close_envs(envs)
        closed_cleanly = True

    if report is None:
        raise RuntimeError("Rollout ended without producing a report")
    report["environment"]["closed_cleanly"] = closed_cleanly
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Stage 4 closed-loop episode completed. Report: {report_path}")


if __name__ == "__main__":
    main()
