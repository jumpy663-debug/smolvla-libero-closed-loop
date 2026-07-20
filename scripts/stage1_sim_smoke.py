#!/usr/bin/env python

"""Run a model-free smoke test on one deterministic LIBERO task."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import lerobot
import mujoco
import numpy as np
import robosuite
import torch
from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action
from lerobot.envs.utils import preprocess_observation
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.utils.io_utils import write_video
from libero.libero import benchmark
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage1_sim_smoke"),
    )
    return parser.parse_args()


def describe_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: describe_tree(child) for key, child in value.items()}
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(np.isfinite(value).all()),
        }
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(torch.isfinite(value).all().item()),
        }
    return {"type": type(value).__name__, "value": value}


def add_batch_dimension(value: Any) -> Any:
    """Match the batched observation structure produced by Gym VectorEnv."""
    if isinstance(value, dict):
        return {key: add_batch_dimension(child) for key, child in value.items()}
    if isinstance(value, np.ndarray):
        return np.expand_dims(value, axis=0)
    return value


def tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert a single BCHW float image in [0, 1] back to HWC uint8."""
    image = image[0].permute(1, 2, 0).mul(255).round().clamp(0, 255).to(torch.uint8)
    return image.cpu().numpy()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")

    suite_factories = benchmark.get_benchmark_dict()
    if args.suite not in suite_factories:
        available = ", ".join(sorted(suite_factories))
        raise ValueError(f"Unknown suite {args.suite!r}; available suites: {available}")

    suite = suite_factories[args.suite]()
    if not 0 <= args.task_id < len(suite.tasks):
        raise ValueError(f"task id must be in [0, {len(suite.tasks) - 1}]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = LiberoEnv(
        task_suite=suite,
        task_id=args.task_id,
        task_suite_name=args.suite,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=args.resolution,
        observation_height=args.resolution,
        init_states=True,
        episode_index=0,
        n_envs=1,
        control_mode="relative",
    )

    frames: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    success_flags: list[bool] = []
    started_at = time.perf_counter()
    closed_cleanly = False

    try:
        observation, reset_info = env.reset(seed=args.seed)
        initial_observation = observation
        initial_processed = LiberoProcessorStep().observation(preprocess_observation(add_batch_dimension(observation)))

        expected_raw_images = {"image", "image2"}
        if set(observation["pixels"]) != expected_raw_images:
            raise AssertionError(
                f"Expected camera keys {sorted(expected_raw_images)}, got {sorted(observation['pixels'])}"
            )
        for key in expected_raw_images:
            image = observation["pixels"][key]
            expected_shape = (args.resolution, args.resolution, 3)
            if image.shape != expected_shape or image.dtype != np.uint8:
                raise AssertionError(
                    f"Unexpected {key} image: shape={image.shape}, dtype={image.dtype}; "
                    f"expected shape={expected_shape}, dtype=uint8"
                )

        state = initial_processed["observation.state"]
        if state.shape != (1, 8) or state.dtype != torch.float32:
            raise AssertionError(f"Unexpected processed state: shape={state.shape}, dtype={state.dtype}")
        if not torch.isfinite(state).all():
            raise AssertionError("Processed state contains non-finite values")

        for key in ("observation.images.image", "observation.images.image2"):
            image = initial_processed[key]
            expected_shape = (1, 3, args.resolution, args.resolution)
            if image.shape != expected_shape or image.dtype != torch.float32:
                raise AssertionError(f"Unexpected processed {key}: shape={image.shape}, dtype={image.dtype}")
            if image.min() < 0 or image.max() > 1:
                raise AssertionError(f"Processed {key} is outside [0, 1]")

        action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
        if action.shape != env.action_space.shape or not env.action_space.contains(action):
            raise AssertionError(f"Invalid no-op action {action}")

        frames.append(np.ascontiguousarray(env.render()))
        for _ in range(args.steps):
            observation, reward, terminated, truncated, info = env.step(action)
            frames.append(np.ascontiguousarray(env.render()))
            rewards.append(float(reward))
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            success_flags.append(bool(info.get("is_success", False)))

        final_processed = LiberoProcessorStep().observation(preprocess_observation(add_batch_dimension(observation)))

        for camera_name, image in initial_observation["pixels"].items():
            Image.fromarray(image).save(args.output_dir / f"initial_{camera_name}.png")
        Image.fromarray(tensor_image_to_uint8(initial_processed["observation.images.image"])).save(
            args.output_dir / "processed_image.png"
        )
        Image.fromarray(tensor_image_to_uint8(initial_processed["observation.images.image2"])).save(
            args.output_dir / "processed_image2.png"
        )
        Image.fromarray(frames[0]).save(args.output_dir / "initial_render.png")
        write_video(args.output_dir / "no_op_rollout.mp4", frames, fps=args.fps)
    finally:
        env.close()
        closed_cleanly = True

    report = {
        "status": "passed",
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": env.task,
        "task_description": env.task_description,
        "seed": args.seed,
        "steps": args.steps,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "closed_cleanly": closed_cleanly,
        "versions": {
            "python": platform.python_version(),
            "lerobot": lerobot.__version__,
            "mujoco": mujoco.__version__,
            "robosuite": robosuite.__version__,
        },
        "runtime": {
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "action_space": {
                "shape": list(env.action_space.shape),
                "dtype": str(env.action_space.dtype),
                "low": env.action_space.low.tolist(),
                "high": env.action_space.high.tolist(),
            },
            "reset_info": reset_info,
            "raw_initial_observation": describe_tree(initial_observation),
            "processed_initial_observation": describe_tree(initial_processed),
            "processed_final_observation": describe_tree(final_processed),
            "rewards": rewards,
            "terminated": terminated_flags,
            "truncated": truncated_flags,
            "success": success_flags,
        },
        "artifacts": {
            "agentview": str(args.output_dir / "initial_image.png"),
            "wrist": str(args.output_dir / "initial_image2.png"),
            "processed_agentview": str(args.output_dir / "processed_image.png"),
            "processed_wrist": str(args.output_dir / "processed_image2.png"),
            "render": str(args.output_dir / "initial_render.png"),
            "video": str(args.output_dir / "no_op_rollout.mp4"),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Stage 1 LIBERO smoke test passed. Report: {report_path}")


if __name__ == "__main__":
    main()
