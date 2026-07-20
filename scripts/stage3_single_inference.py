#!/usr/bin/env python

"""Run one pinned SmolVLA action-chunk inference on a real LIBERO reset observation."""

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

import huggingface_hub
import lerobot
import mujoco
import numpy as np
import robosuite
import torch
import transformers
from huggingface_hub import HfApi, snapshot_download
from lerobot.configs import PreTrainedConfig
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.env_processor import LiberoProcessorStep
from libero.libero import benchmark

BACKBONE_REPO_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BACKBONE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
BACKBONE_ALLOW_PATTERNS = ["*.json", "*.txt", "model.safetensors"]
EXPECTED_POLICY_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"


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
        "--output-dir",
        type=Path,
        default=Path("outputs/stage3_single_inference"),
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    finite = bool(torch.isfinite(tensor).all().item())
    description: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "finite": finite,
    }
    if finite and tensor.numel() > 0 and (tensor.is_floating_point() or tensor.is_complex()):
        value = tensor.detach().float()
        description.update(
            {
                "min": float(value.min().item()),
                "max": float(value.max().item()),
                "mean": float(value.mean().item()),
                "std": float(value.std(unbiased=False).item()),
            }
        )
    return description


def describe_batch(batch: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = describe_tensor(value)
        elif isinstance(value, dict):
            result[key] = describe_batch(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = {"type": type(value).__name__, "value": value}
        elif isinstance(value, list):
            result[key] = {"type": "list", "length": len(value), "preview": value[:2]}
        else:
            result[key] = {"type": type(value).__name__}
    return result


def add_batch_dimension(value: Any) -> Any:
    """Match the nested observation structure produced by a one-worker Gym VectorEnv."""
    if isinstance(value, dict):
        return {key: add_batch_dimension(child) for key, child in value.items()}
    if isinstance(value, np.ndarray):
        return np.expand_dims(value, axis=0)
    return value


def verify_stage2_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    resolved = manifest.get("resolved_revision")
    if resolved != EXPECTED_POLICY_REVISION:
        raise AssertionError(
            f"Unexpected policy revision in stage 2 manifest: {resolved}; expected {EXPECTED_POLICY_REVISION}"
        )
    backbone = manifest.get("external_backbone", {})
    backbone_revision = backbone.get("resolved_revision")
    if backbone_revision != BACKBONE_REVISION:
        raise AssertionError(
            f"Unexpected backbone revision in stage 2 manifest: {backbone_revision}; expected {BACKBONE_REVISION}"
        )
    return manifest


def prepare_backbone(backbone_dir: Path) -> dict[str, Any]:
    backbone_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    api = HfApi()
    info = api.model_info(BACKBONE_REPO_ID, revision=BACKBONE_REVISION, files_metadata=True)
    if info.sha != BACKBONE_REVISION:
        raise AssertionError(f"Backbone revision changed unexpectedly: {info.sha}")

    remote_files: dict[str, dict[str, Any]] = {}
    for sibling in info.siblings or []:
        name = sibling.rfilename
        if name.startswith("onnx/") or name in {".gitattributes", "README.md"}:
            continue
        if not (name.endswith(".json") or name.endswith(".txt") or name == "model.safetensors"):
            continue
        lfs_sha256 = getattr(sibling.lfs, "sha256", None) if sibling.lfs else None
        remote_files[name] = {"size_bytes": sibling.size, "lfs_sha256": lfs_sha256}

    snapshot_path = Path(
        snapshot_download(
            repo_id=BACKBONE_REPO_ID,
            revision=BACKBONE_REVISION,
            local_dir=backbone_dir,
            allow_patterns=BACKBONE_ALLOW_PATTERNS,
            max_workers=8,
        )
    )
    if snapshot_path.resolve() != backbone_dir.resolve():
        raise AssertionError(f"Unexpected backbone path: {snapshot_path}")

    required = {"config.json", "preprocessor_config.json", "tokenizer.json", "model.safetensors"}
    missing = sorted(name for name in required if not (backbone_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"Backbone is missing required runtime files: {missing}")

    local_files: dict[str, dict[str, Any]] = {}
    for name, remote in sorted(remote_files.items()):
        path = backbone_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Expected runtime file was not downloaded: {name}")
        local_sha256 = sha256_file(path)
        if remote["lfs_sha256"] and local_sha256 != remote["lfs_sha256"]:
            raise AssertionError(
                f"Backbone SHA-256 mismatch for {name}: local={local_sha256}, hub={remote['lfs_sha256']}"
            )
        local_files[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": local_sha256,
            "hub_lfs_sha256": remote["lfs_sha256"],
        }

    return {
        "repo_id": BACKBONE_REPO_ID,
        "resolved_revision": info.sha,
        "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "runtime_file_count": len(local_files),
        "runtime_size_bytes": sum(item["size_bytes"] for item in local_files.values()),
        "files": local_files,
    }


def gpu_memory() -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 3 requires a CUDA GPU, but torch.cuda.is_available() is false")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    backbone_dir = args.output_dir / "backbone"
    stage2_manifest = verify_stage2_manifest(args.stage2_manifest)
    backbone_report = prepare_backbone(backbone_dir)
    download_report_path = args.output_dir / "backbone_manifest.json"
    download_report_path.write_text(json.dumps(backbone_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Pinned backbone ready: {backbone_dir}")
    print(f"Backbone manifest: {download_report_path}")
    if args.download_only:
        return

    suite_factories = benchmark.get_benchmark_dict()
    if args.suite not in suite_factories:
        raise ValueError(f"Unknown suite {args.suite!r}")
    suite = suite_factories[args.suite]()
    if not 0 <= args.task_id < len(suite.tasks):
        raise ValueError(f"task id must be in [0, {len(suite.tasks) - 1}]")

    env = LiberoEnv(
        task_suite=suite,
        task_id=args.task_id,
        task_suite_name=args.suite,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        camera_name_mapping={
            "agentview_image": "camera1",
            "robot0_eye_in_hand_image": "camera2",
        },
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=args.resolution,
        observation_height=args.resolution,
        init_states=True,
        episode_index=0,
        n_envs=1,
        control_mode="relative",
    )

    closed_cleanly = False
    started_at = time.perf_counter()
    try:
        observation, reset_info = env.reset(seed=args.seed)
        processed_observation = LiberoProcessorStep().observation(
            preprocess_observation(add_batch_dimension(observation))
        )
        processed_observation["task"] = env.task_description

        config = PreTrainedConfig.from_pretrained(args.checkpoint_dir, local_files_only=True)
        if not isinstance(config, SmolVLAConfig):
            raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
        config.device = "cuda"
        config.vlm_model_name = str(backbone_dir.resolve())
        config.load_vlm_weights = True
        config.empty_cameras = 1
        config.use_amp = False

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(args.checkpoint_dir),
            preprocessor_overrides={
                "rename_observations_processor": {"rename_map": {}},
                "tokenizer_processor": {"tokenizer_name": str(backbone_dir.resolve())},
                "device_processor": {"device": "cuda"},
            },
        )
        model_input = preprocessor(processed_observation)

        if model_input["observation.state"].shape != (1, 8):
            raise AssertionError(f"Expected normalized 8-D LIBERO state, got {model_input['observation.state'].shape}")
        expected_images = {
            "observation.images.camera1",
            "observation.images.camera2",
        }
        if not expected_images.issubset(model_input):
            raise AssertionError(f"Missing model image inputs: {sorted(expected_images - model_input.keys())}")

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

        parameter_count = sum(parameter.numel() for parameter in policy.parameters())
        trainable_parameter_count = sum(
            parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
        )

        policy.reset()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            normalized_action_chunk = policy.predict_action_chunk(model_input)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        inference_memory = gpu_memory()

        action_chunk = postprocessor(normalized_action_chunk)
        if action_chunk.shape != (1, 50, 7):
            raise AssertionError(f"Unexpected action chunk shape: {tuple(action_chunk.shape)}")
        if not torch.isfinite(action_chunk).all():
            raise AssertionError("Action chunk contains non-finite values")

        action_chunk_cpu = action_chunk.detach().float().cpu()
        action_low = torch.from_numpy(env.action_space.low).view(1, 1, -1)
        action_high = torch.from_numpy(env.action_space.high).view(1, 1, -1)
        below_bounds = action_chunk_cpu < action_low
        above_bounds = action_chunk_cpu > action_high
        outside_bounds = below_bounds | above_bounds
        np.save(args.output_dir / "action_chunk.npy", action_chunk_cpu.numpy())
        torch.save(model_input["observation.state"].detach().cpu(), args.output_dir / "normalized_state.pt")

        report = {
            "status": "passed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "stage": "real LIBERO reset observation -> one SmolVLA action chunk",
            "policy": {
                "repo_id": stage2_manifest["repo_id"],
                "revision": stage2_manifest["resolved_revision"],
                "checkpoint_dir": str(args.checkpoint_dir.resolve()),
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_parameter_count,
                "chunk_size": config.chunk_size,
                "action_dimension": config.action_feature.shape[0],
                "num_diffusion_steps": config.num_steps,
                "empty_cameras": config.empty_cameras,
            },
            "backbone": backbone_report,
            "environment": {
                "suite": args.suite,
                "task_id": args.task_id,
                "task_name": env.task,
                "task_description": env.task_description,
                "seed": args.seed,
                "resolution": args.resolution,
                "reset_info": reset_info,
                "closed_cleanly": True,
            },
            "runtime": {
                "mujoco_gl": os.environ.get("MUJOCO_GL"),
                "cuda_device": torch.cuda.get_device_name(),
                "cuda_capability": list(torch.cuda.get_device_capability()),
                "model_load_seconds": round(model_load_seconds, 4),
                "inference_seconds": round(inference_seconds, 4),
                "inference_hz": round(1.0 / inference_seconds, 4),
                "memory_after_model_load": memory_after_load,
                "inference_memory": inference_memory,
                "total_elapsed_seconds": round(time.perf_counter() - started_at, 4),
            },
            "inputs": describe_batch(model_input),
            "outputs": {
                "normalized_action_chunk": describe_tensor(normalized_action_chunk),
                "unnormalized_action_chunk": describe_tensor(action_chunk_cpu),
                "first_action": action_chunk_cpu[0, 0].tolist(),
                "last_action": action_chunk_cpu[0, -1].tolist(),
                "environment_bounds_audit": {
                    "low": env.action_space.low.tolist(),
                    "high": env.action_space.high.tolist(),
                    "outside_value_count": int(outside_bounds.sum().item()),
                    "total_value_count": action_chunk_cpu.numel(),
                    "outside_count_by_dimension": outside_bounds.sum(dim=(0, 1)).tolist(),
                    "maximum_underflow": float(torch.clamp(action_low - action_chunk_cpu, min=0).max().item()),
                    "maximum_overflow": float(torch.clamp(action_chunk_cpu - action_high, min=0).max().item()),
                    "note": (
                        "This records raw postprocessed policy commands. Stage 4 will preserve the "
                        "official evaluation path and test how the simulator consumes them."
                    ),
                },
            },
            "versions": {
                "python": platform.python_version(),
                "lerobot": lerobot.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "huggingface_hub": huggingface_hub.__version__,
                "mujoco": mujoco.__version__,
                "robosuite": robosuite.__version__,
            },
            "artifacts": {
                "report": str((args.output_dir / "report.json").resolve()),
                "backbone_manifest": str(download_report_path.resolve()),
                "action_chunk": str((args.output_dir / "action_chunk.npy").resolve()),
                "normalized_state": str((args.output_dir / "normalized_state.pt").resolve()),
            },
        }
    finally:
        env.close()
        closed_cleanly = True

    report["environment"]["closed_cleanly"] = closed_cleanly
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Stage 3 single-inference test passed. Report: {report_path}")


if __name__ == "__main__":
    main()
