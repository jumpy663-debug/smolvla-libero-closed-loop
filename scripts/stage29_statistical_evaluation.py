#!/usr/bin/env python

"""Run the frozen, resumable SmolVLA × LIBERO statistical evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lerobot
import mujoco
import numpy as np
import robosuite
import torch
import transformers
from lerobot.configs import PreTrainedConfig
from lerobot.envs import close_envs, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video

from stage5_small_benchmark import make_task_env
from statistical_evaluation_common import (
    EpisodeJob,
    atomic_write_text,
    build_jobs,
    canonical_json,
    load_frozen_protocol,
    record_path,
    select_jobs,
    sha256_file,
    trajectory_path,
    validate_resume_record,
    video_path,
)

DEFAULT_ASSET_ROOT = Path("/home/jump/projects/lerobot/outputs/reproduction/smolvla_libero")
DEFAULT_PROTOCOL = Path("protocols/statistical_evaluation_v1.json")
DEFAULT_OUTPUT_DIR = Path("outputs/statistical_evaluation_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=("pilot", "formal", "randomness"), default="formal")
    parser.add_argument("--policies", nargs="+", choices=("pretrained", "expert_only", "lora"))
    parser.add_argument("--horizons", nargs="+", type=int)
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--init-state-indices", nargs="+", type=int)
    parser.add_argument(
        "--max-new-episodes",
        type=int,
        default=None,
        help="Run at most this many missing jobs from the frozen manifest, then exit cleanly.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify the protocol, artifact hashes, and selected manifest without loading a model.",
    )
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def array_digest_update(digest: Any, key: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(key.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())


def nested_observation_arrays(value: Any, prefix: str = "") -> list[tuple[str, np.ndarray]]:
    if isinstance(value, dict):
        rows = []
        for key in sorted(value):
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(nested_observation_arrays(value[key], nested_prefix))
        return rows
    array = np.asarray(value)
    return [(prefix, array)]


def observation_sha256(observation: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, array in nested_observation_arrays(observation):
        array_digest_update(digest, key, array)
    return digest.hexdigest()


def action_trace_sha256(actions: list[np.ndarray]) -> str:
    if not actions:
        raise AssertionError("An episode produced no actions")
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 7 or not np.isfinite(array).all():
        raise AssertionError(f"Invalid action trace: {array.shape}")
    digest = hashlib.sha256()
    array_digest_update(digest, "action", array)
    return digest.hexdigest()


def extract_success(info: dict[str, Any]) -> bool:
    if "final_info" in info:
        final_info = info["final_info"]
        if not isinstance(final_info, dict):
            raise RuntimeError("Unsupported final_info format")
        values = final_info["is_success"]
        return bool(values.tolist()[0])
    if "is_success" in info:
        values = info["is_success"]
        if hasattr(values, "tolist"):
            listed = values.tolist()
            return bool(listed[0] if isinstance(listed, list) else listed)
        return bool(values)
    return False


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise AssertionError(f"Expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    if int(stream["width"]) <= 0 or int(stream["height"]) <= 0:
        raise AssertionError(f"Invalid video dimensions in {path}")
    return stream


def write_trajectory_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def assert_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise AssertionError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def audit_assets(protocol: dict[str, Any], asset_root: Path, repository_root: Path) -> dict[str, Any]:
    checked = []
    for model_name in ("backbone", "pretrained", "expert_only", "lora"):
        model = protocol["models"][model_name]
        directory = asset_root / model["relative_path_under_asset_root"]
        weights = directory / model["weights_file"]
        assert_sha256(weights, model["weights_sha256"])
        checked.append({"name": model_name, "path": str(weights), "sha256": model["weights_sha256"]})
        report_relative = model.get("training_report_relative_path_under_asset_root")
        if report_relative is not None:
            report = asset_root / report_relative
            assert_sha256(report, model["training_report_sha256"])
            checked.append(
                {
                    "name": f"{model_name}_training_report",
                    "path": str(report),
                    "sha256": model["training_report_sha256"],
                }
            )
    for source in protocol["legacy_reuse"]["sources"]:
        path = repository_root / source["path"]
        assert_sha256(path, source["sha256"])
        checked.append({"name": f"legacy:{source['path']}", "path": str(path), "sha256": source["sha256"]})
        source_report = asset_root / source["source_report_relative_path_under_asset_root"]
        assert_sha256(source_report, source["source_report_sha256"])
        checked.append(
            {
                "name": f"legacy_report:{source['path']}",
                "path": str(source_report),
                "sha256": source["source_report_sha256"],
            }
        )
    evidence = protocol["randomness"]["audit"]["same_seed_reproducibility_evidence"]
    evidence_path = repository_root / evidence["source"]
    assert_sha256(evidence_path, evidence["source_sha256"])
    checked.append(
        {
            "name": "same_seed_reproducibility_evidence",
            "path": str(evidence_path),
            "sha256": evidence["source_sha256"],
        }
    )
    return {"status": "passed", "checked_file_count": len(checked), "files": checked}


def model_paths(protocol: dict[str, Any], asset_root: Path, policy_name: str) -> tuple[Path, Path | None]:
    if policy_name == "pretrained":
        model = protocol["models"]["pretrained"]
        return asset_root / model["relative_path_under_asset_root"], None
    if policy_name == "expert_only":
        model = protocol["models"]["expert_only"]
        return asset_root / model["relative_path_under_asset_root"], None
    if policy_name == "lora":
        base = protocol["models"]["pretrained"]
        adapter = protocol["models"]["lora"]
        return (
            asset_root / base["relative_path_under_asset_root"],
            asset_root / adapter["relative_path_under_asset_root"],
        )
    raise ValueError(f"Unknown policy: {policy_name}")


def load_policy_condition(
    *,
    protocol: dict[str, Any],
    asset_root: Path,
    policy_name: str,
    action_horizon: int,
) -> tuple[torch.nn.Module, SmolVLAConfig, Any, Any, float]:
    checkpoint_dir, adapter_dir = model_paths(protocol, asset_root, policy_name)
    backbone_dir = asset_root / protocol["models"]["backbone"]["relative_path_under_asset_root"]
    config = PreTrainedConfig.from_pretrained(checkpoint_dir, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    if action_horizon <= 0 or action_horizon > config.chunk_size:
        raise ValueError(f"Invalid action horizon {action_horizon} for chunk size {config.chunk_size}")
    config.device = "cuda"
    config.vlm_model_name = str(backbone_dir.resolve())
    config.load_vlm_weights = True
    config.empty_cameras = 1
    config.use_amp = False
    config.n_action_steps = action_horizon
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint_dir),
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": {}},
            "tokenizer_processor": {"tokenizer_name": str(backbone_dir.resolve())},
            "device_processor": {"device": "cuda"},
        },
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(checkpoint_dir, config=config, local_files_only=True)
    if adapter_dir is not None:
        from peft import PeftConfig, PeftModel

        adapter_config = PeftConfig.from_pretrained(adapter_dir, local_files_only=True)
        if Path(adapter_config.base_model_name_or_path).resolve() != checkpoint_dir.resolve():
            raise AssertionError("LoRA adapter base model path does not match the frozen pretrained checkpoint")
        policy = PeftModel.from_pretrained(
            policy,
            adapter_dir,
            config=adapter_config,
            is_trainable=False,
            local_files_only=True,
        )
        policy.to("cuda")
    policy.eval()
    torch.cuda.synchronize()
    return policy, config, preprocessor, postprocessor, time.perf_counter() - started


def run_episode(
    *,
    job: EpisodeJob,
    protocol: dict[str, Any],
    protocol_sha256: str,
    output_dir: Path,
    policy: torch.nn.Module,
    config: SmolVLAConfig,
    preprocessor: Any,
    postprocessor: Any,
) -> dict[str, Any]:
    env_cfg, env, envs = make_task_env(
        suite=protocol["environment"]["suite"],
        task_id=job.task_id,
        n_envs=1,
        resolution=int(protocol["environment"]["observation_resolution"][0]),
    )
    frames: list[np.ndarray] = []
    try:
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, config)
        task_name = str(env.call("task")[0])
        task_description = str(env.call("task_description")[0])
        environment_limit = int(env.call("_max_episode_steps")[0])
        if job.mode != "pilot" and environment_limit != job.maximum_control_steps:
            raise AssertionError(
                f"Formal environment limit {environment_limit} differs from {job.maximum_control_steps}"
            )

        random.seed(job.policy_seed)
        np.random.seed(job.policy_seed)
        torch.manual_seed(job.policy_seed)
        torch.cuda.manual_seed_all(job.policy_seed)
        policy.reset()
        env.set_attr("init_state_id", [job.init_state_index])

        episode_started = time.perf_counter()
        observation, _ = env.reset(seed=[job.environment_seed])
        initial_hash = observation_sha256(observation)
        frames.append(np.ascontiguousarray(env.envs[0].render()))
        torch.cuda.reset_peak_memory_stats()

        actions: list[np.ndarray] = []
        rewards: list[float] = []
        terminated_trace: list[bool] = []
        truncated_trace: list[bool] = []
        success_trace: list[bool] = []
        inference_latencies: list[float] = []
        episode_success = False

        control_limit = min(job.maximum_control_steps, environment_limit)
        for step_index in range(control_limit):
            policy_observation = preprocess_observation(observation)
            policy_observation["task"] = list(env.call("task_description"))
            policy_observation = env_preprocessor(policy_observation)
            policy_observation = preprocessor(policy_observation)

            performs_model_inference = len(policy._queues[ACTION]) == 0
            torch.cuda.synchronize()
            inference_started = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(policy_observation)
            torch.cuda.synchronize()
            select_action_seconds = time.perf_counter() - inference_started
            if performs_model_inference:
                inference_latencies.append(select_action_seconds)

            action = postprocessor(action)
            action = env_postprocessor({ACTION: action})[ACTION]
            action_numpy = action.detach().float().cpu().numpy()
            if action_numpy.shape != (1, 7) or not np.isfinite(action_numpy).all():
                raise AssertionError(f"Invalid action at step {step_index}: {action_numpy.shape}")
            actions.append(action_numpy[0].copy())

            observation, reward, terminated, truncated, info = env.step(action_numpy)
            frames.append(np.ascontiguousarray(env.envs[0].render()))
            reward_value = float(np.asarray(reward)[0])
            terminated_value = bool(np.asarray(terminated)[0])
            truncated_value = bool(np.asarray(truncated)[0])
            success_value = extract_success(info)
            episode_success = episode_success or success_value
            rewards.append(reward_value)
            terminated_trace.append(terminated_value)
            truncated_trace.append(truncated_value)
            success_trace.append(success_value)
            if terminated_value or truncated_value:
                break

        torch.cuda.synchronize()
        episode_wall_seconds = time.perf_counter() - episode_started
        peak_memory = int(torch.cuda.max_memory_allocated())
        control_steps = len(actions)
        if control_steps <= 0:
            raise AssertionError("Episode did not execute a control step")
        logical_forwards = math.ceil(control_steps / job.action_horizon)
        if len(inference_latencies) != logical_forwards:
            raise AssertionError(
                f"Expected {logical_forwards} VLM forwards for {job.episode_id}, got {len(inference_latencies)}"
            )
        if episode_success and control_steps >= environment_limit:
            raise AssertionError("A successful episode unexpectedly reached the environment time limit")

        trajectory = trajectory_path(output_dir, job)
        write_trajectory_atomic(
            trajectory,
            {
                "actions": np.asarray(actions, dtype=np.float32),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "success": np.asarray(success_trace, dtype=np.bool_),
                "terminated": np.asarray(terminated_trace, dtype=np.bool_),
                "truncated": np.asarray(truncated_trace, dtype=np.bool_),
            },
        )
        trajectory_hash = sha256_file(trajectory)

        output_video = video_path(output_dir, job)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        frame_array = np.stack(frames, axis=0)
        write_video(output_video, frame_array, fps=env.unwrapped.metadata["render_fps"])
        video_probe = probe_video(output_video)
        video_hash = sha256_file(output_video)
        artifact_finished = time.perf_counter()

        latency = distribution(inference_latencies)
        action_array = np.asarray(actions, dtype=np.float32)
        action_outside_bounds = np.logical_or(action_array < -1.0, action_array > 1.0)
        source = {
            "pilot": "pipeline_validation",
            "formal": "new_formal",
            "randomness": "randomness_audit",
        }[job.mode]
        return {
            "schema_version": 1,
            "protocol_sha256": protocol_sha256,
            "episode_id": job.episode_id,
            "mode": job.mode,
            "source": source,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "policy": job.policy,
            "task_id": job.task_id,
            "task_name": task_name,
            "task_description": task_description,
            "init_state_index": job.init_state_index,
            "official_libero_initial_state": True,
            "environment_seed": job.environment_seed,
            "policy_seed": job.policy_seed,
            "action_horizon": job.action_horizon,
            "maximum_control_steps": job.maximum_control_steps,
            "success": episode_success,
            "control_steps": control_steps,
            "censored": not episode_success and control_steps >= job.maximum_control_steps,
            "sum_reward": float(np.sum(rewards, dtype=np.float64)),
            "max_reward": float(np.max(rewards)),
            "logical_vlm_forward_count": logical_forwards,
            "actual_vlm_forward_count": len(inference_latencies),
            "inference_latencies_seconds": inference_latencies,
            "inference_latency_mean_seconds": latency["mean"],
            "inference_latency_p50_seconds": latency["p50"],
            "inference_latency_p95_seconds": latency["p95"],
            "episode_wall_seconds": episode_wall_seconds,
            "artifact_encoding_and_hash_seconds": artifact_finished - episode_started - episode_wall_seconds,
            "peak_inference_gpu_memory_bytes": peak_memory,
            "initial_observation_sha256": initial_hash,
            "action_sha256": action_trace_sha256(actions),
            "action_bounds_audit": {
                "outside_value_count": int(action_outside_bounds.sum()),
                "total_value_count": int(action_array.size),
                "outside_count_by_dimension": action_outside_bounds.sum(axis=0).tolist(),
                "actions_were_clipped": False,
            },
            "trajectory_path": str(trajectory.resolve()),
            "trajectory_sha256": trajectory_hash,
            "video_path": str(output_video.resolve()),
            "video_sha256": video_hash,
            "video_frame_count": int(frame_array.shape[0]),
            "video_probe": video_probe,
            "versions": {
                "python": platform.python_version(),
                "lerobot": lerobot.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "mujoco": mujoco.__version__,
                "robosuite": robosuite.__version__,
            },
        }
    finally:
        close_envs(envs)


def group_jobs(jobs: list[EpisodeJob]) -> dict[tuple[str, int], list[EpisodeJob]]:
    grouped: dict[tuple[str, int], list[EpisodeJob]] = defaultdict(list)
    for job in jobs:
        grouped[(job.policy, job.action_horizon)].append(job)
    return dict(grouped)


def main() -> None:
    args = parse_args()
    if args.max_new_episodes is not None and args.max_new_episodes <= 0:
        raise ValueError("--max-new-episodes must be positive")
    repository_root = Path(__file__).resolve().parents[1]
    protocol_path = args.protocol if args.protocol.is_absolute() else repository_root / args.protocol
    protocol, protocol_sha256 = load_frozen_protocol(protocol_path)
    assets = audit_assets(protocol, args.asset_root.resolve(), repository_root)
    manifest = build_jobs(protocol, args.mode)
    selected = select_jobs(
        manifest,
        policies=set(args.policies) if args.policies is not None else None,
        horizons=set(args.horizons) if args.horizons is not None else None,
        task_ids=set(args.task_ids) if args.task_ids is not None else None,
        init_state_indices=set(args.init_state_indices) if args.init_state_indices is not None else None,
    )
    if not selected:
        raise ValueError("The requested filters select no jobs from the frozen manifest")

    existing = []
    missing = []
    for job in selected:
        path = record_path(args.output_dir, job)
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            validate_resume_record(record, job, protocol_sha256)
            existing.append(job)
        else:
            missing.append(job)
    if args.max_new_episodes is not None:
        missing = missing[: args.max_new_episodes]

    dry_run_report = {
        "status": "passed",
        "mode": args.mode,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "asset_audit": assets,
        "frozen_manifest_jobs": len(manifest),
        "selected_jobs": len(selected),
        "validated_resume_jobs": len(existing),
        "missing_jobs_selected_for_this_invocation": len(missing),
        "selected_episode_ids": [job.episode_id for job in selected],
    }
    if args.dry_run:
        print(canonical_json(dry_run_report), end="")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 29 requires a CUDA GPU")

    invocation_started = time.perf_counter()
    completed_this_invocation = []
    condition_runtime = []
    for (policy_name, horizon), jobs in group_jobs(missing).items():
        policy, config, preprocessor, postprocessor, load_seconds = load_policy_condition(
            protocol=protocol,
            asset_root=args.asset_root.resolve(),
            policy_name=policy_name,
            action_horizon=horizon,
        )
        condition_started = time.perf_counter()
        for job in jobs:
            episode_started = time.perf_counter()
            try:
                record = run_episode(
                    job=job,
                    protocol=protocol,
                    protocol_sha256=protocol_sha256,
                    output_dir=args.output_dir,
                    policy=policy,
                    config=config,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                atomic_write_text(record_path(args.output_dir, job), canonical_json(record))
                validate_resume_record(record, job, protocol_sha256)
                completed_this_invocation.append(job.episode_id)
                print(
                    f"Stage 29 mode={job.mode} policy={job.policy} H={job.action_horizon} "
                    f"task={job.task_id} init={job.init_state_index:02d} "
                    f"success={record['success']} steps={record['control_steps']} "
                    f"seconds={time.perf_counter() - episode_started:.1f}",
                    flush=True,
                )
            except Exception as error:
                error_record = {
                    "episode_id": job.episode_id,
                    "protocol_sha256": protocol_sha256,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_at_utc": datetime.now(UTC).isoformat(),
                }
                atomic_write_text(
                    args.output_dir / "errors" / f"{job.episode_id}.json",
                    canonical_json(error_record),
                )
                raise
        condition_runtime.append(
            {
                "policy": policy_name,
                "action_horizon": horizon,
                "model_load_seconds": load_seconds,
                "episode_count": len(jobs),
                "wall_seconds_after_model_load": time.perf_counter() - condition_started,
            }
        )
        del policy, preprocessor, postprocessor
        torch.cuda.empty_cache()

    all_validated = 0
    for job in selected:
        path = record_path(args.output_dir, job)
        if path.exists():
            validate_resume_record(json.loads(path.read_text(encoding="utf-8")), job, protocol_sha256)
            all_validated += 1
    full_manifest_validated = 0
    for job in manifest:
        path = record_path(args.output_dir, job)
        if path.exists():
            validate_resume_record(json.loads(path.read_text(encoding="utf-8")), job, protocol_sha256)
            full_manifest_validated += 1
    invocation = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "filters": {
            "policies": args.policies,
            "horizons": args.horizons,
            "task_ids": args.task_ids,
            "init_state_indices": args.init_state_indices,
            "max_new_episodes": args.max_new_episodes,
        },
        "selected_jobs": len(selected),
        "validated_selected_jobs_after_invocation": all_validated,
        "completed_this_invocation": len(completed_this_invocation),
        "completed_episode_ids": completed_this_invocation,
        "condition_runtime": condition_runtime,
        "wall_seconds": time.perf_counter() - invocation_started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    runtime_path = args.output_dir / "runtime" / f"{args.mode}.json"
    prior_invocations = []
    if runtime_path.exists():
        prior_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if prior_runtime.get("protocol_sha256") != protocol_sha256:
            raise AssertionError("Runtime ledger protocol hash differs from the frozen protocol")
        prior_invocations = list(prior_runtime.get("invocations", []))
    runtime = {
        "schema_version": 1,
        "status": "passed",
        "mode": args.mode,
        "protocol_sha256": protocol_sha256,
        "frozen_manifest_jobs": len(manifest),
        "validated_manifest_jobs": full_manifest_validated,
        "invocations": [*prior_invocations, invocation],
        "asset_audit": assets,
        "cuda_device": torch.cuda.get_device_name(),
    }
    atomic_write_text(runtime_path, canonical_json(runtime))
    print(
        f"Stage 29 {args.mode} invocation passed: completed={len(completed_this_invocation)}, "
        f"validated={all_validated}/{len(selected)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
