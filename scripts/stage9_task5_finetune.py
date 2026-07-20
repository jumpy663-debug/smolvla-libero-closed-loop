#!/usr/bin/env python

"""Short expert-only SmolVLA fine-tune on whole LIBERO Task-5 demonstrations."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lerobot
import torch
import transformers
from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from stage3_single_inference import gpu_memory, verify_stage2_manifest
from stage8_training_smoke import (
    DATASET_REVISION,
    TASK_EPISODES,
    TASK_INDEX,
    TASK_PROMPT,
    default_dataset_root,
    make_optimizer,
    optimizer_to_device,
    parameter_digest,
    prepare_raw_batch,
    save_smoke_checkpoint,
)
from torch.utils.data import DataLoader, Subset


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
        "--dataset-root",
        type=Path,
        default=default_dataset_root(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage9_task5_finetune"),
    )
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--peak-learning-rate", type=float, default=1e-5)
    parser.add_argument("--decay-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--validation-freq", type=int, default=100)
    parser.add_argument("--log-freq", type=int, default=50)
    return parser.parse_args()


def split_episodes(seed: int) -> tuple[list[int], list[int]]:
    episode_ids = list(TASK_EPISODES)
    random.Random(seed).shuffle(episode_ids)
    return sorted(episode_ids[:31]), sorted(episode_ids[31:])


def frame_indices(dataset: LeRobotDataset, episode_ids: list[int]) -> list[int]:
    episode_set = set(episode_ids)
    return [
        index for index, value in enumerate(dataset.hf_dataset["episode_index"]) if int(value.item()) in episode_set
    ]


def midpoint_indices(dataset: LeRobotDataset, episode_ids: list[int]) -> list[int]:
    episode_column = dataset.hf_dataset["episode_index"]
    result = []
    for episode_id in episode_ids:
        positions = [index for index, value in enumerate(episode_column) if int(value.item()) == episode_id]
        if not positions:
            raise AssertionError(f"Episode {episode_id} is missing")
        result.append(positions[len(positions) // 2])
    return result


def learning_rate_at_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_learning_rate: float,
    decay_learning_rate: float,
) -> float:
    if step <= warmup_steps:
        return peak_learning_rate * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return decay_learning_rate + (peak_learning_rate - decay_learning_rate) * cosine


def fixed_validation_losses(
    policy: SmolVLAPolicy,
    preprocessor,
    raw_batches: list[dict[str, Any]],
    camera_keys: list[str],
    seed: int,
) -> list[float]:
    losses = []
    policy.eval()
    for index, raw_batch in enumerate(raw_batches):
        generator = torch.Generator(device="cuda").manual_seed(seed + index)
        noise = torch.randn(
            (1, policy.config.chunk_size, policy.config.max_action_dim),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        probe_time = torch.tensor([0.5], device="cuda", dtype=torch.float32)
        batch = preprocessor(prepare_raw_batch(raw_batch, camera_keys))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch, noise=noise, time=probe_time)
        losses.append(float(loss.item()))
    return losses


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 9 requires CUDA")
    if args.steps <= args.warmup_steps or args.batch_size <= 0:
        raise ValueError("Training steps must exceed warmup steps and batch size must be positive")
    if args.validation_freq <= 0 or args.log_freq <= 0:
        raise ValueError("Validation and logging frequencies must be positive")
    if args.decay_learning_rate <= 0 or args.peak_learning_rate <= args.decay_learning_rate:
        raise ValueError("Learning rates are invalid")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    final_checkpoint_dir = args.output_dir / "checkpoint_final"
    if report_path.exists() or final_checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing Stage-9 run: {args.output_dir}")

    manifest = verify_stage2_manifest(args.stage2_manifest)
    train_episodes, validation_episodes = split_episodes(args.seed)
    if set(train_episodes) & set(validation_episodes):
        raise AssertionError("Training and validation episodes overlap")
    if sorted(train_episodes + validation_episodes) != sorted(TASK_EPISODES):
        raise AssertionError("Episode split does not cover Task 5 exactly")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    delta_timestamps = {
        "observation.images.image": [0.0],
        "observation.images.image2": [0.0],
        "observation.state": [0.0],
        "action": [index / 10 for index in range(50)],
    }
    dataset_started = time.perf_counter()
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=args.dataset_root,
        episodes=TASK_EPISODES,
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
        return_uint8=True,
    )
    task_indices = {int(value.item()) for value in dataset.hf_dataset["task_index"]}
    task_prompts = set(dataset.meta.tasks[dataset.meta.tasks["task_index"] == TASK_INDEX].index.tolist())
    if task_indices != {TASK_INDEX} or task_prompts != {TASK_PROMPT}:
        raise AssertionError(f"Unexpected Task-5 data: {task_indices=}, {task_prompts=}")

    train_indices = frame_indices(dataset, train_episodes)
    validation_indices = frame_indices(dataset, validation_episodes)
    validation_midpoints = midpoint_indices(dataset, validation_episodes)
    if set(train_indices) & set(validation_indices):
        raise AssertionError("Training and validation frames overlap")
    train_subset = Subset(dataset, train_indices)
    dataloader_generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=dataloader_generator,
        drop_last=True,
    )
    train_iterator = iter(dataloader)
    validation_raw_batches = [
        next(iter(DataLoader(Subset(dataset, [index]), batch_size=1))) for index in validation_midpoints
    ]
    camera_keys = list(dataset.meta.camera_keys)
    dataset_seconds = time.perf_counter() - dataset_started

    config = PreTrainedConfig.from_pretrained(args.checkpoint_dir, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    config.device = "cuda"
    config.vlm_model_name = str(args.backbone_dir.resolve())
    config.load_vlm_weights = True
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    config.train_state_proj = True
    config.compile_model = False

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint_dir),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(args.backbone_dir.resolve())},
            "device_processor": {"device": "cuda"},
        },
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint_dir,
        config=config,
        local_files_only=True,
    )
    policy.to("cuda")
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_started
    total_parameters = sum(parameter.numel() for parameter in policy.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    optimizer = make_optimizer(policy, args.peak_learning_rate)
    initial_digest = parameter_digest(policy)

    initial_validation_losses = fixed_validation_losses(
        policy,
        preprocessor,
        validation_raw_batches,
        camera_keys,
        args.seed + 10_000,
    )
    validation_history = [
        {"step": 0, "mean_loss": mean(initial_validation_losses), "losses": initial_validation_losses}
    ]
    print(
        f"train_episodes={len(train_episodes)} train_frames={len(train_indices)} "
        f"validation_episodes={len(validation_episodes)} validation_frames={len(validation_indices)} "
        f"batch_size={args.batch_size} initial_validation={mean(initial_validation_losses):.6f}",
        flush=True,
    )

    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    maximum_peak_allocated = 0
    maximum_peak_reserved = 0
    for step in range(1, args.steps + 1):
        try:
            raw_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(dataloader)
            raw_batch = next(train_iterator)
        batch = preprocessor(prepare_raw_batch(raw_batch, camera_keys))
        learning_rate = learning_rate_at_step(
            step,
            args.steps,
            args.warmup_steps,
            args.peak_learning_rate,
            args.decay_learning_rate,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        step_started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}")
        optimizer.step()
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_started
        memory = gpu_memory()
        maximum_peak_allocated = max(maximum_peak_allocated, memory["peak_allocated_bytes"])
        maximum_peak_reserved = max(maximum_peak_reserved, memory["peak_reserved_bytes"])
        history.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm.item()),
                "learning_rate": learning_rate,
                "seconds": step_seconds,
                "peak_allocated_bytes": memory["peak_allocated_bytes"],
                "peak_reserved_bytes": memory["peak_reserved_bytes"],
            }
        )

        if step == 1:
            print(
                f"batch4_gate=passed step=1 loss={loss.item():.6f} peak={memory['peak_allocated_bytes'] / 1e9:.3f} GB",
                flush=True,
            )
        if step % args.validation_freq == 0 or step == args.steps:
            losses = fixed_validation_losses(
                policy,
                preprocessor,
                validation_raw_batches,
                camera_keys,
                args.seed + 10_000,
            )
            validation_history.append({"step": step, "mean_loss": mean(losses), "losses": losses})
            print(f"step={step} validation_loss={mean(losses):.6f}", flush=True)
        elif step % args.log_freq == 0:
            recent = [row["loss"] for row in history[-args.log_freq :]]
            print(
                f"step={step} train_loss_mean={mean(recent):.6f} lr={learning_rate:.2e} seconds={step_seconds:.3f}",
                flush=True,
            )

    training_seconds = time.perf_counter() - training_started
    final_digest_before_save = parameter_digest(policy)
    checkpoint_report = save_smoke_checkpoint(
        final_checkpoint_dir,
        policy,
        optimizer,
        preprocessor,
        postprocessor,
        args.steps,
    )
    schedule_path = final_checkpoint_dir / "schedule.json"
    schedule_path.write_text(
        json.dumps(
            {
                "type": "linear_warmup_cosine_decay",
                "warmup_steps": args.warmup_steps,
                "peak_learning_rate": args.peak_learning_rate,
                "decay_learning_rate": args.decay_learning_rate,
                "total_steps": args.steps,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    del optimizer, policy
    gc.collect()
    torch.cuda.empty_cache()
    reload_started = time.perf_counter()
    reload_config = PreTrainedConfig.from_pretrained(final_checkpoint_dir / "pretrained_model", local_files_only=True)
    if not isinstance(reload_config, SmolVLAConfig):
        raise TypeError("Final checkpoint did not contain SmolVLAConfig")
    reload_config.device = "cuda"
    reload_config.vlm_model_name = str(args.backbone_dir.resolve())
    reload_config.load_vlm_weights = True
    reloaded_policy = SmolVLAPolicy.from_pretrained(
        final_checkpoint_dir / "pretrained_model",
        config=reload_config,
        local_files_only=True,
    )
    reloaded_policy.to("cuda")
    reloaded_optimizer = make_optimizer(reloaded_policy, args.peak_learning_rate)
    optimizer_state = torch.load(final_checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True)
    reloaded_optimizer.load_state_dict(optimizer_state)
    optimizer_to_device(reloaded_optimizer, torch.device("cuda"))
    torch.cuda.synchronize()
    final_digest_after_reload = parameter_digest(reloaded_policy)
    if final_digest_after_reload != final_digest_before_save:
        raise AssertionError("Final parameter digest changed across checkpoint reload")
    reload_seconds = time.perf_counter() - reload_started

    history_path = args.output_dir / "loss_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    validation_path = args.output_dir / "validation_history.json"
    validation_path.write_text(json.dumps(validation_history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = {
        "status": "passed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "stage": "SmolVLA Task-5 expert-only short fine-tune",
        "scope": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "mixed_precision": "bfloat16 autocast",
            "train_expert_only": config.train_expert_only,
            "freeze_vision_encoder": config.freeze_vision_encoder,
            "train_state_proj": config.train_state_proj,
            "chunk_size": config.chunk_size,
            "scheduler": "linear warmup plus cosine decay",
            "warmup_steps": args.warmup_steps,
            "peak_learning_rate": args.peak_learning_rate,
            "decay_learning_rate": args.decay_learning_rate,
        },
        "dataset": {
            "repo_id": "lerobot/libero",
            "snapshot_revision": DATASET_REVISION,
            "task_index": TASK_INDEX,
            "task_prompt": TASK_PROMPT,
            "task_episode_count": len(TASK_EPISODES),
            "task_frame_count": len(dataset),
            "train_episode_ids": train_episodes,
            "validation_episode_ids": validation_episodes,
            "train_frame_count": len(train_indices),
            "validation_frame_count": len(validation_indices),
            "split_overlap_count": len(set(train_indices) & set(validation_indices)),
            "load_seconds": dataset_seconds,
        },
        "policy": {
            "repo_id": manifest["repo_id"],
            "revision": manifest["resolved_revision"],
            "initial_checkpoint": str(args.checkpoint_dir.resolve()),
            "final_checkpoint": checkpoint_report,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "trainable_percent": 100 * trainable_parameters / total_parameters,
            "initial_trainable_parameter_sha256": initial_digest,
            "final_trainable_parameter_sha256": final_digest_after_reload,
            "weights_changed": initial_digest != final_digest_after_reload,
            "reload_passed": True,
            "reload_seconds": reload_seconds,
            "optimizer_state_entries": len(reloaded_optimizer.state),
            "model_load_seconds": model_load_seconds,
        },
        "optimization": {
            "all_losses_finite": all(math.isfinite(row["loss"]) for row in history),
            "all_grad_norms_finite": all(math.isfinite(row["grad_norm"]) for row in history),
            "first_100_train_loss_mean": mean([row["loss"] for row in history[:100]]),
            "last_100_train_loss_mean": mean([row["loss"] for row in history[-100:]]),
            "minimum_train_loss": min(row["loss"] for row in history),
            "final_train_loss": history[-1]["loss"],
            "initial_validation_loss": validation_history[0]["mean_loss"],
            "best_validation": min(validation_history, key=lambda item: item["mean_loss"]),
            "final_validation_loss": validation_history[-1]["mean_loss"],
            "validation_history": validation_history,
        },
        "runtime": {
            "cuda_device": torch.cuda.get_device_name(),
            "cuda_capability": list(torch.cuda.get_device_capability()),
            "training_seconds": training_seconds,
            "mean_step_seconds": mean([row["seconds"] for row in history]),
            "maximum_peak_allocated_bytes": maximum_peak_allocated,
            "maximum_peak_reserved_bytes": maximum_peak_reserved,
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        },
        "versions": {
            "python": platform.python_version(),
            "lerobot": lerobot.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "artifacts": {
            "report": str(report_path.resolve()),
            "loss_history_csv": str(history_path.resolve()),
            "validation_history_json": str(validation_path.resolve()),
            "schedule_json": str(schedule_path.resolve()),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Stage 9 training completed: validation "
        f"{validation_history[0]['mean_loss']:.6f}->{validation_history[-1]['mean_loss']:.6f}; "
        f"checkpoint={final_checkpoint_dir / 'pretrained_model'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
