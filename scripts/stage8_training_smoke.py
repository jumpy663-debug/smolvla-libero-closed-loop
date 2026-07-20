#!/usr/bin/env python

"""Validate local SmolVLA fine-tuning with a tiny Task-5 overfit run."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
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
from torch.utils.data import DataLoader, Subset

DATASET_REVISION = "2e98211ba27db9322efbaa5ed108b34bb03ca163"
TASK_INDEX = 32
TASK_PROMPT = "pick up the black bowl on the ramekin and place it on the plate"
TASK_EPISODES = [
    1264,
    1265,
    1266,
    1271,
    1288,
    1313,
    1341,
    1343,
    1388,
    1396,
    1404,
    1433,
    1443,
    1462,
    1465,
    1469,
    1481,
    1482,
    1496,
    1498,
    1519,
    1521,
    1524,
    1525,
    1561,
    1562,
    1577,
    1594,
    1598,
    1605,
    1626,
    1635,
    1646,
    1647,
    1680,
    1684,
    1685,
    1689,
    1692,
]
TRAIN_EPISODES = TASK_EPISODES[:8]
HELD_OUT_EPISODES = TASK_EPISODES[8:10]


def default_dataset_root() -> Path:
    """Return LeRobot's default Hugging Face dataset cache location."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    lerobot_home = Path(os.environ.get("HF_LEROBOT_HOME", hf_home / "lerobot"))
    return lerobot_home / "hub" / "datasets--lerobot--libero" / "snapshots" / DATASET_REVISION


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
        default=Path("outputs/stage8_training_smoke"),
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--log-freq", type=int, default=10)
    return parser.parse_args()


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def prepare_raw_batch(batch: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    batch = clone_batch(batch)
    for key in camera_keys:
        if key in batch and batch[key].dtype == torch.uint8:
            batch[key] = batch[key].float().div_(255.0)
    return batch


def midpoint_indices(dataset: LeRobotDataset, episode_ids: list[int]) -> list[int]:
    episode_column = dataset.hf_dataset["episode_index"]
    indices = []
    for episode_id in episode_ids:
        positions = [index for index, value in enumerate(episode_column) if value == episode_id]
        if not positions:
            raise AssertionError(f"Episode {episode_id} has no frames in the selected dataset")
        indices.append(positions[len(positions) // 2])
    return indices


def parameter_digest(policy: SmolVLAPolicy, trainable_only: bool = True) -> str:
    digest = hashlib.sha256()
    for name, parameter in policy.named_parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        raw = parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def fixed_probe_loss(
    policy: SmolVLAPolicy,
    preprocessor,
    raw_batch: dict[str, Any],
    camera_keys: list[str],
    seed: int,
) -> float:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.randn(
        (1, policy.config.chunk_size, policy.config.max_action_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    probe_time = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    batch = preprocessor(prepare_raw_batch(raw_batch, camera_keys))
    policy.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy(batch, noise=noise, time=probe_time)
    return float(loss.item())


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def make_optimizer(policy: SmolVLAPolicy, learning_rate: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )


def save_smoke_checkpoint(
    checkpoint_dir: Path,
    policy: SmolVLAPolicy,
    optimizer: torch.optim.Optimizer,
    preprocessor,
    postprocessor,
    step: int,
) -> dict[str, Any]:
    policy_dir = checkpoint_dir / "pretrained_model"
    policy.save_pretrained(policy_dir)
    preprocessor.save_pretrained(policy_dir)
    postprocessor.save_pretrained(policy_dir)
    optimizer_path = checkpoint_dir / "optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_path)
    state_path = checkpoint_dir / "training_state.json"
    state_path.write_text(json.dumps({"step": step}, indent=2) + "\n", encoding="utf-8")
    return {
        "policy_dir": str(policy_dir.resolve()),
        "model_size_bytes": (policy_dir / "model.safetensors").stat().st_size,
        "optimizer_path": str(optimizer_path.resolve()),
        "optimizer_size_bytes": optimizer_path.stat().st_size,
        "training_state_path": str(state_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 8 requires CUDA")
    if args.steps < 2 or args.learning_rate <= 0 or args.log_freq <= 0:
        raise ValueError("Use at least two steps and positive learning-rate/log-freq values")
    if not (args.dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Dataset snapshot is incomplete: {args.dataset_root}")
    if not (args.backbone_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"Backbone is incomplete: {args.backbone_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed report: {report_path}")
    checkpoint_dir = args.output_dir / "checkpoint_step_1"
    if checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_dir}")

    manifest = verify_stage2_manifest(args.stage2_manifest)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    delta_timestamps = {
        "observation.images.image": [0.0],
        "observation.images.image2": [0.0],
        "observation.state": [0.0],
        "action": [index / 10 for index in range(50)],
    }
    all_probe_episodes = TRAIN_EPISODES + HELD_OUT_EPISODES
    dataset_started = time.perf_counter()
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=args.dataset_root,
        episodes=all_probe_episodes,
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
        return_uint8=True,
    )
    task_indices = {int(value.item()) for value in dataset.hf_dataset["task_index"]}
    task_prompts = set(dataset.meta.tasks[dataset.meta.tasks["task_index"] == TASK_INDEX].index.tolist())
    if task_indices != {TASK_INDEX} or task_prompts != {TASK_PROMPT}:
        raise AssertionError(f"Unexpected Task-5 data: indices={task_indices}, prompts={task_prompts}")

    selected_indices = midpoint_indices(dataset, TRAIN_EPISODES)
    held_out_indices = midpoint_indices(dataset, HELD_OUT_EPISODES)
    train_subset = Subset(dataset, selected_indices)
    train_generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        train_subset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )
    train_iterator = iter(dataloader)
    train_probe_raws = [next(iter(DataLoader(Subset(dataset, [index]), batch_size=1))) for index in selected_indices]
    held_out_probe_raws = [next(iter(DataLoader(Subset(dataset, [index]), batch_size=1))) for index in held_out_indices]
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
    frozen_parameters = total_parameters - trainable_parameters
    trainable_groups = {
        "action_expert": sum(
            parameter.numel()
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad and "lm_expert" in name
        ),
        "state_action_projections": sum(
            parameter.numel()
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad and "lm_expert" not in name
        ),
    }
    optimizer = make_optimizer(policy, args.learning_rate)

    initial_digest = parameter_digest(policy)
    initial_train_probe_losses = [
        fixed_probe_loss(policy, preprocessor, raw_batch, camera_keys, args.seed + 100 + index)
        for index, raw_batch in enumerate(train_probe_raws)
    ]
    initial_held_out_probe_losses = [
        fixed_probe_loss(policy, preprocessor, raw_batch, camera_keys, args.seed + 200 + index)
        for index, raw_batch in enumerate(held_out_probe_raws)
    ]
    initial_train_probe = sum(initial_train_probe_losses) / len(initial_train_probe_losses)
    initial_held_out_probe = sum(initial_held_out_probe_losses) / len(initial_held_out_probe_losses)
    print(
        f"dataset={len(dataset)} frames, tiny_samples={len(train_subset)}, "
        f"trainable={trainable_parameters:,}/{total_parameters:,}, "
        f"initial_train_probe={initial_train_probe:.6f}, "
        f"initial_held_out_probe={initial_held_out_probe:.6f}",
        flush=True,
    )

    history: list[dict[str, Any]] = []
    first_step_parameter_name = "model.state_proj.weight"
    first_step_before = dict(policy.named_parameters())[first_step_parameter_name].detach().float().cpu().clone()
    checkpoint_report: dict[str, Any] | None = None
    reload_report: dict[str, Any] | None = None
    training_started = time.perf_counter()

    for step in range(1, args.steps + 1):
        try:
            raw_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(dataloader)
            raw_batch = next(train_iterator)
        batch = preprocessor(prepare_raw_batch(raw_batch, camera_keys))

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        step_started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_details = policy(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}: {grad_norm.item()}")
        optimizer.step()
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_started
        memory = gpu_memory()
        history.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm.item()),
                "seconds": step_seconds,
                "peak_allocated_bytes": memory["peak_allocated_bytes"],
                "peak_reserved_bytes": memory["peak_reserved_bytes"],
            }
        )

        if step == 1:
            first_step_after = dict(policy.named_parameters())[first_step_parameter_name].detach().float().cpu()
            first_step_delta = float((first_step_after - first_step_before).abs().max().item())
            if first_step_delta <= 0:
                raise AssertionError("The first optimizer step did not change state_proj.weight")
            digest_before_reload = parameter_digest(policy)
            checkpoint_report = save_smoke_checkpoint(
                checkpoint_dir,
                policy,
                optimizer,
                preprocessor,
                postprocessor,
                step,
            )

            del optimizer, policy
            gc.collect()
            torch.cuda.empty_cache()
            reload_started = time.perf_counter()
            reload_config = PreTrainedConfig.from_pretrained(checkpoint_dir / "pretrained_model", local_files_only=True)
            if not isinstance(reload_config, SmolVLAConfig):
                raise TypeError("Reloaded checkpoint did not contain a SmolVLAConfig")
            reload_config.device = "cuda"
            reload_config.vlm_model_name = str(args.backbone_dir.resolve())
            reload_config.load_vlm_weights = True
            policy = SmolVLAPolicy.from_pretrained(
                checkpoint_dir / "pretrained_model",
                config=reload_config,
                local_files_only=True,
            )
            policy.to("cuda")
            optimizer = make_optimizer(policy, args.learning_rate)
            optimizer_state = torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state)
            optimizer_to_device(optimizer, torch.device("cuda"))
            torch.cuda.synchronize()
            digest_after_reload = parameter_digest(policy)
            if digest_after_reload != digest_before_reload:
                raise AssertionError("Trainable parameter digest changed across checkpoint reload")
            reload_report = {
                "passed": True,
                "seconds": time.perf_counter() - reload_started,
                "trainable_parameter_sha256": digest_after_reload,
                "optimizer_state_entries": len(optimizer.state),
                "step": json.loads((checkpoint_dir / "training_state.json").read_text(encoding="utf-8"))["step"],
                "first_step_parameter": first_step_parameter_name,
                "first_step_max_abs_delta": first_step_delta,
            }
            print(
                f"step=1 loss={loss.item():.6f} grad_norm={grad_norm.item():.4f} "
                f"checkpoint_reload=passed peak={memory['peak_allocated_bytes'] / 1e9:.3f} GB",
                flush=True,
            )
        elif step % args.log_freq == 0 or step == args.steps:
            print(
                f"step={step} loss={loss.item():.6f} grad_norm={grad_norm.item():.4f} "
                f"seconds={step_seconds:.3f} peak={memory['peak_allocated_bytes'] / 1e9:.3f} GB",
                flush=True,
            )

    training_seconds = time.perf_counter() - training_started
    final_digest = parameter_digest(policy)
    final_train_probe_losses = [
        fixed_probe_loss(policy, preprocessor, raw_batch, camera_keys, args.seed + 100 + index)
        for index, raw_batch in enumerate(train_probe_raws)
    ]
    final_held_out_probe_losses = [
        fixed_probe_loss(policy, preprocessor, raw_batch, camera_keys, args.seed + 200 + index)
        for index, raw_batch in enumerate(held_out_probe_raws)
    ]
    final_train_probe = sum(final_train_probe_losses) / len(final_train_probe_losses)
    final_held_out_probe = sum(final_held_out_probe_losses) / len(final_held_out_probe_losses)
    train_probe_reduction = (initial_train_probe - final_train_probe) / initial_train_probe
    tiny_overfit_passed = final_train_probe < initial_train_probe and train_probe_reduction >= 0.10

    history_path = args.output_dir / "loss_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    report = {
        "status": "passed" if tiny_overfit_passed else "completed_without_overfit_gate",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "stage": "SmolVLA Task-5 local training and tiny-overfit smoke test",
        "scope": {
            "steps": args.steps,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": args.learning_rate,
            "mixed_precision": "bfloat16 autocast",
            "train_expert_only": config.train_expert_only,
            "freeze_vision_encoder": config.freeze_vision_encoder,
            "train_state_proj": config.train_state_proj,
            "chunk_size": config.chunk_size,
            "n_action_steps": config.n_action_steps,
        },
        "dataset": {
            "repo_id": "lerobot/libero",
            "snapshot_revision": DATASET_REVISION,
            "root": str(args.dataset_root.resolve()),
            "total_episodes": dataset.meta.total_episodes,
            "total_frames": dataset.meta.total_frames,
            "total_tasks": dataset.meta.total_tasks,
            "task_index": TASK_INDEX,
            "task_prompt": TASK_PROMPT,
            "task_episode_count": len(TASK_EPISODES),
            "task_episode_ids": TASK_EPISODES,
            "train_episode_ids": TRAIN_EPISODES,
            "held_out_episode_ids": HELD_OUT_EPISODES,
            "tiny_train_sample_indices": selected_indices,
            "held_out_sample_indices": held_out_indices,
            "loaded_frames": len(dataset),
            "load_seconds": dataset_seconds,
            "camera_keys": camera_keys,
            "feature_shapes": {
                key: dataset.meta.features[key]["shape"] for key in [*camera_keys, "observation.state", "action"]
            },
            "fps": dataset.meta.fps,
        },
        "policy": {
            "repo_id": manifest["repo_id"],
            "revision": manifest["resolved_revision"],
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "backbone_dir": str(args.backbone_dir.resolve()),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "frozen_parameters": frozen_parameters,
            "trainable_percent": 100 * trainable_parameters / total_parameters,
            "trainable_groups": trainable_groups,
            "initial_trainable_parameter_sha256": initial_digest,
            "final_trainable_parameter_sha256": final_digest,
            "model_load_seconds": model_load_seconds,
        },
        "single_batch_update": {
            "passed": bool(reload_report and reload_report["first_step_max_abs_delta"] > 0),
            "loss": history[0]["loss"],
            "grad_norm": history[0]["grad_norm"],
            "seconds": history[0]["seconds"],
            "peak_allocated_bytes": history[0]["peak_allocated_bytes"],
            "peak_reserved_bytes": history[0]["peak_reserved_bytes"],
        },
        "checkpoint": checkpoint_report,
        "reload": reload_report,
        "tiny_overfit": {
            "passed": tiny_overfit_passed,
            "gate": "fixed train-probe loss decreases by at least 10%",
            "initial_train_probe_loss": initial_train_probe,
            "final_train_probe_loss": final_train_probe,
            "initial_train_probe_losses": initial_train_probe_losses,
            "final_train_probe_losses": final_train_probe_losses,
            "train_probe_relative_reduction": train_probe_reduction,
            "initial_held_out_probe_loss": initial_held_out_probe,
            "final_held_out_probe_loss": final_held_out_probe,
            "initial_held_out_probe_losses": initial_held_out_probe_losses,
            "final_held_out_probe_losses": final_held_out_probe_losses,
            "held_out_probe_relative_change": (final_held_out_probe - initial_held_out_probe) / initial_held_out_probe,
            "training_seconds": training_seconds,
            "mean_step_seconds": sum(row["seconds"] for row in history) / len(history),
            "minimum_training_loss": min(row["loss"] for row in history),
            "final_training_loss": history[-1]["loss"],
        },
        "runtime": {
            "cuda_device": torch.cuda.get_device_name(),
            "cuda_capability": list(torch.cuda.get_device_capability()),
            "final_memory": gpu_memory(),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
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
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Stage 8 completed: status={report['status']}, "
        f"train_probe={initial_train_probe:.6f}->{final_train_probe:.6f}, "
        f"report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
