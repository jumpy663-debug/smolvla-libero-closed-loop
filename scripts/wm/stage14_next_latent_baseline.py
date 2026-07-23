#!/usr/bin/env python

"""Train paired action-conditioned and no-action next-latent predictors.

The two models use identical architecture, initialization, batch schedule, and
optimization settings. The no-action control receives zero actions. Only train
shards update weights; validation shards select checkpoints; test remains
untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

LATENT_DIMENSION = 384
CAMERAS = 2
TOKENS_PER_CAMERA = 16
TOKENS_PER_FRAME = CAMERAS * TOKENS_PER_CAMERA
STATE_DIMENSION = 8
ACTION_DIMENSION = 7
SPLITS = ("train", "validation")
VARIANTS = ("action_conditioned", "no_action")
COUNTERFACTUAL_SEED = 1417


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-manifest",
        type=Path,
        default=Path("results/wm/stage13_full_feature_cache/shards.csv"),
    )
    parser.add_argument(
        "--stage13-report",
        type=Path,
        default=Path("results/wm/stage13_full_feature_cache/report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wm/stage14_next_latent_baseline"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage14_next_latent_baseline"),
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--hidden-dimension", type=int, default=192)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1400)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace non-identical Stage 14 outputs.",
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tensor_dict_sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(tensors):
        tensor = tensors[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_text(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_deterministic(path: Path, content: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-identical output: {path}")
    path.write_text(content, encoding="utf-8")


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")


@dataclass
class TransitionStore:
    visual_tokens: torch.Tensor
    observation_state: torch.Tensor
    action: torch.Tensor
    current_indices: torch.Tensor
    next_indices: torch.Tensor
    task_ids: torch.Tensor
    episode_ids: torch.Tensor

    @property
    def frame_count(self) -> int:
        return len(self.visual_tokens)

    @property
    def transition_count(self) -> int:
        return len(self.current_indices)


@dataclass
class Normalization:
    latent_mean: torch.Tensor
    latent_std: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

    def to(self, device: torch.device) -> Normalization:
        return Normalization(
            latent_mean=self.latent_mean.to(device),
            latent_std=self.latent_std.to(device),
            state_mean=self.state_mean.to(device),
            state_std=self.state_std.to(device),
            action_mean=self.action_mean.to(device),
            action_std=self.action_std.to(device),
        )


def load_transition_store(
    manifest_rows: list[dict[str, str]],
    *,
    split: str,
    repo_root: Path,
) -> TransitionStore:
    rows = sorted(
        [row for row in manifest_rows if row["split"] == split],
        key=lambda row: int(row["episode_index"]),
    )
    if not rows:
        raise ValueError(f"No shard rows found for split {split}")
    total_frames = sum(int(row["length"]) for row in rows)
    visual_tokens = torch.empty(
        total_frames,
        TOKENS_PER_FRAME,
        LATENT_DIMENSION,
        dtype=torch.float16,
    )
    observation_state = torch.empty(total_frames, STATE_DIMENSION, dtype=torch.float32)
    actions = torch.empty(total_frames, ACTION_DIMENSION, dtype=torch.float32)
    current_indices: list[torch.Tensor] = []
    next_indices: list[torch.Tensor] = []
    task_ids: list[torch.Tensor] = []
    episode_ids: list[torch.Tensor] = []
    cursor = 0

    for row in rows:
        length = int(row["length"])
        episode_index = int(row["episode_index"])
        task_id = int(row["benchmark_task_id"])
        shard_path = repo_root / row["shard_file"]
        tensors = load_file(shard_path)
        tokens = tensors["visual_tokens"].reshape(length, TOKENS_PER_FRAME, LATENT_DIMENSION)
        if tokens.shape != (length, TOKENS_PER_FRAME, LATENT_DIMENSION):
            raise AssertionError(f"Unexpected visual tensor shape in {shard_path}: {tokens.shape}")
        if tensors["observation_state"].shape != (length, STATE_DIMENSION):
            raise AssertionError(f"Unexpected state shape in {shard_path}")
        if tensors["action"].shape != (length, ACTION_DIMENSION):
            raise AssertionError(f"Unexpected action shape in {shard_path}")
        if not bool(tensors["is_terminal"][-1]) or int(tensors["is_terminal"].sum()) != 1:
            raise AssertionError(f"Invalid terminal mask in {shard_path}")
        frame_slice = slice(cursor, cursor + length)
        visual_tokens[frame_slice].copy_(tokens)
        observation_state[frame_slice].copy_(tensors["observation_state"])
        actions[frame_slice].copy_(tensors["action"])
        transitions = torch.arange(cursor, cursor + length - 1, dtype=torch.int64)
        current_indices.append(transitions)
        next_indices.append(transitions + 1)
        task_ids.append(torch.full((length - 1,), task_id, dtype=torch.int64))
        episode_ids.append(torch.full((length - 1,), episode_index, dtype=torch.int64))
        cursor += length

    if cursor != total_frames:
        raise AssertionError(f"Loaded {cursor} frames, expected {total_frames}")
    store = TransitionStore(
        visual_tokens=visual_tokens,
        observation_state=observation_state,
        action=actions,
        current_indices=torch.cat(current_indices),
        next_indices=torch.cat(next_indices),
        task_ids=torch.cat(task_ids),
        episode_ids=torch.cat(episode_ids),
    )
    if torch.any(store.episode_ids[1:] < store.episode_ids[:-1]):
        raise AssertionError("Transition episode ordering is not monotonic")
    return store


def compute_normalization(store: TransitionStore) -> Normalization:
    latent_values = store.visual_tokens.float()
    latent_mean = latent_values.mean(dim=(0, 1))
    latent_std = latent_values.std(dim=(0, 1), unbiased=False).clamp_min(1e-4)
    state_mean = store.observation_state.mean(dim=0)
    state_std = store.observation_state.std(dim=0, unbiased=False).clamp_min(1e-4)
    action_mean = store.action.mean(dim=0)
    action_std = store.action.std(dim=0, unbiased=False).clamp_min(1e-4)
    return Normalization(
        latent_mean=latent_mean,
        latent_std=latent_std,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
    )


def normalization_json(normalization: Normalization) -> dict[str, list[float]]:
    return {
        "latent_mean": [round(float(value), 8) for value in normalization.latent_mean],
        "latent_std": [round(float(value), 8) for value in normalization.latent_std],
        "state_mean": [round(float(value), 8) for value in normalization.state_mean],
        "state_std": [round(float(value), 8) for value in normalization.state_std],
        "action_mean": [round(float(value), 8) for value in normalization.action_mean],
        "action_std": [round(float(value), 8) for value in normalization.action_std],
    }


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dimension: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dimension)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension * 2),
            nn.GELU(),
            nn.Linear(hidden_dimension * 2, hidden_dimension),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(self.norm(hidden))


class TokenDynamicsModel(nn.Module):
    def __init__(self, hidden_dimension: int, depth: int) -> None:
        super().__init__()
        if hidden_dimension <= 0 or depth <= 0:
            raise ValueError("hidden_dimension and depth must be positive")
        self.hidden_dimension = hidden_dimension
        self.depth = depth
        self.token_projection = nn.Linear(LATENT_DIMENSION, hidden_dimension)
        self.context_projection = nn.Sequential(
            nn.LayerNorm(LATENT_DIMENSION + STATE_DIMENSION + ACTION_DIMENSION),
            nn.Linear(LATENT_DIMENSION + STATE_DIMENSION + ACTION_DIMENSION, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
        )
        self.position_embedding = nn.Parameter(torch.empty(TOKENS_PER_FRAME, hidden_dimension))
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dimension) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(hidden_dimension)
        self.output_projection = nn.Linear(hidden_dimension, LATENT_DIMENSION)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        current_latent: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if current_latent.shape[1:] != (TOKENS_PER_FRAME, LATENT_DIMENSION):
            raise ValueError(f"Unexpected latent shape: {tuple(current_latent.shape)}")
        global_latent = current_latent.mean(dim=1)
        context = self.context_projection(torch.cat([global_latent, state, action], dim=-1))
        hidden = self.token_projection(current_latent) + context[:, None, :] + self.position_embedding[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        delta = self.output_projection(self.output_norm(hidden))
        return current_latent + delta


def make_batch_schedule(
    transition_count: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    if transition_count <= 0 or steps <= 0 or batch_size <= 0:
        raise ValueError("transition_count, steps, and batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        transition_count,
        (steps, batch_size),
        generator=generator,
        dtype=torch.int64,
    )


def normalized_batch(
    store: TransitionStore,
    transition_positions: torch.Tensor,
    normalization: Normalization,
    device: torch.device,
    *,
    action_mode: str,
    shuffled_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_indices = store.current_indices[transition_positions]
    next_indices = store.next_indices[transition_positions]
    current = store.visual_tokens[current_indices].to(device=device, dtype=torch.float32)
    target = store.visual_tokens[next_indices].to(device=device, dtype=torch.float32)
    state = store.observation_state[current_indices].to(device)
    if action_mode == "correct":
        action = store.action[current_indices].to(device)
    elif action_mode == "zero":
        action = torch.zeros(len(current_indices), ACTION_DIMENSION, device=device)
    elif action_mode == "shuffled":
        if shuffled_positions is None:
            raise ValueError("shuffled_positions are required for shuffled actions")
        shuffled_current = store.current_indices[shuffled_positions]
        action = store.action[shuffled_current].to(device)
    else:
        raise ValueError(f"Unknown action mode: {action_mode}")
    current = (current - normalization.latent_mean) / normalization.latent_std
    target = (target - normalization.latent_mean) / normalization.latent_std
    state = (state - normalization.state_mean) / normalization.state_std
    if action_mode != "zero":
        action = (action - normalization.action_mean) / normalization.action_std
    return current, target, state, action


def raw_latent(normalized: torch.Tensor, normalization: Normalization) -> torch.Tensor:
    return normalized * normalization.latent_std + normalization.latent_mean


def motion_scores(store: TransitionStore, batch_size: int = 256) -> torch.Tensor:
    scores = torch.empty(store.transition_count, dtype=torch.float32)
    for start in range(0, store.transition_count, batch_size):
        positions = torch.arange(start, min(start + batch_size, store.transition_count))
        current = store.visual_tokens[store.current_indices[positions]].float()
        target = store.visual_tokens[store.next_indices[positions]].float()
        scores[start : start + len(positions)] = (target - current).square().mean(dim=(1, 2))
    return scores


def evaluate_model(
    model: nn.Module,
    store: TransitionStore,
    normalization: Normalization,
    *,
    device: torch.device,
    batch_size: int,
    action_mode: str,
    high_motion_threshold: float,
) -> dict[str, Any]:
    model.eval()
    normalized_error_sum = 0.0
    raw_error_sum = 0.0
    persistence_normalized_sum = 0.0
    persistence_raw_sum = 0.0
    element_count = 0
    cosine_sum = 0.0
    cosine_count = 0
    high_motion_error_sum = 0.0
    high_motion_element_count = 0
    task_error_sum = {task_id: 0.0 for task_id in range(10)}
    task_element_count = {task_id: 0 for task_id in range(10)}
    scores = motion_scores(store, batch_size=batch_size)
    all_positions = torch.arange(store.transition_count)
    shuffled_all = torch.randperm(
        store.transition_count,
        generator=torch.Generator().manual_seed(COUNTERFACTUAL_SEED),
    )

    with torch.inference_mode():
        for start in range(0, store.transition_count, batch_size):
            positions = all_positions[start : start + batch_size]
            shuffled = shuffled_all[start : start + batch_size]
            current, target, state, action = normalized_batch(
                store,
                positions,
                normalization,
                device,
                action_mode=action_mode,
                shuffled_positions=shuffled,
            )
            prediction = model(current, state, action)
            normalized_squared = (prediction - target).square()
            persistence_normalized = (current - target).square()
            prediction_raw = raw_latent(prediction, normalization)
            target_raw = raw_latent(target, normalization)
            current_raw = raw_latent(current, normalization)
            raw_squared = (prediction_raw - target_raw).square()
            persistence_raw = (current_raw - target_raw).square()

            normalized_error_sum += float(normalized_squared.sum())
            raw_error_sum += float(raw_squared.sum())
            persistence_normalized_sum += float(persistence_normalized.sum())
            persistence_raw_sum += float(persistence_raw.sum())
            element_count += normalized_squared.numel()
            cosine_sum += float(
                F.cosine_similarity(
                    prediction_raw.reshape(-1, LATENT_DIMENSION),
                    target_raw.reshape(-1, LATENT_DIMENSION),
                    dim=-1,
                ).sum()
            )
            cosine_count += prediction_raw.shape[0] * prediction_raw.shape[1]

            high_mask = scores[positions] >= high_motion_threshold
            if bool(high_mask.any()):
                high_error = raw_squared[high_mask]
                high_motion_error_sum += float(high_error.sum())
                high_motion_element_count += high_error.numel()
            batch_task_ids = store.task_ids[positions]
            for task_id in range(10):
                task_mask = batch_task_ids == task_id
                if bool(task_mask.any()):
                    task_error = raw_squared[task_mask]
                    task_error_sum[task_id] += float(task_error.sum())
                    task_element_count[task_id] += task_error.numel()

    normalized_mse = normalized_error_sum / element_count
    raw_mse = raw_error_sum / element_count
    persistence_normalized_mse = persistence_normalized_sum / element_count
    persistence_raw_mse = persistence_raw_sum / element_count
    return {
        "normalized_mse": normalized_mse,
        "raw_mse": raw_mse,
        "token_cosine_similarity": cosine_sum / cosine_count,
        "persistence_normalized_mse": persistence_normalized_mse,
        "persistence_raw_mse": persistence_raw_mse,
        "improvement_vs_persistence_percent": (100 * (persistence_raw_mse - raw_mse) / persistence_raw_mse),
        "high_motion_raw_mse": high_motion_error_sum / high_motion_element_count,
        "per_task_raw_mse": {
            str(task_id): task_error_sum[task_id] / task_element_count[task_id] for task_id in range(10)
        },
    }


def save_model(
    path: Path,
    model: nn.Module,
    metadata: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, path, metadata=metadata)


def train_variant(
    variant: str,
    initial_state: dict[str, torch.Tensor],
    train_store: TransitionStore,
    validation_store: TransitionStore,
    normalization: Normalization,
    batch_schedule: torch.Tensor,
    *,
    device: torch.device,
    args: argparse.Namespace,
    high_motion_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    model = TokenDynamicsModel(args.hidden_dimension, args.depth)
    model.load_state_dict(initial_state)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    action_mode = "correct" if variant == "action_conditioned" else "zero"
    best_validation_mse = math.inf
    best_step = -1
    best_state: dict[str, torch.Tensor] | None = None
    curve_rows: list[dict[str, Any]] = []
    training_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, args.steps + 1):
        model.train()
        positions = batch_schedule[step - 1]
        current, target, state, action = normalized_batch(
            train_store,
            positions,
            normalization,
            device,
            action_mode=action_mode,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(current, state, action)
        loss = F.mse_loss(prediction.float(), target)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.gradient_clip_norm,
        )
        optimizer.step()

        if step == 1 or step % args.validation_interval == 0 or step == args.steps:
            validation = evaluate_model(
                model,
                validation_store,
                normalization,
                device=device,
                batch_size=args.eval_batch_size,
                action_mode=action_mode,
                high_motion_threshold=high_motion_threshold,
            )
            is_best = validation["normalized_mse"] < best_validation_mse
            if is_best:
                best_validation_mse = validation["normalized_mse"]
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            curve_rows.append(
                {
                    "variant": variant,
                    "step": step,
                    "train_normalized_mse": float(loss.detach()),
                    "gradient_norm": float(gradient_norm.detach()),
                    "validation_normalized_mse": validation["normalized_mse"],
                    "validation_raw_mse": validation["raw_mse"],
                    "validation_token_cosine": validation["token_cosine_similarity"],
                    "is_best": is_best,
                }
            )
            print(
                f"{variant} step={step}/{args.steps} "
                f"train={float(loss.detach()):.6f} val={validation['normalized_mse']:.6f} "
                f"best={best_validation_mse:.6f}@{best_step}",
                flush=True,
            )

    if best_state is None:
        raise AssertionError(f"No best checkpoint selected for {variant}")
    model.load_state_dict(best_state)
    final_metrics = evaluate_model(
        model,
        validation_store,
        normalization,
        device=device,
        batch_size=args.eval_batch_size,
        action_mode=action_mode,
        high_motion_threshold=high_motion_threshold,
    )
    checkpoint_path = args.output_dir / f"{variant}_best.safetensors"
    save_model(
        checkpoint_path,
        model,
        metadata={
            "schema_version": "1",
            "variant": variant,
            "best_step": str(best_step),
            "hidden_dimension": str(args.hidden_dimension),
            "depth": str(args.depth),
            "seed": str(args.seed),
        },
    )
    reloaded_model = TokenDynamicsModel(args.hidden_dimension, args.depth)
    reloaded_model.load_state_dict(load_file(checkpoint_path))
    reloaded_model.to(device)
    reload_metrics = evaluate_model(
        reloaded_model,
        validation_store,
        normalization,
        device=device,
        batch_size=args.eval_batch_size,
        action_mode=action_mode,
        high_motion_threshold=high_motion_threshold,
    )
    if canonical_json(final_metrics) != canonical_json(reload_metrics):
        raise AssertionError(f"Checkpoint reload metrics changed for {variant}")
    runtime = {
        "variant": variant,
        "training_seconds": time.perf_counter() - training_start,
        "peak_gpu_memory_bytes": (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0),
    }
    summary = {
        "variant": variant,
        "best_step": best_step,
        "best_validation_normalized_mse": best_validation_mse,
        "checkpoint": (f"outputs/wm/stage14_next_latent_baseline/{checkpoint_path.name}"),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_reload_exact_metrics": True,
        "validation": final_metrics,
    }
    model.to("cpu")
    reloaded_model.to("cpu")
    del model, reloaded_model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, curve_rows, runtime


def rounded_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            result[key] = round(value, 10)
        elif isinstance(value, dict):
            result[key] = rounded_metrics(value)
        else:
            result[key] = value
    return result


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("steps and batch sizes must be positive")
    if args.validation_interval <= 0:
        raise ValueError("--validation-interval must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    set_determinism(args.seed)
    device = torch.device(args.device)
    repo_root = Path.cwd()
    manifest_rows = read_csv(args.shard_manifest)
    if {row["split"] for row in manifest_rows} != set(SPLITS):
        raise AssertionError("Stage 14 manifest must contain only train and validation")
    stage13_report = json.loads(args.stage13_report.read_text(encoding="utf-8"))
    if stage13_report["dataset"]["test_split_cached_episodes"] != 0:
        raise AssertionError("Stage 13 report indicates test leakage")

    load_start = time.perf_counter()
    train_store = load_transition_store(manifest_rows, split="train", repo_root=repo_root)
    validation_store = load_transition_store(
        manifest_rows,
        split="validation",
        repo_root=repo_root,
    )
    loading_seconds = time.perf_counter() - load_start
    if train_store.transition_count != 42265 or validation_store.transition_count != 5168:
        raise AssertionError(
            f"Unexpected transition counts: {train_store.transition_count}/{validation_store.transition_count}"
        )
    normalization = compute_normalization(train_store)
    normalization_device = normalization.to(device)
    validation_motion_scores = motion_scores(validation_store)
    high_motion_threshold = float(torch.quantile(validation_motion_scores, 0.75))

    set_determinism(args.seed)
    template_model = TokenDynamicsModel(args.hidden_dimension, args.depth)
    initial_state = {key: value.detach().cpu().clone() for key, value in template_model.state_dict().items()}
    initial_hash = tensor_dict_sha256(initial_state)
    with torch.inference_mode():
        sample_latent = torch.randn(2, TOKENS_PER_FRAME, LATENT_DIMENSION)
        sample_state = torch.randn(2, STATE_DIMENSION)
        sample_action = torch.randn(2, ACTION_DIMENSION)
        if not torch.equal(
            template_model(sample_latent, sample_state, sample_action),
            sample_latent,
        ):
            raise AssertionError("Zero-initialized model does not equal persistence")
    parameter_count = sum(parameter.numel() for parameter in template_model.parameters())
    batch_schedule = make_batch_schedule(
        train_store.transition_count,
        args.steps,
        args.batch_size,
        args.seed + 1,
    )
    batch_schedule_hash = hashlib.sha256(batch_schedule.numpy().tobytes()).hexdigest()
    del template_model

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    summaries: dict[str, dict[str, Any]] = {}
    curve_rows: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    for variant in VARIANTS:
        summary, variant_curve, runtime = train_variant(
            variant,
            initial_state,
            train_store,
            validation_store,
            normalization_device,
            batch_schedule,
            device=device,
            args=args,
            high_motion_threshold=high_motion_threshold,
        )
        summaries[variant] = summary
        curve_rows.extend(variant_curve)
        runtimes.append(runtime)

    conditioned_model = TokenDynamicsModel(args.hidden_dimension, args.depth)
    conditioned_model.load_state_dict(load_file(args.output_dir / "action_conditioned_best.safetensors"))
    conditioned_model.to(device)
    counterfactual = {
        "correct_action": evaluate_model(
            conditioned_model,
            validation_store,
            normalization_device,
            device=device,
            batch_size=args.eval_batch_size,
            action_mode="correct",
            high_motion_threshold=high_motion_threshold,
        ),
        "zero_action": evaluate_model(
            conditioned_model,
            validation_store,
            normalization_device,
            device=device,
            batch_size=args.eval_batch_size,
            action_mode="zero",
            high_motion_threshold=high_motion_threshold,
        ),
        "shuffled_action": evaluate_model(
            conditioned_model,
            validation_store,
            normalization_device,
            device=device,
            batch_size=args.eval_batch_size,
            action_mode="shuffled",
            high_motion_threshold=high_motion_threshold,
        ),
    }
    conditioned_model.to("cpu")
    conditioned = counterfactual["correct_action"]
    no_action = summaries["no_action"]["validation"]
    comparison = {
        "conditioned_vs_no_action_raw_mse_improvement_percent": (
            100 * (no_action["raw_mse"] - conditioned["raw_mse"]) / no_action["raw_mse"]
        ),
        "correct_vs_zero_action_raw_mse_improvement_percent": (
            100
            * (counterfactual["zero_action"]["raw_mse"] - conditioned["raw_mse"])
            / counterfactual["zero_action"]["raw_mse"]
        ),
        "correct_vs_shuffled_action_raw_mse_improvement_percent": (
            100
            * (counterfactual["shuffled_action"]["raw_mse"] - conditioned["raw_mse"])
            / counterfactual["shuffled_action"]["raw_mse"]
        ),
        "conditioned_vs_no_action_high_motion_improvement_percent": (
            100
            * (no_action["high_motion_raw_mse"] - conditioned["high_motion_raw_mse"])
            / no_action["high_motion_raw_mse"]
        ),
    }
    action_signal_detected = all(
        comparison[key] > 0
        for key in (
            "conditioned_vs_no_action_raw_mse_improvement_percent",
            "correct_vs_zero_action_raw_mse_improvement_percent",
            "correct_vs_shuffled_action_raw_mse_improvement_percent",
        )
    )

    curve_text = csv_text(curve_rows)
    normalization_text = canonical_json(
        {
            "schema_version": 1,
            "source": "train frames only",
            **normalization_json(normalization),
        }
    )
    report = {
        "schema_version": 1,
        "stage": "WM Stage 14 paired next-latent baseline",
        "status": "passed",
        "data": {
            "source_shard_manifest": "results/wm/stage13_full_feature_cache/shards.csv",
            "source_shard_manifest_sha256": sha256_file(args.shard_manifest),
            "train_episodes": 346,
            "train_frames": train_store.frame_count,
            "train_transitions": train_store.transition_count,
            "validation_episodes": 43,
            "validation_frames": validation_store.frame_count,
            "validation_transitions": validation_store.transition_count,
            "test_episodes_used": 0,
            "normalization_source": "train frames only",
            "high_motion_definition": "top validation quartile by raw persistence MSE",
            "high_motion_threshold": high_motion_threshold,
        },
        "model": {
            "architecture": "token-wise residual MLP with global latent/state/action context",
            "parameter_count": parameter_count,
            "hidden_dimension": args.hidden_dimension,
            "depth": args.depth,
            "initialization_sha256": initial_hash,
            "zero_initialized_output_equals_persistence": True,
            "same_architecture_and_initialization_for_both_variants": True,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "validation_interval": args.validation_interval,
            "seed": args.seed,
            "batch_schedule_sha256": batch_schedule_hash,
            "same_batch_schedule_for_both_variants": True,
            "precision": "BF16 autocast training; FP32 validation",
            "checkpoint_selection": "lowest validation normalized MSE",
        },
        "variants": {key: rounded_metrics(value) for key, value in summaries.items()},
        "conditioned_counterfactual": {key: rounded_metrics(value) for key, value in counterfactual.items()},
        "counterfactual_action_permutation_seed": COUNTERFACTUAL_SEED,
        "paired_comparison": rounded_metrics(comparison),
        "artifacts": {
            "training_curve": "training_curve.csv",
            "training_curve_sha256": sha256_text(curve_text),
            "normalization": "normalization.json",
            "normalization_sha256": sha256_text(normalization_text),
        },
        "stage15_readiness": {
            "one_step_dynamics_trained": True,
            "test_split_remains_held_out": True,
            "multi_step_rollout_started": False,
            "action_conditioning_signal_detected": action_signal_detected,
            "multi_step_rollout_ready": action_signal_detected,
            "decision_rule": (
                "Proceed to multi-step rollout only if correct actions outperform the paired "
                "no-action model and degrade under zeroed or shuffled action counterfactuals."
            ),
        },
        "software": {
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "numpy": np.__version__,
        },
    }
    runtime_report = {
        "schema_version": 1,
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
        "loading_seconds": loading_seconds,
        "variants": runtimes,
        "peak_gpu_memory_bytes": (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0),
    }
    write_deterministic(
        args.public_results_dir / "training_curve.csv",
        curve_text,
        overwrite=args.overwrite,
    )
    write_deterministic(
        args.public_results_dir / "normalization.json",
        normalization_text,
        overwrite=args.overwrite,
    )
    write_deterministic(
        args.public_results_dir / "report.json",
        canonical_json(report),
        overwrite=args.overwrite,
    )
    write_deterministic(
        args.public_results_dir / "performance.json",
        canonical_json(runtime_report),
        overwrite=args.overwrite,
    )
    print(
        "Stage 14 passed: "
        f"conditioned_raw_mse={conditioned['raw_mse']:.8f}, "
        f"no_action_raw_mse={no_action['raw_mse']:.8f}, "
        f"improvement={comparison['conditioned_vs_no_action_raw_mse_improvement_percent']:.3f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
