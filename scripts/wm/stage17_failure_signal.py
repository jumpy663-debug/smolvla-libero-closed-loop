#!/usr/bin/env python

"""Evaluate frozen latent-dynamics signals on complete-observation rollouts.

Stage 16 trajectories are encoded with the frozen Stage 13 DINOv2 convention.
Frozen Stage 14 dynamics then score the same first 80 H1/H10 rollout starts in
every episode. Failure separation is exploratory and no model is trained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import os
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoImageProcessor, Dinov2Model

HORIZONS = (1, 10)
MAX_HORIZON = max(HORIZONS)
PREFIX_STARTS = 80
ENCODER_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 80
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 1717
FEATURE_SCHEMA_VERSION = "1"
SCORE_NAMES = (
    "h1_raw_mse",
    "h10_raw_mse",
    "h10_correct_zero_disagreement",
    "h10_correct_no_action_disagreement",
    "h10_action_advantage_vs_zero",
    "h10_persistence_raw_mse",
)


def load_stage_module(name: str) -> ModuleType:
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE12 = load_stage_module("stage12_representation_pilot")
STAGE13 = load_stage_module("stage13_full_feature_cache")
STAGE14 = load_stage_module("stage14_next_latent_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage16-episodes",
        type=Path,
        default=Path("results/wm/stage16_closed_loop_recollection/episodes.csv"),
    )
    parser.add_argument(
        "--stage16-report",
        type=Path,
        default=Path("results/wm/stage16_closed_loop_recollection/report.json"),
    )
    parser.add_argument("--model-root", type=Path, default=STAGE12.default_model_root())
    parser.add_argument(
        "--stage13-manifest",
        type=Path,
        default=Path("results/wm/stage13_full_feature_cache/shards.csv"),
    )
    parser.add_argument(
        "--stage13-report",
        type=Path,
        default=Path("results/wm/stage13_full_feature_cache/report.json"),
    )
    parser.add_argument(
        "--stage14-report",
        type=Path,
        default=Path("results/wm/stage14_next_latent_baseline/report.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/wm/stage14_next_latent_baseline"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wm/stage17_failure_signal"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage17_failure_signal"),
    )
    parser.add_argument("--encoder-batch-size", type=int, default=ENCODER_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite-public", action="store_true")
    return parser.parse_args()


def canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def csv_text(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_deterministic(
    path: Path,
    content: str,
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-identical output: {path}")
    path.write_text(content, encoding="utf-8")


def quat_to_axisangle(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"Expected quaternion [N,4], got {tuple(quaternion.shape)}")
    quaternion = quaternion.float()
    w = quaternion[:, 3].clamp(-1.0, 1.0)
    denominator = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    result = torch.zeros((len(quaternion), 3), dtype=torch.float32)
    mask = denominator > 1e-10
    if bool(mask.any()):
        angle = 2.0 * torch.acos(w[mask])
        axis = quaternion[mask, :3] / denominator[mask, None]
        result[mask] = axis * angle[:, None]
    return result


def policy_state(arrays: dict[str, np.ndarray]) -> torch.Tensor:
    eef_position = torch.from_numpy(arrays["eef_pos"]).float()
    eef_quaternion = torch.from_numpy(arrays["eef_quat"]).float()
    gripper_position = torch.from_numpy(arrays["gripper_qpos"]).float()
    state = torch.cat(
        [eef_position, quat_to_axisangle(eef_quaternion), gripper_position],
        dim=-1,
    )
    if state.shape != (len(eef_position), STAGE14.STATE_DIMENSION):
        raise AssertionError(f"Unexpected policy state shape: {tuple(state.shape)}")
    return state


def policy_images(arrays: dict[str, np.ndarray]) -> torch.Tensor:
    cameras = []
    for key in ("camera1", "camera2"):
        value = torch.from_numpy(arrays[key])
        if value.dtype != torch.uint8 or value.ndim != 4 or value.shape[-1] != 3:
            raise AssertionError(f"Unexpected image array {key}: {tuple(value.shape)} {value.dtype}")
        value = value.permute(0, 3, 1, 2).contiguous()
        # Exact LiberoProcessorStep camera convention used by the policy and
        # matching the orientation of the Stage 13 expert videos.
        cameras.append(torch.flip(value, dims=(-2, -1)))
    images = torch.stack(cameras, dim=1)
    if images.shape[1:] != (2, 3, 360, 360):
        raise AssertionError(f"Unexpected stacked image shape: {tuple(images.shape)}")
    return images


def source_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays


def feature_path(output_dir: Path, row: dict[str, str]) -> Path:
    return output_dir / "features" / f"{row['rollout_id']}.safetensors"


def expected_feature_metadata(
    row: dict[str, str],
    pixel_hash: str | None = None,
    *,
    encoder_batch_size: int = ENCODER_BATCH_SIZE,
) -> dict[str, str]:
    metadata = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "rollout_id": row["rollout_id"],
        "benchmark_task_id": str(int(row["benchmark_task_id"])),
        "episode_index": str(int(row["episode_index"])),
        "success": str(row["success"] == "True").lower(),
        "steps": str(int(row["steps"])),
        "source_content_sha256": row["content_sha256"],
        "source_file_sha256": row["file_sha256"],
        "encoder_model_id": STAGE12.MODEL_ID,
        "encoder_revision": STAGE12.MODEL_REVISION,
        "encoder_batch_size": str(encoder_batch_size),
        "image_transform": "flip height and width before DINOv2",
        "state_layout": "eef_pos(3),eef_axisangle(3),gripper_qpos(2)",
    }
    if pixel_hash is not None:
        metadata["policy_pixels_sha256"] = pixel_hash
    return metadata


def atomic_save_feature(
    path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    save_file(tensors, temporary, metadata=metadata)
    os.replace(temporary, path)


def validate_feature(
    path: Path,
    row: dict[str, str],
    *,
    encoder_batch_size: int = ENCODER_BATCH_SIZE,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors = load_file(path)
    expected_keys = {
        "visual_tokens",
        "observation_state",
        "action",
        "transition_valid",
    }
    if set(tensors) != expected_keys:
        raise AssertionError(f"Unexpected feature keys in {path}: {set(tensors)}")
    steps = int(row["steps"])
    expected_shapes = {
        "visual_tokens": (steps + 1, 2, 16, 384),
        "observation_state": (steps + 1, 8),
        "action": (steps, 7),
        "transition_valid": (steps,),
    }
    for key, shape in expected_shapes.items():
        if tuple(tensors[key].shape) != shape:
            raise AssertionError(f"{path} {key} shape {tuple(tensors[key].shape)} != {shape}")
    if tensors["visual_tokens"].dtype != torch.float16:
        raise AssertionError("Visual tokens must be float16")
    if tensors["observation_state"].dtype != torch.float32:
        raise AssertionError("Observation state must be float32")
    if tensors["action"].dtype != torch.float32:
        raise AssertionError("Action must be float32")
    if tensors["transition_valid"].dtype != torch.bool:
        raise AssertionError("Transition validity must be bool")
    for key in ("visual_tokens", "observation_state", "action"):
        if not bool(torch.isfinite(tensors[key]).all()):
            raise AssertionError(f"Non-finite feature tensor: {key}")
    if int(tensors["transition_valid"].sum()) != int(row["valid_transitions"]):
        raise AssertionError("Valid-transition count differs from Stage 16")
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
    if metadata is None:
        raise AssertionError(f"Missing metadata in {path}")
    for key, value in expected_feature_metadata(
        row,
        encoder_batch_size=encoder_batch_size,
    ).items():
        if metadata.get(key) != value:
            raise AssertionError(f"Feature metadata mismatch: {key}")
    if len(metadata.get("policy_pixels_sha256", "")) != 64:
        raise AssertionError("Missing policy pixel digest")
    return tensors, metadata


def encode_feature_shard(
    row: dict[str, str],
    *,
    repo_root: Path,
    output_dir: Path,
    processor: Any,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, str], bool]:
    path = feature_path(output_dir, row)
    if path.exists():
        tensors, metadata = validate_feature(
            path,
            row,
            encoder_batch_size=batch_size,
        )
        return tensors, metadata, True
    source_path = repo_root / row["local_artifact"]
    if sha256_file(source_path) != row["file_sha256"]:
        raise AssertionError(f"Stage 16 source hash mismatch: {source_path}")
    arrays = source_arrays(source_path)
    images = policy_images(arrays)
    pixel_hash = sha256_bytes(images.numpy().tobytes())
    state = policy_state(arrays)
    action = torch.from_numpy(arrays["action"]).float()
    transition_valid = torch.from_numpy(arrays["transition_valid"]).bool()
    visual_tokens = STAGE13.encode_images(
        images,
        processor,
        encoder,
        device=device,
        batch_size=batch_size,
    )
    tensors = {
        "visual_tokens": visual_tokens,
        "observation_state": state,
        "action": action,
        "transition_valid": transition_valid,
    }
    metadata = expected_feature_metadata(
        row,
        pixel_hash,
        encoder_batch_size=batch_size,
    )
    atomic_save_feature(path, tensors, metadata)
    tensors, metadata = validate_feature(
        path,
        row,
        encoder_batch_size=batch_size,
    )
    return tensors, metadata, False


def fixed_prefix_starts(
    transition_valid: torch.Tensor,
    *,
    starts: int = PREFIX_STARTS,
    horizon: int = MAX_HORIZON,
) -> torch.Tensor:
    if transition_valid.ndim != 1:
        raise ValueError("transition_valid must be one-dimensional")
    if starts <= 0 or horizon <= 0:
        raise ValueError("starts and horizon must be positive")
    required = starts + horizon - 1
    if len(transition_valid) < required:
        raise ValueError(f"Need at least {required} transitions, found {len(transition_valid)}")
    selected = torch.arange(starts, dtype=torch.int64)
    for start in selected.tolist():
        if not bool(transition_valid[start : start + horizon].all()):
            raise AssertionError(f"Invalid transition inside fixed prefix at start {start}")
    return selected


def normalization_from_training(
    manifest_path: Path,
    *,
    repo_root: Path,
) -> Any:
    rows = read_csv(manifest_path)
    if {row["split"] for row in rows} != {"train", "validation"}:
        raise AssertionError("Stage 13 manifest split set changed")
    train = STAGE14.load_transition_store(rows, split="train", repo_root=repo_root)
    normalization = STAGE14.compute_normalization(train)
    del train
    return normalization


def load_dynamics(
    stage14_report: dict[str, Any],
    checkpoint_dir: Path,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    hidden_dimension = int(stage14_report["model"]["hidden_dimension"])
    depth = int(stage14_report["model"]["depth"])
    models = {}
    for variant in STAGE14.VARIANTS:
        checkpoint = checkpoint_dir / f"{variant}_best.safetensors"
        if sha256_file(checkpoint) != stage14_report["variants"][variant]["checkpoint_sha256"]:
            raise AssertionError(f"Checkpoint hash mismatch: {variant}")
        model = STAGE14.TokenDynamicsModel(hidden_dimension, depth)
        model.load_state_dict(load_file(checkpoint))
        model.eval()
        model.requires_grad_(False)
        models[variant] = model.to(device)
    return models


def raw_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalization: Any,
) -> torch.Tensor:
    prediction_raw = STAGE14.raw_latent(prediction, normalization)
    target_raw = STAGE14.raw_latent(target, normalization)
    return (prediction_raw - target_raw).square().mean(dim=(1, 2))


def raw_disagreement(
    first: torch.Tensor,
    second: torch.Tensor,
    normalization: Any,
) -> torch.Tensor:
    first_raw = STAGE14.raw_latent(first, normalization)
    second_raw = STAGE14.raw_latent(second, normalization)
    return (first_raw - second_raw).square().mean(dim=(1, 2))


def score_episode_windows(
    tensors: dict[str, torch.Tensor],
    models: dict[str, torch.nn.Module],
    normalization: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    starts = fixed_prefix_starts(tensors["transition_valid"])
    results: dict[str, list[np.ndarray]] = {
        "h1_raw_mse": [],
        "h10_raw_mse": [],
        "h10_correct_zero_disagreement": [],
        "h10_correct_no_action_disagreement": [],
        "h10_action_advantage_vs_zero": [],
        "h10_persistence_raw_mse": [],
    }
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            initial_raw = tensors["visual_tokens"][batch_starts].reshape(
                -1,
                STAGE14.TOKENS_PER_FRAME,
                STAGE14.LATENT_DIMENSION,
            )
            initial_raw = initial_raw.to(device=device, dtype=torch.float32)
            initial = (initial_raw - normalization.latent_mean) / normalization.latent_std
            correct_prediction = initial.clone()
            zero_prediction = initial.clone()
            no_action_prediction = initial.clone()

            for step in range(1, MAX_HORIZON + 1):
                frame_indices = batch_starts + step - 1
                target_indices = batch_starts + step
                state = tensors["observation_state"][frame_indices].to(device)
                state = (state - normalization.state_mean) / normalization.state_std
                action = tensors["action"][frame_indices].to(device)
                action = (action - normalization.action_mean) / normalization.action_std
                zero = torch.zeros_like(action)
                correct_prediction = models["action_conditioned"](
                    correct_prediction,
                    state,
                    action,
                )
                zero_prediction = models["action_conditioned"](
                    zero_prediction,
                    state,
                    zero,
                )
                no_action_prediction = models["no_action"](
                    no_action_prediction,
                    state,
                    zero,
                )
                if step in HORIZONS:
                    target_raw = tensors["visual_tokens"][target_indices].reshape(
                        -1,
                        STAGE14.TOKENS_PER_FRAME,
                        STAGE14.LATENT_DIMENSION,
                    )
                    target_raw = target_raw.to(device=device, dtype=torch.float32)
                    target = (target_raw - normalization.latent_mean) / normalization.latent_std
                    correct_error = raw_mse(correct_prediction, target, normalization)
                    if step == 1:
                        results["h1_raw_mse"].append(correct_error.cpu().numpy())
                    else:
                        zero_error = raw_mse(zero_prediction, target, normalization)
                        results["h10_raw_mse"].append(correct_error.cpu().numpy())
                        results["h10_correct_zero_disagreement"].append(
                            raw_disagreement(
                                correct_prediction,
                                zero_prediction,
                                normalization,
                            )
                            .cpu()
                            .numpy()
                        )
                        results["h10_correct_no_action_disagreement"].append(
                            raw_disagreement(
                                correct_prediction,
                                no_action_prediction,
                                normalization,
                            )
                            .cpu()
                            .numpy()
                        )
                        results["h10_action_advantage_vs_zero"].append((zero_error - correct_error).cpu().numpy())
                        results["h10_persistence_raw_mse"].append(raw_mse(initial, target, normalization).cpu().numpy())
    concatenated = {key: np.concatenate(value) for key, value in results.items()}
    if any(len(value) != PREFIX_STARTS for value in concatenated.values()):
        raise AssertionError("Episode window score count differs from fixed prefix")
    return concatenated


def roc_auc(scores: np.ndarray, failures: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    failures = np.asarray(failures, dtype=np.bool_)
    positive = scores[failures]
    negative = scores[~failures]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("AUROC requires both labels")
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size)


def hedges_g(scores: np.ndarray, failures: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    failures = np.asarray(failures, dtype=np.bool_)
    positive = scores[failures]
    negative = scores[~failures]
    degrees = len(positive) + len(negative) - 2
    pooled_variance = (
        (len(positive) - 1) * positive.var(ddof=1) + (len(negative) - 1) * negative.var(ddof=1)
    ) / degrees
    if pooled_variance <= 0:
        return 0.0
    correction = 1.0 - 3.0 / (4.0 * (len(positive) + len(negative)) - 9.0)
    return float(correction * (positive.mean() - negative.mean()) / np.sqrt(pooled_variance))


def exact_permutation(
    scores: np.ndarray,
    failures: np.ndarray,
) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=np.float64)
    failures = np.asarray(failures, dtype=np.bool_)
    failure_count = int(failures.sum())
    observed = float(scores[failures].mean() - scores[~failures].mean())
    differences = []
    indices = np.arange(len(scores))
    for selected in itertools.combinations(indices.tolist(), failure_count):
        mask = np.zeros(len(scores), dtype=np.bool_)
        mask[list(selected)] = True
        differences.append(float(scores[mask].mean() - scores[~mask].mean()))
    values = np.asarray(differences)
    p_value = float(np.mean(np.abs(values) >= abs(observed) - 1e-15))
    return {
        "failure_minus_success_mean": observed,
        "two_sided_p_value": p_value,
        "label_assignments": len(values),
    }


def stratified_auc_bootstrap(
    scores: np.ndarray,
    failures: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    scores = np.asarray(scores, dtype=np.float64)
    failures = np.asarray(failures, dtype=np.bool_)
    positive = scores[failures]
    negative = scores[~failures]
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        positive_sample = positive[rng.integers(0, len(positive), len(positive))]
        negative_sample = negative[rng.integers(0, len(negative), len(negative))]
        joined = np.concatenate([positive_sample, negative_sample])
        labels = np.concatenate(
            [
                np.ones(len(positive_sample), dtype=np.bool_),
                np.zeros(len(negative_sample), dtype=np.bool_),
            ]
        )
        values[index] = roc_auc(joined, labels)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def task_centered(scores: np.ndarray, task_ids: np.ndarray) -> np.ndarray:
    centered = np.asarray(scores, dtype=np.float64).copy()
    task_ids = np.asarray(task_ids)
    for task_id in np.unique(task_ids):
        mask = task_ids == task_id
        centered[mask] -= centered[mask].mean()
    return centered


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    if not p_values:
        raise ValueError("p_values must not be empty")
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (total - rank) * float(p_values[name]))
        running_maximum = max(running_maximum, candidate)
        adjusted[name] = running_maximum
    return {name: adjusted[name] for name in p_values}


def score_statistics(
    scores: np.ndarray,
    failures: np.ndarray,
    task_ids: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    failures = np.asarray(failures, dtype=np.bool_)
    task_ids = np.asarray(task_ids, dtype=np.int64)
    mixed_mask = np.isin(task_ids, [4, 8])
    permutation = exact_permutation(scores, failures)
    centered = task_centered(scores, task_ids)
    mixed_permutation = exact_permutation(scores[mixed_mask], failures[mixed_mask])
    return {
        "success_mean": float(scores[~failures].mean()),
        "failure_mean": float(scores[failures].mean()),
        "failure_minus_success_mean": permutation["failure_minus_success_mean"],
        "failure_over_success_ratio": float(scores[failures].mean() / scores[~failures].mean()),
        "higher_score_failure_auroc": roc_auc(scores, failures),
        "auroc_stratified_bootstrap_ci95": stratified_auc_bootstrap(
            scores,
            failures,
            samples=BOOTSTRAP_SAMPLES,
            seed=seed,
        ),
        "hedges_g_failure_minus_success": hedges_g(scores, failures),
        "exact_permutation_two_sided_p": permutation["two_sided_p_value"],
        "exact_permutation_assignments": permutation["label_assignments"],
        "task_centered_auroc": roc_auc(centered, failures),
        "mixed_tasks_4_8": {
            "episodes": int(mixed_mask.sum()),
            "successes": int((~failures[mixed_mask]).sum()),
            "failures": int(failures[mixed_mask].sum()),
            "higher_score_failure_auroc": roc_auc(scores[mixed_mask], failures[mixed_mask]),
            "failure_minus_success_mean": mixed_permutation["failure_minus_success_mean"],
            "exact_permutation_two_sided_p": mixed_permutation["two_sided_p_value"],
            "exact_permutation_assignments": mixed_permutation["label_assignments"],
        },
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.encoder_batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.encoder_batch_size != ENCODER_BATCH_SIZE:
        raise ValueError(f"Stage 17 encoder batch size is pinned to {ENCODER_BATCH_SIZE}")
    if args.model_root.name != STAGE12.MODEL_REVISION:
        raise ValueError("DINOv2 model revision is not pinned")
    repo_root = Path.cwd()
    stage16_report = json.loads(args.stage16_report.read_text(encoding="utf-8"))
    if stage16_report["status"] != "passed" or stage16_report["scope"]["episodes"] != 12:
        raise AssertionError("Stage 16 formal report is not ready")
    if stage16_report["scope"]["test_demonstration_episodes_used"] != 0:
        raise AssertionError("Stage 16 indicates test leakage")
    episode_rows = read_csv(args.stage16_episodes)
    if len(episode_rows) != 12:
        raise AssertionError("Stage 16 episode manifest must contain 12 rows")
    if sum(row["success"] == "True" for row in episode_rows) != 7:
        raise AssertionError("Stage 16 label balance changed")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(args.model_root, local_files_only=True)
    encoder = Dinov2Model.from_pretrained(args.model_root, local_files_only=True).eval()
    encoder.requires_grad_(False)
    encoder.to(device)
    encoder_loaded = time.perf_counter()

    feature_rows: list[dict[str, Any]] = []
    resumed = 0
    encoded = 0
    for position, row in enumerate(episode_rows, start=1):
        tensors, metadata, was_resumed = encode_feature_shard(
            row,
            repo_root=repo_root,
            output_dir=args.output_dir,
            processor=processor,
            encoder=encoder,
            device=device,
            batch_size=args.encoder_batch_size,
        )
        path = feature_path(args.output_dir, row)
        resumed += int(was_resumed)
        encoded += int(not was_resumed)
        feature_rows.append(
            {
                "rollout_id": row["rollout_id"],
                "benchmark_task_id": int(row["benchmark_task_id"]),
                "episode_index": int(row["episode_index"]),
                "success": row["success"] == "True",
                "steps": int(row["steps"]),
                "valid_transitions": int(tensors["transition_valid"].sum()),
                "source_content_sha256": row["content_sha256"],
                "policy_pixels_sha256": metadata["policy_pixels_sha256"],
                "feature_file": (f"outputs/wm/stage17_failure_signal/features/{path.name}"),
                "feature_bytes": path.stat().st_size,
                "feature_sha256": sha256_file(path),
            }
        )
        print(
            f"[{position}/12] {row['rollout_id']} encoded={encoded} resumed={resumed}",
            flush=True,
        )

    # Bit-exact recheck on the shortest episode.
    recheck_row = min(episode_rows, key=lambda row: int(row["steps"]))
    cached, cached_metadata = validate_feature(
        feature_path(args.output_dir, recheck_row),
        recheck_row,
        encoder_batch_size=args.encoder_batch_size,
    )
    arrays = source_arrays(repo_root / recheck_row["local_artifact"])
    images = policy_images(arrays)
    recheck_pixels = sha256_bytes(images.numpy().tobytes())
    recheck_tokens = STAGE13.encode_images(
        images,
        processor,
        encoder,
        device=device,
        batch_size=args.encoder_batch_size,
    )
    if recheck_pixels != cached_metadata["policy_pixels_sha256"]:
        raise AssertionError("Deterministic recheck pixel hash differs")
    if not torch.equal(recheck_tokens, cached["visual_tokens"]):
        raise AssertionError("Deterministic recheck visual tokens differ")
    encoding_finished = time.perf_counter()
    encoder.to("cpu")
    del encoder, processor, recheck_tokens, images, arrays, cached
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stage13_report = json.loads(args.stage13_report.read_text(encoding="utf-8"))
    if stage13_report["dataset"]["test_split_cached_episodes"] != 0:
        raise AssertionError("Stage 13 report indicates test leakage")
    normalization = normalization_from_training(
        args.stage13_manifest,
        repo_root=repo_root,
    ).to(device)
    stage14_report = json.loads(args.stage14_report.read_text(encoding="utf-8"))
    models = load_dynamics(stage14_report, args.checkpoint_dir, device)

    episode_score_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    episode_window_values: dict[str, dict[str, np.ndarray]] = {}
    for row in episode_rows:
        tensors, _ = validate_feature(feature_path(args.output_dir, row), row)
        scores = score_episode_windows(
            tensors,
            models,
            normalization,
            device=device,
            batch_size=args.eval_batch_size,
        )
        episode_window_values[row["rollout_id"]] = scores
        episode_record = {
            "rollout_id": row["rollout_id"],
            "benchmark_task_id": int(row["benchmark_task_id"]),
            "episode_index": int(row["episode_index"]),
            "success": row["success"] == "True",
            "failure": row["success"] != "True",
            "steps": int(row["steps"]),
            "fixed_prefix_starts": PREFIX_STARTS,
            **{name: float(scores[name].mean()) for name in SCORE_NAMES},
        }
        episode_score_rows.append(episode_record)
        for start_index in range(PREFIX_STARTS):
            window_rows.append(
                {
                    "rollout_id": row["rollout_id"],
                    "benchmark_task_id": int(row["benchmark_task_id"]),
                    "episode_index": int(row["episode_index"]),
                    "success": row["success"] == "True",
                    "failure": row["success"] != "True",
                    "start_index": start_index,
                    **{name: float(scores[name][start_index]) for name in SCORE_NAMES},
                }
            )

    failures = np.array([row["failure"] for row in episode_score_rows], dtype=np.bool_)
    task_ids = np.array(
        [row["benchmark_task_id"] for row in episode_score_rows],
        dtype=np.int64,
    )
    statistics = {
        name: score_statistics(
            np.array([row[name] for row in episode_score_rows]),
            failures,
            task_ids,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, name in enumerate(SCORE_NAMES)
    }
    steps = np.array([row["steps"] for row in episode_score_rows], dtype=np.float64)
    statistics["episode_steps_confound_reference"] = score_statistics(
        steps,
        failures,
        task_ids,
        seed=BOOTSTRAP_SEED + len(SCORE_NAMES),
    )
    secondary_names = [name for name in SCORE_NAMES if name != "h10_raw_mse"]
    secondary_raw_p = {name: float(statistics[name]["exact_permutation_two_sided_p"]) for name in secondary_names}
    multiple_testing = {
        "method": "Holm family-wise error correction",
        "family": secondary_names,
        "raw_p_values": secondary_raw_p,
        "adjusted_p_values": holm_adjust(secondary_raw_p),
        "secondary_scores_below_0_05_after_correction": [],
    }
    multiple_testing["secondary_scores_below_0_05_after_correction"] = [
        name for name, value in multiple_testing["adjusted_p_values"].items() if value < 0.05
    ]

    time_bin_rows = []
    for bin_index in range(4):
        begin = bin_index * 20
        end = begin + 20
        for label_name, label_value in (("success", False), ("failure", True)):
            values = []
            for episode_row in episode_score_rows:
                if bool(episode_row["failure"]) == label_value:
                    values.extend(episode_window_values[episode_row["rollout_id"]]["h10_raw_mse"][begin:end])
            time_bin_rows.append(
                {
                    "start_begin": begin,
                    "start_end_exclusive": end,
                    "label": label_name,
                    "episodes": int((failures == label_value).sum()),
                    "windows": len(values),
                    "h10_raw_mse_mean": float(np.mean(values)),
                    "h10_raw_mse_median": float(np.median(values)),
                }
            )

    episode_text = csv_text(episode_score_rows)
    window_text = csv_text(window_rows)
    time_bin_text = csv_text(time_bin_rows)
    statistics_text = canonical_json(
        {
            "schema_version": 1,
            "failure_is_positive_label": True,
            "higher_score_is_prespecified_failure_direction": True,
            "primary_score": "h10_raw_mse",
            "scores": statistics,
            "multiple_testing": multiple_testing,
        }
    )
    feature_text = csv_text(feature_rows)
    primary = statistics["h10_raw_mse"]
    primary_viable = (
        primary["higher_score_failure_auroc"] > 0.5
        and primary["mixed_tasks_4_8"]["higher_score_failure_auroc"] > 0.5
        and primary["failure_minus_success_mean"] > 0
    )
    report = {
        "schema_version": 1,
        "stage": "WM Stage 17 exploratory closed-loop failure signal",
        "status": "passed",
        "scope": {
            "model_parameters_updated": False,
            "episodes": 12,
            "successes": 7,
            "failures": 5,
            "task_ids": [4, 5, 7, 8],
            "fixed_prefix_starts_per_episode": PREFIX_STARTS,
            "windows": 12 * PREFIX_STARTS,
            "horizons": list(HORIZONS),
            "state_context": "oracle ground-truth state at every rollout step",
            "expert_test_episodes_used": 0,
            "failure_classifier_trained": False,
        },
        "anti_confound_protocol": {
            "same_prefix_length_for_every_episode": True,
            "shortest_episode_valid_transitions": min(int(row["valid_transitions"]) for row in episode_rows),
            "required_valid_transitions": PREFIX_STARTS + MAX_HORIZON - 1,
            "mixed_label_tasks": [4, 8],
            "single_label_tasks": {"5": "failure only", "7": "success only"},
            "episode_steps_reported_as_confound_reference": True,
        },
        "encoder": {
            "model_id": STAGE12.MODEL_ID,
            "revision": STAGE12.MODEL_REVISION,
            "camera_transform": "180-degree flip matching LiberoProcessorStep and Stage 13 videos",
            "tokens_per_camera": STAGE12.TOKENS_PER_CAMERA,
            "latent_dimension": STAGE12.LATENT_DIMENSION,
            "batch_size": ENCODER_BATCH_SIZE,
            "feature_shards": 12,
            "deterministic_recheck_rollout_id": recheck_row["rollout_id"],
            "deterministic_recheck_bit_exact": True,
        },
        "dynamics": {
            "source_stage14_report_sha256": sha256_file(args.stage14_report),
            "action_conditioned_checkpoint_sha256": stage14_report["variants"]["action_conditioned"][
                "checkpoint_sha256"
            ],
            "no_action_checkpoint_sha256": stage14_report["variants"]["no_action"]["checkpoint_sha256"],
            "normalization_recomputed_from_stage13_train_only": True,
        },
        "primary_result": {
            "score": "h10_raw_mse",
            **primary,
            "passes_exploratory_viability_rule": primary_viable,
        },
        "artifacts": {
            "feature_manifest": "feature_manifest.csv",
            "feature_manifest_sha256": sha256_bytes(feature_text.encode()),
            "episode_scores": "episode_scores.csv",
            "episode_scores_sha256": sha256_bytes(episode_text.encode()),
            "window_scores": "window_scores.csv",
            "window_scores_sha256": sha256_bytes(window_text.encode()),
            "time_bins": "time_bins.csv",
            "time_bins_sha256": sha256_bytes(time_bin_text.encode()),
            "statistics": "statistics.json",
            "statistics_sha256": sha256_bytes(statistics_text.encode()),
        },
        "stage18_readiness": {
            "closed_loop_scores_computed": True,
            "primary_signal_exploratorily_viable": primary_viable,
            "calibrated_failure_detector_ready": False,
        },
        "limitations": [
            "Only 12 episodes and four tasks are available.",
            "Task 5 contains only failures and Task 7 only successes.",
            "The same exploratory cohort defines and evaluates all secondary scores.",
            "Overlapping rollout windows are summarized at episode level, not treated as independent samples.",
            "Oracle proprioceptive state is supplied at every recursive step.",
            "No result is a held-out or calibrated failure detector.",
        ],
    }
    write_deterministic(
        args.public_results_dir / "feature_manifest.csv",
        feature_text,
        overwrite=args.overwrite_public,
    )
    write_deterministic(
        args.public_results_dir / "episode_scores.csv",
        episode_text,
        overwrite=args.overwrite_public,
    )
    write_deterministic(
        args.public_results_dir / "window_scores.csv",
        window_text,
        overwrite=args.overwrite_public,
    )
    write_deterministic(
        args.public_results_dir / "time_bins.csv",
        time_bin_text,
        overwrite=args.overwrite_public,
    )
    write_deterministic(
        args.public_results_dir / "statistics.json",
        statistics_text,
        overwrite=args.overwrite_public,
    )
    write_deterministic(
        args.public_results_dir / "report.json",
        canonical_json(report),
        overwrite=args.overwrite_public,
    )
    runtime = {
        "schema_version": 1,
        "encoder_load_seconds": encoder_loaded - started,
        "encoding_and_recheck_seconds": encoding_finished - encoder_loaded,
        "total_seconds": time.perf_counter() - started,
        "peak_pytorch_gpu_allocation_bytes": (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0),
        "new_feature_shards": encoded,
        "resumed_feature_shards": resumed,
    }
    (args.output_dir / "runtime.json").write_text(
        canonical_json(runtime),
        encoding="utf-8",
    )
    print(
        "Stage 17 passed: "
        f"H10 AUROC={primary['higher_score_failure_auroc']:.3f}, "
        f"mixed-task AUROC={primary['mixed_tasks_4_8']['higher_score_failure_auroc']:.3f}, "
        f"viable={primary_viable}"
    )


if __name__ == "__main__":
    main()
