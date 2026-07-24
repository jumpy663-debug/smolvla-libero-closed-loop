#!/usr/bin/env python

"""Score Stage 20 paired interventions with the frozen Stage 14 world model.

The protocol is fixed before scoring: H10 starts 10--39 are pre-intervention,
starts 50--79 are post-intervention, and the episode pair is the statistical
unit. No encoder, dynamics, threshold, or classifier parameter is updated.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from stage17_failure_signal import (
    STAGE12,
    STAGE13,
    STAGE14,
    canonical_json,
    csv_text,
    load_dynamics,
    normalization_from_training,
    policy_images,
    policy_state,
    raw_mse,
    read_csv,
    sha256_bytes,
    sha256_file,
    source_arrays,
    write_deterministic,
)
from transformers import AutoImageProcessor, Dinov2Model

HORIZON = 10
PRE_STARTS = tuple(range(10, 40))
POST_STARTS = tuple(range(50, 80))
ENCODER_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 60
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2121
FEATURE_SCHEMA_VERSION = "1"
SCORE_NAMES = (
    "commanded_h10_raw_mse",
    "executed_h10_raw_mse",
    "no_action_h10_raw_mse",
    "persistence_h10_raw_mse",
)
CONDITIONS = ("nominal", "persistent_motion_dropout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage20-episodes",
        type=Path,
        default=Path("results/wm/stage20_paired_interventions/episodes.csv"),
    )
    parser.add_argument(
        "--stage20-pairs",
        type=Path,
        default=Path("results/wm/stage20_paired_interventions/pairs.csv"),
    )
    parser.add_argument(
        "--stage20-report",
        type=Path,
        default=Path("results/wm/stage20_paired_interventions/report.json"),
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
        default=Path("outputs/wm/stage21_paired_wm_scoring"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage21_paired_wm_scoring"),
    )
    parser.add_argument("--encoder-batch-size", type=int, default=ENCODER_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite-public", action="store_true")
    return parser.parse_args()


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
        "pair_id": row["pair_id"],
        "condition": row["condition"],
        "benchmark_task_id": str(int(row["benchmark_task_id"])),
        "init_state_index": str(int(row["init_state_index"])),
        "steps": str(int(row["steps"])),
        "source_content_sha256": row["content_sha256"],
        "source_archive_sha256": row["archive_sha256"],
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


def validate_source_arrays(
    arrays: dict[str, np.ndarray],
    row: dict[str, str],
) -> None:
    steps = int(row["steps"])
    observations = steps + 1
    expected_shapes = {
        "camera1": (observations, 360, 360, 3),
        "camera2": (observations, 360, 360, 3),
        "eef_pos": (observations, 3),
        "eef_quat": (observations, 4),
        "gripper_qpos": (observations, 2),
        "commanded_action": (steps, STAGE14.ACTION_DIMENSION),
        "executed_action": (steps, STAGE14.ACTION_DIMENSION),
        "transition_valid": (steps,),
        "intervention_mask": (steps,),
    }
    for key, shape in expected_shapes.items():
        if key not in arrays or arrays[key].shape != shape:
            actual = None if key not in arrays else arrays[key].shape
            raise AssertionError(f"Unexpected Stage 20 array {key}: {actual} != {shape}")
    if int(arrays["transition_valid"].sum()) != int(row["valid_transitions"]):
        raise AssertionError("Stage 20 valid-transition count changed")
    intervention = arrays["intervention_mask"].astype(bool)
    if row["condition"] == "nominal":
        if bool(intervention.any()):
            raise AssertionError("Nominal episode contains intervention steps")
        if not np.array_equal(arrays["commanded_action"], arrays["executed_action"]):
            raise AssertionError("Nominal commanded and executed actions differ")
    else:
        expected = np.arange(steps) >= 50
        if not np.array_equal(intervention, expected):
            raise AssertionError("Fault intervention mask differs from fixed step 50")
        if not np.array_equal(
            arrays["commanded_action"][:50],
            arrays["executed_action"][:50],
        ):
            raise AssertionError("Fault actions differ before intervention")


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
        "commanded_action",
        "executed_action",
        "transition_valid",
        "intervention_mask",
    }
    if set(tensors) != expected_keys:
        raise AssertionError(f"Unexpected feature keys in {path}: {set(tensors)}")
    steps = int(row["steps"])
    expected_shapes = {
        "visual_tokens": (steps + 1, 2, 16, STAGE14.LATENT_DIMENSION),
        "observation_state": (steps + 1, STAGE14.STATE_DIMENSION),
        "commanded_action": (steps, STAGE14.ACTION_DIMENSION),
        "executed_action": (steps, STAGE14.ACTION_DIMENSION),
        "transition_valid": (steps,),
        "intervention_mask": (steps,),
    }
    for key, shape in expected_shapes.items():
        if tuple(tensors[key].shape) != shape:
            raise AssertionError(f"{path} {key} shape {tuple(tensors[key].shape)} != {shape}")
    expected_dtypes = {
        "visual_tokens": torch.float16,
        "observation_state": torch.float32,
        "commanded_action": torch.float32,
        "executed_action": torch.float32,
        "transition_valid": torch.bool,
        "intervention_mask": torch.bool,
    }
    for key, dtype in expected_dtypes.items():
        if tensors[key].dtype != dtype:
            raise AssertionError(f"Unexpected {key} dtype: {tensors[key].dtype}")
    for key in ("visual_tokens", "observation_state", "commanded_action", "executed_action"):
        if not bool(torch.isfinite(tensors[key]).all()):
            raise AssertionError(f"Non-finite feature tensor: {key}")
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
    if sha256_file(source_path) != row["archive_sha256"]:
        raise AssertionError(f"Stage 20 source hash mismatch: {source_path}")
    arrays = source_arrays(source_path)
    validate_source_arrays(arrays, row)
    images = policy_images(arrays)
    pixel_hash = sha256_bytes(images.numpy().tobytes())
    tensors = {
        "visual_tokens": STAGE13.encode_images(
            images,
            processor,
            encoder,
            device=device,
            batch_size=batch_size,
        ),
        "observation_state": policy_state(arrays),
        "commanded_action": torch.from_numpy(arrays["commanded_action"]).float(),
        "executed_action": torch.from_numpy(arrays["executed_action"]).float(),
        "transition_valid": torch.from_numpy(arrays["transition_valid"]).bool(),
        "intervention_mask": torch.from_numpy(arrays["intervention_mask"]).bool(),
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


def validate_window_starts(
    transition_valid: torch.Tensor,
    starts: tuple[int, ...],
    *,
    horizon: int = HORIZON,
) -> torch.Tensor:
    if transition_valid.ndim != 1:
        raise ValueError("transition_valid must be one-dimensional")
    selected = torch.tensor(starts, dtype=torch.int64)
    for start in selected.tolist():
        if start < 0 or start + horizon > len(transition_valid):
            raise ValueError(f"H{horizon} start {start} is outside the episode")
        if not bool(transition_valid[start : start + horizon].all()):
            raise AssertionError(f"Invalid transition in H{horizon} window at {start}")
    return selected


def score_episode_windows(
    tensors: dict[str, torch.Tensor],
    models: dict[str, torch.nn.Module],
    normalization: Any,
    *,
    starts: tuple[int, ...],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    selected = validate_window_starts(tensors["transition_valid"], starts)
    collected = {name: [] for name in SCORE_NAMES}
    with torch.inference_mode():
        for offset in range(0, len(selected), batch_size):
            batch_starts = selected[offset : offset + batch_size]
            initial_raw = tensors["visual_tokens"][batch_starts].reshape(
                -1,
                STAGE14.TOKENS_PER_FRAME,
                STAGE14.LATENT_DIMENSION,
            )
            initial_raw = initial_raw.to(device=device, dtype=torch.float32)
            initial = (initial_raw - normalization.latent_mean) / normalization.latent_std
            commanded_prediction = initial.clone()
            executed_prediction = initial.clone()
            no_action_prediction = initial.clone()

            for step in range(1, HORIZON + 1):
                frame_indices = batch_starts + step - 1
                state = tensors["observation_state"][frame_indices].to(device)
                state = (state - normalization.state_mean) / normalization.state_std
                commanded = tensors["commanded_action"][frame_indices].to(device)
                commanded = (commanded - normalization.action_mean) / normalization.action_std
                executed = tensors["executed_action"][frame_indices].to(device)
                executed = (executed - normalization.action_mean) / normalization.action_std
                zeros = torch.zeros_like(commanded)
                commanded_prediction = models["action_conditioned"](
                    commanded_prediction,
                    state,
                    commanded,
                )
                executed_prediction = models["action_conditioned"](
                    executed_prediction,
                    state,
                    executed,
                )
                no_action_prediction = models["no_action"](
                    no_action_prediction,
                    state,
                    zeros,
                )

            target_indices = batch_starts + HORIZON
            target_raw = tensors["visual_tokens"][target_indices].reshape(
                -1,
                STAGE14.TOKENS_PER_FRAME,
                STAGE14.LATENT_DIMENSION,
            )
            target_raw = target_raw.to(device=device, dtype=torch.float32)
            target = (target_raw - normalization.latent_mean) / normalization.latent_std
            values = {
                "commanded_h10_raw_mse": raw_mse(
                    commanded_prediction,
                    target,
                    normalization,
                ),
                "executed_h10_raw_mse": raw_mse(
                    executed_prediction,
                    target,
                    normalization,
                ),
                "no_action_h10_raw_mse": raw_mse(
                    no_action_prediction,
                    target,
                    normalization,
                ),
                "persistence_h10_raw_mse": raw_mse(initial, target, normalization),
            }
            for name, value in values.items():
                collected[name].append(value.cpu().numpy())
    scores = {name: np.concatenate(values) for name, values in collected.items()}
    if any(len(values) != len(starts) for values in scores.values()):
        raise AssertionError("Incomplete episode window scores")
    return scores


def exact_sign_flip(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a non-empty vector")
    observed = float(values.mean())
    assignments = np.array(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = (assignments * values[None, :]).mean(axis=1)
    return {
        "observed_mean": observed,
        "two_sided_p_value": float(np.mean(np.abs(null) >= abs(observed) - 1e-15)),
        "sign_assignments": int(len(null)),
    }


def paired_bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or samples <= 0:
        raise ValueError("Invalid bootstrap input")
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[selected].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_statistics(
    values: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    sign_flip = exact_sign_flip(values)
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "pairs": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "positive_pairs": int((values > 0).sum()),
        "negative_pairs": int((values < 0).sum()),
        "zero_pairs": int((values == 0).sum()),
        "paired_effect_dz": (
            float(values.mean() / standard_deviation)
            if standard_deviation > 0
            else math.copysign(math.inf, float(values.mean()))
        ),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": seed,
        "mean_bootstrap_ci95": paired_bootstrap_ci(values, seed=seed),
        "exact_sign_flip_two_sided_p": sign_flip["two_sided_p_value"],
        "exact_sign_flip_assignments": sign_flip["sign_assignments"],
    }


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


def period_for_start(start: int) -> str:
    if start in PRE_STARTS:
        return "pre"
    if start in POST_STARTS:
        return "post"
    raise ValueError(f"Start {start} does not belong to a registered period")


def pair_effect_rows(
    period_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {(row["pair_id"], row["condition"], row["period"]): row for row in period_rows}
    pair_ids = sorted({row["pair_id"] for row in period_rows})
    results = []
    for pair_id in pair_ids:
        reference = by_key[(pair_id, "nominal", "pre")]
        record: dict[str, Any] = {
            "pair_id": pair_id,
            "benchmark_task_id": reference["benchmark_task_id"],
            "init_state_index": reference["init_state_index"],
        }
        for name in SCORE_NAMES:
            nominal_pre = float(by_key[(pair_id, "nominal", "pre")][name])
            nominal_post = float(by_key[(pair_id, "nominal", "post")][name])
            fault_pre = float(by_key[(pair_id, "persistent_motion_dropout", "pre")][name])
            fault_post = float(by_key[(pair_id, "persistent_motion_dropout", "post")][name])
            nominal_delta = nominal_post - nominal_pre
            fault_delta = fault_post - fault_pre
            record[f"{name}_nominal_pre"] = nominal_pre
            record[f"{name}_nominal_post"] = nominal_post
            record[f"{name}_nominal_delta"] = nominal_delta
            record[f"{name}_fault_pre"] = fault_pre
            record[f"{name}_fault_post"] = fault_post
            record[f"{name}_fault_delta"] = fault_delta
            record[f"{name}_difference_in_differences"] = fault_delta - nominal_delta
        record["fault_post_commanded_minus_executed"] = (
            record["commanded_h10_raw_mse_fault_post"] - record["executed_h10_raw_mse_fault_post"]
        )
        results.append(record)
    return results


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.encoder_batch_size != ENCODER_BATCH_SIZE:
        raise ValueError(f"Encoder batch size is pinned to {ENCODER_BATCH_SIZE}")
    if args.eval_batch_size <= 0:
        raise ValueError("Evaluation batch size must be positive")
    if args.model_root.name != STAGE12.MODEL_REVISION:
        raise ValueError("DINOv2 model revision is not pinned")

    repo_root = Path.cwd()
    stage20_report = json.loads(args.stage20_report.read_text(encoding="utf-8"))
    if stage20_report["status"] != "passed":
        raise AssertionError("Stage 20 report is not ready")
    if not stage20_report["stage21_readiness"]["frozen_wm_paired_scoring_can_start"]:
        raise AssertionError("Stage 20 did not authorize frozen WM scoring")
    if sha256_file(args.stage20_episodes) != stage20_report["artifacts"]["episodes_csv_sha256"]:
        raise AssertionError("Stage 20 episodes manifest hash changed")
    if sha256_file(args.stage20_pairs) != stage20_report["artifacts"]["pairs_csv_sha256"]:
        raise AssertionError("Stage 20 pairs manifest hash changed")

    episode_rows = read_csv(args.stage20_episodes)
    if len(episode_rows) != 12:
        raise AssertionError("Stage 20 must contain exactly 12 episodes")
    if set(row["condition"] for row in episode_rows) != set(CONDITIONS):
        raise AssertionError("Stage 20 conditions changed")
    counts = {
        (pair_id, condition): sum(row["pair_id"] == pair_id and row["condition"] == condition for row in episode_rows)
        for pair_id in {row["pair_id"] for row in episode_rows}
        for condition in CONDITIONS
    }
    if len(counts) != 12 or any(value != 1 for value in counts.values()):
        raise AssertionError("Stage 20 does not contain six complete pairs")
    episode_rows.sort(
        key=lambda row: (
            int(row["benchmark_task_id"]),
            int(row["init_state_index"]),
            CONDITIONS.index(row["condition"]),
        )
    )

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
    encoded = 0
    resumed = 0
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
        encoded += int(not was_resumed)
        resumed += int(was_resumed)
        feature_rows.append(
            {
                "rollout_id": row["rollout_id"],
                "pair_id": row["pair_id"],
                "condition": row["condition"],
                "benchmark_task_id": int(row["benchmark_task_id"]),
                "init_state_index": int(row["init_state_index"]),
                "steps": int(row["steps"]),
                "valid_transitions": int(tensors["transition_valid"].sum()),
                "source_content_sha256": row["content_sha256"],
                "policy_pixels_sha256": metadata["policy_pixels_sha256"],
                "feature_file": f"outputs/wm/stage21_paired_wm_scoring/features/{path.name}",
                "feature_bytes": path.stat().st_size,
                "feature_sha256": sha256_file(path),
            }
        )
        print(
            f"[{position}/12] {row['rollout_id']} encoded={encoded} resumed={resumed}",
            flush=True,
        )

    recheck_row = min(episode_rows, key=lambda row: int(row["steps"]))
    cached, cached_metadata = validate_feature(
        feature_path(args.output_dir, recheck_row),
        recheck_row,
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
        raise AssertionError("Deterministic pixel recheck differs")
    if not torch.equal(recheck_tokens, cached["visual_tokens"]):
        raise AssertionError("Deterministic DINOv2 recheck differs")
    encoding_finished = time.perf_counter()
    encoder.to("cpu")
    del encoder, processor, images, arrays, recheck_tokens, cached
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

    window_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []
    nominal_action_equivalence = True
    fault_pre_action_equivalence = True
    for row in episode_rows:
        tensors, _ = validate_feature(feature_path(args.output_dir, row), row)
        for period, starts in (("pre", PRE_STARTS), ("post", POST_STARTS)):
            scores = score_episode_windows(
                tensors,
                models,
                normalization,
                starts=starts,
                device=device,
                batch_size=args.eval_batch_size,
            )
            if row["condition"] == "nominal":
                nominal_action_equivalence &= np.array_equal(
                    scores["commanded_h10_raw_mse"],
                    scores["executed_h10_raw_mse"],
                )
            if row["condition"] == "persistent_motion_dropout" and period == "pre":
                fault_pre_action_equivalence &= np.array_equal(
                    scores["commanded_h10_raw_mse"],
                    scores["executed_h10_raw_mse"],
                )
            period_rows.append(
                {
                    "rollout_id": row["rollout_id"],
                    "pair_id": row["pair_id"],
                    "condition": row["condition"],
                    "benchmark_task_id": int(row["benchmark_task_id"]),
                    "init_state_index": int(row["init_state_index"]),
                    "period": period,
                    "window_starts": len(starts),
                    **{name: float(scores[name].mean()) for name in SCORE_NAMES},
                }
            )
            for index, start in enumerate(starts):
                window_rows.append(
                    {
                        "rollout_id": row["rollout_id"],
                        "pair_id": row["pair_id"],
                        "condition": row["condition"],
                        "benchmark_task_id": int(row["benchmark_task_id"]),
                        "init_state_index": int(row["init_state_index"]),
                        "period": period_for_start(start),
                        "start_index": start,
                        "target_index": start + HORIZON,
                        **{name: float(scores[name][index]) for name in SCORE_NAMES},
                    }
                )
        print(f"scored {row['rollout_id']}", flush=True)

    if not nominal_action_equivalence or not fault_pre_action_equivalence:
        raise AssertionError("Commanded/executed negative-control equivalence failed")
    effects = pair_effect_rows(period_rows)
    score_statistics: dict[str, Any] = {}
    for index, name in enumerate(SCORE_NAMES):
        values = np.array(
            [row[f"{name}_difference_in_differences"] for row in effects],
            dtype=np.float64,
        )
        score_statistics[name] = paired_statistics(
            values,
            seed=BOOTSTRAP_SEED + index,
        )
    mechanism_values = np.array(
        [row["fault_post_commanded_minus_executed"] for row in effects],
        dtype=np.float64,
    )
    mechanism_statistics = paired_statistics(
        mechanism_values,
        seed=BOOTSTRAP_SEED + len(SCORE_NAMES),
    )
    diagnostic_raw_p = {
        "executed_h10_raw_mse_difference_in_differences": score_statistics["executed_h10_raw_mse"][
            "exact_sign_flip_two_sided_p"
        ],
        "no_action_h10_raw_mse_difference_in_differences": score_statistics["no_action_h10_raw_mse"][
            "exact_sign_flip_two_sided_p"
        ],
        "persistence_h10_raw_mse_difference_in_differences": score_statistics["persistence_h10_raw_mse"][
            "exact_sign_flip_two_sided_p"
        ],
        "fault_post_commanded_minus_executed": mechanism_statistics["exact_sign_flip_two_sided_p"],
    }
    diagnostic_adjusted_p = holm_adjust(diagnostic_raw_p)
    task_summary = {}
    for task_id in sorted({int(row["benchmark_task_id"]) for row in effects}):
        task_rows = [row for row in effects if int(row["benchmark_task_id"]) == task_id]
        task_summary[str(task_id)] = {
            "pairs": len(task_rows),
            "commanded_difference_in_differences_mean": float(
                np.mean([row["commanded_h10_raw_mse_difference_in_differences"] for row in task_rows])
            ),
            "executed_difference_in_differences_mean": float(
                np.mean([row["executed_h10_raw_mse_difference_in_differences"] for row in task_rows])
            ),
        }

    primary = score_statistics["commanded_h10_raw_mse"]
    primary_support = (
        primary["mean"] > 0 and primary["positive_pairs"] >= 5 and primary["exact_sign_flip_two_sided_p"] <= 0.05
    )
    mechanism_support = mechanism_statistics["mean"] > 0 and mechanism_statistics["positive_pairs"] >= 5
    scored_finished = time.perf_counter()
    peak_memory = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None

    feature_text = csv_text(feature_rows)
    window_text = csv_text(window_rows)
    period_text = csv_text(period_rows)
    effects_text = csv_text(effects)
    statistics_payload = {
        "schema_version": 1,
        "statistical_unit": "episode pair",
        "overlapping_windows_treated_as_independent_samples": False,
        "primary_score": "commanded_h10_raw_mse",
        "primary_direction": "positive fault(post-pre)-nominal(post-pre)",
        "primary_decision_rule": {
            "mean_greater_than_zero": True,
            "at_least_five_of_six_pairs_positive": True,
            "exact_sign_flip_two_sided_p_at_most": 0.05,
        },
        "scores": score_statistics,
        "mechanism_fault_post_commanded_minus_executed": mechanism_statistics,
        "diagnostic_multiple_testing": {
            "method": "Holm family-wise error correction",
            "family": list(diagnostic_raw_p),
            "raw_p_values": diagnostic_raw_p,
            "adjusted_p_values": diagnostic_adjusted_p,
            "scores_below_0_05_after_correction": [
                name for name, value in diagnostic_adjusted_p.items() if value < 0.05
            ],
        },
        "task_summary": task_summary,
    }
    statistics_text = canonical_json(statistics_payload)
    report = {
        "schema_version": 1,
        "stage": "WM Stage 21 frozen paired intervention scoring",
        "status": "passed",
        "scope": {
            "pairs": 6,
            "episodes": 12,
            "tasks": [4, 7, 8],
            "windows_per_episode": len(PRE_STARTS) + len(POST_STARTS),
            "window_rows": len(window_rows),
            "pair_is_statistical_unit": True,
            "model_parameters_updated": 0,
            "failure_classifier_trained": False,
            "thresholds_fitted": 0,
            "expert_test_episodes_used": 0,
        },
        "protocol": {
            "horizon": HORIZON,
            "pre_start_indices": [PRE_STARTS[0], PRE_STARTS[-1]],
            "post_start_indices": [POST_STARTS[0], POST_STARTS[-1]],
            "intervention_start": 50,
            "pre_windows_cross_intervention": False,
            "primary_score": "commanded_h10_raw_mse",
            "primary_effect": "fault(post-pre)-nominal(post-pre)",
            "state_context": "oracle ground-truth state at every recursive rollout step",
            "diagnostics": [
                "executed_h10_raw_mse",
                "no_action_h10_raw_mse",
                "persistence_h10_raw_mse",
                "fault_post_commanded_minus_executed",
            ],
        },
        "frozen_components": {
            "encoder_model_id": STAGE12.MODEL_ID,
            "encoder_revision": STAGE12.MODEL_REVISION,
            "stage14_report_sha256": sha256_file(args.stage14_report),
            "action_conditioned_checkpoint_sha256": stage14_report["variants"]["action_conditioned"][
                "checkpoint_sha256"
            ],
            "no_action_checkpoint_sha256": stage14_report["variants"]["no_action"]["checkpoint_sha256"],
            "normalization_source": "Stage 13 train split only",
        },
        "negative_controls": {
            "nominal_commanded_equals_executed_scores_bit_exact": nominal_action_equivalence,
            "fault_pre_commanded_equals_executed_scores_bit_exact": fault_pre_action_equivalence,
        },
        "primary_result": primary,
        "mechanism_result": mechanism_statistics,
        "conclusion": {
            "primary_action_consequence_mismatch_supported": primary_support,
            "exploratory_commanded_stronger_than_executed_supported": mechanism_support,
            "diagnostic_survives_holm_correction": (
                diagnostic_adjusted_p["fault_post_commanded_minus_executed"] < 0.05
            ),
            "decision_rule_passed": primary_support,
        },
        "artifacts": {
            "feature_manifest": "feature_manifest.csv",
            "feature_manifest_sha256": hashlib.sha256(feature_text.encode()).hexdigest(),
            "window_scores": "window_scores.csv",
            "window_scores_sha256": hashlib.sha256(window_text.encode()).hexdigest(),
            "episode_period_scores": "episode_period_scores.csv",
            "episode_period_scores_sha256": hashlib.sha256(period_text.encode()).hexdigest(),
            "pair_effects": "pair_effects.csv",
            "pair_effects_sha256": hashlib.sha256(effects_text.encode()).hexdigest(),
            "statistics": "statistics.json",
            "statistics_sha256": hashlib.sha256(statistics_text.encode()).hexdigest(),
            "feature_cache": "local-only under outputs/wm/stage21_paired_wm_scoring/features",
        },
        "determinism": {
            "encoder_batch_size": ENCODER_BATCH_SIZE,
            "deterministic_recheck_rollout_id": recheck_row["rollout_id"],
            "deterministic_recheck_bit_exact": True,
        },
        "runtime": {
            "device": str(device),
            "encoded_feature_shards": encoded,
            "resumed_feature_shards": resumed,
            "encoder_load_seconds": encoder_loaded - started,
            "feature_encoding_and_recheck_seconds": encoding_finished - encoder_loaded,
            "scoring_seconds": scored_finished - encoding_finished,
            "total_seconds": scored_finished - started,
            "peak_torch_gpu_memory_bytes": peak_memory,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "limitations": [
            "Only six episode pairs across three tasks are available.",
            "The intervention is a strong synthetic persistent actuator fault.",
            "H10 windows overlap within an episode; inference therefore uses only six pair-level aggregates.",
            "Oracle robot state is supplied at every recursive WM step.",
            "This stage evaluates a frozen score and does not establish a deployed online shield.",
        ],
        "stage22_readiness": {
            "online_shield_prototype_justified": primary_support,
            "requires_more_pairs_before_general_claim": True,
        },
    }
    report_text = canonical_json(report)
    write_deterministic(
        args.public_results_dir / "feature_manifest.csv",
        feature_text,
        overwrite=args.overwrite_public,
    )
    write_deterministic(
        args.public_results_dir / "window_scores.csv",
        window_text,
        overwrite=args.overwrite_public,
    )
    write_deterministic(
        args.public_results_dir / "episode_period_scores.csv",
        period_text,
        overwrite=args.overwrite_public,
    )
    write_deterministic(
        args.public_results_dir / "pair_effects.csv",
        effects_text,
        overwrite=args.overwrite_public,
    )
    write_deterministic(
        args.public_results_dir / "statistics.json",
        statistics_text,
        overwrite=args.overwrite_public,
    )
    write_deterministic(
        args.public_results_dir / "report.json",
        report_text,
        overwrite=args.overwrite_public,
    )
    print(
        "Stage 21 complete: "
        f"primary mean={primary['mean']:.8f}, "
        f"positive={primary['positive_pairs']}/6, "
        f"exact p={primary['exact_sign_flip_two_sided_p']:.5f}, "
        f"supported={primary_support}",
        flush=True,
    )


if __name__ == "__main__":
    main()
