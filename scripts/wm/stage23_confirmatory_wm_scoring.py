#!/usr/bin/env python

"""Confirm the frozen WM action-sensitivity score on the Stage 22 cohort.

The primary unit is one initial-state pair. For each pair, the post-intervention
commanded-minus-executed H10 raw latent MSE is averaged over the two pinned mild
faults. The one-sided direction and decision rule are fixed before encoding.
No encoder, dynamics, detector, or threshold parameter is updated.
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
from stage21_paired_wm_scoring import (
    STAGE12,
    STAGE13,
    STAGE14,
    canonical_json,
    csv_text,
    load_dynamics,
    normalization_from_training,
    policy_images,
    policy_state,
    read_csv,
    sha256_bytes,
    sha256_file,
    source_arrays,
    write_deterministic,
)
from stage22_collect_confirmatory_interventions import (
    CONDITIONS,
    CONFIRMATORY_POST_END_EXCLUSIVE,
    validate_episode_arrays,
)
from transformers import AutoImageProcessor, Dinov2Model

HORIZON = 10
PRE_STARTS = tuple(range(10, 40))
POST_STARTS = tuple(range(50, 80))
FAULT_CONDITIONS = CONDITIONS[1:]
ENCODER_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 60
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2323
FEATURE_SCHEMA_VERSION = "1"
SCORE_NAMES = (
    "commanded_h10_raw_mse",
    "executed_h10_raw_mse",
    "no_action_h10_raw_mse",
    "persistence_h10_raw_mse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage22-episodes",
        type=Path,
        default=Path("results/wm/stage22_confirmatory_interventions/episodes.csv"),
    )
    parser.add_argument(
        "--stage22-pairs",
        type=Path,
        default=Path("results/wm/stage22_confirmatory_interventions/pairs.csv"),
    )
    parser.add_argument(
        "--stage22-report",
        type=Path,
        default=Path("results/wm/stage22_confirmatory_interventions/report.json"),
    )
    parser.add_argument(
        "--stage21-report",
        type=Path,
        default=Path("results/wm/stage21_paired_wm_scoring/report.json"),
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
        default=Path("outputs/wm/stage23_confirmatory_wm_scoring"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage23_confirmatory_wm_scoring"),
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
        "protocol_sha256": row["protocol_sha256"],
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
    expected_dtypes = {
        "visual_tokens": torch.float16,
        "observation_state": torch.float32,
        "commanded_action": torch.float32,
        "executed_action": torch.float32,
        "transition_valid": torch.bool,
        "intervention_mask": torch.bool,
    }
    for key, shape in expected_shapes.items():
        if tuple(tensors[key].shape) != shape:
            raise AssertionError(f"{key} shape differs in {path}")
        if tensors[key].dtype != expected_dtypes[key]:
            raise AssertionError(f"{key} dtype differs in {path}")
    for key in ("visual_tokens", "observation_state", "commanded_action", "executed_action"):
        if not bool(torch.isfinite(tensors[key]).all()):
            raise AssertionError(f"Non-finite feature: {key}")
    if int(tensors["transition_valid"].sum()) != int(row["valid_transitions"]):
        raise AssertionError("Valid transition count changed")
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
        raise AssertionError("Missing policy pixel hash")
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
        raise AssertionError(f"Stage 22 source hash mismatch: {source_path}")
    arrays = source_arrays(source_path)
    validate_episode_arrays(
        arrays,
        condition=row["condition"],
        resolution=360,
    )
    if int(arrays["transition_valid"].sum()) < CONFIRMATORY_POST_END_EXCLUSIVE:
        raise AssertionError("Stage 22 source does not cover the registered windows")
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
) -> torch.Tensor:
    selected = torch.tensor(starts, dtype=torch.int64)
    for start in selected.tolist():
        if start < 0 or start + HORIZON > len(transition_valid):
            raise ValueError(f"H10 start {start} is outside the episode")
        if not bool(transition_valid[start : start + HORIZON].all()):
            raise AssertionError(f"Invalid transition in H10 window at {start}")
    return selected


def raw_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalization: Any,
) -> torch.Tensor:
    prediction_raw = STAGE14.raw_latent(prediction, normalization)
    target_raw = STAGE14.raw_latent(target, normalization)
    return (prediction_raw - target_raw).square().mean(dim=(1, 2))


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


def exact_sign_flip(
    values: np.ndarray,
    *,
    alternative: str,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a non-empty vector")
    assignments = np.array(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = (assignments * values[None, :]).mean(axis=1)
    observed = float(values.mean())
    if alternative == "greater":
        p_value = float(np.mean(null >= observed - 1e-15))
    elif alternative == "two-sided":
        p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    return {
        "observed_mean": observed,
        "p_value": p_value,
        "sign_assignments": int(len(null)),
    }


def bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int,
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    selected = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_SAMPLES, len(values)),
    )
    means = values[selected].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_statistics(
    values: np.ndarray,
    *,
    seed: int,
    alternative: str,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    exact = exact_sign_flip(values, alternative=alternative)
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
        "mean_bootstrap_ci95": bootstrap_ci(values, seed=seed),
        "exact_sign_flip_alternative": alternative,
        "exact_sign_flip_p": exact["p_value"],
        "exact_sign_flip_assignments": exact["sign_assignments"],
    }


def build_pair_confirmation_rows(
    period_rows: list[dict[str, Any]],
    stage22_pairs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    lookup = {(row["pair_id"], row["condition"], row["period"]): row for row in period_rows}
    source_lookup = {row["pair_id"]: row for row in stage22_pairs}
    results = []
    for pair_id in sorted(source_lookup):
        source = source_lookup[pair_id]
        nominal_pre = lookup[(pair_id, "nominal", "pre")]
        nominal_post = lookup[(pair_id, "nominal", "post")]
        record: dict[str, Any] = {
            "pair_id": pair_id,
            "benchmark_task_id": int(source["benchmark_task_id"]),
            "init_state_index": int(source["init_state_index"]),
        }
        candidate_values = []
        for condition in FAULT_CONDITIONS:
            fault_pre = lookup[(pair_id, condition, "pre")]
            fault_post = lookup[(pair_id, condition, "post")]
            candidate = float(fault_post["commanded_h10_raw_mse"]) - float(fault_post["executed_h10_raw_mse"])
            candidate_values.append(candidate)
            record[f"{condition}_post_commanded_minus_executed"] = candidate
            record[f"{condition}_commanded_difference_in_differences"] = (
                float(fault_post["commanded_h10_raw_mse"])
                - float(fault_pre["commanded_h10_raw_mse"])
                - (float(nominal_post["commanded_h10_raw_mse"]) - float(nominal_pre["commanded_h10_raw_mse"]))
            )
            record[f"{condition}_executed_difference_in_differences"] = (
                float(fault_post["executed_h10_raw_mse"])
                - float(fault_pre["executed_h10_raw_mse"])
                - (float(nominal_post["executed_h10_raw_mse"]) - float(nominal_pre["executed_h10_raw_mse"]))
            )
            pixel_motion = float(source[f"{condition}_post_pixel_mae_mean"])
            eef_motion = float(source[f"{condition}_post_eef_motion_l2_mean"])
            record[f"{condition}_candidate_per_pixel_motion"] = candidate / pixel_motion
            record[f"{condition}_candidate_per_eef_motion"] = candidate / eef_motion
        record["primary_mean_across_interventions"] = float(np.mean(candidate_values))
        record["both_interventions_positive"] = all(value > 0 for value in candidate_values)
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

    stage22_report = json.loads(args.stage22_report.read_text(encoding="utf-8"))
    stage21_report = json.loads(args.stage21_report.read_text(encoding="utf-8"))
    if stage22_report["status"] != "passed":
        raise AssertionError("Stage 22 is not ready")
    if not stage22_report["stage23_readiness"]["frozen_wm_scoring_can_start"]:
        raise AssertionError("Stage 22 did not authorize confirmation scoring")
    if stage22_report["protocol"]["stage23_candidate_direction"] != "positive":
        raise AssertionError("Stage 23 direction changed")
    if stage22_report["scope"]["world_model_scores_computed"] != 0:
        raise AssertionError("Stage 22 was not score blind")
    if stage21_report["conclusion"]["decision_rule_passed"]:
        raise AssertionError("Stage 21 discovery status changed")
    if sha256_file(args.stage22_episodes) != stage22_report["artifacts"]["episodes_csv_sha256"]:
        raise AssertionError("Stage 22 episode manifest changed")
    if sha256_file(args.stage22_pairs) != stage22_report["artifacts"]["pairs_csv_sha256"]:
        raise AssertionError("Stage 22 pair manifest changed")

    episode_rows = read_csv(args.stage22_episodes)
    stage22_pairs = read_csv(args.stage22_pairs)
    if len(episode_rows) != 18 or len(stage22_pairs) != 6:
        raise AssertionError("Stage 22 cohort size changed")
    if set(row["condition"] for row in episode_rows) != set(CONDITIONS):
        raise AssertionError("Stage 22 condition set changed")
    if any(row["world_model_scores_computed"] != "False" for row in episode_rows):
        raise AssertionError("Stage 22 manifest is not score blind")
    episode_rows.sort(
        key=lambda row: (
            int(row["benchmark_task_id"]),
            int(row["init_state_index"]),
            CONDITIONS.index(row["condition"]),
        )
    )

    repo_root = Path.cwd()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(
        args.model_root,
        local_files_only=True,
    )
    encoder = Dinov2Model.from_pretrained(
        args.model_root,
        local_files_only=True,
    ).eval()
    encoder.requires_grad_(False)
    encoder.to(device)
    encoder_loaded = time.perf_counter()

    feature_rows = []
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
                "feature_file": (f"outputs/wm/stage23_confirmatory_wm_scoring/features/{path.name}"),
                "feature_bytes": path.stat().st_size,
                "feature_sha256": sha256_file(path),
            }
        )
        print(
            f"[{position}/18] {row['rollout_id']} encoded={encoded} resumed={resumed}",
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
        raise AssertionError("Deterministic token recheck differs")
    encoding_finished = time.perf_counter()
    encoder.to("cpu")
    del encoder, processor, arrays, images, recheck_tokens, cached
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stage13_report = json.loads(args.stage13_report.read_text(encoding="utf-8"))
    if stage13_report["dataset"]["test_split_cached_episodes"] != 0:
        raise AssertionError("Stage 13 indicates test leakage")
    normalization = normalization_from_training(
        args.stage13_manifest,
        repo_root=repo_root,
    ).to(device)
    stage14_report = json.loads(args.stage14_report.read_text(encoding="utf-8"))
    models = load_dynamics(stage14_report, args.checkpoint_dir, device)

    window_rows = []
    period_rows = []
    nominal_equivalence = True
    fault_pre_equivalence = True
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
                nominal_equivalence &= np.array_equal(
                    scores["commanded_h10_raw_mse"],
                    scores["executed_h10_raw_mse"],
                )
            elif period == "pre":
                fault_pre_equivalence &= np.array_equal(
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
                        "period": period,
                        "start_index": start,
                        "target_index": start + HORIZON,
                        **{name: float(scores[name][index]) for name in SCORE_NAMES},
                    }
                )
        print(f"scored {row['rollout_id']}", flush=True)
    if not nominal_equivalence or not fault_pre_equivalence:
        raise AssertionError("Commanded/executed negative control failed")

    confirmation_rows = build_pair_confirmation_rows(
        period_rows,
        stage22_pairs,
    )
    primary_values = np.array([row["primary_mean_across_interventions"] for row in confirmation_rows])
    primary = paired_statistics(
        primary_values,
        seed=BOOTSTRAP_SEED,
        alternative="greater",
    )
    per_intervention = {}
    for index, condition in enumerate(FAULT_CONDITIONS, start=1):
        values = np.array([row[f"{condition}_post_commanded_minus_executed"] for row in confirmation_rows])
        per_intervention[condition] = paired_statistics(
            values,
            seed=BOOTSTRAP_SEED + index,
            alternative="greater",
        )
    contrast_values = np.array(
        [
            row["action_delay_3_post_commanded_minus_executed"]
            - row["motion_attenuation_0p5_post_commanded_minus_executed"]
            for row in confirmation_rows
        ]
    )
    condition_contrast = paired_statistics(
        contrast_values,
        seed=BOOTSTRAP_SEED + 3,
        alternative="two-sided",
    )
    primary_passed = primary["mean"] > 0 and primary["positive_pairs"] >= 5 and primary["exact_sign_flip_p"] <= 0.05
    discovery_mean = float(stage21_report["mechanism_result"]["mean"])
    confirmation_over_discovery = float(primary["mean"] / discovery_mean)
    scored_finished = time.perf_counter()
    peak_memory = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None

    feature_text = csv_text(feature_rows)
    window_text = csv_text(window_rows)
    period_text = csv_text(period_rows)
    confirmation_text = csv_text(confirmation_rows)
    statistics_payload = {
        "schema_version": 1,
        "statistical_unit": "initial-state pair",
        "primary_score": (
            "mean across two interventions of fault-post commanded_h10_raw_mse minus executed_h10_raw_mse"
        ),
        "primary_direction": "positive",
        "primary_decision_rule": {
            "mean_greater_than_zero": True,
            "at_least_five_of_six_pairs_positive": True,
            "one_sided_exact_sign_flip_p_at_most": 0.05,
        },
        "overlapping_windows_treated_as_independent_samples": False,
        "two_interventions_treated_as_independent_pairs": False,
        "primary": primary,
        "per_intervention_supportive": per_intervention,
        "action_delay_minus_attenuation_contrast_exploratory": condition_contrast,
    }
    statistics_text = canonical_json(statistics_payload)
    report = {
        "schema_version": 1,
        "stage": "WM Stage 23 independent confirmatory action-sensitivity scoring",
        "status": "passed",
        "scope": {
            "pairs": 6,
            "episodes": 18,
            "tasks": [4, 7, 8],
            "conditions": list(CONDITIONS),
            "windows_per_episode": len(PRE_STARTS) + len(POST_STARTS),
            "window_rows": len(window_rows),
            "pair_is_primary_statistical_unit": True,
            "model_parameters_updated": 0,
            "detector_parameters_fitted": 0,
            "thresholds_fitted": 0,
            "expert_test_episodes_used": 0,
        },
        "confirmatory_protocol": {
            "candidate_frozen_in_stage22": True,
            "stage22_report_sha256": sha256_file(args.stage22_report),
            "stage21_discovery_report_sha256": sha256_file(args.stage21_report),
            "discovery_episode_reused": False,
            "horizon": HORIZON,
            "pre_start_indices": [PRE_STARTS[0], PRE_STARTS[-1]],
            "post_start_indices": [POST_STARTS[0], POST_STARTS[-1]],
            "primary_score": statistics_payload["primary_score"],
            "primary_direction": "positive",
            "decision_rule": statistics_payload["primary_decision_rule"],
            "state_context": "oracle ground-truth state at every recursive rollout step",
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
            "nominal_commanded_equals_executed_scores_bit_exact": nominal_equivalence,
            "fault_pre_commanded_equals_executed_scores_bit_exact": (fault_pre_equivalence),
        },
        "primary_result": primary,
        "per_intervention_supportive_results": per_intervention,
        "effect_size_context": {
            "stage21_severe_dropout_discovery_mean": discovery_mean,
            "stage23_mild_intervention_confirmation_mean": primary["mean"],
            "confirmation_over_discovery_ratio": confirmation_over_discovery,
            "confirmation_effect_smaller_than_discovery": (abs(primary["mean"]) < abs(discovery_mean)),
        },
        "conclusion": {
            "independent_confirmation_passed": primary_passed,
            "candidate_direction_replicated": primary["mean"] > 0,
            "decision_rule_passed": primary_passed,
            "motion_attenuation_supportive_p_at_most_0_05": (
                per_intervention["motion_attenuation_0p5"]["exact_sign_flip_p"] <= 0.05
            ),
            "action_delay_supportive_p_at_most_0_05": (per_intervention["action_delay_3"]["exact_sign_flip_p"] <= 0.05),
        },
        "artifacts": {
            "feature_manifest": "feature_manifest.csv",
            "feature_manifest_sha256": hashlib.sha256(feature_text.encode()).hexdigest(),
            "window_scores": "window_scores.csv",
            "window_scores_sha256": hashlib.sha256(window_text.encode()).hexdigest(),
            "episode_period_scores": "episode_period_scores.csv",
            "episode_period_scores_sha256": hashlib.sha256(period_text.encode()).hexdigest(),
            "pair_confirmation": "pair_confirmation.csv",
            "pair_confirmation_sha256": hashlib.sha256(confirmation_text.encode()).hexdigest(),
            "statistics": "statistics.json",
            "statistics_sha256": hashlib.sha256(statistics_text.encode()).hexdigest(),
            "feature_cache": ("local-only under outputs/wm/stage23_confirmatory_wm_scoring/features"),
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
            "Only six initial-state pairs across three tasks are tested.",
            "Both confirmation interventions are synthetic actuator faults.",
            "All 18 episodes succeed, so this is not a natural policy-failure detector test.",
            "Executed-action conditioning assumes actuator feedback is available.",
            "Oracle robot state is supplied at each recursive WM step.",
        ],
        "stage24_readiness": {
            "offline_detector_analysis_justified": primary_passed,
            "online_shield_claim_justified": False,
            "requires_threshold_calibration_on_separate_data": True,
        },
    }
    report_text = canonical_json(report)
    for filename, content in (
        ("feature_manifest.csv", feature_text),
        ("window_scores.csv", window_text),
        ("episode_period_scores.csv", period_text),
        ("pair_confirmation.csv", confirmation_text),
        ("statistics.json", statistics_text),
        ("report.json", report_text),
    ):
        write_deterministic(
            args.public_results_dir / filename,
            content,
            overwrite=args.overwrite_public,
        )
    print(
        "Stage 23 complete: "
        f"primary mean={primary['mean']:.8f}, "
        f"positive={primary['positive_pairs']}/6, "
        f"one-sided exact p={primary['exact_sign_flip_p']:.5f}, "
        f"confirmed={primary_passed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
