#!/usr/bin/env python
# ruff: noqa: E501

"""Discover an observation-residual signal on Stage 26 hidden dynamics pairs.

The six Stage 26 pairs are discovery-only. H10 starts 10--40 calibrate each
episode's online baseline; starts 41--50 are the only fault-influenced windows
and retain bit-exact policy commands across nominal/fault pairs. The primary
signal is action-conditioned H10 error minus no-action H10 error. Frozen-WM,
latent-persistence, state-motion, and pixel-motion scores receive the same
within-episode standardization. No held-out Stage 25 test state is accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    read_csv,
    sha256_bytes,
    sha256_file,
    source_arrays,
    write_deterministic,
)
from stage23_confirmatory_wm_scoring import paired_statistics
from stage26_hidden_dynamics_cohort import (
    INTERVENTION_START,
    LIVE_STEPS,
    POST_HORIZON,
    validate_arrays,
)
from transformers import AutoImageProcessor, Dinov2Model

HORIZON = POST_HORIZON
PRE_STARTS = tuple(range(10, 41))
INFLUENCED_STARTS = tuple(range(41, 51))
ALL_STARTS = PRE_STARTS + INFLUENCED_STARTS
CONDITIONS = ("nominal", "arm_actuator_gain_0p5")
FAULT_CONDITION = CONDITIONS[1]
ENCODER_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2727
FEATURE_SCHEMA_VERSION = "1"
STANDARDIZATION_RELATIVE_FLOOR = 1e-6
STANDARDIZATION_ABSOLUTE_FLOOR = 1e-8
PRIMARY_METRIC = "wm_relative_residual"
RAW_METRICS = (
    PRIMARY_METRIC,
    "wm_action_residual",
    "no_action_residual",
    "latent_persistence_motion",
    "eef_path_length",
    "pixel_endpoint_mae",
)
SIMPLE_BASELINES = (
    "no_action_residual",
    "latent_persistence_motion",
    "eef_path_length",
    "pixel_endpoint_mae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage26-episodes",
        type=Path,
        default=Path("results/wm/stage26_hidden_dynamics_cohort/episodes.csv"),
    )
    parser.add_argument(
        "--stage26-report",
        type=Path,
        default=Path("results/wm/stage26_hidden_dynamics_cohort/report.json"),
    )
    parser.add_argument(
        "--stage26-gate",
        type=Path,
        default=Path("results/wm/stage26_hidden_dynamics_cohort/frozen_gate.json"),
    )
    parser.add_argument(
        "--stage26-output-dir",
        type=Path,
        default=Path("outputs/wm/stage26_hidden_dynamics_cohort"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=STAGE12.default_model_root(),
    )
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
        default=Path("outputs/wm/stage27_hidden_dynamics_residual"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage27_hidden_dynamics_residual"),
    )
    parser.add_argument(
        "--media-path",
        type=Path,
        default=Path("media/wm_stage27_hidden_dynamics_residual.svg"),
    )
    parser.add_argument(
        "--encoder-batch-size",
        type=int,
        default=ENCODER_BATCH_SIZE,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=EVAL_BATCH_SIZE,
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite-public", action="store_true")
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def episode_archive(
    stage26_output_dir: Path,
    pair_id: str,
    condition: str,
) -> Path:
    return stage26_output_dir / "episodes" / f"{pair_id}-{condition}.npz"


def selected_episode_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    selected = [dict(row) for row in rows if row["condition"] in CONDITIONS]
    selected.sort(
        key=lambda row: (
            int(row["benchmark_task_id"]),
            int(row["init_state_index"]),
            CONDITIONS.index(row["condition"]),
        )
    )
    if len(selected) != 12:
        raise AssertionError(f"Stage 27 requires 12 episodes, found {len(selected)}")
    pairs = {}
    for row in selected:
        pairs.setdefault(row["pair_id"], set()).add(row["condition"])
    if len(pairs) != 6 or any(value != set(CONDITIONS) for value in pairs.values()):
        raise AssertionError("Stage 27 requires six complete nominal/fault pairs")
    return selected


def protocol_payload(
    *,
    args: argparse.Namespace,
    stage26_report: dict[str, Any],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "WM Stage 27 hidden-dynamics observation residual discovery",
        "source_pairs": sorted({row["pair_id"] for row in rows}),
        "source_conditions": list(CONDITIONS),
        "source_is_discovery_only": True,
        "stage25_held_out_test_states_accessed": False,
        "horizon": HORIZON,
        "intervention_action_step": INTERVENTION_START,
        "pre_window_starts_inclusive": [PRE_STARTS[0], PRE_STARTS[-1]],
        "fault_influenced_window_starts_inclusive": [
            INFLUENCED_STARTS[0],
            INFLUENCED_STARTS[-1],
        ],
        "influenced_window_semantics": (
            "start 41 target 51 first contains action step 50; start 50 target 60 is fully post-shift"
        ),
        "all_influenced_policy_commands_pairwise_bit_exact": True,
        "primary_metric": {
            "name": PRIMARY_METRIC,
            "raw_definition": ("action-conditioned H10 raw latent MSE minus no-action H10 raw latent MSE"),
            "online_anomaly_definition": (
                "absolute deviation from episode pre mean divided by "
                "episode pre sample standard deviation with frozen floor"
            ),
            "paired_effect": ("mean influenced anomaly(fault) minus mean influenced anomaly(nominal)"),
            "direction": "positive",
            "decision_rule": {
                "mean_greater_than_zero": True,
                "at_least_five_of_six_pairs_positive": True,
                "one_sided_exact_sign_flip_p_at_most": 0.05,
            },
        },
        "supportive_wm_metric": "wm_action_residual",
        "simple_baselines": list(SIMPLE_BASELINES),
        "all_metrics": list(RAW_METRICS),
        "standardization": {
            "calibration_starts": [PRE_STARTS[0], PRE_STARTS[-1]],
            "center": "arithmetic mean",
            "scale": "sample standard deviation",
            "scale_floor": ("max(abs(pre_mean)*1e-6, 1e-8)"),
            "score": "abs(raw_value - pre_mean) / max(pre_std, floor)",
            "fault_labels_used_for_standardization": False,
        },
        "incremental_value_rule": {
            "primary_must_pass_decision_rule": True,
            "simple_baseline_dominates_when": (
                "baseline passes the same decision rule, positive-pair "
                "count >= primary, standardized mean >= primary, and at "
                "least one comparison is strict"
            ),
            "candidate_advances_only_if_not_dominated": True,
        },
        "overlapping_windows_treated_as_independent_samples": False,
        "pair_is_statistical_unit": True,
        "model_or_encoder_parameters_updated": 0,
        "detector_thresholds_fitted": 0,
        "source_hashes": {
            "stage26_episodes": sha256_file(args.stage26_episodes),
            "stage26_report": sha256_file(args.stage26_report),
            "stage26_frozen_gate": sha256_file(args.stage26_gate),
            "stage13_manifest": sha256_file(args.stage13_manifest),
            "stage13_report": sha256_file(args.stage13_report),
            "stage14_report": sha256_file(args.stage14_report),
        },
        "stage26_contract": {
            "formal_pairs": stage26_report["scope"]["formal_discovery_pairs"],
            "action_interface_mismatch_steps_total": stage26_report["scope"]["action_interface_mismatch_steps_total"],
            "world_model_scores_computed": stage26_report["scope"]["world_model_scores_computed"],
        },
    }


def feature_path(output_dir: Path, row: dict[str, str]) -> Path:
    return output_dir / "features" / f"{row['pair_id']}-{row['condition']}.safetensors"


def expected_feature_metadata(
    row: dict[str, str],
    *,
    protocol_sha256: str,
    pixel_sha256: str | None = None,
    encoder_batch_size: int = ENCODER_BATCH_SIZE,
) -> dict[str, str]:
    metadata = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "pair_id": row["pair_id"],
        "condition": row["condition"],
        "benchmark_task_id": str(int(row["benchmark_task_id"])),
        "init_state_index": str(int(row["init_state_index"])),
        "steps": str(int(row["steps"])),
        "source_archive_sha256": row["archive_sha256"],
        "source_protocol_sha256": row["protocol_sha256"],
        "stage27_protocol_sha256": protocol_sha256,
        "encoder_model_id": STAGE12.MODEL_ID,
        "encoder_revision": STAGE12.MODEL_REVISION,
        "encoder_batch_size": str(encoder_batch_size),
        "image_transform": "flip height and width before DINOv2",
        "state_layout": "eef_pos(3),eef_axisangle(3),gripper_qpos(2)",
    }
    if pixel_sha256 is not None:
        metadata["policy_pixels_sha256"] = pixel_sha256
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
    protocol_sha256: str,
    encoder_batch_size: int = ENCODER_BATCH_SIZE,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors = load_file(path)
    expected_shapes = {
        "visual_tokens": (
            LIVE_STEPS + 1,
            2,
            16,
            STAGE14.LATENT_DIMENSION,
        ),
        "observation_state": (
            LIVE_STEPS + 1,
            STAGE14.STATE_DIMENSION,
        ),
        "commanded_action": (LIVE_STEPS, STAGE14.ACTION_DIMENSION),
    }
    if set(tensors) != set(expected_shapes):
        raise AssertionError(f"Unexpected Stage 27 feature keys: {set(tensors)}")
    expected_dtypes = {
        "visual_tokens": torch.float16,
        "observation_state": torch.float32,
        "commanded_action": torch.float32,
    }
    for key, shape in expected_shapes.items():
        if tuple(tensors[key].shape) != shape:
            raise AssertionError(f"Unexpected {key} shape: {tensors[key].shape}")
        if tensors[key].dtype != expected_dtypes[key]:
            raise AssertionError(f"Unexpected {key} dtype")
        if not bool(torch.isfinite(tensors[key]).all()):
            raise AssertionError(f"Non-finite Stage 27 tensor: {key}")
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
    if metadata is None:
        raise AssertionError("Missing Stage 27 feature metadata")
    for key, value in expected_feature_metadata(
        row,
        protocol_sha256=protocol_sha256,
        encoder_batch_size=encoder_batch_size,
    ).items():
        if metadata.get(key) != value:
            raise AssertionError(f"Stage 27 feature metadata mismatch: {key}")
    if len(metadata.get("policy_pixels_sha256", "")) != 64:
        raise AssertionError("Missing Stage 27 policy pixel hash")
    return tensors, metadata


def encode_feature_shard(
    row: dict[str, str],
    *,
    args: argparse.Namespace,
    protocol_sha256: str,
    processor: Any,
    encoder: torch.nn.Module,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, str], bool]:
    path = feature_path(args.output_dir, row)
    if path.exists():
        tensors, metadata = validate_feature(
            path,
            row,
            protocol_sha256=protocol_sha256,
            encoder_batch_size=args.encoder_batch_size,
        )
        return tensors, metadata, True
    source_path = episode_archive(
        args.stage26_output_dir,
        row["pair_id"],
        row["condition"],
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Missing Stage 26 archive: {source_path}")
    if sha256_file(source_path) != row["archive_sha256"]:
        raise AssertionError(f"Stage 26 archive hash changed: {source_path}")
    arrays = source_arrays(source_path)
    validate_arrays(arrays, condition=row["condition"])
    images = policy_images(arrays)
    pixel_sha256 = sha256_bytes(images.numpy().tobytes())
    tensors = {
        "visual_tokens": STAGE13.encode_images(
            images,
            processor,
            encoder,
            device=device,
            batch_size=args.encoder_batch_size,
        ),
        "observation_state": policy_state(arrays),
        "commanded_action": torch.from_numpy(arrays["commanded_action"]).float(),
    }
    metadata = expected_feature_metadata(
        row,
        protocol_sha256=protocol_sha256,
        pixel_sha256=pixel_sha256,
        encoder_batch_size=args.encoder_batch_size,
    )
    atomic_save_feature(path, tensors, metadata)
    tensors, metadata = validate_feature(
        path,
        row,
        protocol_sha256=protocol_sha256,
        encoder_batch_size=args.encoder_batch_size,
    )
    return tensors, metadata, False


def validate_starts(starts: tuple[int, ...]) -> torch.Tensor:
    selected = torch.tensor(starts, dtype=torch.int64)
    if any(start < 0 or start + HORIZON > LIVE_STEPS for start in selected.tolist()):
        raise ValueError("Stage 27 H10 start is outside the episode")
    return selected


def raw_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalization: Any,
) -> torch.Tensor:
    prediction_raw = STAGE14.raw_latent(prediction, normalization)
    target_raw = STAGE14.raw_latent(target, normalization)
    return (prediction_raw - target_raw).square().mean(dim=(1, 2))


def score_wm_windows(
    tensors: dict[str, torch.Tensor],
    models: dict[str, torch.nn.Module],
    normalization: Any,
    *,
    starts: tuple[int, ...],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    selected = validate_starts(starts)
    collected = {
        "wm_action_residual": [],
        "no_action_residual": [],
        "latent_persistence_motion": [],
    }
    with torch.inference_mode():
        for offset in range(0, len(selected), batch_size):
            batch_starts = selected[offset : offset + batch_size]
            initial_raw = tensors["visual_tokens"][batch_starts].reshape(
                -1,
                STAGE14.TOKENS_PER_FRAME,
                STAGE14.LATENT_DIMENSION,
            )
            initial_raw = initial_raw.to(
                device=device,
                dtype=torch.float32,
            )
            initial = (initial_raw - normalization.latent_mean) / normalization.latent_std
            action_prediction = initial.clone()
            no_action_prediction = initial.clone()
            for step in range(HORIZON):
                indices = batch_starts + step
                state = tensors["observation_state"][indices].to(device)
                state = (state - normalization.state_mean) / normalization.state_std
                action = tensors["commanded_action"][indices].to(device)
                action = (action - normalization.action_mean) / normalization.action_std
                action_prediction = models["action_conditioned"](
                    action_prediction,
                    state,
                    action,
                )
                no_action_prediction = models["no_action"](
                    no_action_prediction,
                    state,
                    torch.zeros_like(action),
                )
            target_raw = tensors["visual_tokens"][batch_starts + HORIZON].reshape(
                -1,
                STAGE14.TOKENS_PER_FRAME,
                STAGE14.LATENT_DIMENSION,
            )
            target_raw = target_raw.to(
                device=device,
                dtype=torch.float32,
            )
            target = (target_raw - normalization.latent_mean) / normalization.latent_std
            collected["wm_action_residual"].append(raw_mse(action_prediction, target, normalization).cpu().numpy())
            collected["no_action_residual"].append(raw_mse(no_action_prediction, target, normalization).cpu().numpy())
            collected["latent_persistence_motion"].append(raw_mse(initial, target, normalization).cpu().numpy())
    result = {key: np.concatenate(values) for key, values in collected.items()}
    result[PRIMARY_METRIC] = result["wm_action_residual"] - result["no_action_residual"]
    return result


def score_simple_windows(
    arrays: dict[str, np.ndarray],
    *,
    starts: tuple[int, ...],
) -> dict[str, np.ndarray]:
    eef_values = []
    pixel_values = []
    for start in starts:
        target = start + HORIZON
        eef_values.append(
            np.linalg.norm(
                np.diff(
                    arrays["eef_pos"][start : target + 1],
                    axis=0,
                ),
                axis=1,
            ).sum()
        )
        camera_mae = [
            np.abs(arrays[camera][target].astype(np.float32) - arrays[camera][start].astype(np.float32)).mean()
            for camera in ("camera1", "camera2")
        ]
        pixel_values.append(np.mean(camera_mae))
    return {
        "eef_path_length": np.asarray(eef_values, dtype=np.float64),
        "pixel_endpoint_mae": np.asarray(
            pixel_values,
            dtype=np.float64,
        ),
    }


def standardize_anomaly(
    raw_values: np.ndarray,
    *,
    pre_count: int = len(PRE_STARTS),
) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim != 1 or len(values) <= pre_count:
        raise ValueError("Stage 27 raw values do not cover pre and influenced windows")
    pre = values[:pre_count]
    center = float(pre.mean())
    scale = float(pre.std(ddof=1))
    floor = max(
        abs(center) * STANDARDIZATION_RELATIVE_FLOOR,
        STANDARDIZATION_ABSOLUTE_FLOOR,
    )
    denominator = max(scale, floor)
    anomaly = np.abs(values - center) / denominator
    return anomaly, {
        "pre_mean": center,
        "pre_sample_std": scale,
        "scale_floor": floor,
        "denominator": denominator,
    }


def pair_effect_rows(
    episode_scores: dict[
        tuple[str, str],
        dict[str, np.ndarray],
    ],
    episode_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    source = {row["pair_id"]: row for row in episode_rows if row["condition"] == "nominal"}
    rows = []
    pre_count = len(PRE_STARTS)
    for pair_id in sorted(source):
        nominal = episode_scores[(pair_id, "nominal")]
        fault = episode_scores[(pair_id, FAULT_CONDITION)]
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "benchmark_task_id": int(source[pair_id]["benchmark_task_id"]),
            "init_state_index": int(source[pair_id]["init_state_index"]),
        }
        for metric in RAW_METRICS:
            if not np.array_equal(
                nominal[metric][:pre_count],
                fault[metric][:pre_count],
            ):
                raise AssertionError(f"Pre-window raw metric differs: {pair_id} {metric}")
            nominal_anomaly, _ = standardize_anomaly(nominal[metric])
            fault_anomaly, _ = standardize_anomaly(fault[metric])
            nominal_mean = float(nominal_anomaly[pre_count:].mean())
            fault_mean = float(fault_anomaly[pre_count:].mean())
            row[f"{metric}_nominal_influenced_anomaly_mean"] = nominal_mean
            row[f"{metric}_fault_influenced_anomaly_mean"] = fault_mean
            row[f"{metric}_paired_effect"] = fault_mean - nominal_mean
        rows.append(row)
    return rows


def metric_statistics(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    results = {}
    for index, metric in enumerate(RAW_METRICS):
        values = np.asarray(
            [row[f"{metric}_paired_effect"] for row in rows],
            dtype=np.float64,
        )
        results[metric] = paired_statistics(
            values,
            seed=BOOTSTRAP_SEED + index,
            alternative="greater",
        )
    return results


def holm_adjust(
    p_values: dict[str, float],
) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def passes_pair_gate(statistics: dict[str, Any]) -> bool:
    return (
        float(statistics["mean"]) > 0
        and int(statistics["positive_pairs"]) >= 5
        and float(statistics["exact_sign_flip_p"]) <= 0.05
    )


def baseline_dominates(
    primary: dict[str, Any],
    baseline: dict[str, Any],
) -> bool:
    positive_not_worse = int(baseline["positive_pairs"]) >= int(primary["positive_pairs"])
    mean_not_worse = float(baseline["mean"]) >= float(primary["mean"])
    strictly_better = int(baseline["positive_pairs"]) > int(primary["positive_pairs"]) or float(
        baseline["mean"]
    ) > float(primary["mean"])
    return passes_pair_gate(baseline) and positive_not_worse and mean_not_worse and strictly_better


def result_svg(
    statistics: dict[str, dict[str, Any]],
    conclusion: dict[str, Any],
) -> str:
    primary = statistics[PRIMARY_METRIC]
    no_action = statistics["no_action_residual"]
    pixel = statistics["pixel_endpoint_mae"]
    status = (
        "主假设拒绝；no-action 仅冻结为新数据探索性候选"
        if conclusion["exploratory_candidate_can_be_frozen_for_new_data"]
        else "主假设拒绝；不产生后续候选"
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="600" viewBox="0 0 1200 600">
<rect width="1200" height="600" fill="#f7f8fa"/>
<text x="65" y="65" font-size="31" font-family="sans-serif" font-weight="700" fill="#162033">Stage 27：隐藏动力学 Observation Residual</text>
<text x="65" y="105" font-size="18" font-family="sans-serif" fill="#526070">6-pair discovery；H10 starts 41–50；动作逐位相同；未访问留出确认状态</text>
<rect x="65" y="145" width="1070" height="145" rx="14" fill="#ffffff" stroke="#dce1e8"/>
<text x="95" y="190" font-size="20" font-family="sans-serif" font-weight="700" fill="#162033">WM relative residual anomaly</text>
<text x="95" y="245" font-size="36" font-family="monospace" fill="#e36b4f">{primary["positive_pairs"]}/6 positive · mean {primary["mean"]:+.3f}σ · p={primary["exact_sign_flip_p"]:.5f}</text>
<rect x="65" y="325" width="500" height="145" rx="14" fill="#ffffff" stroke="#dce1e8"/>
<text x="95" y="370" font-size="20" font-family="sans-serif" font-weight="700" fill="#162033">Exploratory no-action latent residual</text>
<text x="95" y="415" font-size="27" font-family="monospace" fill="#d99a31">{no_action["positive_pairs"]}/6 · mean {no_action["mean"]:+.3f}σ</text>
<text x="95" y="450" font-size="16" font-family="sans-serif" fill="#6a7583">raw p={no_action["exact_sign_flip_p"]:.5f} · Holm p={conclusion["exploratory_best_metric_holm_p"]:.5f}</text>
<rect x="635" y="325" width="500" height="145" rx="14" fill="#ffffff" stroke="#dce1e8"/>
<text x="665" y="370" font-size="20" font-family="sans-serif" font-weight="700" fill="#162033">Hand-crafted pixel baseline</text>
<text x="665" y="425" font-size="27" font-family="monospace" fill="#5c78c7">{pixel["positive_pairs"]}/6 · mean {pixel["mean"]:+.3f}σ</text>
<text x="65" y="535" font-size="21" font-family="sans-serif" font-weight="700" fill="#162033">Gate：{status}</text>
<text x="65" y="570" font-size="16" font-family="sans-serif" fill="#6a7583">Discovery 结果不能当作独立 test，也不支持 online shield 结论。</text>
</svg>
"""


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

    stage26_report = json.loads(args.stage26_report.read_text(encoding="utf-8"))
    stage26_gate = json.loads(args.stage26_gate.read_text(encoding="utf-8"))
    if stage26_report["status"] != "passed":
        raise AssertionError("Stage 26 did not pass")
    if not stage26_report["stage27_readiness"]["observation_residual_discovery_can_start"]:
        raise AssertionError("Stage 26 did not authorize Stage 27")
    if stage26_gate["selected_fault_conditions"] != [FAULT_CONDITION]:
        raise AssertionError("Stage 26 selected condition changed")
    if stage26_gate["world_model_scores_viewed"]:
        raise AssertionError("Stage 26 gate was not score blind")
    if stage26_report["scope"]["world_model_scores_computed"] != 0:
        raise AssertionError("Stage 26 already viewed WM scores")

    rows = selected_episode_rows(read_csv(args.stage26_episodes))
    protocol = protocol_payload(
        args=args,
        stage26_report=stage26_report,
        rows=rows,
    )
    protocol_text = canonical_json(protocol)
    protocol_sha256 = sha256_text(protocol_text)
    write_deterministic(
        args.output_dir / "protocol.json",
        protocol_text,
        overwrite=False,
    )

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
    for index, row in enumerate(rows, start=1):
        tensors, metadata, was_resumed = encode_feature_shard(
            row,
            args=args,
            protocol_sha256=protocol_sha256,
            processor=processor,
            encoder=encoder,
            device=device,
        )
        path = feature_path(args.output_dir, row)
        encoded += int(not was_resumed)
        resumed += int(was_resumed)
        feature_rows.append(
            {
                "pair_id": row["pair_id"],
                "condition": row["condition"],
                "benchmark_task_id": int(row["benchmark_task_id"]),
                "init_state_index": int(row["init_state_index"]),
                "steps": int(row["steps"]),
                "source_archive_sha256": row["archive_sha256"],
                "policy_pixels_sha256": metadata["policy_pixels_sha256"],
                "feature_file": (f"outputs/wm/stage27_hidden_dynamics_residual/features/{path.name}"),
                "feature_bytes": path.stat().st_size,
                "feature_sha256": sha256_file(path),
            }
        )
        print(
            f"[{index}/{len(rows)}] {row['pair_id']} {row['condition']} encoded={encoded} resumed={resumed}",
            flush=True,
        )

    recheck_row = rows[0]
    cached, cached_metadata = validate_feature(
        feature_path(args.output_dir, recheck_row),
        recheck_row,
        protocol_sha256=protocol_sha256,
        encoder_batch_size=args.encoder_batch_size,
    )
    recheck_arrays = source_arrays(
        episode_archive(
            args.stage26_output_dir,
            recheck_row["pair_id"],
            recheck_row["condition"],
        )
    )
    recheck_images = policy_images(recheck_arrays)
    if sha256_bytes(recheck_images.numpy().tobytes()) != cached_metadata["policy_pixels_sha256"]:
        raise AssertionError("Stage 27 deterministic pixel recheck failed")
    recheck_tokens = STAGE13.encode_images(
        recheck_images,
        processor,
        encoder,
        device=device,
        batch_size=args.encoder_batch_size,
    )
    if not torch.equal(recheck_tokens, cached["visual_tokens"]):
        raise AssertionError("Stage 27 deterministic token recheck failed")
    encoding_finished = time.perf_counter()
    encoder.to("cpu")
    del (
        encoder,
        processor,
        cached,
        recheck_arrays,
        recheck_images,
        recheck_tokens,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stage13_report = json.loads(args.stage13_report.read_text(encoding="utf-8"))
    if stage13_report["dataset"]["test_split_cached_episodes"] != 0:
        raise AssertionError("Stage 13 report indicates expert test leakage")
    normalization = normalization_from_training(
        args.stage13_manifest,
        repo_root=Path.cwd(),
    ).to(device)
    stage14_report = json.loads(args.stage14_report.read_text(encoding="utf-8"))
    if stage14_report["data"]["test_episodes_used"] != 0:
        raise AssertionError("Stage 14 report indicates expert test leakage")
    models = load_dynamics(
        stage14_report,
        args.checkpoint_dir,
        device,
    )

    episode_scores = {}
    window_rows = []
    calibration_rows = []
    pre_count = len(PRE_STARTS)
    for row in rows:
        tensors, _ = validate_feature(
            feature_path(args.output_dir, row),
            row,
            protocol_sha256=protocol_sha256,
            encoder_batch_size=args.encoder_batch_size,
        )
        source_path = episode_archive(
            args.stage26_output_dir,
            row["pair_id"],
            row["condition"],
        )
        arrays = source_arrays(source_path)
        validate_arrays(arrays, condition=row["condition"])
        scores = score_wm_windows(
            tensors,
            models,
            normalization,
            starts=ALL_STARTS,
            device=device,
            batch_size=args.eval_batch_size,
        )
        scores.update(score_simple_windows(arrays, starts=ALL_STARTS))
        if set(scores) != set(RAW_METRICS):
            raise AssertionError("Stage 27 metric set changed")
        episode_scores[(row["pair_id"], row["condition"])] = scores
        standardized = {}
        for metric in RAW_METRICS:
            anomaly, calibration = standardize_anomaly(scores[metric])
            standardized[metric] = anomaly
            calibration_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "condition": row["condition"],
                    "benchmark_task_id": int(row["benchmark_task_id"]),
                    "init_state_index": int(row["init_state_index"]),
                    "metric": metric,
                    **calibration,
                }
            )
        for position, start in enumerate(ALL_STARTS):
            window_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "condition": row["condition"],
                    "benchmark_task_id": int(row["benchmark_task_id"]),
                    "init_state_index": int(row["init_state_index"]),
                    "period": ("pre" if position < pre_count else "fault_influenced"),
                    "start_index": start,
                    "target_index": start + HORIZON,
                    **{f"{metric}_raw": float(scores[metric][position]) for metric in RAW_METRICS},
                    **{f"{metric}_anomaly": float(standardized[metric][position]) for metric in RAW_METRICS},
                }
            )
        print(
            f"scored {row['pair_id']} {row['condition']}",
            flush=True,
        )

    pair_rows = pair_effect_rows(episode_scores, rows)
    statistics = metric_statistics(pair_rows)
    primary = statistics[PRIMARY_METRIC]
    primary_passed = passes_pair_gate(primary)
    diagnostic_metrics = [metric for metric in RAW_METRICS if metric != PRIMARY_METRIC]
    diagnostic_holm = holm_adjust(
        {metric: float(statistics[metric]["exact_sign_flip_p"]) for metric in diagnostic_metrics}
    )
    exploratory_best = min(
        diagnostic_metrics,
        key=lambda metric: (
            float(statistics[metric]["exact_sign_flip_p"]),
            -int(statistics[metric]["positive_pairs"]),
            -float(statistics[metric]["mean"]),
            metric,
        ),
    )
    dominating = [metric for metric in SIMPLE_BASELINES if baseline_dominates(primary, statistics[metric])]
    conclusion = {
        "primary_pair_gate_passed": primary_passed,
        "primary_positive_pairs": primary["positive_pairs"],
        "simple_baselines_passing_same_gate": [
            metric for metric in SIMPLE_BASELINES if passes_pair_gate(statistics[metric])
        ],
        "simple_baselines_dominating_primary": dominating,
        "incremental_value_not_dominated": not dominating,
        "wm_candidate_advances_to_confirmation": (primary_passed and not dominating),
        "diagnostic_family_holm_adjusted_p": diagnostic_holm,
        "exploratory_best_metric": exploratory_best,
        "exploratory_best_metric_raw_p": statistics[exploratory_best]["exact_sign_flip_p"],
        "exploratory_best_metric_holm_p": diagnostic_holm[exploratory_best],
        "exploratory_best_survives_holm_0_05": (diagnostic_holm[exploratory_best] <= 0.05),
        "exploratory_candidate_can_be_frozen_for_new_data": (
            passes_pair_gate(statistics[exploratory_best]) and int(statistics[exploratory_best]["positive_pairs"]) == 6
        ),
        "detector_threshold_fitted": False,
        "online_shield_justified": False,
    }
    scored_finished = time.perf_counter()
    peak_memory = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None

    feature_text = csv_text(feature_rows)
    calibration_text = csv_text(calibration_rows)
    window_text = csv_text(window_rows)
    pair_text = csv_text(pair_rows)
    statistics_payload = {
        "schema_version": 1,
        "statistical_unit": "initial-state pair",
        "overlapping_windows_treated_as_independent_samples": False,
        "primary_metric": PRIMARY_METRIC,
        "primary_decision_rule": protocol["primary_metric"]["decision_rule"],
        "metrics": statistics,
        "conclusion": conclusion,
    }
    statistics_text = canonical_json(statistics_payload)
    report = {
        "schema_version": 1,
        "stage": "WM Stage 27 hidden dynamics observation residual discovery",
        "status": "passed",
        "scope": {
            "pairs": 6,
            "episodes": 12,
            "tasks": [4, 7, 8],
            "conditions": list(CONDITIONS),
            "windows_per_episode": len(ALL_STARTS),
            "pre_windows_per_episode": len(PRE_STARTS),
            "influenced_windows_per_episode": len(INFLUENCED_STARTS),
            "window_rows": len(window_rows),
            "model_parameters_updated": 0,
            "detector_thresholds_fitted": 0,
            "stage25_held_out_test_states_accessed": False,
        },
        "protocol": protocol,
        "frozen_components": {
            "encoder_model_id": STAGE12.MODEL_ID,
            "encoder_revision": STAGE12.MODEL_REVISION,
            "stage14_report_sha256": sha256_file(args.stage14_report),
            "action_conditioned_checkpoint_sha256": (
                stage14_report["variants"]["action_conditioned"]["checkpoint_sha256"]
            ),
            "no_action_checkpoint_sha256": (stage14_report["variants"]["no_action"]["checkpoint_sha256"]),
            "normalization_source": "Stage 13 train split only",
        },
        "negative_controls": {
            "all_raw_pre_metrics_pairwise_bit_exact": True,
            "all_influenced_policy_commands_pairwise_bit_exact": all(
                row["first_post_h10_command_sha256"]
                == next(
                    other["first_post_h10_command_sha256"]
                    for other in rows
                    if other["pair_id"] == row["pair_id"] and other["condition"] != row["condition"]
                )
                for row in rows
            ),
            "action_interface_mismatch_steps_total": sum(int(row["action_interface_mismatch_steps"]) for row in rows),
            "stage26_world_model_scores_computed": (stage26_report["scope"]["world_model_scores_computed"]),
            "stage25_held_out_test_states_accessed": False,
        },
        "statistics": statistics,
        "conclusion": conclusion,
        "artifacts": {
            "protocol": "protocol.json",
            "protocol_sha256": sha256_text(protocol_text),
            "feature_manifest": "feature_manifest.csv",
            "feature_manifest_sha256": sha256_text(feature_text),
            "calibration": "online_calibration.csv",
            "calibration_sha256": sha256_text(calibration_text),
            "window_scores": "window_scores.csv",
            "window_scores_sha256": sha256_text(window_text),
            "pair_effects": "pair_effects.csv",
            "pair_effects_sha256": sha256_text(pair_text),
            "statistics": "statistics.json",
            "statistics_sha256": sha256_text(statistics_text),
            "feature_cache": ("local-only under outputs/wm/stage27_hidden_dynamics_residual/features"),
        },
        "determinism": {
            "encoder_batch_size": ENCODER_BATCH_SIZE,
            "deterministic_recheck_pair_id": recheck_row["pair_id"],
            "deterministic_recheck_condition": recheck_row["condition"],
            "deterministic_recheck_bit_exact": True,
        },
        "runtime": {
            "device": str(device),
            "encoded_feature_shards": encoded,
            "resumed_feature_shards": resumed,
            "encoder_load_seconds": encoder_loaded - started,
            "feature_encoding_and_recheck_seconds": (encoding_finished - encoder_loaded),
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
            "All six pairs are discovery data and cannot serve as an independent test.",
            "The hidden gain shift is synthetic and its onset is known to the evaluator.",
            "Recursive WM scoring receives oracle robot state at every rollout step.",
            "Within-episode pre standardization assumes a clean startup period.",
            "Overlapping H10 windows are summarized within pair and are not treated as independent samples.",
            "No detector threshold, intervention policy, recovery controller, or shield is fitted.",
        ],
        "stage28_readiness": {
            "preregistered_primary_confirmation_justified": conclusion["wm_candidate_advances_to_confirmation"],
            "exploratory_candidate_freeze_for_new_confirmation_data_justified": conclusion[
                "exploratory_candidate_can_be_frozen_for_new_data"
            ],
            "exploratory_candidate": conclusion["exploratory_best_metric"],
            "must_freeze_candidate_before_collecting_test": True,
            "online_shield_claim_justified": False,
        },
    }
    report_text = canonical_json(report)
    for filename, content in (
        ("protocol.json", protocol_text),
        ("feature_manifest.csv", feature_text),
        ("online_calibration.csv", calibration_text),
        ("window_scores.csv", window_text),
        ("pair_effects.csv", pair_text),
        ("statistics.json", statistics_text),
        ("report.json", report_text),
    ):
        write_deterministic(
            args.public_results_dir / filename,
            content,
            overwrite=args.overwrite_public,
        )
    write_deterministic(
        args.media_path,
        result_svg(statistics, conclusion),
        overwrite=args.overwrite_public,
    )
    print(
        "Stage 27 complete: "
        f"primary={primary['positive_pairs']}/6, "
        f"mean={primary['mean']:+.4f}, "
        f"p={primary['exact_sign_flip_p']:.5f}, "
        "advances="
        f"{conclusion['wm_candidate_advances_to_confirmation']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
