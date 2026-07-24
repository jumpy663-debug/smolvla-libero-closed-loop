#!/usr/bin/env python
# ruff: noqa: E501

"""Confirm the frozen no-action residual on new hidden-dynamics rollouts.

The candidate and six Stage 25 test states are written to disk before loading
SmolVLA or collecting any new trajectory. Each state receives a nominal and
0.5x arm-actuator-gain rollout. The no-action H10 latent residual is the only
confirmatory metric; persistence, EEF, and pixel motion remain descriptive
comparators. No threshold, recovery controller, or shield is fitted.
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
import stage26_hidden_dynamics_cohort as STAGE26
import stage27_hidden_dynamics_residual as STAGE27
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from stage17_failure_signal import (
    STAGE12,
    STAGE13,
    STAGE14,
    canonical_json,
    csv_text,
    normalization_from_training,
    policy_images,
    policy_state,
    sha256_bytes,
    sha256_file,
    source_arrays,
    write_deterministic,
)
from stage23_confirmatory_wm_scoring import paired_statistics
from transformers import AutoImageProcessor, Dinov2Model

TASK_IDS = (4, 7, 8)
EXPECTED_CONFIRMATION = {
    4: (6, 9),
    7: (8, 11),
    8: (10, 23),
}
CONDITIONS = ("nominal", "arm_actuator_gain_0p5")
FAULT_CONDITION = CONDITIONS[1]
HORIZON = STAGE27.HORIZON
PRE_STARTS = STAGE27.PRE_STARTS
INFLUENCED_STARTS = STAGE27.INFLUENCED_STARTS
ALL_STARTS = STAGE27.ALL_STARTS
PRIMARY_METRIC = "no_action_residual"
COMPARATOR_METRICS = (
    "latent_persistence_motion",
    "eef_path_length",
    "pixel_endpoint_mae",
)
RAW_METRICS = (PRIMARY_METRIC, *COMPARATOR_METRICS)
ENCODER_BATCH_SIZE = STAGE27.ENCODER_BATCH_SIZE
EVAL_BATCH_SIZE = STAGE27.EVAL_BATCH_SIZE
BOOTSTRAP_SEED = 2828
FEATURE_SCHEMA_VERSION = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage25-protocol",
        type=Path,
        default=Path("results/wm/stage25_detector_calibration/protocol.json"),
    )
    parser.add_argument(
        "--stage27-report",
        type=Path,
        default=Path("results/wm/stage27_hidden_dynamics_residual/report.json"),
    )
    parser.add_argument(
        "--stage27-statistics",
        type=Path,
        default=Path("results/wm/stage27_hidden_dynamics_residual/statistics.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/reproduction/smolvla_libero/stage2_checkpoint/checkpoint"),
    )
    parser.add_argument(
        "--stage2-manifest",
        type=Path,
        default=Path("outputs/reproduction/smolvla_libero/stage2_checkpoint/manifest.json"),
    )
    parser.add_argument(
        "--backbone-dir",
        type=Path,
        default=Path("outputs/reproduction/smolvla_libero/stage3_single_inference/backbone"),
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
        "--dynamics-checkpoint-dir",
        type=Path,
        default=Path("outputs/wm/stage14_next_latent_baseline"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wm/stage28_confirm_hidden_dynamics_residual"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage28_confirm_hidden_dynamics_residual"),
    )
    parser.add_argument(
        "--media-path",
        type=Path,
        default=Path("media/wm_stage28_confirm_hidden_dynamics_residual.svg"),
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
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--overwrite-public", action="store_true")
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def confirmation_pairs(
    stage25_protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in stage25_protocol["pairs"] if row["split"] == "test"]
    observed = {
        task_id: tuple(int(row["init_state_index"]) for row in rows if int(row["benchmark_task_id"]) == task_id)
        for task_id in TASK_IDS
    }
    if observed != EXPECTED_CONFIRMATION:
        raise AssertionError(f"Stage 28 confirmation states changed: {observed}")
    if len(rows) != 6:
        raise AssertionError("Stage 28 requires six confirmation pairs")
    for row in rows:
        row["stage28_pair_id"] = f"task{int(row['benchmark_task_id']):02d}-init{int(row['init_state_index']):02d}"
        row["stage26_pair_id"] = row["stage28_pair_id"]
        row["pilot_pair"] = False
    return rows


def frozen_candidate_payload(
    *,
    args: argparse.Namespace,
    pairs: list[dict[str, Any]],
    stage27_report: dict[str, Any],
) -> dict[str, Any]:
    discovery = stage27_report["statistics"][PRIMARY_METRIC]
    conclusion = stage27_report["conclusion"]
    return {
        "schema_version": 1,
        "candidate": PRIMARY_METRIC,
        "selection_status": (
            "exploratory candidate selected after Stage 27 discovery; not a preregistered Stage 27 success"
        ),
        "stage27_preregistered_primary_failed": True,
        "stage27_exploratory_result": {
            "positive_pairs": discovery["positive_pairs"],
            "pairs": discovery["pairs"],
            "standardized_mean": discovery["mean"],
            "raw_exact_sign_flip_p": discovery["exact_sign_flip_p"],
            "diagnostic_family_holm_p": conclusion["exploratory_best_metric_holm_p"],
            "survived_holm_0_05": conclusion["exploratory_best_survives_holm_0_05"],
        },
        "raw_definition": (
            "frozen no-action TokenDynamicsModel recursive H10 raw "
            "DINO latent MSE; oracle robot state supplied at every step; "
            "action input fixed to normalized zero"
        ),
        "online_standardization": {
            "pre_starts": [PRE_STARTS[0], PRE_STARTS[-1]],
            "center": "pre arithmetic mean",
            "scale": "pre sample standard deviation",
            "floor": "max(abs(pre_mean)*1e-6, 1e-8)",
            "anomaly": ("abs(raw - pre_mean) / max(pre_sample_std, floor)"),
        },
        "confirmation_effect": (
            "mean anomaly(fault, influenced starts 41..50) minus mean anomaly(nominal, influenced starts 41..50)"
        ),
        "confirmation_direction": "positive",
        "confirmation_decision_rule": {
            "mean_greater_than_zero": True,
            "at_least_five_of_six_pairs_positive": True,
            "one_sided_exact_sign_flip_p_at_most": 0.05,
        },
        "confirmation_pairs": [
            {
                "pair_id": row["stage28_pair_id"],
                "benchmark_task_id": int(row["benchmark_task_id"]),
                "init_state_index": int(row["init_state_index"]),
            }
            for row in pairs
        ],
        "candidate_frozen_before_new_rollout_collection": True,
        "confirmation_scores_viewed_at_freeze": False,
        "detector_threshold": None,
        "source_hashes": {
            "stage25_protocol": sha256_file(args.stage25_protocol),
            "stage27_report": sha256_file(args.stage27_report),
            "stage27_statistics": sha256_file(args.stage27_statistics),
            "stage14_report": sha256_file(args.stage14_report),
        },
    }


def protocol_payload(
    *,
    args: argparse.Namespace,
    pairs: list[dict[str, Any]],
    candidate_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "WM Stage 28 independent hidden-dynamics residual confirmation",
        "frozen_candidate_sha256": candidate_sha256,
        "pairs": pairs,
        "source_split": "Stage 25 test states",
        "states_used_in_stage26_or_stage27_hidden_dynamics": False,
        "states_previously_used_for_stage25_action_fault_test": True,
        "conditions": list(CONDITIONS),
        "intervention_start_zero_based": STAGE26.INTERVENTION_START,
        "diagnostic_prefix_steps": STAGE26.LIVE_STEPS,
        "horizon": HORIZON,
        "pre_window_starts_inclusive": [
            PRE_STARTS[0],
            PRE_STARTS[-1],
        ],
        "fault_influenced_window_starts_inclusive": [
            INFLUENCED_STARTS[0],
            INFLUENCED_STARTS[-1],
        ],
        "hidden_mutation": {
            "target": ("MuJoCo actuator_gainprm[:,0] for robot0_torq_j1..j7"),
            "operation": ("multiply by 0.5 once before env.step at action step 50"),
        },
        "pairing_contract": {
            "initial_observations_bit_exact": True,
            "pre_observations_bit_exact": True,
            "pre_commands_bit_exact": True,
            "first_post_h10_commands_bit_exact": True,
            "commanded_equals_env_step_action_every_step": True,
        },
        "confirmatory_metric": PRIMARY_METRIC,
        "descriptive_comparators": list(COMPARATOR_METRICS),
        "metric_reselection_after_confirmation": False,
        "detector_thresholds_fitted": 0,
        "model_parameters_updated": 0,
        "shield_actions_applied": 0,
        "source_hashes": {
            "stage25_protocol": sha256_file(args.stage25_protocol),
            "stage27_report": sha256_file(args.stage27_report),
            "stage27_statistics": sha256_file(args.stage27_statistics),
        },
    }


def collection_record_rows(
    records: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for pair_id in sorted(records):
        for condition in CONDITIONS:
            record = records[pair_id][condition]
            rows.append({key: value for key, value in record.items() if key not in {"archive_file", "mutation"}})
    return rows


def pair_contract_rows(
    records: dict[str, dict[str, dict[str, Any]]],
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        pair_id = pair["stage28_pair_id"]
        nominal = records[pair_id]["nominal"]
        fault = records[pair_id][FAULT_CONDITION]
        row = {
            "pair_id": pair_id,
            "benchmark_task_id": int(pair["benchmark_task_id"]),
            "init_state_index": int(pair["init_state_index"]),
            "initial_observations_bit_exact": (
                nominal["initial_observation_sha256"] == fault["initial_observation_sha256"]
            ),
            "preintervention_observations_bit_exact": (
                nominal["pre_observation_sha256"] == fault["pre_observation_sha256"]
            ),
            "preintervention_commands_bit_exact": (
                nominal["pre_commanded_action_sha256"] == fault["pre_commanded_action_sha256"]
            ),
            "first_post_h10_commands_bit_exact": (
                nominal["first_post_h10_command_sha256"] == fault["first_post_h10_command_sha256"]
            ),
            "action_interface_mismatch_steps": (
                int(nominal["action_interface_mismatch_steps"]) + int(fault["action_interface_mismatch_steps"])
            ),
            "fault_mutation_verified": bool(fault["mutation"]["verified"]),
        }
        row["pair_contract_passed"] = (
            row["initial_observations_bit_exact"]
            and row["preintervention_observations_bit_exact"]
            and row["preintervention_commands_bit_exact"]
            and row["first_post_h10_commands_bit_exact"]
            and row["action_interface_mismatch_steps"] == 0
            and row["fault_mutation_verified"]
        )
        rows.append(row)
    return rows


def feature_path(
    output_dir: Path,
    pair_id: str,
    condition: str,
) -> Path:
    return output_dir / "features" / f"{pair_id}-{condition}.safetensors"


def expected_feature_metadata(
    *,
    pair: dict[str, Any],
    condition: str,
    source_record: dict[str, Any],
    protocol_sha256: str,
    pixel_sha256: str | None = None,
) -> dict[str, str]:
    metadata = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "pair_id": pair["stage28_pair_id"],
        "condition": condition,
        "benchmark_task_id": str(int(pair["benchmark_task_id"])),
        "init_state_index": str(int(pair["init_state_index"])),
        "steps": str(STAGE26.LIVE_STEPS),
        "source_archive_sha256": source_record["archive_sha256"],
        "stage28_protocol_sha256": protocol_sha256,
        "encoder_model_id": STAGE12.MODEL_ID,
        "encoder_revision": STAGE12.MODEL_REVISION,
        "encoder_batch_size": str(ENCODER_BATCH_SIZE),
        "image_transform": "flip height and width before DINOv2",
        "state_layout": ("eef_pos(3),eef_axisangle(3),gripper_qpos(2)"),
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
    *,
    pair: dict[str, Any],
    condition: str,
    source_record: dict[str, Any],
    protocol_sha256: str,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors = load_file(path)
    expected_shapes = {
        "visual_tokens": (
            STAGE26.LIVE_STEPS + 1,
            2,
            16,
            STAGE14.LATENT_DIMENSION,
        ),
        "observation_state": (
            STAGE26.LIVE_STEPS + 1,
            STAGE14.STATE_DIMENSION,
        ),
    }
    if set(tensors) != set(expected_shapes):
        raise AssertionError(f"Unexpected Stage 28 feature keys: {set(tensors)}")
    expected_dtypes = {
        "visual_tokens": torch.float16,
        "observation_state": torch.float32,
    }
    for key, shape in expected_shapes.items():
        if tuple(tensors[key].shape) != shape:
            raise AssertionError(f"Unexpected Stage 28 {key} shape")
        if tensors[key].dtype != expected_dtypes[key]:
            raise AssertionError(f"Unexpected Stage 28 {key} dtype")
        if not bool(torch.isfinite(tensors[key]).all()):
            raise AssertionError(f"Non-finite Stage 28 tensor: {key}")
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
    if metadata is None:
        raise AssertionError("Missing Stage 28 feature metadata")
    expected = expected_feature_metadata(
        pair=pair,
        condition=condition,
        source_record=source_record,
        protocol_sha256=protocol_sha256,
    )
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise AssertionError(f"Stage 28 feature metadata mismatch: {key}")
    if len(metadata.get("policy_pixels_sha256", "")) != 64:
        raise AssertionError("Missing Stage 28 pixel hash")
    return tensors, metadata


def encode_feature(
    *,
    args: argparse.Namespace,
    pair: dict[str, Any],
    condition: str,
    source_record: dict[str, Any],
    protocol_sha256: str,
    processor: Any,
    encoder: torch.nn.Module,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, str], bool]:
    path = feature_path(
        args.output_dir,
        pair["stage28_pair_id"],
        condition,
    )
    if path.exists():
        tensors, metadata = validate_feature(
            path,
            pair=pair,
            condition=condition,
            source_record=source_record,
            protocol_sha256=protocol_sha256,
        )
        return tensors, metadata, True
    archive_path, _ = STAGE26.episode_paths(
        args.output_dir,
        pair["stage28_pair_id"],
        condition,
    )
    if sha256_file(archive_path) != source_record["archive_sha256"]:
        raise AssertionError(f"Stage 28 source hash changed: {archive_path}")
    arrays = source_arrays(archive_path)
    STAGE26.validate_arrays(arrays, condition=condition)
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
    }
    metadata = expected_feature_metadata(
        pair=pair,
        condition=condition,
        source_record=source_record,
        protocol_sha256=protocol_sha256,
        pixel_sha256=pixel_sha256,
    )
    atomic_save_feature(path, tensors, metadata)
    tensors, metadata = validate_feature(
        path,
        pair=pair,
        condition=condition,
        source_record=source_record,
        protocol_sha256=protocol_sha256,
    )
    return tensors, metadata, False


def load_no_action_model(
    *,
    stage14_report: dict[str, Any],
    checkpoint_dir: Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = checkpoint_dir / "no_action_best.safetensors"
    expected = stage14_report["variants"]["no_action"]["checkpoint_sha256"]
    if sha256_file(checkpoint) != expected:
        raise AssertionError("Stage 28 no-action checkpoint hash changed")
    model = STAGE14.TokenDynamicsModel(
        int(stage14_report["model"]["hidden_dimension"]),
        int(stage14_report["model"]["depth"]),
    )
    model.load_state_dict(load_file(checkpoint))
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def raw_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalization: Any,
) -> torch.Tensor:
    prediction_raw = STAGE14.raw_latent(prediction, normalization)
    target_raw = STAGE14.raw_latent(target, normalization)
    return (prediction_raw - target_raw).square().mean(dim=(1, 2))


def score_no_action_windows(
    tensors: dict[str, torch.Tensor],
    model: torch.nn.Module,
    normalization: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    selected = torch.tensor(ALL_STARTS, dtype=torch.int64)
    no_action_values = []
    persistence_values = []
    with torch.inference_mode():
        for offset in range(0, len(selected), batch_size):
            starts = selected[offset : offset + batch_size]
            initial_raw = tensors["visual_tokens"][starts].reshape(
                -1,
                STAGE14.TOKENS_PER_FRAME,
                STAGE14.LATENT_DIMENSION,
            )
            initial_raw = initial_raw.to(
                device=device,
                dtype=torch.float32,
            )
            initial = (initial_raw - normalization.latent_mean) / normalization.latent_std
            prediction = initial.clone()
            for step in range(HORIZON):
                indices = starts + step
                state = tensors["observation_state"][indices].to(device)
                state = (state - normalization.state_mean) / normalization.state_std
                action = torch.zeros(
                    (len(indices), STAGE14.ACTION_DIMENSION),
                    device=device,
                )
                prediction = model(prediction, state, action)
            target_raw = tensors["visual_tokens"][starts + HORIZON].reshape(
                -1,
                STAGE14.TOKENS_PER_FRAME,
                STAGE14.LATENT_DIMENSION,
            )
            target_raw = target_raw.to(
                device=device,
                dtype=torch.float32,
            )
            target = (target_raw - normalization.latent_mean) / normalization.latent_std
            no_action_values.append(raw_mse(prediction, target, normalization).cpu().numpy())
            persistence_values.append(raw_mse(initial, target, normalization).cpu().numpy())
    return {
        PRIMARY_METRIC: np.concatenate(no_action_values),
        "latent_persistence_motion": np.concatenate(persistence_values),
    }


def score_simple_windows(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    eef_values = []
    pixel_values = []
    for start in ALL_STARTS:
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
        pixel_values.append(
            np.mean(
                [
                    np.abs(arrays[camera][target].astype(np.float32) - arrays[camera][start].astype(np.float32)).mean()
                    for camera in ("camera1", "camera2")
                ]
            )
        )
    return {
        "eef_path_length": np.asarray(
            eef_values,
            dtype=np.float64,
        ),
        "pixel_endpoint_mae": np.asarray(
            pixel_values,
            dtype=np.float64,
        ),
    }


def pair_effect_rows(
    episode_scores: dict[
        tuple[str, str],
        dict[str, np.ndarray],
    ],
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    pre_count = len(PRE_STARTS)
    for pair in pairs:
        pair_id = pair["stage28_pair_id"]
        nominal = episode_scores[(pair_id, "nominal")]
        fault = episode_scores[(pair_id, FAULT_CONDITION)]
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "benchmark_task_id": int(pair["benchmark_task_id"]),
            "init_state_index": int(pair["init_state_index"]),
        }
        for metric in RAW_METRICS:
            if not np.array_equal(
                nominal[metric][:pre_count],
                fault[metric][:pre_count],
            ):
                raise AssertionError(f"Stage 28 pre score differs: {pair_id} {metric}")
            nominal_anomaly, _ = STAGE27.standardize_anomaly(nominal[metric])
            fault_anomaly, _ = STAGE27.standardize_anomaly(fault[metric])
            nominal_mean = float(nominal_anomaly[pre_count:].mean())
            fault_mean = float(fault_anomaly[pre_count:].mean())
            row[f"{metric}_nominal_influenced_anomaly_mean"] = nominal_mean
            row[f"{metric}_fault_influenced_anomaly_mean"] = fault_mean
            row[f"{metric}_paired_effect"] = fault_mean - nominal_mean
        rows.append(row)
    return rows


def metric_statistics(
    pair_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    results = {}
    for index, metric in enumerate(RAW_METRICS):
        values = np.asarray(
            [row[f"{metric}_paired_effect"] for row in pair_rows],
            dtype=np.float64,
        )
        results[metric] = paired_statistics(
            values,
            seed=BOOTSTRAP_SEED + index,
            alternative="greater",
        )
    return results


def confirmation_passed(statistics: dict[str, Any]) -> bool:
    return (
        float(statistics["mean"]) > 0
        and int(statistics["positive_pairs"]) >= 5
        and float(statistics["exact_sign_flip_p"]) <= 0.05
    )


def result_svg(
    statistics: dict[str, dict[str, Any]],
    passed: bool,
) -> str:
    primary = statistics[PRIMARY_METRIC]
    persistence = statistics["latent_persistence_motion"]
    pixel = statistics["pixel_endpoint_mae"]
    status = "独立确认通过" if passed else "独立确认失败"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="600" viewBox="0 0 1200 600">
<rect width="1200" height="600" fill="#f7f8fa"/>
<text x="65" y="65" font-size="31" font-family="sans-serif" font-weight="700" fill="#162033">Stage 28：No-action Residual 独立确认</text>
<text x="65" y="105" font-size="18" font-family="sans-serif" fill="#526070">6 个新 hidden-gain pair；候选在采集前冻结；未重选指标或拟合 threshold</text>
<rect x="65" y="145" width="1070" height="165" rx="14" fill="#ffffff" stroke="#dce1e8"/>
<text x="95" y="190" font-size="20" font-family="sans-serif" font-weight="700" fill="#162033">Frozen no-action H10 latent residual</text>
<text x="95" y="247" font-size="38" font-family="monospace" fill="#e36b4f">{primary["positive_pairs"]}/6 positive · mean {primary["mean"]:+.3f}σ · p={primary["exact_sign_flip_p"]:.5f}</text>
<text x="95" y="285" font-size="17" font-family="sans-serif" fill="#6a7583">Gate: mean &gt; 0 · at least 5/6 positive · exact one-sided p ≤ 0.05</text>
<rect x="65" y="350" width="500" height="125" rx="14" fill="#ffffff" stroke="#dce1e8"/>
<text x="95" y="392" font-size="19" font-family="sans-serif" font-weight="700" fill="#162033">Latent persistence</text>
<text x="95" y="440" font-size="25" font-family="monospace" fill="#d99a31">{persistence["positive_pairs"]}/6 · {persistence["mean"]:+.3f}σ</text>
<rect x="635" y="350" width="500" height="125" rx="14" fill="#ffffff" stroke="#dce1e8"/>
<text x="665" y="392" font-size="19" font-family="sans-serif" font-weight="700" fill="#162033">Pixel endpoint</text>
<text x="665" y="440" font-size="25" font-family="monospace" fill="#5c78c7">{pixel["positive_pairs"]}/6 · {pixel["mean"]:+.3f}σ</text>
<text x="65" y="535" font-size="23" font-family="sans-serif" font-weight="700" fill="#162033">结论：{status}</text>
<text x="65" y="570" font-size="16" font-family="sans-serif" fill="#6a7583">通过也只支持 residual 候选；尚无报警 threshold、在线误报率或 shield 结果。</text>
</svg>
"""


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 28 requires CUDA")
    if args.encoder_batch_size != ENCODER_BATCH_SIZE:
        raise ValueError(f"Encoder batch size is pinned to {ENCODER_BATCH_SIZE}")
    if args.eval_batch_size <= 0:
        raise ValueError("Eval batch size must be positive")
    if args.model_root.name != STAGE12.MODEL_REVISION:
        raise ValueError("DINOv2 model revision is not pinned")

    stage25_protocol = json.loads(args.stage25_protocol.read_text(encoding="utf-8"))
    stage27_report = json.loads(args.stage27_report.read_text(encoding="utf-8"))
    if stage27_report["stage28_readiness"]["exploratory_candidate"] != PRIMARY_METRIC:
        raise AssertionError("Stage 27 candidate changed")
    if not stage27_report["stage28_readiness"]["exploratory_candidate_freeze_for_new_confirmation_data_justified"]:
        raise AssertionError("Stage 27 did not authorize candidate freeze")
    if stage27_report["stage28_readiness"]["preregistered_primary_confirmation_justified"]:
        raise AssertionError("Stage 27 primary failure boundary changed")
    if stage27_report["scope"]["stage25_held_out_test_states_accessed"]:
        raise AssertionError("Stage 27 accessed Stage 25 test states")

    pairs = confirmation_pairs(stage25_protocol)
    candidate = frozen_candidate_payload(
        args=args,
        pairs=pairs,
        stage27_report=stage27_report,
    )
    candidate_text = canonical_json(candidate)
    candidate_sha256 = sha256_text(candidate_text)
    protocol = protocol_payload(
        args=args,
        pairs=pairs,
        candidate_sha256=candidate_sha256,
    )
    protocol_text = canonical_json(protocol)
    protocol_sha256 = sha256_text(protocol_text)
    write_deterministic(
        args.output_dir / "frozen_candidate.json",
        candidate_text,
        overwrite=False,
    )
    write_deterministic(
        args.output_dir / "protocol.json",
        protocol_text,
        overwrite=False,
    )
    print(
        "Stage 28 candidate and protocol frozen before collection",
        flush=True,
    )

    STAGE26.STAGE3.verify_stage2_manifest(args.stage2_manifest)
    config = PreTrainedConfig.from_pretrained(
        args.checkpoint_dir,
        local_files_only=True,
    )
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    config.device = "cuda"
    config.vlm_model_name = str(args.backbone_dir.resolve())
    config.load_vlm_weights = True
    config.empty_cameras = 1
    config.use_amp = False
    config.n_action_steps = STAGE26.STAGE18.ACTION_HORIZON
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint_dir),
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": {}},
            "tokenizer_processor": {
                "tokenizer_name": str(args.backbone_dir.resolve()),
            },
            "device_processor": {"device": "cuda"},
        },
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint_dir,
        config=config,
        local_files_only=True,
    ).eval()
    policy_loaded = time.perf_counter()

    records: dict[str, dict[str, dict[str, Any]]] = {}
    for pair in pairs:
        pair_id = pair["stage28_pair_id"]
        records[pair_id] = {}
        for condition in CONDITIONS:
            records[pair_id][condition] = STAGE26.collect_one(
                args=args,
                pair=pair,
                condition=condition,
                policy=policy,
                config=config,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                protocol_sha256=protocol_sha256,
            )
    collection_finished = time.perf_counter()
    pair_contracts = pair_contract_rows(records, pairs)
    if not all(bool(row["pair_contract_passed"]) for row in pair_contracts):
        raise AssertionError("Stage 28 pair contract failed")
    policy.to("cpu")
    del policy, preprocessor, postprocessor
    torch.cuda.empty_cache()

    device = torch.device("cuda")
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
    for pair in pairs:
        pair_id = pair["stage28_pair_id"]
        for condition in CONDITIONS:
            tensors, metadata, was_resumed = encode_feature(
                args=args,
                pair=pair,
                condition=condition,
                source_record=records[pair_id][condition],
                protocol_sha256=protocol_sha256,
                processor=processor,
                encoder=encoder,
                device=device,
            )
            path = feature_path(
                args.output_dir,
                pair_id,
                condition,
            )
            encoded += int(not was_resumed)
            resumed += int(was_resumed)
            feature_rows.append(
                {
                    "pair_id": pair_id,
                    "condition": condition,
                    "benchmark_task_id": int(pair["benchmark_task_id"]),
                    "init_state_index": int(pair["init_state_index"]),
                    "source_archive_sha256": records[pair_id][condition]["archive_sha256"],
                    "policy_pixels_sha256": metadata["policy_pixels_sha256"],
                    "feature_file": (f"outputs/wm/stage28_confirm_hidden_dynamics_residual/features/{path.name}"),
                    "feature_bytes": path.stat().st_size,
                    "feature_sha256": sha256_file(path),
                }
            )
            print(
                f"encoded {pair_id} {condition} new={encoded} resumed={resumed}",
                flush=True,
            )

    recheck_pair = pairs[0]
    recheck_id = recheck_pair["stage28_pair_id"]
    cached, cached_metadata = validate_feature(
        feature_path(
            args.output_dir,
            recheck_id,
            "nominal",
        ),
        pair=recheck_pair,
        condition="nominal",
        source_record=records[recheck_id]["nominal"],
        protocol_sha256=protocol_sha256,
    )
    recheck_archive, _ = STAGE26.episode_paths(
        args.output_dir,
        recheck_id,
        "nominal",
    )
    recheck_arrays = source_arrays(recheck_archive)
    recheck_images = policy_images(recheck_arrays)
    if sha256_bytes(recheck_images.numpy().tobytes()) != cached_metadata["policy_pixels_sha256"]:
        raise AssertionError("Stage 28 pixel recheck failed")
    recheck_tokens = STAGE13.encode_images(
        recheck_images,
        processor,
        encoder,
        device=device,
        batch_size=args.encoder_batch_size,
    )
    if not torch.equal(recheck_tokens, cached["visual_tokens"]):
        raise AssertionError("Stage 28 token recheck failed")
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
    torch.cuda.empty_cache()

    stage13_report = json.loads(args.stage13_report.read_text(encoding="utf-8"))
    if stage13_report["dataset"]["test_split_cached_episodes"] != 0:
        raise AssertionError("Stage 13 expert test leakage changed")
    normalization = normalization_from_training(
        args.stage13_manifest,
        repo_root=Path.cwd(),
    ).to(device)
    stage14_report = json.loads(args.stage14_report.read_text(encoding="utf-8"))
    if stage14_report["data"]["test_episodes_used"] != 0:
        raise AssertionError("Stage 14 expert test leakage changed")
    model = load_no_action_model(
        stage14_report=stage14_report,
        checkpoint_dir=args.dynamics_checkpoint_dir,
        device=device,
    )

    episode_scores = {}
    calibration_rows = []
    window_rows = []
    pre_count = len(PRE_STARTS)
    for pair in pairs:
        pair_id = pair["stage28_pair_id"]
        for condition in CONDITIONS:
            tensors, _ = validate_feature(
                feature_path(
                    args.output_dir,
                    pair_id,
                    condition,
                ),
                pair=pair,
                condition=condition,
                source_record=records[pair_id][condition],
                protocol_sha256=protocol_sha256,
            )
            archive_path, _ = STAGE26.episode_paths(
                args.output_dir,
                pair_id,
                condition,
            )
            arrays = source_arrays(archive_path)
            STAGE26.validate_arrays(arrays, condition=condition)
            scores = score_no_action_windows(
                tensors,
                model,
                normalization,
                device=device,
                batch_size=args.eval_batch_size,
            )
            scores.update(score_simple_windows(arrays))
            if set(scores) != set(RAW_METRICS):
                raise AssertionError("Stage 28 metric set changed")
            episode_scores[(pair_id, condition)] = scores
            standardized = {}
            for metric in RAW_METRICS:
                anomaly, calibration = STAGE27.standardize_anomaly(scores[metric])
                standardized[metric] = anomaly
                calibration_rows.append(
                    {
                        "pair_id": pair_id,
                        "condition": condition,
                        "benchmark_task_id": int(pair["benchmark_task_id"]),
                        "init_state_index": int(pair["init_state_index"]),
                        "metric": metric,
                        **calibration,
                    }
                )
            for position, start in enumerate(ALL_STARTS):
                window_rows.append(
                    {
                        "pair_id": pair_id,
                        "condition": condition,
                        "benchmark_task_id": int(pair["benchmark_task_id"]),
                        "init_state_index": int(pair["init_state_index"]),
                        "period": ("pre" if position < pre_count else "fault_influenced"),
                        "start_index": start,
                        "target_index": start + HORIZON,
                        **{f"{metric}_raw": float(scores[metric][position]) for metric in RAW_METRICS},
                        **{f"{metric}_anomaly": float(standardized[metric][position]) for metric in RAW_METRICS},
                    }
                )
            print(
                f"scored {pair_id} {condition}",
                flush=True,
            )

    pair_rows = pair_effect_rows(episode_scores, pairs)
    statistics = metric_statistics(pair_rows)
    primary = statistics[PRIMARY_METRIC]
    passed = confirmation_passed(primary)
    scored_finished = time.perf_counter()
    peak_memory = int(torch.cuda.max_memory_allocated())

    collection_text = csv_text(collection_record_rows(records))
    pair_contract_text = csv_text(pair_contracts)
    feature_text = csv_text(feature_rows)
    calibration_text = csv_text(calibration_rows)
    window_text = csv_text(window_rows)
    pair_text = csv_text(pair_rows)
    statistics_payload = {
        "schema_version": 1,
        "statistical_unit": "initial-state pair",
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_selected_before_collection": True,
        "primary_decision_rule": candidate["confirmation_decision_rule"],
        "primary": primary,
        "descriptive_comparators": {metric: statistics[metric] for metric in COMPARATOR_METRICS},
        "confirmation_passed": passed,
        "metric_reselection_performed": False,
    }
    statistics_text = canonical_json(statistics_payload)
    current_runtime = {
        "policy_load_seconds": policy_loaded - started,
        "collection_seconds": collection_finished - policy_loaded,
        "encoder_load_seconds": encoder_loaded - collection_finished,
        "feature_encoding_and_recheck_seconds": (encoding_finished - encoder_loaded),
        "scoring_seconds": scored_finished - encoding_finished,
        "total_seconds": scored_finished - started,
        "encoded_feature_shards": encoded,
        "resumed_feature_shards": resumed,
        "peak_torch_gpu_memory_bytes": peak_memory,
    }
    runtime_path = args.output_dir / "first_complete_runtime.json"
    if runtime_path.exists():
        first_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if first_runtime["protocol_sha256"] != protocol_sha256:
            raise AssertionError("Stage 28 runtime protocol changed")
    else:
        first_runtime = {
            "protocol_sha256": protocol_sha256,
            **current_runtime,
        }
        write_deterministic(
            runtime_path,
            canonical_json(first_runtime),
            overwrite=False,
        )
    report = {
        "schema_version": 1,
        "stage": "WM Stage 28 independent hidden dynamics residual confirmation",
        "status": "passed",
        "scope": {
            "pairs": 6,
            "episodes": 12,
            "tasks": list(TASK_IDS),
            "conditions": list(CONDITIONS),
            "control_steps": 12 * STAGE26.LIVE_STEPS,
            "window_rows": len(window_rows),
            "primary_metrics_tested": 1,
            "detector_thresholds_fitted": 0,
            "model_parameters_updated": 0,
            "shield_actions_applied": 0,
        },
        "candidate_freeze": candidate,
        "protocol": protocol,
        "pairing": {
            "all_pair_contracts_passed": all(row["pair_contract_passed"] for row in pair_contracts),
            "action_interface_mismatch_steps_total": sum(
                int(row["action_interface_mismatch_steps"]) for row in pair_contracts
            ),
        },
        "primary_result": primary,
        "descriptive_comparators": {metric: statistics[metric] for metric in COMPARATOR_METRICS},
        "conclusion": {
            "independent_confirmation_passed": passed,
            "candidate_direction_replicated": (float(primary["mean"]) > 0),
            "candidate_positive_pairs_at_least_five": (int(primary["positive_pairs"]) >= 5),
            "candidate_exact_p_at_most_0_05": (float(primary["exact_sign_flip_p"]) <= 0.05),
            "detector_threshold_fitted": False,
            "online_detector_validated": False,
            "online_shield_justified": False,
        },
        "frozen_components": {
            "encoder_model_id": STAGE12.MODEL_ID,
            "encoder_revision": STAGE12.MODEL_REVISION,
            "no_action_checkpoint_sha256": (stage14_report["variants"]["no_action"]["checkpoint_sha256"]),
            "normalization_source": "Stage 13 train split only",
        },
        "artifacts": {
            "frozen_candidate": "frozen_candidate.json",
            "frozen_candidate_sha256": candidate_sha256,
            "protocol": "protocol.json",
            "protocol_sha256": protocol_sha256,
            "episodes": "episodes.csv",
            "episodes_sha256": sha256_text(collection_text),
            "pair_contracts": "pair_contracts.csv",
            "pair_contracts_sha256": sha256_text(pair_contract_text),
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
            "raw_rollouts_and_features": ("local-only under outputs/wm/stage28_confirm_hidden_dynamics_residual"),
        },
        "determinism": {
            "encoder_batch_size": ENCODER_BATCH_SIZE,
            "recheck_pair_id": recheck_id,
            "recheck_condition": "nominal",
            "recheck_tokens_bit_exact": True,
        },
        "runtime": {
            "first_complete_run": first_runtime,
            "current_run": current_runtime,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "limitations": [
            "The six states were used in Stage 25 for a different action-log-visible fault family.",
            "The hidden-gain confirmation trajectories themselves are new and were collected after candidate freeze.",
            "The candidate was selected exploratorily after Stage 27 and did not survive its diagnostic-family Holm correction.",
            "The gain shift is synthetic and the evaluator knows its onset.",
            "The no-action model receives oracle robot state at every recursive step.",
            "Episode pre-standardization assumes a clean startup period.",
            "No alarm threshold, nominal false-positive rate, latency, recovery policy, or shield is evaluated.",
        ],
        "stage29_readiness": {
            "detector_calibration_and_held_out_event_test_justified": passed,
            "must_use_states_outside_stage26_and_stage28": True,
            "online_shield_claim_justified": False,
        },
    }
    report_text = canonical_json(report)
    for filename, content in (
        ("frozen_candidate.json", candidate_text),
        ("protocol.json", protocol_text),
        ("episodes.csv", collection_text),
        ("pair_contracts.csv", pair_contract_text),
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
        result_svg(statistics, passed),
        overwrite=args.overwrite_public,
    )
    print(
        "Stage 28 complete: "
        f"primary={primary['positive_pairs']}/6, "
        f"mean={primary['mean']:+.4f}, "
        f"p={primary['exact_sign_flip_p']:.5f}, "
        f"confirmed={passed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
