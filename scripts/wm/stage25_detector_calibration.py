#!/usr/bin/env python
# ruff: noqa: E501

"""Calibrate and test a causal WM actuator-mismatch detector.

All states screened in Stages 20 and 22 are excluded.  Two fresh states per
task calibrate one threshold using nominal and 0.5x attenuation prefixes.  Two
other states per task remain hidden until the threshold artifact is frozen;
the test split additionally evaluates a zero-shot three-step action delay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from stage24_online_sidecar import (
    STAGE3,
    STAGE5,
    STAGE17,
    STAGE18,
    AutoImageProcessor,
    Dinov2Model,
    OnlineObservationEncoder,
    canonical_json,
    close_envs,
    collect_live_prefix,
    csv_text,
    distribution,
    make_env_pre_post_processors,
    make_monitor,
    make_pre_post_processors,
    normalization_from_json,
    sha256_file,
    write_output,
)

TASK_IDS = (4, 7, 8)
SPLITS = ("calibration", "test")
CALIBRATION_CONDITIONS = ("nominal", "motion_attenuation_0p5")
TEST_CONDITIONS = (
    "nominal",
    "motion_attenuation_0p5",
    "action_delay_3",
)
SELECTION_SALT = "stage25-independent-detector-calibration-test-v1"
PAIRS_PER_TASK_PER_SPLIT = 2
LIVE_STEPS = 90
INTERVENTION_START = 50
FULL_POST_STARTS = range(50, 80)
INFLUENCED_STARTS = range(41, 81)
STRICT_PRE_STARTS = range(0, 41)
CONSECUTIVE_WINDOWS = 3
MINIMUM_CALIBRATION_DETECTIONS = 5
EXPECTED_SELECTION = {
    4: {
        "calibration": (3, 13),
        "test": (6, 9),
    },
    7: {
        "calibration": (10, 13),
        "test": (8, 11),
    },
    8: {
        "calibration": (16, 5),
        "test": (10, 23),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage18-episodes",
        type=Path,
        default=Path("results/wm/stage18_initial_state_scout/scout_episodes.csv"),
    )
    parser.add_argument(
        "--stage19-extension",
        type=Path,
        default=Path("results/wm/stage19_balanced_cohort/task8_extension.csv"),
    )
    parser.add_argument(
        "--stage20-screen",
        type=Path,
        default=Path("results/wm/stage20_paired_interventions/fresh_reset_screen.csv"),
    )
    parser.add_argument(
        "--stage22-screen",
        type=Path,
        default=Path("results/wm/stage22_confirmatory_interventions/fresh_reset_screen.csv"),
    )
    parser.add_argument(
        "--stage24-report",
        type=Path,
        default=Path("results/wm/stage24_online_sidecar/report.json"),
    )
    parser.add_argument(
        "--stage14-report",
        type=Path,
        default=Path("results/wm/stage14_next_latent_baseline/report.json"),
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path("results/wm/stage14_next_latent_baseline/normalization.json"),
    )
    parser.add_argument(
        "--wm-checkpoint-dir",
        type=Path,
        default=Path("outputs/wm/stage14_next_latent_baseline"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=STAGE17.STAGE12.default_model_root(),
    )
    parser.add_argument(
        "--policy-checkpoint-dir",
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
        "--output-dir",
        type=Path,
        default=Path("outputs/wm/stage25_detector_calibration"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage25_detector_calibration"),
    )
    parser.add_argument(
        "--media-path",
        type=Path,
        default=Path("media/wm_stage25_detector_calibration.svg"),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite-public", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    return STAGE17.read_csv(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def selection_key(task_id: int, init_state_index: int) -> str:
    payload = f"{SELECTION_SALT}|task={task_id}|init={init_state_index}"
    return hashlib.sha256(payload.encode()).hexdigest()


def select_pairs(
    scout_rows: list[dict[str, str]],
    excluded: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    unique = {}
    for row in scout_rows:
        key = (
            int(row["benchmark_task_id"]),
            int(row["init_state_index"]),
        )
        if key in unique:
            raise AssertionError(f"Duplicate scout state: {key}")
        unique[key] = row
    selected = []
    for task_id in TASK_IDS:
        candidates = []
        for (row_task, init_state_index), row in unique.items():
            if row_task == task_id and (row_task, init_state_index) not in excluded and int(row["steps"]) >= LIVE_STEPS:
                candidates.append(row)
        candidates.sort(
            key=lambda row: selection_key(
                task_id,
                int(row["init_state_index"]),
            )
        )
        required = len(SPLITS) * PAIRS_PER_TASK_PER_SPLIT
        if len(candidates) < required:
            raise AssertionError(f"Task {task_id} has only {len(candidates)} eligible states")
        for position, row in enumerate(candidates[:required]):
            split = SPLITS[position // PAIRS_PER_TASK_PER_SPLIT]
            selected.append(
                {
                    "pair_id": (f"{split}-task{task_id:02d}-init{int(row['init_state_index']):02d}"),
                    "split": split,
                    "benchmark_task_id": task_id,
                    "task_name": row["task_name"],
                    "init_state_index": int(row["init_state_index"]),
                    "episode_seed": int(row["episode_seed"]),
                    "historical_success": row["success"] == "True",
                    "historical_steps": int(row["steps"]),
                    "selection_key": selection_key(
                        task_id,
                        int(row["init_state_index"]),
                    ),
                    "selection_uses_wm_score": False,
                }
            )
    observed = {
        task_id: {
            split: tuple(
                int(row["init_state_index"])
                for row in selected
                if int(row["benchmark_task_id"]) == task_id and row["split"] == split
            )
            for split in SPLITS
        }
        for task_id in TASK_IDS
    }
    if observed != EXPECTED_SELECTION:
        raise AssertionError(f"Stage 25 selection changed: {observed}")
    return selected


def protocol_payload(
    *,
    pairs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "selection_salt": SELECTION_SALT,
        "tasks": list(TASK_IDS),
        "pairs_per_task_per_split": PAIRS_PER_TASK_PER_SPLIT,
        "pairs": pairs,
        "all_stage20_and_stage22_screen_states_excluded": True,
        "minimum_historical_steps": LIVE_STEPS,
        "live_steps": LIVE_STEPS,
        "intervention_start_zero_based": INTERVENTION_START,
        "calibration_conditions": list(CALIBRATION_CONDITIONS),
        "test_conditions": list(TEST_CONDITIONS),
        "test_action_delay_is_zero_shot_fault_type": True,
        "wm_score": ("commanded H10 raw latent MSE minus executed H10 raw latent MSE"),
        "event_rule": {
            "comparison": "score > threshold",
            "consecutive_windows": CONSECUTIVE_WINDOWS,
            "strict_pre_start_indices": [
                STRICT_PRE_STARTS.start,
                STRICT_PRE_STARTS.stop - 1,
            ],
            "fault_influenced_start_indices": [
                INFLUENCED_STARTS.start,
                INFLUENCED_STARTS.stop - 1,
            ],
            "full_post_start_indices_for_threshold_candidates": [
                FULL_POST_STARTS.start,
                FULL_POST_STARTS.stop - 1,
            ],
        },
        "threshold_rule": (
            "From positive calibration attenuation full-post scores, form "
            "midpoints between sorted unique values and zero. Select the "
            "largest threshold with zero nominal/pre false alarms and at "
            "least 5/6 calibration fault episodes detected by three "
            "consecutive above-threshold windows."
        ),
        "direct_baseline": ("alarm at first step where L2(commanded arm - executed arm) > 0"),
        "test_scores_accessed_before_threshold_freeze": False,
        "thresholds_fitted": 1,
        "actions_modified_by_detector": 0,
        "source_hashes": {
            "stage18_episodes": sha256_file(args.stage18_episodes),
            "stage19_extension": sha256_file(args.stage19_extension),
            "stage20_screen": sha256_file(args.stage20_screen),
            "stage22_screen": sha256_file(args.stage22_screen),
            "stage24_report": sha256_file(args.stage24_report),
        },
    }


def episode_paths(
    output_dir: Path,
    pair: dict[str, Any],
    condition: str,
) -> tuple[Path, Path]:
    base = output_dir / "episodes" / f"{pair['pair_id']}-{condition}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode())
    digest.update(canonical_json(list(contiguous.shape)).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def save_live_episode(
    *,
    output_dir: Path,
    pair: dict[str, Any],
    condition: str,
    live: dict[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    archive_path, metadata_path = episode_paths(
        output_dir,
        pair,
        condition,
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "commanded_action": live["commanded_action"],
        "executed_action": live["executed_action"],
        "reward": live["reward"],
        "success": live["success"],
    }
    temporary = archive_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(archive_path)
    mismatch = np.linalg.norm(
        arrays["commanded_action"][:, :6] - arrays["executed_action"][:, :6],
        axis=1,
    )
    record = {
        "pair_id": pair["pair_id"],
        "split": pair["split"],
        "condition": condition,
        "benchmark_task_id": int(pair["benchmark_task_id"]),
        "task_name": pair["task_name"],
        "init_state_index": int(pair["init_state_index"]),
        "episode_seed": int(pair["episode_seed"]),
        "steps": int(live["steps"]),
        "success_within_prefix": bool(live["success_within_prefix"]),
        "timeline": live["timeline"],
        "timeline_rows": len(live["timeline"]),
        "encoding_seconds": distribution(live["encoding_seconds"]),
        "scoring_seconds": distribution([float(row["scoring_seconds"]) for row in live["timeline"]]),
        "pre_commanded_action_sha256": array_sha256(arrays["commanded_action"][:INTERVENTION_START]),
        "pre_reward_sha256": array_sha256(arrays["reward"][:INTERVENTION_START]),
        "commanded_action_sha256": array_sha256(arrays["commanded_action"]),
        "executed_action_sha256": array_sha256(arrays["executed_action"]),
        "first_direct_mismatch_step": (int(np.flatnonzero(mismatch > 0)[0]) if bool((mismatch > 0).any()) else None),
        "mean_post_arm_action_mismatch_l2": float(mismatch[INTERVENTION_START:].mean()),
        "protocol_sha256": protocol_sha256,
        "archive_file": archive_path.as_posix(),
        "archive_sha256": sha256_file(archive_path),
    }
    write_output(
        metadata_path,
        canonical_json(record),
        overwrite=False,
    )
    return record


def resume_episode(
    *,
    output_dir: Path,
    pair: dict[str, Any],
    condition: str,
    protocol_sha256: str,
) -> dict[str, Any] | None:
    archive_path, metadata_path = episode_paths(
        output_dir,
        pair,
        condition,
    )
    if not archive_path.exists() and not metadata_path.exists():
        return None
    if not archive_path.exists() or not metadata_path.exists():
        raise RuntimeError(f"Partial Stage 25 output: {pair['pair_id']} {condition}")
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "pair_id": pair["pair_id"],
        "split": pair["split"],
        "condition": condition,
        "benchmark_task_id": int(pair["benchmark_task_id"]),
        "init_state_index": int(pair["init_state_index"]),
        "protocol_sha256": protocol_sha256,
        "steps": LIVE_STEPS,
        "timeline_rows": LIVE_STEPS - 10 + 1,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"Stage 25 resume mismatch: {key}")
    if sha256_file(archive_path) != record["archive_sha256"]:
        raise AssertionError("Stage 25 archive hash changed")
    return record


def score_series(
    episode: dict[str, Any],
    starts: range,
) -> list[dict[str, Any]]:
    return [row for row in episode["timeline"] if int(row["start_index"]) in starts]


def first_consecutive_alarm(
    episode: dict[str, Any],
    *,
    threshold: float,
    starts: range,
    consecutive: int = CONSECUTIVE_WINDOWS,
) -> dict[str, Any] | None:
    rows = sorted(
        score_series(episode, starts),
        key=lambda row: int(row["start_index"]),
    )
    run = 0
    previous_start = None
    for row in rows:
        start = int(row["start_index"])
        if previous_start is None or start != previous_start + 1 or float(row["commanded_minus_executed"]) <= threshold:
            run = 0
        if float(row["commanded_minus_executed"]) > threshold:
            run += 1
        previous_start = start
        if run >= consecutive:
            return row
    return None


def calibration_counts(
    episodes: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, int]:
    nominal = [row for row in episodes if row["condition"] == "nominal"]
    faults = [row for row in episodes if row["condition"] == "motion_attenuation_0p5"]
    nominal_false = sum(
        first_consecutive_alarm(
            row,
            threshold=threshold,
            starts=range(0, LIVE_STEPS - 10 + 1),
        )
        is not None
        for row in nominal
    )
    pre_false = sum(
        first_consecutive_alarm(
            row,
            threshold=threshold,
            starts=STRICT_PRE_STARTS,
        )
        is not None
        for row in faults
    )
    detected = sum(
        first_consecutive_alarm(
            row,
            threshold=threshold,
            starts=INFLUENCED_STARTS,
        )
        is not None
        for row in faults
    )
    return {
        "nominal_false_alarms": nominal_false,
        "fault_pre_false_alarms": pre_false,
        "fault_episodes_detected": detected,
        "fault_episodes": len(faults),
    }


def freeze_threshold(
    calibration_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    fault_post_values = sorted(
        {
            float(row["commanded_minus_executed"])
            for episode in calibration_episodes
            if episode["condition"] == "motion_attenuation_0p5"
            for row in score_series(episode, FULL_POST_STARTS)
            if float(row["commanded_minus_executed"]) > 0
        }
    )
    if not fault_post_values:
        raise AssertionError("Calibration has no positive post scores")
    boundaries = [0.0, *fault_post_values]
    candidates = [
        (left + right) / 2
        for left, right in zip(
            boundaries,
            boundaries[1:],
            strict=False,
        )
    ]
    evaluated = []
    for threshold in candidates:
        counts = calibration_counts(
            calibration_episodes,
            threshold=threshold,
        )
        evaluated.append(
            {
                "threshold": threshold,
                **counts,
            }
        )
    eligible = [
        row
        for row in evaluated
        if row["nominal_false_alarms"] == 0
        and row["fault_pre_false_alarms"] == 0
        and row["fault_episodes_detected"] >= MINIMUM_CALIBRATION_DETECTIONS
    ]
    if not eligible:
        raise AssertionError("No calibration threshold satisfies the frozen rule")
    selected = max(eligible, key=lambda row: float(row["threshold"]))
    return {
        "schema_version": 1,
        "score": ("commanded H10 raw latent MSE minus executed H10 raw latent MSE"),
        "comparison": "strictly greater than threshold",
        "threshold": float(selected["threshold"]),
        "consecutive_windows": CONSECUTIVE_WINDOWS,
        "selection_rule": (
            "largest positive-score midpoint with zero calibration "
            "nominal/pre false alarms and at least 5/6 calibration "
            "attenuation episodes detected"
        ),
        "candidate_thresholds_evaluated": len(evaluated),
        "selected_calibration_counts": {key: int(value) for key, value in selected.items() if key != "threshold"},
        "test_scores_accessed": False,
    }


def wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        raise ValueError("total must be positive")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return [
        0.0 if lower < 1e-12 else lower,
        1.0 if 1.0 - upper < 1e-12 else upper,
    ]


def evaluate_episode(
    episode: dict[str, Any],
    threshold: dict[str, Any],
) -> dict[str, Any]:
    value = float(threshold["threshold"])
    condition = episode["condition"]
    if condition == "nominal":
        wm_alarm = first_consecutive_alarm(
            episode,
            threshold=value,
            starts=range(0, LIVE_STEPS - 10 + 1),
        )
        direct_alarm_step = episode["first_direct_mismatch_step"]
        return {
            "pair_id": episode["pair_id"],
            "split": episode["split"],
            "condition": condition,
            "benchmark_task_id": episode["benchmark_task_id"],
            "init_state_index": episode["init_state_index"],
            "wm_alarm": wm_alarm is not None,
            "wm_alarm_available_after_step": (None if wm_alarm is None else int(wm_alarm["available_after_step"])),
            "wm_latency_steps": None,
            "wm_pre_false_alarm": False,
            "direct_alarm": direct_alarm_step is not None,
            "direct_alarm_step": direct_alarm_step,
            "direct_latency_steps": None,
        }
    pre_alarm = first_consecutive_alarm(
        episode,
        threshold=value,
        starts=STRICT_PRE_STARTS,
    )
    wm_alarm = first_consecutive_alarm(
        episode,
        threshold=value,
        starts=INFLUENCED_STARTS,
    )
    direct_alarm_step = episode["first_direct_mismatch_step"]
    return {
        "pair_id": episode["pair_id"],
        "split": episode["split"],
        "condition": condition,
        "benchmark_task_id": episode["benchmark_task_id"],
        "init_state_index": episode["init_state_index"],
        "wm_alarm": wm_alarm is not None,
        "wm_alarm_available_after_step": (None if wm_alarm is None else int(wm_alarm["available_after_step"])),
        "wm_latency_steps": (None if wm_alarm is None else int(wm_alarm["available_after_step"]) - INTERVENTION_START),
        "wm_pre_false_alarm": pre_alarm is not None,
        "direct_alarm": direct_alarm_step is not None,
        "direct_alarm_step": direct_alarm_step,
        "direct_latency_steps": (None if direct_alarm_step is None else int(direct_alarm_step) - INTERVENTION_START),
    }


def condition_metrics(
    event_rows: list[dict[str, Any]],
    *,
    split: str,
    condition: str,
) -> dict[str, Any]:
    rows = [row for row in event_rows if row["split"] == split and row["condition"] == condition]
    if not rows:
        raise AssertionError(f"No events for {split}/{condition}")
    if condition == "nominal":
        wm_false = sum(bool(row["wm_alarm"]) for row in rows)
        direct_false = sum(bool(row["direct_alarm"]) for row in rows)
        return {
            "episodes": len(rows),
            "wm_false_alarms": wm_false,
            "wm_false_alarm_rate": wm_false / len(rows),
            "wm_false_alarm_rate_wilson95": wilson_interval(
                wm_false,
                len(rows),
            ),
            "direct_false_alarms": direct_false,
            "direct_false_alarm_rate": direct_false / len(rows),
        }
    wm_detected = sum(bool(row["wm_alarm"]) for row in rows)
    direct_detected = sum(bool(row["direct_alarm"]) for row in rows)
    latencies = [int(row["wm_latency_steps"]) for row in rows if row["wm_latency_steps"] is not None]
    direct_latencies = [int(row["direct_latency_steps"]) for row in rows if row["direct_latency_steps"] is not None]
    return {
        "episodes": len(rows),
        "wm_detected": wm_detected,
        "wm_event_recall": wm_detected / len(rows),
        "wm_event_recall_wilson95": wilson_interval(
            wm_detected,
            len(rows),
        ),
        "wm_pre_false_alarms": sum(bool(row["wm_pre_false_alarm"]) for row in rows),
        "wm_latency_steps": distribution(latencies),
        "direct_detected": direct_detected,
        "direct_event_recall": direct_detected / len(rows),
        "direct_latency_steps": distribution(direct_latencies),
    }


def validate_pair_prefixes(
    episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for pair_id in sorted({row["pair_id"] for row in episodes}):
        selected = [row for row in episodes if row["pair_id"] == pair_id]
        pre_actions = {row["pre_commanded_action_sha256"] for row in selected}
        pre_rewards = {row["pre_reward_sha256"] for row in selected}
        row = {
            "pair_id": pair_id,
            "split": selected[0]["split"],
            "benchmark_task_id": selected[0]["benchmark_task_id"],
            "init_state_index": selected[0]["init_state_index"],
            "conditions": len(selected),
            "pre_commanded_actions_bit_exact": len(pre_actions) == 1,
            "pre_rewards_bit_exact": len(pre_rewards) == 1,
        }
        row["preperiod_pairing_passed"] = row["pre_commanded_actions_bit_exact"] and row["pre_rewards_bit_exact"]
        rows.append(row)
    return rows


def load_policy(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, Any, float]:
    STAGE3.verify_stage2_manifest(args.stage2_manifest)
    config = PreTrainedConfig.from_pretrained(
        args.policy_checkpoint_dir,
        local_files_only=True,
    )
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    config.device = "cuda"
    config.vlm_model_name = str(args.backbone_dir.resolve())
    config.load_vlm_weights = True
    config.empty_cameras = 1
    config.use_amp = False
    config.n_action_steps = STAGE18.ACTION_HORIZON
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.policy_checkpoint_dir),
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": {}},
            "tokenizer_processor": {
                "tokenizer_name": str(args.backbone_dir.resolve()),
            },
            "device_processor": {"device": "cuda"},
        },
    )
    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(
        args.policy_checkpoint_dir,
        config=config,
        local_files_only=True,
    ).eval()
    return (
        config,
        policy,
        preprocessor,
        postprocessor,
        time.perf_counter() - started,
    )


def collect_split(
    *,
    split: str,
    pairs: list[dict[str, Any]],
    conditions: tuple[str, ...],
    args: argparse.Namespace,
    policy_bundle: tuple[Any, Any, Any, Any, float],
    observation_encoder: OnlineObservationEncoder,
    models: dict[str, torch.nn.Module],
    normalization: Any,
    device: torch.device,
    protocol_sha256: str,
) -> list[dict[str, Any]]:
    config, policy, preprocessor, postprocessor, _ = policy_bundle
    records = []
    selected_pairs = [row for row in pairs if row["split"] == split]
    total = len(selected_pairs) * len(conditions)
    position = 0
    for pair in selected_pairs:
        for condition in conditions:
            position += 1
            resumed = resume_episode(
                output_dir=args.output_dir,
                pair=pair,
                condition=condition,
                protocol_sha256=protocol_sha256,
            )
            if resumed is not None:
                records.append(resumed)
                print(
                    f"[{split} {position}/{total}] resume {pair['pair_id']} {condition}",
                    flush=True,
                )
                continue
            env_cfg, env, envs = STAGE5.make_task_env(
                suite="libero_spatial",
                task_id=int(pair["benchmark_task_id"]),
                n_envs=1,
                resolution=STAGE18.RESOLUTION,
            )
            try:
                env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, config)
                started = time.perf_counter()
                live = collect_live_prefix(
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    observation_encoder=observation_encoder,
                    monitor=make_monitor(models, normalization, device),
                    task_id=int(pair["benchmark_task_id"]),
                    init_state_index=int(pair["init_state_index"]),
                    condition=condition,
                    live_steps=LIVE_STEPS,
                    capture_frames=False,
                )
                if int(live["steps"]) != LIVE_STEPS:
                    raise AssertionError(f"Episode ended before {LIVE_STEPS}: {pair['pair_id']} {condition}")
                record = save_live_episode(
                    output_dir=args.output_dir,
                    pair=pair,
                    condition=condition,
                    live=live,
                    protocol_sha256=protocol_sha256,
                )
                records.append(record)
                print(
                    f"[{split} {position}/{total}] {pair['pair_id']} "
                    f"{condition} seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
            finally:
                close_envs(envs)
    return records


def result_svg(
    test_metrics: dict[str, dict[str, Any]],
    threshold: float,
) -> str:
    attenuation = test_metrics["motion_attenuation_0p5"]
    delay = test_metrics["action_delay_3"]
    nominal = test_metrics["nominal"]
    attenuation_recall = int(attenuation["wm_detected"])
    delay_recall = int(delay["wm_detected"])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="580" viewBox="0 0 1200 580">
<rect width="1200" height="580" fill="#f7f8fa"/>
<text x="65" y="65" font-size="31" font-family="sans-serif" font-weight="700" fill="#162033">Stage 25：独立阈值校准与故障 onset 评测</text>
<text x="65" y="105" font-size="18" font-family="sans-serif" fill="#526070">6 calibration pairs 冻结 threshold={threshold:.6f}；6 个新 test pairs 零参与调参</text>
<rect x="65" y="145" width="330" height="150" rx="14" fill="#ffffff" stroke="#dce1e8"/>
<text x="95" y="187" font-size="20" font-family="sans-serif" font-weight="700" fill="#162033">Normal 误报</text>
<text x="95" y="245" font-size="38" font-family="monospace" fill="#3478b8">{int(nominal["wm_false_alarms"])}/6</text>
<rect x="435" y="145" width="330" height="150" rx="14" fill="#ffffff" stroke="#dce1e8"/>
<text x="465" y="187" font-size="20" font-family="sans-serif" font-weight="700" fill="#162033">0.5× 衰减检出</text>
<text x="465" y="245" font-size="38" font-family="monospace" fill="#e36b4f">{attenuation_recall}/6</text>
<rect x="805" y="145" width="330" height="150" rx="14" fill="#ffffff" stroke="#dce1e8"/>
<text x="835" y="187" font-size="20" font-family="sans-serif" font-weight="700" fill="#162033">3-step delay 零样本检出</text>
<text x="835" y="245" font-size="38" font-family="monospace" fill="#d99a31">{delay_recall}/6</text>
<text x="65" y="355" font-size="22" font-family="sans-serif" font-weight="700" fill="#162033">必要基线：直接比较 commanded / executed action</text>
<text x="65" y="402" font-size="19" font-family="sans-serif" fill="#32445a">合成执行器故障下，直接 mismatch 无需 WM 即可在 onset 当步报警。</text>
<text x="65" y="450" font-size="19" font-family="sans-serif" fill="#32445a">因此 WM detector 的价值必须在“动作反馈不直接暴露故障”或更真实 dynamics shift 上继续验证。</text>
<text x="65" y="525" font-size="17" font-family="sans-serif" fill="#6a7583">本阶段仍是 logging-only detector benchmark；没有修改动作，也不宣称 online shield。</text>
</svg>
"""


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.model_root.name != STAGE17.STAGE12.MODEL_REVISION:
        raise ValueError("DINOv2 model revision is not pinned")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    scout_rows = read_rows(args.stage18_episodes)
    scout_rows.extend(read_rows(args.stage19_extension))
    excluded = set()
    for path in (args.stage20_screen, args.stage22_screen):
        excluded.update(
            (
                int(row["benchmark_task_id"]),
                int(row["init_state_index"]),
            )
            for row in read_rows(path)
        )
    pairs = select_pairs(scout_rows, excluded)
    protocol = protocol_payload(pairs=pairs, args=args)
    protocol_text = canonical_json(protocol)
    protocol_sha256 = sha256_text(protocol_text)
    write_output(
        args.output_dir / "protocol.json",
        protocol_text,
        overwrite=False,
    )

    stage24_report = json.loads(args.stage24_report.read_text(encoding="utf-8"))
    if not stage24_report["conclusion"]["ready_for_separate_detector_calibration"]:
        raise AssertionError("Stage 24 did not authorize Stage 25")
    stage14_report = json.loads(args.stage14_report.read_text(encoding="utf-8"))
    normalization = normalization_from_json(args.normalization, device)
    models = STAGE17.load_dynamics(
        stage14_report,
        args.wm_checkpoint_dir,
        device,
    )
    processor = AutoImageProcessor.from_pretrained(
        args.model_root,
        local_files_only=True,
    )
    dino = Dinov2Model.from_pretrained(
        args.model_root,
        local_files_only=True,
    ).eval()
    dino.requires_grad_(False)
    dino.to(device)
    observation_encoder = OnlineObservationEncoder(
        processor,
        dino,
        device=device,
    )
    policy_bundle = load_policy(args)
    loaded_seconds = time.perf_counter() - started

    calibration = collect_split(
        split="calibration",
        pairs=pairs,
        conditions=CALIBRATION_CONDITIONS,
        args=args,
        policy_bundle=policy_bundle,
        observation_encoder=observation_encoder,
        models=models,
        normalization=normalization,
        device=device,
        protocol_sha256=protocol_sha256,
    )
    calibration_pairs = validate_pair_prefixes(calibration)
    if not all(row["preperiod_pairing_passed"] for row in calibration_pairs):
        raise AssertionError("Calibration preperiod pairing failed")
    threshold = freeze_threshold(calibration)
    threshold.update(
        {
            "protocol_sha256": protocol_sha256,
            "calibration_episode_metadata_sha256": sha256_text(canonical_json(calibration)),
        }
    )
    threshold_text = canonical_json(threshold)
    threshold_path = args.output_dir / "frozen_threshold.json"
    write_output(threshold_path, threshold_text, overwrite=False)
    print(
        f"threshold frozen before test: {threshold['threshold']:.8f}",
        flush=True,
    )

    test_episodes = collect_split(
        split="test",
        pairs=pairs,
        conditions=TEST_CONDITIONS,
        args=args,
        policy_bundle=policy_bundle,
        observation_encoder=observation_encoder,
        models=models,
        normalization=normalization,
        device=device,
        protocol_sha256=protocol_sha256,
    )
    all_episodes = [*calibration, *test_episodes]
    pair_rows = validate_pair_prefixes(all_episodes)
    if not all(row["preperiod_pairing_passed"] for row in pair_rows):
        raise AssertionError("Test preperiod pairing failed")
    event_rows = [evaluate_episode(episode, threshold) for episode in all_episodes]
    calibration_metrics = {
        condition: condition_metrics(
            event_rows,
            split="calibration",
            condition=condition,
        )
        for condition in CALIBRATION_CONDITIONS
    }
    test_metrics = {
        condition: condition_metrics(
            event_rows,
            split="test",
            condition=condition,
        )
        for condition in TEST_CONDITIONS
    }

    episode_rows = []
    window_rows = []
    for episode in all_episodes:
        episode_rows.append(
            {
                key: value
                for key, value in episode.items()
                if key not in {"timeline", "archive_file"} and not isinstance(value, dict)
            }
        )
        for row in episode["timeline"]:
            window_rows.append(
                {
                    "pair_id": episode["pair_id"],
                    "split": episode["split"],
                    "condition": episode["condition"],
                    "benchmark_task_id": episode["benchmark_task_id"],
                    "init_state_index": episode["init_state_index"],
                    **{
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "condition",
                            "task_id",
                            "init_state_index",
                        }
                    },
                }
            )
    pair_text = csv_text(pair_rows)
    episode_text = csv_text(episode_rows)
    window_text = csv_text(window_rows)
    event_text = csv_text(event_rows)
    public_threshold = {
        **threshold,
        "test_scores_accessed": False,
        "frozen_artifact_sha256": sha256_file(threshold_path),
    }
    public_threshold_text = canonical_json(public_threshold)
    formal_runtime_path = args.output_dir / "formal_runtime.json"
    if formal_runtime_path.exists():
        formal_runtime = json.loads(formal_runtime_path.read_text(encoding="utf-8"))
    else:
        formal_runtime = {
            "schema_version": 1,
            "full_collection_model_load_seconds": loaded_seconds,
            "full_collection_total_seconds": time.perf_counter() - started,
            "full_collection_peak_torch_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        write_output(
            formal_runtime_path,
            canonical_json(formal_runtime),
            overwrite=False,
        )
    report = {
        "schema_version": 1,
        "stage": "WM Stage 25 independent detector calibration and onset test",
        "status": "passed",
        "scope": {
            "tasks": list(TASK_IDS),
            "calibration_pairs": 6,
            "test_pairs": 6,
            "calibration_episodes": len(calibration),
            "test_episodes": len(test_episodes),
            "test_fault_type_seen_during_calibration": ["motion_attenuation_0p5"],
            "test_fault_type_zero_shot": ["action_delay_3"],
            "thresholds_fitted": 1,
            "actions_modified_by_detector": 0,
            "shield_claimed": False,
        },
        "protocol": protocol,
        "frozen_threshold": public_threshold,
        "calibration_metrics": calibration_metrics,
        "test_metrics": test_metrics,
        "direct_action_mismatch_baseline": {
            "purpose": (
                "Necessary baseline because synthetic interventions directly change the logged executed action."
            ),
            "test_attenuation": {
                key: test_metrics["motion_attenuation_0p5"][key]
                for key in (
                    "direct_detected",
                    "direct_event_recall",
                    "direct_latency_steps",
                )
            },
            "test_action_delay": {
                key: test_metrics["action_delay_3"][key]
                for key in (
                    "direct_detected",
                    "direct_event_recall",
                    "direct_latency_steps",
                )
            },
            "test_nominal_false_alarms": test_metrics["nominal"]["direct_false_alarms"],
        },
        "conclusion": {
            "threshold_frozen_before_test_collection": True,
            "test_initial_states_disjoint_from_calibration_and_prior_interventions": True,
            "wm_detector_has_test_signal": (test_metrics["motion_attenuation_0p5"]["wm_detected"] > 0),
            "zero_shot_delay_generalization_observed": (test_metrics["action_delay_3"]["wm_detected"] > 0),
            "direct_mismatch_baseline_is_stronger_or_equal": (
                test_metrics["motion_attenuation_0p5"]["direct_detected"]
                >= test_metrics["motion_attenuation_0p5"]["wm_detected"]
                and test_metrics["action_delay_3"]["direct_detected"] >= test_metrics["action_delay_3"]["wm_detected"]
            ),
            "online_shield_justified": False,
        },
        "runtime": {
            **formal_runtime,
            "current_invocation_model_load_seconds": loaded_seconds,
            "current_invocation_total_seconds": (time.perf_counter() - started),
        },
        "artifacts": {
            "protocol": "protocol.json",
            "protocol_sha256": sha256_text(protocol_text),
            "pairs": "pairs.csv",
            "pairs_sha256": sha256_text(pair_text),
            "episodes": "episodes.csv",
            "episodes_sha256": sha256_text(episode_text),
            "window_scores": "window_scores.csv",
            "window_scores_sha256": sha256_text(window_text),
            "detector_events": "detector_events.csv",
            "detector_events_sha256": sha256_text(event_text),
            "frozen_threshold": "frozen_threshold.json",
            "frozen_threshold_sha256": sha256_text(public_threshold_text),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "limitations": [
            "Only six test pairs across three tasks are evaluated.",
            "All faults are synthetic and directly visible in executed-action feedback.",
            "Nominal commanded and executed actions are identical by construction, making the direct baseline trivial.",
            "The detector targets actuator mismatch, not semantic VLA policy errors.",
            "No action intervention or recovery policy is enabled.",
        ],
        "stage26_readiness": {
            "shield_should_not_be_built_from_this_detector_alone": True,
            "next_experiment_should_hide_direct_fault_label_from_action_feedback": True,
            "evaluate_observation_or_state_dynamics_shift": True,
        },
    }
    report_text = canonical_json(report)
    for filename, content in (
        ("protocol.json", protocol_text),
        ("pairs.csv", pair_text),
        ("episodes.csv", episode_text),
        ("window_scores.csv", window_text),
        ("detector_events.csv", event_text),
        ("frozen_threshold.json", public_threshold_text),
        ("report.json", report_text),
    ):
        write_output(
            args.public_results_dir / filename,
            content,
            overwrite=args.overwrite_public,
        )
    write_output(
        args.media_path,
        result_svg(
            test_metrics,
            float(threshold["threshold"]),
        ),
        overwrite=args.overwrite_public,
    )
    print(
        "Stage 25 complete: "
        f"normal_fp={test_metrics['nominal']['wm_false_alarms']}/6 "
        f"attenuation={test_metrics['motion_attenuation_0p5']['wm_detected']}/6 "
        f"delay={test_metrics['action_delay_3']['wm_detected']}/6",
        flush=True,
    )


if __name__ == "__main__":
    main()
