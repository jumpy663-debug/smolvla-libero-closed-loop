#!/usr/bin/env python

"""Analyze the frozen Stage 29 manifest with paired, initial-state-level statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from statistical_evaluation_common import (
    FORMAL_CORE_TASK_IDS,
    atomic_write_text,
    build_jobs,
    canonical_json,
    load_frozen_protocol,
    record_path,
    sha256_file,
    validate_resume_record,
)

DEFAULT_ASSET_ROOT = Path("/home/jump/projects/lerobot/outputs/reproduction/smolvla_libero")
DEFAULT_PROTOCOL = Path("protocols/statistical_evaluation_v1.json")
DEFAULT_INPUT_DIR = Path("outputs/statistical_evaluation_v1")
DEFAULT_RESULTS_DIR = Path("results/statistical_evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def csv_text(rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> str:
    if not rows and fieldnames is None:
        raise ValueError("fieldnames are required for an empty CSV")
    selected_fieldnames = fieldnames or list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=selected_fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def distribution(values: Iterable[float | int]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        raise ValueError("Wilson interval requires at least one trial")
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
    margin /= denominator
    return [center - margin, center + margin]


def exact_mcnemar_p_value(first_only_success: int, second_only_success: int) -> float:
    if first_only_success < 0 or second_only_success < 0:
        raise ValueError("Discordant counts must be non-negative")
    discordant = first_only_success + second_only_success
    if discordant == 0:
        return 1.0
    lower = min(first_only_success, second_only_success)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def holm_adjust(p_values: list[float]) -> list[float]:
    if any(value < 0 or value > 1 for value in p_values):
        raise ValueError("p-values must be in [0, 1]")
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def paired_bootstrap(
    differences: list[float],
    *,
    resamples: int = 10_000,
    seed: int = 20_260_727,
) -> dict[str, Any]:
    if not differences:
        return {"pairs": 0, "mean_difference": None, "confidence_interval_95": [None, None]}
    array = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "pairs": int(array.size),
        "mean_difference": float(array.mean()),
        "confidence_interval_95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "resamples": resamples,
        "seed": seed,
        "difference_direction": "second minus first",
    }


def action_trace_sha256(actions: np.ndarray) -> str:
    array = np.ascontiguousarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 7 or len(array) == 0:
        raise AssertionError(f"Invalid legacy action trace: {array.shape}")
    digest = hashlib.sha256()
    digest.update(b"action\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update((json.dumps(list(array.shape), indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def resolve_legacy_video(asset_root: Path, csv_video: str) -> Path:
    path = Path(csv_video)
    parts = path.parts
    if parts and parts[0] == "outputs":
        path = Path(*parts[1:])
    return asset_root / path


def legacy_trajectory_path(asset_root: Path, policy: str, horizon: int, task_id: int) -> Path:
    if policy == "pretrained" and horizon == 50:
        return asset_root / "stage6_full_spatial_benchmark" / "trajectories" / f"task_{task_id:02d}.npz"
    if policy == "pretrained":
        return (
            asset_root
            / "stage7_action_horizon_ablation"
            / "trajectories"
            / f"horizon_{horizon}"
            / f"task_{task_id:02d}.npz"
        )
    if policy == "expert_only":
        return asset_root / "stage9_task5_finetune_eval" / "trajectories" / f"task_{task_id:02d}.npz"
    if policy == "lora":
        return asset_root / "stage10_task5_lora_eval" / "trajectories" / f"task_{task_id:02d}.npz"
    raise ValueError(f"Unsupported legacy condition: {policy}/H{horizon}")


def legacy_condition(source_path: str, row: dict[str, str]) -> tuple[str, int]:
    if source_path.endswith("full_spatial_baseline.csv"):
        return "pretrained", 50
    if source_path.endswith("action_horizon_ablation.csv"):
        return "pretrained", int(row["horizon"])
    if source_path.endswith("expert_only_eval.csv"):
        return "expert_only", 10
    if source_path.endswith("lora_eval.csv"):
        return "lora", 10
    raise ValueError(f"Unknown legacy source: {source_path}")


def load_legacy_rows(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    repository_root: Path,
    asset_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    source_runtimes = []
    trajectory_cache: dict[Path, dict[str, np.ndarray]] = {}
    file_hash_cache: dict[Path, str] = {}
    for source in protocol["legacy_reuse"]["sources"]:
        source_path = repository_root / source["path"]
        if sha256_file(source_path) != source["sha256"]:
            raise AssertionError(f"Legacy CSV changed after protocol freeze: {source_path}")
        source_report_path = asset_root / source["source_report_relative_path_under_asset_root"]
        if sha256_file(source_report_path) != source["source_report_sha256"]:
            raise AssertionError(f"Legacy source report changed after protocol freeze: {source_report_path}")
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        source_runtimes.append(
            {
                "source_csv": source["path"],
                "source_report": source["source_report_relative_path_under_asset_root"],
                "source_report_sha256": source["source_report_sha256"],
                "runtime": source_report.get("runtime"),
                "per_horizon": source_report.get("per_horizon"),
            }
        )
        with source_path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                policy, horizon = legacy_condition(source["path"], row)
                if source["path"].endswith("action_horizon_ablation.csv") and horizon == 50:
                    continue
                task_id = int(row["task_id"])
                init_state_index = int(row["init_state_index"])
                steps = int(row["steps"])
                episode_index = int(row["episode_index"])
                trajectory_path = legacy_trajectory_path(asset_root, policy, horizon, task_id)
                if trajectory_path not in trajectory_cache:
                    with np.load(trajectory_path, allow_pickle=False) as archive:
                        trajectory_cache[trajectory_path] = {key: archive[key] for key in archive.files}
                trajectory = trajectory_cache[trajectory_path]
                actions = np.asarray(trajectory["actions"][episode_index, :steps], dtype=np.float32)
                video = resolve_legacy_video(asset_root, row["video"])
                for artifact in (trajectory_path, video):
                    if artifact not in file_hash_cache:
                        file_hash_cache[artifact] = sha256_file(artifact)
                success = parse_bool(row["success"])
                environment_seed = int(row["seed"])
                records.append(
                    {
                        "schema_version": 1,
                        "protocol_sha256": protocol_sha256,
                        "episode_id": (
                            f"formal-{policy}-h{horizon:02d}-task{task_id:02d}-init{init_state_index:02d}-legacy-batch3"
                        ),
                        "mode": "formal",
                        "source": "legacy_reused",
                        "policy": policy,
                        "task_id": task_id,
                        "task_name": row["task_name"],
                        "init_state_index": init_state_index,
                        "official_libero_initial_state": True,
                        "environment_seed": environment_seed,
                        "policy_seed": None,
                        "policy_rng_seed": 1000 + task_id,
                        "policy_rng_batch_size": 3,
                        "action_horizon": horizon,
                        "maximum_control_steps": 280,
                        "success": success,
                        "control_steps": steps,
                        "censored": not success and steps >= 280,
                        "sum_reward": float(row["sum_reward"]),
                        "max_reward": float(row["max_reward"]),
                        "logical_vlm_forward_count": math.ceil(steps / horizon),
                        "actual_vlm_forward_count": None,
                        "inference_latencies_seconds": None,
                        "inference_latency_mean_seconds": None,
                        "inference_latency_p50_seconds": None,
                        "inference_latency_p95_seconds": None,
                        "episode_wall_seconds": None,
                        "peak_inference_gpu_memory_bytes": None,
                        "initial_observation_sha256": None,
                        "action_sha256": action_trace_sha256(actions),
                        "trajectory_path": str(trajectory_path),
                        "trajectory_sha256": file_hash_cache[trajectory_path],
                        "video_path": str(video),
                        "video_sha256": file_hash_cache[video],
                        "legacy_source_csv": source["path"],
                        "legacy_source_csv_sha256": source["sha256"],
                        "legacy_source_report": source["source_report_relative_path_under_asset_root"],
                        "legacy_source_report_sha256": source["source_report_sha256"],
                    }
                )
    if len(records) != int(protocol["experiments"]["legacy_reused_unique_episode_count"]):
        raise AssertionError(f"Expected 78 unique legacy rows, loaded {len(records)}")
    return records, source_runtimes


def load_new_formal_rows(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    input_dir: Path,
) -> list[dict[str, Any]]:
    records = []
    missing = []
    for job in build_jobs(protocol, "formal"):
        path = record_path(input_dir, job)
        if not path.exists():
            missing.append(job.episode_id)
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_resume_record(record, job, protocol_sha256)
        record["policy_rng_seed"] = job.policy_seed
        record["policy_rng_batch_size"] = 1
        records.append(record)
    if missing:
        preview = ", ".join(missing[:5])
        raise AssertionError(f"Formal manifest is incomplete: {len(missing)} missing, first: {preview}")
    return records


def condition_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["policy"]), int(row["action_horizon"])


def pair_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["task_id"]), int(row["init_state_index"])


def success_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in rows)
    trials = len(rows)
    per_task = []
    for task_id in sorted({int(row["task_id"]) for row in rows}):
        task_rows = [row for row in rows if int(row["task_id"]) == task_id]
        task_successes = sum(bool(row["success"]) for row in task_rows)
        per_task.append(
            {
                "task_id": task_id,
                "successes": task_successes,
                "episodes": len(task_rows),
                "success_rate": task_successes / len(task_rows),
                "wilson_95_interval": wilson_interval(task_successes, len(task_rows)),
            }
        )
    new_rows = [row for row in rows if row["source"] == "new_formal"]
    latency_events = [float(value) for row in new_rows for value in (row.get("inference_latencies_seconds") or [])]
    return {
        "successes": successes,
        "episodes": trials,
        "success_rate": successes / trials,
        "wilson_95_interval": wilson_interval(successes, trials),
        "per_task": per_task,
        "steps_to_success": distribution(int(row["control_steps"]) for row in rows if bool(row["success"])),
        "censored_failure_steps": distribution(int(row["control_steps"]) for row in rows if not bool(row["success"])),
        "logical_vlm_forward_count": distribution(int(row["logical_vlm_forward_count"]) for row in rows),
        "runtime_measurement_scope": {
            "new_sequential_episodes": len(new_rows),
            "legacy_episodes_with_null_runtime": trials - len(new_rows),
            "per_forward_latency_seconds": distribution(latency_events),
            "episode_wall_seconds": distribution(
                float(row["episode_wall_seconds"]) for row in new_rows if row.get("episode_wall_seconds") is not None
            ),
            "peak_inference_gpu_memory_bytes": distribution(
                int(row["peak_inference_gpu_memory_bytes"])
                for row in new_rows
                if row.get("peak_inference_gpu_memory_bytes") is not None
            ),
        },
    }


def compare_conditions(
    *,
    experiment: str,
    first_label: str,
    second_label: str,
    first_rows: list[dict[str, Any]],
    second_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first = {pair_key(row): row for row in first_rows}
    second = {pair_key(row): row for row in second_rows}
    if set(first) != set(second):
        raise AssertionError(f"Pair keys differ for {experiment}/{first_label}/{second_label}")
    cells = {
        "both_fail": 0,
        "first_only_success": 0,
        "second_only_success": 0,
        "both_success": 0,
    }
    pair_rows = []
    control_step_differences = []
    common_success_step_differences = []
    logical_forward_differences = []
    actual_forward_differences = []
    wall_differences = []
    for task_id, init_state_index in sorted(first):
        first_row = first[(task_id, init_state_index)]
        second_row = second[(task_id, init_state_index)]
        first_success = bool(first_row["success"])
        second_success = bool(second_row["success"])
        if first_success and second_success:
            cell = "both_success"
        elif first_success:
            cell = "first_only_success"
        elif second_success:
            cell = "second_only_success"
        else:
            cell = "both_fail"
        cells[cell] += 1
        step_difference = int(second_row["control_steps"]) - int(first_row["control_steps"])
        logical_difference = int(second_row["logical_vlm_forward_count"]) - int(first_row["logical_vlm_forward_count"])
        control_step_differences.append(float(step_difference))
        logical_forward_differences.append(float(logical_difference))
        if first_success and second_success:
            common_success_step_differences.append(float(step_difference))
        actual_difference = None
        if (
            first_row.get("actual_vlm_forward_count") is not None
            and second_row.get("actual_vlm_forward_count") is not None
        ):
            actual_difference = int(second_row["actual_vlm_forward_count"]) - int(first_row["actual_vlm_forward_count"])
            actual_forward_differences.append(float(actual_difference))
        wall_difference = None
        if first_row.get("episode_wall_seconds") is not None and second_row.get("episode_wall_seconds") is not None:
            wall_difference = float(second_row["episode_wall_seconds"]) - float(first_row["episode_wall_seconds"])
            wall_differences.append(wall_difference)
        pair_rows.append(
            {
                "experiment": experiment,
                "comparison": f"{first_label}_vs_{second_label}",
                "first": first_label,
                "second": second_label,
                "task_id": task_id,
                "init_state_index": init_state_index,
                "environment_seed": first_row["environment_seed"],
                "first_policy_rng_seed": first_row.get("policy_rng_seed"),
                "second_policy_rng_seed": second_row.get("policy_rng_seed"),
                "policy_rng_batch_size": first_row.get("policy_rng_batch_size"),
                "first_success": first_success,
                "second_success": second_success,
                "cell": cell,
                "first_steps": int(first_row["control_steps"]),
                "second_steps": int(second_row["control_steps"]),
                "step_difference_second_minus_first": step_difference,
                "first_logical_vlm_forwards": int(first_row["logical_vlm_forward_count"]),
                "second_logical_vlm_forwards": int(second_row["logical_vlm_forward_count"]),
                "logical_vlm_forward_difference_second_minus_first": logical_difference,
                "actual_vlm_forward_difference_second_minus_first": actual_difference,
                "episode_wall_difference_second_minus_first": wall_difference,
            }
        )
    raw_p_value = exact_mcnemar_p_value(cells["first_only_success"], cells["second_only_success"])
    return (
        {
            "experiment": experiment,
            "first": first_label,
            "second": second_label,
            **cells,
            "discordant_pairs": cells["first_only_success"] + cells["second_only_success"],
            "raw_p_value": raw_p_value,
            "holm_adjusted_p_value": None,
            "paired_continuous": {
                "control_steps_all_pairs": paired_bootstrap(control_step_differences),
                "steps_to_success_both_succeed": paired_bootstrap(common_success_step_differences),
                "logical_vlm_forward_count": paired_bootstrap(logical_forward_differences),
                "actual_vlm_forward_count_available_pairs": paired_bootstrap(actual_forward_differences),
                "episode_wall_seconds_available_pairs": paired_bootstrap(wall_differences),
            },
        },
        pair_rows,
    )


def apply_family_holm(comparisons: list[dict[str, Any]]) -> None:
    adjusted = holm_adjust([float(item["raw_p_value"]) for item in comparisons])
    for item, value in zip(comparisons, adjusted, strict=True):
        item["holm_adjusted_p_value"] = value


def build_paired_state_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(str(row["policy"]), int(row["action_horizon"]), *pair_key(row)): row for row in rows}
    horizon_rows = []
    adaptation_rows = []
    for task_id in FORMAL_CORE_TASK_IDS:
        for init_state_index in range(10):
            horizon_row: dict[str, Any] = {
                "task_id": task_id,
                "init_state_index": init_state_index,
                "environment_seed": 1000 + init_state_index,
            }
            for horizon in (50, 25, 10):
                row = lookup[("pretrained", horizon, task_id, init_state_index)]
                horizon_row[f"h{horizon}_success"] = bool(row["success"])
                horizon_row[f"h{horizon}_steps"] = int(row["control_steps"])
                horizon_row[f"h{horizon}_source"] = row["source"]
            horizon_rows.append(horizon_row)

            adaptation_row: dict[str, Any] = {
                "task_id": task_id,
                "init_state_index": init_state_index,
                "environment_seed": 1000 + init_state_index,
                "action_horizon": 10,
            }
            for policy in ("pretrained", "expert_only", "lora"):
                row = lookup[(policy, 10, task_id, init_state_index)]
                adaptation_row[f"{policy}_success"] = bool(row["success"])
                adaptation_row[f"{policy}_steps"] = int(row["control_steps"])
                adaptation_row[f"{policy}_source"] = row["source"]
            adaptation_rows.append(adaptation_row)
    return horizon_rows, adaptation_rows


def relative_public_path(
    path_value: str | None,
    repository_root: Path,
    asset_root: Path,
) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository_root.resolve()))
    except ValueError:
        try:
            return str(Path("external_asset_root") / resolved.relative_to(asset_root.resolve()))
        except ValueError as error:
            raise AssertionError(f"Refusing to publish an unresolved absolute path: {resolved}") from error


def flat_episode_row(
    row: dict[str, Any],
    repository_root: Path,
    asset_root: Path,
) -> dict[str, Any]:
    return {
        "episode_id": row["episode_id"],
        "source": row["source"],
        "policy": row["policy"],
        "task_id": row["task_id"],
        "task_name": row["task_name"],
        "init_state_index": row["init_state_index"],
        "environment_seed": row["environment_seed"],
        "policy_rng_seed": row.get("policy_rng_seed"),
        "policy_rng_batch_size": row.get("policy_rng_batch_size"),
        "action_horizon": row["action_horizon"],
        "maximum_control_steps": row["maximum_control_steps"],
        "success": row["success"],
        "control_steps": row["control_steps"],
        "censored": row["censored"],
        "logical_vlm_forward_count": row["logical_vlm_forward_count"],
        "actual_vlm_forward_count": row.get("actual_vlm_forward_count"),
        "inference_latency_mean_seconds": row.get("inference_latency_mean_seconds"),
        "inference_latency_p50_seconds": row.get("inference_latency_p50_seconds"),
        "inference_latency_p95_seconds": row.get("inference_latency_p95_seconds"),
        "episode_wall_seconds": row.get("episode_wall_seconds"),
        "peak_inference_gpu_memory_bytes": row.get("peak_inference_gpu_memory_bytes"),
        "initial_observation_sha256": row.get("initial_observation_sha256"),
        "action_sha256": row["action_sha256"],
        "video": relative_public_path(row["video_path"], repository_root, asset_root),
        "video_sha256": row["video_sha256"],
        "trajectory": relative_public_path(row["trajectory_path"], repository_root, asset_root),
        "trajectory_sha256": row["trajectory_sha256"],
        "protocol_sha256": row["protocol_sha256"],
    }


def annotation_rows(
    rows: list[dict[str, Any]],
    *,
    repository_root: Path,
    asset_root: Path,
    existing_path: Path,
) -> list[dict[str, Any]]:
    existing = {}
    if existing_path.exists():
        with existing_path.open(encoding="utf-8", newline="") as file:
            existing = {row["episode_id"]: row for row in csv.DictReader(file)}
    annotations = []
    for row in rows:
        if bool(row["success"]):
            continue
        episode_id = str(row["episode_id"])
        preserved = existing.get(episode_id, {})
        annotations.append(
            {
                "episode_id": episode_id,
                "policy": row["policy"],
                "task_id": row["task_id"],
                "task_name": row["task_name"],
                "init_state_index": row["init_state_index"],
                "environment_seed": row["environment_seed"],
                "policy_seed": row.get("policy_rng_seed") or "",
                "action_horizon": row["action_horizon"],
                "success": False,
                "control_steps": row["control_steps"],
                "primary_failure_code": preserved.get("primary_failure_code", ""),
                "label_source": preserved.get("label_source", "pending_human_review"),
                "review_status": preserved.get("review_status", "pending"),
                "reviewer": preserved.get("reviewer", ""),
                "notes_zh": preserved.get("notes_zh", ""),
                "video_path": relative_public_path(row["video_path"], repository_root, asset_root),
                "video_sha256": row["video_sha256"],
                "contact_sheet_path": preserved.get("contact_sheet_path", ""),
                "contact_sheet_sha256": preserved.get("contact_sheet_sha256", ""),
                "automatic_progress_fields_json": json.dumps(
                    {
                        "censored": bool(row["censored"]),
                        "logical_vlm_forward_count": int(row["logical_vlm_forward_count"]),
                        "sum_reward": row.get("sum_reward"),
                        "max_reward": row.get("max_reward"),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            }
        )
    return annotations


def failure_code_distribution(
    rows: list[dict[str, Any]],
    *,
    taxonomy_codes: list[str],
) -> dict[str, Any]:
    counts = {code: 0 for code in taxonomy_codes}
    confirmed = 0
    for row in rows:
        code = str(row.get("primary_failure_code", ""))
        if code:
            if code not in counts:
                raise AssertionError(f"Failure annotation uses an unknown taxonomy code: {code}")
            counts[code] += 1
        if row.get("review_status") == "confirmed":
            confirmed += 1
    failures = len(rows)
    return {
        "failures": failures,
        "confirmed_annotations": confirmed,
        "counts": counts,
        "fraction_of_failures": {code: (count / failures if failures else None) for code, count in counts.items()},
    }


def summarize_failure_annotations(
    rows: list[dict[str, Any]],
    *,
    taxonomy_codes: list[str],
) -> dict[str, Any]:
    core_task_ids = set(FORMAL_CORE_TASK_IDS)

    def select(policy: str, horizon: int, *, core_only: bool) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if str(row["policy"]) == policy
            and int(row["action_horizon"]) == horizon
            and (not core_only or int(row["task_id"]) in core_task_ids)
        ]

    report = {
        "annotation_unit": "failed episode",
        "primary_code_only": True,
        "overall": failure_code_distribution(rows, taxonomy_codes=taxonomy_codes),
        "by_condition": {
            "pretrained_h50_all_tasks": failure_code_distribution(
                select("pretrained", 50, core_only=False),
                taxonomy_codes=taxonomy_codes,
            ),
            "pretrained_h50_core_tasks": failure_code_distribution(
                select("pretrained", 50, core_only=True),
                taxonomy_codes=taxonomy_codes,
            ),
            "pretrained_h25_core_tasks": failure_code_distribution(
                select("pretrained", 25, core_only=True),
                taxonomy_codes=taxonomy_codes,
            ),
            "pretrained_h10_core_tasks": failure_code_distribution(
                select("pretrained", 10, core_only=True),
                taxonomy_codes=taxonomy_codes,
            ),
            "expert_only_h10_core_tasks": failure_code_distribution(
                select("expert_only", 10, core_only=True),
                taxonomy_codes=taxonomy_codes,
            ),
            "lora_h10_core_tasks": failure_code_distribution(
                select("lora", 10, core_only=True),
                taxonomy_codes=taxonomy_codes,
            ),
        },
    }
    overall = report["overall"]
    report["status"] = (
        "passed"
        if overall["confirmed_annotations"] == overall["failures"]
        and sum(overall["counts"].values()) == overall["failures"]
        else "incomplete"
    )
    return report


def analyze_randomness(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    formal_rows: list[dict[str, Any]],
    input_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    random_records: dict[tuple[int, int, int], dict[str, Any]] = {}
    missing = []
    for job in build_jobs(protocol, "randomness"):
        path = record_path(input_dir, job)
        if not path.exists():
            missing.append(job.episode_id)
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_resume_record(record, job, protocol_sha256)
        random_records[(job.task_id, job.init_state_index, job.policy_seed)] = record
    if missing:
        return (
            {
                "status": "incomplete",
                "required_additional_episodes": len(build_jobs(protocol, "randomness")),
                "missing_episodes": len(missing),
                "first_missing_episode_ids": missing[:10],
            },
            [],
        )
    formal_lookup = {
        (int(row["task_id"]), int(row["init_state_index"])): row
        for row in formal_rows
        if row["policy"] == "pretrained" and int(row["action_horizon"]) == 10 and int(row["init_state_index"]) in (3, 4)
    }
    rows = []
    for task_id in FORMAL_CORE_TASK_IDS:
        for init_state_index in range(5):
            seed_records = []
            for policy_seed_base in (1000, 2000, 3000):
                policy_seed = policy_seed_base + init_state_index
                if policy_seed_base == 1000 and init_state_index in (3, 4):
                    record = formal_lookup[(task_id, init_state_index)]
                else:
                    record = random_records[(task_id, init_state_index, policy_seed)]
                seed_records.append(record)
            action_hashes = [str(record["action_sha256"]) for record in seed_records]
            successes = [bool(record["success"]) for record in seed_records]
            steps = [int(record["control_steps"]) for record in seed_records]
            rows.append(
                {
                    "task_id": task_id,
                    "init_state_index": init_state_index,
                    "environment_seed": 1000 + init_state_index,
                    "policy_seeds": json.dumps(
                        [1000 + init_state_index, 2000 + init_state_index, 3000 + init_state_index]
                    ),
                    "unique_action_hashes": len(set(action_hashes)),
                    "action_varies_across_policy_seeds": len(set(action_hashes)) > 1,
                    "successes": json.dumps(successes),
                    "success_varies_across_policy_seeds": len(set(successes)) > 1,
                    "steps": json.dumps(steps),
                    "steps_vary_across_policy_seeds": len(set(steps)) > 1,
                }
            )
    return (
        {
            "status": "passed",
            "task_state_units": len(rows),
            "policy_seeds_per_unit": 3,
            "units_with_action_variation": sum(row["action_varies_across_policy_seeds"] for row in rows),
            "units_with_success_variation": sum(row["success_varies_across_policy_seeds"] for row in rows),
            "units_with_step_variation": sum(row["steps_vary_across_policy_seeds"] for row in rows),
            "same_seed_reproducibility_evidence": protocol["randomness"]["audit"]["same_seed_reproducibility_evidence"],
        },
        rows,
    )


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    protocol_path = args.protocol if args.protocol.is_absolute() else repository_root / args.protocol
    protocol, protocol_sha256 = load_frozen_protocol(protocol_path)
    legacy_rows, legacy_source_runtimes = load_legacy_rows(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        repository_root=repository_root,
        asset_root=args.asset_root.resolve(),
    )
    new_rows = load_new_formal_rows(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        input_dir=args.input_dir,
    )
    rows = legacy_rows + new_rows
    expected_unique = int(protocol["experiments"]["unique_formal_episode_count"])
    if len(rows) != expected_unique:
        raise AssertionError(f"Expected {expected_unique} unified episodes, got {len(rows)}")
    unique_conditions = {(str(row["policy"]), int(row["action_horizon"]), *pair_key(row)) for row in rows}
    if len(unique_conditions) != expected_unique:
        raise AssertionError("Unified episode conditions are not unique")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[condition_key(row)].append(row)

    experiment_a_rows = grouped[("pretrained", 50)]
    if len(experiment_a_rows) != 100:
        raise AssertionError("Experiment A must contain 100 episodes")
    experiment_b_groups = {
        f"h{horizon}": [row for row in grouped[("pretrained", horizon)] if int(row["task_id"]) in FORMAL_CORE_TASK_IDS]
        for horizon in (50, 25, 10)
    }
    if any(len(group) != 40 for group in experiment_b_groups.values()):
        raise AssertionError("Every Experiment B horizon must contain 40 paired episodes")
    experiment_c_groups = {policy: grouped[(policy, 10)] for policy in ("pretrained", "expert_only", "lora")}
    if any(len(group) != 40 for group in experiment_c_groups.values()):
        raise AssertionError("Every Experiment C policy must contain 40 paired episodes")

    horizon_comparisons = []
    adaptation_comparisons = []
    all_pair_rows = []
    for first, second in (("h50", "h25"), ("h50", "h10"), ("h25", "h10")):
        comparison, pair_rows = compare_conditions(
            experiment="B_action_horizon",
            first_label=first,
            second_label=second,
            first_rows=experiment_b_groups[first],
            second_rows=experiment_b_groups[second],
        )
        horizon_comparisons.append(comparison)
        all_pair_rows.extend(pair_rows)
    apply_family_holm(horizon_comparisons)
    for first, second in (
        ("pretrained", "expert_only"),
        ("pretrained", "lora"),
        ("expert_only", "lora"),
    ):
        comparison, pair_rows = compare_conditions(
            experiment="C_policy_adaptation",
            first_label=first,
            second_label=second,
            first_rows=experiment_c_groups[first],
            second_rows=experiment_c_groups[second],
        )
        adaptation_comparisons.append(comparison)
        all_pair_rows.extend(pair_rows)
    apply_family_holm(adaptation_comparisons)

    horizon_state_rows, adaptation_state_rows = build_paired_state_tables(rows)
    randomness_report, randomness_rows = analyze_randomness(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        formal_rows=rows,
        input_dir=args.input_dir,
    )
    project_summary = json.loads((repository_root / "results/results_summary.json").read_text(encoding="utf-8"))
    training_costs = {
        "expert_only": project_summary["expert_only"],
        "lora": project_summary["lora"],
        "adaptation_control": project_summary["adaptation_control"],
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    annotations = annotation_rows(
        rows,
        repository_root=repository_root,
        asset_root=args.asset_root.resolve(),
        existing_path=args.results_dir / "failure_annotations.csv",
    )
    taxonomy = json.loads((repository_root / "protocols/failure_taxonomy_v1.json").read_text(encoding="utf-8"))
    taxonomy_codes = [item["code"] for item in taxonomy["categories"]]
    failure_report = summarize_failure_annotations(annotations, taxonomy_codes=taxonomy_codes)
    statistics = {
        "schema_version": 1,
        "status": "passed",
        "protocol_sha256": protocol_sha256,
        "statistical_unit": protocol["statistics"]["primary_statistical_unit"],
        "experiment_A_expanded_pretrained_baseline": success_summary(experiment_a_rows),
        "experiment_B_action_horizon": {
            "conditions": {
                label: success_summary(condition_rows) for label, condition_rows in experiment_b_groups.items()
            },
            "paired_comparisons": horizon_comparisons,
        },
        "experiment_C_policy_adaptation": {
            "conditions": {
                label: success_summary(condition_rows) for label, condition_rows in experiment_c_groups.items()
            },
            "paired_comparisons": adaptation_comparisons,
        },
        "randomness_audit": randomness_report,
        "failure_taxonomy": failure_report,
        "limitations": [
            "Legacy states 0-2 use the frozen historical batch=3 policy RNG stream; new states 3-9 use "
            "batch=1 state-specific policy seeds. Every within-state condition comparison preserves its "
            "own frozen RNG contract.",
            "Per-episode latency, wall time, actual VLM calls, and peak memory are null for legacy rows "
            "and are not imputed or rerun.",
            "LIBERO tasks are heterogeneous; pooled intervals supplement rather than replace per-task results.",
            "All results are local simulation measurements, not physical-robot evaluation or the official "
            "paper benchmark.",
        ],
    }
    costs = {
        "schema_version": 1,
        "protocol_sha256": protocol_sha256,
        "training": training_costs,
        "legacy_condition_runtime_sources": legacy_source_runtimes,
        "new_condition_runtime": {
            f"{policy}_h{horizon}": success_summary(condition_rows)["runtime_measurement_scope"]
            for (policy, horizon), condition_rows in sorted(grouped.items())
        },
    }

    flat_rows = [
        flat_episode_row(row, repository_root, args.asset_root.resolve())
        for row in sorted(
            rows,
            key=lambda row: (
                str(row["policy"]),
                int(row["action_horizon"]),
                int(row["task_id"]),
                int(row["init_state_index"]),
            ),
        )
    ]
    atomic_write_text(args.results_dir / "episodes.csv", csv_text(flat_rows))
    atomic_write_text(args.results_dir / "statistics.json", canonical_json(statistics))
    atomic_write_text(args.results_dir / "costs.json", canonical_json(costs))
    atomic_write_text(args.results_dir / "paired_comparisons.csv", csv_text(all_pair_rows))
    atomic_write_text(args.results_dir / "action_horizon_by_state.csv", csv_text(horizon_state_rows))
    atomic_write_text(args.results_dir / "adaptation_by_state.csv", csv_text(adaptation_state_rows))
    atomic_write_text(args.results_dir / "failure_annotations.csv", csv_text(annotations))
    if randomness_rows:
        atomic_write_text(args.results_dir / "randomness_by_state.csv", csv_text(randomness_rows))
    print(
        f"Stage 30 passed: formal episodes={len(rows)}, failures={len(annotations)}, "
        f"randomness={randomness_report['status']}, protocol={protocol_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
