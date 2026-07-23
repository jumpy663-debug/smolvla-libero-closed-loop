#!/usr/bin/env python

"""Evaluate frozen Stage 14 models with recursive multi-step latent rollouts.

Latents are recursively predicted while the state context remains oracle
ground-truth. The common validation cohort supports horizons 1, 5, 10, and 25.
No model parameters are updated and test remains untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

HORIZONS = (1, 5, 10, 25)
MAX_HORIZON = max(HORIZONS)
ACTION_SHUFFLE_SEED = 1517
BOOTSTRAP_SEED = 1523
BOOTSTRAP_SAMPLES = 10_000
ROLLOUT_MODES = (
    "conditioned_correct",
    "conditioned_zero",
    "conditioned_shuffled",
    "no_action",
    "persistence",
    "conditioned_teacher_forced",
)


def load_stage14_module() -> ModuleType:
    path = Path(__file__).with_name("stage14_next_latent_baseline.py")
    spec = importlib.util.spec_from_file_location("stage14_next_latent_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import Stage 14 helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE14 = load_stage14_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-manifest",
        type=Path,
        default=Path("results/wm/stage13_full_feature_cache/shards.csv"),
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
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage15_multistep_rollout"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
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


def common_rollout_positions(store: Any, max_horizon: int) -> torch.Tensor:
    if max_horizon <= 0:
        raise ValueError("max_horizon must be positive")
    selected: list[torch.Tensor] = []
    for episode_id in torch.unique(store.episode_ids):
        positions = torch.nonzero(store.episode_ids == episode_id, as_tuple=False).flatten()
        valid_count = len(positions) - max_horizon + 1
        if valid_count > 0:
            selected.append(positions[:valid_count])
    if not selected:
        raise ValueError("No rollout windows support the requested horizon")
    return torch.cat(selected)


def shuffled_action_frame_indices(store: Any, seed: int) -> torch.Tensor:
    permutation = torch.randperm(
        store.transition_count,
        generator=torch.Generator().manual_seed(seed),
    )
    shuffled_frames = store.current_indices[permutation]
    mapping = torch.full((store.frame_count,), -1, dtype=torch.int64)
    mapping[store.current_indices] = shuffled_frames
    if bool((mapping[store.current_indices] < 0).any()):
        raise AssertionError("Shuffled action mapping is incomplete")
    return mapping


def append_window_metrics(
    destination: dict[str, list[np.ndarray]],
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalization: Any,
) -> None:
    normalized_mse = (prediction - target).square().mean(dim=(1, 2))
    prediction_raw = STAGE14.raw_latent(prediction, normalization)
    target_raw = STAGE14.raw_latent(target, normalization)
    raw_mse = (prediction_raw - target_raw).square().mean(dim=(1, 2))
    cosine = F.cosine_similarity(prediction_raw, target_raw, dim=-1).mean(dim=1)
    destination["normalized_mse"].append(normalized_mse.double().cpu().numpy())
    destination["raw_mse"].append(raw_mse.double().cpu().numpy())
    destination["token_cosine"].append(cosine.double().cpu().numpy())


def evaluate_rollouts(
    conditioned_model: torch.nn.Module,
    no_action_model: torch.nn.Module,
    store: Any,
    normalization: Any,
    rollout_positions: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], np.ndarray, np.ndarray]:
    conditioned_model.eval()
    no_action_model.eval()
    starts = store.current_indices[rollout_positions]
    episode_ids = store.episode_ids[rollout_positions].numpy()
    task_ids = store.task_ids[rollout_positions].numpy()
    shuffled_frames = shuffled_action_frame_indices(store, ACTION_SHUFFLE_SEED)
    collected: dict[str, dict[int, dict[str, list[np.ndarray]]]] = {
        mode: {horizon: {"normalized_mse": [], "raw_mse": [], "token_cosine": []} for horizon in HORIZONS}
        for mode in ROLLOUT_MODES
    }

    with torch.inference_mode():
        for batch_start in range(0, len(starts), batch_size):
            start_frames = starts[batch_start : batch_start + batch_size]
            start_latent_raw = store.visual_tokens[start_frames].to(
                device=device,
                dtype=torch.float32,
            )
            start_latent = (start_latent_raw - normalization.latent_mean) / normalization.latent_std
            predictions = {
                "conditioned_correct": start_latent.clone(),
                "conditioned_zero": start_latent.clone(),
                "conditioned_shuffled": start_latent.clone(),
                "no_action": start_latent.clone(),
            }
            persistence = start_latent

            for step in range(1, MAX_HORIZON + 1):
                current_frames = start_frames + step - 1
                target_frames = start_frames + step
                ground_truth_current = store.visual_tokens[current_frames].to(
                    device=device,
                    dtype=torch.float32,
                )
                ground_truth_current = (ground_truth_current - normalization.latent_mean) / normalization.latent_std
                target = store.visual_tokens[target_frames].to(
                    device=device,
                    dtype=torch.float32,
                )
                target = (target - normalization.latent_mean) / normalization.latent_std
                state = store.observation_state[current_frames].to(device)
                state = (state - normalization.state_mean) / normalization.state_std
                action = store.action[current_frames].to(device)
                action = (action - normalization.action_mean) / normalization.action_std
                shuffled_action = store.action[shuffled_frames[current_frames]].to(device)
                shuffled_action = (shuffled_action - normalization.action_mean) / normalization.action_std
                zero_action = torch.zeros_like(action)

                predictions["conditioned_correct"] = conditioned_model(
                    predictions["conditioned_correct"],
                    state,
                    action,
                )
                predictions["conditioned_zero"] = conditioned_model(
                    predictions["conditioned_zero"],
                    state,
                    zero_action,
                )
                predictions["conditioned_shuffled"] = conditioned_model(
                    predictions["conditioned_shuffled"],
                    state,
                    shuffled_action,
                )
                predictions["no_action"] = no_action_model(
                    predictions["no_action"],
                    state,
                    zero_action,
                )

                if step in HORIZONS:
                    teacher_forced = conditioned_model(
                        ground_truth_current,
                        state,
                        action,
                    )
                    for mode, prediction in predictions.items():
                        append_window_metrics(
                            collected[mode][step],
                            prediction,
                            target,
                            normalization,
                        )
                    append_window_metrics(
                        collected["persistence"][step],
                        persistence,
                        target,
                        normalization,
                    )
                    append_window_metrics(
                        collected["conditioned_teacher_forced"][step],
                        teacher_forced,
                        target,
                        normalization,
                    )

    metrics: dict[str, dict[int, dict[str, np.ndarray]]] = {
        mode: {
            horizon: {key: np.concatenate(values) for key, values in collected[mode][horizon].items()}
            for horizon in HORIZONS
        }
        for mode in ROLLOUT_MODES
    }
    expected_windows = len(rollout_positions)
    for mode in ROLLOUT_MODES:
        for horizon in HORIZONS:
            if len(metrics[mode][horizon]["raw_mse"]) != expected_windows:
                raise AssertionError(f"Incomplete metrics for {mode} horizon {horizon}")
    return metrics, episode_ids, task_ids


def aggregate_metrics(values: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "normalized_mse": float(values["normalized_mse"].mean()),
        "raw_mse": float(values["raw_mse"].mean()),
        "token_cosine": float(values["token_cosine"].mean()),
    }


def episode_cluster_bootstrap(
    conditioned_errors: np.ndarray,
    no_action_errors: np.ndarray,
    episode_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    episodes = np.unique(episode_ids)
    conditioned_sums = np.array(
        [conditioned_errors[episode_ids == episode].sum() for episode in episodes],
        dtype=np.float64,
    )
    no_action_sums = np.array(
        [no_action_errors[episode_ids == episode].sum() for episode in episodes],
        dtype=np.float64,
    )
    counts = np.array(
        [(episode_ids == episode).sum() for episode in episodes],
        dtype=np.int64,
    )
    episode_delta = no_action_sums / counts - conditioned_sums / counts
    rng = np.random.default_rng(seed)
    improvements = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected = rng.integers(0, len(episodes), len(episodes))
        conditioned_mse = conditioned_sums[selected].sum() / counts[selected].sum()
        no_action_mse = no_action_sums[selected].sum() / counts[selected].sum()
        improvements[sample_index] = 100 * (no_action_mse - conditioned_mse) / no_action_mse
    observed_conditioned = conditioned_sums.sum() / counts.sum()
    observed_no_action = no_action_sums.sum() / counts.sum()
    return {
        "episodes": len(episodes),
        "conditioned_better_episodes": int((episode_delta > 0).sum()),
        "conditioned_worse_episodes": int((episode_delta < 0).sum()),
        "aggregate_improvement_percent": float(100 * (observed_no_action - observed_conditioned) / observed_no_action),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "improvement_percent_ci95": [
            float(np.quantile(improvements, 0.025)),
            float(np.quantile(improvements, 0.975)),
        ],
        "bootstrap_probability_positive": float((improvements > 0).mean()),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("batch size and bootstrap samples must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    start = time.perf_counter()
    stage14_report = json.loads(args.stage14_report.read_text(encoding="utf-8"))
    if stage14_report["data"]["test_episodes_used"] != 0:
        raise AssertionError("Stage 14 report indicates test leakage")
    rows = STAGE14.read_csv(args.shard_manifest)
    if {row["split"] for row in rows} != {"train", "validation"}:
        raise AssertionError("Shard manifest must contain only train and validation")
    train_store = STAGE14.load_transition_store(rows, split="train", repo_root=Path.cwd())
    validation_store = STAGE14.load_transition_store(
        rows,
        split="validation",
        repo_root=Path.cwd(),
    )
    normalization = STAGE14.compute_normalization(train_store).to(device)
    hidden_dimension = int(stage14_report["model"]["hidden_dimension"])
    depth = int(stage14_report["model"]["depth"])
    models = {}
    for variant in STAGE14.VARIANTS:
        checkpoint = args.checkpoint_dir / f"{variant}_best.safetensors"
        if sha256_file(checkpoint) != stage14_report["variants"][variant]["checkpoint_sha256"]:
            raise AssertionError(f"Checkpoint hash mismatch for {variant}")
        model = STAGE14.TokenDynamicsModel(hidden_dimension, depth)
        model.load_state_dict(load_file(checkpoint))
        models[variant] = model.to(device)

    rollout_positions = common_rollout_positions(validation_store, MAX_HORIZON)
    if len(rollout_positions) != 4136:
        raise AssertionError(f"Expected 4136 common rollout windows, got {len(rollout_positions)}")
    metrics, episode_ids, task_ids = evaluate_rollouts(
        models["action_conditioned"],
        models["no_action"],
        validation_store,
        normalization,
        rollout_positions,
        device=device,
        batch_size=args.batch_size,
    )

    metric_rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for mode in ROLLOUT_MODES:
        aggregate[mode] = {}
        for horizon in HORIZONS:
            summary = aggregate_metrics(metrics[mode][horizon])
            aggregate[mode][str(horizon)] = summary
            metric_rows.append(
                {
                    "mode": mode,
                    "horizon": horizon,
                    "windows": len(rollout_positions),
                    **summary,
                }
            )

    comparisons = {}
    bootstrap = {}
    for horizon in HORIZONS:
        correct = aggregate["conditioned_correct"][str(horizon)]["raw_mse"]
        no_action = aggregate["no_action"][str(horizon)]["raw_mse"]
        zero = aggregate["conditioned_zero"][str(horizon)]["raw_mse"]
        shuffled = aggregate["conditioned_shuffled"][str(horizon)]["raw_mse"]
        persistence = aggregate["persistence"][str(horizon)]["raw_mse"]
        teacher = aggregate["conditioned_teacher_forced"][str(horizon)]["raw_mse"]
        comparisons[str(horizon)] = {
            "conditioned_vs_no_action_improvement_percent": 100 * (no_action - correct) / no_action,
            "correct_vs_zero_improvement_percent": 100 * (zero - correct) / zero,
            "correct_vs_shuffled_improvement_percent": 100 * (shuffled - correct) / shuffled,
            "correct_vs_persistence_improvement_percent": 100 * (persistence - correct) / persistence,
            "recursive_vs_teacher_forced_error_ratio": correct / teacher,
        }
        bootstrap[str(horizon)] = episode_cluster_bootstrap(
            metrics["conditioned_correct"][horizon]["raw_mse"],
            metrics["no_action"][horizon]["raw_mse"],
            episode_ids,
            samples=args.bootstrap_samples,
            seed=BOOTSTRAP_SEED + horizon,
        )

    task_rows = []
    for task_id in range(10):
        task_mask = task_ids == task_id
        for mode in (
            "conditioned_correct",
            "conditioned_zero",
            "conditioned_shuffled",
            "no_action",
            "persistence",
        ):
            values = metrics[mode][MAX_HORIZON]["raw_mse"][task_mask]
            task_rows.append(
                {
                    "benchmark_task_id": task_id,
                    "mode": mode,
                    "horizon": MAX_HORIZON,
                    "windows": int(task_mask.sum()),
                    "raw_mse": float(values.mean()),
                }
            )

    action_signal_all_horizons = all(
        comparisons[str(horizon)]["conditioned_vs_no_action_improvement_percent"] > 0
        and comparisons[str(horizon)]["correct_vs_zero_improvement_percent"] > 0
        and comparisons[str(horizon)]["correct_vs_shuffled_improvement_percent"] > 0
        for horizon in HORIZONS
    )
    bootstrap_positive_all_horizons = all(
        bootstrap[str(horizon)]["improvement_percent_ci95"][0] > 0 for horizon in HORIZONS
    )
    metrics_text = csv_text(metric_rows)
    task_text = csv_text(task_rows)
    bootstrap_text = canonical_json(
        {
            "schema_version": 1,
            "unit": "validation episode cluster",
            "common_windows": len(rollout_positions),
            "results": bootstrap,
        }
    )
    report = {
        "schema_version": 1,
        "stage": "WM Stage 15 recursive multi-step latent rollout",
        "status": "passed",
        "scope": {
            "model_parameters_updated": False,
            "checkpoint_source": "Stage 14 best validation checkpoints",
            "validation_episodes": 43,
            "common_windows": len(rollout_positions),
            "horizons": list(HORIZONS),
            "test_episodes_used": 0,
            "latent_is_recursive": True,
            "state_context": "oracle ground-truth state at every rollout step",
            "action_sequence": "ground-truth, normalized zero, or deterministic global shuffle",
            "action_shuffle_seed": ACTION_SHUFFLE_SEED,
        },
        "source": {
            "stage14_report": "results/wm/stage14_next_latent_baseline/report.json",
            "stage14_report_sha256": sha256_file(args.stage14_report),
            "conditioned_checkpoint_sha256": stage14_report["variants"]["action_conditioned"]["checkpoint_sha256"],
            "no_action_checkpoint_sha256": stage14_report["variants"]["no_action"]["checkpoint_sha256"],
        },
        "aggregate_metrics": aggregate,
        "paired_comparisons": comparisons,
        "bootstrap": bootstrap,
        "stage16_readiness": {
            "multi_step_rollout_evaluated": True,
            "action_signal_all_horizons": action_signal_all_horizons,
            "bootstrap_ci_positive_all_horizons": bootstrap_positive_all_horizons,
            "test_split_remains_held_out": True,
            "closed_loop_integration_started": False,
        },
        "artifacts": {
            "rollout_metrics": "rollout_metrics.csv",
            "rollout_metrics_sha256": sha256_text(metrics_text),
            "task_h25": "task_h25.csv",
            "task_h25_sha256": sha256_text(task_text),
            "paired_bootstrap": "paired_bootstrap.json",
            "paired_bootstrap_sha256": sha256_text(bootstrap_text),
        },
        "limitations": [
            "Ground-truth proprioceptive state is provided at every recursive step.",
            "Rollouts are offline on expert validation trajectories, not closed-loop simulation.",
            "Frozen DINOv2 latent error is not a calibrated pixel or task-success metric.",
        ],
    }
    write_deterministic(
        args.public_results_dir / "rollout_metrics.csv",
        metrics_text,
        overwrite=args.overwrite,
    )
    write_deterministic(
        args.public_results_dir / "task_h25.csv",
        task_text,
        overwrite=args.overwrite,
    )
    write_deterministic(
        args.public_results_dir / "paired_bootstrap.json",
        bootstrap_text,
        overwrite=args.overwrite,
    )
    write_deterministic(
        args.public_results_dir / "report.json",
        canonical_json(report),
        overwrite=args.overwrite,
    )
    performance_path = args.public_results_dir / "performance.json"
    if not performance_path.exists():
        write_deterministic(
            performance_path,
            canonical_json(
                {
                    "schema_version": 1,
                    "device": str(device),
                    "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
                    "batch_size": args.batch_size,
                    "evaluation_seconds": time.perf_counter() - start,
                    "peak_gpu_memory_bytes": (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0),
                }
            ),
            overwrite=False,
        )
    print(
        "Stage 15 passed: "
        + ", ".join(
            f"h{h}={comparisons[str(h)]['conditioned_vs_no_action_improvement_percent']:.3f}%" for h in HORIZONS
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
