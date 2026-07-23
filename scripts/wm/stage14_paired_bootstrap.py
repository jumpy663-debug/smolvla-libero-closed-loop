#!/usr/bin/env python

"""Run episode-cluster bootstrap for the paired Stage 14 validation errors."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

BOOTSTRAP_SEED = 1423
BOOTSTRAP_SAMPLES = 10_000


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
        "--output",
        type=Path,
        default=Path("results/wm/stage14_next_latent_baseline/paired_bootstrap.json"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def paired_episode_error_sums(
    conditioned_model: torch.nn.Module,
    no_action_model: torch.nn.Module,
    store: Any,
    normalization: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    episodes = sorted(int(value) for value in torch.unique(store.episode_ids))
    episode_to_position = {episode_id: index for index, episode_id in enumerate(episodes)}
    conditioned_sums = np.zeros(len(episodes), dtype=np.float64)
    no_action_sums = np.zeros(len(episodes), dtype=np.float64)
    element_counts = np.zeros(len(episodes), dtype=np.int64)

    conditioned_model.eval()
    no_action_model.eval()
    with torch.inference_mode():
        for start in range(0, store.transition_count, batch_size):
            positions = torch.arange(start, min(start + batch_size, store.transition_count))
            current, target, state, action = STAGE14.normalized_batch(
                store,
                positions,
                normalization,
                device,
                action_mode="correct",
            )
            target_raw = STAGE14.raw_latent(target, normalization)
            conditioned_raw = STAGE14.raw_latent(
                conditioned_model(current, state, action),
                normalization,
            )
            no_action_raw = STAGE14.raw_latent(
                no_action_model(current, state, torch.zeros_like(action)),
                normalization,
            )
            conditioned_per_transition = (conditioned_raw - target_raw).double().square().sum(dim=(1, 2)).cpu().numpy()
            no_action_per_transition = (no_action_raw - target_raw).double().square().sum(dim=(1, 2)).cpu().numpy()
            episode_ids = store.episode_ids[positions].numpy()
            elements_per_transition = STAGE14.TOKENS_PER_FRAME * STAGE14.LATENT_DIMENSION
            for episode_id in np.unique(episode_ids):
                output_position = episode_to_position[int(episode_id)]
                mask = episode_ids == episode_id
                conditioned_sums[output_position] += conditioned_per_transition[mask].sum()
                no_action_sums[output_position] += no_action_per_transition[mask].sum()
                element_counts[output_position] += int(mask.sum()) * elements_per_transition
    return conditioned_sums, no_action_sums, element_counts, episodes


def cluster_bootstrap(
    conditioned_sums: np.ndarray,
    no_action_sums: np.ndarray,
    element_counts: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0 or seed < 0:
        raise ValueError("Bootstrap samples must be positive and seed non-negative")
    if not (len(conditioned_sums) == len(no_action_sums) == len(element_counts) and len(conditioned_sums) > 0):
        raise ValueError("Episode arrays must have equal non-zero lengths")
    rng = np.random.default_rng(seed)
    episode_count = len(conditioned_sums)
    improvements = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected = rng.integers(0, episode_count, episode_count)
        conditioned_mse = conditioned_sums[selected].sum() / element_counts[selected].sum()
        no_action_mse = no_action_sums[selected].sum() / element_counts[selected].sum()
        improvements[sample_index] = 100 * (no_action_mse - conditioned_mse) / no_action_mse
    conditioned_episode_mse = conditioned_sums / element_counts
    no_action_episode_mse = no_action_sums / element_counts
    episode_deltas = no_action_episode_mse - conditioned_episode_mse
    conditioned_mse = conditioned_sums.sum() / element_counts.sum()
    no_action_mse = no_action_sums.sum() / element_counts.sum()
    return {
        "episodes": episode_count,
        "conditioned_raw_mse_float64": float(conditioned_mse),
        "no_action_raw_mse_float64": float(no_action_mse),
        "aggregate_improvement_percent_float64": float(100 * (no_action_mse - conditioned_mse) / no_action_mse),
        "conditioned_better_episodes": int((episode_deltas > 0).sum()),
        "conditioned_worse_episodes": int((episode_deltas < 0).sum()),
        "tied_episodes": int((episode_deltas == 0).sum()),
        "median_episode_raw_mse_delta": float(np.median(episode_deltas)),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "improvement_percent_mean": float(improvements.mean()),
        "improvement_percent_ci95": [
            float(np.quantile(improvements, 0.025)),
            float(np.quantile(improvements, 0.975)),
        ],
        "bootstrap_probability_positive": float((improvements > 0).mean()),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    report = json.loads(args.stage14_report.read_text(encoding="utf-8"))
    if report["data"]["test_episodes_used"] != 0:
        raise AssertionError("Stage 14 report indicates test leakage")
    hidden_dimension = int(report["model"]["hidden_dimension"])
    depth = int(report["model"]["depth"])
    rows = STAGE14.read_csv(args.shard_manifest)
    train_store = STAGE14.load_transition_store(rows, split="train", repo_root=Path.cwd())
    validation_store = STAGE14.load_transition_store(
        rows,
        split="validation",
        repo_root=Path.cwd(),
    )
    normalization = STAGE14.compute_normalization(train_store).to(device)
    models = {}
    for variant in STAGE14.VARIANTS:
        checkpoint = args.checkpoint_dir / f"{variant}_best.safetensors"
        expected_hash = report["variants"][variant]["checkpoint_sha256"]
        if STAGE14.sha256_file(checkpoint) != expected_hash:
            raise AssertionError(f"Checkpoint hash mismatch for {variant}")
        model = STAGE14.TokenDynamicsModel(hidden_dimension, depth)
        model.load_state_dict(load_file(checkpoint))
        models[variant] = model.to(device)

    start = time.perf_counter()
    conditioned_sums, no_action_sums, counts, episodes = paired_episode_error_sums(
        models["action_conditioned"],
        models["no_action"],
        validation_store,
        normalization,
        device=device,
        batch_size=args.batch_size,
    )
    analysis = cluster_bootstrap(
        conditioned_sums,
        no_action_sums,
        counts,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    if analysis["conditioned_better_episodes"] + analysis["conditioned_worse_episodes"] != len(episodes):
        raise AssertionError("Unexpected tied episode in paired analysis")
    output = {
        "schema_version": 1,
        "stage": "WM Stage 14 episode-cluster bootstrap",
        "status": "passed",
        "source": {
            "stage14_report": "results/wm/stage14_next_latent_baseline/report.json",
            "stage14_report_sha256": STAGE14.sha256_file(args.stage14_report),
            "conditioned_checkpoint_sha256": report["variants"]["action_conditioned"]["checkpoint_sha256"],
            "no_action_checkpoint_sha256": report["variants"]["no_action"]["checkpoint_sha256"],
            "validation_episodes": len(episodes),
            "validation_transitions": validation_store.transition_count,
            "test_episodes_used": 0,
        },
        "method": {
            "unit": "episode cluster",
            "resampling": "sample 43 validation episodes with replacement",
            "metric": "frame-and-token-weighted raw MSE improvement percent",
            "float_reduction": "float64",
        },
        "result": analysis,
        "interpretation": (
            "The paired action-conditioned gain is positive across the 95% episode-cluster "
            "bootstrap interval, but remains small in absolute percentage terms."
        ),
    }
    STAGE14.write_deterministic(
        args.output,
        STAGE14.canonical_json(output),
        overwrite=args.overwrite,
    )
    print(
        "Stage 14 bootstrap passed: "
        f"wins={analysis['conditioned_better_episodes']}/{analysis['episodes']}, "
        f"ci95={analysis['improvement_percent_ci95']}, "
        f"seconds={time.perf_counter() - start:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
