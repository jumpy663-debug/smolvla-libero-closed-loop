#!/usr/bin/env python

"""Audit LIBERO Spatial demonstrations and existing SmolVLA rollout artifacts.

This stage creates deterministic episode-level train/validation/test splits and
publication-safe manifests. It does not decode images, extract visual features,
or train a world model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

DATASET_REVISION = "2e98211ba27db9322efbaa5ed108b34bb03ca163"
HORIZONS = (10, 25, 50)
SPLIT_NAMES = ("train", "validation", "test")
SPATIAL_TASKS = {
    0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
    1: "pick up the black bowl next to the ramekin and place it on the plate",
    2: "pick up the black bowl from table center and place it on the plate",
    3: "pick up the black bowl on the cookie box and place it on the plate",
    4: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    5: "pick up the black bowl on the ramekin and place it on the plate",
    6: "pick up the black bowl next to the cookie box and place it on the plate",
    7: "pick up the black bowl on the stove and place it on the plate",
    8: "pick up the black bowl next to the plate and place it on the plate",
    9: "pick up the black bowl on the wooden cabinet and place it on the plate",
}
HARD_BENCHMARK_TASKS = frozenset({4, 5, 7, 8})


def default_dataset_root() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    lerobot_home = Path(os.environ.get("HF_LEROBOT_HOME", hf_home / "lerobot"))
    return lerobot_home / "hub" / "datasets--lerobot--libero" / "snapshots" / DATASET_REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument(
        "--rollout-root",
        type=Path,
        required=True,
        help="Root containing the existing stage6/stage7/stage9/stage10 artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/wm/stage11_data_audit"),
    )
    parser.add_argument("--split-seed", type=int, default=1000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing Stage 11 output only when intentionally regenerating it.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stable_score(seed: int, task_index: int, episode_index: int) -> str:
    return sha256_text(f"{seed}:{task_index}:{episode_index}")


def assign_episode_splits(
    episodes_by_task: dict[int, list[int]],
    seed: int,
) -> dict[int, str]:
    assignments: dict[int, str] = {}
    for task_index, episode_ids in sorted(episodes_by_task.items()):
        ranked = sorted(
            episode_ids,
            key=lambda episode_id: stable_score(seed, task_index, episode_id),
        )
        validation_count = max(1, round(len(ranked) * 0.1))
        test_count = max(1, round(len(ranked) * 0.1))
        train_count = len(ranked) - validation_count - test_count
        if train_count <= 0:
            raise ValueError(f"Task {task_index} has too few episodes for an 80/10/10 split")
        split_by_rank = ["train"] * train_count + ["validation"] * validation_count + ["test"] * test_count
        for episode_id, split in zip(ranked, split_by_rank, strict=True):
            if episode_id in assignments:
                raise AssertionError(f"Episode {episode_id} was assigned more than once")
            assignments[episode_id] = split
    return assignments


def canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def csv_text(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_deterministic(path: Path, content: str, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content and not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-identical output: {path}")
        if existing == content:
            return
    path.write_text(content, encoding="utf-8")


def logical_path(path: Path, root: Path) -> str:
    # Keep the logical snapshot path instead of following Hugging Face cache
    # symlinks into the sibling blobs directory.
    return path.absolute().relative_to(root.absolute()).as_posix()


def inventory_digest(paths: list[Path], root: Path) -> tuple[str, int]:
    rows = []
    total_bytes = 0
    for path in sorted(paths):
        size = path.stat().st_size
        total_bytes += size
        resolved = path.resolve()
        rows.append(
            {
                "path": logical_path(path, root),
                "bytes": size,
                "blob": resolved.name,
            }
        )
    return sha256_text(canonical_json(rows)), total_bytes


def resolve_task_mapping(tasks: pd.DataFrame) -> dict[int, int]:
    prompt_to_dataset_index = {str(prompt): int(row["task_index"]) for prompt, row in tasks.iterrows()}
    missing = [prompt for prompt in SPATIAL_TASKS.values() if prompt not in prompt_to_dataset_index]
    if missing:
        raise AssertionError(f"LIBERO Spatial prompts missing from dataset metadata: {missing}")
    mapping = {
        benchmark_task_id: prompt_to_dataset_index[prompt] for benchmark_task_id, prompt in SPATIAL_TASKS.items()
    }
    if len(set(mapping.values())) != len(mapping):
        raise AssertionError(f"Spatial task mapping is not one-to-one: {mapping}")
    return mapping


def map_episode_data_files(dataset_root: Path) -> tuple[dict[int, str], list[Path]]:
    """Map each episode to the parquet fragment that actually contains it.

    The aggregated LIBERO snapshot currently has stale ``data/file_index``
    values in ``meta/episodes``. Reading the parquet fragments themselves keeps
    the published manifest usable without depending on that metadata field.
    """
    data_root = dataset_root / "data"
    parquet_dataset = pads.dataset(data_root, format="parquet")
    episode_to_file: dict[int, str] = {}
    data_files: list[Path] = []
    for fragment in parquet_dataset.get_fragments():
        path = Path(fragment.path)
        if not path.is_absolute():
            path = dataset_root / path
        data_files.append(path)
        episode_ids = set(fragment.to_table(columns=["episode_index"]).column("episode_index").to_pylist())
        for episode_id_value in episode_ids:
            episode_id = int(episode_id_value)
            if episode_id in episode_to_file:
                raise AssertionError(f"Episode {episode_id} spans multiple parquet files")
            episode_to_file[episode_id] = logical_path(path, dataset_root)
    return episode_to_file, data_files


def audit_expert_dataset(
    dataset_root: Path,
    dataset_revision: str,
    split_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[int]]]:
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    required_metadata = [info_path, episodes_path, tasks_path, dataset_root / "meta" / "stats.json"]
    missing_metadata = [path for path in required_metadata if not path.is_file()]
    if missing_metadata:
        raise FileNotFoundError(f"Dataset metadata is incomplete: {missing_metadata}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info["total_episodes"]) <= 0 or int(info["total_frames"]) <= 0:
        raise AssertionError(f"Invalid dataset totals: {info}")
    tasks = pd.read_parquet(tasks_path)
    episodes = pd.read_parquet(episodes_path)
    benchmark_to_dataset = resolve_task_mapping(tasks)
    dataset_to_benchmark = {value: key for key, value in benchmark_to_dataset.items()}

    parquet_dataset = pads.dataset(dataset_root / "data", format="parquet")
    frame_table = parquet_dataset.to_table(columns=["episode_index", "task_index", "frame_index"])
    frame_data = frame_table.to_pandas()
    frame_stats = (
        frame_data.groupby("episode_index")
        .agg(
            frame_count=("frame_index", "size"),
            frame_min=("frame_index", "min"),
            frame_max=("frame_index", "max"),
            task_index_min=("task_index", "min"),
            task_index_max=("task_index", "max"),
        )
        .reset_index()
    )
    if len(frame_stats) != int(info["total_episodes"]):
        raise AssertionError(f"Frame data covers {len(frame_stats)} episodes, expected {info['total_episodes']}")
    if not (frame_stats["task_index_min"] == frame_stats["task_index_max"]).all():
        raise AssertionError("At least one episode contains multiple task_index values")

    merged = episodes.merge(frame_stats, on="episode_index", validate="one_to_one")
    if not (merged["length"] == merged["frame_count"]).all():
        raise AssertionError("Episode metadata length disagrees with parquet frame count")
    if not (merged["frame_min"] == 0).all():
        raise AssertionError("At least one episode does not start at frame_index 0")
    if not (merged["frame_max"] == merged["length"] - 1).all():
        raise AssertionError("At least one episode has a non-contiguous final frame_index")

    spatial = merged[merged["task_index_min"].isin(dataset_to_benchmark)].copy()
    episodes_by_task = {
        int(task_index): [int(value) for value in group["episode_index"].tolist()]
        for task_index, group in spatial.groupby("task_index_min")
    }
    assignments = assign_episode_splits(episodes_by_task, split_seed)

    episode_to_data_file, data_files = map_episode_data_files(dataset_root)
    if set(episode_to_data_file) != set(int(value) for value in episodes["episode_index"]):
        raise AssertionError("Parquet episode set does not match meta/episodes")
    metadata_data_files = {
        int(row["episode_index"]): (
            f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
        )
        for _, row in episodes.iterrows()
    }
    metadata_data_file_mismatches = sum(
        episode_to_data_file[episode_id] != metadata_path for episode_id, metadata_path in metadata_data_files.items()
    )
    camera_keys = ["observation.images.image", "observation.images.image2"]
    video_files = set()
    for camera_key in camera_keys:
        for _, row in episodes.iterrows():
            video_files.add(
                dataset_root
                / f"videos/{camera_key}"
                / f"chunk-{int(row[f'videos/{camera_key}/chunk_index']):03d}"
                / f"file-{int(row[f'videos/{camera_key}/file_index']):03d}.mp4"
            )
    missing_files = [path for path in sorted(set(data_files) | video_files) if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Dataset snapshot is missing {len(missing_files)} referenced files; first missing path: {missing_files[0]}"
        )

    expert_rows: list[dict[str, Any]] = []
    split_ids: dict[str, list[int]] = {split: [] for split in SPLIT_NAMES}
    for _, row in spatial.sort_values("episode_index").iterrows():
        episode_index = int(row["episode_index"])
        dataset_task_index = int(row["task_index_min"])
        benchmark_task_id = dataset_to_benchmark[dataset_task_index]
        split = assignments[episode_index]
        split_ids[split].append(episode_index)
        record: dict[str, Any] = {
            "episode_index": episode_index,
            "split": split,
            "benchmark_task_id": benchmark_task_id,
            "dataset_task_index": dataset_task_index,
            "task_prompt": SPATIAL_TASKS[benchmark_task_id],
            "length": int(row["length"]),
            "data_file": episode_to_data_file[episode_index],
            "camera1_video_file": (
                "videos/observation.images.image/"
                f"chunk-{int(row['videos/observation.images.image/chunk_index']):03d}/"
                f"file-{int(row['videos/observation.images.image/file_index']):03d}.mp4"
            ),
            "camera1_from_seconds": float(row["videos/observation.images.image/from_timestamp"]),
            "camera1_to_seconds": float(row["videos/observation.images.image/to_timestamp"]),
            "camera2_video_file": (
                "videos/observation.images.image2/"
                f"chunk-{int(row['videos/observation.images.image2/chunk_index']):03d}/"
                f"file-{int(row['videos/observation.images.image2/file_index']):03d}.mp4"
            ),
            "camera2_from_seconds": float(row["videos/observation.images.image2/from_timestamp"]),
            "camera2_to_seconds": float(row["videos/observation.images.image2/to_timestamp"]),
        }
        for horizon in HORIZONS:
            record[f"valid_windows_h{horizon}"] = max(int(row["length"]) - horizon, 0)
        expert_rows.append(record)

    episode_ids_by_split = {split: set(values) for split, values in split_ids.items()}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            if episode_ids_by_split[left] & episode_ids_by_split[right]:
                raise AssertionError(f"Episode leakage between {left} and {right}")
    if set().union(*episode_ids_by_split.values()) != set(assignments):
        raise AssertionError("Split union does not match the Spatial episode set")

    spatial_frames = sum(int(row["length"]) for row in expert_rows)
    hard_frames = sum(
        int(row["length"]) for row in expert_rows if int(row["benchmark_task_id"]) in HARD_BENCHMARK_TASKS
    )
    bytes_per_frame = 2 * 16 * 384 * 2
    data_inventory_sha256, data_bytes = inventory_digest(data_files, dataset_root)
    video_inventory_sha256, video_bytes = inventory_digest(list(video_files), dataset_root)
    expert_summary = {
        "dataset": {
            "repo_id": "lerobot/libero",
            "revision": dataset_revision,
            "codebase_version": info["codebase_version"],
            "fps": info["fps"],
            "all_tasks": int(info["total_tasks"]),
            "all_episodes": int(info["total_episodes"]),
            "all_frames": int(info["total_frames"]),
            "snapshot_complete": True,
            "data_files": len(data_files),
            "video_files": len(video_files),
            "data_bytes": data_bytes,
            "video_bytes": video_bytes,
            "data_inventory_sha256": data_inventory_sha256,
            "video_inventory_sha256": video_inventory_sha256,
            "metadata_data_file_mismatches": metadata_data_file_mismatches,
            "data_file_mapping_source": "episode_index scanned from parquet fragments",
            "metadata_sha256": {logical_path(path, dataset_root): sha256_file(path) for path in required_metadata},
        },
        "spatial_subset": {
            "tasks": len(SPATIAL_TASKS),
            "episodes": len(expert_rows),
            "frames": spatial_frames,
            "hard_subset_tasks": sorted(HARD_BENCHMARK_TASKS),
            "hard_subset_episodes": sum(int(row["benchmark_task_id"]) in HARD_BENCHMARK_TASKS for row in expert_rows),
            "hard_subset_frames": hard_frames,
            "task_mapping": {
                str(benchmark_id): dataset_index for benchmark_id, dataset_index in sorted(benchmark_to_dataset.items())
            },
        },
        "splits": {
            split: {
                "episodes": sum(row["split"] == split for row in expert_rows),
                "frames": sum(int(row["length"]) for row in expert_rows if row["split"] == split),
                "valid_windows": {
                    str(horizon): sum(
                        int(row[f"valid_windows_h{horizon}"]) for row in expert_rows if row["split"] == split
                    )
                    for horizon in HORIZONS
                },
            }
            for split in SPLIT_NAMES
        },
        "per_task": {
            str(benchmark_task_id): {
                "dataset_task_index": benchmark_to_dataset[benchmark_task_id],
                "prompt": SPATIAL_TASKS[benchmark_task_id],
                "episodes": sum(int(row["benchmark_task_id"]) == benchmark_task_id for row in expert_rows),
                "frames": sum(
                    int(row["length"]) for row in expert_rows if int(row["benchmark_task_id"]) == benchmark_task_id
                ),
                "split_episodes": {
                    split: sum(
                        int(row["benchmark_task_id"]) == benchmark_task_id and row["split"] == split
                        for row in expert_rows
                    )
                    for split in SPLIT_NAMES
                },
            }
            for benchmark_task_id in sorted(SPATIAL_TASKS)
        },
        "feature_cache_budget": {
            "assumption": {
                "cameras": 2,
                "tokens_per_camera": 16,
                "latent_dimension": 384,
                "dtype": "float16",
                "bytes_per_frame": bytes_per_frame,
            },
            "all_spatial_bytes": spatial_frames * bytes_per_frame,
            "hard_subset_bytes": hard_frames * bytes_per_frame,
        },
        "leakage_checks": {
            "episode_level_split": True,
            "train_validation_disjoint": True,
            "train_test_disjoint": True,
            "validation_test_disjoint": True,
            "frame_indices_contiguous_within_episode": True,
            "note": (
                "LIBERO metadata does not expose simulator initial-state identifiers for expert "
                "demonstrations, so overlap with benchmark init-state IDs cannot be audited."
            ),
        },
    }
    return expert_rows, expert_summary, split_ids


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Expected True/False, got {value!r}")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rollout_sources(rollout_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "experiment": "stage6_full_spatial_benchmark",
            "policy": "pretrained",
            "csv": rollout_root / "stage6_full_spatial_benchmark" / "episodes.csv",
            "row_filter": lambda row: True,
            "horizon": lambda row: 50,
            "trajectory": lambda row: (
                rollout_root / "stage6_full_spatial_benchmark" / "trajectories" / f"task_{int(row['task_id']):02d}.npz"
            ),
            "video": lambda row: (
                rollout_root
                / "stage6_full_spatial_benchmark"
                / "videos"
                / f"task_{int(row['task_id']):02d}"
                / f"episode_{int(row['episode_index']):02d}.mp4"
            ),
        },
        {
            "experiment": "stage7_action_horizon_ablation",
            "policy": "pretrained",
            "csv": rollout_root / "stage7_action_horizon_ablation" / "episodes.csv",
            "row_filter": lambda row: row["source"] == "stage7_ablation",
            "horizon": lambda row: int(row["horizon"]),
            "trajectory": lambda row: (
                rollout_root
                / "stage7_action_horizon_ablation"
                / "trajectories"
                / f"horizon_{int(row['horizon'])}"
                / f"task_{int(row['task_id']):02d}.npz"
            ),
            "video": lambda row: (
                rollout_root
                / "stage7_action_horizon_ablation"
                / "videos"
                / f"horizon_{int(row['horizon'])}"
                / f"task_{int(row['task_id']):02d}"
                / f"episode_{int(row['episode_index']):02d}.mp4"
            ),
        },
        {
            "experiment": "stage9_task5_finetune_eval",
            "policy": "expert_only",
            "csv": rollout_root / "stage9_task5_finetune_eval" / "episodes.csv",
            "row_filter": lambda row: True,
            "horizon": lambda row: 10,
            "trajectory": lambda row: (
                rollout_root / "stage9_task5_finetune_eval" / "trajectories" / f"task_{int(row['task_id']):02d}.npz"
            ),
            "video": lambda row: (
                rollout_root
                / "stage9_task5_finetune_eval"
                / "videos"
                / f"task_{int(row['task_id']):02d}"
                / f"episode_{int(row['episode_index']):02d}.mp4"
            ),
        },
        {
            "experiment": "stage10_task5_lora_eval",
            "policy": "lora_rank16",
            "csv": rollout_root / "stage10_task5_lora_eval" / "episodes.csv",
            "row_filter": lambda row: True,
            "horizon": lambda row: 10,
            "trajectory": lambda row: (
                rollout_root / "stage10_task5_lora_eval" / "trajectories" / f"task_{int(row['task_id']):02d}.npz"
            ),
            "video": lambda row: (
                rollout_root
                / "stage10_task5_lora_eval"
                / "videos"
                / f"task_{int(row['task_id']):02d}"
                / f"episode_{int(row['episode_index']):02d}.mp4"
            ),
        },
    ]


def audit_rollouts(rollout_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    file_hashes: dict[Path, str] = {}
    duplicate_keys = set()
    selected_bytes = 0
    source_reports: dict[str, Any] = {}

    for source in rollout_sources(rollout_root):
        csv_path = source["csv"]
        report_path = csv_path.parent / "report.json"
        if not csv_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"Missing rollout source files under {csv_path.parent}")
        source_reports[source["experiment"]] = {
            "episodes_csv_sha256": sha256_file(csv_path),
            "report_sha256": sha256_file(report_path),
        }
        for csv_row in load_csv(csv_path):
            if not source["row_filter"](csv_row):
                continue
            task_id = int(csv_row["task_id"])
            episode_index = int(csv_row["episode_index"])
            horizon = int(source["horizon"](csv_row))
            trajectory_path = source["trajectory"](csv_row)
            video_path = source["video"](csv_row)
            if not trajectory_path.is_file() or not video_path.is_file():
                raise FileNotFoundError(
                    f"Missing trajectory/video for {source['experiment']} task={task_id} episode={episode_index}"
                )
            with np.load(trajectory_path) as trajectory:
                required_arrays = {"actions", "rewards", "successes", "dones", "done_indices"}
                if not required_arrays.issubset(trajectory.files):
                    raise AssertionError(f"{trajectory_path} is missing {required_arrays - set(trajectory.files)}")
                actions = trajectory["actions"]
                rewards = trajectory["rewards"]
                done_indices = trajectory["done_indices"]
                if actions.ndim != 3 or actions.shape[-1] != 7:
                    raise AssertionError(f"Unexpected action shape in {trajectory_path}: {actions.shape}")
                if not np.isfinite(actions).all() or not np.isfinite(rewards).all():
                    raise AssertionError(f"Non-finite rollout arrays in {trajectory_path}")
                if episode_index >= actions.shape[0]:
                    raise AssertionError(f"Episode slot {episode_index} missing in {trajectory_path}")
                steps = int(csv_row["steps"])
                if int(done_indices[episode_index]) + 1 != steps:
                    raise AssertionError(f"CSV/trajectory step mismatch in {trajectory_path}, slot {episode_index}")

            key = (
                source["policy"],
                horizon,
                task_id,
                int(csv_row["init_state_index"]),
                int(csv_row["seed"]),
            )
            if key in duplicate_keys:
                raise AssertionError(f"Duplicate canonical rollout key: {key}")
            duplicate_keys.add(key)
            for path in (trajectory_path, video_path):
                if path not in file_hashes:
                    file_hashes[path] = sha256_file(path)
                    selected_bytes += path.stat().st_size
            rows.append(
                {
                    "rollout_id": (
                        f"{source['policy']}-h{horizon}-task{task_id:02d}-init{int(csv_row['init_state_index']):02d}"
                    ),
                    "source_experiment": source["experiment"],
                    "policy": source["policy"],
                    "horizon": horizon,
                    "benchmark_task_id": task_id,
                    "task_name": csv_row["task_name"],
                    "episode_index": episode_index,
                    "init_state_index": int(csv_row["init_state_index"]),
                    "seed": int(csv_row["seed"]),
                    "success": parse_bool(csv_row["success"]),
                    "steps": steps,
                    "trajectory_file": logical_path(trajectory_path, rollout_root),
                    "trajectory_slot": episode_index,
                    "trajectory_sha256": file_hashes[trajectory_path],
                    "render_video_file": logical_path(video_path, rollout_root),
                    "render_video_sha256": file_hashes[video_path],
                    "has_actions": True,
                    "has_rewards": True,
                    "has_render_video": True,
                    "has_observation_state": False,
                    "has_dual_camera_observations": False,
                    "wm_v1_eligible": False,
                    "visual_action_only_candidate": True,
                }
            )

    rows.sort(
        key=lambda row: (
            str(row["policy"]),
            int(row["horizon"]),
            int(row["benchmark_task_id"]),
            int(row["init_state_index"]),
        )
    )
    counts = Counter((str(row["policy"]), int(row["horizon"])) for row in rows)
    successes = Counter((str(row["policy"]), int(row["horizon"])) for row in rows if bool(row["success"]))
    per_policy_horizon = {
        f"{policy}:h{horizon}": {
            "episodes": counts[(policy, horizon)],
            "successes": successes[(policy, horizon)],
        }
        for policy, horizon in sorted(counts)
    }
    rollout_summary = {
        "episodes": len(rows),
        "unique_canonical_keys": len(duplicate_keys),
        "selected_artifact_bytes": selected_bytes,
        "per_policy_horizon": per_policy_horizon,
        "source_reports": source_reports,
        "modality_audit": {
            "actions": len(rows),
            "render_video": len(rows),
            "observation_state": 0,
            "dual_camera_observations": 0,
            "wm_v1_eligible": 0,
            "visual_action_only_candidates": len(rows),
        },
        "decision": (
            "Existing rollout artifacts are excluded from WM-v1 training because they do not "
            "contain synchronized proprioception and dual-camera policy observations. They remain "
            "usable for visual-action-only pilots and as a deterministic recollection index."
        ),
    }
    return rows, rollout_summary


def main() -> None:
    args = parse_args()
    if args.split_seed < 0:
        raise ValueError("--split-seed must be non-negative")
    if args.dataset_revision != args.dataset_root.name:
        raise ValueError(
            f"Dataset revision {args.dataset_revision} does not match snapshot directory {args.dataset_root.name}"
        )

    expert_rows, expert_summary, split_ids = audit_expert_dataset(
        args.dataset_root,
        args.dataset_revision,
        args.split_seed,
    )
    rollout_rows, rollout_summary = audit_rollouts(args.rollout_root)

    expert_fields = list(expert_rows[0])
    rollout_fields = list(rollout_rows[0])
    expert_csv = csv_text(expert_rows, expert_fields)
    rollout_csv = csv_text(rollout_rows, rollout_fields)
    splits_text = canonical_json(
        {
            "schema_version": 1,
            "seed": args.split_seed,
            "strategy": "per-task deterministic SHA-256 ranking with 80/10/10 episode split",
            "episode_ids": split_ids,
        }
    )
    report = {
        "schema_version": 1,
        "stage": "WM Stage 11 dataset audit and episode-level split",
        "status": "passed",
        "horizons": list(HORIZONS),
        "expert_demonstrations": expert_summary,
        "existing_rollouts": rollout_summary,
        "published_manifests": {
            "expert_episodes": "expert_episodes.csv",
            "expert_episodes_sha256": sha256_text(expert_csv),
            "rollout_episodes": "rollout_episodes.csv",
            "rollout_episodes_sha256": sha256_text(rollout_csv),
            "splits": "splits.json",
            "splits_sha256": sha256_text(splits_text),
        },
        "stage12_readiness": {
            "expert_demonstrations_ready": True,
            "existing_rollouts_ready_for_wm_v1": False,
            "feature_extraction_can_start_without_rollout_recollection": True,
            "recommended_scope": (
                "Start representation pilot on expert LIBERO Spatial demonstrations; defer "
                "rollout recollection until the frozen feature representation is selected."
            ),
        },
    }
    report_text = canonical_json(report)

    write_deterministic(args.output_dir / "expert_episodes.csv", expert_csv, overwrite=args.overwrite)
    write_deterministic(args.output_dir / "rollout_episodes.csv", rollout_csv, overwrite=args.overwrite)
    write_deterministic(args.output_dir / "splits.json", splits_text, overwrite=args.overwrite)
    write_deterministic(args.output_dir / "report.json", report_text, overwrite=args.overwrite)
    print(
        "Stage 11 passed: "
        f"expert_episodes={len(expert_rows)}, rollout_episodes={len(rollout_rows)}, "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
