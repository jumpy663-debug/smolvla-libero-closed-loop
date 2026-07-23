#!/usr/bin/env python

"""Extract resumable full DINOv2 features for LIBERO Spatial train/validation.

Each episode is written as one atomic Safetensors shard. Existing valid shards
are verified and skipped, enabling safe resumption after interruption. The test
split is intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import safetensors
import torch
import transformers
from lerobot.datasets.video_utils import decode_video_frames
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoImageProcessor, Dinov2Model


def load_stage12_module() -> ModuleType:
    script_path = Path(__file__).with_name("stage12_representation_pilot.py")
    spec = importlib.util.spec_from_file_location("stage12_representation_pilot", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import shared representation helpers from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE12 = load_stage12_module()
DATASET_REVISION = STAGE12.DATASET_REVISION
MODEL_ID = STAGE12.MODEL_ID
MODEL_REVISION = STAGE12.MODEL_REVISION
TOKENS_PER_CAMERA = STAGE12.TOKENS_PER_CAMERA
LATENT_DIMENSION = STAGE12.LATENT_DIMENSION
CACHE_SPLITS = ("train", "validation")
SHARD_SCHEMA_VERSION = "1"
EXPECTED_KEYS = {
    "visual_tokens",
    "observation_state",
    "action",
    "dataset_index",
    "frame_index",
    "timestamp",
    "is_terminal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=STAGE12.default_dataset_root())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/wm/stage11_data_audit/expert_episodes.csv"),
    )
    parser.add_argument("--model-root", type=Path, default=STAGE12.default_model_root())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wm/stage13_full_feature_cache"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage13_full_feature_cache"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--recheck-seed", type=int, default=1300)
    parser.add_argument(
        "--overwrite-invalid",
        action="store_true",
        help="Replace an existing shard only if it fails validation.",
    )
    parser.add_argument(
        "--overwrite-public-results",
        action="store_true",
        help="Replace non-identical public manifest/report files.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def shard_relative_path(episode_index: int) -> Path:
    return Path("shards") / f"episode_{episode_index:06d}.safetensors"


def expected_metadata(row: dict[str, str], pixel_sha256: str | None = None) -> dict[str, str]:
    metadata = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "dataset_repo_id": "lerobot/libero",
        "dataset_revision": DATASET_REVISION,
        "encoder_model_id": MODEL_ID,
        "encoder_revision": MODEL_REVISION,
        "visual_token_layout": "frame,camera,spatial_token,channel",
        "split": row["split"],
        "benchmark_task_id": str(int(row["benchmark_task_id"])),
        "dataset_task_index": str(int(row["dataset_task_index"])),
        "episode_index": str(int(row["episode_index"])),
        "length": str(int(row["length"])),
        "task_prompt": row["task_prompt"],
    }
    if pixel_sha256 is not None:
        metadata["decoded_pixels_sha256"] = pixel_sha256
    return metadata


def validate_cache_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No cache rows selected")
    episode_ids = [int(row["episode_index"]) for row in rows]
    if len(episode_ids) != len(set(episode_ids)):
        raise AssertionError("Duplicate episode IDs in cache selection")
    if {row["split"] for row in rows} != set(CACHE_SPLITS):
        raise AssertionError(f"Cache selection must contain exactly {CACHE_SPLITS}")
    if any(row["split"] == "test" for row in rows):
        raise AssertionError("Test episodes must not be extracted")
    for split in CACHE_SPLITS:
        task_ids = {int(row["benchmark_task_id"]) for row in rows if row["split"] == split}
        if task_ids != set(range(10)):
            raise AssertionError(f"Split {split} does not cover benchmark tasks 0-9: {task_ids}")


def load_episode(
    dataset_root: Path,
    row: dict[str, str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, str]:
    episode_index = int(row["episode_index"])
    length = int(row["length"])
    table = pq.read_table(
        dataset_root / row["data_file"],
        columns=[
            "episode_index",
            "frame_index",
            "timestamp",
            "index",
            "observation.state",
            "action",
        ],
        filters=[("episode_index", "=", episode_index)],
    )
    records = sorted(table.to_pylist(), key=lambda record: int(record["frame_index"]))
    if len(records) != length:
        raise AssertionError(f"Episode {episode_index} has {len(records)} rows, expected {length}")
    frame_indices = [int(record["frame_index"]) for record in records]
    if frame_indices != list(range(length)):
        raise AssertionError(f"Episode {episode_index} frame indices are not contiguous")
    timestamps = [float(record["timestamp"]) for record in records]

    cameras: list[torch.Tensor] = []
    for camera_index in (1, 2):
        query_timestamps = [float(row[f"camera{camera_index}_from_seconds"]) + timestamp for timestamp in timestamps]
        frames = decode_video_frames(
            dataset_root / row[f"camera{camera_index}_video_file"],
            query_timestamps,
            tolerance_s=STAGE12.VIDEO_TOLERANCE_SECONDS,
            backend="pyav",
            return_uint8=True,
        )
        expected_shape = (length, 3, 256, 256)
        if tuple(frames.shape) != expected_shape or frames.dtype != torch.uint8:
            raise AssertionError(
                f"Episode {episode_index} camera {camera_index}: "
                f"{tuple(frames.shape)} {frames.dtype}, expected {expected_shape} uint8"
            )
        cameras.append(frames)
    images = torch.stack(cameras, dim=1)
    pixel_sha256 = sha256_bytes(images.contiguous().numpy().tobytes())

    tensors = {
        "observation_state": torch.tensor(
            [record["observation.state"] for record in records],
            dtype=torch.float32,
        ),
        "action": torch.tensor(
            [record["action"] for record in records],
            dtype=torch.float32,
        ),
        "dataset_index": torch.tensor(
            [int(record["index"]) for record in records],
            dtype=torch.int64,
        ),
        "frame_index": torch.tensor(frame_indices, dtype=torch.int64),
        "timestamp": torch.tensor(timestamps, dtype=torch.float32),
        "is_terminal": torch.arange(length, dtype=torch.int64) == length - 1,
    }
    if tensors["observation_state"].shape != (length, 8):
        raise AssertionError(f"Episode {episode_index} has invalid state shape")
    if tensors["action"].shape != (length, 7):
        raise AssertionError(f"Episode {episode_index} has invalid action shape")
    if not torch.isfinite(tensors["observation_state"]).all():
        raise AssertionError(f"Episode {episode_index} has non-finite states")
    if not torch.isfinite(tensors["action"]).all():
        raise AssertionError(f"Episode {episode_index} has non-finite actions")
    return tensors, images, pixel_sha256


def encode_images(
    images: torch.Tensor,
    processor: Any,
    model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    flat_images = images.reshape(-1, *images.shape[2:])
    output_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(flat_images), batch_size):
            pixels = STAGE12.prepare_pixels(
                processor,
                flat_images[start : start + batch_size],
            ).to(device)
            hidden = model(pixel_values=pixels).last_hidden_state
            output_batches.append(STAGE12.pool_patch_tokens(hidden).cpu())
    flat_tokens = torch.cat(output_batches, dim=0)
    tokens = flat_tokens.reshape(
        len(images),
        2,
        TOKENS_PER_CAMERA,
        LATENT_DIMENSION,
    ).to(torch.float16)
    if not torch.isfinite(tokens).all():
        raise AssertionError("Encoded visual tokens contain non-finite values")
    return tokens


def atomic_save_shard(
    path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    save_file(tensors, temporary_path, metadata=metadata)
    os.replace(temporary_path, path)


def validate_shard(
    path: Path,
    row: dict[str, str],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors = load_file(path)
    if set(tensors) != EXPECTED_KEYS:
        raise AssertionError(f"{path} has unexpected tensor keys: {set(tensors)}")
    length = int(row["length"])
    expected_shapes = {
        "visual_tokens": (length, 2, TOKENS_PER_CAMERA, LATENT_DIMENSION),
        "observation_state": (length, 8),
        "action": (length, 7),
        "dataset_index": (length,),
        "frame_index": (length,),
        "timestamp": (length,),
        "is_terminal": (length,),
    }
    for key, shape in expected_shapes.items():
        if tuple(tensors[key].shape) != shape:
            raise AssertionError(f"{path} tensor {key} shape {tuple(tensors[key].shape)} != {shape}")
    expected_dtypes = {
        "visual_tokens": torch.float16,
        "observation_state": torch.float32,
        "action": torch.float32,
        "dataset_index": torch.int64,
        "frame_index": torch.int64,
        "timestamp": torch.float32,
        "is_terminal": torch.bool,
    }
    for key, dtype in expected_dtypes.items():
        if tensors[key].dtype != dtype:
            raise AssertionError(f"{path} tensor {key} dtype {tensors[key].dtype} != {dtype}")
    if not torch.equal(tensors["frame_index"], torch.arange(length, dtype=torch.int64)):
        raise AssertionError(f"{path} frame indices are not contiguous")
    if int(tensors["is_terminal"].sum()) != 1 or not bool(tensors["is_terminal"][-1]):
        raise AssertionError(f"{path} terminal mask is invalid")
    for key in ("visual_tokens", "observation_state", "action", "timestamp"):
        if not torch.isfinite(tensors[key]).all():
            raise AssertionError(f"{path} tensor {key} contains non-finite values")
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
    if metadata is None:
        raise AssertionError(f"{path} has no metadata")
    expected = expected_metadata(row)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise AssertionError(f"{path} metadata {key}={metadata.get(key)!r} != {value!r}")
    pixel_sha256 = metadata.get("decoded_pixels_sha256", "")
    if len(pixel_sha256) != 64:
        raise AssertionError(f"{path} has invalid decoded pixel digest")
    return tensors, metadata


@dataclass
class FeatureStatsAccumulator:
    element_count: int = 0
    value_sum: float = 0.0
    value_square_sum: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    sample_count: int = 0
    embedding_sum: np.ndarray | None = None
    embedding_square_sum: np.ndarray | None = None
    camera_cosine_sum: float = 0.0
    camera_cosine_minimum: float = float("inf")
    camera_cosine_maximum: float = float("-inf")

    def update(self, visual_tokens: torch.Tensor) -> None:
        values = visual_tokens.numpy().astype(np.float64)
        self.element_count += values.size
        self.value_sum += float(values.sum())
        self.value_square_sum += float(np.square(values).sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        sample_embeddings = values.mean(axis=(1, 2))
        if self.embedding_sum is None:
            self.embedding_sum = np.zeros(sample_embeddings.shape[1], dtype=np.float64)
            self.embedding_square_sum = np.zeros(sample_embeddings.shape[1], dtype=np.float64)
        self.embedding_sum += sample_embeddings.sum(axis=0)
        self.embedding_square_sum += np.square(sample_embeddings).sum(axis=0)
        self.sample_count += len(sample_embeddings)
        camera_embeddings = values.mean(axis=2)
        dot = (camera_embeddings[:, 0] * camera_embeddings[:, 1]).sum(axis=1)
        norms = np.linalg.norm(camera_embeddings[:, 0], axis=1) * np.linalg.norm(
            camera_embeddings[:, 1],
            axis=1,
        )
        cosine = dot / np.maximum(norms, 1e-12)
        self.camera_cosine_sum += float(cosine.sum())
        self.camera_cosine_minimum = min(self.camera_cosine_minimum, float(cosine.min()))
        self.camera_cosine_maximum = max(self.camera_cosine_maximum, float(cosine.max()))

    def finalize(self) -> dict[str, float | bool]:
        if self.element_count == 0 or self.sample_count == 0:
            raise ValueError("No feature statistics were accumulated")
        if self.embedding_sum is None or self.embedding_square_sum is None:
            raise AssertionError("Embedding accumulators are missing")
        mean = self.value_sum / self.element_count
        variance = max(self.value_square_sum / self.element_count - mean * mean, 0.0)
        embedding_mean = self.embedding_sum / self.sample_count
        embedding_variance = np.maximum(
            self.embedding_square_sum / self.sample_count - np.square(embedding_mean),
            0.0,
        )
        return {
            "finite": True,
            "minimum": round(self.minimum, 8),
            "maximum": round(self.maximum, 8),
            "mean": round(mean, 8),
            "standard_deviation": round(float(np.sqrt(variance)), 8),
            "mean_channel_standard_deviation": round(float(np.sqrt(embedding_variance).mean()), 8),
            "camera_embedding_cosine_mean": round(
                self.camera_cosine_sum / self.sample_count,
                8,
            ),
            "camera_embedding_cosine_minimum": round(self.camera_cosine_minimum, 8),
            "camera_embedding_cosine_maximum": round(self.camera_cosine_maximum, 8),
        }


def select_recheck_rows(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    selected = []
    for split in CACHE_SPLITS:
        candidates = [row for row in rows if row["split"] == split]
        selected.append(
            min(
                candidates,
                key=lambda row: sha256_bytes(f"{seed}:{split}:{row['episode_index']}".encode()),
            )
        )
    return selected


def write_progress(
    output_dir: Path,
    *,
    completed: int,
    total: int,
    new_shards: int,
    resumed_shards: int,
    last_episode: int,
    elapsed_seconds: float,
) -> None:
    progress = {
        "completed": completed,
        "total": total,
        "new_shards": new_shards,
        "resumed_shards": resumed_shards,
        "last_episode": last_episode,
        "elapsed_seconds": elapsed_seconds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_dir / "progress.json.tmp"
    temporary_path.write_text(canonical_json(progress), encoding="utf-8")
    os.replace(temporary_path, output_dir / "progress.json")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.recheck_seed < 0:
        raise ValueError("--recheck-seed must be non-negative")
    if args.dataset_root.name != DATASET_REVISION:
        raise ValueError(f"Dataset snapshot must be revision {DATASET_REVISION}")
    if args.model_root.name != MODEL_REVISION:
        raise ValueError(f"Model snapshot must be revision {MODEL_REVISION}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    manifest_rows = read_csv(args.manifest)
    cache_rows = sorted(
        [row for row in manifest_rows if row["split"] in CACHE_SPLITS],
        key=lambda row: int(row["episode_index"]),
    )
    test_rows = [row for row in manifest_rows if row["split"] == "test"]
    validate_cache_rows(cache_rows)
    if len(cache_rows) != 389 or sum(int(row["length"]) for row in cache_rows) != 47822:
        raise AssertionError("Stage 11 train/validation totals changed unexpectedly")
    if len(test_rows) != 43:
        raise AssertionError("Stage 11 test split changed unexpectedly")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    processor = AutoImageProcessor.from_pretrained(args.model_root, local_files_only=True)
    model = Dinov2Model.from_pretrained(args.model_root, local_files_only=True).eval()
    model.requires_grad_(False)
    model.to(device)

    run_start = time.perf_counter()
    decoding_seconds = 0.0
    encoding_seconds = 0.0
    new_shards = 0
    resumed_shards = 0
    shard_rows: list[dict[str, Any]] = []
    stats = FeatureStatsAccumulator()

    for position, row in enumerate(cache_rows, start=1):
        episode_index = int(row["episode_index"])
        relative_path = shard_relative_path(episode_index)
        shard_path = args.output_dir / relative_path
        tensors: dict[str, torch.Tensor]
        metadata: dict[str, str]
        if shard_path.exists():
            try:
                tensors, metadata = validate_shard(shard_path, row)
                resumed_shards += 1
            except (AssertionError, RuntimeError, ValueError) as error:
                if not args.overwrite_invalid:
                    raise RuntimeError(
                        f"Invalid existing shard {shard_path}; rerun with --overwrite-invalid"
                    ) from error
                shard_path.unlink()
                tensors = {}
                metadata = {}
        else:
            tensors = {}
            metadata = {}

        if not tensors:
            decode_start = time.perf_counter()
            tensors, images, pixel_sha256 = load_episode(args.dataset_root, row)
            decoding_seconds += time.perf_counter() - decode_start
            encode_start = time.perf_counter()
            tensors["visual_tokens"] = encode_images(
                images,
                processor,
                model,
                device=device,
                batch_size=args.batch_size,
            )
            encoding_seconds += time.perf_counter() - encode_start
            metadata = expected_metadata(row, pixel_sha256)
            atomic_save_shard(shard_path, tensors, metadata)
            tensors, metadata = validate_shard(shard_path, row)
            new_shards += 1

        stats.update(tensors["visual_tokens"])
        shard_rows.append(
            {
                "episode_index": episode_index,
                "split": row["split"],
                "benchmark_task_id": int(row["benchmark_task_id"]),
                "dataset_task_index": int(row["dataset_task_index"]),
                "length": int(row["length"]),
                "shard_file": (f"outputs/wm/stage13_full_feature_cache/{relative_path.as_posix()}"),
                "shard_bytes": shard_path.stat().st_size,
                "shard_sha256": sha256_file(shard_path),
                "decoded_pixels_sha256": metadata["decoded_pixels_sha256"],
                "source_data_file": row["data_file"],
                "camera1_video_file": row["camera1_video_file"],
                "camera2_video_file": row["camera2_video_file"],
            }
        )
        write_progress(
            args.output_dir,
            completed=position,
            total=len(cache_rows),
            new_shards=new_shards,
            resumed_shards=resumed_shards,
            last_episode=episode_index,
            elapsed_seconds=time.perf_counter() - run_start,
        )
        if position == 1 or position % 10 == 0 or position == len(cache_rows):
            print(
                f"[{position}/{len(cache_rows)}] episode={episode_index} new={new_shards} resumed={resumed_shards}",
                flush=True,
            )

    recheck_rows = select_recheck_rows(cache_rows, args.recheck_seed)
    recheck_results = []
    for row in recheck_rows:
        episode_index = int(row["episode_index"])
        source_tensors, images, pixel_sha256 = load_episode(args.dataset_root, row)
        visual_tokens = encode_images(
            images,
            processor,
            model,
            device=device,
            batch_size=args.batch_size,
        )
        cached, metadata = validate_shard(
            args.output_dir / shard_relative_path(episode_index),
            row,
        )
        source_tensors["visual_tokens"] = visual_tokens
        tensor_exact = all(torch.equal(source_tensors[key], cached[key]) for key in EXPECTED_KEYS)
        pixel_exact = pixel_sha256 == metadata["decoded_pixels_sha256"]
        if not tensor_exact or not pixel_exact:
            raise AssertionError(
                f"Deterministic recheck failed for episode {episode_index}: "
                f"tensor_exact={tensor_exact}, pixel_exact={pixel_exact}"
            )
        recheck_results.append(
            {
                "split": row["split"],
                "episode_index": episode_index,
                "decoded_pixels_exact": pixel_exact,
                "all_tensors_bit_exact": tensor_exact,
            }
        )

    manifest_text = csv_text(shard_rows)
    inventory_rows = [
        {
            "shard_file": row["shard_file"],
            "shard_bytes": row["shard_bytes"],
            "shard_sha256": row["shard_sha256"],
        }
        for row in shard_rows
    ]
    total_cache_bytes = sum(int(row["shard_bytes"]) for row in shard_rows)
    split_summary = {
        split: {
            "episodes": sum(row["split"] == split for row in shard_rows),
            "frames": sum(int(row["length"]) for row in shard_rows if row["split"] == split),
            "cache_bytes": sum(int(row["shard_bytes"]) for row in shard_rows if row["split"] == split),
        }
        for split in CACHE_SPLITS
    }
    model_files = [
        args.model_root / "config.json",
        args.model_root / "preprocessor_config.json",
        args.model_root / "model.safetensors",
    ]
    report = {
        "schema_version": 1,
        "stage": "WM Stage 13 resumable full feature cache",
        "status": "passed",
        "dataset": {
            "repo_id": "lerobot/libero",
            "revision": DATASET_REVISION,
            "source_manifest": "results/wm/stage11_data_audit/expert_episodes.csv",
            "source_manifest_sha256": sha256_file(args.manifest),
            "cached_splits": list(CACHE_SPLITS),
            "test_split_episodes": len(test_rows),
            "test_split_cached_episodes": 0,
        },
        "encoder": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "tokens_per_camera": TOKENS_PER_CAMERA,
            "latent_dimension": LATENT_DIMENSION,
            "output_dtype": "float16",
            "model_file_sha256": {path.name: sha256_file(path) for path in model_files},
        },
        "cache": {
            "format": "one atomic Safetensors shard per episode",
            "episodes": len(shard_rows),
            "frames": sum(int(row["length"]) for row in shard_rows),
            "camera_images": 2 * sum(int(row["length"]) for row in shard_rows),
            "shards": len(shard_rows),
            "total_bytes": total_cache_bytes,
            "inventory_sha256": sha256_bytes(canonical_json(inventory_rows).encode()),
            "split_summary": split_summary,
            "tensor_schema": {
                "visual_tokens": "[frames, 2, 16, 384] float16",
                "observation_state": "[frames, 8] float32",
                "action": "[frames, 7] float32",
                "dataset_index": "[frames] int64",
                "frame_index": "[frames] int64",
                "timestamp": "[frames] float32",
                "is_terminal": "[frames] bool",
            },
        },
        "validation": {
            "all_shards_valid": True,
            "episode_boundaries_preserved": True,
            "frame_indices_contiguous": True,
            "all_features_states_actions_finite": True,
            "feature_statistics": stats.finalize(),
            "deterministic_recheck_seed": args.recheck_seed,
            "deterministic_recheck": recheck_results,
        },
        "artifacts": {
            "shard_manifest": "shards.csv",
            "shard_manifest_sha256": sha256_bytes(manifest_text.encode()),
            "feature_cache_root": "outputs/wm/stage13_full_feature_cache/shards",
        },
        "stage14_readiness": {
            "latent_dynamics_training_ready": True,
            "test_split_remains_held_out": True,
            "world_model_training_started": False,
            "recommended_first_baseline": (
                "Train a deterministic action-conditioned next-latent predictor on train shards; "
                "select checkpoints only with validation shards."
            ),
        },
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
            "numpy": np.__version__,
        },
    }
    write_deterministic(
        args.public_results_dir / "shards.csv",
        manifest_text,
        overwrite=args.overwrite_public_results,
    )
    write_deterministic(
        args.public_results_dir / "report.json",
        canonical_json(report),
        overwrite=args.overwrite_public_results,
    )
    runtime = {
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
        "batch_size": args.batch_size,
        "new_shards": new_shards,
        "resumed_shards": resumed_shards,
        "decoding_seconds_for_new_shards": decoding_seconds,
        "encoding_seconds_for_new_shards": encoding_seconds,
        "total_seconds": time.perf_counter() - run_start,
        "peak_gpu_memory_bytes": (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0),
    }
    runtime_text = canonical_json(runtime)
    if new_shards > 0:
        extraction_runtime_path = args.output_dir / "runtime_extraction.json"
        if not extraction_runtime_path.exists():
            extraction_runtime_path.write_text(runtime_text, encoding="utf-8")
    (args.output_dir / "runtime.json").write_text(runtime_text, encoding="utf-8")
    print(
        "Stage 13 passed: "
        f"episodes={len(shard_rows)}, frames={report['cache']['frames']}, "
        f"new={new_shards}, resumed={resumed_shards}, "
        f"cache_bytes={total_cache_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
