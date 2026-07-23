#!/usr/bin/env python

"""Extract a deterministic DINOv2 representation pilot for LIBERO Spatial.

The pilot covers one episode per task in both train and validation splits. It
stores frozen dual-camera visual tokens together with aligned robot states and
actions, but it does not train a world model or extract the full dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
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

DATASET_REVISION = "2e98211ba27db9322efbaa5ed108b34bb03ca163"
MODEL_ID = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
PILOT_SPLITS = ("train", "validation")
CAMERA_COUNT = 2
TOKENS_PER_CAMERA = 16
TOKEN_GRID_SIZE = 4
LATENT_DIMENSION = 384
FPS = 10.0
VIDEO_TOLERANCE_SECONDS = 0.051


def default_hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def default_dataset_root() -> Path:
    lerobot_home = Path(os.environ.get("HF_LEROBOT_HOME", default_hf_home() / "lerobot"))
    return lerobot_home / "hub" / "datasets--lerobot--libero" / "snapshots" / DATASET_REVISION


def default_model_root() -> Path:
    return default_hf_home() / "hub" / "models--facebook--dinov2-small" / "snapshots" / MODEL_REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/wm/stage11_data_audit/expert_episodes.csv"),
    )
    parser.add_argument("--model-root", type=Path, default=default_model_root())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wm/stage12_representation_pilot"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage12_representation_pilot"),
    )
    parser.add_argument("--selection-seed", type=int, default=1200)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--frames-per-episode", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing non-identical pilot outputs.",
    )
    return parser.parse_args()


def canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_score(seed: int, split: str, task_id: int, episode_id: int) -> str:
    return sha256_bytes(f"{seed}:{split}:{task_id}:{episode_id}".encode())


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


def even_frame_indices(length: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("Frame count must be positive")
    if length < count:
        raise ValueError(f"Episode length {length} is smaller than requested frame count {count}")
    if count == 1:
        return [length // 2]
    indices = [round(index * (length - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise AssertionError(f"Even frame selection produced duplicates: {indices}")
    return indices


def select_pilot_episodes(
    manifest_rows: list[dict[str, str]],
    *,
    splits: tuple[str, ...],
    episodes_per_task: int,
    seed: int,
) -> list[dict[str, str]]:
    if episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive")
    selected: list[dict[str, str]] = []
    task_ids = sorted({int(row["benchmark_task_id"]) for row in manifest_rows})
    for split in splits:
        for task_id in task_ids:
            candidates = [
                row for row in manifest_rows if row["split"] == split and int(row["benchmark_task_id"]) == task_id
            ]
            ranked = sorted(
                candidates,
                key=lambda row: stable_score(seed, split, task_id, int(row["episode_index"])),
            )
            if len(ranked) < episodes_per_task:
                raise ValueError(
                    f"Split {split} task {task_id} has {len(ranked)} episodes; {episodes_per_task} requested"
                )
            selected.extend(ranked[:episodes_per_task])
    return selected


def pool_patch_tokens(last_hidden_state: torch.Tensor, grid_size: int = TOKEN_GRID_SIZE) -> torch.Tensor:
    if last_hidden_state.ndim != 3:
        raise ValueError(f"Expected [batch, tokens, channels], got {tuple(last_hidden_state.shape)}")
    patch_tokens = last_hidden_state[:, 1:, :]
    patch_grid = math.isqrt(patch_tokens.shape[1])
    if patch_grid * patch_grid != patch_tokens.shape[1]:
        raise ValueError(f"Patch token count is not square: {patch_tokens.shape[1]}")
    if patch_grid % grid_size != 0:
        raise ValueError(f"Patch grid {patch_grid} cannot be evenly pooled to {grid_size}")
    batch_size, _, channels = patch_tokens.shape
    spatial = patch_tokens.reshape(batch_size, patch_grid, patch_grid, channels)
    block_size = patch_grid // grid_size
    pooled = spatial.reshape(
        batch_size,
        grid_size,
        block_size,
        grid_size,
        block_size,
        channels,
    ).mean(dim=(2, 4))
    return pooled.reshape(batch_size, grid_size * grid_size, channels)


def load_aligned_episode_rows(
    dataset_root: Path,
    manifest_row: dict[str, str],
    frame_indices: list[int],
) -> dict[int, dict[str, Any]]:
    episode_id = int(manifest_row["episode_index"])
    table = pq.read_table(
        dataset_root / manifest_row["data_file"],
        columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"],
        filters=[("episode_index", "=", episode_id)],
    )
    frame_rows = table.to_pylist()
    by_frame = {int(row["frame_index"]): row for row in frame_rows}
    if len(by_frame) != int(manifest_row["length"]):
        raise AssertionError(
            f"Episode {episode_id} parquet length {len(by_frame)} != manifest {manifest_row['length']}"
        )
    missing = set(frame_indices) - set(by_frame)
    if missing:
        raise AssertionError(f"Episode {episode_id} is missing selected frames: {sorted(missing)}")
    return {frame_index: by_frame[frame_index] for frame_index in frame_indices}


def tensor_sha256(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def decode_pilot_samples(
    selected_episodes: list[dict[str, str]],
    dataset_root: Path,
    frames_per_episode: int,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_rows: list[dict[str, Any]] = []
    image_batches: list[torch.Tensor] = []
    states: list[list[float]] = []
    actions: list[list[float]] = []

    for manifest_row in selected_episodes:
        episode_id = int(manifest_row["episode_index"])
        frame_indices = even_frame_indices(int(manifest_row["length"]), frames_per_episode)
        aligned_rows = load_aligned_episode_rows(dataset_root, manifest_row, frame_indices)
        timestamps = [float(aligned_rows[index]["timestamp"]) for index in frame_indices]
        camera_frames: list[torch.Tensor] = []
        camera_query_timestamps: list[list[float]] = []
        for camera_index in (1, 2):
            query_timestamps = [
                float(manifest_row[f"camera{camera_index}_from_seconds"]) + timestamp for timestamp in timestamps
            ]
            frames = decode_video_frames(
                dataset_root / manifest_row[f"camera{camera_index}_video_file"],
                query_timestamps,
                tolerance_s=VIDEO_TOLERANCE_SECONDS,
                backend="pyav",
                return_uint8=True,
            )
            expected_shape = (frames_per_episode, 3, 256, 256)
            if tuple(frames.shape) != expected_shape or frames.dtype != torch.uint8:
                raise AssertionError(
                    f"Episode {episode_id} camera {camera_index} decoded "
                    f"{tuple(frames.shape)} {frames.dtype}, expected {expected_shape} uint8"
                )
            camera_frames.append(frames)
            camera_query_timestamps.append(query_timestamps)

        episode_images = torch.stack(camera_frames, dim=1)
        image_batches.append(episode_images)
        for local_index, frame_index in enumerate(frame_indices):
            aligned = aligned_rows[frame_index]
            state = [float(value) for value in aligned["observation.state"]]
            action = [float(value) for value in aligned["action"]]
            if len(state) != 8 or len(action) != 7:
                raise AssertionError(
                    f"Episode {episode_id} frame {frame_index} has state/action shapes {len(state)}/{len(action)}"
                )
            states.append(state)
            actions.append(action)
            sample_rows.append(
                {
                    "sample_index": len(sample_rows),
                    "split": manifest_row["split"],
                    "benchmark_task_id": int(manifest_row["benchmark_task_id"]),
                    "dataset_task_index": int(manifest_row["dataset_task_index"]),
                    "episode_index": episode_id,
                    "frame_index": frame_index,
                    "episode_timestamp_seconds": timestamps[local_index],
                    "camera1_video_file": manifest_row["camera1_video_file"],
                    "camera1_query_timestamp_seconds": camera_query_timestamps[0][local_index],
                    "camera1_pixel_sha256": tensor_sha256(episode_images[local_index, 0]),
                    "camera2_video_file": manifest_row["camera2_video_file"],
                    "camera2_query_timestamp_seconds": camera_query_timestamps[1][local_index],
                    "camera2_pixel_sha256": tensor_sha256(episode_images[local_index, 1]),
                }
            )

    images = torch.cat(image_batches, dim=0)
    state_tensor = torch.tensor(states, dtype=torch.float32)
    action_tensor = torch.tensor(actions, dtype=torch.float32)
    if not torch.isfinite(state_tensor).all() or not torch.isfinite(action_tensor).all():
        raise AssertionError("Pilot state/action tensors contain non-finite values")
    return sample_rows, images, state_tensor, action_tensor


def prepare_pixels(processor: Any, images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected image batch [batch, 3, height, width], got {tuple(images.shape)}")
    image_list = [image.permute(1, 2, 0).contiguous().numpy() for image in images]
    pixel_values = processor(images=image_list, return_tensors="pt")["pixel_values"]
    if pixel_values.dtype != torch.float32 or pixel_values.ndim != 4:
        raise AssertionError(f"Unexpected processor output: {pixel_values.dtype} {pixel_values.shape}")
    return pixel_values


def encode_visual_tokens(
    images: torch.Tensor,
    processor: Any,
    model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    flat_images = images.reshape(-1, *images.shape[2:])
    output_batches: list[torch.Tensor] = []
    repeat_exact = False
    repeat_max_abs_difference = float("nan")
    encoding_start = time.perf_counter()

    with torch.inference_mode():
        for start in range(0, len(flat_images), batch_size):
            batch = flat_images[start : start + batch_size]
            pixels = prepare_pixels(processor, batch).to(device)
            outputs = model(pixel_values=pixels).last_hidden_state
            pooled = pool_patch_tokens(outputs)
            if start == 0:
                repeated = pool_patch_tokens(model(pixel_values=pixels).last_hidden_state)
                repeat_exact = torch.equal(pooled, repeated)
                repeat_max_abs_difference = float((pooled - repeated).abs().max().item())
                if not repeat_exact:
                    raise AssertionError(f"Repeated frozen encoder pass was not bit-exact: {repeat_max_abs_difference}")
            output_batches.append(pooled.cpu())

    flat_tokens = torch.cat(output_batches, dim=0)
    if flat_tokens.shape != (len(flat_images), TOKENS_PER_CAMERA, LATENT_DIMENSION):
        raise AssertionError(f"Unexpected pooled token shape: {tuple(flat_tokens.shape)}")
    tokens = flat_tokens.reshape(
        len(images),
        CAMERA_COUNT,
        TOKENS_PER_CAMERA,
        LATENT_DIMENSION,
    ).to(torch.float16)
    if not torch.isfinite(tokens).all():
        raise AssertionError("Visual tokens contain non-finite values")
    return tokens, {
        "encoding_seconds": time.perf_counter() - encoding_start,
        "repeat_first_batch_bit_exact": repeat_exact,
        "repeat_first_batch_max_abs_difference": repeat_max_abs_difference,
    }


def compare_or_save_cache(
    path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        existing = load_file(path)
        if set(existing) != set(tensors):
            raise FileExistsError(f"Existing cache keys differ: {path}")
        for key, tensor in tensors.items():
            if not torch.equal(existing[key], tensor):
                raise FileExistsError(f"Existing cache tensor differs for {key}: {path}")
        with safe_open(path, framework="pt") as handle:
            if handle.metadata() != metadata:
                raise FileExistsError(f"Existing cache metadata differs: {path}")
        return
    save_file(tensors, path, metadata=metadata)


def rounded(value: float) -> float:
    return round(value, 8)


def feature_statistics(tokens: torch.Tensor) -> dict[str, Any]:
    values = tokens.float().numpy().astype(np.float64)
    sample_embeddings = values.mean(axis=(1, 2))
    unique_samples = len(np.unique(sample_embeddings, axis=0))
    camera_embeddings = values.mean(axis=2)
    dot = (camera_embeddings[:, 0] * camera_embeddings[:, 1]).sum(axis=1)
    norms = np.linalg.norm(camera_embeddings[:, 0], axis=1) * np.linalg.norm(camera_embeddings[:, 1], axis=1)
    camera_cosine = dot / np.maximum(norms, 1e-12)
    return {
        "finite": bool(np.isfinite(values).all()),
        "minimum": rounded(float(values.min())),
        "maximum": rounded(float(values.max())),
        "mean": rounded(float(values.mean())),
        "standard_deviation": rounded(float(values.std())),
        "mean_channel_standard_deviation": rounded(float(sample_embeddings.std(axis=0).mean())),
        "unique_sample_embeddings": unique_samples,
        "camera_embedding_cosine_mean": rounded(float(camera_cosine.mean())),
        "camera_embedding_cosine_minimum": rounded(float(camera_cosine.min())),
        "camera_embedding_cosine_maximum": rounded(float(camera_cosine.max())),
    }


def main() -> None:
    args = parse_args()
    if args.selection_seed < 0:
        raise ValueError("--selection-seed must be non-negative")
    if args.dataset_root.name != DATASET_REVISION:
        raise ValueError(f"Dataset snapshot must be revision {DATASET_REVISION}")
    if args.model_root.name != MODEL_REVISION:
        raise ValueError(f"Model snapshot must be revision {MODEL_REVISION}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    required_model_files = [
        args.model_root / "config.json",
        args.model_root / "preprocessor_config.json",
        args.model_root / "model.safetensors",
    ]
    missing_model_files = [path for path in required_model_files if not path.is_file()]
    if missing_model_files:
        raise FileNotFoundError(f"Frozen encoder snapshot is incomplete: {missing_model_files}")

    manifest_rows = read_csv(args.manifest)
    selected_episodes = select_pilot_episodes(
        manifest_rows,
        splits=PILOT_SPLITS,
        episodes_per_task=args.episodes_per_task,
        seed=args.selection_seed,
    )
    decode_start = time.perf_counter()
    sample_rows, images, states, actions = decode_pilot_samples(
        selected_episodes,
        args.dataset_root,
        args.frames_per_episode,
    )
    decoding_seconds = time.perf_counter() - decode_start

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    processor = AutoImageProcessor.from_pretrained(args.model_root, local_files_only=True)
    model = Dinov2Model.from_pretrained(args.model_root, local_files_only=True).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        model.requires_grad_(False)
    model.to(device)
    visual_tokens, encoder_checks = encode_visual_tokens(
        images,
        processor,
        model,
        device=device,
        batch_size=args.batch_size,
    )
    peak_gpu_memory_bytes = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0

    split_codes = {split: index for index, split in enumerate(PILOT_SPLITS)}
    cache_tensors = {
        "visual_tokens": visual_tokens.contiguous(),
        "observation_state": states.contiguous(),
        "action": actions.contiguous(),
        "sample_index": torch.arange(len(sample_rows), dtype=torch.int64),
        "episode_index": torch.tensor(
            [int(row["episode_index"]) for row in sample_rows],
            dtype=torch.int64,
        ),
        "frame_index": torch.tensor(
            [int(row["frame_index"]) for row in sample_rows],
            dtype=torch.int64,
        ),
        "benchmark_task_id": torch.tensor(
            [int(row["benchmark_task_id"]) for row in sample_rows],
            dtype=torch.int64,
        ),
        "split_code": torch.tensor(
            [split_codes[str(row["split"])] for row in sample_rows],
            dtype=torch.int8,
        ),
    }
    cache_metadata = {
        "schema_version": "1",
        "dataset_repo_id": "lerobot/libero",
        "dataset_revision": DATASET_REVISION,
        "encoder_model_id": MODEL_ID,
        "encoder_revision": MODEL_REVISION,
        "visual_token_layout": "sample,camera,spatial_token,channel",
        "split_codes": canonical_json(split_codes).strip(),
    }
    cache_path = args.output_dir / "pilot_features.safetensors"
    compare_or_save_cache(cache_path, cache_tensors, cache_metadata, overwrite=args.overwrite)

    reloaded = load_file(cache_path)
    cache_reload_exact = set(reloaded) == set(cache_tensors) and all(
        torch.equal(reloaded[key], value) for key, value in cache_tensors.items()
    )
    if not cache_reload_exact:
        raise AssertionError("Reloaded feature cache does not match in-memory tensors")

    samples_text = csv_text(sample_rows)
    sample_counts = {split: sum(row["split"] == split for row in sample_rows) for split in PILOT_SPLITS}
    task_coverage = {
        split: sorted({int(row["benchmark_task_id"]) for row in sample_rows if row["split"] == split})
        for split in PILOT_SPLITS
    }
    stats = feature_statistics(visual_tokens)
    report = {
        "schema_version": 1,
        "stage": "WM Stage 12 frozen representation pilot",
        "status": "passed",
        "dataset": {
            "repo_id": "lerobot/libero",
            "revision": DATASET_REVISION,
            "manifest": "results/wm/stage11_data_audit/expert_episodes.csv",
            "manifest_sha256": sha256_file(args.manifest),
        },
        "selection": {
            "seed": args.selection_seed,
            "splits": list(PILOT_SPLITS),
            "episodes_per_task": args.episodes_per_task,
            "frames_per_episode": args.frames_per_episode,
            "episodes": len(selected_episodes),
            "samples": len(sample_rows),
            "camera_images": int(images.shape[0] * images.shape[1]),
            "samples_per_split": sample_counts,
            "task_coverage": task_coverage,
        },
        "encoder": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "input_shape": list(images.shape),
            "processor_output_resolution": [224, 224],
            "patch_size": int(model.config.patch_size),
            "native_patch_grid": [16, 16],
            "pooling": "non-overlapping 4x4 mean pooling over 16x16 patch tokens",
            "output_shape": list(visual_tokens.shape),
            "output_dtype": str(visual_tokens.dtype).removeprefix("torch."),
            "model_file_sha256": {path.name: sha256_file(path) for path in required_model_files},
        },
        "aligned_modalities": {
            "visual_tokens": list(visual_tokens.shape),
            "observation_state": list(states.shape),
            "action": list(actions.shape),
        },
        "validation": {
            "video_decode_backend": "pyav",
            "decoded_image_shape": list(images.shape),
            "decoded_image_dtype": str(images.dtype).removeprefix("torch."),
            "feature_statistics": stats,
            "repeat_first_batch_bit_exact": encoder_checks["repeat_first_batch_bit_exact"],
            "repeat_first_batch_max_abs_difference": encoder_checks["repeat_first_batch_max_abs_difference"],
            "cache_reload_bit_exact": cache_reload_exact,
            "state_finite": bool(torch.isfinite(states).all()),
            "action_finite": bool(torch.isfinite(actions).all()),
        },
        "artifacts": {
            "feature_cache": "outputs/wm/stage12_representation_pilot/pilot_features.safetensors",
            "feature_cache_bytes": cache_path.stat().st_size,
            "feature_cache_sha256": sha256_file(cache_path),
            "samples": "samples.csv",
            "samples_sha256": sha256_bytes(samples_text.encode()),
        },
        "stage13_readiness": {
            "full_feature_extraction_ready": True,
            "world_model_training_started": False,
            "decision": (
                "The frozen dual-camera representation is finite, non-collapsed, deterministic "
                "within the tested hardware path, and aligned with state/action rows."
            ),
        },
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
            "numpy": np.__version__,
        },
    }
    if stats["unique_sample_embeddings"] != len(sample_rows):
        raise AssertionError("At least two pilot samples have identical mean embeddings")
    if stats["standard_deviation"] <= 0 or stats["mean_channel_standard_deviation"] <= 0:
        raise AssertionError(f"Frozen representation appears collapsed: {stats}")

    write_deterministic(
        args.public_results_dir / "samples.csv",
        samples_text,
        overwrite=args.overwrite,
    )
    write_deterministic(
        args.public_results_dir / "report.json",
        canonical_json(report),
        overwrite=args.overwrite,
    )
    runtime = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "batch_size": args.batch_size,
        "decoding_seconds": decoding_seconds,
        "encoding_seconds": encoder_checks["encoding_seconds"],
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runtime.json").write_text(canonical_json(runtime), encoding="utf-8")
    print(
        "Stage 12 passed: "
        f"samples={len(sample_rows)}, images={images.shape[0] * images.shape[1]}, "
        f"tokens={tuple(visual_tokens.shape)}, cache={cache_path}"
    )


if __name__ == "__main__":
    main()
