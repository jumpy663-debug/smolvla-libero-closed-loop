#!/usr/bin/env python

"""Download, pin, and statically audit the official SmolVLA LIBERO checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import huggingface_hub
import lerobot
import safetensors
from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="lerobot/smolvla_libero")
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub revision to resolve. The manifest always records the immutable commit SHA.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage2_checkpoint"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def collect_step_types(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"type", "name", "registry_name"} and isinstance(child, str) and "processor" in child.lower():
                found.append(child)
            found.extend(collect_step_types(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(collect_step_types(child))
    return list(dict.fromkeys(found))


def inspect_safetensors(path: Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        sample_shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys[:10]}
        metadata = handle.metadata()
    return {
        "tensor_count": len(keys),
        "first_tensor_names": keys[:10],
        "first_tensor_shapes": sample_shapes,
        "metadata": metadata,
    }


def safetensors_shapes(path: Path) -> dict[str, list[int]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            key: list(handle.get_slice(key).get_shape())
            for key in handle.keys()  # noqa: SIM118 - safe_open exposes an explicit keys() API.
        }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoint"

    api = HfApi()
    model_info = api.model_info(args.repo_id, revision=args.revision, files_metadata=True)
    if not model_info.sha:
        raise RuntimeError(f"Hub did not return an immutable commit for {args.repo_id}@{args.revision}")
    resolved_revision = model_info.sha

    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            revision=resolved_revision,
            local_dir=checkpoint_dir,
            max_workers=8,
        )
    )
    if snapshot_path.resolve() != checkpoint_dir.resolve():
        raise AssertionError(f"Unexpected snapshot path: {snapshot_path}")

    required_files = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "train_config.json",
    ]
    missing = [name for name in required_files if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing required files: {missing}")

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(checkpoint_dir.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(checkpoint_dir).parts:
            continue
        relative = path.relative_to(checkpoint_dir).as_posix()
        files[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    remote_files: dict[str, dict[str, Any]] = {}
    for sibling in model_info.siblings or []:
        lfs = sibling.lfs
        remote_sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        remote_files[sibling.rfilename] = {
            "size_bytes": sibling.size,
            "blob_id": sibling.blob_id,
            "lfs_sha256": remote_sha256,
        }
        if remote_sha256 and sibling.rfilename in files:
            local_sha256 = files[sibling.rfilename]["sha256"]
            if local_sha256 != remote_sha256:
                raise AssertionError(
                    f"SHA-256 mismatch for {sibling.rfilename}: local={local_sha256}, hub={remote_sha256}"
                )

    config = load_json(checkpoint_dir / "config.json")
    train_config = load_json(checkpoint_dir / "train_config.json")
    preprocessor = load_json(checkpoint_dir / "policy_preprocessor.json")
    postprocessor = load_json(checkpoint_dir / "policy_postprocessor.json")
    preprocessor_stats_path = checkpoint_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    postprocessor_stats_path = checkpoint_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    preprocessor_stats_shapes = safetensors_shapes(preprocessor_stats_path)
    postprocessor_stats_shapes = safetensors_shapes(postprocessor_stats_path)

    backbone_repo_id = config.get("vlm_model_name")
    if not isinstance(backbone_repo_id, str) or not backbone_repo_id:
        raise AssertionError(f"Invalid VLM backbone repo id: {backbone_repo_id!r}")
    backbone_info = api.model_info(backbone_repo_id, revision="main", files_metadata=True)
    if not backbone_info.sha:
        raise RuntimeError(f"Hub did not return an immutable commit for {backbone_repo_id}@main")
    backbone_runtime_files = {
        sibling.rfilename: sibling.size
        for sibling in (backbone_info.siblings or [])
        if not sibling.rfilename.startswith("onnx/")
        and sibling.rfilename
        not in {
            ".gitattributes",
            "README.md",
        }
    }

    input_features = config.get("input_features", {})
    output_features = config.get("output_features", {})
    visual_features = {key: value for key, value in input_features.items() if value.get("type") == "VISUAL"}
    state_features = {key: value for key, value in input_features.items() if value.get("type") == "STATE"}
    action_feature = output_features.get("action")
    declared_state_shape = next(iter(state_features.values())).get("shape") if state_features else None
    statistics_state_shape = preprocessor_stats_shapes.get("observation.state.mean")

    if config.get("type") != "smolvla":
        raise AssertionError(f"Expected SmolVLA config, got type={config.get('type')!r}")
    if action_feature is None or action_feature.get("shape") != [7]:
        raise AssertionError(f"Expected a 7-D action feature, got {action_feature}")
    if len(visual_features) != 3:
        raise AssertionError(f"Expected three visual slots, got {sorted(visual_features)}")
    if not state_features:
        raise AssertionError("Checkpoint has no state input feature")
    if config.get("chunk_size") != 50 or config.get("n_action_steps") != 50:
        raise AssertionError(
            "Unexpected action chunk settings: "
            f"chunk_size={config.get('chunk_size')}, n_action_steps={config.get('n_action_steps')}"
        )

    weight_path = checkpoint_dir / "model.safetensors"
    if weight_path.stat().st_size < 100 * 1024 * 1024:
        raise AssertionError("model.safetensors is unexpectedly small")

    report = {
        "status": "passed",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "checkpoint_dir": str(checkpoint_dir),
        "reproduction_command": (
            f"uv run --no-sync python scripts/stage2_prepare_checkpoint.py --revision {resolved_revision}"
        ),
        "versions": {
            "python": platform.python_version(),
            "lerobot": lerobot.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "safetensors": safetensors.__version__,
        },
        "hub_metadata": {
            "model_id": model_info.id,
            "commit_sha": resolved_revision,
            "private": bool(model_info.private),
            "gated": model_info.gated,
            "pipeline_tag": model_info.pipeline_tag,
            "library_name": model_info.library_name,
            "tags": model_info.tags,
            "files": remote_files,
        },
        "external_backbone": {
            "repo_id": backbone_repo_id,
            "resolved_revision": backbone_info.sha,
            "runtime_files": backbone_runtime_files,
            "runtime_files_total_size_bytes": sum(size or 0 for size in backbone_runtime_files.values()),
            "downloaded_in_stage_2": False,
            "reason": (
                "The checkpoint files are complete, but SmolVLA construction also initializes the "
                "referenced VLM and tokenizer. Stage 3 will fetch only these runtime files, excluding "
                "the backbone repository's unrelated ONNX exports."
            ),
        },
        "checkpoint_summary": {
            "total_files": len(files),
            "total_size_bytes": sum(item["size_bytes"] for item in files.values()),
            "policy_type": config["type"],
            "input_features": input_features,
            "output_features": output_features,
            "visual_feature_names": sorted(visual_features),
            "state_features": state_features,
            "declared_state_shape": declared_state_shape,
            "statistics_state_shape": statistics_state_shape,
            "declared_and_statistics_state_shapes_match": (declared_state_shape == statistics_state_shape),
            "chunk_size": config.get("chunk_size"),
            "n_action_steps": config.get("n_action_steps"),
            "num_flow_steps": config.get("num_steps"),
            "empty_cameras_in_checkpoint": config.get("empty_cameras"),
            "vlm_model_name": config.get("vlm_model_name"),
            "load_vlm_weights": config.get("load_vlm_weights"),
            "use_amp": config.get("use_amp"),
            "normalization_mapping": config.get("normalization_mapping"),
            "training_dataset": train_config.get("dataset", {}).get("repo_id"),
            "preprocessor_step_types": collect_step_types(preprocessor),
            "postprocessor_step_types": collect_step_types(postprocessor),
            "preprocessor_rename_map": preprocessor.get("steps", [{}])[0].get("config", {}).get("rename_map"),
            "preprocessor_statistics_shapes": preprocessor_stats_shapes,
            "postprocessor_statistics_shapes": postprocessor_stats_shapes,
            "weights": inspect_safetensors(weight_path),
        },
        "compatibility_notes": [
            "The checkpoint declares a 6-D state feature, while its saved normalization statistics "
            "contain 8-D state tensors. The current LeRobot LIBERO processor also emits 8-D state; "
            "stage 3 must validate this path with a real observation before closed-loop evaluation.",
            "The checkpoint has three visual slots. Current LeRobot CI maps the two LIBERO cameras "
            "to camera1/camera2 and overrides policy.empty_cameras=1 for the third slot.",
            "The checkpoint references an external SmolVLM2 backbone. Its exact commit is recorded "
            "above; only runtime PyTorch/tokenizer files should be downloaded in stage 3.",
        ],
        "files": files,
    }

    report_path = args.output_dir / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "REVISION").write_text(resolved_revision + "\n", encoding="utf-8")
    (args.output_dir / "BACKBONE_REVISION").write_text(backbone_info.sha + "\n", encoding="utf-8")

    print(json.dumps(report["checkpoint_summary"], indent=2, ensure_ascii=False))
    print(f"Resolved revision: {resolved_revision}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Manifest: {report_path}")


if __name__ == "__main__":
    main()
