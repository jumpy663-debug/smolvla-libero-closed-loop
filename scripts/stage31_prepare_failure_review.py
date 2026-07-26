#!/usr/bin/env python

"""Create deterministic contact sheets and apply reviewed failure labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from statistical_evaluation_common import atomic_write_text, sha256_file

DEFAULT_ANNOTATIONS = Path("results/statistical_evaluation/failure_annotations.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/statistical_evaluation_v1/failure_review")
DEFAULT_LABELS = Path("annotations/failure_labels_v1.json")
DEFAULT_ASSET_ROOT = Path("/home/jump/projects/lerobot/outputs/reproduction/smolvla_libero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--frames-per-sheet", type=int, default=12)
    parser.add_argument("--review-page-size", type=int, default=16)
    parser.add_argument("--apply-labels", action="store_true")
    return parser.parse_args()


def csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def ffprobe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise AssertionError(f"Expected one video stream in {path}")
    stream = streams[0]
    stream["width"] = int(stream["width"])
    stream["height"] = int(stream["height"])
    stream["nb_read_frames"] = int(stream["nb_read_frames"])
    return stream


def sample_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0 or sample_count <= 0:
        raise ValueError("Frame and sample counts must be positive")
    if sample_count == 1:
        return [0]
    return [round(index * (frame_count - 1) / (sample_count - 1)) for index in range(sample_count)]


def safe_drawtext(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def create_contact_sheet(
    *,
    video_path: Path,
    output_path: Path,
    label: str,
    sample_count: int,
) -> tuple[list[int], dict[str, Any]]:
    stream = ffprobe_video(video_path)
    indices = sample_indices(stream["nb_read_frames"], sample_count)
    select_expression = "+".join(f"eq(n\\,{index})" for index in indices)
    columns = 4
    rows = math.ceil(sample_count / columns)
    tile_width = 240
    tile_height = 180
    filter_graph = (
        f"select='{select_expression}',"
        f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,"
        f"pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"tile={columns}x{rows}:padding=2:margin=2:color=white,"
        f"drawtext=text='{safe_drawtext(label)}':x=8:y=8:"
        "fontcolor=white:fontsize=22:box=1:boxcolor=black@0.75"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}.png")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            filter_graph,
            "-frames:v",
            "1",
            str(temporary),
        ],
        check=True,
    )
    os.replace(temporary, output_path)
    return indices, stream


def create_review_page(contact_sheets: list[Path], output_path: Path) -> None:
    if not contact_sheets:
        raise ValueError("A review page requires at least one contact sheet")
    columns = 2 if len(contact_sheets) <= 4 else 4
    rows = math.ceil(len(contact_sheets) / columns)
    sheet_width = 960 if columns == 2 else 480
    sheet_height = 540 if columns == 2 else 270
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in contact_sheets:
        command.extend(["-i", str(path)])
    filters = []
    for index in range(len(contact_sheets)):
        filters.append(f"[{index}:v]scale={sheet_width}:{sheet_height}[s{index}]")
    layout = "|".join(
        f"{(index % columns) * sheet_width}_{(index // columns) * sheet_height}" for index in range(len(contact_sheets))
    )
    inputs = "".join(f"[s{index}]" for index in range(len(contact_sheets)))
    filters.append(
        f"{inputs}xstack=inputs={len(contact_sheets)}:layout={layout}:fill=white,"
        f"pad=1920:{rows * sheet_height}:0:0:white[out]"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}.png")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-frames:v",
            "1",
            str(temporary),
        ]
    )
    subprocess.run(command, check=True)
    os.replace(temporary, output_path)
    expected_height = rows * sheet_height
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    if int(stream["width"]) != 1920 or int(stream["height"]) != expected_height:
        raise AssertionError(f"Unexpected review page dimensions: {stream}")


def load_taxonomy(repository_root: Path) -> set[str]:
    path = repository_root / "protocols/failure_taxonomy_v1.json"
    taxonomy = json.loads(path.read_text(encoding="utf-8"))
    return {item["code"] for item in taxonomy["categories"]}


def resolve_public_artifact_path(path_value: str, *, repository_root: Path, asset_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "external_asset_root":
        return asset_root / Path(*path.parts[1:])
    return repository_root / path


def apply_labels(
    rows: list[dict[str, str]],
    *,
    labels_path: Path,
    allowed_codes: set[str],
) -> None:
    labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    indexed_codes = labels_payload.get("codes_by_annotation_index")
    if indexed_codes is not None:
        episode_id_sequence = "".join(f"{row['episode_id']}\n" for row in rows)
        actual_sequence_sha256 = hashlib.sha256(episode_id_sequence.encode()).hexdigest()
        expected_sequence_sha256 = labels_payload["episode_id_sequence_sha256"]
        if actual_sequence_sha256 != expected_sequence_sha256:
            raise AssertionError(
                "Failure-label episode sequence drifted: "
                f"expected={expected_sequence_sha256} actual={actual_sequence_sha256}"
            )
        if len(indexed_codes) != len(rows):
            raise AssertionError(f"Failure-label count mismatch: labels={len(indexed_codes)} rows={len(rows)}")
        notes = labels_payload.get("notes_by_annotation_index", {})
        label_source = labels_payload["label_source"]
        for index, (row, code) in enumerate(zip(rows, indexed_codes, strict=True)):
            if code not in allowed_codes:
                raise AssertionError(f"Unknown failure code {code} at annotation index {index}")
            note = str(notes.get(str(index), ""))
            if code == "other" and not note.strip():
                raise AssertionError(f"The other category requires notes at annotation index {index}")
            row["primary_failure_code"] = code
            row["label_source"] = label_source
            row["review_status"] = "confirmed"
            row["reviewer"] = labels_payload["reviewer"]
            row["notes_zh"] = note
        return

    labels = labels_payload["labels"]
    if len(labels) != len({item["episode_id"] for item in labels}):
        raise AssertionError("Failure-label episode IDs are not unique")
    by_episode = {item["episode_id"]: item for item in labels}
    row_ids = {row["episode_id"] for row in rows}
    if set(by_episode) != row_ids:
        missing = sorted(row_ids - set(by_episode))
        extra = sorted(set(by_episode) - row_ids)
        raise AssertionError(f"Failure labels do not cover the annotation CSV: missing={missing[:3]} extra={extra[:3]}")
    for row in rows:
        label = by_episode[row["episode_id"]]
        code = label["primary_failure_code"]
        if code not in allowed_codes:
            raise AssertionError(f"Unknown failure code {code} for {row['episode_id']}")
        if code == "other" and not str(label.get("notes_zh", "")).strip():
            raise AssertionError(f"The other category requires notes: {row['episode_id']}")
        row["primary_failure_code"] = code
        row["label_source"] = label["label_source"]
        row["review_status"] = "confirmed"
        row["reviewer"] = labels_payload["reviewer"]
        row["notes_zh"] = str(label.get("notes_zh", ""))


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    with args.annotations.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise AssertionError("Failure annotation CSV is empty")

    contact_paths = []
    for index, row in enumerate(rows):
        video_path = resolve_public_artifact_path(
            row["video_path"],
            repository_root=repository_root,
            asset_root=args.asset_root.resolve(),
        )
        if sha256_file(video_path) != row["video_sha256"]:
            raise AssertionError(f"Video hash mismatch for {row['episode_id']}")
        contact_path = args.output_dir / "contact_sheets" / f"{index:03d}_{row['episode_id']}.png"
        label = (
            f"{index:03d} {row['policy']} H{row['action_horizon']} "
            f"T{int(row['task_id']):02d} I{int(row['init_state_index']):02d}"
        )
        indices, stream = create_contact_sheet(
            video_path=video_path,
            output_path=contact_path,
            label=label,
            sample_count=args.frames_per_sheet,
        )
        contact_paths.append(contact_path)
        row["contact_sheet_path"] = str(contact_path)
        row["contact_sheet_sha256"] = sha256_file(contact_path)
        automatic = json.loads(row["automatic_progress_fields_json"])
        automatic["contact_sheet_source_frame_count"] = stream["nb_read_frames"]
        automatic["contact_sheet_sample_indices"] = indices
        row["automatic_progress_fields_json"] = json.dumps(automatic, sort_keys=True, ensure_ascii=False)

    review_manifest = []
    for page_index, start in enumerate(range(0, len(contact_paths), args.review_page_size)):
        page_contacts = contact_paths[start : start + args.review_page_size]
        page_path = args.output_dir / "review_pages" / f"review_page_{page_index:02d}.png"
        create_review_page(page_contacts, page_path)
        for slot, contact_path in enumerate(page_contacts):
            annotation_index = start + slot
            review_manifest.append(
                {
                    "review_page": str(page_path),
                    "review_page_sha256": sha256_file(page_path),
                    "slot": slot,
                    "annotation_index": annotation_index,
                    "episode_id": rows[annotation_index]["episode_id"],
                    "contact_sheet": str(contact_path),
                    "contact_sheet_sha256": rows[annotation_index]["contact_sheet_sha256"],
                }
            )

    if args.apply_labels:
        apply_labels(rows, labels_path=args.labels, allowed_codes=load_taxonomy(repository_root))
    atomic_write_text(args.annotations, csv_text(rows))
    atomic_write_text(args.output_dir / "review_manifest.csv", csv_text(review_manifest))
    print(
        f"Stage 31 passed: failures={len(rows)}, review_pages={math.ceil(len(rows) / args.review_page_size)}, "
        f"labels_applied={args.apply_labels}",
        flush=True,
    )


if __name__ == "__main__":
    main()
