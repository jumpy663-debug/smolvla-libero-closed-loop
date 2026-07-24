#!/usr/bin/env python

"""Extend the score-blind scout and freeze a balanced three-task cohort.

Task 5 is retained as a policy-floor stress set. Task 8 is evaluated on the
entire pre-registered official initial-state block 15--26. If Task 4/7/8 each
contain at least three successes and three failures, this script freezes a
score-blind 12-episode calibration and 6-episode held-out test cohort.
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

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.envs import close_envs, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

COHORT_TASK_IDS = (4, 7, 8)
STRESS_TASK_ID = 5
EXTENSION_TASK_ID = 8
EXTENSION_START = 15
EXTENSION_END = 26
REQUIRED_PER_LABEL_PER_TASK = 3
CALIBRATION_PER_LABEL_PER_TASK = 2
TEST_PER_LABEL_PER_TASK = 1
SELECTION_SALT = "stage19-three-task-balanced-cohort-v1"


def load_stage18() -> ModuleType:
    path = Path(__file__).with_name("stage18_initial_state_scout.py")
    spec = importlib.util.spec_from_file_location("stage18_initial_state_scout_for_stage19", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE18 = load_stage18()
STAGE3 = STAGE18.STAGE3
STAGE5 = STAGE18.STAGE5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--stage17-report",
        type=Path,
        default=Path("results/wm/stage17_failure_signal/report.json"),
    )
    parser.add_argument(
        "--stage18-report",
        type=Path,
        default=Path("results/wm/stage18_initial_state_scout/report.json"),
    )
    parser.add_argument(
        "--stage18-episodes",
        type=Path,
        default=Path("results/wm/stage18_initial_state_scout/scout_episodes.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wm/stage19_balanced_cohort"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage19_balanced_cohort"),
    )
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--overwrite-public", action="store_true")
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
    if not rows:
        raise ValueError("Cannot serialize empty rows")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def read_stage18_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = dict(raw)
        for key in ("benchmark_task_id", "init_state_index", "episode_seed", "steps"):
            row[key] = int(row[key])
        row["success"] = row["success"] == "True"
        row["cumulative_reward"] = float(row["cumulative_reward"])
        for key in (
            "official_libero_initial_state",
            "complete_observations_saved",
            "test_demonstration_episode_used",
        ):
            row[key] = row[key] == "True"
        row["source_stage"] = 18
        rows.append(row)
    if len(rows) != 48:
        raise AssertionError(f"Expected 48 Stage 18 rows, got {len(rows)}")
    return rows


def public_episode_row(record: dict[str, Any], *, source_stage: int) -> dict[str, Any]:
    row = STAGE18.public_episode_row(record)
    row["source_stage"] = source_stage
    return row


def selection_key(row: dict[str, Any]) -> str:
    payload = (
        f"{SELECTION_SALT}|task={int(row['benchmark_task_id'])}|"
        f"success={int(bool(row['success']))}|init={int(row['init_state_index'])}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_balanced_cohort(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keys = [(int(row["benchmark_task_id"]), int(row["init_state_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("Candidate pool contains duplicate task/initial-state keys")
    selected: list[dict[str, Any]] = []
    cells: dict[str, Any] = {}
    blocking: list[dict[str, Any]] = []
    for task_id in COHORT_TASK_IDS:
        cells[str(task_id)] = {}
        for success in (False, True):
            label = "success" if success else "failure"
            candidates = sorted(
                (row for row in rows if int(row["benchmark_task_id"]) == task_id and bool(row["success"]) == success),
                key=selection_key,
            )
            enough = len(candidates) >= REQUIRED_PER_LABEL_PER_TASK
            cells[str(task_id)][label] = {
                "available": len(candidates),
                "required": REQUIRED_PER_LABEL_PER_TASK,
                "ready": enough,
            }
            if not enough:
                blocking.append(
                    {
                        "benchmark_task_id": task_id,
                        "label": label,
                        "available": len(candidates),
                        "required": REQUIRED_PER_LABEL_PER_TASK,
                        "deficit": REQUIRED_PER_LABEL_PER_TASK - len(candidates),
                    }
                )
                continue
            for rank, row in enumerate(candidates[:REQUIRED_PER_LABEL_PER_TASK]):
                split = "calibration" if rank < CALIBRATION_PER_LABEL_PER_TASK else "test"
                selected.append(
                    {
                        "rollout_id": row["rollout_id"],
                        "benchmark_task_id": task_id,
                        "task_name": row["task_name"],
                        "init_state_index": int(row["init_state_index"]),
                        "episode_seed": int(row["episode_seed"]),
                        "success": success,
                        "split": split,
                        "selection_rank_within_task_label": rank,
                        "selection_key": selection_key(row),
                        "source_stage": int(row["source_stage"]),
                        "scout_initial_observation_sha256": row["initial_observation_sha256"],
                        "scout_action_sha256": row["action_sha256"],
                        "stage17_score_inspected_for_selection": False,
                        "complete_observation_recollection_required": True,
                    }
                )
    ready = not blocking
    selected.sort(
        key=lambda row: (
            str(row["split"]),
            int(row["benchmark_task_id"]),
            bool(row["success"]),
            int(row["selection_rank_within_task_label"]),
        )
    )
    expected = len(COHORT_TASK_IDS) * REQUIRED_PER_LABEL_PER_TASK * 2
    if ready and len(selected) != expected:
        raise AssertionError(f"Expected {expected} selected episodes, got {len(selected)}")
    return selected if ready else [], {
        "ready": ready,
        "per_task": cells,
        "blocking_cells": blocking,
    }


def protocol_payload(
    *,
    policy_sha256: str,
    backbone_sha256: str,
    stage17_sha256: str,
    stage18_report_sha256: str,
    stage18_episodes_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite": "libero_spatial",
        "extension_task_id": EXTENSION_TASK_ID,
        "extension_initial_state_indices": list(range(EXTENSION_START, EXTENSION_END + 1)),
        "extension_must_finish_full_block": True,
        "cohort_task_ids": list(COHORT_TASK_IDS),
        "stress_task_id": STRESS_TASK_ID,
        "stress_task_role": "policy-floor external stress set; excluded from within-task AUROC",
        "episode_batch_size": 1,
        "episode_seed_formula": (f"{STAGE18.BASE_SEED} + task_id * 100 + init_state_index"),
        "n_action_steps": STAGE18.ACTION_HORIZON,
        "resolution": STAGE18.RESOLUTION,
        "required_per_task_label": REQUIRED_PER_LABEL_PER_TASK,
        "calibration_per_task_label": CALIBRATION_PER_LABEL_PER_TASK,
        "test_per_task_label": TEST_PER_LABEL_PER_TASK,
        "selection_salt": SELECTION_SALT,
        "selection_uses_stage17_scores": False,
        "policy_sha256": policy_sha256,
        "backbone_sha256": backbone_sha256,
        "stage17_report_sha256": stage17_sha256,
        "stage18_report_sha256": stage18_report_sha256,
        "stage18_episodes_sha256": stage18_episodes_sha256,
    }


def record_path(output_dir: Path, init_state_index: int) -> Path:
    return output_dir / "episodes" / f"task_08_init_{init_state_index:02d}.json"


def validate_record(
    record: dict[str, Any],
    *,
    init_state_index: int,
    protocol_sha256: str,
) -> None:
    expected = {
        "benchmark_task_id": EXTENSION_TASK_ID,
        "init_state_index": init_state_index,
        "episode_seed": STAGE18.episode_seed(EXTENSION_TASK_ID, init_state_index),
        "protocol_sha256": protocol_sha256,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"Resume record mismatch for {key}: {record.get(key)} != {value}")
    if int(record["steps"]) <= 0 or int(record["steps"]) > 280:
        raise AssertionError("Invalid episode length")
    for key in ("initial_observation_sha256", "action_sha256"):
        if len(str(record[key])) != 64:
            raise AssertionError(f"Invalid {key}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 19 requires a CUDA GPU")

    stage17 = json.loads(args.stage17_report.read_text(encoding="utf-8"))
    stage18_report = json.loads(args.stage18_report.read_text(encoding="utf-8"))
    if stage17.get("status") != "passed":
        raise AssertionError("Stage 17 must have passed")
    if stage17["primary_result"]["score"] != "h10_raw_mse":
        raise AssertionError("The frozen Stage 17 primary score changed")
    if stage18_report.get("status") != "insufficient_balance":
        raise AssertionError("Stage 19 expects the audited Stage 18 insufficient-balance result")
    expected_blocking = {(5, "success", 3), (8, "failure", 1)}
    actual_blocking = {
        (int(row["benchmark_task_id"]), str(row["label"]), int(row["deficit"]))
        for row in stage18_report["label_balance"]["blocking_cells"]
    }
    if actual_blocking != expected_blocking:
        raise AssertionError(f"Unexpected Stage 18 blocking cells: {actual_blocking}")

    stage18_rows = read_stage18_rows(args.stage18_episodes)
    stage2_manifest = STAGE3.verify_stage2_manifest(args.stage2_manifest)
    backbone_manifest = json.loads((args.backbone_dir.parent / "backbone_manifest.json").read_text(encoding="utf-8"))
    if stage2_manifest["resolved_revision"] != STAGE3.EXPECTED_POLICY_REVISION:
        raise AssertionError("Policy revision differs from the pinned revision")
    if backbone_manifest["resolved_revision"] != STAGE3.BACKBONE_REVISION:
        raise AssertionError("Backbone revision differs from the pinned revision")

    policy_sha256 = sha256_file(args.checkpoint_dir / "model.safetensors")
    backbone_sha256 = sha256_file(args.backbone_dir / "model.safetensors")
    protocol = protocol_payload(
        policy_sha256=policy_sha256,
        backbone_sha256=backbone_sha256,
        stage17_sha256=sha256_file(args.stage17_report),
        stage18_report_sha256=sha256_file(args.stage18_report),
        stage18_episodes_sha256=sha256_file(args.stage18_episodes),
    )
    protocol_text = canonical_json(protocol)
    protocol_sha256 = sha256_text(protocol_text)
    STAGE18.write_deterministic(args.output_dir / "protocol.json", protocol_text)

    config = PreTrainedConfig.from_pretrained(args.checkpoint_dir, local_files_only=True)
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
        pretrained_path=str(args.checkpoint_dir),
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": {}},
            "tokenizer_processor": {"tokenizer_name": str(args.backbone_dir.resolve())},
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
    )
    policy.eval()
    model_loaded = time.perf_counter()

    missing = [
        init_state_index
        for init_state_index in range(EXTENSION_START, EXTENSION_END + 1)
        if not record_path(args.output_dir, init_state_index).exists()
    ]
    if not missing:
        print("Stage 19 resume: all 12 Task 8 extension labels already exist", flush=True)
    else:
        env_cfg, env, envs = STAGE5.make_task_env(
            suite="libero_spatial",
            task_id=EXTENSION_TASK_ID,
            n_envs=1,
            resolution=STAGE18.RESOLUTION,
        )
        try:
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, config)
            for init_state_index in missing:
                episode_started = time.perf_counter()
                record = STAGE18.run_episode(
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task_id=EXTENSION_TASK_ID,
                    init_state_index=init_state_index,
                )
                record["protocol_sha256"] = protocol_sha256
                STAGE18.write_deterministic(
                    record_path(args.output_dir, init_state_index),
                    canonical_json(record),
                )
                print(
                    f"Stage 19 task=8 init={init_state_index:02d} "
                    f"success={record['success']} steps={record['steps']} "
                    f"seconds={time.perf_counter() - episode_started:.1f}",
                    flush=True,
                )
        finally:
            close_envs(envs)

    scout_finished = time.perf_counter()
    runtime = {
        "schema_version": 1,
        "model_load_seconds": model_loaded - started,
        "scout_or_resume_seconds_before_audit": scout_finished - started,
        "peak_pytorch_gpu_allocation_bytes": torch.cuda.max_memory_allocated(),
    }
    if args.local_only:
        runtime["total_seconds_this_invocation"] = time.perf_counter() - started
        runtime["determinism_audit_included"] = False
        (args.output_dir / "runtime.json").write_text(canonical_json(runtime), encoding="utf-8")
        print("Stage 19 local-only passed", flush=True)
        return

    extension_records: list[dict[str, Any]] = []
    for init_state_index in range(EXTENSION_START, EXTENSION_END + 1):
        path = record_path(args.output_dir, init_state_index)
        if not path.exists():
            raise AssertionError(f"Missing formal extension record: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_record(
            record,
            init_state_index=init_state_index,
            protocol_sha256=protocol_sha256,
        )
        extension_records.append(record)

    # Predeclared deterministic replay: first index in the extension block.
    audit_init = EXTENSION_START
    env_cfg, env, envs = STAGE5.make_task_env(
        suite="libero_spatial",
        task_id=EXTENSION_TASK_ID,
        n_envs=1,
        resolution=STAGE18.RESOLUTION,
    )
    try:
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, config)
        audit = STAGE18.run_episode(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task_id=EXTENSION_TASK_ID,
            init_state_index=audit_init,
        )
    finally:
        close_envs(envs)
    reference = next(row for row in extension_records if int(row["init_state_index"]) == audit_init)
    audit_fields = ("success", "steps", "initial_observation_sha256", "action_sha256")
    if any(audit[field] != reference[field] for field in audit_fields):
        raise AssertionError("Stage 19 deterministic replay did not reproduce bit-exactly")
    runtime["total_seconds_this_invocation"] = time.perf_counter() - started
    runtime["determinism_audit_included"] = True
    (args.output_dir / "runtime.json").write_text(canonical_json(runtime), encoding="utf-8")

    extension_rows = [public_episode_row(record, source_stage=19) for record in extension_records]
    candidate_rows = [row for row in stage18_rows if int(row["benchmark_task_id"]) in COHORT_TASK_IDS]
    candidate_rows.extend(extension_rows)
    selected, readiness = build_balanced_cohort(candidate_rows)
    stress_rows = [
        {
            **row,
            "role": "policy_floor_stress_set",
            "included_in_balanced_cohort": False,
        }
        for row in stage18_rows
        if int(row["benchmark_task_id"]) == STRESS_TASK_ID
    ]
    extension_csv = csv_text(extension_rows)
    stress_csv = csv_text(stress_rows)
    cohort_csv = csv_text(selected) if selected else ""
    extension_successes = sum(bool(row["success"]) for row in extension_rows)
    report = {
        "schema_version": 1,
        "stage": "WM Stage 19 score-blind balanced cohort freeze",
        "status": "passed" if readiness["ready"] else "insufficient_balance",
        "scope": {
            "extension_task_id": EXTENSION_TASK_ID,
            "extension_initial_state_indices": list(range(EXTENSION_START, EXTENSION_END + 1)),
            "extension_episodes": len(extension_rows),
            "extension_successes": extension_successes,
            "extension_failures": len(extension_rows) - extension_successes,
            "cohort_candidate_episodes": len(candidate_rows),
            "stress_episodes": len(stress_rows),
            "test_demonstration_episodes_used": 0,
            "complete_observations_saved": 0,
        },
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "frozen_world_model": {
            "stage17_primary_score": stage17["primary_result"]["score"],
            "failure_direction": "higher score means more likely failure",
            "world_model_scores_computed_in_stage19": 0,
            "thresholds_fitted_in_stage19": 0,
            "selection_uses_stage17_scores": False,
        },
        "task5_stress_set": {
            "role": "policy-floor external stress set",
            "episodes": len(stress_rows),
            "successes": sum(bool(row["success"]) for row in stress_rows),
            "failures": sum(not bool(row["success"]) for row in stress_rows),
            "included_in_balanced_cohort": False,
            "reason": (
                "Task 5 has no successes in official initial states 3-14 and cannot "
                "support within-task success/failure discrimination."
            ),
        },
        "cohort_readiness": readiness,
        "cohort": {
            "frozen": readiness["ready"],
            "episodes": len(selected),
            "task_ids": list(COHORT_TASK_IDS),
            "successes": sum(bool(row["success"]) for row in selected),
            "failures": sum(not bool(row["success"]) for row in selected),
            "calibration_episodes": sum(row["split"] == "calibration" for row in selected),
            "test_episodes": sum(row["split"] == "test" for row in selected),
            "calibration_test_episode_disjoint": readiness["ready"]
            and len({row["rollout_id"] for row in selected}) == len(selected),
            "selection_rule": (
                "Within each task and label, sort by SHA-256 of the fixed Stage 19 "
                "salt and canonical episode key; take three, assigning ranks 0-1 "
                "to calibration and rank 2 to test."
            ),
        },
        "determinism_audit": {
            "task_id": EXTENSION_TASK_ID,
            "init_state_index": audit_init,
            "bit_exact_fields": list(audit_fields),
            "passed": True,
        },
        "artifacts": {
            "task8_extension_csv": "task8_extension.csv",
            "task8_extension_csv_sha256": sha256_text(extension_csv),
            "task5_stress_set_csv": "task5_stress_set.csv",
            "task5_stress_set_csv_sha256": sha256_text(stress_csv),
            "balanced_cohort_csv": "balanced_cohort.csv" if selected else None,
            "balanced_cohort_csv_sha256": sha256_text(cohort_csv) if selected else None,
            "raw_observations_committed": False,
        },
        "stage20_readiness": {
            "balanced_cohort_frozen": readiness["ready"],
            "complete_observation_recollection_can_start": readiness["ready"],
            "stage17_score_remains_frozen": True,
            "continued_natural_failure_search_recommended": False,
            "paired_action_intervention_pilot_recommended": not readiness["ready"],
        },
        "limitations": [
            "The cohort is score-blind but label-aware by construction.",
            "Task 5 is excluded from AUROC rather than converted into an artificial positive class.",
            "No complete observations or world-model features are collected in this stage.",
            "The test split remains untouched by world-model scoring and threshold calibration.",
        ],
    }
    STAGE18.write_deterministic(
        args.public_results_dir / "task8_extension.csv",
        extension_csv,
        overwrite=args.overwrite_public,
    )
    STAGE18.write_deterministic(
        args.public_results_dir / "task5_stress_set.csv",
        stress_csv,
        overwrite=args.overwrite_public,
    )
    if selected:
        STAGE18.write_deterministic(
            args.public_results_dir / "balanced_cohort.csv",
            cohort_csv,
            overwrite=args.overwrite_public,
        )
    STAGE18.write_deterministic(
        args.public_results_dir / "report.json",
        canonical_json(report),
        overwrite=args.overwrite_public,
    )
    print(
        f"Stage 19 {report['status']}: extension={extension_successes}/12 success, "
        f"cohort={len(selected)}, calibration={report['cohort']['calibration_episodes']}, "
        f"test={report['cohort']['test_episodes']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
