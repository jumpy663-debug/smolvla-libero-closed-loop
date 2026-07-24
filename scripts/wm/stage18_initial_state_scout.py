#!/usr/bin/env python

"""Scout unseen LIBERO initial states for a balanced WM failure cohort.

Stage 17's world-model score stays frozen. This stage only runs the pretrained
SmolVLA policy on official LIBERO initial states 3--14, records lightweight
episode labels and action hashes, then pre-registers an episode-disjoint
calibration/test cohort without inspecting any world-model score.
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
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.envs import close_envs, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION

TASK_IDS = (4, 5, 7, 8)
INIT_STATE_START = 3
INIT_STATE_END = 14
BASE_SEED = 180_000
ACTION_HORIZON = 10
RESOLUTION = 360
REQUIRED_PER_LABEL_PER_TASK = 3
CALIBRATION_PER_LABEL_PER_TASK = 2
TEST_PER_LABEL_PER_TASK = 1
SELECTION_SALT = "stage18-balanced-cohort-v1"
CAMERA_KEYS = ("camera1", "camera2")


def load_script_module(name: str) -> ModuleType:
    path = Path(__file__).parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE3 = load_script_module("stage3_single_inference")
STAGE5 = load_script_module("stage5_small_benchmark")
STAGE16 = load_script_module("wm/stage16_collect_closed_loop")


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
        "--stage16-report",
        type=Path,
        default=Path("results/wm/stage16_closed_loop_recollection/report.json"),
    )
    parser.add_argument(
        "--stage17-report",
        type=Path,
        default=Path("results/wm/stage17_failure_signal/report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wm/stage18_initial_state_scout"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage18_initial_state_scout"),
    )
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(TASK_IDS))
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
        raise ValueError("Cannot serialize an empty row list")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_deterministic(path: Path, content: str, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-identical output: {path}")
    path.write_text(content, encoding="utf-8")


def episode_seed(task_id: int, init_state_index: int) -> int:
    if task_id < 0 or init_state_index < 0:
        raise ValueError("Task and initial-state indices must be non-negative")
    return BASE_SEED + task_id * 100 + init_state_index


def array_digest_update(digest: Any, key: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(key.encode())
    digest.update(b"\0")
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(canonical_json(list(array.shape)).encode())
    digest.update(array.tobytes())


def initial_observation_sha256(observation: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    pixels = observation["pixels"]
    robot_state = observation["robot_state"]
    for camera in CAMERA_KEYS:
        array_digest_update(digest, camera, np.asarray(pixels[camera][0]))
    for name, path in STAGE16.STATE_PATHS.items():
        array_digest_update(digest, name, np.asarray(STAGE16.nested_array(robot_state, path)[0]))
    return digest.hexdigest()


def action_trace_sha256(actions: list[np.ndarray]) -> str:
    if not actions:
        raise ValueError("An episode must contain at least one action")
    stacked = np.asarray(actions, dtype=np.float32)
    if stacked.ndim != 2 or stacked.shape[1] != 7:
        raise ValueError(f"Unexpected action trace shape: {stacked.shape}")
    digest = hashlib.sha256()
    array_digest_update(digest, "action", stacked)
    return digest.hexdigest()


def selection_key(row: dict[str, Any]) -> str:
    payload = (
        f"{SELECTION_SALT}|task={int(row['benchmark_task_id'])}|"
        f"success={int(bool(row['success']))}|init={int(row['init_state_index'])}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_balanced_selection(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    task_balance: dict[str, Any] = {}
    ready_tasks: list[int] = []
    blocking_cells: list[dict[str, Any]] = []
    ready = True
    for task_id in TASK_IDS:
        task_balance[str(task_id)] = {}
        task_ready = True
        for success in (False, True):
            label = "success" if success else "failure"
            candidates = sorted(
                (row for row in rows if int(row["benchmark_task_id"]) == task_id and bool(row["success"]) == success),
                key=selection_key,
            )
            enough = len(candidates) >= REQUIRED_PER_LABEL_PER_TASK
            ready &= enough
            task_ready &= enough
            task_balance[str(task_id)][label] = {
                "available": len(candidates),
                "required": REQUIRED_PER_LABEL_PER_TASK,
                "ready": enough,
            }
            if not enough:
                blocking_cells.append(
                    {
                        "benchmark_task_id": task_id,
                        "label": label,
                        "available": len(candidates),
                        "required": REQUIRED_PER_LABEL_PER_TASK,
                        "deficit": REQUIRED_PER_LABEL_PER_TASK - len(candidates),
                    }
                )
                continue
            chosen = candidates[:REQUIRED_PER_LABEL_PER_TASK]
            for rank, row in enumerate(chosen):
                split = "calibration" if rank < CALIBRATION_PER_LABEL_PER_TASK else "test"
                selected.append(
                    {
                        "rollout_id": row["rollout_id"],
                        "benchmark_task_id": task_id,
                        "task_name": row["task_name"],
                        "init_state_index": int(row["init_state_index"]),
                        "success": success,
                        "split": split,
                        "selection_rank_within_task_label": rank,
                        "selection_key": selection_key(row),
                        "episode_seed": int(row["episode_seed"]),
                        "scout_action_sha256": row["action_sha256"],
                        "stage17_score_inspected_for_selection": False,
                        "complete_observation_recollection_required": True,
                    }
                )
        if task_ready:
            ready_tasks.append(task_id)
    selected.sort(
        key=lambda row: (
            int(row["benchmark_task_id"]),
            str(row["split"]),
            bool(row["success"]),
            int(row["selection_rank_within_task_label"]),
        )
    )
    expected = len(TASK_IDS) * REQUIRED_PER_LABEL_PER_TASK * 2
    if ready and len(selected) != expected:
        raise AssertionError(f"Expected {expected} selected episodes, got {len(selected)}")
    return selected if ready else [], {
        "ready": ready,
        "ready_tasks": ready_tasks,
        "blocking_cells": blocking_cells,
        "per_task": task_balance,
    }


def protocol_payload(
    *,
    args: argparse.Namespace,
    policy_sha256: str,
    backbone_sha256: str,
    stage16_sha256: str,
    stage17_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite": args.suite,
        "task_ids": list(TASK_IDS),
        "initial_state_indices": list(range(INIT_STATE_START, INIT_STATE_END + 1)),
        "episode_batch_size": 1,
        "episode_seed_formula": f"{BASE_SEED} + task_id * 100 + init_state_index",
        "n_action_steps": ACTION_HORIZON,
        "resolution": RESOLUTION,
        "policy_sha256": policy_sha256,
        "backbone_sha256": backbone_sha256,
        "stage16_report_sha256": stage16_sha256,
        "stage17_report_sha256": stage17_sha256,
        "required_per_task_label": REQUIRED_PER_LABEL_PER_TASK,
        "calibration_per_task_label": CALIBRATION_PER_LABEL_PER_TASK,
        "test_per_task_label": TEST_PER_LABEL_PER_TASK,
        "selection_salt": SELECTION_SALT,
        "selection_uses_stage17_scores": False,
    }


def record_path(output_dir: Path, task_id: int, init_state_index: int) -> Path:
    return output_dir / "episodes" / f"task_{task_id:02d}_init_{init_state_index:02d}.json"


def validate_record(
    record: dict[str, Any],
    *,
    task_id: int,
    init_state_index: int,
    protocol_sha256: str,
) -> None:
    expected = {
        "benchmark_task_id": task_id,
        "init_state_index": init_state_index,
        "episode_seed": episode_seed(task_id, init_state_index),
        "protocol_sha256": protocol_sha256,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"Resume record mismatch for {key}: {record.get(key)} != {value}")
    if int(record["steps"]) <= 0 or int(record["steps"]) > 280:
        raise AssertionError("Invalid episode length")
    for key in ("initial_observation_sha256", "action_sha256"):
        value = str(record[key])
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise AssertionError(f"Invalid {key}")


def run_episode(
    *,
    env: Any,
    policy: torch.nn.Module,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    task_id: int,
    init_state_index: int,
) -> dict[str, Any]:
    seed = episode_seed(task_id, init_state_index)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    policy.reset()
    env.set_attr("init_state_id", [init_state_index])
    observation, _ = env.reset(seed=[seed])
    initial_hash = initial_observation_sha256(observation)
    max_steps = int(env.call("_max_episode_steps")[0])
    task_name = str(env.call("task")[0])
    actions: list[np.ndarray] = []
    cumulative_reward = 0.0
    episode_success = False

    for step in range(max_steps):
        policy_observation = preprocess_observation(observation)
        try:
            policy_observation["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            policy_observation["task"] = list(env.call("task"))
        policy_observation = env_preprocessor(policy_observation)
        policy_observation = preprocessor(policy_observation)
        with torch.inference_mode():
            action = policy.select_action(policy_observation)
        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
        action_numpy = action.detach().float().cpu().numpy()
        if action_numpy.shape != (1, 7) or not np.isfinite(action_numpy).all():
            raise AssertionError(f"Invalid action at step {step}: {action_numpy.shape}")
        actions.append(action_numpy[0].copy())

        observation, reward, terminated, truncated, info = env.step(action_numpy)
        cumulative_reward += float(np.asarray(reward)[0])
        successes = STAGE16.extract_successes(info, 1)
        episode_success = episode_success or successes[0]
        if bool(np.asarray(terminated)[0]) or bool(np.asarray(truncated)[0]):
            break
    else:
        step = max_steps - 1

    steps = step + 1
    if episode_success and steps >= max_steps:
        raise AssertionError("A successful episode unexpectedly reached the time limit")
    return {
        "rollout_id": f"pretrained-h10-task{task_id:02d}-init{init_state_index:02d}",
        "benchmark_task_id": task_id,
        "task_name": task_name,
        "init_state_index": init_state_index,
        "episode_seed": seed,
        "success": episode_success,
        "steps": steps,
        "cumulative_reward": cumulative_reward,
        "initial_observation_sha256": initial_hash,
        "action_sha256": action_trace_sha256(actions),
        "official_libero_initial_state": True,
        "complete_observations_saved": False,
        "test_demonstration_episode_used": False,
    }


def public_episode_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "rollout_id",
            "benchmark_task_id",
            "task_name",
            "init_state_index",
            "episode_seed",
            "success",
            "steps",
            "cumulative_reward",
            "initial_observation_sha256",
            "action_sha256",
            "official_libero_initial_state",
            "complete_observations_saved",
            "test_demonstration_episode_used",
        )
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 18 requires a CUDA GPU")
    if args.suite != "libero_spatial":
        raise ValueError("Stage 18 protocol is pinned to libero_spatial")
    if len(set(args.task_ids)) != len(args.task_ids):
        raise ValueError("--task-ids must not contain duplicates")
    if any(task_id not in TASK_IDS for task_id in args.task_ids):
        raise ValueError(f"Stage 18 task IDs must be a subset of {TASK_IDS}")

    stage16 = json.loads(args.stage16_report.read_text(encoding="utf-8"))
    stage17 = json.loads(args.stage17_report.read_text(encoding="utf-8"))
    if stage16.get("status") != "passed" or stage17.get("status") != "passed":
        raise AssertionError("Stage 16 and Stage 17 reports must both have passed")
    if stage17["primary_result"]["score"] != "h10_raw_mse":
        raise AssertionError("The frozen Stage 17 primary score changed")

    stage2_manifest = STAGE3.verify_stage2_manifest(args.stage2_manifest)
    backbone_manifest = json.loads((args.backbone_dir.parent / "backbone_manifest.json").read_text(encoding="utf-8"))
    if stage2_manifest["resolved_revision"] != STAGE3.EXPECTED_POLICY_REVISION:
        raise AssertionError("Policy revision differs from the pinned revision")
    if backbone_manifest["resolved_revision"] != STAGE3.BACKBONE_REVISION:
        raise AssertionError("Backbone revision differs from the pinned revision")

    policy_sha256 = sha256_file(args.checkpoint_dir / "model.safetensors")
    backbone_sha256 = sha256_file(args.backbone_dir / "model.safetensors")
    protocol = protocol_payload(
        args=args,
        policy_sha256=policy_sha256,
        backbone_sha256=backbone_sha256,
        stage16_sha256=sha256_file(args.stage16_report),
        stage17_sha256=sha256_file(args.stage17_report),
    )
    protocol_text = canonical_json(protocol)
    protocol_sha256 = sha256_text(protocol_text)
    write_deterministic(args.output_dir / "protocol.json", protocol_text)

    config = PreTrainedConfig.from_pretrained(args.checkpoint_dir, local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    config.device = "cuda"
    config.vlm_model_name = str(args.backbone_dir.resolve())
    config.load_vlm_weights = True
    config.empty_cameras = 1
    config.use_amp = False
    config.n_action_steps = ACTION_HORIZON
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

    all_records: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        for init_state_index in range(INIT_STATE_START, INIT_STATE_END + 1):
            path = record_path(args.output_dir, task_id, init_state_index)
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                validate_record(
                    record,
                    task_id=task_id,
                    init_state_index=init_state_index,
                    protocol_sha256=protocol_sha256,
                )
                all_records.append(record)

    requested = set(args.task_ids)
    for task_id in TASK_IDS:
        missing = [
            init_state_index
            for init_state_index in range(INIT_STATE_START, INIT_STATE_END + 1)
            if not record_path(args.output_dir, task_id, init_state_index).exists()
        ]
        if task_id not in requested or not missing:
            if task_id in requested:
                print(f"Stage 18 resume: task {task_id} already has all 12 labels", flush=True)
            continue
        env_cfg, env, envs = STAGE5.make_task_env(
            suite=args.suite,
            task_id=task_id,
            n_envs=1,
            resolution=RESOLUTION,
        )
        try:
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, config)
            for init_state_index in missing:
                episode_started = time.perf_counter()
                record = run_episode(
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task_id=task_id,
                    init_state_index=init_state_index,
                )
                record["protocol_sha256"] = protocol_sha256
                write_deterministic(
                    record_path(args.output_dir, task_id, init_state_index),
                    canonical_json(record),
                )
                all_records.append(record)
                print(
                    f"Stage 18 task={task_id} init={init_state_index:02d} "
                    f"success={record['success']} steps={record['steps']} "
                    f"seconds={time.perf_counter() - episode_started:.1f}",
                    flush=True,
                )
        finally:
            close_envs(envs)

    finished = time.perf_counter()
    runtime = {
        "schema_version": 1,
        "model_load_seconds": model_loaded - started,
        "total_seconds_this_invocation": finished - started,
        "peak_pytorch_gpu_allocation_bytes": torch.cuda.max_memory_allocated(),
        "task_ids_requested": args.task_ids,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }
    (args.output_dir / "runtime.json").write_text(canonical_json(runtime), encoding="utf-8")
    if args.local_only:
        print(f"Stage 18 local-only passed: tasks={args.task_ids}", flush=True)
        return
    if set(args.task_ids) != set(TASK_IDS):
        raise ValueError("Public finalization requires the complete pinned task set")

    all_records = []
    for task_id in TASK_IDS:
        for init_state_index in range(INIT_STATE_START, INIT_STATE_END + 1):
            path = record_path(args.output_dir, task_id, init_state_index)
            if not path.exists():
                raise AssertionError(f"Missing formal scout record: {path}")
            record = json.loads(path.read_text(encoding="utf-8"))
            validate_record(
                record,
                task_id=task_id,
                init_state_index=init_state_index,
                protocol_sha256=protocol_sha256,
            )
            all_records.append(record)

    # Re-run one predeclared episode after all records exist. Exact initial
    # observation and action hashes guard the single-episode recollection path.
    audit_task, audit_init = TASK_IDS[0], INIT_STATE_START
    env_cfg, env, envs = STAGE5.make_task_env(
        suite=args.suite,
        task_id=audit_task,
        n_envs=1,
        resolution=RESOLUTION,
    )
    try:
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, config)
        audit = run_episode(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task_id=audit_task,
            init_state_index=audit_init,
        )
    finally:
        close_envs(envs)
    reference = next(
        row
        for row in all_records
        if int(row["benchmark_task_id"]) == audit_task and int(row["init_state_index"]) == audit_init
    )
    audit_fields = (
        "success",
        "steps",
        "initial_observation_sha256",
        "action_sha256",
    )
    if any(audit[field] != reference[field] for field in audit_fields):
        raise AssertionError("Deterministic single-episode audit did not reproduce bit-exactly")
    runtime["scout_or_resume_seconds_before_audit"] = finished - started
    runtime["total_seconds_this_invocation"] = time.perf_counter() - started
    runtime["determinism_audit_included"] = True
    (args.output_dir / "runtime.json").write_text(canonical_json(runtime), encoding="utf-8")

    public_rows = [public_episode_row(row) for row in all_records]
    selected, readiness = build_balanced_selection(public_rows)
    scout_csv = csv_text(public_rows)
    selected_csv = csv_text(selected) if selected else ""
    successes = sum(bool(row["success"]) for row in public_rows)
    report = {
        "schema_version": 1,
        "stage": "WM Stage 18 unseen initial-state label scout",
        "status": "passed" if readiness["ready"] else "insufficient_balance",
        "scope": {
            "suite": args.suite,
            "policy": "pretrained SmolVLA",
            "task_ids": list(TASK_IDS),
            "initial_state_indices": list(range(INIT_STATE_START, INIT_STATE_END + 1)),
            "episodes": len(public_rows),
            "successes": successes,
            "failures": len(public_rows) - successes,
            "complete_observations_saved": 0,
            "test_demonstration_episodes_used": 0,
        },
        "frozen_before_scout": {
            "stage17_primary_score": stage17["primary_result"]["score"],
            "stage17_failure_direction": "higher score means more likely failure",
            "stage17_report_sha256": protocol["stage17_report_sha256"],
            "world_model_scores_computed_in_stage18": 0,
        },
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "label_balance": readiness,
        "selection": {
            "ready": readiness["ready"],
            "selected_episodes": len(selected),
            "calibration_episodes": sum(row["split"] == "calibration" for row in selected),
            "test_episodes": sum(row["split"] == "test" for row in selected),
            "selection_rule": (
                "Within each task and label, sort unseen episodes by SHA-256 of the fixed "
                "selection salt and canonical episode key; take the first three, assigning "
                "ranks 0-1 to calibration and rank 2 to test."
            ),
            "selection_uses_stage17_scores": False,
            "stage16_exploratory_episodes_selected": 0,
        },
        "determinism_audit": {
            "task_id": audit_task,
            "init_state_index": audit_init,
            "bit_exact_fields": list(audit_fields),
            "passed": True,
        },
        "artifacts": {
            "scout_episodes_csv": "scout_episodes.csv",
            "scout_episodes_csv_sha256": sha256_text(scout_csv),
            "selected_episodes_csv": "selected_episodes.csv" if selected else None,
            "selected_episodes_csv_sha256": sha256_text(selected_csv) if selected else None,
            "raw_observations_committed": False,
        },
        "stage19_readiness": {
            "balanced_episode_list_frozen": readiness["ready"],
            "complete_observation_recollection_can_start": readiness["ready"],
            "stage17_score_remains_frozen": True,
        },
        "limitations": [
            "Labels are known because this is a case-control cohort construction stage.",
            "No world-model score, threshold, classifier, or policy parameter is fitted here.",
            "Any future selected test cohort is score-blind but not label-blind.",
            "Complete observations must be recollected before world-model evaluation.",
        ],
    }
    write_deterministic(
        args.public_results_dir / "scout_episodes.csv",
        scout_csv,
        overwrite=args.overwrite_public,
    )
    if selected:
        write_deterministic(
            args.public_results_dir / "selected_episodes.csv",
            selected_csv,
            overwrite=args.overwrite_public,
        )
    write_deterministic(
        args.public_results_dir / "report.json",
        canonical_json(report),
        overwrite=args.overwrite_public,
    )
    print(
        f"Stage 18 {report['status']}: episodes={len(public_rows)}, "
        f"success={successes}, failure={len(public_rows) - successes}, "
        f"selected={len(selected)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
