#!/usr/bin/env python
# ruff: noqa: E501

"""Build a score-blind paired cohort with hidden MuJoCo dynamics shifts.

The action passed through the policy/environment interface is unchanged.
At step 50, either arm actuator gain or arm joint damping is modified inside
MuJoCo.  A three-pair pilot gates which shifts advance to the six-pair
discovery cohort; no DINO feature or world-model score is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import stage22_collect_confirmatory_interventions as STAGE22
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.envs import close_envs, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION

STAGE16 = STAGE22.STAGE16
STAGE18 = STAGE22.STAGE18
STAGE3 = STAGE22.STAGE3
STAGE5 = STAGE22.STAGE5

TASK_IDS = (4, 7, 8)
CONDITIONS = (
    "nominal",
    "arm_actuator_gain_0p5",
    "arm_joint_damping_10x",
)
FAULT_CONDITIONS = CONDITIONS[1:]
INTERVENTION_START = 50
LIVE_STEPS = 90
POST_HORIZON = 10
ACTUATOR_GAIN_SCALE = 0.5
JOINT_DAMPING_SCALE = 10.0
PILOT_MINIMUM_PASSING_PAIRS = 2
MINIMUM_EEF_DIVERGENCE_M = 0.001
MINIMUM_PIXEL_MAE = 0.25
EXPECTED_DISCOVERY = {
    4: (3, 13),
    7: (10, 13),
    8: (16, 5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage25-protocol",
        type=Path,
        default=Path("results/wm/stage25_detector_calibration/protocol.json"),
    )
    parser.add_argument(
        "--stage25-report",
        type=Path,
        default=Path("results/wm/stage25_detector_calibration/report.json"),
    )
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
        "--output-dir",
        type=Path,
        default=Path("outputs/wm/stage26_hidden_dynamics_cohort"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage26_hidden_dynamics_cohort"),
    )
    parser.add_argument(
        "--media-path",
        type=Path,
        default=Path("media/wm_stage26_hidden_dynamics_cohort.svg"),
    )
    parser.add_argument("--overwrite-public", action="store_true")
    return parser.parse_args()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode())
    digest.update(canonical_json(list(contiguous.shape)).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def write_output(
    path: Path,
    content: str,
    *,
    overwrite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-identical output: {path}")
    path.write_text(content, encoding="utf-8")


def csv_text(rows: list[dict[str, Any]]) -> str:
    return STAGE22.STAGE20.csv_text(rows)


def discovery_pairs(stage25_protocol: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [dict(row) for row in stage25_protocol["pairs"] if row["split"] == "calibration"]
    observed = {
        task_id: tuple(int(row["init_state_index"]) for row in selected if int(row["benchmark_task_id"]) == task_id)
        for task_id in TASK_IDS
    }
    if observed != EXPECTED_DISCOVERY:
        raise AssertionError(f"Stage 26 discovery states changed: {observed}")
    if len(selected) != 6:
        raise AssertionError("Stage 26 requires six discovery pairs")
    for row in selected:
        row["stage26_pair_id"] = f"task{int(row['benchmark_task_id']):02d}-init{int(row['init_state_index']):02d}"
        row["pilot_pair"] = int(row["init_state_index"]) == EXPECTED_DISCOVERY[int(row["benchmark_task_id"])][0]
    return selected


def protocol_payload(
    *,
    pairs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tasks": list(TASK_IDS),
        "pairs": pairs,
        "source_split": "Stage 25 calibration states reused as score-blind hidden-dynamics discovery states",
        "stage25_state_selection_used_wm_score": False,
        "conditions": list(CONDITIONS),
        "intervention_start_zero_based": INTERVENTION_START,
        "diagnostic_prefix_steps": LIVE_STEPS,
        "first_post_horizon_steps": POST_HORIZON,
        "action_interface_contract": (
            "commanded_action and action passed to env.step are bit-exact under every condition"
        ),
        "mutations": {
            "arm_actuator_gain_0p5": {
                "target": "MuJoCo actuator_gainprm[:,0] for robot0_torq_j1..j7",
                "operation": "multiply by 0.5 once before env.step at step 50",
            },
            "arm_joint_damping_10x": {
                "target": "MuJoCo dof_damping for robot0_joint1..joint7",
                "operation": "multiply by 10 once before env.step at step 50",
            },
        },
        "pilot_pairs": [row["stage26_pair_id"] for row in pairs if row["pilot_pair"]],
        "pilot_gate_per_condition": {
            "minimum_passing_pairs_out_of_three": PILOT_MINIMUM_PASSING_PAIRS,
            "each_pair_requires": {
                "action_interface_mismatch_steps": 0,
                "preintervention_actions_bit_exact": True,
                "preintervention_observations_bit_exact": True,
                "first_post_h10_commands_bit_exact": True,
                "mutation_verified": True,
                "max_first_h10_eef_divergence_m_at_least": (MINIMUM_EEF_DIVERGENCE_M),
                "observation_60_dual_camera_pixel_mae_at_least": (MINIMUM_PIXEL_MAE),
            },
        },
        "remaining_pairs_collected_only_after_gate_freeze": True,
        "world_model_loaded_during_collection": False,
        "world_model_scores_computed": 0,
        "source_hashes": {
            "stage25_protocol": sha256_file(args.stage25_protocol),
            "stage25_report": sha256_file(args.stage25_report),
        },
    }


def underlying_sim(vector_env: Any) -> Any:
    raw_env = vector_env.envs[0]
    if raw_env._env is None:
        raise RuntimeError("LIBERO environment has not been reset")
    return raw_env._env.env.sim


def arm_parameter_indices(model: Any) -> tuple[list[int], list[int]]:
    actuator_indices = []
    for joint_number in range(1, 8):
        name = f"robot0_torq_j{joint_number}"
        index = int(model.actuator_name2id(name))
        if index < 0:
            raise KeyError(f"Missing actuator: {name}")
        actuator_indices.append(index)
    dof_indices = []
    for joint_number in range(1, 8):
        name = f"robot0_joint{joint_number}"
        joint_id = int(model.joint_name2id(name))
        if joint_id < 0:
            raise KeyError(f"Missing joint: {name}")
        dof_indices.append(int(model.jnt_dofadr[joint_id]))
    if len(set(actuator_indices)) != 7 or len(set(dof_indices)) != 7:
        raise AssertionError("Arm parameter indices are not unique")
    return actuator_indices, dof_indices


def parameter_snapshot(sim: Any) -> dict[str, list[float]]:
    actuator_indices, dof_indices = arm_parameter_indices(sim.model)
    return {
        "arm_actuator_gain": [float(sim.model.actuator_gainprm[index, 0]) for index in actuator_indices],
        "arm_joint_damping": [float(sim.model.dof_damping[index]) for index in dof_indices],
    }


def apply_hidden_dynamics_shift(
    vector_env: Any,
    condition: str,
) -> dict[str, Any]:
    sim = underlying_sim(vector_env)
    before = parameter_snapshot(sim)
    actuator_indices, dof_indices = arm_parameter_indices(sim.model)
    if condition == "arm_actuator_gain_0p5":
        sim.model.actuator_gainprm[actuator_indices, 0] *= ACTUATOR_GAIN_SCALE
    elif condition == "arm_joint_damping_10x":
        sim.model.dof_damping[dof_indices] *= JOINT_DAMPING_SCALE
    elif condition != "nominal":
        raise ValueError(f"Unknown hidden dynamics condition: {condition}")
    sim.forward()
    after = parameter_snapshot(sim)
    if condition == "nominal":
        verified = before == after
    elif condition == "arm_actuator_gain_0p5":
        verified = (
            np.allclose(
                after["arm_actuator_gain"],
                np.asarray(before["arm_actuator_gain"]) * ACTUATOR_GAIN_SCALE,
                rtol=0,
                atol=0,
            )
            and before["arm_joint_damping"] == after["arm_joint_damping"]
        )
    else:
        verified = (
            np.allclose(
                after["arm_joint_damping"],
                np.asarray(before["arm_joint_damping"]) * JOINT_DAMPING_SCALE,
                rtol=0,
                atol=0,
            )
            and before["arm_actuator_gain"] == after["arm_actuator_gain"]
        )
    if not verified:
        raise AssertionError(f"Hidden dynamics mutation failed: {condition}")
    return {
        "condition": condition,
        "before": before,
        "after": after,
        "verified": verified,
    }


def collect_condition(
    *,
    env: Any,
    policy: torch.nn.Module,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    task_id: int,
    init_state_index: int,
    condition: str,
) -> tuple[
    dict[str, list[np.ndarray]],
    dict[str, torch.Tensor],
    str,
    dict[str, Any],
]:
    seed = STAGE18.episode_seed(task_id, init_state_index)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    policy.reset()
    env.set_attr("init_state_id", [init_state_index])
    observation, _ = env.reset(seed=[seed])
    initial_hash = STAGE18.initial_observation_sha256(observation)
    captured = {key: [] for key in (*STAGE16.CAMERA_KEYS, *STAGE16.STATE_PATHS)}
    STAGE16.capture_observation(observation, captured)
    commanded_actions = []
    interface_actions = []
    rewards = []
    successes = []
    mutation_mask = []
    mutation_record = None

    for step in range(LIVE_STEPS):
        policy_observation = preprocess_observation(observation)
        try:
            policy_observation["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            policy_observation["task"] = list(env.call("task"))
        policy_observation = env_preprocessor(policy_observation)
        policy_observation = preprocessor(policy_observation)
        with torch.inference_mode():
            commanded = policy.select_action(policy_observation)
        commanded = postprocessor(commanded)
        commanded = env_postprocessor({ACTION: commanded})[ACTION]
        commanded_numpy = commanded.detach().float().cpu().numpy()
        if commanded_numpy.shape != (1, 7):
            raise AssertionError("Unexpected Stage 26 action shape")
        if step == INTERVENTION_START:
            mutation_record = apply_hidden_dynamics_shift(env, condition)
        interface_action = commanded_numpy.copy()
        observation, reward, terminated, truncated, info = env.step(interface_action)
        STAGE16.capture_observation(observation, captured)
        if bool((terminated | truncated)[0]) and step + 1 < LIVE_STEPS:
            raise RuntimeError(
                "Stage 26 diagnostic prefix terminated early at "
                f"step {step + 1}: task={task_id}, "
                f"init={init_state_index}, condition={condition}"
            )
        step_successes = STAGE16.extract_successes(info, 1)
        commanded_actions.append(torch.from_numpy(commanded_numpy.copy()))
        interface_actions.append(torch.from_numpy(interface_action.copy()))
        rewards.append(torch.from_numpy(np.asarray(reward).copy()))
        successes.append(torch.tensor(step_successes, dtype=torch.bool))
        mutation_mask.append(
            torch.tensor(
                [condition != "nominal" and step >= INTERVENTION_START],
                dtype=torch.bool,
            )
        )
    if mutation_record is None:
        raise AssertionError("Stage 26 mutation point was not reached")
    return (
        captured,
        {
            "commanded_action": torch.cat(commanded_actions),
            "interface_action": torch.cat(interface_actions),
            "reward": torch.cat(rewards),
            "success": torch.cat(successes),
            "mutation_mask": torch.cat(mutation_mask),
        },
        initial_hash,
        mutation_record,
    )


def episode_paths(
    output_dir: Path,
    pair_id: str,
    condition: str,
) -> tuple[Path, Path]:
    base = output_dir / "episodes" / f"{pair_id}-{condition}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def validate_arrays(
    arrays: dict[str, np.ndarray],
    *,
    condition: str,
) -> None:
    required = {
        *STAGE16.CAMERA_KEYS,
        *STAGE16.STATE_PATHS,
        "commanded_action",
        "interface_action",
        "reward",
        "success",
        "mutation_mask",
    }
    if set(arrays) != required:
        raise AssertionError(f"Unexpected Stage 26 keys: {sorted(set(arrays) ^ required)}")
    for key in ("commanded_action", "interface_action"):
        if arrays[key].shape != (LIVE_STEPS, 7):
            raise AssertionError(f"Unexpected {key} shape")
        if arrays[key].dtype != np.float32:
            raise AssertionError(f"Unexpected {key} dtype")
    if not np.array_equal(
        arrays["commanded_action"],
        arrays["interface_action"],
    ):
        raise AssertionError("Stage 26 action interface contains mismatch")
    for camera in STAGE16.CAMERA_KEYS:
        if arrays[camera].shape != (
            LIVE_STEPS + 1,
            STAGE18.RESOLUTION,
            STAGE18.RESOLUTION,
            3,
        ):
            raise AssertionError(f"Unexpected camera shape: {camera}")
    for key in STAGE16.STATE_PATHS:
        if len(arrays[key]) != LIVE_STEPS + 1:
            raise AssertionError(f"Unexpected state length: {key}")
    expected_mask = (
        np.zeros(LIVE_STEPS, dtype=np.bool_) if condition == "nominal" else np.arange(LIVE_STEPS) >= INTERVENTION_START
    )
    if not np.array_equal(arrays["mutation_mask"], expected_mask):
        raise AssertionError("Stage 26 mutation mask changed")


def load_episode(path: Path, condition: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_arrays(arrays, condition=condition)
    return arrays


def save_episode(
    *,
    output_dir: Path,
    pair: dict[str, Any],
    condition: str,
    captured: dict[str, list[np.ndarray]],
    rollout: dict[str, torch.Tensor],
    initial_hash: str,
    mutation: dict[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    archive_path, metadata_path = episode_paths(
        output_dir,
        pair["stage26_pair_id"],
        condition,
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.stack([batch[0] for batch in values]) for key, values in captured.items()}
    arrays.update(
        {
            "commanded_action": rollout["commanded_action"].float().numpy(),
            "interface_action": rollout["interface_action"].float().numpy(),
            "reward": rollout["reward"].float().numpy(),
            "success": rollout["success"].numpy().astype(np.bool_),
            "mutation_mask": rollout["mutation_mask"].numpy().astype(np.bool_),
        }
    )
    validate_arrays(arrays, condition=condition)
    temporary = archive_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(archive_path)
    record = {
        "pair_id": pair["stage26_pair_id"],
        "benchmark_task_id": int(pair["benchmark_task_id"]),
        "task_name": pair["task_name"],
        "init_state_index": int(pair["init_state_index"]),
        "episode_seed": int(pair["episode_seed"]),
        "pilot_pair": bool(pair["pilot_pair"]),
        "condition": condition,
        "steps": LIVE_STEPS,
        "success_within_prefix": bool(arrays["success"].any()),
        "initial_observation_sha256": initial_hash,
        "pre_observation_sha256": STAGE22.STAGE20.observation_prefix_sha256(
            arrays,
            observations=INTERVENTION_START + 1,
        ),
        "pre_commanded_action_sha256": array_sha256(arrays["commanded_action"][:INTERVENTION_START]),
        "first_post_h10_command_sha256": array_sha256(
            arrays["commanded_action"][INTERVENTION_START : INTERVENTION_START + POST_HORIZON]
        ),
        "commanded_action_sha256": array_sha256(arrays["commanded_action"]),
        "interface_action_sha256": array_sha256(arrays["interface_action"]),
        "action_interface_mismatch_steps": int(
            np.any(
                arrays["commanded_action"] != arrays["interface_action"],
                axis=1,
            ).sum()
        ),
        "mutation": mutation,
        "protocol_sha256": protocol_sha256,
        "archive_file": archive_path.as_posix(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
    }
    write_output(metadata_path, canonical_json(record))
    return record


def resume_episode(
    *,
    output_dir: Path,
    pair: dict[str, Any],
    condition: str,
    protocol_sha256: str,
) -> dict[str, Any] | None:
    archive_path, metadata_path = episode_paths(
        output_dir,
        pair["stage26_pair_id"],
        condition,
    )
    if not archive_path.exists() and not metadata_path.exists():
        return None
    if not archive_path.exists() or not metadata_path.exists():
        raise RuntimeError(f"Partial Stage 26 output: {pair['stage26_pair_id']} {condition}")
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "pair_id": pair["stage26_pair_id"],
        "benchmark_task_id": int(pair["benchmark_task_id"]),
        "init_state_index": int(pair["init_state_index"]),
        "condition": condition,
        "protocol_sha256": protocol_sha256,
        "steps": LIVE_STEPS,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise AssertionError(f"Stage 26 resume mismatch: {key}")
    if sha256_file(archive_path) != record["archive_sha256"]:
        raise AssertionError("Stage 26 archive hash changed")
    load_episode(archive_path, condition)
    return record


def paired_diagnostic(
    *,
    output_dir: Path,
    pair: dict[str, Any],
    records: dict[str, dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    nominal_path, _ = episode_paths(
        output_dir,
        pair["stage26_pair_id"],
        "nominal",
    )
    fault_path, _ = episode_paths(
        output_dir,
        pair["stage26_pair_id"],
        condition,
    )
    nominal = load_episode(nominal_path, "nominal")
    fault = load_episode(fault_path, condition)
    begin = INTERVENTION_START + 1
    end = INTERVENTION_START + POST_HORIZON + 1
    eef_divergence = np.linalg.norm(
        fault["eef_pos"][begin:end] - nominal["eef_pos"][begin:end],
        axis=1,
    )
    joint_divergence = np.linalg.norm(
        fault["joint_pos"][begin:end] - nominal["joint_pos"][begin:end],
        axis=1,
    )
    target = INTERVENTION_START + POST_HORIZON
    pixel_values = []
    for camera in STAGE16.CAMERA_KEYS:
        pixel_values.append(
            np.abs(fault[camera][target].astype(np.float32) - nominal[camera][target].astype(np.float32)).mean()
        )
    record = {
        "pair_id": pair["stage26_pair_id"],
        "benchmark_task_id": int(pair["benchmark_task_id"]),
        "init_state_index": int(pair["init_state_index"]),
        "pilot_pair": bool(pair["pilot_pair"]),
        "condition": condition,
        "initial_observations_bit_exact": (
            records["nominal"]["initial_observation_sha256"] == records[condition]["initial_observation_sha256"]
        ),
        "preintervention_observations_bit_exact": (
            records["nominal"]["pre_observation_sha256"] == records[condition]["pre_observation_sha256"]
        ),
        "preintervention_commands_bit_exact": (
            records["nominal"]["pre_commanded_action_sha256"] == records[condition]["pre_commanded_action_sha256"]
        ),
        "first_post_h10_commands_bit_exact": (
            records["nominal"]["first_post_h10_command_sha256"] == records[condition]["first_post_h10_command_sha256"]
        ),
        "action_interface_mismatch_steps": int(records[condition]["action_interface_mismatch_steps"]),
        "mutation_verified": bool(records[condition]["mutation"]["verified"]),
        "first_h10_eef_divergence_m_max": float(eef_divergence.max()),
        "first_h10_eef_divergence_m_at_observation_60": float(eef_divergence[-1]),
        "first_h10_joint_divergence_l2_max": float(joint_divergence.max()),
        "observation_60_dual_camera_pixel_mae": float(np.mean(pixel_values)),
    }
    record["state_effect_gate"] = record["first_h10_eef_divergence_m_max"] >= MINIMUM_EEF_DIVERGENCE_M
    record["visual_effect_gate"] = record["observation_60_dual_camera_pixel_mae"] >= MINIMUM_PIXEL_MAE
    record["pair_gate_passed"] = (
        record["initial_observations_bit_exact"]
        and record["preintervention_observations_bit_exact"]
        and record["preintervention_commands_bit_exact"]
        and record["first_post_h10_commands_bit_exact"]
        and record["action_interface_mismatch_steps"] == 0
        and record["mutation_verified"]
        and record["state_effect_gate"]
        and record["visual_effect_gate"]
    )
    return record


def gate_conditions(
    diagnostics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    results = {}
    for condition in FAULT_CONDITIONS:
        rows = [row for row in diagnostics if row["condition"] == condition and row["pilot_pair"]]
        if len(rows) != len(TASK_IDS):
            raise AssertionError(f"Incomplete pilot diagnostics: {condition}")
        passing = sum(bool(row["pair_gate_passed"]) for row in rows)
        results[condition] = {
            "pilot_pairs": len(rows),
            "passing_pairs": passing,
            "minimum_passing_pairs": PILOT_MINIMUM_PASSING_PAIRS,
            "advanced_to_discovery": (passing >= PILOT_MINIMUM_PASSING_PAIRS),
            "eef_divergence_m_max_across_pairs": float(max(row["first_h10_eef_divergence_m_max"] for row in rows)),
            "pixel_mae_median": float(np.median([row["observation_60_dual_camera_pixel_mae"] for row in rows])),
        }
    return results


def collect_one(
    *,
    args: argparse.Namespace,
    pair: dict[str, Any],
    condition: str,
    policy: torch.nn.Module,
    config: SmolVLAConfig,
    preprocessor: Any,
    postprocessor: Any,
    protocol_sha256: str,
) -> dict[str, Any]:
    resumed = resume_episode(
        output_dir=args.output_dir,
        pair=pair,
        condition=condition,
        protocol_sha256=protocol_sha256,
    )
    if resumed is not None:
        print(
            f"resume {pair['stage26_pair_id']} {condition}",
            flush=True,
        )
        return resumed
    env_cfg, env, envs = STAGE5.make_task_env(
        suite="libero_spatial",
        task_id=int(pair["benchmark_task_id"]),
        n_envs=1,
        resolution=STAGE18.RESOLUTION,
    )
    try:
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, config)
        started = time.perf_counter()
        captured, rollout, initial_hash, mutation = collect_condition(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task_id=int(pair["benchmark_task_id"]),
            init_state_index=int(pair["init_state_index"]),
            condition=condition,
        )
        record = save_episode(
            output_dir=args.output_dir,
            pair=pair,
            condition=condition,
            captured=captured,
            rollout=rollout,
            initial_hash=initial_hash,
            mutation=mutation,
            protocol_sha256=protocol_sha256,
        )
        print(
            f"{pair['stage26_pair_id']} {condition}: "
            f"seconds={time.perf_counter() - started:.1f} "
            f"bytes={record['archive_bytes']}",
            flush=True,
        )
        return record
    finally:
        close_envs(envs)


def result_svg(
    gate: dict[str, dict[str, Any]],
    formal_diagnostics: list[dict[str, Any]],
) -> str:
    gain = gate["arm_actuator_gain_0p5"]
    damping = gate["arm_joint_damping_10x"]
    eef_median = float(np.median([row["first_h10_eef_divergence_m_max"] for row in formal_diagnostics]))
    pixel_median = float(np.median([row["observation_60_dual_camera_pixel_mae"] for row in formal_diagnostics]))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="570" viewBox="0 0 1200 570">
<rect width="1200" height="570" fill="#f7f8fa"/>
<text x="65" y="65" font-size="31" font-family="sans-serif" font-weight="700" fill="#162033">Stage 26：隐藏动力学偏移队列</text>
<text x="65" y="105" font-size="18" font-family="sans-serif" fill="#526070">action 接口逐位相同；只改变 MuJoCo 内部动力学；全程不查看 WM 分数</text>
<rect x="65" y="145" width="500" height="150" rx="14" fill="#ffffff" stroke="#dce1e8"/>
<text x="95" y="188" font-size="20" font-family="sans-serif" font-weight="700" fill="#162033">0.5× arm actuator gain</text>
<text x="95" y="245" font-size="35" font-family="monospace" fill="#e36b4f">{gain["passing_pairs"]}/3 pilot gate</text>
<rect x="635" y="145" width="500" height="150" rx="14" fill="#ffffff" stroke="#dce1e8"/>
<text x="665" y="188" font-size="20" font-family="sans-serif" font-weight="700" fill="#162033">10× arm joint damping</text>
<text x="665" y="245" font-size="35" font-family="monospace" fill="#d99a31">{damping["passing_pairs"]}/3 pilot gate</text>
<text x="65" y="355" font-size="22" font-family="sans-serif" font-weight="700" fill="#162033">通过条件的 6-pair discovery cohort</text>
<text x="65" y="405" font-size="19" font-family="sans-serif" fill="#32445a">H10 EEF divergence median: {eef_median * 1000:.2f} mm</text>
<text x="65" y="448" font-size="19" font-family="sans-serif" fill="#32445a">Observation 60 dual-camera pixel MAE median: {pixel_median:.3f}</text>
<text x="65" y="515" font-size="17" font-family="sans-serif" fill="#6a7583">下一阶段才首次计算 observation residual；本图不包含任何 WM score。</text>
</svg>
"""


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 26 requires CUDA for SmolVLA")
    stage25_protocol = json.loads(args.stage25_protocol.read_text(encoding="utf-8"))
    stage25_report = json.loads(args.stage25_report.read_text(encoding="utf-8"))
    if not stage25_report["stage26_readiness"]["evaluate_observation_or_state_dynamics_shift"]:
        raise AssertionError("Stage 25 did not authorize hidden dynamics")
    pairs = discovery_pairs(stage25_protocol)
    protocol = protocol_payload(pairs=pairs, args=args)
    protocol_text = canonical_json(protocol)
    protocol_sha256 = sha256_text(protocol_text)
    write_output(args.output_dir / "protocol.json", protocol_text)

    STAGE3.verify_stage2_manifest(args.stage2_manifest)
    config = PreTrainedConfig.from_pretrained(
        args.checkpoint_dir,
        local_files_only=True,
    )
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
            "tokenizer_processor": {
                "tokenizer_name": str(args.backbone_dir.resolve()),
            },
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
    ).eval()
    model_loaded = time.perf_counter()

    records: dict[str, dict[str, dict[str, Any]]] = {}
    pilot_pairs = [row for row in pairs if row["pilot_pair"]]
    for pair in pilot_pairs:
        records[pair["stage26_pair_id"]] = {}
        for condition in CONDITIONS:
            records[pair["stage26_pair_id"]][condition] = collect_one(
                args=args,
                pair=pair,
                condition=condition,
                policy=policy,
                config=config,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                protocol_sha256=protocol_sha256,
            )
    pilot_diagnostics = []
    for pair in pilot_pairs:
        for condition in FAULT_CONDITIONS:
            pilot_diagnostics.append(
                paired_diagnostic(
                    output_dir=args.output_dir,
                    pair=pair,
                    records=records[pair["stage26_pair_id"]],
                    condition=condition,
                )
            )
    gate = gate_conditions(pilot_diagnostics)
    selected_faults = tuple(condition for condition in FAULT_CONDITIONS if gate[condition]["advanced_to_discovery"])
    if not selected_faults:
        raise AssertionError("No hidden dynamics condition passed the pilot")
    gate_payload = {
        "schema_version": 1,
        "protocol_sha256": protocol_sha256,
        "selected_fault_conditions": list(selected_faults),
        "rejected_fault_conditions": [condition for condition in FAULT_CONDITIONS if condition not in selected_faults],
        "pilot_results": gate,
        "world_model_scores_viewed": False,
    }
    gate_text = canonical_json(gate_payload)
    write_output(args.output_dir / "frozen_gate.json", gate_text)
    print(f"Stage 26 pilot gate frozen: {selected_faults}", flush=True)

    remaining_pairs = [row for row in pairs if not row["pilot_pair"]]
    for pair in remaining_pairs:
        records[pair["stage26_pair_id"]] = {}
        for condition in ("nominal", *selected_faults):
            records[pair["stage26_pair_id"]][condition] = collect_one(
                args=args,
                pair=pair,
                condition=condition,
                policy=policy,
                config=config,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                protocol_sha256=protocol_sha256,
            )
    formal_diagnostics = []
    for pair in pairs:
        for condition in selected_faults:
            formal_diagnostics.append(
                paired_diagnostic(
                    output_dir=args.output_dir,
                    pair=pair,
                    records=records[pair["stage26_pair_id"]],
                    condition=condition,
                )
            )
    if not all(
        bool(row["initial_observations_bit_exact"])
        and bool(row["preintervention_observations_bit_exact"])
        and bool(row["preintervention_commands_bit_exact"])
        and bool(row["first_post_h10_commands_bit_exact"])
        and int(row["action_interface_mismatch_steps"]) == 0
        and bool(row["mutation_verified"])
        for row in formal_diagnostics
    ):
        raise AssertionError("Stage 26 formal pairing contract failed")

    episode_rows = []
    all_records = []
    for pair_id in sorted(records):
        for condition in CONDITIONS:
            if condition not in records[pair_id]:
                continue
            record = records[pair_id][condition]
            all_records.append(record)
            episode_rows.append(
                {key: value for key, value in record.items() if key not in {"archive_file", "mutation"}}
            )
    episode_text = csv_text(episode_rows)
    pilot_text = csv_text(pilot_diagnostics)
    formal_text = csv_text(formal_diagnostics)
    public_gate = {
        **gate_payload,
        "frozen_local_artifact_sha256": sha256_file(args.output_dir / "frozen_gate.json"),
    }
    public_gate_text = canonical_json(public_gate)
    peak_memory = int(torch.cuda.max_memory_allocated())
    total_seconds = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "stage": "WM Stage 26 score-blind hidden dynamics cohort",
        "status": "passed",
        "scope": {
            "tasks": list(TASK_IDS),
            "pilot_pairs": len(pilot_pairs),
            "formal_discovery_pairs": len(pairs),
            "collected_episodes": len(all_records),
            "selected_fault_conditions": list(selected_faults),
            "rejected_fault_conditions": public_gate["rejected_fault_conditions"],
            "diagnostic_prefix_steps": LIVE_STEPS,
            "world_model_loaded_during_collection": False,
            "world_model_scores_computed": 0,
            "action_interface_mismatch_steps_total": int(
                sum(row["action_interface_mismatch_steps"] for row in all_records)
            ),
        },
        "protocol": protocol,
        "pilot_gate": public_gate,
        "formal_discovery_summary": {
            condition: {
                "pairs": len([row for row in formal_diagnostics if row["condition"] == condition]),
                "pair_gates_passed": sum(
                    bool(row["pair_gate_passed"]) for row in formal_diagnostics if row["condition"] == condition
                ),
                "eef_divergence_m_median": float(
                    np.median(
                        [
                            row["first_h10_eef_divergence_m_max"]
                            for row in formal_diagnostics
                            if row["condition"] == condition
                        ]
                    )
                ),
                "pixel_mae_median": float(
                    np.median(
                        [
                            row["observation_60_dual_camera_pixel_mae"]
                            for row in formal_diagnostics
                            if row["condition"] == condition
                        ]
                    )
                ),
            }
            for condition in selected_faults
        },
        "negative_controls": {
            "all_initial_observations_bit_exact": all(
                row["initial_observations_bit_exact"] for row in formal_diagnostics
            ),
            "all_preintervention_observations_bit_exact": all(
                row["preintervention_observations_bit_exact"] for row in formal_diagnostics
            ),
            "all_preintervention_commands_bit_exact": all(
                row["preintervention_commands_bit_exact"] for row in formal_diagnostics
            ),
            "all_first_post_h10_commands_bit_exact": all(
                row["first_post_h10_commands_bit_exact"] for row in formal_diagnostics
            ),
            "commanded_equals_interface_action_every_step": (
                all(row["action_interface_mismatch_steps"] == 0 for row in all_records)
            ),
        },
        "runtime": {
            "policy_load_seconds": model_loaded - started,
            "total_seconds": total_seconds,
            "peak_torch_gpu_memory_bytes": peak_memory,
        },
        "artifacts": {
            "protocol": "protocol.json",
            "protocol_sha256": sha256_text(protocol_text),
            "frozen_gate": "frozen_gate.json",
            "frozen_gate_sha256": sha256_text(public_gate_text),
            "episodes": "episodes.csv",
            "episodes_sha256": sha256_text(episode_text),
            "pilot_diagnostics": "pilot_diagnostics.csv",
            "pilot_diagnostics_sha256": sha256_text(pilot_text),
            "paired_diagnostics": "paired_diagnostics.csv",
            "paired_diagnostics_sha256": sha256_text(formal_text),
            "raw_archives": ("local-only under outputs/wm/stage26_hidden_dynamics_cohort/episodes"),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "limitations": [
            "The discovery states were previously used in Stage 25 for a different synthetic fault family.",
            "MuJoCo parameter shifts remain synthetic and do not model a specific real robot failure rate.",
            "The cohort contains only 90-step diagnostic prefixes, not full success-rate evaluations.",
            "No world-model score is evaluated in this stage.",
        ],
        "stage27_readiness": {
            "observation_residual_discovery_can_start": True,
            "must_compare_against_state_and_pixel_heuristics": True,
            "shield_claim_justified": False,
        },
    }
    report_text = canonical_json(report)
    for filename, content in (
        ("protocol.json", protocol_text),
        ("frozen_gate.json", public_gate_text),
        ("episodes.csv", episode_text),
        ("pilot_diagnostics.csv", pilot_text),
        ("paired_diagnostics.csv", formal_text),
        ("report.json", report_text),
    ):
        write_output(
            args.public_results_dir / filename,
            content,
            overwrite=args.overwrite_public,
        )
    write_output(
        args.media_path,
        result_svg(gate, formal_diagnostics),
        overwrite=args.overwrite_public,
    )
    print(
        f"Stage 26 complete: selected={selected_faults}, episodes={len(all_records)}, WM scores=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
