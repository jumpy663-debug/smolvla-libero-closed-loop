#!/usr/bin/env python

"""Attach the frozen H10 world model as a non-intervening SmolVLA sidecar.

Stage 24 first replays every registered Stage 23 window through the causal
monitor API.  It then runs one paired nominal/fault LIBERO diagnostic prefix
with the same monitor inside the real SmolVLA control loop.  No threshold is
fit and the monitor never changes an action.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import time
from pathlib import Path
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
from lerobot.utils.io_utils import write_video
from online_monitor import (
    STAGE14,
    STAGE17,
    OnlineObservationEncoder,
    OnlineWorldModelMonitor,
    normalization_from_json,
)
from safetensors.torch import load_file
from stage22_collect_confirmatory_interventions import (
    INTERVENTION_START,
    STAGE3,
    STAGE5,
    STAGE18,
    apply_intervention,
)
from transformers import AutoImageProcessor, Dinov2Model

HORIZON = 10
REGISTERED_STARTS = tuple(range(10, 40)) + tuple(range(50, 80))
LIVE_CONDITIONS = ("nominal", "motion_attenuation_0p5")
DEFAULT_LIVE_TASK_ID = 4
DEFAULT_LIVE_INIT_STATE_INDEX = 7
DEFAULT_LIVE_STEPS = 90
SCORE_TOLERANCE = 2e-5
ENCODER_TOLERANCE = 2e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage22-episodes",
        type=Path,
        default=Path("results/wm/stage22_confirmatory_interventions/episodes.csv"),
    )
    parser.add_argument(
        "--stage23-windows",
        type=Path,
        default=Path("results/wm/stage23_confirmatory_wm_scoring/window_scores.csv"),
    )
    parser.add_argument(
        "--stage23-feature-dir",
        type=Path,
        default=Path("outputs/wm/stage23_confirmatory_wm_scoring/features"),
    )
    parser.add_argument(
        "--stage14-report",
        type=Path,
        default=Path("results/wm/stage14_next_latent_baseline/report.json"),
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path("results/wm/stage14_next_latent_baseline/normalization.json"),
    )
    parser.add_argument(
        "--wm-checkpoint-dir",
        type=Path,
        default=Path("outputs/wm/stage14_next_latent_baseline"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=STAGE17.STAGE12.default_model_root(),
    )
    parser.add_argument(
        "--policy-checkpoint-dir",
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
        default=Path("outputs/wm/stage24_online_sidecar"),
    )
    parser.add_argument(
        "--public-results-dir",
        type=Path,
        default=Path("results/wm/stage24_online_sidecar"),
    )
    parser.add_argument(
        "--media-path",
        type=Path,
        default=Path("media/wm_stage24_online_sidecar.svg"),
    )
    parser.add_argument("--live-task-id", type=int, default=DEFAULT_LIVE_TASK_ID)
    parser.add_argument(
        "--live-init-state-index",
        type=int,
        default=DEFAULT_LIVE_INIT_STATE_INDEX,
    )
    parser.add_argument("--live-steps", type=int, default=DEFAULT_LIVE_STEPS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--replay-only", action="store_true")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot serialize an empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_output(
    path: Path,
    content: str,
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-identical output: {path}")
    path.write_text(content, encoding="utf-8")


def distribution(values: list[float]) -> dict[str, int | float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "total": float(array.sum()),
    }


def make_monitor(
    models: dict[str, torch.nn.Module],
    normalization: Any,
    device: torch.device,
) -> OnlineWorldModelMonitor:
    return OnlineWorldModelMonitor(
        action_model=models["action_conditioned"],
        no_action_model=models["no_action"],
        normalization=normalization,
        device=device,
        horizon=HORIZON,
    )


def replay_validation(
    *,
    episode_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    feature_dir: Path,
    models: dict[str, torch.nn.Module],
    normalization: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = {(row["rollout_id"], int(row["start_index"])): row for row in reference_rows}
    if len(reference) != 1080:
        raise AssertionError("Stage 23 registered window set changed")
    comparisons = []
    scoring_seconds = []
    for episode in episode_rows:
        rollout_id = episode["rollout_id"]
        tensors = load_file(feature_dir / f"{rollout_id}.safetensors")
        monitor = make_monitor(models, normalization, device)
        monitor.reset(
            tensors["visual_tokens"][0].reshape(
                STAGE14.TOKENS_PER_FRAME,
                STAGE14.LATENT_DIMENSION,
            ),
            tensors["observation_state"][0],
        )
        maximum_target = max(REGISTERED_STARTS) + HORIZON
        for step in range(maximum_target):
            score = monitor.step(
                next_latent=tensors["visual_tokens"][step + 1].reshape(
                    STAGE14.TOKENS_PER_FRAME,
                    STAGE14.LATENT_DIMENSION,
                ),
                next_state=tensors["observation_state"][step + 1],
                commanded_action=tensors["commanded_action"][step],
                executed_action=tensors["executed_action"][step],
            )
            if score is None or score.start_index not in REGISTERED_STARTS:
                continue
            expected = reference[(rollout_id, score.start_index)]
            errors = {}
            for name in (
                "commanded_h10_raw_mse",
                "executed_h10_raw_mse",
                "no_action_h10_raw_mse",
                "persistence_h10_raw_mse",
            ):
                errors[name] = abs(float(getattr(score, name)) - float(expected[name]))
            comparisons.append(
                {
                    "rollout_id": rollout_id,
                    "condition": episode["condition"],
                    "start_index": score.start_index,
                    "target_index": score.target_index,
                    **{f"{name}_abs_error": value for name, value in errors.items()},
                }
            )
            scoring_seconds.append(score.scoring_seconds)
    if len(comparisons) != len(reference):
        raise AssertionError(f"Compared {len(comparisons)} windows, expected {len(reference)}")
    maximum_error = max(float(value) for row in comparisons for key, value in row.items() if key.endswith("_abs_error"))
    return comparisons, {
        "episodes": len(episode_rows),
        "registered_windows": len(comparisons),
        "maximum_absolute_score_difference": maximum_error,
        "tolerance": SCORE_TOLERANCE,
        "within_tolerance": maximum_error <= SCORE_TOLERANCE,
        "per_window_scoring_seconds": distribution(scoring_seconds),
    }


def archived_observation(
    arrays: dict[str, np.ndarray],
    index: int,
) -> dict[str, Any]:
    return {
        "pixels": {
            "camera1": arrays["camera1"][index : index + 1],
            "camera2": arrays["camera2"][index : index + 1],
        },
        "robot_state": {
            "eef": {
                "pos": arrays["eef_pos"][index : index + 1],
                "quat": arrays["eef_quat"][index : index + 1],
            },
            "gripper": {
                "qpos": arrays["gripper_qpos"][index : index + 1],
            },
        },
    }


def validate_online_encoder(
    *,
    episode: dict[str, str],
    feature_dir: Path,
    encoder: OnlineObservationEncoder,
) -> dict[str, Any]:
    source_path = Path(episode["local_artifact"])
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    cached = load_file(feature_dir / f"{episode['rollout_id']}.safetensors")
    indices = (0, 50, 89)
    maximum = 0.0
    state_maximum = 0.0
    seconds = []
    for index in indices:
        tokens, state, elapsed = encoder.encode(archived_observation(arrays, index))
        expected_tokens = cached["visual_tokens"][index].reshape(
            STAGE14.TOKENS_PER_FRAME,
            STAGE14.LATENT_DIMENSION,
        )
        maximum = max(maximum, float((tokens.float() - expected_tokens.float()).abs().max()))
        state_maximum = max(
            state_maximum,
            float((state - cached["observation_state"][index]).abs().max()),
        )
        seconds.append(elapsed)
    return {
        "rollout_id": episode["rollout_id"],
        "observation_indices": list(indices),
        "maximum_absolute_token_difference": maximum,
        "maximum_absolute_state_difference": state_maximum,
        "token_tolerance": ENCODER_TOLERANCE,
        "within_tolerance": maximum <= ENCODER_TOLERANCE and state_maximum == 0.0,
        "per_observation_encoding_seconds": distribution(seconds),
    }


def annotate_frame(
    frame: np.ndarray,
    *,
    condition: str,
    target_index: int,
    score: float | None,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.ascontiguousarray(frame))
    draw = ImageDraw.Draw(image)
    label = f"{condition} | obs={target_index:03d} | H10 gap=" + ("warming up" if score is None else f"{score:+.5f}")
    draw.rectangle((8, 8, min(image.width - 8, 510), 36), fill=(0, 0, 0))
    draw.text((16, 15), label, fill=(255, 255, 255))
    return np.asarray(image)


def collect_live_prefix(
    *,
    env: Any,
    policy: torch.nn.Module,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    observation_encoder: OnlineObservationEncoder,
    monitor: OnlineWorldModelMonitor,
    task_id: int,
    init_state_index: int,
    condition: str,
    live_steps: int,
    capture_frames: bool = True,
) -> dict[str, Any]:
    seed = STAGE18.episode_seed(task_id, init_state_index)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    policy.reset()
    env.set_attr("init_state_id", [init_state_index])
    observation, _ = env.reset(seed=[seed])
    initial_tokens, initial_state, initial_encode_seconds = observation_encoder.encode(observation)
    monitor.reset(initial_tokens, initial_state)
    commanded_actions = []
    executed_actions = []
    reward_values = []
    success_values = []
    timeline: list[dict[str, Any]] = []
    encode_seconds = [initial_encode_seconds]
    command_history: list[np.ndarray] = []
    latest_score: float | None = None
    frames = []
    if capture_frames:
        frames.append(
            annotate_frame(
                np.ascontiguousarray(env.envs[0].render()),
                condition=condition,
                target_index=0,
                score=None,
            )
        )

    for step in range(live_steps):
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
        executed_numpy, intervention_active = apply_intervention(
            commanded_numpy,
            command_history,
            step=step,
            condition=condition,
        )
        command_history.append(commanded_numpy.copy())
        observation, reward, terminated, truncated, info = env.step(executed_numpy)
        next_tokens, next_state, elapsed = observation_encoder.encode(observation)
        encode_seconds.append(elapsed)
        score = monitor.step(
            next_latent=next_tokens,
            next_state=next_state,
            commanded_action=commanded_numpy[0],
            executed_action=executed_numpy[0],
        )
        if score is not None:
            latest_score = score.commanded_minus_executed
            timeline.append(
                {
                    "condition": condition,
                    "task_id": task_id,
                    "init_state_index": init_state_index,
                    "intervention_start": (INTERVENTION_START if condition != "nominal" else ""),
                    "intervention_active_at_available_step": intervention_active,
                    **score.to_dict(),
                }
            )
        commanded_actions.append(commanded_numpy[0].copy())
        executed_actions.append(executed_numpy[0].copy())
        reward_values.append(float(np.asarray(reward)[0]))
        step_successes = STAGE18.STAGE16.extract_successes(info, 1)
        success_values.append(bool(step_successes[0]))
        if capture_frames:
            rendered_frame = np.ascontiguousarray(env.envs[0].render())
            frames.append(
                annotate_frame(
                    rendered_frame,
                    condition=condition,
                    target_index=step + 1,
                    score=latest_score,
                )
            )
        if bool((terminated | truncated)[0]):
            break
    return {
        "condition": condition,
        "task_id": task_id,
        "init_state_index": init_state_index,
        "seed": seed,
        "steps": len(commanded_actions),
        "success_within_prefix": any(success_values),
        "commanded_action": np.stack(commanded_actions).astype(np.float32),
        "executed_action": np.stack(executed_actions).astype(np.float32),
        "reward": np.asarray(reward_values, dtype=np.float32),
        "success": np.asarray(success_values, dtype=np.bool_),
        "timeline": timeline,
        "frames": frames,
        "encoding_seconds": encode_seconds,
    }


def compare_live_with_archive(
    live: dict[str, Any],
    archived_row: dict[str, str],
) -> dict[str, Any]:
    with np.load(archived_row["local_artifact"], allow_pickle=False) as archive:
        steps = int(live["steps"])
        commanded_match = np.array_equal(
            live["commanded_action"],
            archive["commanded_action"][:steps],
        )
        executed_match = np.array_equal(
            live["executed_action"],
            archive["executed_action"][:steps],
        )
        reward_match = np.array_equal(
            live["reward"],
            archive["reward"][:steps],
        )
    return {
        "reference_rollout_id": archived_row["rollout_id"],
        "prefix_steps": int(live["steps"]),
        "commanded_actions_bit_exact": commanded_match,
        "executed_actions_bit_exact": executed_match,
        "rewards_bit_exact": reward_match,
        "all_control_outputs_bit_exact": (commanded_match and executed_match and reward_match),
    }


def compare_live_scores(
    live: dict[str, Any],
    reference_rows: list[dict[str, str]],
) -> dict[str, Any]:
    lookup = {
        (row["condition"], int(row["start_index"])): row
        for row in reference_rows
        if int(row["benchmark_task_id"]) == int(live["task_id"])
        and int(row["init_state_index"]) == int(live["init_state_index"])
    }
    differences = []
    for row in live["timeline"]:
        start = int(row["start_index"])
        key = (live["condition"], start)
        if start in REGISTERED_STARTS and key in lookup:
            expected = lookup[key]
            differences.extend(
                [
                    abs(float(row["commanded_h10_raw_mse"]) - float(expected["commanded_h10_raw_mse"])),
                    abs(float(row["executed_h10_raw_mse"]) - float(expected["executed_h10_raw_mse"])),
                ]
            )
    if len(differences) != len(REGISTERED_STARTS) * 2:
        raise AssertionError("Live run did not cover every registered H10 window")
    maximum = max(differences)
    return {
        "registered_windows": len(differences) // 2,
        "maximum_absolute_score_difference": maximum,
        "tolerance": ENCODER_TOLERANCE,
        "within_tolerance": maximum <= ENCODER_TOLERANCE,
    }


def period_gap_mean(
    timeline: list[dict[str, Any]],
    starts: range,
) -> float:
    values = [float(row["commanded_minus_executed"]) for row in timeline if int(row["start_index"]) in starts]
    if len(values) != len(starts):
        raise AssertionError("Live timeline is missing a registered period")
    return float(np.mean(values))


def run_live_pair(
    *,
    args: argparse.Namespace,
    models: dict[str, torch.nn.Module],
    normalization: Any,
    observation_encoder: OnlineObservationEncoder,
    episode_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if args.live_steps < max(REGISTERED_STARTS) + HORIZON:
        raise ValueError("--live-steps must be at least 90")
    stage2_manifest = STAGE3.verify_stage2_manifest(args.stage2_manifest)
    config = PreTrainedConfig.from_pretrained(
        args.policy_checkpoint_dir,
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
        pretrained_path=str(args.policy_checkpoint_dir),
        preprocessor_overrides={
            "rename_observations_processor": {"rename_map": {}},
            "tokenizer_processor": {
                "tokenizer_name": str(args.backbone_dir.resolve()),
            },
            "device_processor": {"device": "cuda"},
        },
    )
    load_started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(
        args.policy_checkpoint_dir,
        config=config,
        local_files_only=True,
    ).eval()
    model_load_seconds = time.perf_counter() - load_started

    archived_lookup = {
        (
            int(row["benchmark_task_id"]),
            int(row["init_state_index"]),
            row["condition"],
        ): row
        for row in episode_rows
    }
    live_rows = []
    timeline_rows = []
    started = time.perf_counter()
    for condition in LIVE_CONDITIONS:
        env_cfg, env, envs = STAGE5.make_task_env(
            suite="libero_spatial",
            task_id=args.live_task_id,
            n_envs=1,
            resolution=STAGE18.RESOLUTION,
        )
        try:
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg,
                config,
            )
            live = collect_live_prefix(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                observation_encoder=observation_encoder,
                monitor=make_monitor(models, normalization, device),
                task_id=args.live_task_id,
                init_state_index=args.live_init_state_index,
                condition=condition,
                live_steps=args.live_steps,
            )
            key = (
                args.live_task_id,
                args.live_init_state_index,
                condition,
            )
            if key not in archived_lookup:
                raise KeyError(f"Missing Stage 22 paired reference: {key}")
            control_comparison = compare_live_with_archive(
                live,
                archived_lookup[key],
            )
            score_comparison = compare_live_scores(live, reference_rows)
            trajectory_path = args.output_dir / f"{condition}_trajectory.npz"
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                trajectory_path,
                commanded_action=live["commanded_action"],
                executed_action=live["executed_action"],
                reward=live["reward"],
                success=live["success"],
            )
            video_path = args.media_path.parent / f"wm_stage24_{condition}_monitor.mp4"
            write_video(
                video_path,
                live["frames"],
                fps=env.unwrapped.metadata["render_fps"],
            )
            pre_mean = period_gap_mean(live["timeline"], range(10, 40))
            post_mean = period_gap_mean(live["timeline"], range(50, 80))
            live_rows.append(
                {
                    "condition": condition,
                    "task_id": args.live_task_id,
                    "init_state_index": args.live_init_state_index,
                    "steps": live["steps"],
                    "success_within_prefix": live["success_within_prefix"],
                    "pre_h10_gap_mean": pre_mean,
                    "post_h10_gap_mean": post_mean,
                    "post_minus_pre_gap": post_mean - pre_mean,
                    "encoding_seconds": distribution(live["encoding_seconds"]),
                    "scoring_seconds": distribution([float(row["scoring_seconds"]) for row in live["timeline"]]),
                    "control_comparison": control_comparison,
                    "score_comparison": score_comparison,
                    "trajectory": trajectory_path.as_posix(),
                    "trajectory_sha256": sha256_file(trajectory_path),
                    "video": video_path.as_posix(),
                    "video_sha256": sha256_file(video_path),
                }
            )
            timeline_rows.extend(live["timeline"])
            print(
                f"live {condition}: steps={live['steps']} "
                f"post_gap={post_mean:+.6f} "
                f"control_exact={control_comparison['all_control_outputs_bit_exact']}",
                flush=True,
            )
        finally:
            close_envs(envs)
    return (
        live_rows,
        timeline_rows,
        {
            "policy_model_load_seconds": model_load_seconds,
            "paired_live_wall_seconds": time.perf_counter() - started,
            "policy_revision": stage2_manifest["resolved_revision"],
            "peak_torch_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
    )


def result_svg(live_rows: list[dict[str, Any]]) -> str:
    nominal = next(row for row in live_rows if row["condition"] == "nominal")
    fault = next(row for row in live_rows if row["condition"] == "motion_attenuation_0p5")
    scale = 8000.0
    nominal_width = max(2.0, abs(float(nominal["post_h10_gap_mean"])) * scale)
    fault_width = max(2.0, abs(float(fault["post_h10_gap_mean"])) * scale)

    def text(
        x: float,
        y: float,
        size: int,
        content: str,
        *,
        fill: str = "#162033",
        weight: int | None = None,
        family: str = "sans-serif",
    ) -> str:
        weight_attribute = "" if weight is None else f' font-weight="{weight}"'
        return (
            f'<text x="{x}" y="{y}" font-size="{size}" '
            f'font-family="{family}"{weight_attribute} fill="{fill}">'
            f"{content}</text>"
        )

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" viewBox="0 0 1200 560">',
        '<rect width="1200" height="560" fill="#f7f8fa"/>',
        text(70, 70, 32, "Stage 24：WM 在线旁路接入 SmolVLA 闭环", weight=700),
        text(
            70,
            108,
            19,
            "H10 分数随真实观测到达后在线产生；只记录，不修改动作，不拟合阈值",
            fill="#526070",
        ),
        '<rect x="70" y="150" width="1060" height="110" rx="14" fill="#ffffff" stroke="#dce1e8"/>',
        text(100, 190, 21, "历史轨迹因果回放", weight=700),
        text(
            100,
            227,
            19,
            f"18 episodes · 1,080 registered windows · online/offline max Δ ≤ {SCORE_TOLERANCE:g}",
            fill="#32445a",
        ),
        text(
            70,
            315,
            22,
            "实时闭环 post-window commanded − executed H10 error",
            weight=700,
        ),
        text(90, 370, 19, "正常执行", fill="#32445a"),
        f'<rect x="260" y="343" width="{nominal_width:.2f}" height="35" rx="5" fill="#4f86c6"/>',
        text(
            280 + nominal_width,
            368,
            18,
            f"{float(nominal['post_h10_gap_mean']):+.6f}",
            family="monospace",
        ),
        text(90, 435, 19, "0.5× 动作衰减", fill="#32445a"),
        f'<rect x="260" y="408" width="{fault_width:.2f}" height="35" rx="5" fill="#e36b4f"/>',
        text(
            280 + fault_width,
            433,
            18,
            f"{float(fault['post_h10_gap_mean']):+.6f}",
            family="monospace",
        ),
        text(
            70,
            510,
            17,
            "边界：当前是有 10 步观测延迟的 actuator-mismatch monitor，不是自然失败 detector 或 proactive shield。",
            fill="#6a7583",
        ),
        "</svg>",
    ]
    return "\n".join(elements) + "\n"


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.model_root.name != STAGE17.STAGE12.MODEL_REVISION:
        raise ValueError("DINOv2 model revision is not pinned")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    episode_rows = read_csv(args.stage22_episodes)
    reference_rows = read_csv(args.stage23_windows)
    if len(episode_rows) != 18 or len(reference_rows) != 1080:
        raise AssertionError("Stage 22/23 public cohort changed")
    stage14_report = json.loads(args.stage14_report.read_text(encoding="utf-8"))
    if sha256_file(args.normalization) != stage14_report["artifacts"]["normalization_sha256"]:
        raise AssertionError("Stage 14 normalization hash changed")
    normalization = normalization_from_json(args.normalization, device)
    models = STAGE17.load_dynamics(
        stage14_report,
        args.wm_checkpoint_dir,
        device,
    )
    processor = AutoImageProcessor.from_pretrained(
        args.model_root,
        local_files_only=True,
    )
    dino = Dinov2Model.from_pretrained(
        args.model_root,
        local_files_only=True,
    ).eval()
    dino.requires_grad_(False)
    dino.to(device)
    observation_encoder = OnlineObservationEncoder(
        processor,
        dino,
        device=device,
    )
    loaded_seconds = time.perf_counter() - started

    replay_rows, replay_report = replay_validation(
        episode_rows=episode_rows,
        reference_rows=reference_rows,
        feature_dir=args.stage23_feature_dir,
        models=models,
        normalization=normalization,
        device=device,
    )
    if not replay_report["within_tolerance"]:
        raise AssertionError(f"Online/offline score mismatch: {replay_report}")
    encoder_report = validate_online_encoder(
        episode=episode_rows[0],
        feature_dir=args.stage23_feature_dir,
        encoder=observation_encoder,
    )
    if not encoder_report["within_tolerance"]:
        raise AssertionError(f"Online encoder mismatch: {encoder_report}")
    print(
        "replay validation: "
        f"windows={replay_report['registered_windows']} "
        f"max_delta={replay_report['maximum_absolute_score_difference']:.3e}",
        flush=True,
    )

    if args.replay_only:
        print("Stage 24 replay-only validation complete", flush=True)
        return

    live_rows, timeline_rows, live_runtime = run_live_pair(
        args=args,
        models=models,
        normalization=normalization,
        observation_encoder=observation_encoder,
        episode_rows=episode_rows,
        reference_rows=reference_rows,
        device=device,
    )
    if not all(row["control_comparison"]["all_control_outputs_bit_exact"] for row in live_rows):
        raise AssertionError("Sidecar changed or failed to reproduce control outputs")
    if not all(row["score_comparison"]["within_tolerance"] for row in live_rows):
        raise AssertionError("Live sidecar scores differ from Stage 23 references")

    replay_text = csv_text(replay_rows)
    timeline_text = csv_text(timeline_rows)
    live_summary_text = canonical_json(live_rows)
    report = {
        "schema_version": 1,
        "stage": "WM Stage 24 online non-intervening SmolVLA sidecar",
        "status": "passed",
        "scope": {
            "monitor_mode": "online sidecar; logging only",
            "horizon": HORIZON,
            "observation_delay_control_steps": HORIZON,
            "actions_modified_by_monitor": 0,
            "thresholds_fitted": 0,
            "detector_claimed": False,
            "shield_claimed": False,
            "replay_episodes": len(episode_rows),
            "replay_registered_windows": len(replay_rows),
            "live_conditions": list(LIVE_CONDITIONS),
            "live_diagnostic_prefix_steps_per_condition": args.live_steps,
        },
        "frozen_components": {
            "encoder_model_id": STAGE17.STAGE12.MODEL_ID,
            "encoder_revision": STAGE17.STAGE12.MODEL_REVISION,
            "stage14_report_sha256": sha256_file(args.stage14_report),
            "normalization_sha256": sha256_file(args.normalization),
            "action_conditioned_checkpoint_sha256": stage14_report["variants"]["action_conditioned"][
                "checkpoint_sha256"
            ],
            "no_action_checkpoint_sha256": stage14_report["variants"]["no_action"]["checkpoint_sha256"],
        },
        "causal_semantics": {
            "input_at_transition": ("current oracle state plus commanded and executed action"),
            "target": "DINO latent of observation arriving H=10 transitions later",
            "score": "commanded H10 raw latent MSE minus executed H10 raw latent MSE",
            "score_available_after_step": "start_index + H - 1",
            "future_data_access_before_arrival": False,
        },
        "replay_equivalence": replay_report,
        "online_encoder_equivalence": encoder_report,
        "live_pair": live_rows,
        "conclusion": {
            "wm_runs_inside_smolvla_libero_loop": True,
            "sidecar_preserves_control_outputs_bit_exact": True,
            "live_scores_match_offline_reference_within_tolerance": True,
            "nominal_gap_is_zero": (float(live_rows[0]["post_h10_gap_mean"]) == 0.0),
            "attenuation_gap_is_positive": (float(live_rows[1]["post_h10_gap_mean"]) > 0.0),
            "ready_for_separate_detector_calibration": True,
            "online_shield_justified": False,
        },
        "runtime": {
            "frozen_wm_and_encoder_load_seconds": loaded_seconds,
            **live_runtime,
            "total_seconds": time.perf_counter() - started,
        },
        "artifacts": {
            "replay_comparison": "replay_comparison.csv",
            "replay_comparison_sha256": hashlib.sha256(replay_text.encode()).hexdigest(),
            "live_timeline": "live_timeline.csv",
            "live_timeline_sha256": hashlib.sha256(timeline_text.encode()).hexdigest(),
            "live_summary": "live_summary.json",
            "live_summary_sha256": hashlib.sha256(live_summary_text.encode()).hexdigest(),
            "videos": [
                {
                    "condition": row["condition"],
                    "local_path": row["video"],
                    "sha256": row["video_sha256"],
                }
                for row in live_rows
            ],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "limitations": [
            "The score is retrospective and becomes available after ten control steps.",
            "The live demonstration is one paired initial state, not a detector benchmark.",
            "No threshold, alarm rule, or recovery action is selected in this stage.",
            "The score targets commanded/executed actuator mismatch, not semantic VLA errors.",
            "Executed-action feedback and oracle robot state are assumed available.",
        ],
        "stage25_readiness": {
            "collect_separate_calibration_and_test_episodes": True,
            "fit_or_freeze_threshold_on_stage23_confirmation_data": False,
            "calibrate_detector_before_any_shield_claim": True,
        },
    }
    report_text = canonical_json(report)
    for filename, content in (
        ("replay_comparison.csv", replay_text),
        ("live_timeline.csv", timeline_text),
        ("live_summary.json", live_summary_text),
        ("report.json", report_text),
    ):
        write_output(
            args.public_results_dir / filename,
            content,
            overwrite=args.overwrite_public,
        )
    write_output(
        args.media_path,
        result_svg(live_rows),
        overwrite=args.overwrite_public,
    )
    print(
        "Stage 24 complete: online sidecar integrated, control outputs preserved, no threshold or shield claim",
        flush=True,
    )


if __name__ == "__main__":
    main()
