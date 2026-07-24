#!/usr/bin/env python

"""Reusable delayed online monitor for the frozen Stage 14 latent world model.

The monitor consumes observations in arrival order.  Once H transitions have
completed, it emits the score for the window that started H control steps ago.
It never changes the commanded or executed action.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch


def load_stage_module(name: str) -> ModuleType:
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE17 = load_stage_module("stage17_failure_signal")
STAGE13 = STAGE17.STAGE13
STAGE14 = STAGE17.STAGE14


@dataclass(frozen=True)
class OnlineScore:
    """One score emitted when the target observation becomes available."""

    start_index: int
    target_index: int
    available_after_step: int
    horizon: int
    commanded_h10_raw_mse: float
    executed_h10_raw_mse: float
    commanded_minus_executed: float
    no_action_h10_raw_mse: float
    persistence_h10_raw_mse: float
    scoring_seconds: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def normalization_from_json(path: Path, device: torch.device) -> Any:
    """Load the deployable, train-only normalization artifact."""

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("source") != "train frames only":
        raise AssertionError("Unexpected Stage 14 normalization metadata")
    expected = {
        "latent_mean": STAGE14.LATENT_DIMENSION,
        "latent_std": STAGE14.LATENT_DIMENSION,
        "state_mean": STAGE14.STATE_DIMENSION,
        "state_std": STAGE14.STATE_DIMENSION,
        "action_mean": STAGE14.ACTION_DIMENSION,
        "action_std": STAGE14.ACTION_DIMENSION,
    }
    tensors = {}
    for key, length in expected.items():
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != length:
            raise AssertionError(f"Unexpected normalization field: {key}")
        tensor = torch.tensor(values, dtype=torch.float32, device=device)
        if not bool(torch.isfinite(tensor).all()):
            raise AssertionError(f"Non-finite normalization field: {key}")
        if key.endswith("_std") and not bool((tensor > 0).all()):
            raise AssertionError(f"Non-positive normalization scale: {key}")
        tensors[key] = tensor
    return STAGE14.Normalization(**tensors)


def observation_to_model_inputs(
    observation: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one raw LIBERO vector observation to Stage 13/14 conventions."""

    pixels = observation["pixels"]
    robot_state = observation["robot_state"]
    cameras = []
    for key in ("camera1", "camera2"):
        value = np.asarray(pixels[key])
        if value.shape != (1, 360, 360, 3) or value.dtype != np.uint8:
            raise AssertionError(f"Unexpected live camera {key}: {value.shape} {value.dtype}")
        camera = torch.from_numpy(value.copy()).permute(0, 3, 1, 2).contiguous()
        cameras.append(torch.flip(camera, dims=(-2, -1)))
    images = torch.stack(cameras, dim=1)

    def state_array(*path: str) -> np.ndarray:
        value: Any = robot_state
        for key in path:
            value = value[key]
        result = np.asarray(value, dtype=np.float32)
        if result.shape[0] != 1 or not np.isfinite(result).all():
            raise AssertionError(f"Unexpected live state {'.'.join(path)}: {result.shape}")
        return result

    state_arrays = {
        "eef_pos": state_array("eef", "pos"),
        "eef_quat": state_array("eef", "quat"),
        "gripper_qpos": state_array("gripper", "qpos"),
    }
    state = STAGE17.policy_state(state_arrays)
    return images, state


class OnlineObservationEncoder:
    """Encode one newly arrived dual-camera observation with frozen DINOv2."""

    def __init__(
        self,
        processor: Any,
        encoder: torch.nn.Module,
        *,
        device: torch.device,
    ) -> None:
        self.processor = processor
        self.encoder = encoder
        self.device = device

    def encode(
        self,
        observation: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        images, state = observation_to_model_inputs(observation)
        started = time.perf_counter()
        tokens = STAGE13.encode_images(
            images,
            self.processor,
            self.encoder,
            device=self.device,
            batch_size=2,
        ).reshape(STAGE14.TOKENS_PER_FRAME, STAGE14.LATENT_DIMENSION)
        elapsed = time.perf_counter() - started
        return tokens, state[0], elapsed


class OnlineWorldModelMonitor:
    """Causal H-step scorer with a fixed H-step observation delay."""

    def __init__(
        self,
        *,
        action_model: torch.nn.Module,
        no_action_model: torch.nn.Module,
        normalization: Any,
        device: torch.device,
        horizon: int = 10,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.action_model = action_model
        self.no_action_model = no_action_model
        self.normalization = normalization
        self.device = device
        self.horizon = horizon
        self._latents: deque[torch.Tensor] = deque(maxlen=horizon + 1)
        self._states: deque[torch.Tensor] = deque(maxlen=horizon + 1)
        self._commanded: deque[torch.Tensor] = deque(maxlen=horizon)
        self._executed: deque[torch.Tensor] = deque(maxlen=horizon)
        self.transitions_seen = 0

    @staticmethod
    def _checked_vector(
        value: torch.Tensor | np.ndarray,
        *,
        shape: tuple[int, ...],
        name: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains non-finite values")
        return tensor

    def reset(
        self,
        initial_latent: torch.Tensor | np.ndarray,
        initial_state: torch.Tensor | np.ndarray,
    ) -> None:
        self._latents.clear()
        self._states.clear()
        self._commanded.clear()
        self._executed.clear()
        self.transitions_seen = 0
        self._latents.append(
            self._checked_vector(
                initial_latent,
                shape=(STAGE14.TOKENS_PER_FRAME, STAGE14.LATENT_DIMENSION),
                name="initial_latent",
            )
        )
        self._states.append(
            self._checked_vector(
                initial_state,
                shape=(STAGE14.STATE_DIMENSION,),
                name="initial_state",
            )
        )

    def step(
        self,
        *,
        next_latent: torch.Tensor | np.ndarray,
        next_state: torch.Tensor | np.ndarray,
        commanded_action: torch.Tensor | np.ndarray,
        executed_action: torch.Tensor | np.ndarray,
    ) -> OnlineScore | None:
        if not self._latents:
            raise RuntimeError("reset must be called before step")
        self._commanded.append(
            self._checked_vector(
                commanded_action,
                shape=(STAGE14.ACTION_DIMENSION,),
                name="commanded_action",
            )
        )
        self._executed.append(
            self._checked_vector(
                executed_action,
                shape=(STAGE14.ACTION_DIMENSION,),
                name="executed_action",
            )
        )
        self._latents.append(
            self._checked_vector(
                next_latent,
                shape=(STAGE14.TOKENS_PER_FRAME, STAGE14.LATENT_DIMENSION),
                name="next_latent",
            )
        )
        self._states.append(
            self._checked_vector(
                next_state,
                shape=(STAGE14.STATE_DIMENSION,),
                name="next_state",
            )
        )
        self.transitions_seen += 1
        if self.transitions_seen < self.horizon:
            return None
        return self._score_closed_window()

    def _score_closed_window(self) -> OnlineScore:
        if not (
            len(self._latents) == self.horizon + 1
            and len(self._states) == self.horizon + 1
            and len(self._commanded) == self.horizon
            and len(self._executed) == self.horizon
        ):
            raise AssertionError("Online monitor buffers are inconsistent")
        started = time.perf_counter()
        initial_raw = self._latents[0].to(self.device)[None, :]
        initial = (initial_raw - self.normalization.latent_mean) / self.normalization.latent_std
        commanded_prediction = initial.clone()
        executed_prediction = initial.clone()
        no_action_prediction = initial.clone()
        with torch.inference_mode():
            for offset in range(self.horizon):
                state = self._states[offset].to(self.device)[None, :]
                state = (state - self.normalization.state_mean) / self.normalization.state_std
                commanded = self._commanded[offset].to(self.device)[None, :]
                commanded = (commanded - self.normalization.action_mean) / self.normalization.action_std
                executed = self._executed[offset].to(self.device)[None, :]
                executed = (executed - self.normalization.action_mean) / self.normalization.action_std
                commanded_prediction = self.action_model(
                    commanded_prediction,
                    state,
                    commanded,
                )
                executed_prediction = self.action_model(
                    executed_prediction,
                    state,
                    executed,
                )
                no_action_prediction = self.no_action_model(
                    no_action_prediction,
                    state,
                    torch.zeros_like(commanded),
                )
            target_raw = self._latents[-1].to(self.device)[None, :]
            target = (target_raw - self.normalization.latent_mean) / self.normalization.latent_std

            def raw_mse(prediction: torch.Tensor) -> float:
                prediction_raw = STAGE14.raw_latent(prediction, self.normalization)
                reconstructed_target = STAGE14.raw_latent(target, self.normalization)
                return float((prediction_raw - reconstructed_target).square().mean().item())

            commanded_error = raw_mse(commanded_prediction)
            executed_error = raw_mse(executed_prediction)
            no_action_error = raw_mse(no_action_prediction)
            persistence_error = raw_mse(initial)
        elapsed = time.perf_counter() - started
        target_index = self.transitions_seen
        return OnlineScore(
            start_index=target_index - self.horizon,
            target_index=target_index,
            available_after_step=target_index - 1,
            horizon=self.horizon,
            commanded_h10_raw_mse=commanded_error,
            executed_h10_raw_mse=executed_error,
            commanded_minus_executed=commanded_error - executed_error,
            no_action_h10_raw_mse=no_action_error,
            persistence_h10_raw_mse=persistence_error,
            scoring_seconds=elapsed,
        )
