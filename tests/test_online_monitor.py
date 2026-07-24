from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def load_module(filename: str, name: str):
    path = Path(__file__).parents[1] / "scripts" / "wm" / filename
    scripts_path = str(path.parent)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ONLINE = load_module("online_monitor.py", "online_monitor_for_test")
STAGE23 = load_module(
    "stage23_confirmatory_wm_scoring.py",
    "stage23_for_online_monitor_test",
)


def normalization():
    stage14 = ONLINE.STAGE14
    return stage14.Normalization(
        latent_mean=torch.zeros(stage14.LATENT_DIMENSION),
        latent_std=torch.ones(stage14.LATENT_DIMENSION),
        state_mean=torch.zeros(stage14.STATE_DIMENSION),
        state_std=torch.ones(stage14.STATE_DIMENSION),
        action_mean=torch.zeros(stage14.ACTION_DIMENSION),
        action_std=torch.ones(stage14.ACTION_DIMENSION),
    )


def test_monitor_emits_only_after_horizon_and_keeps_causal_indices() -> None:
    torch.manual_seed(24)
    stage14 = ONLINE.STAGE14
    action_model = stage14.TokenDynamicsModel(hidden_dimension=8, depth=1).eval()
    no_action_model = stage14.TokenDynamicsModel(hidden_dimension=8, depth=1).eval()
    monitor = ONLINE.OnlineWorldModelMonitor(
        action_model=action_model,
        no_action_model=no_action_model,
        normalization=normalization(),
        device=torch.device("cpu"),
        horizon=3,
    )
    latent = torch.randn(stage14.TOKENS_PER_FRAME, stage14.LATENT_DIMENSION)
    state = torch.zeros(stage14.STATE_DIMENSION)
    action = torch.zeros(stage14.ACTION_DIMENSION)
    monitor.reset(latent, state)
    assert (
        monitor.step(
            next_latent=latent,
            next_state=state,
            commanded_action=action,
            executed_action=action,
        )
        is None
    )
    assert (
        monitor.step(
            next_latent=latent,
            next_state=state,
            commanded_action=action,
            executed_action=action,
        )
        is None
    )
    score = monitor.step(
        next_latent=latent,
        next_state=state,
        commanded_action=action,
        executed_action=action,
    )
    assert score is not None
    assert (score.start_index, score.target_index, score.available_after_step) == (0, 3, 2)
    assert score.commanded_minus_executed == 0.0


def test_online_scores_match_offline_window_semantics() -> None:
    torch.manual_seed(2400)
    stage14 = ONLINE.STAGE14
    horizon = 3
    steps = 7
    action_model = stage14.TokenDynamicsModel(hidden_dimension=8, depth=1).eval()
    no_action_model = stage14.TokenDynamicsModel(hidden_dimension=8, depth=1).eval()
    models = {
        "action_conditioned": action_model,
        "no_action": no_action_model,
    }
    norm = normalization()
    tensors = {
        "visual_tokens": torch.randn(
            steps + 1,
            2,
            16,
            stage14.LATENT_DIMENSION,
        ).half(),
        "observation_state": torch.randn(steps + 1, stage14.STATE_DIMENSION),
        "commanded_action": torch.randn(steps, stage14.ACTION_DIMENSION),
        "executed_action": torch.randn(steps, stage14.ACTION_DIMENSION),
        "transition_valid": torch.ones(steps, dtype=torch.bool),
    }
    old_horizon = STAGE23.HORIZON
    STAGE23.HORIZON = horizon
    try:
        offline = STAGE23.score_episode_windows(
            tensors,
            models,
            norm,
            starts=tuple(range(steps - horizon + 1)),
            device=torch.device("cpu"),
            batch_size=steps,
        )
    finally:
        STAGE23.HORIZON = old_horizon

    monitor = ONLINE.OnlineWorldModelMonitor(
        action_model=action_model,
        no_action_model=no_action_model,
        normalization=norm,
        device=torch.device("cpu"),
        horizon=horizon,
    )
    monitor.reset(
        tensors["visual_tokens"][0].reshape(32, 384),
        tensors["observation_state"][0],
    )
    online = []
    for step in range(steps):
        score = monitor.step(
            next_latent=tensors["visual_tokens"][step + 1].reshape(32, 384),
            next_state=tensors["observation_state"][step + 1],
            commanded_action=tensors["commanded_action"][step],
            executed_action=tensors["executed_action"][step],
        )
        if score is not None:
            online.append(score)
    assert len(online) == steps - horizon + 1
    for index, score in enumerate(online):
        assert score.start_index == index
        assert score.commanded_h10_raw_mse == pytest.approx(
            float(offline["commanded_h10_raw_mse"][index]),
            abs=1e-6,
        )
        assert score.executed_h10_raw_mse == pytest.approx(
            float(offline["executed_h10_raw_mse"][index]),
            abs=1e-6,
        )
        assert score.no_action_h10_raw_mse == pytest.approx(
            float(offline["no_action_h10_raw_mse"][index]),
            abs=1e-6,
        )
