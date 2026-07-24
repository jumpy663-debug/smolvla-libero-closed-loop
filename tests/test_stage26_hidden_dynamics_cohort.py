from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_stage26():
    path = Path(__file__).parents[1] / "scripts" / "wm" / "stage26_hidden_dynamics_cohort.py"
    scripts_path = str(path.parent)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location(
        "stage26_hidden_dynamics_cohort_for_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE26 = load_stage26()
ROOT = Path(__file__).parents[1]


class FakeModel:
    def __init__(self) -> None:
        self.actuator_gainprm = np.zeros((9, 10), dtype=np.float64)
        self.actuator_gainprm[:, 0] = [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1000,
            1000,
        ]
        self.dof_damping = np.array(
            [0.1] * 5 + [0.01] * 2 + [100, 100],
            dtype=np.float64,
        )
        self.jnt_dofadr = np.arange(9)

    @staticmethod
    def actuator_name2id(name: str) -> int:
        return int(name.rsplit("j", 1)[1]) - 1

    @staticmethod
    def joint_name2id(name: str) -> int:
        return int(name.rsplit("joint", 1)[1]) - 1


class FakeSim:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.forward_calls = 0

    def forward(self) -> None:
        self.forward_calls += 1


def fake_vector_env() -> tuple[SimpleNamespace, FakeSim]:
    sim = FakeSim()
    vector = SimpleNamespace(envs=[SimpleNamespace(_env=SimpleNamespace(env=SimpleNamespace(sim=sim)))])
    return vector, sim


def test_actuator_gain_shift_changes_only_arm_gain() -> None:
    env, sim = fake_vector_env()
    before = STAGE26.parameter_snapshot(sim)
    result = STAGE26.apply_hidden_dynamics_shift(
        env,
        "arm_actuator_gain_0p5",
    )
    after = STAGE26.parameter_snapshot(sim)
    assert result["verified"]
    assert np.array_equal(
        after["arm_actuator_gain"],
        np.asarray(before["arm_actuator_gain"]) * 0.5,
    )
    assert after["arm_joint_damping"] == before["arm_joint_damping"]
    assert sim.forward_calls == 1


def test_joint_damping_shift_changes_only_arm_damping() -> None:
    env, sim = fake_vector_env()
    before = STAGE26.parameter_snapshot(sim)
    result = STAGE26.apply_hidden_dynamics_shift(
        env,
        "arm_joint_damping_10x",
    )
    after = STAGE26.parameter_snapshot(sim)
    assert result["verified"]
    assert np.array_equal(
        after["arm_joint_damping"],
        np.asarray(before["arm_joint_damping"]) * 10,
    )
    assert after["arm_actuator_gain"] == before["arm_actuator_gain"]


def test_pilot_gate_requires_two_of_three_complete_pair_effects() -> None:
    diagnostics = []
    for condition in STAGE26.FAULT_CONDITIONS:
        for index in range(3):
            diagnostics.append(
                {
                    "condition": condition,
                    "pilot_pair": True,
                    "pair_gate_passed": (index < 2 if condition == "arm_actuator_gain_0p5" else index < 1),
                    "first_h10_eef_divergence_m_max": 0.01,
                    "observation_60_dual_camera_pixel_mae": 1.0,
                }
            )
    gate = STAGE26.gate_conditions(diagnostics)
    assert gate["arm_actuator_gain_0p5"]["advanced_to_discovery"]
    assert not gate["arm_joint_damping_10x"]["advanced_to_discovery"]


def test_discovery_pairs_preserve_stage25_pilot_order() -> None:
    pairs = []
    for task_id, init_states in STAGE26.EXPECTED_DISCOVERY.items():
        for init_state in init_states:
            pairs.append(
                {
                    "benchmark_task_id": task_id,
                    "init_state_index": init_state,
                    "split": "calibration",
                }
            )
    selected = STAGE26.discovery_pairs({"pairs": pairs})
    assert [row["init_state_index"] for row in selected if row["pilot_pair"]] == [3, 10, 16]


def test_public_hidden_dynamics_cohort_passes_contract() -> None:
    result_dir = ROOT / "results" / "wm" / "stage26_hidden_dynamics_cohort"
    report = json.loads((result_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["scope"]["selected_fault_conditions"] == ["arm_actuator_gain_0p5"]
    assert report["scope"]["rejected_fault_conditions"] == ["arm_joint_damping_10x"]
    assert report["scope"]["collected_episodes"] == 15
    assert report["scope"]["action_interface_mismatch_steps_total"] == 0
    assert report["scope"]["world_model_scores_computed"] == 0
    assert all(report["negative_controls"].values())
    summary = report["formal_discovery_summary"]["arm_actuator_gain_0p5"]
    assert summary["pairs"] == 6
    assert summary["pair_gates_passed"] == 6

    with (result_dir / "paired_diagnostics.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {row["condition"] for row in rows} == {"arm_actuator_gain_0p5"}
    assert all(row["pair_gate_passed"] == "True" for row in rows)
    assert all(row["first_post_h10_commands_bit_exact"] == "True" for row in rows)

    for key in (
        "protocol",
        "frozen_gate",
        "episodes",
        "pilot_diagnostics",
        "paired_diagnostics",
    ):
        path = result_dir / report["artifacts"][key]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == report["artifacts"][f"{key}_sha256"]
