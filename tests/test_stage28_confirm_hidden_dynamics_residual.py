from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def load_stage28():
    path = Path(__file__).parents[1] / "scripts" / "wm" / "stage28_confirm_hidden_dynamics_residual.py"
    scripts_path = str(path.parent)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location(
        "stage28_confirm_hidden_dynamics_residual_for_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE28 = load_stage28()
ROOT = Path(__file__).parents[1]


def test_confirmation_states_are_pinned_and_disjoint_from_discovery() -> None:
    protocol = {
        "pairs": [
            {
                "benchmark_task_id": task_id,
                "init_state_index": init_state,
                "split": "test",
            }
            for task_id, init_states in STAGE28.EXPECTED_CONFIRMATION.items()
            for init_state in init_states
        ]
    }
    selected = STAGE28.confirmation_pairs(protocol)
    assert len(selected) == 6
    assert {(row["benchmark_task_id"], row["init_state_index"]) for row in selected}.isdisjoint(
        {
            (4, 3),
            (4, 13),
            (7, 10),
            (7, 13),
            (8, 16),
            (8, 5),
        }
    )


def test_confirmation_gate_is_fixed_to_pair_level_evidence() -> None:
    assert STAGE28.confirmation_passed(
        {
            "mean": 0.5,
            "positive_pairs": 5,
            "exact_sign_flip_p": 0.03125,
        }
    )
    assert not STAGE28.confirmation_passed(
        {
            "mean": 0.5,
            "positive_pairs": 4,
            "exact_sign_flip_p": 0.03125,
        }
    )
    assert not STAGE28.confirmation_passed(
        {
            "mean": 0.5,
            "positive_pairs": 6,
            "exact_sign_flip_p": 0.078125,
        }
    )


def test_public_stage28_result_rejects_unreplicated_candidate() -> None:
    result_dir = ROOT / "results" / "wm" / "stage28_confirm_hidden_dynamics_residual"
    report = json.loads((result_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["scope"]["pairs"] == 6
    assert report["scope"]["episodes"] == 12
    assert report["candidate_freeze"]["candidate_frozen_before_new_rollout_collection"]
    assert not report["candidate_freeze"]["confirmation_scores_viewed_at_freeze"]
    assert report["pairing"]["all_pair_contracts_passed"]
    assert report["pairing"]["action_interface_mismatch_steps_total"] == 0
    assert report["primary_result"]["positive_pairs"] == 4
    assert report["primary_result"]["mean"] < 0
    assert report["primary_result"]["exact_sign_flip_p"] == 0.640625
    assert not report["conclusion"]["independent_confirmation_passed"]
    assert not report["stage29_readiness"]["detector_calibration_and_held_out_event_test_justified"]
    assert report["descriptive_comparators"]["latent_persistence_motion"]["positive_pairs"] == 6

    with (result_dir / "pair_contracts.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert all(row["pair_contract_passed"] == "True" for row in rows)

    for key in (
        "frozen_candidate",
        "protocol",
        "episodes",
        "pair_contracts",
        "feature_manifest",
        "calibration",
        "window_scores",
        "pair_effects",
        "statistics",
    ):
        path = result_dir / report["artifacts"][key]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == report["artifacts"][f"{key}_sha256"]
