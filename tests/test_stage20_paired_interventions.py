from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "wm" / "stage20_collect_paired_interventions.py"
SPEC = importlib.util.spec_from_file_location("stage20_collect_paired_interventions", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
STAGE20 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE20
SPEC.loader.exec_module(STAGE20)


def scout_rows() -> list[dict]:
    rows = []
    for task_id in STAGE20.TASK_IDS:
        for offset in range(4):
            init_state_index = 3 + offset
            rows.append(
                {
                    "benchmark_task_id": task_id,
                    "init_state_index": init_state_index,
                    "episode_seed": STAGE20.STAGE18.episode_seed(task_id, init_state_index),
                    "task_name": f"task-{task_id}",
                    "success": True,
                    "steps": 100 + offset,
                    "initial_observation_sha256": f"obs-{task_id}-{init_state_index}",
                    "action_sha256": f"action-{task_id}-{init_state_index}",
                    "source_stage": 18,
                }
            )
    return rows


def valid_arrays(condition: str, steps: int = 60, resolution: int = 4) -> dict[str, np.ndarray]:
    observations = steps + 1
    commanded = np.ones((steps, 7), dtype=np.float32)
    executed = commanded.copy()
    mask = np.zeros(steps, dtype=np.bool_)
    if condition == "persistent_motion_dropout":
        mask[STAGE20.INTERVENTION_START :] = True
        executed[STAGE20.INTERVENTION_START :, :6] = 0
    return {
        "camera1": np.zeros((observations, resolution, resolution, 3), dtype=np.uint8),
        "camera2": np.ones((observations, resolution, resolution, 3), dtype=np.uint8),
        "eef_pos": np.zeros((observations, 3), dtype=np.float32),
        "eef_quat": np.zeros((observations, 4), dtype=np.float32),
        "eef_mat": np.zeros((observations, 3, 3), dtype=np.float32),
        "gripper_qpos": np.zeros((observations, 2), dtype=np.float32),
        "gripper_qvel": np.zeros((observations, 2), dtype=np.float32),
        "joint_pos": np.zeros((observations, 7), dtype=np.float32),
        "joint_vel": np.zeros((observations, 7), dtype=np.float32),
        "commanded_action": commanded,
        "executed_action": executed,
        "reward": np.zeros(steps, dtype=np.float32),
        "success": np.zeros(steps, dtype=np.bool_),
        "done": np.array([False] * (steps - 1) + [True], dtype=np.bool_),
        "transition_valid": np.ones(steps, dtype=np.bool_),
        "intervention_mask": mask,
    }


class Stage20PairedInterventionsTest(unittest.TestCase):
    def test_screen_selection_is_deterministic_and_balanced(self) -> None:
        rows = scout_rows()
        first = STAGE20.select_screen_candidates(rows)
        second = STAGE20.select_screen_candidates(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        for task_id in STAGE20.TASK_IDS:
            self.assertEqual(
                sum(row["benchmark_task_id"] == task_id for row in first),
                4,
            )

    def test_selection_ignores_world_model_scores(self) -> None:
        rows = scout_rows()
        first = STAGE20.select_screen_candidates(rows)
        for row in rows:
            row["h10_raw_mse"] = float(row["init_state_index"]) * 100
        self.assertEqual(first, STAGE20.select_screen_candidates(rows))

    def test_pair_freeze_uses_fresh_reset_successes(self) -> None:
        candidates = STAGE20.select_screen_candidates(scout_rows())
        screen = []
        for candidate in candidates:
            success = candidate["screen_rank_within_task"] in (1, 2)
            screen.append(
                {
                    "benchmark_task_id": candidate["benchmark_task_id"],
                    "init_state_index": candidate["init_state_index"],
                    "success": success,
                    "steps": 100,
                    "initial_observation_sha256": f"fresh-{candidate['candidate_id']}",
                    "action_sha256": f"action-{candidate['candidate_id']}",
                }
            )
        selected = STAGE20.freeze_pairs(candidates, screen)
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(row["screen_rank_within_task"] in (1, 2) for row in selected))

    def test_pair_freeze_rejects_insufficient_stable_successes(self) -> None:
        candidates = STAGE20.select_screen_candidates(scout_rows())
        screen = [
            {
                "benchmark_task_id": candidate["benchmark_task_id"],
                "init_state_index": candidate["init_state_index"],
                "success": candidate["screen_rank_within_task"] == 0,
                "steps": 100,
                "initial_observation_sha256": f"fresh-{candidate['candidate_id']}",
                "action_sha256": f"action-{candidate['candidate_id']}",
            }
            for candidate in candidates
        ]
        with self.assertRaisesRegex(AssertionError, "stable fresh-reset successes"):
            STAGE20.freeze_pairs(candidates, screen)

    def test_nominal_validation_rejects_modified_action(self) -> None:
        arrays = valid_arrays("nominal")
        arrays["executed_action"][0, 0] = 0
        with self.assertRaisesRegex(AssertionError, "Nominal"):
            STAGE20.validate_episode_arrays(
                arrays,
                condition="nominal",
                resolution=4,
            )

    def test_fault_validation_accepts_pinned_schedule(self) -> None:
        arrays = valid_arrays("persistent_motion_dropout")
        STAGE20.validate_episode_arrays(
            arrays,
            condition="persistent_motion_dropout",
            resolution=4,
        )

    def test_fault_validation_rejects_gripper_change(self) -> None:
        arrays = valid_arrays("persistent_motion_dropout")
        arrays["executed_action"][STAGE20.INTERVENTION_START, 6] = 0
        with self.assertRaisesRegex(AssertionError, "gripper"):
            STAGE20.validate_episode_arrays(
                arrays,
                condition="persistent_motion_dropout",
                resolution=4,
            )

    def test_observation_prefix_hash_changes_with_pixels(self) -> None:
        arrays = valid_arrays("nominal")
        first = STAGE20.observation_prefix_sha256(arrays, observations=10)
        arrays["camera1"][5, 0, 0, 0] = 1
        second = STAGE20.observation_prefix_sha256(arrays, observations=10)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
