from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "wm" / "stage18_initial_state_scout.py"
SPEC = importlib.util.spec_from_file_location("stage18_initial_state_scout", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
STAGE18 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE18
SPEC.loader.exec_module(STAGE18)


def balanced_rows(per_label: int = 4) -> list[dict]:
    rows = []
    for task_id in STAGE18.TASK_IDS:
        for success in (False, True):
            for offset in range(per_label):
                init_state_index = 3 + offset + (10 if success else 0)
                rows.append(
                    {
                        "rollout_id": f"task{task_id}-init{init_state_index}",
                        "benchmark_task_id": task_id,
                        "task_name": f"task-{task_id}",
                        "init_state_index": init_state_index,
                        "success": success,
                        "episode_seed": STAGE18.episode_seed(task_id, init_state_index),
                        "action_sha256": f"{task_id:02d}{init_state_index:02d}".ljust(64, "0"),
                    }
                )
    return rows


class Stage18InitialStateScoutTest(unittest.TestCase):
    def test_episode_seed_is_stable_and_unique(self) -> None:
        seeds = {
            STAGE18.episode_seed(task_id, init_state_index)
            for task_id in STAGE18.TASK_IDS
            for init_state_index in range(3, 15)
        }
        self.assertEqual(len(seeds), 48)
        self.assertEqual(STAGE18.episode_seed(4, 3), 180403)

    def test_action_hash_rejects_wrong_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            STAGE18.action_trace_sha256([np.zeros(6, dtype=np.float32)])

    def test_balanced_selection_is_deterministic(self) -> None:
        rows = balanced_rows()
        first, first_readiness = STAGE18.build_balanced_selection(rows)
        second, second_readiness = STAGE18.build_balanced_selection(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(first_readiness, second_readiness)
        self.assertTrue(first_readiness["ready"])

    def test_balanced_selection_has_episode_disjoint_splits(self) -> None:
        selected, readiness = STAGE18.build_balanced_selection(balanced_rows())
        self.assertTrue(readiness["ready"])
        self.assertEqual(len(selected), 24)
        calibration = {row["rollout_id"] for row in selected if row["split"] == "calibration"}
        test = {row["rollout_id"] for row in selected if row["split"] == "test"}
        self.assertEqual(len(calibration), 16)
        self.assertEqual(len(test), 8)
        self.assertTrue(calibration.isdisjoint(test))
        for task_id in STAGE18.TASK_IDS:
            for success in (False, True):
                cell = [row for row in selected if row["benchmark_task_id"] == task_id and row["success"] == success]
                self.assertEqual(sum(row["split"] == "calibration" for row in cell), 2)
                self.assertEqual(sum(row["split"] == "test" for row in cell), 1)

    def test_insufficient_cell_blocks_selection(self) -> None:
        rows = [row for row in balanced_rows() if not (row["benchmark_task_id"] == 5 and row["success"])]
        selected, readiness = STAGE18.build_balanced_selection(rows)
        self.assertEqual(selected, [])
        self.assertFalse(readiness["ready"])
        self.assertFalse(readiness["per_task"]["5"]["success"]["ready"])
        self.assertNotIn(5, readiness["ready_tasks"])
        self.assertEqual(
            [cell["benchmark_task_id"] for cell in readiness["blocking_cells"]],
            [5],
        )


if __name__ == "__main__":
    unittest.main()
