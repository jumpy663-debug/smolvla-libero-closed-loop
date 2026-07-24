from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "wm" / "stage19_freeze_balanced_cohort.py"
SPEC = importlib.util.spec_from_file_location("stage19_freeze_balanced_cohort", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
STAGE19 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE19
SPEC.loader.exec_module(STAGE19)


def candidate_rows(per_label: int = 4) -> list[dict]:
    rows = []
    for task_id in STAGE19.COHORT_TASK_IDS:
        for success in (False, True):
            for offset in range(per_label):
                init_state_index = 3 + offset + (10 if success else 0)
                rows.append(
                    {
                        "rollout_id": f"task{task_id}-init{init_state_index}",
                        "benchmark_task_id": task_id,
                        "task_name": f"task-{task_id}",
                        "init_state_index": init_state_index,
                        "episode_seed": STAGE19.STAGE18.episode_seed(task_id, init_state_index),
                        "success": success,
                        "initial_observation_sha256": f"obs-{task_id}-{init_state_index}",
                        "action_sha256": f"act-{task_id}-{init_state_index}",
                        "source_stage": 18,
                    }
                )
    return rows


class Stage19BalancedCohortTest(unittest.TestCase):
    def test_selection_is_order_independent(self) -> None:
        rows = candidate_rows()
        first, first_readiness = STAGE19.build_balanced_cohort(rows)
        second, second_readiness = STAGE19.build_balanced_cohort(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(first_readiness, second_readiness)

    def test_frozen_cohort_is_balanced_and_disjoint(self) -> None:
        selected, readiness = STAGE19.build_balanced_cohort(candidate_rows())
        self.assertTrue(readiness["ready"])
        self.assertEqual(len(selected), 18)
        self.assertEqual(sum(row["split"] == "calibration" for row in selected), 12)
        self.assertEqual(sum(row["split"] == "test" for row in selected), 6)
        self.assertEqual(len({row["rollout_id"] for row in selected}), 18)
        for task_id in STAGE19.COHORT_TASK_IDS:
            for success in (False, True):
                cell = [row for row in selected if row["benchmark_task_id"] == task_id and row["success"] == success]
                self.assertEqual(len(cell), 3)
                self.assertEqual(sum(row["split"] == "calibration" for row in cell), 2)
                self.assertEqual(sum(row["split"] == "test" for row in cell), 1)

    def test_insufficient_task8_failure_blocks_freeze(self) -> None:
        rows = [
            row
            for row in candidate_rows()
            if not (row["benchmark_task_id"] == 8 and not row["success"] and row["init_state_index"] > 4)
        ]
        selected, readiness = STAGE19.build_balanced_cohort(rows)
        self.assertEqual(selected, [])
        self.assertFalse(readiness["ready"])
        self.assertEqual(
            readiness["blocking_cells"],
            [
                {
                    "benchmark_task_id": 8,
                    "label": "failure",
                    "available": 2,
                    "required": 3,
                    "deficit": 1,
                }
            ],
        )

    def test_duplicate_candidate_is_rejected(self) -> None:
        rows = candidate_rows()
        with self.assertRaisesRegex(AssertionError, "duplicate"):
            STAGE19.build_balanced_cohort(rows + [dict(rows[0])])

    def test_selection_key_is_score_independent(self) -> None:
        row = candidate_rows()[0]
        first = STAGE19.selection_key(row)
        row["h10_raw_mse"] = 999.0
        self.assertEqual(first, STAGE19.selection_key(row))


if __name__ == "__main__":
    unittest.main()
