from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "wm" / "stage17_failure_signal.py"
SPEC = importlib.util.spec_from_file_location("stage17_failure_signal", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
STAGE17 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE17
SPEC.loader.exec_module(STAGE17)


class Stage17FailureSignalTest(unittest.TestCase):
    def test_identity_quaternion_maps_to_zero_axisangle(self) -> None:
        quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        result = STAGE17.quat_to_axisangle(quaternion)
        self.assertTrue(torch.equal(result, torch.zeros(1, 3)))

    def test_fixed_prefix_rejects_invalid_transition(self) -> None:
        valid = torch.ones(90, dtype=torch.bool)
        valid[88] = False
        with self.assertRaisesRegex(AssertionError, "Invalid transition"):
            STAGE17.fixed_prefix_starts(valid, starts=80, horizon=10)

    def test_roc_auc_preserves_prespecified_direction(self) -> None:
        failures = np.array([False, False, True, True])
        self.assertEqual(STAGE17.roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), failures), 1.0)
        self.assertEqual(STAGE17.roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), failures), 0.0)

    def test_exact_permutation_uses_all_label_assignments(self) -> None:
        failures = np.array([False, False, True, True])
        result = STAGE17.exact_permutation(
            np.array([0.1, 0.2, 0.8, 0.9]),
            failures,
        )
        self.assertEqual(result["label_assignments"], 6)
        self.assertGreater(result["failure_minus_success_mean"], 0)

    def test_task_centering_removes_task_means(self) -> None:
        scores = np.array([1.0, 3.0, 10.0, 14.0])
        tasks = np.array([4, 4, 8, 8])
        centered = STAGE17.task_centered(scores, tasks)
        self.assertAlmostEqual(float(centered[tasks == 4].mean()), 0.0)
        self.assertAlmostEqual(float(centered[tasks == 8].mean()), 0.0)

    def test_holm_adjustment_is_monotonic_in_rank_order(self) -> None:
        adjusted = STAGE17.holm_adjust({"a": 0.01, "b": 0.03, "c": 0.8})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertEqual(adjusted["c"], 0.8)


if __name__ == "__main__":
    unittest.main()
