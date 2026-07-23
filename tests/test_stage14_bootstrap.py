from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "wm" / "stage14_paired_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("stage14_paired_bootstrap", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOOTSTRAP
SPEC.loader.exec_module(BOOTSTRAP)


class Stage14BootstrapTest(unittest.TestCase):
    def test_positive_paired_effect_has_positive_interval(self) -> None:
        conditioned = np.array([8.0, 18.0, 28.0, 38.0])
        no_action = np.array([10.0, 20.0, 30.0, 40.0])
        counts = np.array([10, 10, 10, 10])

        result = BOOTSTRAP.cluster_bootstrap(
            conditioned,
            no_action,
            counts,
            samples=1000,
            seed=1423,
        )

        self.assertEqual(result["conditioned_better_episodes"], 4)
        self.assertGreater(result["improvement_percent_ci95"][0], 0)
        self.assertEqual(result["bootstrap_probability_positive"], 1.0)

    def test_bootstrap_is_deterministic(self) -> None:
        conditioned = np.array([9.0, 19.0, 29.0])
        no_action = np.array([10.0, 20.0, 30.0])
        counts = np.array([10, 20, 30])

        first = BOOTSTRAP.cluster_bootstrap(
            conditioned,
            no_action,
            counts,
            samples=100,
            seed=5,
        )
        second = BOOTSTRAP.cluster_bootstrap(
            conditioned,
            no_action,
            counts,
            samples=100,
            seed=5,
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
