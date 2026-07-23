from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "wm" / "stage15_multistep_rollout.py"
SPEC = importlib.util.spec_from_file_location("stage15_multistep_rollout", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
STAGE15 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE15
SPEC.loader.exec_module(STAGE15)


class Stage15RolloutTest(unittest.TestCase):
    def test_common_windows_do_not_cross_episode_boundary(self) -> None:
        store = SimpleNamespace(
            episode_ids=torch.tensor([10, 10, 10, 10, 11, 11, 11]),
        )

        positions = STAGE15.common_rollout_positions(store, max_horizon=3)

        self.assertEqual(positions.tolist(), [0, 1, 4])

    def test_shuffled_action_mapping_is_a_permutation(self) -> None:
        store = SimpleNamespace(
            transition_count=5,
            frame_count=7,
            current_indices=torch.tensor([0, 1, 3, 4, 5]),
        )

        mapping = STAGE15.shuffled_action_frame_indices(store, seed=1517)

        self.assertEqual(
            sorted(mapping[store.current_indices].tolist()),
            sorted(store.current_indices.tolist()),
        )

    def test_episode_bootstrap_detects_positive_effect(self) -> None:
        conditioned = np.array([0.8, 0.9, 1.8, 1.9])
        no_action = np.array([1.0, 1.1, 2.0, 2.1])
        episode_ids = np.array([10, 10, 11, 11])

        result = STAGE15.episode_cluster_bootstrap(
            conditioned,
            no_action,
            episode_ids,
            samples=1000,
            seed=5,
        )

        self.assertEqual(result["conditioned_better_episodes"], 2)
        self.assertGreater(result["improvement_percent_ci95"][0], 0)
        self.assertEqual(result["bootstrap_probability_positive"], 1.0)


if __name__ == "__main__":
    unittest.main()
