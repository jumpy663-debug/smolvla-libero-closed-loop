from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "wm" / "stage12_representation_pilot.py"
SPEC = importlib.util.spec_from_file_location("stage12_representation_pilot", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
STAGE12 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE12)


class Stage12PilotTest(unittest.TestCase):
    def test_even_frame_indices_include_episode_boundaries(self) -> None:
        self.assertEqual(STAGE12.even_frame_indices(100, 4), [0, 33, 66, 99])
        self.assertEqual(STAGE12.even_frame_indices(101, 1), [50])

    def test_even_frame_indices_reject_invalid_request(self) -> None:
        with self.assertRaises(ValueError):
            STAGE12.even_frame_indices(3, 4)
        with self.assertRaises(ValueError):
            STAGE12.even_frame_indices(10, 0)

    def test_selection_is_deterministic_and_covers_each_split_task(self) -> None:
        rows = [
            {
                "split": split,
                "benchmark_task_id": str(task),
                "episode_index": str(1000 * split_index + 10 * task + episode),
            }
            for split_index, split in enumerate(STAGE12.PILOT_SPLITS)
            for task in range(3)
            for episode in range(4)
        ]
        first = STAGE12.select_pilot_episodes(
            rows,
            splits=STAGE12.PILOT_SPLITS,
            episodes_per_task=1,
            seed=1200,
        )
        second = STAGE12.select_pilot_episodes(
            rows,
            splits=STAGE12.PILOT_SPLITS,
            episodes_per_task=1,
            seed=1200,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(
            {(row["split"], int(row["benchmark_task_id"])) for row in first},
            {(split, task) for split in STAGE12.PILOT_SPLITS for task in range(3)},
        )

    def test_patch_pooling_preserves_spatial_order(self) -> None:
        cls = torch.tensor([[[-1.0]]])
        patches = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1)
        hidden = torch.cat([cls, patches], dim=1)

        pooled = STAGE12.pool_patch_tokens(hidden, grid_size=2)

        self.assertEqual(tuple(pooled.shape), (1, 4, 1))
        self.assertTrue(
            torch.equal(
                pooled.flatten(),
                torch.tensor([2.5, 4.5, 10.5, 12.5]),
            )
        )


if __name__ == "__main__":
    unittest.main()
