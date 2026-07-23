from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "wm" / "stage13_full_feature_cache.py"
SPEC = importlib.util.spec_from_file_location("stage13_full_feature_cache", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
STAGE13 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE13
SPEC.loader.exec_module(STAGE13)


def manifest_row(split: str = "train", episode_index: int = 42, length: int = 3) -> dict[str, str]:
    return {
        "split": split,
        "benchmark_task_id": "4",
        "dataset_task_index": "31",
        "episode_index": str(episode_index),
        "length": str(length),
        "task_prompt": "test task",
    }


class Stage13CacheTest(unittest.TestCase):
    def test_shard_path_is_stable(self) -> None:
        self.assertEqual(
            STAGE13.shard_relative_path(42).as_posix(),
            "shards/episode_000042.safetensors",
        )

    def test_recheck_selection_has_one_episode_per_split(self) -> None:
        rows = [
            manifest_row(split=split, episode_index=100 * split_index + episode)
            for split_index, split in enumerate(STAGE13.CACHE_SPLITS)
            for episode in range(4)
        ]
        first = STAGE13.select_recheck_rows(rows, seed=1300)
        second = STAGE13.select_recheck_rows(rows, seed=1300)
        self.assertEqual(first, second)
        self.assertEqual({row["split"] for row in first}, set(STAGE13.CACHE_SPLITS))

    def test_feature_statistics_match_direct_numpy_result(self) -> None:
        values = torch.arange(2 * 2 * 16 * 384, dtype=torch.float32).reshape(2, 2, 16, 384)
        accumulator = STAGE13.FeatureStatsAccumulator()
        accumulator.update(values[:1].to(torch.float16))
        accumulator.update(values[1:].to(torch.float16))

        result = accumulator.finalize()
        expected = values.to(torch.float16).numpy().astype(np.float64)

        self.assertAlmostEqual(result["mean"], float(expected.mean()), places=6)
        self.assertAlmostEqual(result["standard_deviation"], float(expected.std()), places=6)

    def test_valid_shard_is_accepted(self) -> None:
        row = manifest_row()
        length = int(row["length"])
        tensors = {
            "visual_tokens": torch.zeros(length, 2, 16, 384, dtype=torch.float16),
            "observation_state": torch.zeros(length, 8, dtype=torch.float32),
            "action": torch.zeros(length, 7, dtype=torch.float32),
            "dataset_index": torch.arange(length, dtype=torch.int64),
            "frame_index": torch.arange(length, dtype=torch.int64),
            "timestamp": torch.arange(length, dtype=torch.float32) / 10,
            "is_terminal": torch.tensor([False, False, True]),
        }
        metadata = STAGE13.expected_metadata(row, "a" * 64)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "episode.safetensors"
            save_file(tensors, path, metadata=metadata)

            loaded, loaded_metadata = STAGE13.validate_shard(path, row)

            self.assertEqual(set(loaded), STAGE13.EXPECTED_KEYS)
            self.assertEqual(loaded_metadata["decoded_pixels_sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
