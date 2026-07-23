from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "wm" / "stage11_build_dataset_manifest.py"
SPEC = importlib.util.spec_from_file_location("stage11_build_dataset_manifest", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
STAGE11 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE11)


class Stage11SplitTest(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self) -> None:
        episodes = {
            30: list(range(100, 140)),
            31: list(range(200, 239)),
        }
        first = STAGE11.assign_episode_splits(episodes, seed=1000)
        second = STAGE11.assign_episode_splits(episodes, seed=1000)

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(episodes[30]) | set(episodes[31]))
        for task_episode_ids in episodes.values():
            counts = {
                split: sum(first[episode_id] == split for episode_id in task_episode_ids)
                for split in STAGE11.SPLIT_NAMES
            }
            self.assertGreater(counts["train"], counts["validation"])
            self.assertGreater(counts["train"], counts["test"])
            self.assertGreaterEqual(counts["validation"], 1)
            self.assertGreaterEqual(counts["test"], 1)

    def test_split_changes_with_seed(self) -> None:
        episodes = {30: list(range(100, 140))}
        first = STAGE11.assign_episode_splits(episodes, seed=1000)
        second = STAGE11.assign_episode_splits(episodes, seed=1001)

        self.assertNotEqual(first, second)

    def test_too_few_episodes_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            STAGE11.assign_episode_splits({30: [1, 2]}, seed=1000)

    def test_episode_file_mapping_uses_parquet_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data" / "chunk-000"
            data_dir.mkdir(parents=True)
            pq.write_table(
                pa.table({"episode_index": [10, 10, 11]}),
                data_dir / "file-004.parquet",
            )
            pq.write_table(
                pa.table({"episode_index": [12, 12]}),
                data_dir / "file-009.parquet",
            )

            episode_to_file, files = STAGE11.map_episode_data_files(root)

            self.assertEqual(
                episode_to_file,
                {
                    10: "data/chunk-000/file-004.parquet",
                    11: "data/chunk-000/file-004.parquet",
                    12: "data/chunk-000/file-009.parquet",
                },
            )
            self.assertEqual({path.name for path in files}, {"file-004.parquet", "file-009.parquet"})


if __name__ == "__main__":
    unittest.main()
