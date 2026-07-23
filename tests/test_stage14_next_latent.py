from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "wm" / "stage14_next_latent_baseline.py"
SPEC = importlib.util.spec_from_file_location("stage14_next_latent_baseline", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
STAGE14 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE14
SPEC.loader.exec_module(STAGE14)


class Stage14NextLatentTest(unittest.TestCase):
    def test_zero_initialized_model_is_exact_persistence(self) -> None:
        torch.manual_seed(1)
        model = STAGE14.TokenDynamicsModel(hidden_dimension=32, depth=1)
        latent = torch.randn(3, 32, 384)
        state = torch.randn(3, 8)
        action = torch.randn(3, 7)

        prediction = model(latent, state, action)

        self.assertTrue(torch.equal(prediction, latent))

    def test_batch_schedule_is_deterministic(self) -> None:
        first = STAGE14.make_batch_schedule(100, steps=5, batch_size=8, seed=1401)
        second = STAGE14.make_batch_schedule(100, steps=5, batch_size=8, seed=1401)
        different = STAGE14.make_batch_schedule(100, steps=5, batch_size=8, seed=1402)

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))

    def test_transition_store_never_crosses_episode_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = []
            for episode_index, length in ((10, 3), (11, 4)):
                tensors = {
                    "visual_tokens": torch.zeros(length, 2, 16, 384, dtype=torch.float16),
                    "observation_state": torch.zeros(length, 8),
                    "action": torch.zeros(length, 7),
                    "is_terminal": torch.arange(length) == length - 1,
                }
                relative = Path(f"episode_{episode_index}.safetensors")
                save_file(tensors, root / relative)
                manifest.append(
                    {
                        "split": "train",
                        "episode_index": str(episode_index),
                        "benchmark_task_id": "0",
                        "length": str(length),
                        "shard_file": relative.as_posix(),
                    }
                )

            store = STAGE14.load_transition_store(
                manifest,
                split="train",
                repo_root=root,
            )

            self.assertEqual(store.frame_count, 7)
            self.assertEqual(store.transition_count, 5)
            self.assertEqual(store.current_indices.tolist(), [0, 1, 3, 4, 5])
            self.assertEqual(store.next_indices.tolist(), [1, 2, 4, 5, 6])
            self.assertEqual(store.episode_ids.tolist(), [10, 10, 11, 11, 11])

    def test_normalization_uses_all_train_frames(self) -> None:
        store = STAGE14.TransitionStore(
            visual_tokens=torch.tensor(
                [[[-1.0] * 384] * 32, [[[1.0] * 384] * 32][0]],
                dtype=torch.float16,
            ),
            observation_state=torch.tensor([[-1.0] * 8, [1.0] * 8]),
            action=torch.tensor([[-2.0] * 7, [2.0] * 7]),
            current_indices=torch.tensor([0]),
            next_indices=torch.tensor([1]),
            task_ids=torch.tensor([0]),
            episode_ids=torch.tensor([10]),
        )

        normalization = STAGE14.compute_normalization(store)

        self.assertTrue(torch.equal(normalization.latent_mean, torch.zeros(384)))
        self.assertTrue(torch.equal(normalization.latent_std, torch.ones(384)))
        self.assertTrue(torch.equal(normalization.state_mean, torch.zeros(8)))
        self.assertTrue(torch.equal(normalization.action_std, torch.full((7,), 2.0)))


if __name__ == "__main__":
    unittest.main()
