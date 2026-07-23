from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "wm" / "stage16_collect_closed_loop.py"
SPEC = importlib.util.spec_from_file_location("stage16_collect_closed_loop", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import {SCRIPT_PATH}")
STAGE16 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE16
SPEC.loader.exec_module(STAGE16)


def valid_arrays(steps: int = 3, resolution: int = 4) -> dict[str, np.ndarray]:
    observations = steps + 1
    arrays = {
        "camera1": np.zeros((observations, resolution, resolution, 3), dtype=np.uint8),
        "camera2": np.ones((observations, resolution, resolution, 3), dtype=np.uint8),
        "eef_pos": np.zeros((observations, 3), dtype=np.float32),
        "eef_quat": np.zeros((observations, 4), dtype=np.float32),
        "eef_mat": np.zeros((observations, 3, 3), dtype=np.float32),
        "gripper_qpos": np.zeros((observations, 2), dtype=np.float32),
        "gripper_qvel": np.zeros((observations, 2), dtype=np.float32),
        "joint_pos": np.zeros((observations, 7), dtype=np.float32),
        "joint_vel": np.zeros((observations, 7), dtype=np.float32),
        "action": np.zeros((steps, 7), dtype=np.float32),
        "reward": np.zeros(steps, dtype=np.float32),
        "success": np.zeros(steps, dtype=np.bool_),
        "done": np.array([False] * (steps - 1) + [True], dtype=np.bool_),
        "transition_valid": np.ones(steps, dtype=np.bool_),
    }
    return arrays


class Stage16RecollectionTest(unittest.TestCase):
    def test_success_transition_excludes_automatic_reset(self) -> None:
        mask = STAGE16.transition_valid_mask(steps=4, success=True)
        self.assertEqual(mask.tolist(), [True, True, True, False])

    def test_failure_keeps_time_limit_transition(self) -> None:
        mask = STAGE16.transition_valid_mask(steps=4, success=False)
        self.assertTrue(bool(mask.all()))

    def test_content_hash_is_key_order_independent(self) -> None:
        arrays = valid_arrays()
        reversed_arrays = dict(reversed(list(arrays.items())))
        self.assertEqual(
            STAGE16.episode_content_sha256(arrays),
            STAGE16.episode_content_sha256(reversed_arrays),
        )

    def test_validation_rejects_success_crossing_transition(self) -> None:
        arrays = valid_arrays()
        arrays["success"][-1] = True
        with self.assertRaisesRegex(AssertionError, "automatic reset"):
            STAGE16.validate_episode_arrays(arrays, resolution=4)


if __name__ == "__main__":
    unittest.main()
