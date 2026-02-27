from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from phase_two.exp1_rank_p import load_partial_uncertainty_trials


class Exp1ResumeTests(unittest.TestCase):
    def test_load_partial_uncertainty_trials_valid_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            partial_path = Path(tmp_dir) / "exp1_partial.npz"
            trials = np.arange(12, dtype=np.float32).reshape(3, 4)
            np.savez_compressed(
                partial_path,
                uncertainty_trials=trials,
                p_value=np.asarray([0.01], dtype=np.float64),
                passes=np.asarray([64], dtype=np.int64),
                completed_trials=np.asarray([2], dtype=np.int64),
                total_trials=np.asarray([3], dtype=np.int64),
            )

            loaded = load_partial_uncertainty_trials(
                partial_path,
                expected_trials=3,
                expected_num_images=4,
                expected_passes=64,
                expected_p=0.01,
            )

            self.assertEqual(len(loaded), 2)
            np.testing.assert_array_equal(loaded[0], trials[0])
            np.testing.assert_array_equal(loaded[1], trials[1])

    def test_load_partial_uncertainty_trials_ignores_mismatched_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            partial_path = Path(tmp_dir) / "exp1_partial.npz"
            trials = np.arange(8, dtype=np.float32).reshape(2, 4)
            np.savez_compressed(
                partial_path,
                uncertainty_trials=trials,
                p_value=np.asarray([0.02], dtype=np.float64),
                passes=np.asarray([32], dtype=np.int64),
                completed_trials=np.asarray([2], dtype=np.int64),
                total_trials=np.asarray([2], dtype=np.int64),
            )

            loaded = load_partial_uncertainty_trials(
                partial_path,
                expected_trials=2,
                expected_num_images=4,
                expected_passes=64,
                expected_p=0.02,
            )
            self.assertEqual(loaded, [])


if __name__ == "__main__":
    unittest.main()
