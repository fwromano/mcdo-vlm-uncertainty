from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


class RunWrapperSmokeTests(unittest.TestCase):
    def test_phase_one_help_does_not_trigger_unbound_variable(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["RUN_DEVICE"] = "cpu"

        proc = subprocess.run(
            ["./run", "phase", "one", "--help"],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )

        combined = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, msg=combined)
        self.assertIn("run_phase1_fast.py", combined)
        self.assertNotIn("unbound variable", combined)


if __name__ == "__main__":
    unittest.main()
