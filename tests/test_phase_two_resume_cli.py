from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


class PhaseTwoResumeCliTests(unittest.TestCase):
    def test_run_phase2_help_mentions_exp1_resume(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            ["python", "-m", "phase_two.run_phase2", "--help"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        combined = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, msg=combined)
        self.assertIn("--exp1-resume", combined)


if __name__ == "__main__":
    unittest.main()
