from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = REPO_ROOT / "scripts" / "sealed_evaluation_defaults.sh"
FREEZE = REPO_ROOT / "scripts" / "freeze_final_protocol.sh"
SUBMIT = REPO_ROOT / "scripts" / "submit_sealed_control_matrix.sh"


class SealedLauncherContractTest(unittest.TestCase):
    def _resolved_batch(self, value: str | None = None) -> str:
        env = dict(os.environ)
        env.pop("CONDITIONAL_BATCH_SIZE", None)
        if value is not None:
            env["CONDITIONAL_BATCH_SIZE"] = value
        completed = subprocess.run(
            [
                "bash",
                "-c",
                'source scripts/sealed_evaluation_defaults.sh; printf "%s" "$CONDITIONAL_BATCH_SIZE"',
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return completed.stdout

    def test_freeze_and_submit_source_one_batch_contract(self) -> None:
        self.assertTrue(DEFAULTS.is_file())
        freeze = FREEZE.read_text(encoding="utf-8")
        submit = SUBMIT.read_text(encoding="utf-8")
        source = "source scripts/sealed_evaluation_defaults.sh"
        self.assertIn(source, freeze)
        self.assertIn(source, submit)
        self.assertIn('--conditional-batch-size "$CONDITIONAL_BATCH_SIZE"', freeze)
        self.assertIn('CONDITIONAL_BATCH_SIZE="$CONDITIONAL_BATCH_SIZE"', submit)
        self.assertNotIn("CONDITIONAL_BATCH_SIZE:-", freeze)
        self.assertNotIn("CONDITIONAL_BATCH_SIZE=16", submit)
        self.assertIn(
            "--evaluator-script scripts/sealed_evaluation_defaults.sh", freeze
        )
        self.assertEqual(self._resolved_batch(), "16")
        self.assertEqual(self._resolved_batch("7"), "7")

    def test_batch_contract_rejects_invalid_values(self) -> None:
        for value in ("0", "-1", "1.5", "false"):
            with self.subTest(value=value):
                completed = subprocess.run(
                    ["bash", "-c", "source scripts/sealed_evaluation_defaults.sh"],
                    cwd=REPO_ROOT,
                    env={**os.environ, "CONDITIONAL_BATCH_SIZE": value},
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
