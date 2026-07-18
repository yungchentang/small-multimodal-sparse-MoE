from pathlib import Path
import os
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SH = REPO_ROOT / "run.sh"
SYNTHETIC_DEMO_SH = REPO_ROOT / "scripts" / "run_synthetic_demo.sh"


class RunModeContractTest(unittest.TestCase):
    def run_mode(self, mode: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("SOURCE_COMMIT_SHA", None)
        return subprocess.run(
            ["bash", str(RUN_SH), mode],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            check=False,
            text=True,
        )

    def test_final_fails_closed_and_points_to_frozen_commands(self) -> None:
        result = self.run_mode("final")

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "final is disabled; freeze the protocol, then use "
            "scripts/submit_sealed_control_matrix.sh and "
            "scripts/submit_sealed_analysis.sh\n",
        )

    def test_usage_exposes_synthetic_demo_not_final(self) -> None:
        result = self.run_mode("not-a-mode")

        self.assertEqual(result.returncode, 2)
        self.assertIn("synthetic-demo", result.stderr)
        self.assertNotIn("|final", result.stderr)

    def test_legacy_pipeline_is_named_synthetic_demo(self) -> None:
        runner = RUN_SH.read_text(encoding="utf-8")
        demo_case = runner.split("  synthetic-demo)\n", 1)[1].split("    ;;\n", 1)[0]

        self.assertIn("training.calibrate_top2", demo_case)
        self.assertIn("training.train", demo_case)
        self.assertIn("evaluation.evaluate", demo_case)

        wrapper = SYNTHETIC_DEMO_SH.read_text(encoding="utf-8")
        self.assertIn("bash run.sh synthetic-demo", wrapper)
        self.assertNotIn("bash run.sh final", wrapper)

    def test_no_argument_release_command_defaults_to_smoke(self) -> None:
        runner = RUN_SH.read_text(encoding="utf-8")

        self.assertIn('MODE="${1:-smoke}"', runner)
        self.assertIn("  smoke)\n", runner)
        smoke_case = runner.split("  smoke)\n", 1)[1].split("    ;;\n", 1)[0]
        self.assertIn('OUT="${OUT:-outputs/smoke}"', smoke_case)


if __name__ == "__main__":
    unittest.main()
