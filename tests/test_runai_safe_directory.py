from pathlib import Path
import unittest


class RunAISafeDirectoryTest(unittest.TestCase):
    def test_launcher_trusts_only_the_mounted_repository(self) -> None:
        launcher = Path("scripts/submit_runai.sh").read_text(encoding="utf-8")
        self.assertIn("--environment GIT_CONFIG_COUNT=1", launcher)
        self.assertIn("--environment GIT_CONFIG_KEY_0=safe.directory", launcher)
        self.assertIn('--environment GIT_CONFIG_VALUE_0="$REPO_DIR"', launcher)
        self.assertNotIn("safe.directory=*", launcher)


    def test_conditional_launchers_declare_explicit_evaluation_scope(self) -> None:
        run_script = Path("run.sh").read_text(encoding="utf-8")
        submitter = Path("scripts/submit_runai.sh").read_text(
            encoding="utf-8"
        )
        development = Path(
            "scripts/submit_development_candidate_matrix.sh"
        ).read_text(encoding="utf-8")
        followup = Path(
            "scripts/submit_followup_development_matrix.sh"
        ).read_text(encoding="utf-8")
        final = Path(
            "scripts/submit_sealed_control_matrix.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '--evaluation-scope "${EVALUATION_SCOPE:?EVALUATION_SCOPE is required}"',
            run_script,
        )
        self.assertIn(
            '--environment EVALUATION_SCOPE="${EVALUATION_SCOPE:-}"',
            submitter,
        )
        self.assertIn("EVALUATION_SCOPE=development", development)
        self.assertIn("EVALUATION_SCOPE=development", followup)
        self.assertIn("EVALUATION_SCOPE=final", final)


if __name__ == "__main__":
    unittest.main()
