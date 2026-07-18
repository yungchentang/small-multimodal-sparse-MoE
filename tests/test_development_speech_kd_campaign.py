"""Tests for the provenance-aware development speech KD launcher."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "submit_development_speech_kd_campaign.sh"


class DevelopmentSpeechKDCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        scripts = self.repo / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(LAUNCHER, scripts / LAUNCHER.name)
        self.data = self.repo / "development_data"
        self.data.mkdir()
        (self.data / "manifest.json").write_text("{}", encoding="utf-8")
        self.split_manifest = self.repo / "development_split_manifest.json"
        self.split_manifest.write_text("{}", encoding="utf-8")
        self.stage_b = self.repo / "stage_b.pt"
        self.stage_a = self.repo / "stage_a.pt"
        self.manifest = self.repo / "manifest.json"
        self.stage_b.write_bytes(b"stage-b")
        self.stage_a.write_bytes(b"stage-a")
        self.manifest.write_text('{"development_only": true}', encoding="utf-8")
        subprocess.run(["git", "init"], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Speech KD Test"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Test fixture"],
            cwd=self.repo,
            check=True,
            capture_output=True,
        )
        self.source_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True
        ).strip()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run_launcher(
        self,
        *,
        drop: tuple[str, ...] = (),
        **updates: str,
    ) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "SOURCE_COMMIT_SHA": self.source_sha,
            "DATA_DIR": str(self.data),
            "DEVELOPMENT_SPLIT_MANIFEST": str(self.split_manifest.resolve()),
            "DEVELOPMENT_SPEECH_SOURCE_SHA256": "a" * 64,
            "BASE_OUT": str(self.repo / "outputs"),
            "STAGE_B_CHECKPOINT": str(self.stage_b),
            "STAGE_B_CHECKPOINT_SHA256": self.digest(self.stage_b),
            "MULTIMODAL_INITIAL_CHECKPOINT": str(self.stage_a),
            "MULTIMODAL_INITIAL_CHECKPOINT_SHA256": self.digest(self.stage_a),
            "MULTIMODAL_INITIAL_MANIFEST": str(self.manifest),
            "DRY_RUN": "1",
            "STAMP": "test",
        }
        env.update(updates)
        for name in drop:
            env.pop(name, None)
        return subprocess.run(
            ["bash", str(self.repo / "scripts" / LAUNCHER.name)],
            cwd=self.repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_dry_run_is_single_factor_frozen_a0_plus_kd(self) -> None:
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("arm=A0_KD", result.stdout)
        self.assertIn("gpu=1", result.stdout)
        self.assertIn("main_steps=500", result.stdout)
        self.assertIn("alignment_steps=400", result.stdout)
        self.assertIn("kl_coef=1.0", result.stdout)
        self.assertIn("router=0 experts=0 lm_head=0", result.stdout)
        self.assertIn("multimodal_scope=image", result.stdout)
        self.assertIn("stage_a_manifest=" + str(self.manifest), result.stdout)

    def test_strict_manifest_digest_and_parent_symlink_fail_closed(self) -> None:
        for variable in (
            "DEVELOPMENT_SPLIT_MANIFEST",
            "DEVELOPMENT_SPEECH_SOURCE_SHA256",
        ):
            with self.subTest(variable=variable):
                result = self.run_launcher(drop=(variable,))
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"{variable} is required", result.stderr)

        alias_parent = Path(self.temporary.name) / "alias-parent"
        alias_parent.symlink_to(self.repo, target_is_directory=True)
        result = self.run_launcher(DATA_DIR=str(alias_parent / self.data.name))
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsafe DATA_DIR symlink component", result.stderr)

    def test_checkpoint_hash_mismatch_fails_closed(self) -> None:
        result = self.run_launcher(STAGE_B_CHECKPOINT_SHA256="a" * 64)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Stage B checkpoint SHA-256 mismatch", result.stderr)

    def test_source_mismatch_fails_before_submission(self) -> None:
        result = self.run_launcher(SOURCE_COMMIT_SHA="0" * 40)
        self.assertEqual(result.returncode, 2)
        self.assertIn("source commit mismatch", result.stderr)

    def test_runner_and_runai_forward_kd_environment(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        run_sh = (ROOT / "run.sh").read_text(encoding="utf-8")
        submit = (ROOT / "scripts" / "submit_runai.sh").read_text(encoding="utf-8")
        self.assertIn("--speech-behavior-kl-coef", run_sh)
        self.assertIn("--speech-behavior-kl-temperature", run_sh)
        self.assertIn("--environment SPEECH_BEHAVIOR_KL_COEF=", submit)
        self.assertIn("--environment SPEECH_BEHAVIOR_KL_TEMPERATURE=", submit)
        launcher = LAUNCHER.read_text(encoding="utf-8")
        for required in (
            "SOURCE_COMMIT_SHA",
            "STAGE_B_CHECKPOINT_SHA256",
            "MULTIMODAL_INITIAL_CHECKPOINT_SHA256",
            "MULTIMODAL_INITIAL_MANIFEST",
            "sha256sum",
            "TRAIN_ROUTER_GATES=0 TRAIN_EXPERTS=0 TRAIN_LM_HEAD=0",
            "GPU=1",
            "validate_development_multimodal_runtime_manifest",
            'DEVELOPMENT_SPLIT_MANIFEST="$DEVELOPMENT_SPLIT_MANIFEST"',
            'DEVELOPMENT_SPEECH_SOURCE_SHA256="$DEVELOPMENT_SPEECH_SOURCE_SHA256"',
            "IMAGE_EVAL_SAMPLES=137",
            "SPEECH_EVAL_SAMPLES=137",
            "RETRIEVAL_EVAL_SAMPLES=137",
            "CONDITIONAL_EVAL_SAMPLES=137",
        ):
            self.assertIn(required, launcher)


if __name__ == "__main__":
    unittest.main()
