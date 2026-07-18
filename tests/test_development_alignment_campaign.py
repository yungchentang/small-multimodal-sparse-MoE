"""Tests for the development-only Stage A campaign launcher."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "submit_development_alignment_campaign.sh"


class DevelopmentAlignmentCampaignTest(unittest.TestCase):
    def run_script(self, **updates: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(updates)
        return subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_dry_run_emits_matched_image_and_speech_arms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "development-data"
            data.mkdir()
            split_manifest = Path(directory) / "development-splits.json"
            split_manifest.write_text("{}", encoding="utf-8")
            result = self.run_script(
                SOURCE_COMMIT_SHA=subprocess.check_output(
                    [
                        "git",
                        "-c",
                        f"safe.directory={REPO_ROOT}",
                        "rev-parse",
                        "HEAD",
                    ],
                    cwd=REPO_ROOT,
                    text=True,
                ).strip(),
                DATA_DIR=str(data / ".." / data.name),
                DEVELOPMENT_SPLIT_MANIFEST=str(split_manifest),
                DEVELOPMENT_SPEECH_SOURCE_SHA256="a" * 64,
                BASE_OUT=str(Path(directory) / "outputs"),
                STAMP="test",
                ALLOW_SMOKE="1",
                SCREEN_STEPS="12",
                ALIGNMENT_PRETRAIN_STEPS="8",
                ONLY="I_LINEAR,I_NORM,S_ATTN6",
                DRY_RUN="1",
                ALLOW_DIRTY_DRY_RUN="1",
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("arm=I_LINEAR", result.stdout)
        self.assertIn("modality=image", result.stdout)
        self.assertIn("image_bridge=linear_projector", result.stdout)
        self.assertIn("arm=I_NORM", result.stdout)
        self.assertIn("image_bridge=linear_projector_norm", result.stdout)
        self.assertIn(f"development_split_manifest={split_manifest.resolve()}", result.stdout)
        self.assertIn(f"data_dir={data.resolve()}", result.stdout)
        self.assertIn("arm=S_ATTN6", result.stdout)
        self.assertIn("modality=speech", result.stdout)
        self.assertIn("audio_bridge=attention_pool", result.stdout)

    def test_runai_entrypoints_forward_alignment_controls(self) -> None:
        run_sh = (REPO_ROOT / "run.sh").read_text(encoding="utf-8")
        submit = (REPO_ROOT / "scripts" / "submit_runai.sh").read_text(encoding="utf-8")
        for cli in (
            "--image-alignment-target",
            "--image-bridge-type",
            "--audio-bridge-type",
            "--audio-max-seconds",
            "--speech-unfreeze-last-blocks",
            "--alignment-pretrain-modalities",
        ):
            self.assertIn(cli, run_sh)
        for variable in (
            "IMAGE_ALIGNMENT_TARGET",
            "IMAGE_BRIDGE_TYPE",
            "AUDIO_BRIDGE_TYPE",
            "AUDIO_MAX_SECONDS",
            "SPEECH_UNFREEZE_LAST_BLOCKS",
            "SPEECH_UNFREEZE_LAYER_NORM",
            "ALIGNMENT_PRETRAIN_MODALITIES",
            "SOURCE_COMMIT_SHA",
            "DEVELOPMENT_SPLIT_MANIFEST",
            "DEVELOPMENT_SPEECH_SOURCE_SHA256",
        ):
            self.assertIn(f"--environment {variable}=", submit)

    def test_launcher_pins_group_clean_batch_one_protocol(self) -> None:
        launcher = SCRIPT.read_text(encoding="utf-8")
        for token in (
            "I_NORM|image|linear_projector_norm",
            "TRAIN_BATCH_SIZE=1 EVAL_BATCH_SIZE=1",
            'SPEECH_TEACHER_BANK_BATCH_SIZE="$SPEECH_TEACHER_BANK_BATCH_SIZE"',
            "IMAGE_EVAL_SAMPLES=137",
            "SPEECH_EVAL_SAMPLES=137",
            "RETRIEVAL_EVAL_SAMPLES=137",
            "CONDITIONAL_EVAL_SAMPLES=137",
            "CONDITIONAL_BATCH_SIZE=1",
        ):
            with self.subTest(token=token):
                self.assertIn(token, launcher)

    def test_runtime_validation_keeps_reserved_eval_split_file_unread(self) -> None:
        for script_name in (
            "submit_development_alignment_campaign.sh",
            "submit_development_mm_norm_kd_campaign.sh",
        ):
            launcher = (REPO_ROOT / "scripts" / script_name).read_text(
                encoding="utf-8"
            )
            with self.subTest(script=script_name):
                self.assertIn(
                    "validate_development_multimodal_runtime_manifest", launcher
                )
                self.assertIn("expected_source_commit_sha=sys.argv[2]", launcher)
                self.assertNotIn(
                    "validate_development_multimodal_split_manifest as validate",
                    launcher,
                )
    def test_canonical_data_dir_does_not_bypass_sealed_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed_data = root / "sealed-data"
            sealed_data.mkdir()
            alias = root / "development-data"
            alias.symlink_to(sealed_data, target_is_directory=True)
            split_manifest = root / "manifest.json"
            split_manifest.write_text("{}", encoding="utf-8")
            result = self.run_script(
                SOURCE_COMMIT_SHA=subprocess.check_output(
                    [
                        "git", "-c", f"safe.directory={REPO_ROOT}",
                        "rev-parse", "HEAD",
                    ],
                    cwd=REPO_ROOT,
                    text=True,
                ).strip(),
                DATA_DIR=str(alias),
                DEVELOPMENT_SPLIT_MANIFEST=str(split_manifest),
                DEVELOPMENT_SPEECH_SOURCE_SHA256="a" * 64,
                DRY_RUN="1",
                ALLOW_DIRTY_DRY_RUN="1",
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsafe DATA_DIR symlink component", result.stderr)

    def test_data_dir_parent_symlink_alias_is_rejected_before_realpath(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_parent = root / "real-parent"
            data = real_parent / "development-data"
            data.mkdir(parents=True)
            alias_parent = root / "alias-parent"
            alias_parent.symlink_to(real_parent, target_is_directory=True)
            split_manifest = root / "manifest.json"
            split_manifest.write_text("{}", encoding="utf-8")
            result = self.run_script(
                SOURCE_COMMIT_SHA=subprocess.check_output(
                    [
                        "git", "-c", f"safe.directory={REPO_ROOT}",
                        "rev-parse", "HEAD",
                    ],
                    cwd=REPO_ROOT,
                    text=True,
                ).strip(),
                DATA_DIR=str(alias_parent / data.name),
                DEVELOPMENT_SPLIT_MANIFEST=str(split_manifest),
                DEVELOPMENT_SPEECH_SOURCE_SHA256="a" * 64,
                DRY_RUN="1",
                ALLOW_DIRTY_DRY_RUN="1",
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsafe DATA_DIR symlink component", result.stderr)
        self.assertIn(str(alias_parent), result.stderr)

    def test_source_mismatch_fails_before_submission(self) -> None:
        result = self.run_script(SOURCE_COMMIT_SHA="0" * 40, DRY_RUN="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source commit mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
