from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "submit_development_selected_expert_campaign.sh"


class DevelopmentSelectedExpertCampaignTests(unittest.TestCase):
    def _run_strict_fixture(
        self,
        root: Path,
        *,
        drop: tuple[str, ...] = (),
        parent_symlink: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        source_sha = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        real_parent = root / "real-parent"
        data = real_parent / "data"
        data.mkdir(parents=True)
        (data / "manifest.json").write_text("{}", encoding="utf-8")
        split_manifest = root / "development_split_manifest.json"
        split_manifest.write_text("{}", encoding="utf-8")
        selection = root / "selection.json"
        stage_b = root / "stage_b.pt"
        image = root / "image.pt"
        manifest = root / "manifest.json"
        for path in (selection, stage_b, image, manifest):
            path.write_text("evidence", encoding="utf-8")
        data_value = data
        if parent_symlink:
            alias_parent = root / "alias-parent"
            alias_parent.symlink_to(real_parent, target_is_directory=True)
            data_value = alias_parent / data.name
        env = {
            **os.environ,
            "SOURCE_COMMIT_SHA": source_sha,
            "DATA_DIR": str(data_value),
            "DEVELOPMENT_SPLIT_MANIFEST": str(split_manifest.resolve()),
            "DEVELOPMENT_SPEECH_SOURCE_SHA256": "a" * 64,
            "BASE_OUT": str(root / "outputs"),
            "EXPERT_SELECTION_JSON": str(selection),
            "STAGE_B_CHECKPOINT": str(stage_b),
            "STAGE_B_CHECKPOINT_SHA256": "a" * 64,
            "MULTIMODAL_INITIAL_CHECKPOINT": str(image),
            "MULTIMODAL_INITIAL_CHECKPOINT_SHA256": "b" * 64,
            "MULTIMODAL_INITIAL_MANIFEST": str(manifest),
            "ONLY": "A0",
            "DRY_RUN": "1",
            "ALLOW_DIRTY_DRY_RUN": "1",
            "STAMP": "test",
        }
        for name in drop:
            env.pop(name, None)
        return subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_dry_run_is_factorized_and_requires_exact_initialization(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        source_sha = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "manifest.json").write_text("{}", encoding="utf-8")
            split_manifest = root / "development_split_manifest.json"
            split_manifest.write_text("{}", encoding="utf-8")
            selection = root / "selection.json"
            stage_b = root / "stage_b.pt"
            image = root / "image.pt"
            manifest = root / "manifest.json"
            for path in (selection, stage_b, image, manifest):
                path.write_text("evidence", encoding="utf-8")
            env = {
                **os.environ,
                "SOURCE_COMMIT_SHA": source_sha,
                "DATA_DIR": str(data),
                "DEVELOPMENT_SPLIT_MANIFEST": str(split_manifest.resolve()),
                "DEVELOPMENT_SPEECH_SOURCE_SHA256": "a" * 64,
                "BASE_OUT": str(root / "outputs"),
                "EXPERT_SELECTION_JSON": str(selection),
                "STAGE_B_CHECKPOINT": str(stage_b),
                "STAGE_B_CHECKPOINT_SHA256": "a" * 64,
                "MULTIMODAL_INITIAL_CHECKPOINT": str(image),
                "MULTIMODAL_INITIAL_CHECKPOINT_SHA256": "b" * 64,
                "MULTIMODAL_INITIAL_MANIFEST": str(manifest),
                "ONLY": "A0,A3,C1,C2",
                "DRY_RUN": "1",
            "ALLOW_DIRTY_DRY_RUN": "1",
                "STAMP": "test",
            }
            result = subprocess.run(
                ["bash", str(LAUNCHER)], cwd=ROOT, env=env,
                text=True, capture_output=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("arm=A0", lines[0])
        self.assertIn("selected=0", lines[0])
        self.assertIn("arm=A3", lines[1])
        self.assertIn("bias=0.001", lines[2])
        self.assertIn("router=1", lines[3])
        self.assertTrue(all("selected=1" in line for line in lines[1:]))
        self.assertTrue(all("lm_head=0" in line for line in lines))
        self.assertTrue(all("multimodal_scope=image" in line for line in lines))
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('ALLOW_SELECTED_EXPERT_ROUTER_TUNING="$router"', text)

    def test_strict_manifest_digest_and_parent_symlink_fail_closed(self) -> None:
        for variable in (
            "DEVELOPMENT_SPLIT_MANIFEST",
            "DEVELOPMENT_SPEECH_SOURCE_SHA256",
        ):
            with self.subTest(variable=variable), tempfile.TemporaryDirectory() as temporary:
                result = self._run_strict_fixture(
                    Path(temporary), drop=(variable,)
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"{variable} is required", result.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            result = self._run_strict_fixture(
                Path(temporary), parent_symlink=True
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("unsafe DATA_DIR symlink component", result.stderr)

    def test_launcher_rejects_missing_checkpoint_hashes(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("STAGE_B_CHECKPOINT_SHA256:?", text)
        self.assertIn("MULTIMODAL_INITIAL_CHECKPOINT_SHA256:?", text)
        self.assertIn("MULTIMODAL_INITIAL_MANIFEST:?", text)
        self.assertIn("refusing sealed/synthetic path", text)
        self.assertIn("screening runs require 500-1000 steps", text)
        self.assertIn("alignment pretraining must be positive and shorter", text)
        self.assertIn("validate_development_multimodal_runtime_manifest", text)
        self.assertIn('DEVELOPMENT_SPLIT_MANIFEST="$DEVELOPMENT_SPLIT_MANIFEST"', text)
        self.assertIn(
            'DEVELOPMENT_SPEECH_SOURCE_SHA256="$DEVELOPMENT_SPEECH_SOURCE_SHA256"',
            text,
        )
        for count in (
            "IMAGE_EVAL_SAMPLES=137",
            "SPEECH_EVAL_SAMPLES=137",
            "RETRIEVAL_EVAL_SAMPLES=137",
            "CONDITIONAL_EVAL_SAMPLES=137",
        ):
            self.assertIn(count, text)
        run_script = (ROOT / "run.sh").read_text(encoding="utf-8")
        submit_script = (ROOT / "scripts" / "submit_runai.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--multimodal-initial-manifest", run_script)
        self.assertIn("MULTIMODAL_INITIAL_MANIFEST", run_script)
        self.assertIn(
            '--environment MULTIMODAL_INITIAL_MANIFEST="${MULTIMODAL_INITIAL_MANIFEST:-}"',
            submit_script,
        )

    def test_launcher_rejects_synthetic_multimodal_manifest_path(self) -> None:
        source_sha = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "manifest.json").write_text("{}", encoding="utf-8")
            split_manifest = root / "development_split_manifest.json"
            split_manifest.write_text("{}", encoding="utf-8")
            selection = root / "selection.json"
            stage_b = root / "stage_b.pt"
            image = root / "image.pt"
            synthetic_manifest = root / "synthetic_manifest.json"
            for path in (selection, stage_b, image, synthetic_manifest):
                path.write_text("evidence", encoding="utf-8")
            env = {
                **os.environ,
                "SOURCE_COMMIT_SHA": source_sha,
                "DATA_DIR": str(data),
                "DEVELOPMENT_SPLIT_MANIFEST": str(split_manifest.resolve()),
                "DEVELOPMENT_SPEECH_SOURCE_SHA256": "a" * 64,
                "BASE_OUT": str(root / "outputs"),
                "EXPERT_SELECTION_JSON": str(selection),
                "STAGE_B_CHECKPOINT": str(stage_b),
                "STAGE_B_CHECKPOINT_SHA256": "a" * 64,
                "MULTIMODAL_INITIAL_CHECKPOINT": str(image),
                "MULTIMODAL_INITIAL_CHECKPOINT_SHA256": "b" * 64,
                "MULTIMODAL_INITIAL_MANIFEST": str(synthetic_manifest),
                "DRY_RUN": "1",
            "ALLOW_DIRTY_DRY_RUN": "1",
            }
            result = subprocess.run(
                ["bash", str(LAUNCHER)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing sealed/synthetic path", result.stderr)

    def test_unknown_arm_fails_closed(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("unknown development arm", text)

    def test_runner_records_source_and_completion_provenance(self) -> None:
        text = (ROOT / "training" / "olmoe_real_subset_runs.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"source_commit_sha": source_commit_sha', text)
        self.assertIn('"e3_checkpoint_sha256": e3["checkpoint_sha256"]', text)
        self.assertIn('"status": "completed"', text)
        self.assertIn('"run_provenance": run_provenance', text)


if __name__ == "__main__":
    unittest.main()
