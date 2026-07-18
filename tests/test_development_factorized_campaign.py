from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "submit_development_factorized_campaign.sh"


class DevelopmentFactorizedCampaignTests(unittest.TestCase):
    def _run(
        self,
        *,
        only: str = "",
        selection_name: str = "selection.json",
        drop: tuple[str, ...] = (),
        parent_symlink: bool = False,
    ):
        source_sha = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        data_dir = root / "development_data"
        data_dir.mkdir()
        (data_dir / "manifest.json").write_text("{}", encoding="utf-8")
        split_manifest = root / "development_split_manifest.json"
        split_manifest.write_text("{}", encoding="utf-8")
        selection = root / selection_name
        selection.write_text("{}", encoding="utf-8")
        env = {
            **os.environ,
            "SOURCE_COMMIT_SHA": source_sha,
            "DATA_DIR": str(data_dir),
            "DEVELOPMENT_SPLIT_MANIFEST": str(split_manifest.resolve()),
            "DEVELOPMENT_SPEECH_SOURCE_SHA256": "a" * 64,
            "BASE_OUT": str(root / "campaign"),
            "EXPERT_SELECTION_JSON": str(selection),
            "DRY_RUN": "1",
            "ALLOW_DIRTY_DRY_RUN": "1",
            "STAMP": "test",
        }
        if parent_symlink:
            alias_parent = root / "alias-parent"
            alias_parent.symlink_to(root, target_is_directory=True)
            env["DATA_DIR"] = str(alias_parent / data_dir.name)
        if only:
            env["ONLY"] = only
        for name in drop:
            env.pop(name, None)
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=ROOT,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        return temporary, result, source_sha

    def test_launcher_is_syntactically_valid_and_factorized(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        text = LAUNCHER.read_text(encoding="utf-8")
        for arm in ("A0", "A1", "A2", "A3", "A4", "A5"):
            self.assertEqual(text.count(f'"{arm}|'), 1)
        self.assertIn("SOURCE_COMMIT_SHA", text)
        self.assertIn("safe.directory=$REPO_ROOT", text)
        self.assertIn("diff --quiet", text)
        self.assertIn("refusing sealed/synthetic path", text)
        self.assertIn("refusing overwrite", text)
        self.assertIn("500-1000 steps", text)
        self.assertIn("ALIGNMENT_PRETRAIN_STEPS", text)
        self.assertIn("EXPERT_SELECTION_JSON", text)
        self.assertIn("SUBMIT_REPO_DIR", text)
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
        runner = (ROOT / "run.sh").read_text(encoding="utf-8")
        submit = (ROOT / "scripts" / "submit_runai.sh").read_text(
            encoding="utf-8"
        )
        for variable in (
            "STAGE_B_CHECKPOINT",
            "STAGE_B_CHECKPOINT_SHA256",
            "MULTIMODAL_INITIAL_CHECKPOINT",
            "MULTIMODAL_INITIAL_CHECKPOINT_SHA256",
            "MULTIMODAL_INITIALIZATION_SCOPE",
        ):
            self.assertIn(variable, runner)
            self.assertIn(variable, submit)

    def test_default_dry_run_emits_five_supported_single_factor_arms(self) -> None:
        temporary, result, source_sha = self._run()
        self.addCleanup(temporary.cleanup)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line for line in result.stdout.splitlines() if line]
        self.assertEqual(len(lines), 5)
        self.assertTrue(all("source=" + source_sha in line for line in lines))
        by_arm = {line.split()[0].split("=", 1)[1]: line for line in lines}
        self.assertEqual(set(by_arm), {"A0", "A1", "A2", "A3", "A5"})
        self.assertIn("bias=0.001", by_arm["A1"])
        self.assertIn("router=1", by_arm["A2"])
        self.assertIn("selection=ESFT-Gate", by_arm["A3"])
        self.assertIn("expert_mode=full", by_arm["A3"])
        self.assertIn("speech_blocks=1", by_arm["A5"])
        self.assertIn("speech_ln=1", by_arm["A5"])

    def test_strict_manifest_digest_and_parent_symlink_fail_closed(self) -> None:
        for variable in (
            "DEVELOPMENT_SPLIT_MANIFEST",
            "DEVELOPMENT_SPEECH_SOURCE_SHA256",
        ):
            with self.subTest(variable=variable):
                temporary, result, _ = self._run(drop=(variable,))
                self.addCleanup(temporary.cleanup)
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"{variable} is required", result.stderr)
        temporary, result, _ = self._run(parent_symlink=True)
        self.addCleanup(temporary.cleanup)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsafe DATA_DIR symlink component", result.stderr)

    def test_a4_lora_fallback_fails_closed(self) -> None:
        temporary, result, _ = self._run(only="A4")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(result.returncode, 2)
        self.assertIn("A4 expert LoRA is unavailable and fails closed", result.stderr)

    def test_selection_path_rejects_sealed_evidence(self) -> None:
        temporary, result, _ = self._run(only="A3", selection_name="sealed-selection.json")
        self.addCleanup(temporary.cleanup)
        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing sealed/synthetic path", result.stderr)


if __name__ == "__main__":
    unittest.main()
