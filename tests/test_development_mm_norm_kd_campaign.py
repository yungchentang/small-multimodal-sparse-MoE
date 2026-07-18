"""Integration tests for the strict dual-initializer MM NORM+KD launcher."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from PIL import Image

from scripts.audit_requirements import _development_image_pixel_sha256, sha256_file
from scripts.development_split_provenance import (
    DATA_CONTRACT_ARTIFACTS,
    DATA_CONTRACT_POLICY,
    DATA_CONTRACT_SNAPSHOT_SEMANTICS,
    annotate_speech_partition_rows,
    speech_partition_commitments,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "submit_development_mm_norm_kd_campaign.sh"


def write_pcm_wav(path: Path, sample: int = 0) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(int(sample).to_bytes(2, "little", signed=True))


class DevelopmentMMNormKDCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        scripts = self.repo / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(LAUNCHER, scripts / LAUNCHER.name)
        shutil.copy2(
            ROOT / "scripts" / "materialize_eval_splits.py",
            scripts / "materialize_eval_splits.py",
        )
        (self.repo / ".gitignore").write_text(
            "development_splits/\nreal_development_data/image_captions.jsonl\nreal_development_data/speech_transcripts.jsonl\n"
            "real_development_data/*.wav\n",
            encoding="utf-8",
        )
        self.data = self.repo / "real_development_data"
        self.data.mkdir()
        (self.data / "manifest.json").write_text(
            json.dumps({"dataset": "test"}), encoding="utf-8"
        )
        for filename, row in (
            ("text_tasks.jsonl", {"id": "prefix-0"}),
            ("text_blocks_train.jsonl", {"id": "train-0"}),
            ("text_blocks_eval.jsonl", {"id": "eval-0"}),
            ("image_captions.jsonl", {"id": "image-placeholder"}),
            ("speech_transcripts.jsonl", {"id": "speech-placeholder"}),
        ):
            (self.data / filename).write_text(
                json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
            )
        self.stage_b = self.repo / "stage_b.pt"
        self.baseline_checkpoint = self.repo / "image_linear.pt"
        self.norm_checkpoint = self.repo / "image_norm50.pt"
        self.speech_checkpoint = self.repo / "speech_last1_ln.pt"
        self.baseline_manifest = self.repo / "image_linear_manifest.json"
        self.norm_manifest = self.repo / "image_norm50_manifest.json"
        self.speech_manifest = self.repo / "speech_manifest.json"
        self.development_split_manifest = (
            self.repo / "development_splits" / "manifest.json"
        )
        self.stage_b.write_bytes(b"stage-b")
        self.baseline_checkpoint.write_bytes(b"image-linear")
        self.norm_checkpoint.write_bytes(b"image-norm50")
        self.speech_checkpoint.write_bytes(b"speech-last1-ln")
        self.write_manifest("baseline")
        self.write_manifest("norm")
        self.write_manifest("speech")
        subprocess.run(["git", "init"], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "MM NORM KD Test"],
            cwd=self.repo,
            check=True,
        )
        self.commit_fixture("Initial fixture")
        self.write_development_split_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def manifest_payload(self, profile: str) -> dict:
        checkpoints = {
            "baseline": self.baseline_checkpoint,
            "norm": self.norm_checkpoint,
            "speech": self.speech_checkpoint,
        }
        checkpoint = checkpoints[profile]
        scope = "speech" if profile == "speech" else "image"
        args = {
            "final_steps": 500,
            "alignment_pretrain_steps": 400,
            "alignment_pretrain_modalities": scope,
        }
        if profile in {"baseline", "norm"}:
            args.update(
                image_bridge_type=(
                    "linear_projector" if profile == "baseline"
                    else "linear_projector_norm"
                ),
                image_prefix_tokens=50,
            )
        else:
            args.update(
                speech_unfreeze_last_blocks=1,
                speech_unfreeze_layer_norm=True,
                audio_bridge_type="attention_pool",
                audio_prefix_tokens=64,
            )
        provenance = {
            "source_commit_sha": "a" * 40,
            "runai_job_name": f"stage-a-{profile}",
            "runai_project": "test-project",
            "resolved_data_root": str(self.data.resolve()),
            "resolved_output_root": str((self.repo / f"stage_a_{profile}").resolve()),
            "final_main_steps": 500,
            "alignment_pretrain_steps": 400,
            "checkpoint_completed_step": 500,
            "policy": "development_only_stage_a_multimodal_initialization",
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
        }
        return {
            "source_commit_sha": provenance["source_commit_sha"],
            "runai_job_name": provenance["runai_job_name"],
            "runai_project": provenance["runai_project"],
            "args": args,
            "run_provenance": provenance,
            "completion": {
                "status": "completed",
                "e3_checkpoint_path": str(checkpoint.resolve()),
                "e3_checkpoint_sha256": self.digest(checkpoint),
                "e3_steps": 500,
            },
        }

    def write_manifest(self, profile: str) -> None:
        paths = {
            "baseline": self.baseline_manifest,
            "norm": self.norm_manifest,
            "speech": self.speech_manifest,
        }
        path = paths[profile]
        path.write_text(
            json.dumps(self.manifest_payload(profile), sort_keys=True),
            encoding="utf-8",
        )

    def write_development_split_manifest(self) -> None:
        root = self.development_split_manifest.parent
        root.mkdir(parents=True, exist_ok=True)
        image_records = {}
        for split, color in (
            ("train", (255, 0, 0)),
            ("dev", (0, 255, 0)),
            ("eval", (0, 0, 255)),
        ):
            path = root / f"{split}.png"
            Image.new("RGB", (2, 2), color).save(path)
            image_records[split] = {
                "path": str(path.resolve()),
                "media_sha256": sha256_file(path),
                "content_sha256": _development_image_pixel_sha256(path),
            }
        counts = {"train": 5000, "dev": 137, "eval": 113}
        rows = {}
        for split, count in counts.items():
            image_record = image_records[split]
            rows[f"image_{split}"] = [
                {
                    "id": f"image-{split}-{index}",
                    "partition": split,
                    "task": "image",
                    "caption": f"caption {index}",
                    "image_path": image_record["path"],
                    "media_sha256": image_record["media_sha256"],
                    "content_sha256": image_record["content_sha256"],
                    "resized_content_sha256": image_record["content_sha256"],
                }
                for index in range(count)
            ]
            audio_path = self.data / f"{split}.wav"
            write_pcm_wav(audio_path, len(rows))
            rows[f"speech_{split}"] = [
                {
                    "id": f"speech-{split}-{index}",
                    "partition": split,
                    "task": "speech",
                    "transcript": f"transcript {index}",
                    "audio_path": str(audio_path.resolve()),
                    "audio_sha256": sha256_file(audio_path),
                    "source_dataset": "openslr/librispeech_asr",
                    "speaker_id": f"{split}-speaker",
                }
                for index in range(count)
            ]
        raw_speech_rows = [
            row
            for split in ("train", "dev", "eval")
            for row in rows[f"speech_{split}"]
        ]
        speech_source_path = self.data / "speech_transcripts.jsonl"
        speech_source_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in raw_speech_rows
            ),
            encoding="utf-8",
        )
        speech_commitments = speech_partition_commitments(
            raw_speech_rows,
            source_path=str(speech_source_path.resolve()),
            source_sha256=sha256_file(speech_source_path),
        )
        files = {}
        for name, values in rows.items():
            modality, split = name.split("_", 1)
            output_values = (
                annotate_speech_partition_rows(values, split)
                if modality == "speech"
                else values
            )
            path = root / f"{name}.jsonl"
            path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in output_values
                ),
                encoding="utf-8",
            )
            files[name] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": len(output_values),
            }
            if modality == "speech":
                files[name]["source_partition_membership_sha256"] = (
                    speech_commitments["partitions"][split][
                        "membership_root_sha256"
                    ]
                )
        image_source_path = self.data / "image_captions.jsonl"
        image_source_rows = [
            row
            for split in ("train", "dev", "eval")
            for row in rows[f"image_{split}"]
        ]
        image_source_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in image_source_rows
            ),
            encoding="utf-8",
        )
        source_files = {
            "image": {
                "path": str(image_source_path.resolve()),
                "sha256": sha256_file(image_source_path),
                "bytes": image_source_path.stat().st_size,
                "rows": len(image_source_rows),
            },
            "speech": {
                "path": str(speech_source_path.resolve()),
                "sha256": sha256_file(speech_source_path),
                "bytes": speech_source_path.stat().st_size,
                "rows": len(raw_speech_rows),
            },
        }
        artifact_records = {}
        for artifact_name, (filename, is_jsonl) in DATA_CONTRACT_ARTIFACTS.items():
            artifact_path = self.data / filename
            record = {
                "path": str(artifact_path.resolve()),
                "sha256": sha256_file(artifact_path),
                "bytes": artifact_path.stat().st_size,
            }
            if is_jsonl:
                record["rows"] = len(
                    artifact_path.read_text(encoding="utf-8").splitlines()
                )
            artifact_records[artifact_name] = record
        builder_path = self.repo / "scripts" / "materialize_eval_splits.py"
        manifest = {
            "schema_version": 3,
            "data_dir": str(self.data.resolve()),
            "builder": {
                "path": str(builder_path.resolve()),
                "sha256": sha256_file(builder_path),
                "source_commit_sha": self.source_sha,
                "source_matches_commit": True,
                "command": "python scripts/materialize_eval_splits.py",
            },
            "real_subset": True,
            "synthetic_evidence_used": False,
            "sealed_data_used": False,
            "dev_count": 137,
            "eval_count": 113,
            "split_policy": {
                "image": "seeded_exact_image_content_disjoint_v1",
                "speech": "explicit_source_partition",
            },
            "counts": {name: len(values) for name, values in rows.items()},
            "image_content_partition": {
                "policy": "seeded_exact_image_content_disjoint_v1",
                "seed": 42,
                "group_key": "content_sha256_of_decoded_resized_rgb_pixels",
                "row_counts": counts,
                "group_counts": {"train": 1, "dev": 1, "eval": 1},
                "pairwise_group_overlaps": {
                    "train_dev": [],
                    "train_eval": [],
                    "dev_eval": [],
                },
                "pairwise_group_overlap_count": 0,
                "source_group_count": 3,
            },
            "speech_group_key": ["source_dataset", "speaker_id"],
            "speech_group_overlap": {
                "train_dev": [],
                "train_eval": [],
                "dev_eval": [],
            },
            "files": files,
            "data_contract": {
                "policy": DATA_CONTRACT_POLICY,
                "data_dir": str(self.data.resolve()),
                "snapshot_semantics": DATA_CONTRACT_SNAPSHOT_SEMANTICS,
                "artifacts": artifact_records,
            },
            "source_files": source_files,
            "speech_partition_commitments": speech_commitments,
        }
        self.development_split_manifest.write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )

    def commit_fixture(self, message: str) -> None:
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.repo,
            check=True,
            capture_output=True,
        )
        self.source_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True
        ).strip()
        if self.development_split_manifest.is_file():
            payload = json.loads(
                self.development_split_manifest.read_text(encoding="utf-8")
            )
            payload["builder"]["source_commit_sha"] = self.source_sha
            self.development_split_manifest.write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )

    def run_launcher(self, **updates: str) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "SOURCE_COMMIT_SHA": self.source_sha,
            "DATA_DIR": str(self.data),
            "DEVELOPMENT_SPLIT_MANIFEST": str(
                self.development_split_manifest.resolve()
            ),
            "DEVELOPMENT_SPEECH_SOURCE_SHA256": self.digest(
                self.data / "speech_transcripts.jsonl"
            ),
            "PYTHONPATH": str(ROOT),
            "BASE_OUT": str(self.repo / "outputs"),
            "STAGE_B_CHECKPOINT": str(self.stage_b.resolve()),
            "STAGE_B_CHECKPOINT_SHA256": self.digest(self.stage_b),
            "BASELINE_LINEAR_INITIAL_CHECKPOINT": str(self.baseline_checkpoint.resolve()),
            "BASELINE_LINEAR_INITIAL_CHECKPOINT_SHA256": self.digest(
                self.baseline_checkpoint
            ),
            "BASELINE_LINEAR_INITIAL_MANIFEST": str(self.baseline_manifest.resolve()),
            "BASELINE_LINEAR_INITIAL_MANIFEST_SHA256": self.digest(
                self.baseline_manifest
            ),
            "NORM_IMAGE_INITIAL_CHECKPOINT": str(self.norm_checkpoint.resolve()),
            "NORM_IMAGE_INITIAL_CHECKPOINT_SHA256": self.digest(
                self.norm_checkpoint
            ),
            "NORM_IMAGE_INITIAL_MANIFEST": str(self.norm_manifest.resolve()),
            "NORM_IMAGE_INITIAL_MANIFEST_SHA256": self.digest(self.norm_manifest),
            "SPEECH_INITIAL_CHECKPOINT": str(self.speech_checkpoint.resolve()),
            "SPEECH_INITIAL_CHECKPOINT_SHA256": self.digest(
                self.speech_checkpoint
            ),
            "SPEECH_INITIAL_MANIFEST": str(self.speech_manifest.resolve()),
            "SPEECH_INITIAL_MANIFEST_SHA256": self.digest(self.speech_manifest),
            "DRY_RUN": "1",
            "STAMP": "test",
        }
        env.update(updates)
        return subprocess.run(
            ["bash", str(self.repo / "scripts" / LAUNCHER.name)],
            cwd=self.repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_only_dry_run_selects_exact_matched_initializers_per_arm(self) -> None:
        expected = {
            "C0": ("baseline", "linear_projector", "none", "0"),
            "C_IMAGE_NORM_ONLY": (
                "norm",
                "linear_projector_norm",
                "none",
                "0",
            ),
            "C_SPEECH_INIT_ONLY": (
                "baseline",
                "linear_projector",
                "verified_last1_ln",
                "0",
            ),
            "C_DUAL": ("norm", "linear_projector_norm", "verified_last1_ln", "0"),
            "C_DUAL_KD025": (
                "norm",
                "linear_projector_norm",
                "verified_last1_ln",
                "0.25",
            ),
        }
        for arm, (image_kind, bridge, speech_kind, kd_coef) in expected.items():
            with self.subTest(arm=arm):
                result = self.run_launcher(ONLY=arm)
                self.assertEqual(result.returncode, 0, result.stderr)
                lines = result.stdout.splitlines()
                self.assertEqual(len(lines), 1)
                line = lines[0]
                self.assertIn(f"arm={arm} ", line)
                self.assertIn(f"image_initializer={image_kind} ", line)
                self.assertIn(f"image_bridge={bridge} ", line)
                self.assertIn(f"speech_initializer={speech_kind} ", line)
                self.assertIn(f"kd_coef={kd_coef} ", line)
                image_checkpoint = (
                    self.baseline_checkpoint if image_kind == "baseline"
                    else self.norm_checkpoint
                )
                self.assertIn(f"image_checkpoint={image_checkpoint.resolve()} ", line)
                expected_speech = (
                    self.speech_checkpoint.resolve() if speech_kind != "none"
                    else "none"
                )
                self.assertIn(f"speech_checkpoint={expected_speech} ", line)
                self.assertIn("gpu=1 top_k=2", line)
                self.assertIn("alignment=speech:400 main=500", line)
                self.assertIn(
                    f"development_split_manifest={self.development_split_manifest.resolve()}",
                    line,
                )
                self.assertIn("modality_cycle=text,image,speech", line)
                self.assertIn("router=0 experts=0 lm_head=0", line)
                self.assertIn("kd_temperature=1", line)

    def test_image_norm_only_does_not_require_speech_initializer(self) -> None:
        result = self.run_launcher(
            ONLY="C_IMAGE_NORM_ONLY",
            SPEECH_INITIAL_CHECKPOINT="",
            SPEECH_INITIAL_CHECKPOINT_SHA256="",
            SPEECH_INITIAL_MANIFEST="",
            SPEECH_INITIAL_MANIFEST_SHA256="",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertIn("speech_initializer=none", line)
        self.assertIn("speech_checkpoint=none", line)
        self.assertIn("gpu=1 top_k=2", line)

    def test_promotion_steps_are_explicit_and_fail_closed(self) -> None:
        promotion = self.run_launcher(
            ONLY="C_DUAL", FINAL_STEPS="3000", ALIGNMENT_PRETRAIN_STEPS="400"
        )
        self.assertEqual(promotion.returncode, 0, promotion.stderr)
        self.assertIn("alignment=speech:400 main=3000", promotion.stdout)

        invalid = self.run_launcher(ONLY="C_DUAL", FINAL_STEPS="499")
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("FINAL_STEPS must be an integer >= 500", invalid.stderr)

    def test_multimodal_cycle_is_explicit_and_fail_closed(self) -> None:
        multimodal = self.run_launcher(
            ONLY="C_DUAL", MODALITY_CYCLE="text,image,speech"
        )
        self.assertEqual(multimodal.returncode, 0, multimodal.stderr)
        self.assertIn("modality_cycle=text,image,speech", multimodal.stdout)

        missing_text = self.run_launcher(
            ONLY="C_DUAL", MODALITY_CYCLE="image,speech"
        )
        self.assertEqual(missing_text.returncode, 2)
        self.assertIn("MODALITY_CYCLE must include text", missing_text.stderr)

        invalid = self.run_launcher(
            ONLY="C_DUAL", MODALITY_CYCLE="text,image,video,speech"
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("invalid MODALITY_CYCLE entry: video", invalid.stderr)

    def test_image_objective_requires_image_cycle_and_batch_is_explicit(self) -> None:
        invalid = self.run_launcher(
            ONLY="C_DUAL",
            MODALITY_CYCLE="text,speech,speech",
            IMAGE_CONDITIONAL_RANKING_COEF="0.5",
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("image objective requires image in MODALITY_CYCLE", invalid.stderr)

        frozen_image = self.run_launcher(
            ONLY="C_DUAL",
            MODALITY_CYCLE="text,speech,speech",
            IMAGE_CONDITIONAL_RANKING_COEF="0.0",
            IMAGE_CONTRASTIVE_COEF="0.0",
            TRAIN_BATCH_SIZE="1",
        )
        self.assertEqual(frozen_image.returncode, 0, frozen_image.stderr)

        bad_batch = self.run_launcher(ONLY="C_DUAL", TRAIN_BATCH_SIZE="0")
        self.assertEqual(bad_batch.returncode, 2)
        self.assertIn("TRAIN_BATCH_SIZE must be a positive integer", bad_batch.stderr)

    def test_image_objective_screen_knobs_are_explicit_and_fail_closed(self) -> None:
        result = self.run_launcher(
            ONLY="C_DUAL",
            MODALITY_CYCLE="text,image,speech",
            CONDITIONAL_RANKING_NEGATIVE_MODE="hard_text",
            IMAGE_CONDITIONAL_RANKING_COEF="3.0",
            IMAGE_CONTRASTIVE_COEF="0.2",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("negative_mode=hard_text", result.stdout)
        self.assertIn("image_rank_coef=3.0", result.stdout)
        self.assertIn("image_contrastive_coef=0.2", result.stdout)

        invalid = self.run_launcher(
            ONLY="C_DUAL", CONDITIONAL_RANKING_NEGATIVE_MODE="oracle"
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("invalid CONDITIONAL_RANKING_NEGATIVE_MODE", invalid.stderr)

    def test_data_dir_parent_symlink_alias_is_rejected_before_realpath(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            alias_parent = Path(directory) / "alias-parent"
            alias_parent.symlink_to(self.repo, target_is_directory=True)
            result = self.run_launcher(
                DATA_DIR=str(alias_parent / self.data.name),
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsafe DATA_DIR symlink component", result.stderr)
        self.assertIn(str(alias_parent), result.stderr)

    def test_checkpoint_hash_and_manifest_scope_fail_closed(self) -> None:
        mismatch = self.run_launcher(STAGE_B_CHECKPOINT_SHA256="0" * 64)
        self.assertEqual(mismatch.returncode, 2)
        self.assertIn("Stage B checkpoint SHA-256 mismatch", mismatch.stderr)

        payload = self.manifest_payload("speech")
        payload["args"]["alignment_pretrain_modalities"] = "image"
        self.speech_manifest.write_text(json.dumps(payload), encoding="utf-8")
        self.commit_fixture("Cross-scope speech manifest")
        cross_scope = self.run_launcher()
        self.assertNotEqual(cross_scope.returncode, 0)
        self.assertIn(
            "Stage A speech manifest mismatch for alignment_pretrain_modalities",
            cross_scope.stderr,
        )

    def test_launcher_pins_exact_provenance_clean_a0_settings(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        required_tokens = (
            "CAPACITY_FACTOR=8.0 AUX_COEF=0.02",
            'IMAGE_CONDITIONAL_RANKING_COEF="$IMAGE_CONDITIONAL_RANKING_COEF"',
            'IMAGE_CONTRASTIVE_COEF="$IMAGE_CONTRASTIVE_COEF"',
            "SPEECH_CONTRASTIVE_COEF=0.2",
            "CONTRASTIVE_COEF=0.2",
            "CENTER_POSITIVE_WEIGHT=1.0",
            "RAW_POSITIVE_WEIGHT=0.0",
            "CONTRASTIVE_NEGATIVES=128",
            "IMAGE_CONTRASTIVE_NEGATIVES=-1",
            "SPEECH_CONTRASTIVE_NEGATIVES=-1",
            "IMAGE_CONTRASTIVE_TEMPERATURE=0.07",
            "SPEECH_CONTRASTIVE_TEMPERATURE=0.04",
            "IMAGE_CENTER_POSITIVE_WEIGHT=1.5",
            "IMAGE_RAW_POSITIVE_WEIGHT=0.05",
            "SPEECH_CENTER_POSITIVE_WEIGHT=5.0",
            "SPEECH_RAW_POSITIVE_WEIGHT=0.0",
            "RETRIEVAL_HEAD_LEARNING_RATE=0.0",
            "LM_HEAD_LEARNING_RATE=0.00001",
            "ROUTER_LEARNING_RATE=0.000002",
            "EXPERT_LEARNING_RATE=0.000001",
            "EXPERT_ANCHOR_COEFFICIENT=0.01",
            "EXPERT_UPDATE_MODE=full",
            "WEIGHT_DECAY=0.0",
            "GRAD_CLIP=5.0",
            "IMAGE_EVAL_SAMPLES=137",
            "SPEECH_EVAL_SAMPLES=137",
            "RETRIEVAL_EVAL_SAMPLES=137",
            "CONDITIONAL_EVAL_SAMPLES=137",
            "CONDITIONAL_NEGATIVES=9",
            "CONDITIONAL_BATCH_SIZE=1",
            "EVAL_BATCH_SIZE=1",
            "CONDITIONAL_RANKING_NEGATIVES=9",
            'CONDITIONAL_RANKING_NEGATIVE_MODE="$CONDITIONAL_RANKING_NEGATIVE_MODE"',
            "CONDITIONAL_RANKING_HARD_POOL_SIZE=512",
            "CONDITIONAL_RANKING_TEMPERATURE=0.7",
        )
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, launcher)
        default_line = next(
            line for line in launcher.splitlines() if line.startswith("ONLY_RAW=")
        )
        self.assertNotIn("C0,", default_line)

    def test_unknown_arm_and_plumbing_fail_closed(self) -> None:
        unknown = self.run_launcher(ONLY="C0,OTHER")
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("unsupported MM attribution arm", unknown.stderr)
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("TRAIN_ROUTER_GATES=0 TRAIN_EXPERTS=0 TRAIN_LM_HEAD=0", launcher)
        self.assertIn("MULTIMODAL_INITIALIZATION_SCOPE=image", launcher)
        self.assertIn("SPEECH_UNFREEZE_LAST_BLOCKS=1", launcher)
        self.assertIn("BASELINE_LINEAR_INITIAL_CHECKPOINT", launcher)
        self.assertIn("NORM_IMAGE_INITIAL_CHECKPOINT", launcher)
        run_sh = (ROOT / "run.sh").read_text(encoding="utf-8")
        submit = (ROOT / "scripts" / "submit_runai.sh").read_text(encoding="utf-8")
        for variable in (
            "DEVELOPMENT_SPLIT_MANIFEST",
            "SPEECH_INITIAL_CHECKPOINT",
            "SPEECH_INITIAL_CHECKPOINT_SHA256",
            "SPEECH_INITIAL_MANIFEST",
        ):
            self.assertIn(variable, run_sh)
            self.assertIn(variable, submit)


if __name__ == "__main__":
    unittest.main()
