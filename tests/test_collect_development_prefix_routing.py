import hashlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

from scripts import collect_development_prefix_routing as collector
from scripts import materialize_eval_splits as split_producer
from scripts import stage_b_checkpoint_provenance as stage_b_producer


REPO_ROOT = Path(__file__).resolve().parents[1]


class DevelopmentPrefixRoutingTest(unittest.TestCase):
    def test_rejects_sealed_and_synthetic_paths(self):
        for path in ("data/sealed_eval/image.jsonl", "outputs/synthetic-routing"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "development-only collector rejects"):
                    collector.reject_forbidden_paths([path])
        collector.reject_forbidden_paths(["data/development/image.jsonl"])

    def test_checkpoint_sha_is_required_and_must_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint-state")
            expected = hashlib.sha256(b"checkpoint-state").hexdigest()
            self.assertEqual(collector.verify_checkpoint_sha(checkpoint, expected), expected)
            with self.assertRaisesRegex(ValueError, "exact 64-hex"):
                collector.verify_checkpoint_sha(checkpoint, "")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                collector.verify_checkpoint_sha(checkpoint, "0" * 64)

    def test_train_dev_media_ids_must_be_disjoint_per_modality(self):
        train = [{"id": 1, "image_path": "train/a.png"}]
        dev = [{"id": 1, "image_path": "dev/a.png"}]
        with self.assertRaisesRegex(ValueError, "media ID overlap"):
            collector.assert_disjoint_media_ids(train, dev, "image")
        collector.assert_disjoint_media_ids(train, [{"id": 2}], "image")
        with self.assertRaisesRegex(ValueError, "media ID overlap"):
            collector.assert_disjoint_media_ids(
                [{"media_sha256": "abc"}], [{"content_sha256": "abc"}], "image"
            )

    def test_train_dev_media_ids_allow_duplicate_rows_within_split(self):
        train = [
            {"media_sha256": "same", "image_path": "train/a-1.png"},
            {"media_sha256": "same", "image_path": "train/a-2.png"},
        ]
        collector.assert_disjoint_media_ids(
            train,
            [{"media_sha256": "different", "image_path": "dev/b.png"}],
            "image",
        )

    def test_relative_media_path_resolves_from_verified_data_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_dir = root / "splits"
            data_dir = root / "data"
            split_dir.mkdir()
            (data_dir / "audio").mkdir(parents=True)
            audio = data_dir / "audio" / "sample.wav"
            audio.write_bytes(b"wave")
            rows = [{"audio_path": "audio/sample.wav"}]

            collector.resolve_media_paths(
                rows, split_dir / "speech_train.jsonl", "speech", data_dir
            )

            self.assertEqual(rows[0]["audio_path"], str(audio.resolve()))

    def test_checkpoint_must_declare_e3_top2_bridge_state(self):
        valid = {
            "last_row": {"experiment_id": collector.E3_EXPERIMENT_ID, "top_k": 2},
            "args": {},
            "image_resampler": {},
            "audio_resampler": {},
        }
        collector.assert_top2_checkpoint_state(valid)
        invalid = {**valid, "last_row": {**valid["last_row"], "top_k": 8}}
        with self.assertRaisesRegex(ValueError, "not Top-2"):
            collector.assert_top2_checkpoint_state(invalid)

    def test_e3_stage_b_source_path_sha_and_restore_flags_must_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_b = Path(temp_dir) / "stage_b.pt"
            stage_b.write_bytes(b"stage-b")
            digest = hashlib.sha256(b"stage-b").hexdigest()
            state = {
                "trainable_meta": {
                    "stage_b_initialization": {
                        "path": str(stage_b.resolve()),
                        "sha256": digest,
                        "policy": "development_only_stage_b_top8_to_top2_initialization",
                        "sealed_evidence_used": False,
                        "synthetic_evidence_used": False,
                        "state_restored": True,
                        "final_inference_top_k": 2,
                    }
                },
                "last_row": {
                    "stage_b_checkpoint_state_restored": True,
                    "source_stage_b_checkpoint_sha256": digest,
                },
                "run_provenance": {
                    "source_commit_sha": "a" * 40,
                    "runai_job_name": "e3-test",
                    "runai_project": "test-project",
                    "sealed_evidence_used": False,
                    "synthetic_evidence_used": False,
                },
            }
            provenance = collector.validate_e3_stage_b_source(
                state, stage_b, digest
            )
            self.assertEqual(provenance["recorded_sha256"], digest)

            mismatched = {
                **state,
                "last_row": {
                    **state["last_row"],
                    "source_stage_b_checkpoint_sha256": "0" * 64,
                },
            }
            with self.assertRaisesRegex(ValueError, "source SHA-256 mismatch"):
                collector.validate_e3_stage_b_source(mismatched, stage_b, digest)
            with self.assertRaisesRegex(ValueError, "source path mismatch"):
                collector.validate_e3_stage_b_source(
                    state, Path(temp_dir) / "other.pt", digest
                )

    def test_stage_b_is_restored_before_e3_adapter_overlay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            e3 = Path(temp_dir) / "e3.pt"
            e3.write_bytes(b"e3")
            e3_digest = hashlib.sha256(b"e3").hexdigest()
            stage_b = Path(temp_dir) / "stage_b.pt"
            stage_b.write_bytes(b"stage-b")
            digest = hashlib.sha256(b"stage-b").hexdigest()
            events = []
            e3_state = {
                "last_row": {
                    "experiment_id": collector.E3_EXPERIMENT_ID,
                    "top_k": 2,
                    "stage_b_checkpoint_state_restored": True,
                    "source_stage_b_checkpoint_sha256": digest,
                },
                "args": {},
                "image_resampler": {},
                "audio_resampler": {},
                "trainable_meta": {
                    "stage_b_initialization": {
                        "path": str(stage_b.resolve()),
                        "sha256": digest,
                        "policy": "development_only_stage_b_top8_to_top2_initialization",
                        "sealed_evidence_used": False,
                        "synthetic_evidence_used": False,
                        "state_restored": True,
                        "final_inference_top_k": 2,
                    }
                },
                "run_provenance": {
                    "source_commit_sha": "a" * 40,
                    "runai_job_name": "e3-test",
                    "runai_project": "test-project",
                    "sealed_evidence_used": False,
                    "synthetic_evidence_used": False,
                },
            }

            class FakeWrapper:
                def __init__(self, lm):
                    self.lm = lm

                def to(self, _device):
                    return self

                def eval(self):
                    return self

            lm = SimpleNamespace(
                model=SimpleNamespace(layers=[]),
                config=SimpleNamespace(),
                parameters=lambda: iter([SimpleNamespace(device="cpu")]),
            )
            wrapper = FakeWrapper(lm)
            stage_b_run_provenance = {
                "source_commit_sha": "a" * 40,
                "runai_job_name": "stage-b-test",
                "runai_project": "test-project",
                "producer_code": {
                    "path": "training/distill_olmoe_top2_real.py",
                    "sha256": "b" * 64,
                },
                "dataset_split_provenance": {"policy": "unit-test"},
                "sealed_evidence_used": False,
                "synthetic_evidence_used": False,
            }
            real_runs = types.ModuleType("training.olmoe_real_subset_runs")
            runtime_identity = {
                "model_name_or_path": "fake/base",
                "config_name_or_path": "fake/base",
                "revision": None,
                "commit_hash": "a" * 40,
                "config_sha256": "b" * 64,
                "architecture_sha256": "c" * 64,
                "pre_routing_state_sha256": "d" * 64,
            }
            real_runs.base_model_identity = lambda *_args: runtime_identity
            real_runs.load_stage_b_initialization_checkpoint = lambda *_args: (
                {"provenance": stage_b_run_provenance},
                {
                    "path": str(stage_b.resolve()),
                    "sha256": digest,
                    "size_bytes": stage_b.stat().st_size,
                },
            )
            def restore_stage_b(*args):
                self.assertEqual(args[3], runtime_identity)
                events.append("stage_b")
                return {}

            real_runs.restore_stage_b_student_initialization = restore_stage_b
            real_runs.restore_training_checkpoint = (
                lambda *_args, **_kwargs: events.append("e3")
            )
            real_runs.make_wrapper = lambda *_args: wrapper
            required_runs = types.ModuleType("training.olmoe_required_runs")

            def load_model(*_args, **kwargs):
                identity_fn = kwargs["pre_routing_identity_fn"]
                return (
                    lm,
                    object(),
                    {"pre_routing_model_identity": identity_fn(lm)},
                )

            required_runs.load_model = load_model
            required_runs.load_encoders = (
                lambda *_args: (object(), object(), object(), object())
            )
            required_runs.iter_olmoe_mlp_layers = lambda *_args: []
            with (
                mock.patch.object(
                    collector.torch, "load", return_value=e3_state, create=True
                ),
                mock.patch.object(
                    collector,
                    "load_stage_b_companion_manifest",
                    return_value={
                        "source_commit_sha": "a" * 40,
                        "run_provenance": stage_b_run_provenance,
                    },
                ),
                mock.patch.object(
                    collector,
                    "load_e3_checkpoint_manifest",
                    return_value={"source_commit_sha": "a" * 40},
                ),
                mock.patch.dict(
                    sys.modules,
                    {
                        "training.olmoe_real_subset_runs": real_runs,
                        "training.olmoe_required_runs": required_runs,
                    },
                ),
            ):
                result = collector.load_e3_wrapper(
                    e3,
                    e3_digest,
                    Path(temp_dir) / "e3_manifest.json",
                    "1" * 64,
                    None,
                    stage_b,
                    digest,
                    Path(temp_dir) / "stage_b_companion.json",
                    "2" * 64,
                    Path(temp_dir) / "split_manifest.json",
                    "3" * 64,
                    REPO_ROOT,
                )
            self.assertEqual(events, ["stage_b", "e3"])
            self.assertEqual(
                result[-3]["restoration_order"], ["stage_b_student", "e3_adapter"]
            )

    def test_hand_built_old_stage_b_companion_cannot_bless_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "stage_b.pt"
            checkpoint.write_bytes(b"stage-b")
            checkpoint_sha = collector.sha256_file(checkpoint)
            run_manifest = root / "manifest.json"
            run_manifest.write_text(
                json.dumps(
                    {
                        "data_policy": "development_only_real_manifests",
                        "final_inference_top_k": 2,
                        "args": {
                            "base_model": "unit-test-model",
                            "student_top_k": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            metrics = root / "metrics.json"
            metrics.write_text(
                json.dumps(
                    {
                        "checkpoint_provenance": {
                            "stage": "B",
                            "saved_checkpoint": str(checkpoint.resolve()),
                            "saved_checkpoint_sha256": checkpoint_sha,
                            "saved_checkpoint_size_bytes": checkpoint.stat().st_size,
                        }
                    }
                ),
                encoding="utf-8",
            )
            source_commit = subprocess.check_output(
                [
                    "git",
                    "-c",
                    f"safe.directory={REPO_ROOT}",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=REPO_ROOT,
                text=True,
            ).strip()
            code_bytes = subprocess.check_output(
                [
                    "git",
                    "-c",
                    f"safe.directory={REPO_ROOT}",
                    "show",
                    f"{source_commit}:training/distill_olmoe_top2_real.py",
                ],
                cwd=REPO_ROOT,
            )
            code_sha = hashlib.sha256(code_bytes).hexdigest()
            dataset_files = {}
            for role, split_name in (
                ("text_tasks", "development_calibration_source"),
                ("train", "train"),
                ("development_eval", "development_eval"),
            ):
                source_path = root / f"{role}.jsonl"
                source_path.write_text("{}\n", encoding="utf-8")
                dataset_files[role] = {
                    "path": str(source_path.resolve()),
                    "sha256": collector.sha256_file(source_path),
                    "size_bytes": source_path.stat().st_size,
                    "rows": 1,
                    "split": split_name,
                }
            dataset = {
                "policy": "development_only_real_manifests",
                "data_dir": str(root.resolve()),
                "files": dataset_files,
                "sealed_evidence_used": False,
                "synthetic_evidence_used": False,
            }
            run_provenance = {
                "source_commit_sha": source_commit,
                "runai_job_name": "stage-b-job",
                "runai_project": "project",
                "policy": "development_only_stage_b_top8_to_top2",
                "resolved_data_root": str(root.resolve()),
                "producer_code": {
                    "path": "training/distill_olmoe_top2_real.py",
                    "sha256": code_sha,
                },
                "dataset_split_provenance": dataset,
                "sealed_evidence_used": False,
                "synthetic_evidence_used": False,
            }
            run_manifest.write_text(
                json.dumps(
                    {
                        "source_commit_sha": source_commit,
                        "runai_job_name": "stage-b-job",
                        "runai_project": "project",
                        "run_provenance": run_provenance,
                        "dataset_split_provenance": dataset,
                        "data_policy": "development_only_real_manifests",
                        "final_inference_top_k": 2,
                        "args": {
                            "base_model": "unit-test-model",
                            "student_top_k": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            metrics.write_text(
                json.dumps(
                    {
                        "checkpoint_provenance": {
                            **run_provenance,
                            "stage": "B",
                            "saved_checkpoint": str(checkpoint.resolve()),
                            "saved_checkpoint_sha256": checkpoint_sha,
                            "saved_checkpoint_size_bytes": checkpoint.stat().st_size,
                        }
                    }
                ),
                encoding="utf-8",
            )
            companion = root / "stage_b_companion.json"
            payload = {
                "schema_version": 1,
                "artifact_type": "development_stage_b_checkpoint_companion",
                "development_only": True,
                "sealed_evidence_used": False,
                "synthetic_evidence_used": False,
                "source_commit_sha": source_commit,
                "runai_job_name": "stage-b-job",
                "runai_project": "project",
                "dataset_split_provenance": dataset,
                "checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "sha256": checkpoint_sha,
                    "size_bytes": checkpoint.stat().st_size,
                },
                "run_manifest": {
                    "path": str(run_manifest.resolve()),
                    "sha256": collector.sha256_file(run_manifest),
                },
                "metrics": {
                    "path": str(metrics.resolve()),
                    "sha256": collector.sha256_file(metrics),
                },
                "code": {
                    "path": "training/distill_olmoe_top2_real.py",
                    "sha256": code_sha,
                },
            }
            companion.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run_uuid"):
                collector.load_stage_b_companion_manifest(
                    companion,
                    collector.sha256_file(companion),
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_sha,
                    checkpoint_state={},
                    expected_base_model="unit-test-model",
                    repo_root=REPO_ROOT,
                )

    def test_stage_b_companion_real_producer_consumer_integration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_repo = root / "source_repo"
            training_source = (
                source_repo / "training" / "distill_olmoe_top2_real.py"
            )
            training_source.parent.mkdir(parents=True)
            training_source.write_bytes(
                (REPO_ROOT / "training" / "distill_olmoe_top2_real.py").read_bytes()
            )
            subprocess.run(["git", "init", "-q"], cwd=source_repo, check=True)
            subprocess.run(["git", "add", "."], cwd=source_repo, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Stage B Test",
                    "-c",
                    "user.email=stage-b@example.invalid",
                    "commit",
                    "-qm",
                    "Stage B producer fixture",
                ],
                cwd=source_repo,
                check=True,
            )
            source_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source_repo, text=True
            ).strip()

            data = root / "development_data"
            data.mkdir()
            source_files = {}
            for role, split_name in (
                ("text_tasks", "development_calibration_source"),
                ("train", "train"),
                ("development_eval", "development_eval"),
            ):
                source_path = data / f"{role}.jsonl"
                source_path.write_text("{}\n", encoding="utf-8")
                source_files[role] = {
                    "path": source_path,
                    "rows": 1,
                    "split": split_name,
                }
            run = stage_b_producer.build_stage_b_run_provenance(
                repo_root=source_repo,
                data_dir=data,
                source_files=source_files,
                environment={
                    "SOURCE_COMMIT_SHA": source_commit,
                    "RUNAI_JOB_NAME": "fresh-stage-b-job",
                    "RUNAI_PROJECT": "project",
                },
            )
            run_identity = stage_b_producer.build_checkpoint_run_identity(run)
            run_manifest = root / "manifest.json"
            run_manifest.write_text(
                json.dumps(
                    {
                        "source_commit_sha": source_commit,
                        "runai_job_name": "fresh-stage-b-job",
                        "runai_project": "project",
                        "run_identity": run_identity,
                        "run_provenance": run,
                        "dataset_split_provenance": run[
                            "dataset_split_provenance"
                        ],
                        "data_policy": "development_only_real_manifests",
                        "final_inference_top_k": 2,
                        "args": {
                            "base_model": "unit-test-model",
                            "student_top_k": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = root / "checkpoint_final.pt"
            checkpoint_state = {
                "checkpoint_version": 2,
                "final_inference_top_k": 2,
                "provenance": {**run, "stage": "B"},
                "run_identity": run_identity,
            }
            torch.save(checkpoint_state, checkpoint)
            checkpoint_sha = collector.sha256_file(checkpoint)
            metrics = root / "metrics.json"
            metrics.write_text(
                json.dumps(
                    {
                        "checkpoint_provenance": {
                            **run,
                            "stage": "B",
                            "saved_checkpoint": str(checkpoint.resolve()),
                            "saved_checkpoint_sha256": checkpoint_sha,
                            "saved_checkpoint_size_bytes": checkpoint.stat().st_size,
                        }
                    }
                ),
                encoding="utf-8",
            )
            companion = root / "stage_b_companion_manifest.json"
            produced = stage_b_producer.write_stage_b_companion(
                output_path=companion,
                checkpoint_path=checkpoint,
                metrics_path=metrics,
                run_manifest_path=run_manifest,
                repo_root=source_repo,
            )
            consumed = collector.load_stage_b_companion_manifest(
                companion,
                collector.sha256_file(companion),
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_state=checkpoint_state,
                expected_base_model="unit-test-model",
                repo_root=source_repo,
            )
            self.assertEqual(consumed["source_commit_sha"], source_commit)
            self.assertEqual(consumed["runai_job_name"], "fresh-stage-b-job")
            self.assertEqual(
                produced["dataset_split_provenance"],
                consumed["dataset_split_provenance"],
            )

            def bind_metrics(candidate: Path) -> None:
                candidate_sha = collector.sha256_file(candidate)
                metrics.write_text(
                    json.dumps(
                        {
                            "checkpoint_provenance": {
                                **run,
                                "stage": "B",
                                "saved_checkpoint": str(candidate.resolve()),
                                "saved_checkpoint_sha256": candidate_sha,
                                "saved_checkpoint_size_bytes": candidate.stat().st_size,
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            arbitrary = root / "arbitrary_checkpoint.pt"
            arbitrary.write_bytes(b"not-a-torch-checkpoint")
            bind_metrics(arbitrary)
            with self.assertRaisesRegex(ValueError, "cannot be safely loaded"):
                stage_b_producer.write_stage_b_companion(
                    output_path=root / "arbitrary_companion.json",
                    checkpoint_path=arbitrary,
                    metrics_path=metrics,
                    run_manifest_path=run_manifest,
                    repo_root=source_repo,
                )

            old_checkpoint = root / "old_checkpoint_without_identity.pt"
            torch.save(
                {
                    "checkpoint_version": 2,
                    "final_inference_top_k": 2,
                    "provenance": run,
                },
                old_checkpoint,
            )
            bind_metrics(old_checkpoint)
            with self.assertRaisesRegex(
                ValueError, "lacks matching internal run identity"
            ):
                stage_b_producer.write_stage_b_companion(
                    output_path=root / "old_checkpoint_companion.json",
                    checkpoint_path=old_checkpoint,
                    metrics_path=metrics,
                    run_manifest_path=run_manifest,
                    repo_root=source_repo,
                )

    def test_e3_manifest_binds_resolvable_source_job_stage_b_and_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e3 = root / "e3.pt"
            stage_b = root / "stage_b.pt"
            split = root / "split.json"
            e3.write_bytes(b"e3")
            stage_b.write_bytes(b"stage-b")
            split.write_text("{}", encoding="utf-8")
            split_sha = collector.sha256_file(split)
            e3_sha = collector.sha256_file(e3)
            stage_b_sha = collector.sha256_file(stage_b)
            source_commit = subprocess.check_output(
                [
                    "git",
                    "-c",
                    f"safe.directory={REPO_ROOT}",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=REPO_ROOT,
                text=True,
            ).strip()
            run = {
                "source_commit_sha": source_commit,
                "development_split_manifest_sha256": split_sha,
                "runai_job_name": "e3-job",
                "runai_project": "project",
                "sealed_evidence_used": False,
                "synthetic_evidence_used": False,
            }
            split_source = {
                "path": str(split.resolve()),
                "sha256": split_sha,
            }
            checkpoint_args = {
                "stage_b_checkpoint": str(stage_b.resolve()),
                "stage_b_checkpoint_sha256": stage_b_sha,
                "development_split_manifest": str(split.resolve()),
                "development_split_manifest_sha256": split_sha,
                "development_split_manifest_source": split_source,
            }
            payload = {
                "source_commit_sha": source_commit,
                "runai_job_name": "e3-job",
                "runai_project": "project",
                "run_provenance": run,
                "development_split_provenance": {
                    "manifest_path": str(split.resolve()),
                    "manifest_sha256": split_sha,
                },
                "completion": {
                    "status": "completed",
                    "e3_checkpoint_path": str(e3.resolve()),
                    "e3_checkpoint_sha256": e3_sha,
                    "e3_checkpoint_size_bytes": e3.stat().st_size,
                },
                "stage_b_initialization": {
                    "path": str(stage_b.resolve()),
                    "sha256": stage_b_sha,
                },
                "args": checkpoint_args,
            }
            manifest = root / "e3_manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            provenance = collector.load_e3_checkpoint_manifest(
                manifest,
                collector.sha256_file(manifest),
                checkpoint=e3,
                checkpoint_sha256=e3_sha,
                checkpoint_state={
                    "run_provenance": run,
                    "args": checkpoint_args,
                },
                stage_b_checkpoint=stage_b,
                stage_b_checkpoint_sha256=stage_b_sha,
                split_manifest=split,
                split_manifest_sha256=split_sha,
                repo_root=REPO_ROOT,
            )
            self.assertEqual(provenance["runai_job_name"], "e3-job")

            unresolved = "f" * 40
            payload["source_commit_sha"] = unresolved
            payload["run_provenance"] = {**run, "source_commit_sha": unresolved}
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot be resolved"):
                collector.load_e3_checkpoint_manifest(
                    manifest,
                    collector.sha256_file(manifest),
                    checkpoint=e3,
                    checkpoint_sha256=e3_sha,
                    checkpoint_state={
                        "run_provenance": payload["run_provenance"],
                        "args": checkpoint_args,
                    },
                    stage_b_checkpoint=stage_b,
                    stage_b_checkpoint_sha256=stage_b_sha,
                    split_manifest=split,
                    split_manifest_sha256=split_sha,
                    repo_root=REPO_ROOT,
                )

            payload["source_commit_sha"] = source_commit
            payload["run_provenance"] = run
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            bad_checkpoint_args = {
                **checkpoint_args,
                "development_split_manifest_source": {
                    "path": str(split.resolve()),
                    "sha256": "0" * 64,
                },
            }
            with self.assertRaisesRegex(
                ValueError, "checkpoint args strict split source mismatch"
            ):
                collector.load_e3_checkpoint_manifest(
                    manifest,
                    collector.sha256_file(manifest),
                    checkpoint=e3,
                    checkpoint_sha256=e3_sha,
                    checkpoint_state={
                        "run_provenance": run,
                        "args": bad_checkpoint_args,
                    },
                    stage_b_checkpoint=stage_b,
                    stage_b_checkpoint_sha256=stage_b_sha,
                    split_manifest=split,
                    split_manifest_sha256=split_sha,
                    repo_root=REPO_ROOT,
                )
            split.write_text('{"replacement": true}', encoding="utf-8")
            replacement_sha = collector.sha256_file(split)
            with self.assertRaisesRegex(
                ValueError, "E3 manifest strict split bytes mismatch"
            ):
                collector.load_e3_checkpoint_manifest(
                    manifest,
                    collector.sha256_file(manifest),
                    checkpoint=e3,
                    checkpoint_sha256=e3_sha,
                    checkpoint_state={
                        "run_provenance": run,
                        "args": checkpoint_args,
                    },
                    stage_b_checkpoint=stage_b,
                    stage_b_checkpoint_sha256=stage_b_sha,
                    split_manifest=split,
                    split_manifest_sha256=replacement_sha,
                    repo_root=REPO_ROOT,
                )

    def test_producer_manifest_release_v4_is_consumed_without_schema_guessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data"
            images = []
            for index in range(5):
                image_path = data / "images" / f"{index}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (4, 4), (index * 20, 10, 30)).save(image_path)
                images.append(
                    {
                        "id": index,
                        "task": "image",
                        "source": "real-coco",
                        "caption": f"caption {index}",
                        "image_path": str(image_path),
                    }
                )
            speech = []
            for split, count in (("train", 3), ("dev", 1), ("eval", 1)):
                for _index in range(count):
                    row_id = len(speech)
                    speech.append(
                        {
                            "id": row_id,
                            "task": "speech",
                            "source": "real-librispeech",
                            "source_dataset": "real-librispeech",
                            "partition": split,
                            "speaker_id": f"{split}-{row_id}",
                            "audio_sha256": hashlib.sha256(
                                f"audio-{row_id}".encode("utf-8")
                            ).hexdigest(),
                            "transcript": f"speech {row_id}",
                        }
                    )
            data.mkdir(parents=True, exist_ok=True)
            (data / "manifest.json").write_text("{}\n", encoding="utf-8")
            for filename in (
                "text_blocks_train.jsonl",
                "text_blocks_eval.jsonl",
                "text_tasks.jsonl",
            ):
                (data / filename).write_text('{"id": 0}\n', encoding="utf-8")
            (data / "image_captions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in images),
                encoding="utf-8",
            )
            (data / "speech_transcripts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in speech),
                encoding="utf-8",
            )
            output = root / "development_content_group_disjoint_splits_v4_test"
            produced = split_producer.materialize(
                data, output, dev_count=1, eval_count=1
            )
            manifest_path = output / "manifest.json"
            for key in ("image_dev", "image_eval", "speech_dev", "speech_eval"):
                Path(produced["files"][key]["path"]).unlink()
            payload, records = collector.load_strict_split_manifest(
                manifest_path,
                collector.sha256_file(manifest_path),
                collection_split="train",
                supplied_paths={
                    key: Path(produced["files"][key]["path"])
                    for key in ("image_train", "speech_train")
                },
                repo_root=REPO_ROOT,
            )
            self.assertEqual(set(records), {"image_train", "speech_train"})
            self.assertEqual(payload["schema_version"], produced["schema_version"])
            self.assertEqual(
                set(payload["_verified"]["unread_files"]),
                {"image_dev", "speech_dev", "image_eval", "speech_eval"},
            )
            for key, record in records.items():
                rows = collector.read_manifest(Path(record["path"]))
                collector.validate_strict_split_rows(rows, key)
            with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
                collector.load_strict_split_manifest(
                    manifest_path,
                    "0" * 64,
                    collection_split="train",
                    supplied_paths={
                        key: Path(produced["files"][key]["path"])
                        for key in ("image_train", "speech_train")
                    },
                    repo_root=REPO_ROOT,
                )
            alternate = root / "alternate_image_train.jsonl"
            alternate.write_text("{}\n", encoding="utf-8")
            mismatched_paths = {
                "image_train": alternate,
                "speech_train": Path(produced["files"]["speech_train"]["path"]),
            }
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                collector.load_strict_split_manifest(
                    manifest_path,
                    collector.sha256_file(manifest_path),
                    collection_split="train",
                    supplied_paths=mismatched_paths,
                    repo_root=REPO_ROOT,
                )

    def test_strict_rows_fail_closed_on_source_and_split(self):
        valid = {
            "task": "image",
            "source": "unit-test-source",
            "eval_split_name": "image_train",
        }
        collector.validate_strict_split_rows([valid], "image_train")
        with self.assertRaisesRegex(ValueError, "missing source"):
            collector.validate_strict_split_rows(
                [{**valid, "source": ""}], "image_train"
            )
        with self.assertRaisesRegex(ValueError, "split mismatch"):
            collector.validate_strict_split_rows(
                [{**valid, "eval_split_name": "image_dev"}], "image_train"
            )

    def test_canonical_per_layer_rows_and_conservation(self):
        logits0 = torch.tensor(
            [[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]], dtype=torch.float32
        )
        logits1 = torch.tensor(
            [[0.0, 4.0, 1.0, 2.0], [4.0, 0.0, 2.0, 1.0]], dtype=torch.float32
        )
        outputs = SimpleNamespace(router_logits=(logits0, logits1))
        rows = collector.canonical_layer_rows(
            outputs,
            modality="image_prefix",
            batch_size=1,
            prefix_tokens=2,
            num_experts=4,
        )
        self.assertEqual([row["layer"] for row in rows], [0, 1])
        for row in rows:
            self.assertEqual(row["top_k"], 2)
            self.assertEqual(row["token_count"], 2)
            self.assertEqual(sum(row["attempted_expert_counts"]), 4)
            self.assertEqual(len(row["gate_score_sums"]), 4)
            self.assertTrue(row["conservation_ok"])
            self.assertFalse(row["selection_bias_applied"])
            self.assertEqual(
                row["selection_accounting_source"], "router_logits_pre_capacity"
            )

        aggregate = collector.aggregate_layer_rows(rows + rows)
        outer = collector.build_outer_row(
            split="train",
            modality="image_prefix",
            sample_count=2,
            batch_count=2,
            layer_rows=aggregate,
            num_experts=4,
            accounting_sources=["router_logits_attempted_only"],
            source_manifest_key="image_train",
            source_manifest_sha256="1" * 64,
            strict_split_manifest_sha256="2" * 64,
        )
        self.assertTrue(outer["real_subset"])
        self.assertTrue(outer["prefix_routing_included"])
        self.assertTrue(outer["shared_olmoe_prefix_path"])
        self.assertEqual(outer["prefix_observed_assignments"], 16)
        self.assertEqual(outer["source_manifest_key"], "image_train")

    def test_dynamic_expert_bias_changes_attempted_top2_accounting(self):
        logits = torch.tensor([[5.0, 4.0, 0.0, 0.0]], dtype=torch.float32)
        outputs = SimpleNamespace(router_logits=(logits,))
        raw = collector.canonical_layer_rows(
            outputs,
            modality="audio_prefix",
            batch_size=1,
            prefix_tokens=1,
            num_experts=4,
        )[0]
        biased = collector.canonical_layer_rows(
            outputs,
            modality="audio_prefix",
            batch_size=1,
            prefix_tokens=1,
            num_experts=4,
            expert_biases=(torch.tensor([0.0, 0.0, 8.0, 0.0]),),
            normalize_topk_probs=(False,),
        )[0]
        self.assertEqual(raw["attempted_expert_counts"], [1, 1, 0, 0])
        self.assertEqual(biased["attempted_expert_counts"], [1, 0, 1, 0])
        self.assertTrue(biased["selection_bias_applied"])
        self.assertEqual(
            biased["selection_accounting_source"],
            "router_logits_plus_dynamic_expert_bias_pre_capacity",
        )

    def test_conservation_failure_is_rejected(self):
        row = {
            "layer": 0,
            "modality": "audio_prefix",
            "token_count": 2,
            "top_k": 2,
            "attempted_expert_counts": [1, 1, 1, 0],
            "gate_score_sums": [0.5, 0.5, 0.5, 0.0],
            "conservation_ok": False,
        }
        with self.assertRaisesRegex(ValueError, "conservation failure"):
            collector.validate_canonical_layer_row(row)

    def test_stage_b_launcher_invokes_fresh_companion_producer(self):
        training = (
            REPO_ROOT / "training" / "distill_olmoe_top2_real.py"
        ).read_text(encoding="utf-8")
        launcher = (
            REPO_ROOT / "scripts" / "submit_top2_distill_runai.sh"
        ).read_text(encoding="utf-8")
        for token in (
            "build_stage_b_run_provenance",
            "write_stage_b_companion",
            "stage_b_companion_manifest.json",
            'SOURCE_COMMIT_SHA="${SOURCE_COMMIT_SHA:?SOURCE_COMMIT_SHA is required}"',
            'export REPO_DIR="${REPO_DIR:-${REPO_ROOT}}"',
            "fresh Stage-B provenance requires a clean producer checkout",
        ):
            with self.subTest(token=token):
                self.assertTrue(token in training or token in launcher)

    def test_launcher_wires_clean_commit_single_gpu_and_all_inputs(self):
        launcher = (REPO_ROOT / "scripts" / "submit_development_prefix_routing.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            "SOURCE_COMMIT_SHA",
            "CHECKPOINT_MANIFEST",
            "EXPECTED_CHECKPOINT_MANIFEST_SHA256",
            "STAGE_B_CHECKPOINT",
            "EXPECTED_STAGE_B_CHECKPOINT_SHA256",
            "STAGE_B_COMPANION_MANIFEST",
            "EXPECTED_STAGE_B_COMPANION_MANIFEST_SHA256",
            "DEVELOPMENT_SPLIT_MANIFEST",
            "EXPECTED_DEVELOPMENT_SPLIT_MANIFEST_SHA256",
            "COLLECTION_SPLIT",
            "status --porcelain --untracked-files=all",
            "-g 1",
            "scripts.collect_development_prefix_routing",
            "--expected-checkpoint-sha256",
            "--checkpoint-manifest",
            "--expected-checkpoint-manifest-sha256",
            "--stage-b-checkpoint",
            "--expected-stage-b-checkpoint-sha256",
            "--stage-b-companion-manifest",
            "--expected-stage-b-companion-manifest-sha256",
            "--development-split-manifest",
            "--expected-development-split-manifest-sha256",
            "--collection-split",
            "--train-image-manifest",
            "--train-speech-manifest",
            "--dev-image-manifest",
            "--dev-speech-manifest",
            "--sample-count",
        ):
            with self.subTest(token=token):
                self.assertIn(token, launcher)
        self.assertNotIn('-g "${GPU', launcher)
        self.assertIn("routing_args=(", launcher)
        self.assertIn('"${routing_args[@]}" "${gamma_args[@]}"', launcher)
        self.assertIn('if [ "$COLLECTION_SPLIT" = "train-dev" ]', launcher)
        self.assertIn('python" -m scripts.collect_development_prefix_routing', launcher)
        self.assertNotIn("scripts/collect_development_prefix_routing.py", launcher)


if __name__ == "__main__":
    unittest.main()
