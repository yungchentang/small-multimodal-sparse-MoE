import json
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.development_split_provenance import (
    DATA_CONTRACT_ARTIFACTS,
    DATA_CONTRACT_POLICY,
    DATA_CONTRACT_SNAPSHOT_SEMANTICS,
    annotate_speech_partition_rows,
    speech_partition_commitments,
)
from training import olmoe_real_subset_runs as real


def write_pcm_wav(path: Path, sample: int = 0) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(int(sample).to_bytes(2, "little", signed=True))


REPO_ROOT = Path(__file__).resolve().parents[1]


def committed_builder_fixture(root: Path) -> tuple[Path, str]:
    repo = root / "builder_repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    builder = scripts / "materialize_eval_splits.py"
    shutil.copy2(REPO_ROOT / "scripts" / builder.name, builder)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Loader Test"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "add", "scripts/materialize_eval_splits.py"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Fixture builder"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    return builder.resolve(), commit


class ImageGroupNegativeSelectionTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "source": "coco",
                "source_image_id": 7,
                "image_path": "images/copy-a.png",
                "caption": "a red train beside a station",
            },
            {
                "source": "coco",
                "source_image_id": 7,
                "image_path": "images/copy-b.png",
                "caption": "a red train beside the same station",
            },
            {
                "source": "coco",
                "source_image_id": 8,
                "image_path": "images/b.png",
                "caption": "a blue boat on calm water",
            },
            {
                "source": "coco",
                "source_image_id": 9,
                "image_path": "images/c.png",
                "caption": "a green bicycle near a wall",
            },
            {
                "source": "coco",
                "source_image_id": 10,
                "image_path": "images/d.png",
                "caption": "a yellow bus on a road",
            },
        ]

    def test_all_image_ranking_modes_exclude_duplicate_image_rows(self):
        positive_group = real.image_group_id(self.records[0])
        self.assertEqual(positive_group, real.image_group_id(self.records[1]))

        for mode in ("hard_text", "stride", "random"):
            with self.subTest(mode=mode):
                candidates = real.local_candidate_indices(
                    self.records,
                    "caption",
                    0,
                    negatives=3,
                    mode=mode,
                    hard_pool_size=5,
                    modality="image",
                )
                self.assertEqual(candidates[0], 0)
                self.assertEqual(len(candidates), 4)
                self.assertTrue(
                    all(
                        real.image_group_id(self.records[index]) != positive_group
                        for index in candidates[1:]
                    )
                )

    def test_image_ranking_fails_without_enough_cross_group_rows(self):
        for mode in ("hard_text", "stride", "random"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(
                    ValueError, "Not enough cross-image-group negatives"
                ):
                    real.local_candidate_indices(
                        self.records[:3],
                        "caption",
                        0,
                        negatives=2,
                        mode=mode,
                        modality="image",
                    )


class DevelopmentPartitionLoaderTest(unittest.TestCase):
    def test_strict_development_manifest_requires_16000_runtime_sample_rate(
        self,
    ) -> None:
        rejected = SimpleNamespace(
            development_split_manifest="/tmp/development-split.json",
            sample_rate=8000,
        )
        with self.assertRaisesRegex(
            ValueError, "strict development split manifest requires --sample-rate=16000"
        ):
            real.validate_runtime_sample_rate(rejected)
        non_integer = SimpleNamespace(
            development_split_manifest="/tmp/development-split.json",
            sample_rate=16000.5,
        )
        with self.assertRaisesRegex(ValueError, "sample rate must be an integer"):
            real.validate_runtime_sample_rate(non_integer)

        accepted = SimpleNamespace(
            development_split_manifest="/tmp/development-split.json",
            sample_rate=16000,
        )
        self.assertEqual(real.validate_runtime_sample_rate(accepted), 16000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            split_manifest = root / "development_split.json"
            split_manifest.write_text("{}", encoding="utf-8")
            accepted.data_dir = str(root)
            accepted.output_dir = str(root / "output")
            accepted.final_steps = 1
            accepted.alignment_pretrain_steps = 0
            accepted.development_split_manifest = str(split_manifest)
            accepted.development_split_manifest_sha256 = real.sha256_file(
                split_manifest
            )
            with patch.dict(
                "os.environ",
                {
                    "SOURCE_COMMIT_SHA": "a" * 40,
                    "RUNAI_JOB_NAME": "strict-sample-rate-test",
                    "RUNAI_PROJECT": "test-project",
                },
            ):
                provenance = real.build_stage_a_run_provenance(accepted, 1)
            self.assertEqual(provenance["runtime_sample_rate"], 16000)

    def test_strict_runtime_data_dir_rejects_symlink_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = (root / "canonical").resolve()
            canonical.mkdir()
            alias = root / "alias"
            alias.symlink_to(canonical, target_is_directory=True)
            rejected = [
                str(alias),
                "~",
                f"{canonical}/.",
                f"{canonical}//",
            ]
            for value in rejected:
                with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "absolute canonical directory"
                ):
                    real.resolve_runtime_data_dir(
                        value,
                        strict_development_manifest=True,
                    )
            self.assertEqual(
                real.resolve_runtime_data_dir(
                    canonical,
                    strict_development_manifest=True,
                ),
                canonical,
            )

    def test_loader_uses_train_and_dev_while_proving_eval_is_reserved(self):
        counts = {
            "image_train": 2,
            "image_dev": 1,
            "image_eval": 1,
            "speech_train": 2,
            "speech_dev": 1,
            "speech_eval": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = {
                "image_train": [
                    {"content_sha256": "a" * 64, "caption": "a"},
                    {"content_sha256": "a" * 64, "caption": "b"},
                ],
                "image_dev": [{"content_sha256": "b" * 64, "caption": "c"}],
                "image_eval": [{"content_sha256": "c" * 64, "caption": "d"}],
                "speech_train": [
                    {
                        "id": "speech-train-0",
                        "partition": "train",
                        "source_dataset": "real",
                        "speaker_id": "train",
                        "transcript": "a",
                    },
                    {
                        "id": "speech-train-1",
                        "partition": "train",
                        "source_dataset": "real",
                        "speaker_id": "train",
                        "transcript": "b",
                    },
                ],
                "speech_dev": [
                    {
                        "id": "speech-dev-0",
                        "partition": "dev",
                        "source_dataset": "real",
                        "speaker_id": "dev",
                        "transcript": "c",
                    }
                ],
                "speech_eval": [
                    {
                        "id": "speech-eval-0",
                        "partition": "eval",
                        "source_dataset": "real",
                        "speaker_id": "eval",
                        "transcript": "d",
                    }
                ],
            }
            data_dir = (root / "data").resolve()
            data_dir.mkdir()
            audio_dir = data_dir / "audio"
            audio_dir.mkdir()
            for split in ("train", "dev", "eval"):
                for row in rows[f"speech_{split}"]:
                    audio_path = audio_dir / f"{row['id']}.wav"
                    write_pcm_wav(audio_path, len(row["id"]))
                    row["audio_path"] = str(audio_path.relative_to(data_dir))
                    row["audio_sha256"] = real.sha256_file(audio_path)
            speech_source_path = data_dir / "speech_transcripts.jsonl"
            speech_source_rows = [
                row
                for split in ("train", "dev", "eval")
                for row in rows[f"speech_{split}"]
            ]
            speech_source_path.write_text(
                "".join(json.dumps(row) + "\n" for row in speech_source_rows),
                encoding="utf-8",
            )
            image_source_path = data_dir / "image_captions.jsonl"
            image_source_rows = [
                row
                for split in ("train", "dev", "eval")
                for row in rows[f"image_{split}"]
            ]
            image_source_path.write_text(
                "".join(json.dumps(row) + "\n" for row in image_source_rows),
                encoding="utf-8",
            )
            (data_dir / "manifest.json").write_text(
                json.dumps({"dataset": "test"}) + "\n", encoding="utf-8"
            )
            for filename in (
                "text_tasks.jsonl",
                "text_blocks_train.jsonl",
                "text_blocks_eval.jsonl",
            ):
                (data_dir / filename).write_text(
                    json.dumps({"id": filename}) + "\n", encoding="utf-8"
                )
            speech_commitments = speech_partition_commitments(
                speech_source_rows,
                source_path=str(speech_source_path),
                source_sha256=real.sha256_file(speech_source_path),
            )
            files = {}
            for name, values in rows.items():
                path = root / f"{name}.jsonl"
                modality, split = name.split("_", 1)
                output_values = (
                    annotate_speech_partition_rows(values, split)
                    if modality == "speech"
                    else values
                )
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in output_values),
                    encoding="utf-8",
                )
                files[name] = {
                    "path": str(path.resolve()),
                    "sha256": real.sha256_file(path),
                    "rows": len(output_values),
                }
                if modality == "speech":
                    files[name]["source_partition_membership_sha256"] = (
                        speech_commitments["partitions"][split][
                            "membership_root_sha256"
                        ]
                    )
            builder_path, builder_commit = committed_builder_fixture(root)
            artifact_records = {}
            for artifact_name, (filename, is_jsonl) in DATA_CONTRACT_ARTIFACTS.items():
                artifact_path = data_dir / filename
                record = {
                    "path": str(artifact_path),
                    "sha256": real.sha256_file(artifact_path),
                    "bytes": artifact_path.stat().st_size,
                }
                if is_jsonl:
                    record["rows"] = len(
                        artifact_path.read_text(encoding="utf-8").splitlines()
                    )
                artifact_records[artifact_name] = record
            manifest = {
                "schema_version": 3,
                "data_dir": str(data_dir),
                "builder": {
                    "path": str(builder_path.resolve()),
                    "sha256": real.sha256_file(builder_path),
                    "source_commit_sha": builder_commit,
                    "source_matches_commit": True,
                    "command": "python scripts/materialize_eval_splits.py",
                },
                "real_subset": True,
                "synthetic_evidence_used": False,
                "sealed_data_used": False,
                "dev_count": counts["image_dev"],
                "eval_count": counts["image_eval"],
                "split_policy": {
                    "image": "seeded_exact_image_content_disjoint_v1",
                    "speech": "explicit_source_partition",
                },
                "speech_group_key": ["source_dataset", "speaker_id"],
                "speech_group_overlap": {
                    "train_dev": [],
                    "train_eval": [],
                    "dev_eval": [],
                },
                "counts": counts,
                "files": files,
                "speech_partition_commitments": speech_commitments,
                "data_contract": {
                    "policy": DATA_CONTRACT_POLICY,
                    "data_dir": str(data_dir),
                    "snapshot_semantics": DATA_CONTRACT_SNAPSHOT_SEMANTICS,
                    "artifacts": artifact_records,
                },
                "source_files": {
                    "image": {
                        "path": str(image_source_path),
                        "sha256": real.sha256_file(image_source_path),
                        "bytes": image_source_path.stat().st_size,
                        "rows": len(image_source_rows),
                    },
                    "speech": {
                        "path": str(speech_source_path.resolve()),
                        "sha256": real.sha256_file(speech_source_path),
                        "bytes": speech_source_path.stat().st_size,
                        "rows": len(speech_source_rows),
                    }
                },
                "image_content_partition": {
                    "policy": "seeded_exact_image_content_disjoint_v1",
                    "seed": 42,
                    "group_key": (
                        "content_sha256_of_decoded_resized_rgb_pixels"
                    ),
                    "row_counts": {
                        split: counts[f"image_{split}"]
                        for split in ("train", "dev", "eval")
                    },
                    "pairwise_group_overlap_count": 0,
                    "pairwise_group_overlaps": {
                        "train_dev": [],
                        "train_eval": [],
                        "dev_eval": [],
                    },
                    "group_counts": {"train": 1, "dev": 1, "eval": 1},
                    "source_group_count": 3,
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            original_counts = real.DEVELOPMENT_SPLIT_COUNTS
            real.DEVELOPMENT_SPLIT_COUNTS = counts
            reserved_paths = {
                root / "image_eval.jsonl",
                root / "speech_eval.jsonl",
            }
            original_open = Path.open

            original_read_bytes = Path.read_bytes
            read_counts = {}

            def counting_read_bytes(path):
                read_counts[path] = read_counts.get(path, 0) + 1
                return original_read_bytes(path)
            def guarded_open(path, *args, **kwargs):

                if path in reserved_paths:
                    raise AssertionError(f"reserved eval file opened: {path}")
                return original_open(path, *args, **kwargs)

            try:
                with (
                    patch.object(Path, "open", new=guarded_open),
                    patch.object(Path, "read_bytes", new=counting_read_bytes),
                ):
                    image_train, image_dev, speech_train, speech_dev, provenance = (
                        real.load_development_multimodal_partitions(
                            manifest_path,
                            expected_data_dir=data_dir,
                            expected_speech_source_sha256=real.sha256_file(
                                speech_source_path
                            ),
                        )
                    )
            finally:
                real.DEVELOPMENT_SPLIT_COUNTS = original_counts
            for snapshot_path in (
                manifest_path,
                *(data_dir / filename for filename, _ in DATA_CONTRACT_ARTIFACTS.values()),
                *(root / f"{name}.jsonl" for name in real.DEVELOPMENT_SELECTED_FILES),
            ):
                self.assertEqual(
                    read_counts.get(snapshot_path, 0),
                    1,
                    f"{snapshot_path} was not read as exactly one bytes snapshot",
                )
            for reserved_path in reserved_paths:
                self.assertEqual(read_counts.get(reserved_path, 0), 0)

            for row_index, row in enumerate((*speech_train, *speech_dev)):
                self.assertFalse(Path(row["audio_path"]).is_absolute())
                resolved_audio_path = real.resolve_speech_audio_path(
                    str(row["audio_path"]),
                    data_dir=data_dir,
                    row_index=row_index,
                )
                payload, digest = real.audio_file_snapshot(
                    str(resolved_audio_path),
                    row["audio_sha256"],
                    require_expected=True,
                )
                self.assertEqual(digest, real.sha256_file(resolved_audio_path))
                self.assertTrue(payload)
            self.assertEqual(
                real.speech_partition_record(
                    speech_train, "train", annotated=True
                ),
                speech_commitments["partitions"]["train"],
            )
            self.assertEqual(provenance["data_dir"], str(data_dir))
            self.assertEqual(
                provenance["speech_audio_verification"][
                    "audio_rows_verified"
                ],
                counts["speech_train"] + counts["speech_dev"],
            )

            modified_audio = real.resolve_speech_audio_path(
                str(speech_train[0]["audio_path"]),
                data_dir=data_dir,
                row_index=0,
            )
            original_audio = modified_audio.read_bytes()
            modified_audio.write_bytes(original_audio + b"modified")
            original_counts = real.DEVELOPMENT_SPLIT_COUNTS
            real.DEVELOPMENT_SPLIT_COUNTS = counts
            try:
                with self.assertRaisesRegex(ValueError, "audio SHA256 mismatch"):
                    real.load_development_multimodal_partitions(
                        manifest_path,
                        expected_data_dir=data_dir,
                        expected_speech_source_sha256=real.sha256_file(
                            speech_source_path
                        ),
                    )
            finally:
                modified_audio.write_bytes(original_audio)
                real.DEVELOPMENT_SPLIT_COUNTS = original_counts

            missing_trust = dict(provenance)
            missing_trust["trusted_digest_verified"] = False
            missing_trust["expected_speech_source_sha256"] = None
            original_counts = real.DEVELOPMENT_SPLIT_COUNTS
            real.DEVELOPMENT_SPLIT_COUNTS = counts
            try:
                with self.assertRaisesRegex(
                    ValueError, "externally trusted source digest"
                ):
                    real.validate_speech_shared_split_binding(missing_trust, 2)
            finally:
                real.DEVELOPMENT_SPLIT_COUNTS = original_counts

            original_counts = real.DEVELOPMENT_SPLIT_COUNTS
            real.DEVELOPMENT_SPLIT_COUNTS = counts
            try:
                with self.assertRaisesRegex(
                    ValueError, "trusted runtime digest"
                ):
                    real.load_development_multimodal_partitions(
                        manifest_path,
                        expected_data_dir=data_dir,
                        expected_speech_source_sha256="0" * 64,
                    )
            finally:
                real.DEVELOPMENT_SPLIT_COUNTS = original_counts

            alternate_data_dir = (root / "alternate_data").resolve()
            alternate_data_dir.mkdir()
            substituted_manifest = dict(manifest)
            substituted_manifest["data_dir"] = str(alternate_data_dir)
            substituted_path = root / "substituted_manifest.json"
            substituted_path.write_text(
                json.dumps(substituted_manifest), encoding="utf-8"
            )
            original_counts = real.DEVELOPMENT_SPLIT_COUNTS
            real.DEVELOPMENT_SPLIT_COUNTS = counts
            try:
                with self.assertRaisesRegex(
                    ValueError, "disagrees with runtime data_dir"
                ):
                    real.load_development_multimodal_partitions(
                        substituted_path,
                        expected_data_dir=data_dir,
                    )
            finally:
                real.DEVELOPMENT_SPLIT_COUNTS = original_counts

            self.assertEqual(len(image_train), 2)
            self.assertEqual(len(image_dev), 1)
            self.assertEqual(len(speech_train), 2)
            self.assertEqual(len(speech_dev), 1)
            self.assertEqual(provenance["selection_splits"], ["train", "dev"])
            self.assertEqual(provenance["reserved_unused_split"], "eval")
            self.assertEqual(
                provenance["reserved_unused_counts"], {"image": 1, "speech": 1}
            )
            self.assertTrue(provenance["strict_manifest_verified"])
            self.assertEqual(provenance["source_commit_sha"], builder_commit)
            self.assertEqual(provenance["builder"]["path"], str(builder_path.resolve()))
            self.assertEqual(
                provenance["builder"]["sha256"], real.sha256_file(builder_path)
            )
            self.assertTrue(provenance["builder"]["source_commit_exists"])
            self.assertTrue(provenance["builder"]["current_bytes_match_commit"])
            self.assertEqual(
                provenance["source_files"]["speech"]["sha256"],
                real.sha256_file(speech_source_path),
            )
            self.assertTrue(
                provenance["source_files"]["speech"][
                    "content_used_for_partition_verification"
                ]
            )
            self.assertTrue(
                provenance["source_files"]["speech"]["derivation_verified"]
            )
            self.assertTrue(
                provenance["source_files"]["speech"]["data_contract"][
                    "all_artifacts_verified_this_run"
                ]
            )
            self.assertFalse(
                provenance["source_files"]["speech"]["reserved_eval_split_file_opened"]
            )
            self.assertTrue(
                provenance["source_files"]["speech"][
                    "raw_source_eval_rows_read_for_partition_verification"
                ]
            )
            self.assertTrue(provenance["manifest_hash_and_parse_same_bytes"])
            self.assertTrue(provenance["trusted_digest_verified"])
            self.assertFalse(provenance["reserved_files_opened"])
            for name in ("image_eval", "speech_eval"):
                self.assertEqual(
                    provenance["files"][name]["read_status"], "reserved_unread"
                )
                self.assertFalse(provenance["files"][name]["content_opened"])
                self.assertFalse(
                    provenance["files"][name]["sha256_verified_this_run"]
                )


class PerModalityCursorTest(unittest.TestCase):
    def test_repeated_cycles_cover_each_dataset_without_fixed_residues(self):
        modality_cycle = ("text", "image", "speech")
        datasets = {
            modality: [{"row": index} for index in range(12)]
            for modality in modality_cycle
        }
        cursors = {modality: 0 for modality in modality_cycle}
        seen = {modality: set() for modality in modality_cycle}
        starts = {modality: [] for modality in modality_cycle}

        for _ in range(6):
            for modality in modality_cycle:
                indices, _rows, provenance = real.sample_next_modality_batch(
                    datasets[modality], modality, cursors, batch_size=2
                )
                seen[modality].update(indices)
                starts[modality].append(provenance["data_cursor_start"])

        for modality in modality_cycle:
            self.assertEqual(seen[modality], set(range(12)))
            self.assertEqual(starts[modality], [0, 2, 4, 6, 8, 10])
            self.assertEqual(cursors[modality], 12)


class FinalRoutingAggregateTest(unittest.TestCase):
    @staticmethod
    def row(modality, step):
        values = {
            "text": (10, 1.0, 0.1, 0.01),
            "image": (20, 2.0, 0.2, 0.02),
            "speech": (30, 4.0, 0.4, 0.04),
        }
        assignments, entropy, inactive, overflow = values[modality]
        return {
            "step": step,
            "modality": modality,
            "routing_attempted_assignments_total": assignments,
            "gate_entropy_mean": entropy,
            "inactive_expert_ratio_mean": inactive,
            "capacity_overflow_ratio_mean": overflow,
            "dynamic_expert_bias_inactive_proxy": inactive / 2.0,
            "dynamic_expert_bias_overflow_proxy": overflow / 2.0,
        }

    def test_aggregate_is_invariant_to_last_modality(self):
        first_cycle = ("text", "image", "speech")
        second_cycle = ("speech", "text", "image")
        first = real.assignment_weighted_final_cycle_summary(
            [self.row(modality, step) for step, modality in enumerate(first_cycle, 1)],
            first_cycle,
        )
        second = real.assignment_weighted_final_cycle_summary(
            [self.row(modality, step) for step, modality in enumerate(second_cycle, 1)],
            second_cycle,
        )

        self.assertEqual(first["overall"], second["overall"])
        self.assertEqual(first["by_modality"], second["by_modality"])
        self.assertAlmostEqual(
            first["overall"]["gate_entropy_mean_assignment_weighted"],
            (10.0 + 40.0 + 120.0) / 60.0,
        )
        self.assertEqual(
            first["overall"]["attempted_assignment_weight_total"],
            60,
        )


if __name__ == "__main__":
    unittest.main()
