"""Tests for provenance-rich multimodal split materialization."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from scripts.development_split_provenance import verify_speech_partition_derivation


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_eval_splits", ROOT / "scripts" / "materialize_eval_splits.py"
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write_pcm_wav(path: Path, sample: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(int(sample).to_bytes(2, "little", signed=True))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.name == "speech_transcripts.jsonl":
        for index, row in enumerate(rows):
            audio_path = path.parent / "audio" / f"fixture_{index}.wav"
            write_pcm_wav(audio_path, index)
            row["audio_path"] = str(audio_path.resolve())
            row["audio_sha256"] = module.sha256_file(audio_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)


def write_stage_a_contract_files(data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    (data / "manifest.json").write_text(
        json.dumps({"dataset": "test"}) + "\n", encoding="utf-8"
    )
    for name in (
        "text_tasks.jsonl",
        "text_blocks_train.jsonl",
        "text_blocks_eval.jsonl",
    ):
        write_jsonl(data / name, [{"id": name, "text": name}])


def image_rows(data: Path, group_sizes: list[int]) -> list[dict]:
    rows: list[dict] = []
    for group_index, size in enumerate(group_sizes):
        original = data / "source_images" / f"group_{group_index}.png"
        write_image(original, (group_index * 31 % 255, group_index * 47 % 255, group_index * 59 % 255))
        for caption_index in range(size):
            copied = data / "images" / f"group_{group_index}_caption_{caption_index}.png"
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(original.read_bytes())
            rows.append(
                {
                    "id": len(rows),
                    "task": "image",
                    "caption": f"group {group_index} caption {caption_index}",
                    "image_path": str(copied),
                    "source": "real-coco",
                }
            )
    return rows


class MaterializeEvalSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        builder_path = ROOT / "scripts" / "materialize_eval_splits.py"
        self.builder_patch = patch.object(
            module,
            "builder_provenance",
            return_value={
                "path": str(builder_path.resolve()),
                "sha256": module.sha256_file(builder_path),
                "source_commit_sha": "a" * 40,
                "source_matches_commit": True,
                "command": "python scripts/materialize_eval_splits.py",
            },
        )
        self.builder_patch.start()
        self.addCleanup(self.builder_patch.stop)

    def test_named_speech_partitions_and_image_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            output = root / "splits"
            write_stage_a_contract_files(data)
            write_jsonl(
                data / "image_captions.jsonl",
                image_rows(data, [1] * 8),
            )
            speech = []
            for split, speakers in (("train", ("a", "b")), ("dev", ("c", "d")), ("eval", ("e", "f"))):
                for speaker in speakers:
                    speech.append(
                        {
                            "id": len(speech),
                            "partition": split,
                            "speaker_id": speaker,
                            "source_dataset": "real",
                            "transcript": speaker,
                        }
                    )
            write_jsonl(data / "speech_transcripts.jsonl", speech)
            manifest = module.materialize(data, output, dev_count=2, eval_count=2)
            self.assertEqual(manifest["split_policy"]["speech"], "explicit_source_partition")
            self.assertEqual(
                manifest["split_policy"]["image"],
                "seeded_exact_image_content_disjoint_v1",
            )
            self.assertEqual(
                manifest["image_content_partition"]["pairwise_group_overlap_count"], 0
            )
            self.assertEqual(manifest["counts"]["speech_train"], 2)
            self.assertEqual(manifest["counts"]["image_train"], 4)
            self.assertTrue(all(not values for values in manifest["speech_group_overlap"].values()))
            verification = verify_speech_partition_derivation(
                manifest,
                expected_partition_rows={"train": 2, "dev": 2, "eval": 2},
                observed_partition_rows={
                    split: [
                        json.loads(line)
                        for line in (
                            output / f"speech_{split}.jsonl"
                        ).read_text().splitlines()
                    ]
                    for split in ("train", "dev")
                },
            )
            self.assertTrue(verification["derivation_verified"])
            self.assertEqual(
                verification["exact_reproduction_splits"], ["dev", "train"]
            )
            self.assertFalse(verification["reserved_eval_split_file_opened"])
            self.assertTrue(
                verification["raw_source_eval_rows_read_for_partition_verification"]
            )
            self.assertEqual(
                [json.loads(line)["id"] for line in (output / "speech_dev.jsonl").read_text().splitlines()],
                [2, 3],
            )

    def test_image_content_groups_are_hashed_and_partitioned_without_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            write_stage_a_contract_files(data)
            write_jsonl(data / "image_captions.jsonl", image_rows(data, [3, 2, 2, 1]))
            speech = []
            for split, count in (("train", 5), ("dev", 2), ("eval", 1)):
                for _ in range(count):
                    speech.append(
                        {
                            "id": len(speech),
                            "partition": split,
                            "speaker_id": f"{split}-{len(speech)}",
                            "source_dataset": "real",
                            "transcript": split,
                        }
                    )
            write_jsonl(data / "speech_transcripts.jsonl", speech)
            output = root / "splits"
            manifest = module.materialize(
                data, output, dev_count=2, eval_count=1, image_split_seed=7
            )
            image_splits = {
                split: [
                    json.loads(line)
                    for line in (output / f"image_{split}.jsonl").read_text().splitlines()
                ]
                for split in ("train", "dev", "eval")
            }
            groups = {
                split: {row["content_sha256"] for row in rows}
                for split, rows in image_splits.items()
            }
            self.assertEqual({split: len(rows) for split, rows in image_splits.items()}, {
                "train": 5,
                "dev": 2,
                "eval": 1,
            })
            self.assertFalse(groups["train"] & groups["dev"])
            self.assertFalse(groups["train"] & groups["eval"])
            self.assertFalse(groups["dev"] & groups["eval"])
            self.assertTrue(
                all(
                    row["content_sha256"] == row["resized_content_sha256"]
                    and len(row["media_sha256"]) == 64
                    for rows in image_splits.values()
                    for row in rows
                )
            )
            self.assertEqual(
                manifest["image_content_partition"]["source_group_count"], 4
            )
            self.assertEqual(
                manifest["builder"]["sha256"],
                module.sha256_file(Path(manifest["builder"]["path"])),
            )
            self.assertEqual(len(manifest["builder"]["source_commit_sha"]), 40)

    def test_source_manifests_are_each_read_once_as_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            write_stage_a_contract_files(data)
            write_jsonl(data / "image_captions.jsonl", image_rows(data, [1] * 5))
            speech = [
                {
                    "id": index,
                    "partition": split,
                    "speaker_id": split,
                    "source_dataset": "real",
                    "transcript": split,
                }
                for index, split in enumerate(("train", "dev", "eval"))
            ]
            write_jsonl(data / "speech_transcripts.jsonl", speech)
            source_names = {
                filename
                for filename, _is_jsonl in module.DATA_CONTRACT_ARTIFACTS.values()
            }
            read_counts = {name: 0 for name in source_names}
            original_read_bytes = Path.read_bytes

            def counted_read_bytes(path: Path) -> bytes:
                if path.name in read_counts:
                    read_counts[path.name] += 1
                return original_read_bytes(path)

            with patch.object(
                Path,
                "read_bytes",
                autospec=True,
                side_effect=counted_read_bytes,
            ):
                manifest = module.materialize(
                    data, root / "out", dev_count=1, eval_count=1
                )

            self.assertEqual(
                read_counts,
                {name: 1 for name in source_names},
            )
            self.assertEqual(
                manifest["source_snapshot_policy"],
                module.SOURCE_SNAPSHOT_POLICY,
            )
            self.assertTrue(
                all(
                    record["snapshot_semantics"]
                    == "rows_and_sha256_derived_from_the_same_single_bytes_read"
                    for record in manifest["source_files"].values()
                )
            )
            self.assertEqual(
                set(manifest["data_contract"]["artifacts"]),
                set(module.DATA_CONTRACT_ARTIFACTS),
            )
            self.assertTrue(
                all(
                    record["bytes"] == Path(record["path"]).stat().st_size
                    for record in manifest["data_contract"]["artifacts"].values()
                )
            )

    def test_source_snapshot_survives_post_read_toctou_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            write_stage_a_contract_files(data)
            write_jsonl(data / "image_captions.jsonl", image_rows(data, [1] * 5))
            speech = [
                {
                    "id": index,
                    "partition": split,
                    "speaker_id": split,
                    "source_dataset": "real",
                    "transcript": split,
                }
                for index, split in enumerate(("train", "dev", "eval"))
            ]
            write_jsonl(data / "speech_transcripts.jsonl", speech)
            source_paths = {
                filename: data / filename
                for filename, _is_jsonl in module.DATA_CONTRACT_ARTIFACTS.values()
            }
            original_payloads = {
                name: path.read_bytes() for name, path in source_paths.items()
            }
            read_counts = {name: 0 for name in source_paths}
            original_read_bytes = Path.read_bytes

            def mutate_after_read(path: Path) -> bytes:
                payload = original_read_bytes(path)
                if path.name in source_paths:
                    read_counts[path.name] += 1
                    path.write_bytes(b"{}\n")
                return payload

            with patch.object(
                Path,
                "read_bytes",
                autospec=True,
                side_effect=mutate_after_read,
            ):
                manifest = module.materialize(
                    data, root / "out", dev_count=1, eval_count=1
                )

            self.assertEqual(
                read_counts,
                {name: 1 for name in source_paths},
            )
            for modality, source_name in (
                ("image", "image_captions.jsonl"),
                ("speech", "speech_transcripts.jsonl"),
            ):
                expected_sha = module.hashlib.sha256(
                    original_payloads[source_name]
                ).hexdigest()
                self.assertEqual(
                    manifest["source_files"][modality]["sha256"], expected_sha
                )
                self.assertNotEqual(
                    module.sha256_file(source_paths[source_name]), expected_sha
                )
            self.assertEqual(manifest["source_files"]["image"]["rows"], 5)
            self.assertEqual(manifest["source_files"]["speech"]["rows"], 3)
            self.assertEqual(
                manifest["speech_partition_commitments"]["source_sha256"],
                manifest["source_files"]["speech"]["sha256"],
            )

    def test_speaker_overlap_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            write_stage_a_contract_files(data)
            write_jsonl(data / "image_captions.jsonl", image_rows(data, [1] * 5))
            write_jsonl(
                data / "speech_transcripts.jsonl",
                [
                    {"partition": "train", "speaker_id": "same", "source_dataset": "real"},
                    {"partition": "dev", "speaker_id": "same", "source_dataset": "real"},
                    {"partition": "eval", "speaker_id": "other", "source_dataset": "real"},
                ],
            )
            with self.assertRaisesRegex(module.SplitError, "speaker overlap"):
                module.materialize(data, root / "out", dev_count=1, eval_count=1)

    def test_requires_canonical_manifest_and_rejects_artifact_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            with self.assertRaisesRegex(module.SplitError, "dataset_manifest"):
                module.materialize(data, root / "missing", dev_count=1, eval_count=1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            write_stage_a_contract_files(data)
            target = root / "text_tasks.jsonl"
            write_jsonl(target, [{"id": "target"}])
            (data / "text_tasks.jsonl").unlink()
            (data / "text_tasks.jsonl").symlink_to(target)
            with self.assertRaisesRegex(module.SplitError, "text_tasks.*non-symlink"):
                module.materialize(data, root / "symlink", dev_count=1, eval_count=1)

    def test_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            output = root / "out"
            output.mkdir()
            with self.assertRaisesRegex(module.SplitError, "existing output"):
                module.materialize(data, output, dev_count=1, eval_count=1)


if __name__ == "__main__":
    unittest.main()
