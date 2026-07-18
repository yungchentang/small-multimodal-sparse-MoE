"""Temporary-fixture tests for development routing split materialization."""

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_development_routing_splits.py"
SPEC = importlib.util.spec_from_file_location("build_development_routing_splits", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_wav(path: Path, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack("<h", 0) * 16)


class DevelopmentRoutingSplitTests(unittest.TestCase):
    def make_fixture(
        self, count: int = 5
    ) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        source = root / "data" / "clean_subset"
        output = root / "outputs" / "routing"
        (source / "images").mkdir(parents=True)
        (source / "audio").mkdir()
        image_rows = []
        speech_rows = []
        for index in range(count):
            image = source / "images" / f"image_{index}.png"
            audio = source / "audio" / f"audio_{index}.wav"
            image.write_bytes(b"png" + bytes([index]))
            write_wav(audio)
            image_rows.append(
                {
                    "id": index,
                    "task": "image",
                    "image_path": f"images/{image.name}",
                    "caption": f"caption {index}",
                    "source": "public-image:train",
                    "preprocess": {"resize": [224, 224], "custom": index},
                }
            )
            speech_rows.append(
                {
                    "id": index,
                    "task": "speech",
                    "audio_path": f"audio/{audio.name}",
                    "transcript": f"transcript {index}",
                    "sample_rate": 16_000,
                    "source": "public-speech:train",
                    "preprocess": {"resampled_to": 16_000, "custom": index},
                }
            )
        write_jsonl(source / "image_captions.jsonl", image_rows)
        write_jsonl(source / "speech_transcripts.jsonl", speech_rows)
        return temporary, source, output

    def test_builds_deterministic_splits_and_manifest_without_semantic_changes(self) -> None:
        temporary, source, output = self.make_fixture()
        with temporary:
            original_images = read_jsonl(source / "image_captions.jsonl")
            original_speech = read_jsonl(source / "speech_transcripts.jsonl")
            manifest = builder.build_splits(source, output, train_count=3, dev_count=2)

            self.assertEqual([row["id"] for row in read_jsonl(output / "image_train.jsonl")], [0, 1, 2])
            self.assertEqual([row["id"] for row in read_jsonl(output / "image_dev.jsonl")], [3, 4])
            self.assertEqual([row["id"] for row in read_jsonl(output / "speech_train.jsonl")], [0, 1, 2])
            self.assertEqual([row["id"] for row in read_jsonl(output / "speech_dev.jsonl")], [3, 4])
            reconstruction = read_jsonl(output / "reconstruction_dev.jsonl")
            self.assertEqual(len(reconstruction), 4)
            self.assertEqual(
                [row["text"] for row in reconstruction],
                ["caption 3", "caption 4", "transcript 3", "transcript 4"],
            )
            self.assertTrue(all(row["real_subset"] is True for row in reconstruction))
            self.assertTrue(all(row["split"] == "dev" for row in reconstruction))

            for modality, originals, field in (
                ("image", original_images, "image_path"),
                ("speech", original_speech, "audio_path"),
            ):
                rebuilt = read_jsonl(output / f"{modality}_train.jsonl") + read_jsonl(
                    output / f"{modality}_dev.jsonl"
                )
                for original, row in zip(originals, rebuilt):
                    self.assertTrue(Path(row[field]).is_absolute())
                    self.assertTrue(Path(row[field]).is_file())
                    without_path = dict(row)
                    without_path[field] = original[field]
                    self.assertEqual(without_path, original)

            on_disk_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk_manifest, manifest)
            self.assertFalse(manifest["semantic_rows_changed"])
            self.assertFalse(manifest["sealed_data_used"])
            self.assertEqual(manifest["counts"]["reconstruction_dev"], 4)
            self.assertIn("caption/transcript", manifest["split_policy"]["reconstruction"])
            self.assertEqual(manifest["id_ranges"]["image_train"], {"first_id": 0, "last_id": 2})
            self.assertEqual(manifest["id_ranges"]["speech_dev"], {"first_id": 3, "last_id": 4})
            for record in manifest["source_files"].values():
                self.assertEqual(record["sha256"], builder.sha256_file(Path(record["path"])))
            for record in manifest["output_files"].values():
                self.assertEqual(record["sha256"], builder.sha256_file(Path(record["path"])))

    def test_wrong_counts_fail_without_creating_output(self) -> None:
        temporary, source, output = self.make_fixture(count=4)
        with temporary:
            with self.assertRaisesRegex(builder.SplitBuildError, "wrong image source count"):
                builder.build_splits(source, output, train_count=3, dev_count=2)
            self.assertFalse(output.exists())

    def test_existing_output_fails_before_any_replacement(self) -> None:
        temporary, source, output = self.make_fixture()
        with temporary:
            output.mkdir(parents=True)
            existing = output / "image_train.jsonl"
            existing.write_text("keep\n", encoding="utf-8")
            with self.assertRaisesRegex(builder.SplitBuildError, "refusing existing output"):
                builder.build_splits(source, output, train_count=3, dev_count=2)
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep\n")
            self.assertEqual(list(output.iterdir()), [existing])

    def test_existing_empty_output_directory_fails(self) -> None:
        temporary, source, output = self.make_fixture()
        with temporary:
            output.mkdir(parents=True)
            with self.assertRaisesRegex(builder.SplitBuildError, "existing output directory"):
                builder.build_splits(source, output, train_count=3, dev_count=2)

    def test_duplicate_ids_paths_and_train_dev_overlap_fail(self) -> None:
        cases = (
            ("duplicate image ID in train", 1, "id", 0),
            ("duplicate image path", 1, "image_path", "images/image_0.png"),
            ("train/dev overlap", 4, "id", 0),
        )
        for expected, index, field, value in cases:
            with self.subTest(expected=expected):
                temporary, source, output = self.make_fixture()
                with temporary:
                    rows = read_jsonl(source / "image_captions.jsonl")
                    rows[index][field] = value
                    write_jsonl(source / "image_captions.jsonl", rows)
                    with self.assertRaisesRegex(builder.SplitBuildError, expected):
                        builder.build_splits(source, output, train_count=3, dev_count=2)

    def test_missing_media_and_non_16k_metadata_fail(self) -> None:
        cases = (
            ("image_captions.jsonl", 0, "image_path", "images/missing.png", "missing media file"),
            ("speech_transcripts.jsonl", 0, "sample_rate", 8_000, "non-16k speech metadata"),
            ("speech_transcripts.jsonl", 0, "preprocess", {"resampled_to": 8_000}, "non-16k speech metadata"),
        )
        for filename, index, field, value, expected in cases:
            with self.subTest(field=field, value=value):
                temporary, source, output = self.make_fixture()
                with temporary:
                    rows = read_jsonl(source / filename)
                    rows[index][field] = value
                    write_jsonl(source / filename, rows)
                    with self.assertRaisesRegex(builder.SplitBuildError, expected):
                        builder.build_splits(source, output, train_count=3, dev_count=2)

    def test_forbidden_source_output_and_media_paths_fail(self) -> None:
        temporary, source, output = self.make_fixture()
        with temporary:
            with self.assertRaisesRegex(builder.SplitBuildError, "forbidden sealed path"):
                builder.build_splits(source, output.parent / "sealed-routing", train_count=3, dev_count=2)

        temporary, source, output = self.make_fixture()
        with temporary:
            rows = read_jsonl(source / "image_captions.jsonl")
            rows[0]["image_path"] = "synthetic/image.png"
            write_jsonl(source / "image_captions.jsonl", rows)
            with self.assertRaisesRegex(builder.SplitBuildError, "forbidden synthetic path"):
                builder.build_splits(source, output, train_count=3, dev_count=2)


if __name__ == "__main__":
    unittest.main()
