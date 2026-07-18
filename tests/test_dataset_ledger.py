"""Tiny-fixture tests for the independently recomputed dataset ledger."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import tempfile
import unittest
import wave
import zlib
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_dataset_ledger.py"
SPEC = importlib.util.spec_from_file_location("build_dataset_ledger", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def write_png(path: Path, width: int = 2, height: int = 2, color_type: int = 2, seed: int = 0) -> None:
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    raw = b"".join(bytes([0]) + bytes([seed]) * (width * channels) for _ in range(height))
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    payload = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", payload) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def write_wav(path: Path, sample_rate: int = 16_000, channels: int = 1, samples: int = 16, seed: int = 0) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack("<h", seed) * channels * samples)


class DatasetLedgerTests(unittest.TestCase):
    def make_fixture(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "data"
        (root / "images").mkdir(parents=True)
        (root / "audio").mkdir()
        write_png(root / "images" / "one.png")
        write_png(root / "images" / "two.png", seed=1)
        write_wav(root / "audio" / "one.wav")
        write_wav(root / "audio" / "two.wav", seed=1)
        tasks = [{"task": task, "text": "private text"} for task in ledger.TASK_LABELS]
        blocks = [{"task": task, "input_ids": [1, 2]} for task in ledger.TASK_LABELS]
        write_jsonl(root / "text_tasks.jsonl", tasks)
        write_jsonl(root / "text_blocks_train.jsonl", blocks)
        write_jsonl(root / "text_blocks_eval.jsonl", blocks)
        write_jsonl(root / "image_captions.jsonl", [{"task": "image", "image_path": "images/one.png"}, {"task": "image", "image_path": "images/two.png"}])
        write_jsonl(root / "speech_transcripts.jsonl", [{"task": "speech", "audio_path": "audio/one.wav"}, {"task": "speech", "audio_path": "audio/two.wav"}])
        # Deliberately false, and the ledger must never consume it.
        (root / "manifest.json").write_text(json.dumps({"counts": {"text_blocks_train": 999999}}), encoding="utf-8")
        return temporary, root

    def args(self, root: Path, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "data_root": str(root), "output": str(root / "ledger.json"), "force": False, "allow_short": True,
            "media_validation": "full", "media_sample_size": 10, "image_width": 2, "image_height": 2,
            "image_mode": "RGB", "audio_sample_rate": 16000, "audio_channels": 1, "audio_num_samples": 16,
            "image_eval_tail": 0, "speech_eval_tail": 0, "sealed_root": None, "sealed_reference_index": [],
            "hf_cache_root": None, "hf_ref_registry": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_positive_ledger_recomputes_not_manifest_counts_and_is_public_safe(self) -> None:
        temporary, root = self.make_fixture()
        with temporary:
            document = ledger.build_ledger(self.args(root))
            self.assertEqual(document["schema_version"], 2)
            self.assertEqual(document["ledger_type"], "recomputed_dataset_ledger")
            self.assertEqual(set(document["files"]), {"image_rows", "speech_rows", "packed_tasks"})
            self.assertEqual(document["jsonl_files"]["text_blocks_train"]["rows"], 5)
            self.assertEqual(document["task_counts"]["packed_train_blocks"]["text"], 1)
            self.assertEqual(document["media"]["image"]["unique_paths"], 2)
            self.assertEqual(document["counts"]["image_rows"], 2)
            self.assertEqual(document["counts"]["speech_rows"], 2)
            self.assertEqual(document["counts"]["packed_rows"], 5)
            self.assertEqual(document["counts"]["packed_task_counts"], {task: 1 for task in ledger.TASK_LABELS})
            self.assertNotIn("private text", json.dumps(document))
            self.assertNotIn("manifest.json", json.dumps(document))

    def test_schema_v2_records_physical_files_and_task_shards(self) -> None:
        temporary, root = self.make_fixture()
        with temporary:
            output = root / "evidence" / "dataset_ledger.json"
            document = ledger.build_ledger(self.args(root, output=str(output)))
            files = document["files"]
            for role, source_name in (
                ("image_rows", "image_captions.jsonl"),
                ("speech_rows", "speech_transcripts.jsonl"),
            ):
                record = files[role]
                source = (root / source_name).resolve()
                self.assertEqual(Path(record["path"]), source)
                self.assertEqual(record["sha256"], ledger.sha256_file(source))
                self.assertEqual(record["bytes"], source.stat().st_size)
                self.assertEqual(record["rows"], 2)

            self.assertEqual(set(files["packed_tasks"]), set(ledger.TASK_LABELS))
            for task in ledger.TASK_LABELS:
                record = files["packed_tasks"][task]
                shard = Path(record["path"])
                self.assertTrue(shard.is_absolute())
                self.assertEqual(shard.parent, output.parent.resolve())
                self.assertEqual(shard.name, f"dataset_ledger.packed.{task}.jsonl")
                self.assertTrue(shard.is_file())
                self.assertFalse(shard.is_symlink())
                self.assertEqual(record["sha256"], ledger.sha256_file(shard))
                self.assertEqual(record["bytes"], shard.stat().st_size)
                shard_rows, evidence = ledger.load_jsonl(shard)
                self.assertEqual(record["rows"], evidence["rows"])
                self.assertEqual([row["task"] for row in shard_rows], [task])

    def test_shards_are_idempotent_and_force_controls_replacement(self) -> None:
        temporary, root = self.make_fixture()
        with temporary:
            args = self.args(root)
            first = ledger.build_ledger(args)
            paths = {task: Path(record["path"]) for task, record in first["files"]["packed_tasks"].items()}
            mtimes = {task: path.stat().st_mtime_ns for task, path in paths.items()}
            second = ledger.build_ledger(args)
            self.assertEqual(first, second)
            self.assertEqual(mtimes, {task: path.stat().st_mtime_ns for task, path in paths.items()})

            expected_text_hash = first["files"]["packed_tasks"]["text"]["sha256"]
            paths["text"].write_text('{"task":"text","corrupt":true}\n', encoding="utf-8")
            code_mtime = paths["code"].stat().st_mtime_ns
            with self.assertRaisesRegex(ledger.LedgerError, "refusing to overwrite existing packed shard"):
                ledger.build_ledger(args)
            self.assertIn("corrupt", paths["text"].read_text(encoding="utf-8"))
            self.assertEqual(paths["code"].stat().st_mtime_ns, code_mtime)

            repaired = ledger.build_ledger(self.args(root, force=True))
            self.assertEqual(ledger.sha256_file(paths["text"]), expected_text_hash)
            self.assertEqual(repaired["files"]["packed_tasks"]["text"]["sha256"], expected_text_hash)

    def test_main_refuses_existing_output_before_shard_side_effects(self) -> None:
        temporary, root = self.make_fixture()
        with temporary:
            output = root / "ledger.json"
            output.write_text("existing\n", encoding="utf-8")
            argv = [
                "--data-root", str(root), "--output", str(output), "--allow-short",
                "--image-width", "2", "--image-height", "2", "--image-mode", "RGB",
                "--audio-sample-rate", "16000", "--audio-channels", "1",
                "--audio-num-samples", "16", "--image-eval-tail", "0",
                "--speech-eval-tail", "0",
            ]
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite existing ledger"):
                ledger.main(argv)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing\n")
            self.assertEqual(list(root.glob("ledger.packed.*.jsonl")), [])

    def test_missing_media_fails(self) -> None:
        temporary, root = self.make_fixture()
        with temporary:
            write_jsonl(root / "image_captions.jsonl", [{"task": "image", "image_path": "images/nope.png"}])
            with self.assertRaisesRegex(ledger.LedgerError, "missing media"):
                ledger.build_ledger(self.args(root))

    def test_duplicate_media_reference_fails(self) -> None:
        temporary, root = self.make_fixture()
        with temporary:
            write_jsonl(root / "image_captions.jsonl", [{"task": "image", "image_path": "images/one.png"}, {"task": "image", "image_path": "images/one.png"}])
            with self.assertRaisesRegex(ledger.LedgerError, "duplicate image media reference"):
                ledger.build_ledger(self.args(root))

    def test_repeated_image_content_with_distinct_captions_is_counted(self) -> None:
        temporary, root = self.make_fixture()
        with temporary:
            (root / "images" / "two.png").write_bytes((root / "images" / "one.png").read_bytes())
            write_jsonl(
                root / "image_captions.jsonl",
                [
                    {"task": "image", "image_path": "images/one.png", "caption": "first caption"},
                    {"task": "image", "image_path": "images/two.png", "caption": "second caption"},
                ],
            )
            document = ledger.build_ledger(self.args(root))
            self.assertEqual(document["media"]["image"]["referenced_rows"], 2)
            self.assertEqual(document["media"]["image"]["unique_media_sha256"], 1)
            self.assertEqual(document["media"]["image"]["duplicate_content_rows"], 1)
            self.assertEqual(document["media"]["image"]["unique_media_target_pairs"], 2)

    def test_wrong_preprocessing_fails(self) -> None:
        temporary, root = self.make_fixture()
        with temporary:
            with self.assertRaisesRegex(ledger.LedgerError, "image preprocessing mismatch"):
                ledger.build_ledger(self.args(root, image_width=3))

    def test_final_scale_minimums_fail_unless_debug_allow_short(self) -> None:
        temporary, root = self.make_fixture()
        with temporary:
            with self.assertRaisesRegex(ledger.LedgerError, "final-scale minimum gates failed"):
                ledger.build_ledger(self.args(root, allow_short=False))
            self.assertTrue(ledger.build_ledger(self.args(root, allow_short=True))["minimum_gate"]["passed"])

    def test_sealed_overlap_fails(self) -> None:
        temporary, root = self.make_fixture()
        with temporary:
            image_hash = hashlib.sha256((root / "images" / "one.png").read_bytes()).hexdigest()
            reference = root / "sealed.jsonl"
            write_jsonl(reference, [{"media_sha256": image_hash}])
            with self.assertRaisesRegex(ledger.LedgerError, "sealed overlap failure"):
                ledger.build_ledger(self.args(root, sealed_reference_index=[str(reference)]))

    def test_hf_ref_mismatch_fails(self) -> None:
        temporary, root = self.make_fixture()
        with temporary:
            cache = root / "cache" / "datasets--org--repo" / "refs"
            cache.mkdir(parents=True)
            ref_path = cache / "main"
            ref_path.write_text("a" * 40, encoding="utf-8")
            registry = root / "refs.json"
            registry.write_text(json.dumps({"repos": {"org/repo": {"ref": "b" * 40, "mtime_ns": ref_path.stat().st_mtime_ns}}}), encoding="utf-8")
            with self.assertRaisesRegex(ledger.LedgerError, "external_cache_ref_provenance: ref mismatch"):
                ledger.build_ledger(self.args(root, hf_cache_root=str(root / "cache"), hf_ref_registry=str(registry)))

    def test_hf_ref_match_and_atomic_overwrite_refusal(self) -> None:
        temporary, root = self.make_fixture()
        with temporary:
            cache = root / "cache" / "datasets--org--repo" / "refs"
            cache.mkdir(parents=True)
            ref_path = cache / "main"
            ref_path.write_text("a" * 40, encoding="utf-8")
            registry = root / "refs.json"
            registry.write_text(json.dumps({"repos": {"org/repo": {"ref": "a" * 40, "mtime_ns": ref_path.stat().st_mtime_ns}}}), encoding="utf-8")
            args = self.args(root, hf_cache_root=str(root / "cache"), hf_ref_registry=str(registry))
            document = ledger.build_ledger(args)
            self.assertEqual(document["external_cache_ref_provenance"]["kind"], "external_cache_ref_provenance")
            ledger.write_atomic(Path(args.output), document, force=False)
            with self.assertRaisesRegex(ledger.LedgerError, "refusing to overwrite"):
                ledger.write_atomic(Path(args.output), document, force=False)


if __name__ == "__main__":
    unittest.main()
