"""Pure-function tests for the sealed evaluation materializer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "datasets" / "build_sealed_eval.py"
SPEC = importlib.util.spec_from_file_location("build_sealed_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sealed_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sealed_eval)


class ImageGroupingTests(unittest.TestCase):
    def test_groups_repeated_images_and_canonicalizes_captions(self) -> None:
        hash_a = "a" * 64
        hash_b = "b" * 64
        groups = sealed_eval.group_image_records(
            [
                {
                    "content_sha256": hash_a,
                    "caption": "Zebra near a tree",
                    "source_ids": ["coco_image:9"],
                    "source_index": 3,
                },
                {
                    "content_sha256": hash_b,
                    "captions": ["A second image"],
                    "source_ids": ["coco_image:10"],
                    "source_index": 1,
                },
                {
                    "content_sha256": hash_a,
                    "captions": ["apple on grass", "Zebra near a tree"],
                    "source_ids": ["caption_row:2"],
                    "source_index": 4,
                },
            ]
        )

        self.assertEqual([group["content_sha256"] for group in groups], [hash_b, hash_a])
        grouped = groups[1]
        self.assertEqual(grouped["canonical_caption"], "apple on grass")
        self.assertEqual(grouped["captions"], ["apple on grass", "Zebra near a tree"])
        self.assertEqual(grouped["caption_count"], 2)
        self.assertEqual(grouped["source_row_count"], 2)
        self.assertEqual(grouped["source_ids"], ["caption_row:2", "coco_image:9"])


class ImageSelectionTests(unittest.TestCase):
    class FakeImage:
        def __init__(self, content_hash: str) -> None:
            self.content_hash = content_hash

        def convert(self, _mode: str):
            return self

        def copy(self):
            return self

    def _source_rows(self):
        return [
            {
                "id": position,
                "cocoid": position,
                "caption": f"caption {position}",
                "image": self.FakeImage(character * 64),
            }
            for position, character in enumerate(("a", "b", "c", "d"), start=1)
        ]

    def _load_with_exclusions(self, sample_count, excluded_hashes, excluded_source_ids):
        rows = self._source_rows()
        media_hashes = {
            row["image"].content_hash: str(position) * 64
            for position, row in enumerate(rows, start=1)
        }
        helpers = type(
            "Helpers",
            (),
            {"extract_image": staticmethod(lambda row: row["image"])},
        )()
        summary = {}
        with (
            patch.object(sealed_eval, "_load_real_subset_helpers", return_value=helpers),
            patch.object(sealed_eval, "load_dataset_ref", return_value=rows),
            patch.object(
                sealed_eval,
                "_canonical_image_hash",
                side_effect=lambda image: image.content_hash,
            ),
            patch.object(
                sealed_eval,
                "_image_output_hashes",
                side_effect=lambda image: ("f" * 64, media_hashes[image.content_hash]),
            ),
        ):
            result = sealed_eval._load_image_groups(
                sample_count,
                excluded_hashes,
                excluded_source_ids,
                summary,
            )
        return result, summary, media_hashes

    def test_old_first_n_images_are_skipped_for_deterministic_next_n(self) -> None:
        rows = self._source_rows()
        first_media_hash = "1" * 64
        (groups, images, _source), summary, _media_hashes = self._load_with_exclusions(
            2,
            {first_media_hash},
            {"coco_image:2"},
        )

        expected_hashes = [rows[2]["image"].content_hash, rows[3]["image"].content_hash]
        self.assertEqual([group["content_sha256"] for group in groups], expected_hashes)
        self.assertEqual(list(images), expected_hashes)
        self.assertEqual(summary["excluded_candidates"], 2)
        (repeated_groups, _images, _source), _summary, _hashes = self._load_with_exclusions(
            2,
            {first_media_hash},
            {"coco_image:2"},
        )
        self.assertEqual(
            [group["content_sha256"] for group in repeated_groups], expected_hashes
        )

    def test_no_exclusions_preserve_first_n_image_selection(self) -> None:
        rows = self._source_rows()
        (groups, images, _source), summary, _hashes = self._load_with_exclusions(
            2, set(), set()
        )
        expected_hashes = [rows[0]["image"].content_hash, rows[1]["image"].content_hash]
        self.assertEqual([group["content_sha256"] for group in groups], expected_hashes)
        self.assertEqual(list(images), expected_hashes)
        self.assertEqual(summary["excluded_candidates"], 0)

    def test_image_selection_fails_closed_when_disjoint_candidates_are_insufficient(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "enough unique images"):
            self._load_with_exclusions(
                2,
                {"1" * 64, "2" * 64},
                {"coco_image:3"},
            )


class SpeakerSelectionTests(unittest.TestCase):
    def test_round_robin_is_balanced_and_stable(self) -> None:
        rows = [
            {"speaker_id": 2, "source_id": "2-b", "source_index": 3},
            {"speaker_id": 1, "source_id": "1-b", "source_index": 1},
            {"speaker_id": 3, "source_id": "3-a", "source_index": 4},
            {"speaker_id": 1, "source_id": "1-a", "source_index": 0},
            {"speaker_id": 2, "source_id": "2-a", "source_index": 2},
        ]
        selected = sealed_eval.balanced_speaker_round_robin(rows, 5)
        self.assertEqual(
            [row["source_id"] for row in selected],
            ["1-a", "2-a", "3-a", "1-b", "2-b"],
        )
        counts = {speaker: 0 for speaker in (1, 2, 3)}
        for row in selected:
            counts[row["speaker_id"]] += 1
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_round_robin_skips_old_rows_and_selects_next_disjoint_rows(self) -> None:
        rows = [
            {"speaker_id": 2, "source_id": "2-b", "source_index": 3},
            {"speaker_id": 1, "source_id": "1-b", "source_index": 2},
            {"speaker_id": 2, "source_id": "2-a", "source_index": 1},
            {"speaker_id": 1, "source_id": "1-a", "source_index": 0},
        ]
        summary = {}
        selected = sealed_eval.balanced_speaker_round_robin(
            rows,
            2,
            {"1-a"},
            candidate_is_excluded=lambda row: row["source_id"] == "2-a",
            selection_summary=summary,
        )

        self.assertEqual([row["source_id"] for row in selected], ["1-b", "2-b"])
        self.assertEqual(summary["excluded_candidates"], 2)
        repeated = sealed_eval.balanced_speaker_round_robin(rows, 2, {"1-a", "2-a"})
        self.assertEqual(repeated, selected)


class HashAndProvenanceTests(unittest.TestCase):
    def test_hash_and_provenance_are_explicit_and_stable(self) -> None:
        expected = hashlib.sha256(b"abc").hexdigest()
        self.assertEqual(sealed_eval.sha256_bytes(b"abc"), expected)
        provenance = sealed_eval.make_provenance(
            "openslr/librispeech_asr",
            "clean",
            "test",
            ["utterance:2", "utterance:1", "utterance:1"],
            "LibriSpeech test-clean",
        )
        self.assertEqual(provenance["source_ids"], ["utterance:1", "utterance:2"])
        self.assertEqual(provenance["config"], "clean")
        self.assertEqual(provenance["split"], "test")
        sealed_eval.validate_source_policy(provenance, "speech")
        image_ids = sealed_eval._image_source_ids(
            {"id": 7, "cocoid": 42},
            7,
            "Multimodal-Fatima/COCO_captions_validation",
            "validation",
        )
        self.assertIn(
            "hf_row:Multimodal-Fatima/COCO_captions_validation:validation:7",
            image_ids,
        )
        self.assertIn("coco_image:42", image_ids)

    def test_reserved_index_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts"):
            sealed_eval._validate_index_name("image_test.jsonl")

    def test_training_split_is_rejected(self) -> None:
        provenance = sealed_eval.make_provenance(
            "Multimodal-Fatima/COCO_captions_validation", None, "train", []
        )
        with self.assertRaisesRegex(ValueError, "training split"):
            sealed_eval.validate_source_policy(provenance, "image")


class SourceRevisionTests(unittest.TestCase):
    def test_sealed_profiles_use_exact_revisions_without_test_fallback(self) -> None:
        self.assertEqual(len(sealed_eval.IMAGE_SOURCES), 1)
        image_source = sealed_eval.IMAGE_SOURCES[0]
        self.assertEqual(
            image_source["dataset"],
            "Multimodal-Fatima/COCO_captions_validation",
        )
        self.assertEqual(
            image_source["revision"],
            "bfa149029bb1e2975cb0b9bea8ad948db9e9ddb2",
        )
        self.assertNotIn("_test", image_source["dataset"])
        self.assertEqual(
            sealed_eval.SPEECH_SOURCE["revision"],
            "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
        )

    def test_image_loader_propagates_sealed_revision(self) -> None:
        source = sealed_eval.IMAGE_SOURCES[0]
        with patch.object(sealed_eval, "_load_real_subset_helpers", return_value=object()):
            with patch.object(
                sealed_eval,
                "load_dataset_ref",
                side_effect=RuntimeError("mocked stop"),
            ) as loader:
                with self.assertRaisesRegex(RuntimeError, "no approved image source"):
                    sealed_eval._load_image_groups(1)

        loader.assert_called_once_with(
            source["dataset"],
            source["config"],
            revision=source["revision"],
            split=source["split"],
        )

    def test_manifest_preserves_sources_and_adds_hf_sources(self) -> None:
        image_source = sealed_eval.make_provenance(
            "Multimodal-Fatima/COCO_captions_validation",
            None,
            "validation",
            [],
            "COCO 2014 validation-derived data",
        )
        speech_source = sealed_eval.make_provenance(
            "openslr/librispeech_asr",
            "clean",
            "test",
            [],
            "LibriSpeech test-clean",
        )
        groups = [
            {
                "source_row_count": 1,
                "caption_count": 1,
                "source_ids": [],
            }
        ]

        def fake_write_images(output_dir, _groups, _images):
            (output_dir / "images").mkdir(parents=True)
            row = {"id": "image-0000", "preprocessing": {"resize": [224, 224]}}
            return [row], [dict(row)]

        def fake_write_speech(output_dir, *_args):
            (output_dir / "audio").mkdir(parents=True)
            row = {
                "id": "speech-0000",
                "speaker_id": 1,
                "preprocessing": {"sample_rate_hz": 16000},
            }
            return [row], [dict(row)], speech_source

        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                output_dir=str(Path(directory) / "sealed"),
                image_samples=1,
                speech_samples=1,
                speech_duration_seconds=1.0,
                index_file=sealed_eval.DEFAULT_INDEX_NAME,
                exclude_index=[],
                force=False,
            )
            with (
                patch.object(
                    sealed_eval,
                    "_load_image_groups",
                    return_value=(groups, {}, image_source),
                ),
                patch.object(sealed_eval, "_write_images", side_effect=fake_write_images),
                patch.object(sealed_eval, "_write_speech", side_effect=fake_write_speech),
            ):
                manifest = sealed_eval.build_sealed_eval(args)

        self.assertEqual(manifest["sources"]["image"], image_source)
        self.assertEqual(manifest["sources"]["speech"], speech_source)
        self.assertNotIn("revision", manifest["sources"]["image"])
        self.assertEqual(
            manifest["hf_sources"]["image"],
            {
                "repo_id": "Multimodal-Fatima/COCO_captions_validation",
                "revision": "bfa149029bb1e2975cb0b9bea8ad948db9e9ddb2",
                "config": None,
                "split": "validation",
            },
        )
        self.assertEqual(
            manifest["hf_sources"]["speech"]["revision"],
            "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
        )
        self.assertFalse(manifest["exclusions"]["active"])
        self.assertEqual(manifest["exclusions"]["reference_indexes"], [])
        self.assertEqual(
            manifest["exclusions"]["selection_policy"]["name"],
            "deterministic_exclude_then_select",
        )


class OverlapTests(unittest.TestCase):
    def test_overlap_checks_hashes_and_source_ids(self) -> None:
        candidates = [
            {
                "media_sha256": "1" * 64,
                "content_sha256": "2" * 64,
                "source": {"source_ids": ["coco_image:7"]},
            }
        ]
        clean = sealed_eval.overlap_report(
            candidates,
            [{"media_sha256": "3" * 64, "source_ids": ["coco_image:8"]}],
        )
        self.assertTrue(clean["passed"])
        sealed_eval.assert_no_overlap(clean, "image")

        overlapping = sealed_eval.overlap_report(
            candidates,
            [
                {
                    "media_sha256": "2" * 64,
                    "source": {"source_ids": ["coco_image:7"]},
                }
            ],
        )
        self.assertFalse(overlapping["passed"])
        self.assertEqual(overlapping["hash_overlap_count"], 1)
        self.assertEqual(overlapping["source_id_overlap_count"], 1)
        with self.assertRaisesRegex(ValueError, "hashes=1 source_ids=1"):
            sealed_eval.assert_no_overlap(overlapping, "image")

    def test_single_row_jsonl_reference_is_loaded_once(self) -> None:
        row = {"media_sha256": "4" * 64, "source_ids": ["coco_image:4"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.jsonl"
            sealed_eval._write_jsonl(path, [row])
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(sealed_eval._read_reference_records([path]), [row])

    def test_multiline_jsonl_reference_preserves_row_count(self) -> None:
        rows = [
            {"media_sha256": "5" * 64, "source_ids": ["utterance:5"]},
            {"media_sha256": "6" * 64, "source_ids": ["utterance:6"]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.jsonl"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertEqual(sealed_eval._read_reference_records([path]), rows)

    def test_exclusion_indexes_parse_modality_keys_and_record_provenance(self) -> None:
        image_content_hash = "a" * 64
        image_media_hash = "b" * 64
        speech_media_hash = "c" * 64
        rows = [
            {
                "modality": "image",
                "content_sha256": image_content_hash,
                "media_sha256": image_media_hash,
                "source": {"source_ids": ["coco_image:1"]},
            },
            {
                "modality": "speech",
                "media_sha256": speech_media_hash,
                "source_id": "1-2-3",
                "utterance_id": "1-2-3",
                "source": {"source_ids": ["utterance:1-2-3"]},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exclude.jsonl"
            sealed_eval._write_jsonl(path, rows)
            exclusions = sealed_eval._read_exclusion_indexes([path])

            self.assertEqual(
                exclusions["reference_indexes"],
                [
                    {
                        "path": str(path),
                        "sha256": sealed_eval.sha256_file(path),
                        "rows": 2,
                    }
                ],
            )

        self.assertEqual(
            exclusions["image_hashes"],
            {image_content_hash, image_media_hash},
        )
        self.assertEqual(exclusions["image_source_ids"], {"coco_image:1"})
        self.assertEqual(exclusions["speech_media_hashes"], {speech_media_hash})
        self.assertEqual(
            exclusions["speech_source_ids"],
            {"1-2-3", "utterance:1-2-3"},
        )


if __name__ == "__main__":
    unittest.main()
