from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from dataclasses import replace
import unittest

import torch
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "freeze_evaluation_protocol.py"
SPEC = importlib.util.spec_from_file_location("freeze_evaluation_protocol", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
freezer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = freezer
SPEC.loader.exec_module(freezer)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FreezeEvaluationProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.selected_root = self.root / "checkpoint_root"
        self.selected_root.mkdir()
        self.checkpoint = self.selected_root / "checkpoint_final.pt"
        torch.save(
            {
                "args": {"capacity_factor": 7.0, "top_k": 2},
                "run_provenance": {"source_commit_sha": "a" * 40},
                "gamma_provenance": {"sha256": "b" * 64},
                "trainable_meta": {"trainable_params": 1},
                "last_row": {"step": 1},
            },
            self.checkpoint,
        )
        self.evaluator = self.root / "evaluate.py"
        self.evaluator.write_text("print('evaluate')\n", encoding="utf-8")
        self.analysis = self.root / "paired.py"
        self.analysis.write_text("print('paired')\n", encoding="utf-8")
        self.image_test = self.root / "image_test.jsonl"
        self.speech_test = self.root / "speech_test.jsonl"
        self.manifest = self.root / "sealed_eval_manifest.json"
        self.output = self.root / "frozen_protocol.json"
        self.image_dir = self.root / "images"
        self.audio_dir = self.root / "audio"
        self.image_dir.mkdir()
        self.audio_dir.mkdir()
        self.image_media = [
            self.image_dir / "image-0000.png",
            self.image_dir / "image-0001.png",
        ]
        self.speech_media = self.audio_dir / "speech-0000.wav"
        self.image_media[0].write_bytes(b"image-zero")
        self.image_media[1].write_bytes(b"image-one")
        self.speech_media.write_bytes(b"speech-zero")
        self.image_rows = [
            {
                "id": "image-0000",
                "image_path": "images/image-0000.png",
                "caption": "secret caption",
                "media_sha256": sha256(self.image_media[0]),
                "content_sha256": "b" * 64,
                "source": {"source_ids": ["coco_image:1"]},
                "group_id": "image-group-1",
            },
            {
                "id": "image-0001",
                "image_path": "images/image-0001.png",
                "caption": "another secret caption",
                "media_sha256": sha256(self.image_media[1]),
                "content_sha256": "d" * 64,
                "source": {"source_ids": ["coco_image:2"]},
                "group_id": "image-group-2",
            },
        ]
        self.speech_rows = [
            {
                "id": "speech-0000",
                "audio_path": "audio/speech-0000.wav",
                "transcript": "secret transcript",
                "media_sha256": sha256(self.speech_media),
                "source_id": "1-2-3",
                "source": {"source_ids": ["utterance:1-2-3"]},
                "group_id": "speech-group-1",
            }
        ]
        self.write_bundle()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def manifest_value(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "sealed": True,
            "sources": {
                "image": {
                    "dataset": "Multimodal-Fatima/COCO_captions_validation",
                    "config": None,
                    "split": "validation",
                    "partition": "COCO 2014 validation-derived data",
                },
                "speech": {
                    "dataset": "openslr/librispeech_asr",
                    "config": "clean",
                    "split": "test",
                    "partition": "LibriSpeech test-clean",
                },
            },
            "counts": {
                "image_rows": len(self.image_rows),
                "speech_rows": len(self.speech_rows),
                "total_rows": len(self.image_rows) + len(self.speech_rows),
            },
            "overlap_assertions": {
                "source_partition_policy": {
                    "image_non_training_validation_style": True,
                    "speech_is_librispeech_test_clean": True,
                    "passed": True,
                },
                "image": {
                    "passed": True,
                    "hash_overlap_count": 0,
                    "source_id_overlap_count": 0,
                },
                "speech": {
                    "passed": True,
                    "hash_overlap_count": 0,
                    "source_id_overlap_count": 0,
                },
            },
            "files": {
                "image_test.jsonl": {
                    "sha256": sha256(self.image_test),
                    "rows": len(self.image_rows),
                },
                "speech_test.jsonl": {
                    "sha256": sha256(self.speech_test),
                    "rows": len(self.speech_rows),
                },
            },
        }

    def write_bundle(self) -> None:
        self.write_jsonl(self.image_test, self.image_rows)
        self.write_jsonl(self.speech_test, self.speech_rows)
        self.manifest.write_text(
            json.dumps(self.manifest_value(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def config(self, output: Path | None = None) -> freezer.FreezeConfig:
        return freezer.FreezeConfig(
            selected_root=self.selected_root,
            checkpoint=self.checkpoint,
            checkpoint_args={"capacity_factor": 7.0, "top_k": 2},
            sealed_manifest=self.manifest,
            image_test=self.image_test,
            speech_test=self.speech_test,
            evaluator_scripts=[
                self.evaluator,
                Path(freezer.__file__).with_name(
                    "sealed_position_allocator.py"
                ),
            ],
            paired_analysis_scripts=[self.analysis],
            output=output or self.output,
            runai_project="runai-lidiap-alignai-yctang",
            candidate_seed=17,
            control_seed=23,
            candidate_sizes=[5, 10, 50],
            candidate_protocols=["random", "hard-negative"],
            hard_negative_protocol="same-source semantic nearest neighbors, fixed before unsealing",
            image_query_count=250,
            speech_query_count=250,
            evaluation_cells=[
                {
                    "id": "r5",
                    "candidate_count": 5,
                    "negative_mode": "random",
                    "role": "secondary",
                },
                {
                    "id": "h10",
                    "candidate_count": 10,
                    "negative_mode": "hard_text",
                    "role": "primary",
                },
                {
                    "id": "f50",
                    "candidate_count": 50,
                    "negative_mode": "full_matrix",
                    "role": "secondary",
                },
            ],
        )

    def test_refuses_overwrite(self) -> None:
        freezer.freeze_protocol(self.config())
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            freezer.freeze_protocol(self.config())

    def test_verify_detects_hash_drift(self) -> None:
        freezer.freeze_protocol(self.config())
        self.evaluator.write_text("print('changed')\n", encoding="utf-8")
        with self.assertRaisesRegex(freezer.ProtocolError, "hash drift"):
            freezer.verify_protocol(self.output)

    def test_rejects_invalid_split_and_failed_overlap(self) -> None:
        manifest = self.manifest_value()
        manifest["sources"]["image"]["split"] = "train"
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(freezer.ProtocolError, "non-training"):
            freezer.freeze_protocol(self.config())

        manifest = self.manifest_value()
        manifest["overlap_assertions"]["speech"]["passed"] = False
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(freezer.ProtocolError, "overlap assertion did not pass"):
            freezer.freeze_protocol(self.config())

    def test_rejects_duplicate_media_source_and_group_ids(self) -> None:
        cases = (
            ("media_sha256", "duplicate media ID"),
            ("source", "duplicate source ID"),
            ("group_id", "duplicate group ID"),
        )
        for index, (field, message) in enumerate(cases):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.image_rows)
                if field == "source":
                    rows[1][field] = copy.deepcopy(rows[0][field])
                else:
                    rows[1][field] = rows[0][field]
                self.image_rows = rows
                self.write_bundle()
                output = self.root / f"duplicate-{index}.json"
                with self.assertRaisesRegex(freezer.ProtocolError, message):
                    freezer.freeze_protocol(self.config(output))
                self.image_rows = copy.deepcopy(
                    [
                        {
                            "id": "image-0000",
                            "image_path": "images/image-0000.png",
                            "caption": "secret caption",
                            "media_sha256": sha256(self.image_media[0]),
                            "content_sha256": "b" * 64,
                            "source": {"source_ids": ["coco_image:1"]},
                            "group_id": "image-group-1",
                        },
                        {
                            "id": "image-0001",
                            "image_path": "images/image-0001.png",
                            "caption": "another secret caption",
                            "media_sha256": sha256(self.image_media[1]),
                            "content_sha256": "d" * 64,
                            "source": {"source_ids": ["coco_image:2"]},
                            "group_id": "image-group-2",
                        },
                    ]
                )

    def test_successful_verify_and_content_free_protocol(self) -> None:
        protocol = freezer.freeze_protocol(self.config())
        verified = freezer.verify_protocol(self.output)
        self.assertEqual(verified["protocol_content_sha256"], protocol["protocol_content_sha256"])
        self.assertEqual(verified["controls"], list(freezer.REQUIRED_CONTROLS))
        self.assertEqual(verified["sealed_metrics_policy"], freezer.SEALED_METRICS_POLICY)
        self.assertEqual(verified["holm_families"][0]["cell_ids"], ["h10"])
        self.assertEqual(verified["evaluation_matrix"][1]["role"], "primary")
        self.assertEqual(len(verified["evaluation_runs"]), 15)
        allocator = verified["gold_position_allocator"]
        self.assertEqual(allocator["name"], freezer.ALLOCATOR_NAME)
        self.assertEqual(allocator["version"], freezer.ALLOCATOR_VERSION)
        self.assertEqual(len(allocator["plans"]), 6)
        for plan in allocator["plans"]:
            self.assertLessEqual(
                max(plan["position_counts"]) - min(plan["position_counts"]),
                1,
            )
        self.assertEqual(
            {run["max_length"] for run in verified["evaluation_runs"]},
            {512},
        )
        self.assertEqual(
            {run["conditional_batch_size"] for run in verified["evaluation_runs"]},
            {8},
        )
        serialized_rows = json.dumps(verified["sealed_rows"])
        self.assertNotIn("secret caption", serialized_rows)
        self.assertNotIn("secret transcript", serialized_rows)
        self.assertEqual(
            {tuple(sorted(row)) for row in verified["sealed_rows"]},
            {
                (
                    "group_ids_sha256",
                    "media_ids_sha256",
                    "media_relative_path_sha256",
                    "media_sha256",
                    "media_size_bytes",
                    "modality",
                    "row_id",
                    "row_sha256",
                    "source_ids_sha256",
                )
            },
        )

    def test_verify_rejects_incomplete_schema_v2_evaluation_run(self) -> None:
        freezer.freeze_protocol(self.config())
        protocol = json.loads(self.output.read_text(encoding="utf-8"))
        del protocol["evaluation_runs"][0]["max_length"]
        unhashed = dict(protocol)
        unhashed.pop("protocol_content_sha256")
        protocol["protocol_content_sha256"] = freezer.canonical_sha256(unhashed)
        self.output.write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            freezer.ProtocolError, "evaluation run contract is incomplete"
        ):
            freezer.verify_protocol(self.output)

    def test_freeze_requires_exactly_one_allocator_source_fingerprint(self) -> None:
        allocator = Path(freezer.__file__).with_name(
            "sealed_position_allocator.py"
        )
        with self.assertRaisesRegex(
            freezer.ProtocolError, "exactly one resolved"
        ):
            freezer.freeze_protocol(
                replace(
                    self.config(output=self.root / "missing-allocator.json"),
                    evaluator_scripts=[self.evaluator],
                )
            )
        with self.assertRaisesRegex(
            freezer.ProtocolError, "exactly one resolved"
        ):
            freezer.freeze_protocol(
                replace(
                    self.config(output=self.root / "duplicate-allocator.json"),
                    evaluator_scripts=[self.evaluator, allocator, allocator],
                )
            )

    def test_verify_rejects_removed_or_tampered_allocator_fingerprint(
        self,
    ) -> None:
        freezer.freeze_protocol(self.config())
        original = json.loads(self.output.read_text(encoding="utf-8"))
        cases = {
            "removed": (
                lambda protocol: protocol["inputs"]["evaluator_scripts"].pop(),
                "exactly one resolved",
            ),
            "tampered": (
                lambda protocol: protocol["inputs"]["evaluator_scripts"][-1].update(
                    {"sha256": "0" * 64}
                ),
                "current SHA256",
            ),
        }
        for name, (mutate, message) in cases.items():
            with self.subTest(name=name):
                protocol = copy.deepcopy(original)
                mutate(protocol)
                unhashed = dict(protocol)
                unhashed.pop("protocol_content_sha256")
                protocol["protocol_content_sha256"] = freezer.canonical_sha256(
                    unhashed
                )
                self.output.write_text(
                    json.dumps(protocol, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(freezer.ProtocolError, message):
                    freezer.verify_protocol(self.output)

    def test_verify_rejects_rehashed_unbalanced_assignment_plan(self) -> None:
        freezer.freeze_protocol(self.config())
        protocol = json.loads(self.output.read_text(encoding="utf-8"))
        plan = protocol["gold_position_allocator"]["plans"][0]
        plan["positions"] = [0] * plan["query_count"]
        plan["position_counts"] = [plan["query_count"]] + [
            0
        ] * (plan["candidate_count"] - 1)
        plan["positions_sha256"] = freezer.canonical_sha256(plan["positions"])
        protocol["gold_position_allocator"]["plans_sha256"] = (
            freezer.canonical_sha256(
                protocol["gold_position_allocator"]["plans"]
            )
        )
        unhashed = dict(protocol)
        unhashed.pop("protocol_content_sha256")
        protocol["protocol_content_sha256"] = freezer.canonical_sha256(unhashed)
        self.output.write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            freezer.ProtocolError,
            "invalid gold-position allocator",
        ):
            freezer.verify_protocol(self.output)

    def test_rejects_actual_media_replacement_and_symlink(self) -> None:
        self.image_media[0].write_bytes(b"replacement")
        with self.assertRaisesRegex(
            freezer.ProtocolError, "media_sha256 does not match actual bytes"
        ):
            freezer.freeze_protocol(self.config())

        self.image_media[0].unlink()
        self.image_media[0].symlink_to(self.image_media[1])
        self.image_rows[0]["media_sha256"] = sha256(self.image_media[1])
        self.write_bundle()
        with self.assertRaisesRegex(freezer.ProtocolError, "symlink"):
            freezer.freeze_protocol(self.config())


if __name__ == "__main__":
    unittest.main()
