import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_specialization_and_quality import (
    load_specialization_rows,
    parse_args,
    text_readability,
)


class SpecializationQualityTest(unittest.TestCase):
    def test_stage_b_arguments_are_required(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--run-output-dir",
                    "run",
                    "--checkpoint",
                    "e3.pt",
                ]
            )

    def test_accepts_exact_stage_b_path_and_sha(self):
        args = parse_args(
            [
                "--run-output-dir",
                "run",
                "--checkpoint",
                "e3.pt",
                "--stage-b-checkpoint",
                "stage-b.pt",
                "--stage-b-checkpoint-sha256",
                "a" * 64,
            ]
        )
        self.assertEqual(args.stage_b_checkpoint, "stage-b.pt")
        self.assertEqual(args.stage_b_checkpoint_sha256, "a" * 64)

    def test_requires_paired_development_manifests(self):
        args = parse_args(
            [
                "--run-output-dir", "run", "--checkpoint", "e3.pt",
                "--stage-b-checkpoint", "stage-b.pt",
                "--stage-b-checkpoint-sha256", "a" * 64,
                "--image-manifest", "image_dev.jsonl",
            ]
        )
        with self.assertRaisesRegex(ValueError, "provided together"):
            load_specialization_rows(args, Path("data"))

    def test_loads_only_exact_development_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "image.bin").write_bytes(b"image")
            (data_dir / "audio.bin").write_bytes(b"audio")
            image_manifest = root / "image_dev.jsonl"
            speech_manifest = root / "speech_dev.jsonl"
            image_manifest.write_text(
                json.dumps(
                    {
                        "id": 1,
                        "task": "image",
                        "eval_split_name": "image_dev",
                        "image_path": str(data_dir / "image.bin"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            speech_manifest.write_text(
                json.dumps(
                    {
                        "id": 2,
                        "task": "speech",
                        "eval_split_name": "speech_dev",
                        "audio_path": str(data_dir / "audio.bin"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--run-output-dir", "run", "--checkpoint", "e3.pt",
                    "--stage-b-checkpoint", "stage-b.pt",
                    "--stage-b-checkpoint-sha256", "a" * 64,
                    "--image-manifest", str(image_manifest),
                    "--speech-manifest", str(speech_manifest),
                    "--image-eval-count", "1",
                    "--speech-eval-count", "1",
                ]
            )
            image_rows, speech_rows, provenance = load_specialization_rows(
                args, data_dir
            )
            self.assertEqual(len(image_rows), 1)
            self.assertEqual(len(speech_rows), 1)
            self.assertTrue(provenance["development_only"])
            self.assertEqual(provenance["image"]["split_role"], "image_dev")

            image_manifest.write_text(
                json.dumps(
                    {
                        "id": 1,
                        "task": "image",
                        "eval_split_name": "image_eval",
                        "image_path": str(data_dir / "image.bin"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "image_dev"):
                load_specialization_rows(args, data_dir)

    def test_readable_generation_passes(self):
        result = text_readability("A person walks beside a red car.")
        self.assertTrue(result["readable"])
        self.assertEqual(result["replacement_characters"], 0)

    def test_mojibake_and_empty_generation_fail(self):
        self.assertFalse(text_readability("")["readable"])
        self.assertFalse(text_readability("bad \ufffd text")["readable"])
        self.assertFalse(text_readability("Fran\u00c3\u00a7ais")["readable"])


if __name__ == "__main__":
    unittest.main()
