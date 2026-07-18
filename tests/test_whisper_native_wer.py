from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import eval_whisper_native_wer as native_wer


class WhisperNativeWerTest(unittest.TestCase):
    def test_normalization_is_deterministic_and_transparent(self) -> None:
        self.assertEqual(
            native_wer.normalize_text("  HÉLLO, world’s!  42% "),
            "héllo world s 42",
        )
        self.assertEqual(native_wer.normalize_text("ＡＢＣ"), "abc")

    def test_levenshtein_counts_and_wer(self) -> None:
        cases = (
            ("a b c", "a x c", {"substitutions": 1, "deletions": 0, "insertions": 0}),
            ("a b c", "a c", {"substitutions": 0, "deletions": 1, "insertions": 0}),
            ("a c", "a b c", {"substitutions": 0, "deletions": 0, "insertions": 1}),
        )
        for reference, hypothesis, expected in cases:
            self.assertEqual(
                native_wer.levenshtein_counts(reference.split(), hypothesis.split()),
                expected,
            )
        record = native_wer.word_error_record("A B C D", "a x d y")
        self.assertEqual(record["word_errors"], 3)
        self.assertEqual(record["wer"], 0.75)

    def test_runtime_identity_requires_runai_and_rejects_dirty_source(self) -> None:
        with self.assertRaises(ValueError):
            native_wer.verify_runtime_identity(Path("/repo"), "a" * 40, {})
        completed = [
            mock.Mock(stdout="a" * 40 + "\n"),
            mock.Mock(stdout="?? unexpected.txt\n"),
        ]
        with mock.patch.object(native_wer.subprocess, "run", side_effect=completed):
            with self.assertRaisesRegex(ValueError, "clean source worktree"):
                native_wer.verify_runtime_identity(
                    Path("/repo"),
                    "a" * 40,
                    {
                        "SOURCE_COMMIT_SHA": "a" * 40,
                        "RUNAI_JOB_NAME": "job",
                        "RUNAI_PROJECT": "project",
                    },
                )

    @staticmethod
    def _wav_payload(sample_rate: int, samples: int, channels: int = 2) -> bytes:
        timeline = np.arange(samples, dtype=np.float64) / sample_rate
        signal = (0.25 * np.sin(2.0 * np.pi * 220.0 * timeline) * 32767).astype(
            "<i2"
        )
        frames = np.repeat(signal[:, None], channels, axis=1).tobytes()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(frames)
        return buffer.getvalue()

    def test_processor_uses_fixed_whisper_length_and_attention_mask(self) -> None:
        payload = self._wav_payload(16_000, 1_600, channels=1)
        processor = mock.Mock()
        processor.feature_extractor.n_samples = 480_000
        processor.return_value = {"input_features": mock.Mock()}
        processor.batch_decode.return_value = ["hello"]
        model = mock.Mock()
        model.generate.return_value = [[1]]
        row = {
            "source_dataset": "source",
            "utterance_id": "utt",
            "audio_path": "audio.wav",
            "audio_sha256": hashlib.sha256(payload).hexdigest(),
            "transcript": "hello",
        }
        fake_torch = types.SimpleNamespace(
            inference_mode=lambda: contextlib.nullcontext()
        )
        with (
            mock.patch.dict("sys.modules", {"torch": fake_torch}),
            mock.patch.object(
                native_wer,
                "secure_speech_audio_snapshot",
                return_value=(Path("/data/audio.wav"), payload),
            ),
        ):
            results = native_wer.evaluate_speech_dev(
                [row],
                data_dir=Path("/data"),
                processor=processor,
                model=model,
                device="cpu",
                batch_size=1,
                max_seconds=6.0,
            )
        self.assertEqual(results[0]["wer"], 0.0)
        kwargs = processor.call_args.kwargs
        self.assertEqual(kwargs["padding"], "max_length")
        self.assertTrue(kwargs["truncation"])
        self.assertEqual(kwargs["max_length"], 480_000)
        self.assertTrue(kwargs["return_attention_mask"])

    def test_audio_resampling_and_truncation_metadata(self) -> None:
        payload = self._wav_payload(8_000, 56_000)
        audio, metadata = native_wer.preprocess_audio_payload(
            payload, target_sample_rate=16_000, max_seconds=6.0
        )
        self.assertEqual(audio.shape, (96_000,))
        self.assertEqual(metadata["original_samples"], 56_000)
        self.assertEqual(metadata["original_sample_rate"], 8_000)
        self.assertEqual(metadata["original_channels"], 2)
        self.assertEqual(metadata["processed_samples"], 96_000)
        self.assertEqual(metadata["processed_sample_rate"], 16_000)
        self.assertTrue(metadata["resampled"])
        self.assertTrue(metadata["truncated"])

    def test_loader_selects_only_speech_dev_and_proves_eval_unopened(self) -> None:
        train = [{"partition": "train"}]
        dev = [{"partition": "dev"}, {"partition": "dev"}]
        provenance = {
            "reserved_files_opened": False,
            "selection_splits": ["train", "dev"],
            "files": {
                "speech_eval": {
                    "content_opened": False,
                    "read_status": "reserved_unread",
                }
            },
        }
        fake_module = mock.Mock()
        fake_module.load_development_multimodal_partitions.return_value = (
            [],
            [],
            train,
            dev,
            provenance,
        )
        with mock.patch.dict(
            "sys.modules", {"training.olmoe_real_subset_runs": fake_module}
        ):
            selected, observed = native_wer.load_speech_dev(
                Path("/data"),
                Path("/split.json"),
                "a" * 40,
                "b" * 64,
                2,
            )
        self.assertIs(selected, dev)
        self.assertNotIn(train[0], selected)
        self.assertEqual(observed["wer_selection"]["split"], "speech_dev")
        self.assertTrue(
            observed["wer_selection"]["speech_eval_metadata_only_unopened"]
        )
        kwargs = fake_module.load_development_multimodal_partitions.call_args.kwargs
        self.assertEqual(kwargs["expected_source_commit_sha"], "a" * 40)
        self.assertEqual(kwargs["expected_speech_source_sha256"], "b" * 64)

    def test_output_is_exclusive_and_records_hashes(self) -> None:
        results = [
            {
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "reference_words": 1,
            }
        ]
        metrics = native_wer.summarize_results(results, {"source": "test"})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "wer"
            hashes = native_wer.write_outputs_exclusively(output, results, metrics)
            per_path = output / "per_utterance.jsonl"
            metrics_path = output / "metrics.json"
            self.assertEqual(
                hashes["per_utterance_jsonl_sha256"],
                hashlib.sha256(per_path.read_bytes()).hexdigest(),
            )
            stored = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["split"], "speech_dev")
            self.assertEqual(stored["row_count"], 1)
            self.assertIn("output_hashes", stored["provenance"])
            with self.assertRaises(FileExistsError):
                native_wer.write_outputs_exclusively(output, results, metrics)

    def test_preexisting_and_forbidden_outputs_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                native_wer.validate_output_path(existing)
            with self.assertRaises(ValueError):
                native_wer.validate_output_path(root / "sealed_result")

    def test_launcher_pins_source_seed_model_and_one_gpu(self) -> None:
        launcher = Path("scripts/submit_whisper_native_wer.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('MODEL_ID="openai/whisper-base.en"', launcher)
        self.assertIn(
            'MODEL_REVISION="911407f4214e0e1d82085af863093ec0b66f9cd6"',
            launcher,
        )
        self.assertIn("SEED=42", launcher)
        self.assertIn('GPU="${GPU:-1}"', launcher)
        self.assertIn('if [ "$GPU" != "1" ]', launcher)
        self.assertIn("status --porcelain --untracked-files=all", launcher)
        self.assertIn('--environment SOURCE_COMMIT_SHA="$SOURCE_COMMIT_SHA"', launcher)
        self.assertIn("--expected-rows 137", launcher)
        self.assertIn("--max-seconds 6.0", launcher)
        self.assertIn(': "${VENV_PATH:?shared VENV_PATH is required}"', launcher)
        self.assertIn('OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_MOUNT/$JOB_NAME}"', launcher)
        self.assertNotIn("for CONFIG", launcher)


if __name__ == "__main__":
    unittest.main()
