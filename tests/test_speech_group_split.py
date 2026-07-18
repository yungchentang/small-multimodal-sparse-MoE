"""Focused tests for real speech provenance and group-disjoint splits."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

from scripts.development_split_provenance import verify_speech_audio_rows


ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


real_subset = load_module("speech_group_real_subset", ROOT / "datasets" / "build_real_subset.py")
audit = load_module("speech_group_audit", ROOT / "scripts" / "audit_speech_group_split.py")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_pcm_wav(path: Path, sample: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(int(sample).to_bytes(2, "little", signed=True))


def source_fingerprint(label: str) -> dict:
    payload = label.encode("utf-8")
    return {
        "algorithm": "sha256",
        "semantics": "encoded_audio_bytes_v1",
        "value": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


class StrictSpeechAudioPathTest(unittest.TestCase):
    def test_valid_pcm_wav_records_deterministic_format_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory).resolve()
            audio_path = data_dir / "sample.wav"
            write_pcm_wav(audio_path, 7)
            row = {
                "id": "valid-0",
                "audio_path": "sample.wav",
                "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            }

            result = verify_speech_audio_rows([row], data_dir=data_dir)

            self.assertEqual(
                result["audio_format_summary"],
                {
                    "parser": "python_stdlib_wave",
                    "formats": [{
                        "container": "WAV",
                        "encoding": "PCM",
                        "channels": 1,
                        "sample_width_bytes": 2,
                        "sample_rate_hz": 16000,
                        "unique_files": 1,
                        "min_frame_count": 1,
                        "max_frame_count": 1,
                        "total_frame_count": 1,
                    }],
                },
            )
            self.assertEqual(len(result["audio_format_binding_root_sha256"]), 64)
            self.assertTrue(result["audio_wav_decoded_this_run"])

    def test_builder_relative_output_uses_data_root_relative_audio_paths(self) -> None:
        candidate = types.SimpleNamespace(
            repo="openslr/librispeech_asr",
            config="clean",
            split="train.100",
            label="openslr/librispeech_asr/clean:train.100",
        )
        raw_rows = []
        for index in range(4):
            speaker = str(100 + index)
            utterance = f"{speaker}-1-0000"
            raw_rows.append({
                "id": utterance,
                "speaker_id": speaker,
                "chapter_id": "1",
                "text": f"fixture transcript {index}",
                "audio": {
                    "array": real_subset.np.zeros(32, dtype=real_subset.np.float32),
                    "sampling_rate": 16000,
                },
            })

        def fake_write(path, _audio, _sample_rate):
            write_pcm_wav(Path(path), 7)

        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                with mock.patch.object(
                    real_subset, "SPEECH_SOURCE_PROFILE", (candidate,)
                ), mock.patch.object(
                    real_subset,
                    "iter_dataset_rows",
                    return_value=iter(raw_rows),
                ), mock.patch.dict(
                    sys.modules,
                    {"soundfile": types.SimpleNamespace(write=fake_write)},
                ):
                    rows, _sources, _summary = real_subset.build_audio_rows(
                        Path("relative-data"),
                        max_samples=4,
                        target_sr=16000,
                        max_seconds=0.001,
                        errors=[],
                        heldout_rows=2,
                        split_seed=0,
                    )
                data_dir = (Path(directory) / "relative-data").resolve()
                self.assertEqual(len(rows), 4)
                for row in rows:
                    audio_path = Path(row["audio_path"])
                    self.assertFalse(audio_path.is_absolute())
                    self.assertEqual(audio_path.parts[0], "audio")
                    self.assertTrue((data_dir / audio_path).is_file())
                verified = verify_speech_audio_rows(rows, data_dir=data_dir)
                self.assertEqual(verified["audio_rows_verified"], 4)
                self.assertEqual(verified["unique_audio_files_verified"], 4)
                self.assertTrue(verified["audio_wav_decoded_this_run"])
            finally:
                os.chdir(previous)

    def test_parent_symlink_outside_data_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = (root / "data").resolve()
            outside_dir = (root / "outside").resolve()
            data_dir.mkdir()
            outside_dir.mkdir()
            outside_audio = outside_dir / "sample.wav"
            write_pcm_wav(outside_audio)
            (data_dir / "linked").symlink_to(outside_dir, target_is_directory=True)
            row = {
                "source_dataset": "real",
                "utterance_id": "outside-0",
                "audio_path": "linked/sample.wav",
                "audio_sha256": hashlib.sha256(
                    outside_audio.read_bytes()
                ).hexdigest(),
            }

            with self.assertRaisesRegex(ValueError, "symlink component"):
                verify_speech_audio_rows([row], data_dir=data_dir)

    def test_symlink_followed_by_parent_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data_dir = root / "data"
            outside_dir = root / "outside"
            data_dir.mkdir()
            outside_dir.mkdir()
            inside_audio = data_dir / "sample.wav"
            external_audio = root / "sample.wav"
            write_pcm_wav(inside_audio)
            external_audio.write_bytes(inside_audio.read_bytes())
            (data_dir / "linked").symlink_to(
                outside_dir, target_is_directory=True
            )
            row = {
                "source_dataset": "real",
                "utterance_id": "traversal-0",
                "audio_path": "linked/../sample.wav",
                "audio_sha256": hashlib.sha256(
                    inside_audio.read_bytes()
                ).hexdigest(),
            }

            with self.assertRaisesRegex(ValueError, "parent traversal"):
                verify_speech_audio_rows([row], data_dir=data_dir)


def make_rows(root: Path, group_sizes: tuple[int, ...] = (2, 2, 2, 2)) -> list[dict]:
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for speaker_index, group_size in enumerate(group_sizes, 1):
        speaker = str(100 + speaker_index)
        chapter = str(1000 + speaker_index)
        for utterance_index in range(group_size):
            utterance_id = f"{speaker}-{chapter}-{utterance_index:04d}"
            relative_path = Path("audio") / f"{utterance_id}.wav"
            audio_path = root / relative_path
            write_pcm_wav(audio_path, len(rows))
            rows.append(
                {
                    "id": len(rows),
                    "task": "speech",
                    "source": "openslr/librispeech_asr/clean:train.100",
                    "source_dataset": "openslr/librispeech_asr",
                    "source_config": "clean",
                    "source_split": "train.100",
                    "speaker_id": speaker,
                    "chapter_id": chapter,
                    "utterance_id": utterance_id,
                    "audio_path": str(relative_path),
                    "source_audio_fingerprint": source_fingerprint(utterance_id),
                    "audio_fingerprint": real_subset.path_stat_fingerprint(
                        audio_path, str(relative_path)
                    ),
                    "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                    "transcript": f"transcript {utterance_id}",
                }
            )
    return rows


class SpeechProvenanceTests(unittest.TestCase):
    def test_metadata_or_canonical_path_preserves_source_ids(self) -> None:
        candidate = real_subset.SPEECH_SOURCE_PROFILE[0]
        from_metadata = real_subset.extract_librispeech_provenance(
            {
                "speaker_id": 103,
                "chapter_id": 1240,
                "id": "103-1240-0007",
            },
            candidate,
        )
        from_path = real_subset.extract_librispeech_provenance(
            {"audio": {"path": "/cache/251/136532/251-136532-0014.flac"}},
            candidate,
        )
        self.assertEqual(from_metadata["speaker_id"], "103")
        self.assertEqual(from_metadata["chapter_id"], "1240")
        self.assertEqual(from_metadata["utterance_id"], "103-1240-0007")
        self.assertEqual(from_metadata["source_split"], "train.360")
        self.assertEqual(from_path["utterance_id"], "251-136532-0014")

    def test_missing_or_conflicting_metadata_fails_closed(self) -> None:
        candidate = real_subset.SPEECH_SOURCE_PROFILE[0]
        with self.assertRaisesRegex(real_subset.SpeechProvenanceError, "missing LibriSpeech speaker_id"):
            real_subset.extract_librispeech_provenance(
                {"id": 0, "audio": {"array": [0.0], "sampling_rate": 16000}},
                candidate,
            )
        with self.assertRaisesRegex(real_subset.SpeechProvenanceError, "conflicting LibriSpeech speaker_id"):
            real_subset.extract_librispeech_provenance(
                {
                    "speaker_id": 103,
                    "chapter_id": 1240,
                    "id": "104-1240-0007",
                },
                candidate,
            )


class SpeechPartitionTests(unittest.TestCase):
    def test_partition_is_deterministic_and_speaker_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = make_rows(Path(directory))
            first, first_summary = real_subset.partition_speech_rows(rows, 4, 73)
            second, second_summary = real_subset.partition_speech_rows(rows, 4, 73)

        first_assignment = [(row["utterance_id"], row["partition"]) for row in first]
        second_assignment = [(row["utterance_id"], row["partition"]) for row in second]
        self.assertEqual(first_assignment, second_assignment)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_summary["seed"], 73)
        self.assertEqual(first_summary["row_counts"], {"train": 4, "dev": 2, "eval": 2})
        self.assertEqual(first_summary["group_counts"], {"train": 2, "dev": 1, "eval": 1})
        self.assertEqual(first_summary["overlap_audit"]["pairwise_group_overlap_count"], 0)
        self.assertEqual([row["partition"] for row in first], ["train"] * 4 + ["dev"] * 2 + ["eval"] * 2)

    def test_insufficient_or_inexact_group_scale_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = make_rows(Path(directory), (2, 2))
            with self.assertRaisesRegex(real_subset.SpeechProvenanceError, "at least 3 speaker groups"):
                real_subset.partition_speech_rows(rows, 2, 0)
        with tempfile.TemporaryDirectory() as directory:
            rows = make_rows(Path(directory), (2, 2, 2))
            with self.assertRaisesRegex(real_subset.SpeechProvenanceError, "cannot form the exact"):
                real_subset.partition_speech_rows(rows, 3, 0)


class SpeechAuditTests(unittest.TestCase):
    def test_new_root_passes_and_manifest_counts_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, summary = real_subset.partition_speech_rows(make_rows(root), 4, 11)
            ledger = real_subset.speech_partition_ledger(rows, summary)
            write_jsonl(root / "speech_transcripts.jsonl", rows)
            ledger_text = json.dumps(ledger, indent=2, ensure_ascii=False, sort_keys=True)
            (root / "speech_partition_ledger.json").write_text(ledger_text, encoding="utf-8")
            manifest = {
                "counts": {
                    "speech_transcripts": 8,
                    "speech_train_utterances": 4,
                    "speech_eval_utterances": 4,
                    "speech_dev_utterances": 2,
                    "speech_final_eval_utterances": 2,
                },
                "speech_partition": summary,
                "speech_partition_ledger": {
                    "path": str(root / "speech_partition_ledger.json"),
                    "sha256": hashlib.sha256(ledger_text.encode("utf-8")).hexdigest(),
                    "rows": 8,
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = audit.audit_root(root)
            output = root / "audit.json"
            self.assertEqual(audit.main([str(root), "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["speaker_group_overlap_count"], 0)
        self.assertEqual(report["row_counts"], manifest["speech_partition"]["row_counts"])

    def test_legacy_root_missing_metadata_fails_with_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(
                root / "speech_transcripts.jsonl",
                [{"id": 0, "task": "speech", "audio_path": "audio/legacy.wav"}],
            )
            with self.assertRaisesRegex(audit.AuditError, "missing required provenance field source_dataset"):
                audit.audit_root(root)


if __name__ == "__main__":
    unittest.main()
