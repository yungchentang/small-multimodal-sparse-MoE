#!/usr/bin/env python3
"""Development-only native Whisper WER diagnostic with strict provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import unicodedata
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hf_sources import load_pretrained, resolve_model
from scripts.development_split_provenance import (
    secure_speech_audio_snapshot,
    speech_source_row_identity,
)


MODEL_ID = "openai/whisper-base.en"
MODEL_REVISION = "911407f4214e0e1d82085af863093ec0b66f9cd6"
SEED = 42
TARGET_SAMPLE_RATE = 16_000
DEFAULT_MAX_SECONDS = 6.0
DEFAULT_EXPECTED_ROWS = 137
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_PATH_TERMS = ("sealed", "synthetic")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def normalize_text(text: str) -> str:
    """NFKC-casefold text, replace punctuation/symbols with spaces, collapse whitespace."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    characters = [
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    ]
    return " ".join("".join(characters).split())


def levenshtein_counts(
    reference_words: Sequence[str], hypothesis_words: Sequence[str]
) -> dict[str, int]:
    """Return deterministic word edit counts, preferring S, then D, then I on ties."""
    previous = [(0, 0, index) for index in range(len(hypothesis_words) + 1)]
    for ref_index, reference_word in enumerate(reference_words, start=1):
        current = [(0, ref_index, 0)]
        for hyp_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            if reference_word == hypothesis_word:
                current.append(previous[hyp_index - 1])
                continue
            substitution = previous[hyp_index - 1]
            deletion = previous[hyp_index]
            insertion = current[hyp_index - 1]
            candidates = (
                ((sum(substitution) + 1, 0), (substitution[0] + 1, substitution[1], substitution[2])),
                ((sum(deletion) + 1, 1), (deletion[0], deletion[1] + 1, deletion[2])),
                ((sum(insertion) + 1, 2), (insertion[0], insertion[1], insertion[2] + 1)),
            )
            current.append(min(candidates, key=lambda item: item[0])[1])
        previous = current
    substitutions, deletions, insertions = previous[-1]
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def word_error_record(reference: str, hypothesis: str) -> dict[str, Any]:
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()
    if not reference_words:
        raise ValueError("WER reference is empty after normalization")
    counts = levenshtein_counts(reference_words, hypothesis_words)
    errors = sum(counts.values())
    return {
        "reference_normalized": normalized_reference,
        "hypothesis_normalized": normalized_hypothesis,
        **counts,
        "reference_words": len(reference_words),
        "word_errors": errors,
        "wer": errors / len(reference_words),
    }


def preprocess_audio_payload(
    payload: bytes,
    *,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> tuple[np.ndarray, dict[str, Any]]:
    import io

    if target_sample_rate <= 0 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("audio preprocessing requires positive sample rate and duration")
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            original_channels = int(wav_file.getnchannels())
            sample_width = int(wav_file.getsampwidth())
            original_sample_rate = int(wav_file.getframerate())
            original_samples = int(wav_file.getnframes())
            if wav_file.getcomptype() != "NONE":
                raise ValueError("compressed WAV audio is unsupported")
            frames = wav_file.readframes(original_samples)
    except wave.Error as exc:
        raise ValueError("audio payload is not a valid uncompressed PCM WAV") from exc
    if (
        original_samples <= 0
        or original_channels <= 0
        or original_sample_rate <= 0
        or sample_width not in {1, 2, 3, 4}
    ):
        raise ValueError("decoded audio is empty or has an invalid sample rate")
    expected_bytes = original_samples * original_channels * sample_width
    if len(frames) != expected_bytes:
        raise ValueError("decoded WAV frame count does not match its header")
    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        triples = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        integers = (
            triples[:, 0].astype(np.int32)
            | (triples[:, 1].astype(np.int32) << 8)
            | (triples[:, 2].astype(np.int32) << 16)
        )
        integers = np.where(integers & 0x800000, integers - 0x1000000, integers)
        audio = integers.astype(np.float32) / 8388608.0
    else:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    audio = audio.reshape(original_samples, original_channels)
    if not np.isfinite(audio).all():
        raise ValueError("decoded audio contains non-finite samples")
    mono = audio.mean(axis=1, dtype=np.float32)
    resampled = int(original_sample_rate) != target_sample_rate
    if resampled:
        output_samples = max(
            1, int(round(len(mono) * target_sample_rate / int(original_sample_rate)))
        )
        source_positions = np.arange(len(mono), dtype=np.float64)
        target_positions = np.arange(output_samples, dtype=np.float64) * (
            int(original_sample_rate) / target_sample_rate
        )
        mono = np.interp(target_positions, source_positions, mono).astype(np.float32)
    max_samples = int(round(target_sample_rate * max_seconds))
    truncated = len(mono) > max_samples
    if truncated:
        mono = mono[:max_samples]
    metadata = {
        "original_samples": original_samples,
        "original_sample_rate": int(original_sample_rate),
        "original_channels": original_channels,
        "processed_samples": int(len(mono)),
        "processed_sample_rate": target_sample_rate,
        "resampled": resampled,
        "truncated": truncated,
    }
    return np.ascontiguousarray(mono, dtype=np.float32), metadata


def reject_forbidden_path(path: Path, label: str) -> None:
    lowered = str(path).lower()
    for term in FORBIDDEN_PATH_TERMS:
        if term in lowered:
            raise ValueError(f"{label} contains forbidden {term!r} path")


def require_canonical_file(path: Path, label: str) -> Path:
    reject_forbidden_path(path, label)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute regular non-symlink file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{label} must be canonical")
    return resolved


def require_canonical_directory(path: Path, label: str) -> Path:
    reject_forbidden_path(path, label)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be an absolute canonical directory")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{label} must be canonical")
    return resolved


def validate_output_path(path: Path) -> Path:
    reject_forbidden_path(path, "output directory")
    if not path.is_absolute():
        raise ValueError("output directory must be absolute")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing preexisting output path: {path}")
    parent = require_canonical_directory(path.parent, "output parent")
    return parent / path.name


def verify_runtime_identity(
    repo_root: Path,
    source_commit_sha: str,
    environ: Mapping[str, str],
) -> dict[str, str]:
    if COMMIT_RE.fullmatch(source_commit_sha) is None:
        raise ValueError("source commit must be an exact lowercase 40-hex SHA")
    if environ.get("SOURCE_COMMIT_SHA") != source_commit_sha:
        raise ValueError("SOURCE_COMMIT_SHA must exactly match --source-commit-sha")
    runai_job_name = environ.get("RUNAI_JOB_NAME", "").strip()
    runai_project = environ.get("RUNAI_PROJECT", "").strip()
    if not runai_job_name or not runai_project:
        raise ValueError("RUNAI_JOB_NAME and RUNAI_PROJECT are required")
    actual_commit = subprocess.run(
        ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != source_commit_sha:
        raise ValueError(
            f"source commit mismatch: expected {source_commit_sha}, found {actual_commit}"
        )
    dirty = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("native Whisper WER requires a clean source worktree")
    return {
        "source_commit_sha": source_commit_sha,
        "runai_job_name": runai_job_name,
        "runai_project": runai_project,
    }


def load_speech_dev(
    data_dir: Path,
    split_manifest: Path,
    source_commit_sha: str,
    trusted_speech_source_sha256: str,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if SHA256_RE.fullmatch(trusted_speech_source_sha256) is None:
        raise ValueError("trusted speech source SHA must be exact lowercase SHA256")
    from training.olmoe_real_subset_runs import (
        load_development_multimodal_partitions,
    )

    _, _, speech_train, speech_dev, provenance = (
        load_development_multimodal_partitions(
            split_manifest,
            expected_source_commit_sha=source_commit_sha,
            expected_data_dir=data_dir,
            expected_speech_source_sha256=trusted_speech_source_sha256,
        )
    )
    if len(speech_dev) != expected_rows:
        raise ValueError(
            f"speech_dev row count mismatch: expected {expected_rows}, found {len(speech_dev)}"
        )
    if provenance.get("reserved_files_opened") is not False:
        raise ValueError("loader provenance does not prove reserved eval stayed unopened")
    speech_eval = provenance.get("files", {}).get("speech_eval", {})
    if (
        speech_eval.get("content_opened") is not False
        or speech_eval.get("read_status") != "reserved_unread"
    ):
        raise ValueError("speech_eval must remain metadata-only and unopened")
    if provenance.get("selection_splits") != ["train", "dev"]:
        raise ValueError("loader provenance has unexpected development selection")
    provenance = dict(provenance)
    provenance["wer_selection"] = {
        "split": "speech_dev",
        "rows": len(speech_dev),
        "speech_train_rows_verified_not_evaluated": len(speech_train),
        "speech_eval_metadata_only_unopened": True,
    }
    return speech_dev, provenance


def load_whisper_model(device: str) -> tuple[Any, Any, dict[str, str]]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    try:
        from transformers import WhisperForConditionalGeneration

        model_factory: Any = WhisperForConditionalGeneration
    except ImportError:
        model_factory = AutoModelForSpeechSeq2Seq
    ref = resolve_model(MODEL_ID)
    if ref.revision != MODEL_REVISION:
        raise ValueError("hf_sources.py Whisper revision does not match evaluator pin")
    processor = load_pretrained(AutoProcessor, ref)
    model = load_pretrained(model_factory, ref)
    model.to(torch.device(device))
    model.eval()
    return processor, model, ref.as_dict()


def whisper_processor_n_samples(processor: Any) -> int:
    feature_extractor = getattr(processor, "feature_extractor", None)
    value = getattr(feature_extractor, "n_samples", None)
    if value is None:
        value = getattr(processor, "n_samples", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("Whisper processor must expose a positive integer n_samples")
    return value


def evaluate_speech_dev(
    rows: Sequence[Mapping[str, Any]],
    *,
    data_dir: Path,
    processor: Any,
    model: Any,
    device: str,
    batch_size: int,
    max_seconds: float,
) -> list[dict[str, Any]]:
    import torch

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    results: list[dict[str, Any]] = []
    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        waveforms: list[np.ndarray] = []
        pending: list[dict[str, Any]] = []
        for offset, row in enumerate(batch_rows):
            row_index = batch_start + offset
            raw_path = row.get("audio_path")
            expected_sha = row.get("audio_sha256")
            reference = row.get("transcript")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"speech_dev row {row_index} has no audio_path")
            if not isinstance(expected_sha, str) or SHA256_RE.fullmatch(expected_sha) is None:
                raise ValueError(f"speech_dev row {row_index} has no exact audio_sha256")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"speech_dev row {row_index} has no transcript")
            audio_path, payload = secure_speech_audio_snapshot(
                raw_path, data_dir=data_dir, row_index=row_index
            )
            actual_sha = sha256_bytes(payload)
            if actual_sha != expected_sha:
                raise ValueError(f"speech_dev row {row_index} audio SHA256 mismatch")
            waveform, audio_metadata = preprocess_audio_payload(
                payload,
                target_sample_rate=TARGET_SAMPLE_RATE,
                max_seconds=max_seconds,
            )
            waveforms.append(waveform)
            pending.append(
                {
                    "row_index": row_index,
                    "row_identity_sha256": speech_source_row_identity(row, row_index),
                    "source_dataset": row.get("source_dataset"),
                    "utterance_id": row.get("utterance_id"),
                    "audio_path": str(audio_path),
                    "audio_sha256": actual_sha,
                    "reference": reference,
                    "audio": audio_metadata,
                }
            )
        processor_n_samples = whisper_processor_n_samples(processor)
        encoded = processor(
            waveforms,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=processor_n_samples,
            return_attention_mask=True,
        )
        model_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs)
        hypotheses = processor.batch_decode(generated_ids, skip_special_tokens=True)
        if len(hypotheses) != len(pending):
            raise ValueError("Whisper returned a different number of transcripts")
        for record, hypothesis in zip(pending, hypotheses):
            results.append(
                {
                    **record,
                    "hypothesis": str(hypothesis),
                    **word_error_record(record["reference"], str(hypothesis)),
                }
            )
    return results


def summarize_results(
    results: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    totals = {
        key: sum(int(row[key]) for row in results)
        for key in ("substitutions", "deletions", "insertions", "reference_words")
    }
    if totals["reference_words"] <= 0:
        raise ValueError("total reference word count must be positive")
    word_errors = (
        totals["substitutions"] + totals["deletions"] + totals["insertions"]
    )
    return {
        "schema_version": 1,
        "diagnostic": "native_whisper_development_wer",
        "split": "speech_dev",
        "row_count": len(results),
        **totals,
        "word_errors": word_errors,
        "wer": word_errors / totals["reference_words"],
        "provenance": dict(provenance),
    }


def atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        temporary_path.unlink()
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_outputs_exclusively(
    output_dir: Path,
    results: Sequence[Mapping[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(mode=0o750, parents=False, exist_ok=False)
    per_payload = b"".join(canonical_json_bytes(dict(row)) for row in results)
    per_hash = sha256_bytes(per_payload)
    metrics["provenance"]["output_hashes"] = {
        "per_utterance_jsonl_sha256": per_hash,
        "metrics_json_content_sha256": None,
        "metrics_json_hash_scope": (
            "canonical metrics JSON with metrics_json_content_sha256 set to null"
        ),
    }
    content_hash = sha256_bytes(canonical_json_bytes(metrics))
    metrics["provenance"]["output_hashes"]["metrics_json_content_sha256"] = content_hash
    metrics_payload = canonical_json_bytes(metrics)
    per_path = output_dir / "per_utterance.jsonl"
    metrics_path = output_dir / "metrics.json"
    atomic_write(per_path, per_payload)
    atomic_write(metrics_path, metrics_payload)
    return {
        "per_utterance_jsonl_sha256": sha256_file(per_path),
        "metrics_json_sha256": sha256_file(metrics_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--development-split-manifest", type=Path, required=True)
    parser.add_argument("--trusted-speech-source-sha256", required=True)
    parser.add_argument("--source-commit-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    if args.expected_rows <= 0:
        raise ValueError("expected speech_dev rows must be positive")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("native Whisper WER requires the assigned CUDA GPU")
    script_path = Path(__file__).resolve(strict=True)
    repo_root = script_path.parents[1]
    runtime_identity = verify_runtime_identity(
        repo_root, args.source_commit_sha, os.environ
    )
    data_dir = require_canonical_directory(args.data_dir, "data_dir")
    split_manifest = require_canonical_file(
        args.development_split_manifest, "development split manifest"
    )
    output_dir = validate_output_path(args.output_dir)
    speech_dev, loader_provenance = load_speech_dev(
        data_dir,
        split_manifest,
        args.source_commit_sha,
        args.trusted_speech_source_sha256,
        args.expected_rows,
    )
    split_manifest_sha256 = sha256_file(split_manifest)
    if loader_provenance.get("manifest_sha256") != split_manifest_sha256:
        raise ValueError("development split manifest changed after loader snapshot")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    processor, model, model_ref = load_whisper_model(args.device)
    processor_n_samples = whisper_processor_n_samples(processor)
    results = evaluate_speech_dev(
        speech_dev,
        data_dir=data_dir,
        processor=processor,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
        max_seconds=args.max_seconds,
    )
    provenance = {
        **runtime_identity,
        "evaluator_path": str(script_path),
        "evaluator_sha256": sha256_file(script_path),
        "data_dir": str(data_dir),
        "development_split_manifest_path": str(split_manifest),
        "development_split_manifest_sha256": split_manifest_sha256,
        "trusted_speech_source_sha256": args.trusted_speech_source_sha256,
        "model": model_ref,
        "seed": SEED,
        "preprocessing": {
            "decode": "stdlib_wave_uncompressed_pcm_to_float32_then_channel_mean",
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "resampling": "deterministic_numpy_linear_interpolation",
            "max_seconds": args.max_seconds,
            "truncate_order": "after_resampling_before_processor",
            "processor_padding": "max_length",
            "processor_truncation": True,
            "processor_max_length_samples": processor_n_samples,
            "processor_return_attention_mask": True,
            "batch_size": args.batch_size,
        },
        "text_normalization": (
            "unicode_NFKC_then_casefold_then_punctuation_and_symbols_to_space_"
            "then_whitespace_collapse"
        ),
        "loader_provenance": loader_provenance,
        "sealed_evidence_used": False,
        "synthetic_evidence_used": False,
    }
    metrics = summarize_results(results, provenance)
    output_hashes = write_outputs_exclusively(output_dir, results, metrics)
    print(json.dumps({"output_dir": str(output_dir), **output_hashes}, sort_keys=True))


if __name__ == "__main__":
    main()
