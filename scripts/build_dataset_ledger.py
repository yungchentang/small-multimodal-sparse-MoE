#!/usr/bin/env python3
"""Recompute a public-safe provenance ledger for a packed multimodal dataset.

The ledger intentionally does not read ``manifest.json``.  Counts, digests,
media presence, and preprocessing claims are recomputed from the actual inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import tempfile
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2
LEDGER_TYPE = "recomputed_dataset_ledger"
TASK_LABELS = ("text", "code", "reasoning", "math", "education")
MINIMUM_BLOCKS = {"text": 10_000, "code": 1_000, "reasoning": 1, "math": 1, "education": 1}
REF_RE = re.compile(r"^[0-9a-f]{40}$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_COLOR_MODES = {0: "L", 2: "RGB", 4: "LA", 6: "RGBA"}


class LedgerError(ValueError):
    """Raised when provenance cannot be independently established."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return "external/" + fingerprint(path.name)[:16]


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(f"invalid JSONL {path.name}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise LedgerError(f"invalid JSONL {path.name}:{line_number}: object required")
            rows.append(row)
    return rows, {"rows": len(rows), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def path_exists(path: Path) -> bool:
    """Return True for regular paths and broken symlinks."""

    return path.exists() or path.is_symlink()


def ensure_output_available(path: Path, force: bool) -> None:
    """Refuse an existing ledger before shard materialization can begin."""

    if not path_exists(path):
        return
    if not force:
        raise LedgerError(f"refusing to overwrite existing ledger: {path}")
    if path.is_dir():
        raise LedgerError(f"ledger output is a directory: {path}")


def dataset_file_record(path: Path, rows: int) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise LedgerError(f"dataset file is missing or not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "rows": rows,
    }


def canonical_jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for row in rows
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def packed_shard_path(output: Path, task: str) -> Path:
    return output.parent / f"{output.stem}.packed.{task}.jsonl"


def preflight_materializations(materializations: list[tuple[Path, bytes]], force: bool) -> None:
    """Check every destination before writing any shard."""

    for path, payload in materializations:
        if not path_exists(path):
            continue
        if path.is_dir():
            raise LedgerError(f"packed shard destination is a directory: {path}")
        if not path.is_symlink() and path.is_file() and path.read_bytes() == payload:
            continue
        if not force:
            raise LedgerError(f"refusing to overwrite existing packed shard: {path}")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    if not path.is_symlink() and path.is_file() and path.read_bytes() == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=path.name + ".", delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def materialize_packed_shards(
    output: Path,
    rows: list[dict[str, Any]],
    force: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    rows_by_task = {task: [] for task in TASK_LABELS}
    for row in rows:
        rows_by_task[str(row["task"])].append(row)
    payloads = {task: canonical_jsonl_bytes(rows_by_task[task]) for task in TASK_LABELS}
    paths = {task: packed_shard_path(output, task).resolve() for task in TASK_LABELS}
    materializations = [(paths[task], payloads[task]) for task in TASK_LABELS]
    preflight_materializations(materializations, force)
    for shard_path, payload in materializations:
        write_bytes_atomic(shard_path, payload)
    counts = {task: len(rows_by_task[task]) for task in TASK_LABELS}
    records = {task: dataset_file_record(paths[task], counts[task]) for task in TASK_LABELS}
    return records, counts


def resolve_media(value: Any, data_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise LedgerError("media path must be a non-empty string")
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [data_root / raw, raw, data_root.parent / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise LedgerError(f"missing media file: {value}")


def png_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(29)
    if len(header) != 29 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise LedgerError(f"unsupported or invalid PNG: {path.name}")
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", header[16:29]
    )
    if color_type not in PNG_COLOR_MODES or bit_depth != 8 or compression != 0 or filter_method != 0:
        raise LedgerError(f"unsupported PNG preprocessing: {path.name}")
    return {"width": width, "height": height, "mode": PNG_COLOR_MODES[color_type], "interlace": interlace}


def wav_metadata(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as handle:
            return {
                "sample_rate": handle.getframerate(),
                "channels": handle.getnchannels(),
                "num_samples": handle.getnframes(),
                "sample_width": handle.getsampwidth(),
                "compression": handle.getcomptype(),
            }
    except (wave.Error, EOFError) as exc:
        raise LedgerError(f"invalid WAV {path.name}: {exc}") from exc


def checked_indexes(total: int, mode: str, sample_size: int) -> set[int]:
    if mode == "full" or total <= sample_size:
        return set(range(total))
    # Input order is deterministic.  Spreading picks avoids a prefix-only sample.
    return {(index * total) // sample_size for index in range(sample_size)}


def require_task_rows(rows: Iterable[dict[str, Any]], allowed: set[str], label: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        task = row.get("task")
        if task not in allowed:
            raise LedgerError(f"{label} row {index} has unsupported task {task!r}")
        counts[str(task)] += 1
    return counts


def media_ledger(
    rows: list[dict[str, Any]],
    *,
    data_root: Path,
    field: str,
    expected_task: str,
    media_type: str,
    check_mode: str,
    sample_size: int,
    expected_image_size: tuple[int, int],
    expected_image_mode: str,
    expected_audio_rate: int,
    expected_audio_channels: int,
    expected_audio_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require_task_rows(rows, {expected_task}, media_type)
    paths: list[Path] = []
    hashes: list[str] = []
    pair_hashes: list[str] = []
    checked = checked_indexes(len(rows), check_mode, sample_size)
    for index, row in enumerate(rows):
        path = resolve_media(row.get(field), data_root)
        paths.append(path)
        media_hash = sha256_file(path)
        hashes.append(media_hash)
        pair_hashes.append(
            fingerprint(
                {
                    'media_sha256': media_hash,
                    'target': row.get('caption') if media_type == 'image' else row.get('transcript'),
                }
            )
        )
        if index not in checked:
            continue
        if media_type == "image":
            actual = png_metadata(path)
            if (actual["width"], actual["height"]) != expected_image_size or actual["mode"] != expected_image_mode:
                raise LedgerError(
                    f"image preprocessing mismatch for {path.name}: "
                    f"got {actual['width']}x{actual['height']} {actual['mode']}"
                )
        else:
            actual = wav_metadata(path)
            expected = (expected_audio_rate, expected_audio_channels, expected_audio_samples)
            observed = (actual["sample_rate"], actual["channels"], actual["num_samples"])
            if observed != expected or actual["compression"] != "NONE":
                raise LedgerError(f"audio preprocessing mismatch for {path.name}: got {observed}")
    if len(set(paths)) != len(paths):
        raise LedgerError(f"duplicate {media_type} media reference")
    if len(set(pair_hashes)) != len(pair_hashes):
        raise LedgerError(f"duplicate {media_type} media-target pair")
    if media_type != "image" and len(set(hashes)) != len(hashes):
        raise LedgerError(f"duplicate {media_type} media content hash")
    public_rows = [
        {"media_sha256": digest, "source_ids": row_source_ids(row)}
        for digest, row in zip(hashes, rows)
    ]
    return (
        {
            "referenced_rows": len(rows),
            "unique_paths": len(set(paths)),
            "unique_media_sha256": len(set(hashes)),
            "duplicate_content_rows": len(hashes) - len(set(hashes)),
            "unique_media_target_pairs": len(set(pair_hashes)),
            "checked_rows": len(checked),
            "validation_mode": check_mode,
            "aggregate_media_sha256": fingerprint(sorted(hashes)),
            "aggregate_media_target_sha256": fingerprint(sorted(pair_hashes)),
        },
        public_rows,
    )


def row_source_ids(row: dict[str, Any]) -> set[str]:
    values = row.get("source_ids", row.get("source_id", []))
    if isinstance(values, str):
        return {values}
    if isinstance(values, list) and all(isinstance(value, str) for value in values):
        return set(values)
    return set()


def train_count(rows: list[dict[str, Any]], eval_tail: int, label: str) -> tuple[int, int, str]:
    explicit = [row.get("split_role", row.get("split")) for row in rows]
    if any(value is not None for value in explicit):
        train = sum(1 for value in explicit if value == "train")
        eval_count = len(rows) - train
        return train, eval_count, "explicit_row_split"
    if eval_tail < 0 or eval_tail > len(rows):
        raise LedgerError(f"{label} eval tail is outside row count")
    return len(rows) - eval_tail, eval_tail, "deterministic_tail"


def assert_minimums(block_counts: Counter[str], image_train: int, speech_train: int, allow_short: bool) -> None:
    if allow_short:
        return
    failures = [f"{task}={block_counts[task]} < {minimum}" for task, minimum in MINIMUM_BLOCKS.items() if block_counts[task] < minimum]
    if image_train < 5_000:
        failures.append(f"image_train_pairs={image_train} < 5000")
    if speech_train < 5_000:
        failures.append(f"speech_train_utterances={speech_train} < 5000")
    if failures:
        raise LedgerError("final-scale minimum gates failed: " + "; ".join(failures))


def index_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, evidence = load_jsonl(path)
    return rows, {"relative_path": path.name, **evidence}


def sealed_ledger(
    sealed_root: Path | None,
    reference_indexes: list[Path],
    candidate_media: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    paths = list(reference_indexes)
    if sealed_root is not None:
        root_index = sealed_root / "sealed_eval_index.jsonl"
        if not root_index.is_file():
            raise LedgerError("sealed root must contain sealed_eval_index.jsonl")
        paths.append(root_index)
    if not paths:
        return None
    unique_paths = sorted({path.resolve() for path in paths})
    evidence: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for path in unique_paths:
        rows, file_evidence = index_records(path)
        evidence.append(file_evidence)
        reference_rows.extend(rows)
    reference_hashes = {
        value
        for row in reference_rows
        for key, value in row.items()
        if key in {"media_sha256", "content_sha256", "resized_content_sha256"}
        and isinstance(value, str)
    }
    reference_ids = set().union(*(row_source_ids(row.get("source", row)) for row in reference_rows)) if reference_rows else set()
    assertions: dict[str, Any] = {}
    for modality, rows in candidate_media.items():
        hashes = {row["media_sha256"] for row in rows}
        source_ids = set().union(*(row["source_ids"] for row in rows)) if rows else set()
        hash_overlap = hashes & reference_hashes
        source_overlap = source_ids & reference_ids
        report = {
            "candidate_count": len(rows),
            "hash_overlap_count": len(hash_overlap),
            "source_id_overlap_count": len(source_overlap),
            "passed": not hash_overlap and not source_overlap,
        }
        assertions[modality] = report
        if not report["passed"]:
            raise LedgerError(f"sealed overlap failure for {modality}: hashes={len(hash_overlap)} source_ids={len(source_overlap)}")
    return {"reference_indexes": evidence, "reference_rows": len(reference_rows), "overlap_assertions": assertions}


def registry_entries(registry_path: Path) -> dict[str, dict[str, Any]]:
    try:
        document = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerError(f"external_cache_ref_provenance: invalid registry: {exc}") from exc
    raw = document.get("repos", document) if isinstance(document, dict) else None
    if not isinstance(raw, dict):
        raise LedgerError("external_cache_ref_provenance: registry must contain a repos object")
    entries: dict[str, dict[str, Any]] = {}
    for repo, entry in raw.items():
        if not isinstance(repo, str) or not isinstance(entry, dict):
            raise LedgerError("external_cache_ref_provenance: invalid repo entry")
        ref = entry.get("ref")
        mtime_ns = entry.get("mtime_ns")
        if not isinstance(ref, str) or REF_RE.fullmatch(ref) is None or not isinstance(mtime_ns, int):
            raise LedgerError(f"external_cache_ref_provenance: {repo} needs exact ref and mtime_ns")
        entries[repo] = entry
    return entries


def hf_cache_ledger(cache_root: Path | None, registry_path: Path | None) -> dict[str, Any] | None:
    if cache_root is None and registry_path is None:
        return None
    if cache_root is None or registry_path is None:
        raise LedgerError("external_cache_ref_provenance: --hf-cache-root and --hf-ref-registry must be supplied together")
    entries = registry_entries(registry_path)
    checked: list[dict[str, Any]] = []
    for repo, entry in sorted(entries.items()):
        directory = entry.get("cache_dir", "datasets--" + repo.replace("/", "--"))
        ref_name = entry.get("ref_name", "main")
        if not isinstance(directory, str) or not isinstance(ref_name, str) or Path(directory).is_absolute() or ".." in Path(directory).parts:
            raise LedgerError(f"external_cache_ref_provenance: unsafe cache location for {repo}")
        ref_path = cache_root / directory / "refs" / ref_name
        if not ref_path.is_file():
            raise LedgerError(f"external_cache_ref_provenance: missing cache ref for {repo}")
        actual_ref = ref_path.read_text(encoding="utf-8").strip()
        actual_mtime = ref_path.stat().st_mtime_ns
        if actual_ref != entry["ref"] or actual_mtime != entry["mtime_ns"]:
            raise LedgerError(f"external_cache_ref_provenance: ref mismatch for {repo}")
        checked.append({"repo_fingerprint": fingerprint(repo), "ref": actual_ref, "mtime_ns": actual_mtime})
    return {"kind": "external_cache_ref_provenance", "registry_sha256": sha256_file(registry_path), "checked_repos": checked}


def build_ledger(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    if not data_root.is_dir():
        raise LedgerError("data root does not exist")
    filenames = {
        "text_tasks": "text_tasks.jsonl",
        "text_blocks_train": "text_blocks_train.jsonl",
        "text_blocks_eval": "text_blocks_eval.jsonl",
        "image_captions": "image_captions.jsonl",
        "speech_transcripts": "speech_transcripts.jsonl",
    }
    rows: dict[str, list[dict[str, Any]]] = {}
    files: dict[str, dict[str, Any]] = {}
    for label, name in filenames.items():
        path = data_root / name
        if not path.is_file():
            raise LedgerError(f"missing required JSONL: {name}")
        rows[label], files[label] = load_jsonl(path)
        files[label]["relative_path"] = name
    task_counts = require_task_rows(rows["text_tasks"], set(TASK_LABELS), "text_tasks")
    train_blocks = require_task_rows(rows["text_blocks_train"], set(TASK_LABELS), "text_blocks_train")
    require_task_rows(rows["text_blocks_eval"], set(TASK_LABELS), "text_blocks_eval")
    for index, row in enumerate(rows["text_blocks_train"]):
        if not isinstance(row.get("input_ids"), list):
            raise LedgerError(f"text_blocks_train row {index} is not a packed token block")
    image_summary, image_public = media_ledger(
        rows["image_captions"], data_root=data_root, field="image_path", expected_task="image", media_type="image",
        check_mode=args.media_validation, sample_size=args.media_sample_size,
        expected_image_size=(args.image_width, args.image_height), expected_image_mode=args.image_mode,
        expected_audio_rate=args.audio_sample_rate, expected_audio_channels=args.audio_channels, expected_audio_samples=args.audio_num_samples,
    )
    speech_summary, speech_public = media_ledger(
        rows["speech_transcripts"], data_root=data_root, field="audio_path", expected_task="speech", media_type="speech",
        check_mode=args.media_validation, sample_size=args.media_sample_size,
        expected_image_size=(args.image_width, args.image_height), expected_image_mode=args.image_mode,
        expected_audio_rate=args.audio_sample_rate, expected_audio_channels=args.audio_channels, expected_audio_samples=args.audio_num_samples,
    )
    image_train, image_eval, image_policy = train_count(rows["image_captions"], args.image_eval_tail, "image")
    speech_train, speech_eval, speech_policy = train_count(rows["speech_transcripts"], args.speech_eval_tail, "speech")
    assert_minimums(train_blocks, image_train, speech_train, args.allow_short)
    sealed = sealed_ledger(
        Path(args.sealed_root).resolve() if args.sealed_root else None,
        [Path(path).resolve() for path in args.sealed_reference_index],
        {"image": image_public, "speech": speech_public},
    )
    external = hf_cache_ledger(
        Path(args.hf_cache_root).resolve() if args.hf_cache_root else None,
        Path(args.hf_ref_registry).resolve() if args.hf_ref_registry else None,
    )
    packed_records, packed_counts = materialize_packed_shards(
        output, rows["text_blocks_train"], args.force
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "ledger_type": LEDGER_TYPE,
        "files": {
            "image_rows": dataset_file_record(
                data_root / filenames["image_captions"], len(rows["image_captions"])
            ),
            "speech_rows": dataset_file_record(
                data_root / filenames["speech_transcripts"], len(rows["speech_transcripts"])
            ),
            "packed_tasks": packed_records,
        },
        "counts": {
            "image_rows": len(rows["image_captions"]),
            "speech_rows": len(rows["speech_transcripts"]),
            "packed_task_counts": packed_counts,
            "packed_rows": sum(packed_counts.values()),
        },
        "provenance_policy": "recomputed_from_inputs_no_manifest_counts_or_revisions",
        "data_root_fingerprint": fingerprint(relative_path(data_root, data_root.parent)),
        "jsonl_files": files,
        "task_counts": {"text_tasks": dict(sorted(task_counts.items())), "packed_train_blocks": dict(sorted(train_blocks.items()))},
        "media": {
            "image": {**image_summary, "train_pairs": image_train, "eval_pairs": image_eval, "split_policy": image_policy},
            "speech": {**speech_summary, "train_utterances": speech_train, "eval_utterances": speech_eval, "split_policy": speech_policy},
        },
        "expected_preprocessing": {
            "image": {"width": args.image_width, "height": args.image_height, "mode": args.image_mode},
            "speech": {"sample_rate": args.audio_sample_rate, "channels": args.audio_channels, "num_samples": args.audio_num_samples},
        },
        "minimum_gate": {"allow_short": args.allow_short, "passed": True},
        "sealed": sealed,
        "external_cache_ref_provenance": external,
    }


def write_atomic(path: Path, document: dict[str, Any], force: bool) -> None:
    ensure_output_available(path, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-short", action="store_true", help="debug only: bypass final-scale minimum gates")
    parser.add_argument("--media-validation", choices=("full", "sample"), default="full")
    parser.add_argument("--media-sample-size", type=int, default=100)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-mode", default="RGB")
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-channels", type=int, default=1)
    parser.add_argument("--audio-num-samples", type=int, default=96000)
    parser.add_argument("--image-eval-tail", type=int, default=250)
    parser.add_argument("--speech-eval-tail", type=int, default=250)
    parser.add_argument("--sealed-root")
    parser.add_argument("--sealed-reference-index", action="append", default=[])
    parser.add_argument("--hf-cache-root")
    parser.add_argument("--hf-ref-registry")
    args = parser.parse_args(argv)
    if args.media_sample_size <= 0:
        parser.error("--media-sample-size must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output).resolve()
    try:
        ensure_output_available(output, args.force)
        ledger = build_ledger(args)
        write_atomic(output, ledger, args.force)
    except LedgerError as exc:
        raise SystemExit(f"dataset ledger failed: {exc}") from exc


if __name__ == "__main__":
    main()
