#!/usr/bin/env python3
"""Fail-closed audit for real speech provenance and group-disjoint partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


PARTITIONS = ("train", "dev", "eval")
POLICY = "seeded_exact_heldout_speaker_disjoint_v1"
PATH_STAT_SEMANTICS = "path_utf8_size_bytes_mtime_ns_v1_not_audio_content"
SOURCE_FINGERPRINT_SEMANTICS = {
    PATH_STAT_SEMANTICS,
    "encoded_audio_bytes_v1",
    "decoded_audio_array_dtype_shape_rate_bytes_v1",
}
LIBRISPEECH_ID_RE = re.compile(r"^(\d+)-(\d+)-(\d+)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LEDGER_FIELDS = (
    "id", "partition", "source_dataset", "source_config", "source_split",
    "speaker_id", "chapter_id", "utterance_id", "audio_path",
    "source_audio_fingerprint", "audio_fingerprint",
)


class AuditError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing required file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {path.name}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AuditError(f"missing required file: {path.name}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(f"invalid JSONL {path.name}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise AuditError(f"expected object at {path.name}:{line_number}")
        rows.append(row)
    if not rows:
        raise AuditError(f"speech manifest has no rows: {path.name}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_file(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise AuditError(f"audio_path must be a non-empty string, got {value!r}")
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [root / raw, raw, root.parent / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise AuditError(f"missing materialized audio: {value}")


def require_fingerprint(value: Any, *, source: bool, row_index: int) -> dict[str, Any]:
    label = "source_audio_fingerprint" if source else "audio_fingerprint"
    if not isinstance(value, dict):
        raise AuditError(f"speech row {row_index} missing {label}")
    allowed = SOURCE_FINGERPRINT_SEMANTICS if source else {PATH_STAT_SEMANTICS}
    if value.get("algorithm") != "sha256" or value.get("semantics") not in allowed:
        raise AuditError(f"speech row {row_index} has unsupported {label} semantics")
    if not isinstance(value.get("value"), str) or not SHA256_RE.fullmatch(value["value"]):
        raise AuditError(f"speech row {row_index} has invalid {label} value")
    return value


def projected_record(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in LEDGER_FIELDS}


def audit_root(data_root: Path) -> dict[str, Any]:
    root = data_root.resolve()
    rows_path = root / "speech_transcripts.jsonl"
    manifest_path = root / "manifest.json"
    ledger_path = root / "speech_partition_ledger.json"
    rows = load_jsonl(rows_path)

    partition_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in PARTITIONS}
    partition_groups: dict[str, set[str]] = {name: set() for name in PARTITIONS}
    source_ids: set[tuple[str, str]] = set()
    observed_order = []
    for index, row in enumerate(rows):
        for field in (
            "source_dataset", "source_config", "source_split", "speaker_id", "chapter_id",
            "utterance_id", "partition", "audio_path",
        ):
            if row.get(field) in (None, ""):
                raise AuditError(f"speech row {index} missing required provenance field {field}")
        if row.get("task") != "speech":
            raise AuditError(f"speech row {index} has invalid task {row.get('task')!r}")
        if row["source_dataset"] != "openslr/librispeech_asr":
            raise AuditError(
                f"speech row {index} has unsupported source_dataset {row['source_dataset']!r}"
            )
        match = LIBRISPEECH_ID_RE.fullmatch(str(row["utterance_id"]))
        if match is None:
            raise AuditError(f"speech row {index} has invalid LibriSpeech utterance_id")
        if str(int(match.group(1))) != str(row["speaker_id"]):
            raise AuditError(f"speech row {index} utterance_id/speaker_id mismatch")
        if str(int(match.group(2))) != str(row["chapter_id"]):
            raise AuditError(f"speech row {index} utterance_id/chapter_id mismatch")
        partition = str(row["partition"])
        if partition not in partition_rows:
            raise AuditError(f"speech row {index} has unsupported partition {partition!r}")
        source_id = (str(row["source_dataset"]), str(row["utterance_id"]))
        if source_id in source_ids:
            raise AuditError(f"duplicate source utterance ID: {source_id!r}")
        source_ids.add(source_id)
        partition_rows[partition].append(row)
        partition_groups[partition].add(f"{row['source_dataset']}:{row['speaker_id']}")
        observed_order.append(PARTITIONS.index(partition))

        require_fingerprint(row.get("source_audio_fingerprint"), source=True, row_index=index)
        fingerprint = require_fingerprint(row.get("audio_fingerprint"), source=False, row_index=index)
        audio_path = resolve_file(row["audio_path"], root)
        stat = audio_path.stat()
        actual = {
            "path": row["audio_path"],
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if fingerprint.get("value") != sha256_json(actual):
            raise AuditError(f"speech row {index} materialized audio path/stat fingerprint mismatch")
        if fingerprint.get("size_bytes") != stat.st_size or fingerprint.get("mtime_ns") != stat.st_mtime_ns:
            raise AuditError(f"speech row {index} materialized audio stat metadata mismatch")

    if observed_order != sorted(observed_order):
        raise AuditError("speech rows are not ordered train,dev,eval for legacy tail safety")
    if any(not partition_rows[name] for name in PARTITIONS):
        raise AuditError("train/dev/eval speech partitions must all be non-empty")

    manifest = load_json(manifest_path)
    ledger = load_json(ledger_path)
    manifest_summary = manifest.get("speech_partition")
    if not isinstance(manifest_summary, dict):
        raise AuditError("manifest.json missing speech_partition metadata")
    if manifest_summary.get("policy") != POLICY:
        raise AuditError(f"unsupported speech partition policy: {manifest_summary.get('policy')!r}")
    if ledger.get("policy") != POLICY:
        raise AuditError(f"speech ledger has unsupported policy: {ledger.get('policy')!r}")

    overlaps = {
        f"{left}_{right}": sorted(partition_groups[left] & partition_groups[right])
        for left, right in (("train", "dev"), ("train", "eval"), ("dev", "eval"))
    }
    overlap_count = sum(len(values) for values in overlaps.values())
    if overlap_count:
        raise AuditError(f"speaker group overlap detected: {overlaps}")
    row_counts = {name: len(partition_rows[name]) for name in PARTITIONS}
    group_counts = {name: len(partition_groups[name]) for name in PARTITIONS}
    heldout_rows = row_counts["dev"] + row_counts["eval"]
    if manifest_summary.get("row_counts") != row_counts:
        raise AuditError("manifest speech partition row_counts mismatch")
    if manifest_summary.get("group_counts") != group_counts:
        raise AuditError("manifest speech partition group_counts mismatch")
    if manifest_summary.get("heldout_row_target") != heldout_rows:
        raise AuditError("manifest heldout_row_target mismatch")
    overlap_audit = manifest_summary.get("overlap_audit", {})
    if overlap_audit.get("pairwise_group_overlap_count") != 0:
        raise AuditError("manifest overlap audit is not zero")
    if overlap_audit.get("pairwise_group_overlaps") != overlaps:
        raise AuditError("manifest overlap audit details mismatch")

    ledger_summary = {key: value for key, value in ledger.items() if key != "records"}
    if ledger_summary != manifest_summary:
        raise AuditError("speech ledger summary does not match manifest speech_partition")
    expected_records = [projected_record(row) for row in rows]
    if ledger.get("records") != expected_records:
        raise AuditError("speech ledger records do not match speech_transcripts.jsonl")
    ledger_ref = manifest.get("speech_partition_ledger")
    if not isinstance(ledger_ref, dict):
        raise AuditError("manifest.json missing speech_partition_ledger reference")
    if ledger_ref.get("rows") != len(rows):
        raise AuditError("manifest speech ledger row count mismatch")
    if ledger_ref.get("sha256") != sha256_file(ledger_path):
        raise AuditError("manifest speech ledger sha256 mismatch")

    counts = manifest.get("counts", {})
    expected_counts = {
        "speech_transcripts": len(rows),
        "speech_train_utterances": row_counts["train"],
        "speech_eval_utterances": heldout_rows,
        "speech_dev_utterances": row_counts["dev"],
        "speech_final_eval_utterances": row_counts["eval"],
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            raise AuditError(f"manifest count mismatch for {key}: {counts.get(key)!r} != {expected}")
    return {
        "status": "PASS",
        "data_root": str(root),
        "policy": POLICY,
        "seed": manifest_summary.get("seed"),
        "row_counts": row_counts,
        "group_counts": group_counts,
        "speaker_group_overlap_count": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_root(args.data_root)
    except AuditError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
