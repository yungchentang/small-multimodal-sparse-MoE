#!/usr/bin/env python3
"""Freeze and verify a fail-closed sealed-evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

try:
    from scripts.sealed_position_allocator import (
        ALLOCATOR_NAME,
        ALLOCATOR_VERSION,
        AssignmentPlanError,
        build_allocator_manifest,
        validate_allocator_manifest,
    )
except ImportError:  # Direct execution from the scripts directory.
    from sealed_position_allocator import (  # type: ignore[no-redef]
        ALLOCATOR_NAME,
        ALLOCATOR_VERSION,
        AssignmentPlanError,
        build_allocator_manifest,
        validate_allocator_manifest,
    )


SCHEMA_VERSION = 2
PROTOCOL_NAME = "sealed_evaluation_protocol"
REQUIRED_CONTROLS = ("real", "shuffled", "zero", "norm-matched-random", "no-prefix")
SEALED_METRICS_POLICY = (
    "Sealed metrics cannot trigger retraining, checkpoint reselection, candidate reselection, "
    "hyperparameter changes, protocol changes, or additional candidate screening."
)
DEFAULT_PAIRED_TESTS = (
    "paired exact McNemar test for R@1",
    "paired sign-flip permutation test for mean reciprocal rank",
    "paired bootstrap 95% confidence interval for metric differences",
)
FIXED_CLAIM_DECISION_RULES = (
    {
        "id": "complete_predeclared_matrix",
        "rule": (
            "No claim is permitted unless every frozen modality, candidate size, candidate "
            "protocol, control, and query count is complete and passes integrity checks."
        ),
    },
    {
        "id": "paired_positive_effect",
        "rule": (
            "A positive claim requires the real-prefix condition to improve in the declared "
            "direction over every control on aligned queries, with two-sided alpha=0.05 after "
            "Holm correction across the frozen primary comparisons."
        ),
    },
    {
        "id": "confidence_interval_excludes_zero",
        "rule": (
            "The paired 95% confidence interval for each claimed primary effect must exclude "
            "zero in the claimed direction."
        ),
    },
    {
        "id": "hard_negative_required",
        "rule": (
            "Generalization claims require the frozen hard-negative protocol to pass; random "
            "or stride-only candidate results are insufficient."
        ),
    },
    {
        "id": "no_adaptation_from_sealed_metrics",
        "rule": SEALED_METRICS_POLICY,
    },
)


class ProtocolError(ValueError):
    """Raised when a protocol cannot be frozen or verified safely."""


@dataclass(frozen=True)
class FreezeConfig:
    selected_root: Path
    checkpoint: Path
    checkpoint_args: Mapping[str, Any]
    sealed_manifest: Path
    image_test: Path
    speech_test: Path
    evaluator_scripts: Sequence[Path]
    paired_analysis_scripts: Sequence[Path]
    output: Path
    runai_project: str
    candidate_seed: int
    control_seed: int
    candidate_sizes: Sequence[int]
    candidate_protocols: Sequence[str]
    hard_negative_protocol: str
    image_query_count: int
    speech_query_count: int
    checkpoint_args_source: Path | None = None
    paired_tests: Sequence[str] = DEFAULT_PAIRED_TESTS
    evaluation_cells: Sequence[Mapping[str, Any]] = ()
    query_offset: int = 0
    candidate_offset: int = 0
    tie_epsilon: float = 1e-8
    candidate_permutation: str = "query_identity_seeded"
    randomize_positive_position: bool = True
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 12345
    protocol_name: str = "sealed_evaluation_v1"
    eval_split_name: str = "sealed_test"
    max_length: int = 512
    conditional_batch_size: int = 8


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"value is not JSON-serializable: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


CHECKPOINT_PROVENANCE_FIELDS = (
    "run_provenance",
    "gamma_provenance",
    "trainable_meta",
    "last_row",
)


def checkpoint_provenance_payload(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        field: state.get(field)
        for field in CHECKPOINT_PROVENANCE_FIELDS
    }


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError:
        return False
    return True


def checkpoint_identity(
    checkpoint_path: Path,
    selected_root: Path,
) -> Tuple[Dict[str, Any], Mapping[str, Any]]:
    raw_path = checkpoint_path.expanduser()
    if raw_path.is_symlink():
        raise ProtocolError("checkpoint cannot be a symlink")
    path = raw_path.resolve(strict=True)
    root = selected_root.expanduser().resolve(strict=True)
    if not path.is_file() or not _path_is_within(path, root):
        raise ProtocolError("checkpoint must be a regular file under selected root")
    payload = path.read_bytes()
    try:
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ProtocolError(f"cannot load checkpoint with weights_only=True: {exc}") from exc
    if not isinstance(state, Mapping):
        raise ProtocolError("checkpoint payload must be a mapping")
    args = state.get("args")
    if not isinstance(args, Mapping):
        raise ProtocolError("checkpoint is missing args mapping")
    provenance = checkpoint_provenance_payload(state)
    identity = {
        "path": str(path),
        "type": "file",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "args_sha256": canonical_sha256(args),
        "provenance_sha256": canonical_sha256(provenance),
        "args_provenance_sha256": canonical_sha256({
            "args": dict(args),
            "provenance": provenance,
        }),
    }
    return identity, state


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ProtocolError(f"expected a regular file: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "type": "file",
        "sha256": sha256_file(resolved),
        "bytes": stat.st_size,
    }


def _directory_fingerprint(path: Path) -> Dict[str, Any]:
    resolved = path.resolve(strict=True)
    records: List[Dict[str, Any]] = []
    for child in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_symlink():
            raise ProtocolError(f"directory inputs cannot contain symlinks: {child}")
        if not child.is_file():
            continue
        relative = child.relative_to(resolved).as_posix()
        records.append(
            {
                "relative_path": relative,
                "sha256": sha256_file(child),
                "bytes": child.stat().st_size,
            }
        )
    if not records:
        raise ProtocolError(f"selected root contains no regular files: {resolved}")
    return {
        "path": str(resolved),
        "type": "directory",
        "sha256": canonical_sha256(records),
        "file_count": len(records),
        "files": records,
    }


def fingerprint_path(path: Path) -> Dict[str, Any]:
    raw = path.expanduser()
    if raw.is_symlink():
        raise ProtocolError(f"input cannot be a symlink: {path}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"input does not exist: {path}: {exc}") from exc
    if resolved.is_file():
        return _file_fingerprint(resolved)
    if resolved.is_dir():
        return _directory_fingerprint(resolved)
    raise ProtocolError(f"input must be a regular file or directory: {resolved}")


def require_current_allocator_source_fingerprint(
    records: Any,
) -> Dict[str, Any]:
    """Require one canonical fingerprint for the allocator source in use."""

    if not isinstance(records, list) or not records:
        raise ProtocolError(
            "evaluator scripts must include exactly one "
            "scripts/sealed_position_allocator.py fingerprint"
        )
    allocator_source = Path(__file__).with_name(
        "sealed_position_allocator.py"
    ).resolve(strict=True)
    matches: List[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ProtocolError(
                f"evaluator_scripts[{index}] fingerprint must be an object"
            )
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ProtocolError(
                f"evaluator_scripts[{index}] fingerprint path is missing"
            )
        raw_path = Path(path_value).expanduser()
        if raw_path.is_symlink():
            raise ProtocolError(
                f"evaluator_scripts[{index}] fingerprint path cannot be a symlink"
            )
        try:
            resolved = raw_path.resolve(strict=True)
        except OSError as exc:
            raise ProtocolError(
                f"evaluator_scripts[{index}] fingerprint path cannot be resolved: {exc}"
            ) from exc
        if resolved == allocator_source:
            matches.append(record)
    if len(matches) != 1:
        raise ProtocolError(
            "evaluator scripts must include exactly one resolved "
            "scripts/sealed_position_allocator.py fingerprint"
        )
    current = fingerprint_path(allocator_source)
    if dict(matches[0]) != current:
        raise ProtocolError(
            "scripts/sealed_position_allocator.py fingerprint does not match "
            "the current SHA256"
        )
    return current


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ProtocolError(f"cannot open JSONL file {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ProtocolError(f"{path}:{line_number}: each row must be an object")
            rows.append(row)
    if not rows:
        raise ProtocolError(f"sealed data file has no rows: {path}")
    return rows


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be an object")
    return value


def _require_exact_int(value: Any, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ProtocolError(f"{label} mismatch: expected {expected}, got {value!r}")


def _validate_sources(manifest: Mapping[str, Any]) -> None:
    sources = _require_mapping(manifest.get("sources"), "manifest.sources")
    image = _require_mapping(sources.get("image"), "manifest.sources.image")
    speech = _require_mapping(sources.get("speech"), "manifest.sources.speech")

    image_split = str(image.get("split", "")).strip().casefold()
    image_partition = str(image.get("partition", "")).strip().casefold()
    if not image.get("dataset") or not image_split:
        raise ProtocolError("image source must identify a dataset and split")
    if "train" in image_split:
        raise ProtocolError("image source must be non-training validation-style data")
    if image_split not in {"validation", "valid", "val"} and "validation" not in image_partition:
        raise ProtocolError("image source must be validation-style data")

    speech_tuple = (
        str(speech.get("dataset", "")).casefold(),
        str(speech.get("config", "")).casefold(),
        str(speech.get("split", "")).casefold(),
    )
    if speech_tuple != ("openslr/librispeech_asr", "clean", "test"):
        raise ProtocolError("speech source must be openslr/librispeech_asr clean test")
    if "test-clean" not in str(speech.get("partition", "")).casefold():
        raise ProtocolError("speech source partition must be LibriSpeech test-clean")


def _validate_overlap(manifest: Mapping[str, Any]) -> None:
    assertions = _require_mapping(
        manifest.get("overlap_assertions"), "manifest.overlap_assertions"
    )
    policy = _require_mapping(
        assertions.get("source_partition_policy"),
        "manifest.overlap_assertions.source_partition_policy",
    )
    for key in (
        "image_non_training_validation_style",
        "speech_is_librispeech_test_clean",
        "passed",
    ):
        if policy.get(key) is not True:
            raise ProtocolError(f"source partition assertion did not pass: {key}")
    for modality in ("image", "speech"):
        report = _require_mapping(assertions.get(modality), f"overlap assertion {modality}")
        if report.get("passed") is not True:
            raise ProtocolError(f"{modality} overlap assertion did not pass")
        for count_key in ("hash_overlap_count", "source_id_overlap_count"):
            value = report.get(count_key)
            if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                raise ProtocolError(f"{modality} {count_key} must be zero")


def _as_id_values(value: Any) -> Iterable[str]:
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    return tuple(str(item) for item in values if str(item))


def _row_id_sets(row: Mapping[str, Any]) -> Tuple[str, set[str], set[str], set[str]]:
    row_id = str(row.get("id", "")).strip()
    if not row_id:
        raise ProtocolError("every sealed row must have a non-empty id")

    media_ids = set(_as_id_values(row.get("media_id")))
    media_ids.update(_as_id_values(row.get("media_sha256")))
    if not media_ids:
        raise ProtocolError(f"sealed row {row_id!r} has no media identifier or media_sha256")

    source_ids: set[str] = set()
    for key in ("source_id", "utterance_id", "source_ids"):
        source_ids.update(_as_id_values(row.get(key)))
    source = row.get("source")
    if isinstance(source, Mapping):
        for key in ("id", "source_id", "source_ids"):
            source_ids.update(_as_id_values(source.get(key)))
    if not source_ids:
        raise ProtocolError(f"sealed row {row_id!r} has no source identifier")

    group_ids: set[str] = set()
    group_ids.update(_as_id_values(row.get("group_id")))
    group = row.get("group")
    if isinstance(group, Mapping):
        for key in ("id", "group_id"):
            group_ids.update(_as_id_values(group.get(key)))
    elif group is not None:
        group_ids.update(_as_id_values(group))
    return row_id, media_ids, source_ids, group_ids


def _media_bytes(
    row: Mapping[str, Any],
    modality: str,
    manifest_path: Path,
) -> Tuple[bytes, Path]:
    media_key = "image_path" if modality == "image" else "audio_path"
    value = row.get(media_key)
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"sealed {modality} row is missing {media_key}")
    root = manifest_path.parent.resolve(strict=True)
    raw_path = Path(value).expanduser()
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ProtocolError(
            f"sealed {modality} media path escapes manifest root: {value}"
        ) from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ProtocolError(f"sealed {modality} media path contains a symlink: {value}")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"sealed {modality} media does not exist: {value}") from exc
    if not resolved.is_file():
        raise ProtocolError(f"sealed {modality} media is not a regular file: {value}")
    try:
        return resolved.read_bytes(), resolved
    except OSError as exc:
        raise ProtocolError(f"cannot read sealed {modality} media: {value}") from exc


def _content_free_rows(
    image_rows: Sequence[Mapping[str, Any]],
    speech_rows: Sequence[Mapping[str, Any]],
    image_path: Path,
    speech_path: Path,
) -> List[Dict[str, Any]]:
    seen: Dict[str, set[str]] = {
        "row": set(),
        "media": set(),
        "source": set(),
        "group": set(),
    }
    output: List[Dict[str, Any]] = []
    for modality, rows, manifest_path in (
        ("image", image_rows, image_path),
        ("speech", speech_rows, speech_path),
    ):
        for row in rows:
            row_id, media_ids, source_ids, group_ids = _row_id_sets(row)
            identifiers = {
                "row": {row_id},
                "media": media_ids,
                "source": source_ids,
                "group": group_ids,
            }
            for namespace, values in identifiers.items():
                duplicates = seen[namespace].intersection(values)
                if duplicates:
                    raise ProtocolError(
                        f"duplicate {namespace} ID(s): {sorted(duplicates)!r}"
                    )
                seen[namespace].update(values)
            media_payload, media_path = _media_bytes(row, modality, manifest_path)
            actual_media_sha256 = hashlib.sha256(media_payload).hexdigest()
            expected_media_sha256 = row.get("media_sha256")
            if (
                not isinstance(expected_media_sha256, str)
                or len(expected_media_sha256) != 64
                or expected_media_sha256.lower() != actual_media_sha256
            ):
                raise ProtocolError(
                    f"sealed row {row_id!r} media_sha256 does not match actual bytes"
                )
            output.append(
                {
                    "modality": modality,
                    "row_id": row_id,
                    "row_sha256": canonical_sha256(row),
                    "media_ids_sha256": canonical_sha256(sorted(media_ids)),
                    "source_ids_sha256": canonical_sha256(sorted(source_ids)),
                    "group_ids_sha256": canonical_sha256(sorted(group_ids)),
                    "media_sha256": actual_media_sha256,
                    "media_size_bytes": len(media_payload),
                    "media_relative_path_sha256": canonical_sha256(
                        media_path.relative_to(manifest_path.parent.resolve(strict=True)).as_posix()
                    ),
                }
            )
    return output


def validate_sealed_bundle(
    manifest_path: Path, image_path: Path, speech_path: Path
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    manifest = _read_json(manifest_path)
    if manifest.get("sealed") is not True:
        raise ProtocolError("sealed manifest must contain sealed=true")
    _validate_sources(manifest)
    _validate_overlap(manifest)

    image_rows = _read_jsonl(image_path)
    speech_rows = _read_jsonl(speech_path)
    counts = _require_mapping(manifest.get("counts"), "manifest.counts")
    _require_exact_int(counts.get("image_rows"), len(image_rows), "image row count")
    _require_exact_int(counts.get("speech_rows"), len(speech_rows), "speech row count")
    _require_exact_int(
        counts.get("total_rows"), len(image_rows) + len(speech_rows), "total row count"
    )

    files = _require_mapping(manifest.get("files"), "manifest.files")
    for path, rows in ((image_path, image_rows), (speech_path, speech_rows)):
        record = _require_mapping(files.get(path.name), f"manifest.files.{path.name}")
        expected_hash = str(record.get("sha256", ""))
        actual_hash = sha256_file(path)
        if len(expected_hash) != 64 or expected_hash != actual_hash:
            raise ProtocolError(f"manifest hash mismatch for {path.name}")
        _require_exact_int(record.get("rows"), len(rows), f"manifest row count for {path.name}")

    return manifest, _content_free_rows(
        image_rows,
        speech_rows,
        image_path,
        speech_path,
    )


def _git_command(repo: Path, args: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolError(f"git command failed: {' '.join(args)}: {exc}") from exc
    return result.stdout


def git_state(start: Path | None = None) -> Dict[str, Any]:
    location = (start or Path(__file__)).resolve()
    if location.is_file():
        location = location.parent
    try:
        root = _git_command(location, ("rev-parse", "--show-toplevel")).decode().strip()
    except ProtocolError:
        return {"available": False, "root": None, "head": None, "dirty_diff_sha256": None}
    repo = Path(root)
    head = _git_command(repo, ("rev-parse", "HEAD")).decode().strip()
    diff = _git_command(repo, ("diff", "--binary", "HEAD", "--", "."))
    untracked_raw = _git_command(
        repo, ("ls-files", "--others", "--exclude-standard", "-z")
    )
    untracked_files = sorted(
        value.decode("utf-8")
        for value in untracked_raw.split(b"\0")
        if value
    )
    untracked_records = []
    for relative in untracked_files:
        path = repo / relative
        if not path.is_file() or path.is_symlink():
            raise ProtocolError(f"untracked protocol input is not a regular file: {relative}")
        untracked_records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "available": True,
        "root": str(repo.resolve()),
        "head": head,
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "dirty_diff_bytes": len(diff),
        "untracked_files": untracked_records,
        "untracked_manifest_sha256": canonical_sha256(untracked_records),
    }


def validate_evaluation_cells(
    cells: Sequence[Mapping[str, Any]], candidate_sizes: Sequence[int]
) -> List[Dict[str, Any]]:
    sizes = set(candidate_sizes)
    if not cells:
        raise ProtocolError("evaluation matrix must contain at least one cell")
    normalized: List[Dict[str, Any]] = []
    cell_ids: set[str] = set()
    represented_sizes: set[int] = set()
    primary_hard_cells = 0
    for raw in cells:
        if not isinstance(raw, Mapping):
            raise ProtocolError("each evaluation cell must be an object")
        cell_id = str(raw.get("id", "")).strip()
        size = raw.get("candidate_count")
        negative_mode = str(raw.get("negative_mode", "")).strip()
        role = str(raw.get("role", "")).strip()
        if not cell_id or cell_id in cell_ids:
            raise ProtocolError("evaluation cell IDs must be non-empty and unique")
        if isinstance(size, bool) or not isinstance(size, int) or size not in sizes:
            raise ProtocolError(f"evaluation cell {cell_id!r} uses an unfrozen candidate size")
        if negative_mode not in {"random", "hard_text", "full_matrix"}:
            raise ProtocolError(f"evaluation cell {cell_id!r} has an invalid negative mode")
        if role not in {"primary", "secondary"}:
            raise ProtocolError(f"evaluation cell {cell_id!r} has an invalid role")
        normalized.append({
            "id": cell_id,
            "candidate_count": size,
            "negative_mode": negative_mode,
            "role": role,
        })
        cell_ids.add(cell_id)
        represented_sizes.add(size)
        primary_hard_cells += int(role == "primary" and negative_mode == "hard_text")
    if represented_sizes != sizes:
        raise ProtocolError("evaluation matrix must represent every frozen candidate size")
    if primary_hard_cells != 1:
        raise ProtocolError("evaluation matrix must contain exactly one primary hard-text cell")
    return normalized


CONTROL_RUNS = {
    "real": ("shared_prefix", "real"),
    "shuffled": ("shared_prefix", "shuffled"),
    "zero": ("shared_prefix", "zero"),
    "norm-matched-random": ("shared_prefix", "random"),
    "no-prefix": ("no_prefix_lm", "real"),
}


def build_evaluation_runs(
    cells: Sequence[Mapping[str, Any]],
    config: FreezeConfig,
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    allocator = build_allocator_manifest(
        cells,
        candidate_seed=config.candidate_seed,
        query_counts={
            "image": config.image_query_count,
            "speech": config.speech_query_count,
        },
        query_offset=config.query_offset,
    )
    plans = {
        (str(plan["cell_id"]), str(plan["modality"])): plan
        for plan in allocator["plans"]
    }
    for cell in cells:
        cell_id = str(cell["id"])
        candidate_count = int(cell["candidate_count"])
        image_plan = plans[(cell_id, "image")]
        speech_plan = plans[(cell_id, "speech")]
        full_matrix = (
            str(cell["negative_mode"]) == "full_matrix"
            or candidate_count == int(config.image_query_count)
        )
        negative_mode = (
            "random"
            if str(cell["negative_mode"]) == "full_matrix"
            else str(cell["negative_mode"])
        )
        for control, (eval_path, prefix_control) in CONTROL_RUNS.items():
            runs.append({
                "id": f"{cell_id}:{control}",
                "cell_id": cell_id,
                "role": str(cell["role"]),
                "negative_mode": negative_mode,
                "requested_candidate_count": candidate_count,
                "conditional_negatives": -1 if full_matrix else candidate_count - 1,
                "conditional_candidates": candidate_count,
                "conditional_queries": int(config.image_query_count),
                "image_query_count": int(config.image_query_count),
                "speech_query_count": int(config.speech_query_count),
                "image_eval_samples": int(config.image_query_count),
                "speech_eval_samples": int(config.speech_query_count),
                "max_length": int(config.max_length),
                "conditional_batch_size": int(config.conditional_batch_size),
                "query_offset": int(config.query_offset),
                "candidate_offset": int(config.candidate_offset),
                "tie_epsilon": float(config.tie_epsilon),
                "candidate_permutation": str(config.candidate_permutation),
                "randomize_positive_position": bool(
                    config.randomize_positive_position
                ),
                "control": control,
                "prefix_control": prefix_control,
                "eval_path": eval_path,
                "candidate_seed": int(config.candidate_seed),
                "gold_position_allocator_name": ALLOCATOR_NAME,
                "gold_position_allocator_version": ALLOCATOR_VERSION,
                "gold_position_assignment_plans_sha256": allocator[
                    "plans_sha256"
                ],
                "image_gold_position_assignment_id": image_plan[
                    "assignment_id"
                ],
                "image_gold_positions_sha256": image_plan[
                    "positions_sha256"
                ],
                "speech_gold_position_assignment_id": speech_plan[
                    "assignment_id"
                ],
                "speech_gold_positions_sha256": speech_plan[
                    "positions_sha256"
                ],
                "control_seed": int(config.control_seed),
                "bootstrap_samples": int(config.bootstrap_samples),
                "bootstrap_seed": int(config.bootstrap_seed),
                "protocol_name": str(config.protocol_name),
                "eval_split_name": str(config.eval_split_name),
            })
    return runs


def validate_evaluation_runs(
    runs: Any,
    cells: Sequence[Mapping[str, Any]],
    allocator_manifest: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if not isinstance(runs, list):
        raise ProtocolError("evaluation_runs must be a list")
    expected_pairs = {
        (str(cell["id"]), control)
        for cell in cells
        for control in REQUIRED_CONTROLS
    }
    observed_pairs: set[Tuple[str, str]] = set()
    observed_contracts: set[str] = set()
    required_fields = {
        "id", "cell_id", "role", "negative_mode",
        "requested_candidate_count", "conditional_negatives",
        "conditional_candidates", "conditional_queries",
        "image_query_count", "speech_query_count", "query_offset",
        "image_eval_samples", "speech_eval_samples",
        "max_length", "conditional_batch_size",
        "candidate_offset", "tie_epsilon", "candidate_permutation",
        "randomize_positive_position", "control", "prefix_control",
        "eval_path", "candidate_seed", "control_seed",
        "gold_position_allocator_name", "gold_position_allocator_version",
        "gold_position_assignment_plans_sha256",
        "image_gold_position_assignment_id", "image_gold_positions_sha256",
        "speech_gold_position_assignment_id", "speech_gold_positions_sha256",
        "bootstrap_samples", "bootstrap_seed", "protocol_name",
        "eval_split_name",
    }
    plans = {
        (str(plan["cell_id"]), str(plan["modality"])): plan
        for plan in allocator_manifest["plans"]
    }
    for run in runs:
        if not isinstance(run, dict) or set(run) != required_fields:
            raise ProtocolError("evaluation run contract is incomplete")
        pair = (str(run["cell_id"]), str(run["control"]))
        if pair in observed_pairs:
            raise ProtocolError("evaluation run contracts must be unique")
        observed_pairs.add(pair)
        contract_signature = canonical_sha256({
            key: value
            for key, value in run.items()
            if key not in {"id", "cell_id", "role"}
        })
        if contract_signature in observed_contracts:
            raise ProtocolError(
                "evaluation matrix contains non-unique metric contracts"
            )
        observed_contracts.add(contract_signature)
        if run["negative_mode"] not in {"random", "hard_text"}:
            raise ProtocolError("evaluation run has invalid negative_mode")
        if run["eval_path"] not in {"shared_prefix", "no_prefix_lm"}:
            raise ProtocolError("evaluation run has invalid eval_path")
        for field in (
            "requested_candidate_count", "conditional_negatives",
            "conditional_candidates", "conditional_queries",
            "image_query_count", "speech_query_count", "query_offset",
            "image_eval_samples", "speech_eval_samples",
            "max_length", "conditional_batch_size",
            "candidate_offset", "candidate_seed", "control_seed",
            "bootstrap_samples", "bootstrap_seed",
        ):
            if isinstance(run[field], bool) or not isinstance(run[field], int):
                raise ProtocolError(f"evaluation run {field} must be an integer")
        if not isinstance(run["randomize_positive_position"], bool):
            raise ProtocolError("evaluation run randomization flag must be boolean")
        image_plan = plans.get((str(run["cell_id"]), "image"))
        speech_plan = plans.get((str(run["cell_id"]), "speech"))
        if image_plan is None or speech_plan is None:
            raise ProtocolError("evaluation run has no frozen gold-position plan")
        expected_assignment_binding = {
            "gold_position_allocator_name": ALLOCATOR_NAME,
            "gold_position_allocator_version": ALLOCATOR_VERSION,
            "gold_position_assignment_plans_sha256": allocator_manifest[
                "plans_sha256"
            ],
            "image_gold_position_assignment_id": image_plan["assignment_id"],
            "image_gold_positions_sha256": image_plan["positions_sha256"],
            "speech_gold_position_assignment_id": speech_plan["assignment_id"],
            "speech_gold_positions_sha256": speech_plan["positions_sha256"],
        }
        if any(run.get(key) != value for key, value in expected_assignment_binding.items()):
            raise ProtocolError("evaluation run gold-position binding is invalid")
        tie_epsilon = run["tie_epsilon"]
        if (
            isinstance(tie_epsilon, bool)
            or not isinstance(tie_epsilon, (int, float))
            or not math.isfinite(float(tie_epsilon))
            or float(tie_epsilon) < 0.0
        ):
            raise ProtocolError("evaluation run tie_epsilon is invalid")
    if observed_pairs != expected_pairs:
        raise ProtocolError("evaluation runs do not cover the exact matrix")
    return runs


def holm_families(cells: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    primary_ids = [str(cell["id"]) for cell in cells if str(cell["role"]) == "primary"]
    shared = {
        "alpha": 0.05,
        "method": "holm_bonferroni",
        "cell_ids": primary_ids,
        "modalities": ["image", "speech"],
        "controls": list(REQUIRED_CONTROLS[1:]),
    }
    return [
        {
            **shared,
            "id": "primary_r1_mcnemar",
            "metric": "r_at_1",
            "test": "paired_exact_mcnemar",
        },
        {
            **shared,
            "id": "primary_mrr_permutation",
            "metric": "mrr",
            "test": "paired_sign_flip_permutation",
        },
    ]


def _validate_config(config: FreezeConfig) -> None:
    if config.output.exists():
        raise FileExistsError(f"refusing to overwrite existing protocol: {config.output}")
    if not isinstance(config.checkpoint_args, Mapping):
        raise ProtocolError("checkpoint args must be a JSON object")
    if not config.evaluator_scripts or not config.paired_analysis_scripts:
        raise ProtocolError("at least one evaluator and paired-analysis script are required")
    require_current_allocator_source_fingerprint(
        [fingerprint_path(path) for path in config.evaluator_scripts]
    )
    if not config.runai_project.strip():
        raise ProtocolError("Run:AI project must be non-empty")
    sizes = list(config.candidate_sizes)
    if not sizes or any(isinstance(size, bool) or not isinstance(size, int) or size <= 1 for size in sizes):
        raise ProtocolError("candidate sizes must contain positive integers greater than one")
    if len(set(sizes)) != len(sizes):
        raise ProtocolError("candidate sizes must be unique")
    protocols = [value.strip() for value in config.candidate_protocols]
    if not protocols or any(not value for value in protocols) or len(set(protocols)) != len(protocols):
        raise ProtocolError("candidate protocols must be non-empty and unique")
    if not config.hard_negative_protocol.strip():
        raise ProtocolError("hard-negative protocol must be non-empty")
    if config.image_query_count <= 0 or config.speech_query_count <= 0:
        raise ProtocolError("query counts must be positive")
    if config.max_length <= 0 or config.conditional_batch_size <= 0:
        raise ProtocolError(
            "max_length and conditional_batch_size must be positive"
        )
    if config.image_query_count != config.speech_query_count:
        raise ProtocolError(
            "conditional evaluator requires equal image and speech query counts"
        )
    if not config.paired_tests or any(not value.strip() for value in config.paired_tests):
        raise ProtocolError("paired tests must be non-empty")
    validate_evaluation_cells(config.evaluation_cells, sizes)
    for field, value in (
        ("query_offset", config.query_offset),
        ("candidate_offset", config.candidate_offset),
        ("bootstrap_samples", config.bootstrap_samples),
        ("bootstrap_seed", config.bootstrap_seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProtocolError(f"{field} must be a non-negative integer")
    if (
        isinstance(config.tie_epsilon, bool)
        or not isinstance(config.tie_epsilon, (int, float))
        or not math.isfinite(float(config.tie_epsilon))
        or float(config.tie_epsilon) < 0.0
    ):
        raise ProtocolError("tie epsilon must be finite and non-negative")
    if config.candidate_permutation != "query_identity_seeded":
        raise ProtocolError("unsupported candidate permutation policy")
    if config.randomize_positive_position is not True:
        raise ProtocolError("sealed evaluation requires randomized positive positions")
    if not config.protocol_name.strip() or not config.eval_split_name.strip():
        raise ProtocolError("protocol and eval split names must be non-empty")

    output = config.output.resolve(strict=False)
    selected = config.selected_root.resolve(strict=True)
    checkpoint_identity(config.checkpoint, config.selected_root)
    if selected.is_dir() and (output == selected or selected in output.parents):
        raise ProtocolError("output must be outside the selected root so freezing cannot cause drift")


def _build_input_fingerprints(config: FreezeConfig) -> Dict[str, Any]:
    evaluator_scripts = [
        fingerprint_path(path) for path in config.evaluator_scripts
    ]
    require_current_allocator_source_fingerprint(evaluator_scripts)
    inputs: Dict[str, Any] = {
        "selected_root": fingerprint_path(config.selected_root),
        "sealed_manifest": fingerprint_path(config.sealed_manifest),
        "image_test": fingerprint_path(config.image_test),
        "speech_test": fingerprint_path(config.speech_test),
        "evaluator_scripts": evaluator_scripts,
        "paired_analysis_scripts": [
            fingerprint_path(path) for path in config.paired_analysis_scripts
        ],
    }
    if config.checkpoint_args_source is None:
        inputs["checkpoint_args"] = {
            "type": "inline-json",
            "sha256": canonical_sha256(config.checkpoint_args),
        }
    else:
        inputs["checkpoint_args"] = fingerprint_path(config.checkpoint_args_source)
    return inputs


def freeze_protocol(config: FreezeConfig) -> Dict[str, Any]:
    _validate_config(config)
    artifact, checkpoint_state = checkpoint_identity(
        config.checkpoint, config.selected_root
    )
    checkpoint_args = _require_mapping(
        checkpoint_state.get("args"), "checkpoint.args"
    )
    if dict(checkpoint_args) != dict(config.checkpoint_args):
        raise ProtocolError(
            "checkpoint-stored args disagree with --checkpoint-args-json"
        )
    _, row_records = validate_sealed_bundle(
        config.sealed_manifest, config.image_test, config.speech_test
    )
    evaluation_cells = validate_evaluation_cells(
        config.evaluation_cells, config.candidate_sizes
    )
    gold_position_allocator = build_allocator_manifest(
        evaluation_cells,
        candidate_seed=config.candidate_seed,
        query_counts={
            "image": config.image_query_count,
            "speech": config.speech_query_count,
        },
        query_offset=config.query_offset,
    )
    protocol: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "inputs": _build_input_fingerprints(config),
        "checkpoint": {
            "selected_root": str(config.selected_root.resolve(strict=True)),
            "artifact": artifact,
            "args": dict(config.checkpoint_args),
            "args_sha256": canonical_sha256(config.checkpoint_args),
            "provenance_sha256": artifact["provenance_sha256"],
            "args_provenance_sha256": artifact[
                "args_provenance_sha256"
            ],
        },
        "git": git_state(),
        "runai_project": config.runai_project.strip(),
        "seeds": {
            "candidate_seed": config.candidate_seed,
            "control_seed": config.control_seed,
        },
        "candidate_sets": {
            "sizes": list(config.candidate_sizes),
            "protocols": [value.strip() for value in config.candidate_protocols],
        },
        "gold_position_allocator": gold_position_allocator,
        "evaluation_matrix": evaluation_cells,
        "evaluation_runs": build_evaluation_runs(evaluation_cells, config),
        "controls": list(REQUIRED_CONTROLS),
        "hard_negative_protocol": config.hard_negative_protocol.strip(),
        "query_counts": {
            "image": config.image_query_count,
            "speech": config.speech_query_count,
        },
        "random_positive_requirement": {
            "required": True,
            "rule": (
                "The gold candidate position must be randomized from candidate_seed for every "
                "query and balanced within one occurrence at each candidate size."
            ),
        },
        "paired_tests": [value.strip() for value in config.paired_tests],
        "holm_families": holm_families(evaluation_cells),
        "fixed_claim_decision_rules": list(FIXED_CLAIM_DECISION_RULES),
        "sealed_metrics_policy": SEALED_METRICS_POLICY,
        "sealed_rows": row_records,
    }
    protocol["protocol_content_sha256"] = canonical_sha256(protocol)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with config.output.open("x", encoding="utf-8") as handle:
            json.dump(protocol, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing protocol: {config.output}")
    return protocol


def _verify_fingerprint(stored: Mapping[str, Any], label: str) -> None:
    if stored.get("type") == "inline-json":
        return
    path_value = stored.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ProtocolError(f"{label}: stored path is missing")
    actual = fingerprint_path(Path(path_value))
    if actual != dict(stored):
        raise ProtocolError(f"{label}: input hash drift detected for {path_value}")


def verify_protocol(protocol_path: Path, *, verify_git_state: bool = False) -> Dict[str, Any]:
    protocol = _read_json(protocol_path)
    stored_digest = protocol.pop("protocol_content_sha256", None)
    if not isinstance(stored_digest, str) or canonical_sha256(protocol) != stored_digest:
        raise ProtocolError("protocol content hash mismatch")
    protocol["protocol_content_sha256"] = stored_digest
    if protocol.get("schema_version") != SCHEMA_VERSION or protocol.get("protocol") != PROTOCOL_NAME:
        raise ProtocolError("unsupported protocol schema")

    inputs = _require_mapping(protocol.get("inputs"), "protocol.inputs")
    require_current_allocator_source_fingerprint(
        inputs.get("evaluator_scripts")
    )
    for role in ("selected_root", "sealed_manifest", "image_test", "speech_test"):
        _verify_fingerprint(_require_mapping(inputs.get(role), f"inputs.{role}"), role)
    for role in ("evaluator_scripts", "paired_analysis_scripts"):
        records = inputs.get(role)
        if not isinstance(records, list) or not records:
            raise ProtocolError(f"inputs.{role} must be a non-empty list")
        for index, record in enumerate(records):
            _verify_fingerprint(_require_mapping(record, f"inputs.{role}[{index}]"), role)
    checkpoint_args_input = _require_mapping(inputs.get("checkpoint_args"), "inputs.checkpoint_args")
    _verify_fingerprint(checkpoint_args_input, "checkpoint_args")

    checkpoint = _require_mapping(protocol.get("checkpoint"), "protocol.checkpoint")
    selected_root = Path(str(checkpoint.get("selected_root", "")))
    artifact = _require_mapping(
        checkpoint.get("artifact"), "protocol.checkpoint.artifact"
    )
    actual_artifact, checkpoint_state = checkpoint_identity(
        Path(str(artifact.get("path", ""))), selected_root
    )
    if actual_artifact != dict(artifact):
        raise ProtocolError("checkpoint artifact identity drift detected")
    checkpoint_args = _require_mapping(checkpoint.get("args"), "protocol.checkpoint.args")
    if canonical_sha256(checkpoint_args) != checkpoint.get("args_sha256"):
        raise ProtocolError("checkpoint args hash mismatch")
    stored_checkpoint_args = _require_mapping(
        checkpoint_state.get("args"), "checkpoint.args"
    )
    if dict(stored_checkpoint_args) != dict(checkpoint_args):
        raise ProtocolError("checkpoint-stored args disagree with frozen args")
    if (
        checkpoint.get("provenance_sha256")
        != actual_artifact["provenance_sha256"]
        or checkpoint.get("args_provenance_sha256")
        != actual_artifact["args_provenance_sha256"]
    ):
        raise ProtocolError("checkpoint args/provenance digest mismatch")
    if checkpoint_args_input.get("type") == "inline-json":
        if checkpoint_args_input.get("sha256") != canonical_sha256(checkpoint_args):
            raise ProtocolError("inline checkpoint args drift detected")
    else:
        source_args = _read_json(Path(str(checkpoint_args_input["path"])))
        if source_args != dict(checkpoint_args):
            raise ProtocolError("checkpoint args file content disagrees with frozen args")

    _, row_records = validate_sealed_bundle(
        Path(str(inputs["sealed_manifest"]["path"])),
        Path(str(inputs["image_test"]["path"])),
        Path(str(inputs["speech_test"]["path"])),
    )
    if row_records != protocol.get("sealed_rows"):
        raise ProtocolError("sealed row IDs or hashes drifted")
    if protocol.get("controls") != list(REQUIRED_CONTROLS):
        raise ProtocolError("required controls changed")
    candidate_sets = _require_mapping(protocol.get("candidate_sets"), "protocol.candidate_sets")
    raw_sizes = candidate_sets.get("sizes")
    if not isinstance(raw_sizes, list):
        raise ProtocolError("protocol candidate sizes must be a list")
    raw_cells = protocol.get("evaluation_matrix")
    if not isinstance(raw_cells, list):
        raise ProtocolError("protocol evaluation matrix must be a list")
    cells = validate_evaluation_cells(raw_cells, raw_sizes)
    if cells != raw_cells:
        raise ProtocolError("evaluation matrix is not canonical")
    query_counts = _require_mapping(
        protocol.get("query_counts"), "protocol.query_counts"
    )
    seeds = _require_mapping(protocol.get("seeds"), "protocol.seeds")
    runs = protocol.get("evaluation_runs")
    if not isinstance(runs, list) or not runs:
        raise ProtocolError("evaluation_runs must be a non-empty list")
    query_offsets = {
        run.get("query_offset")
        for run in runs
        if isinstance(run, Mapping)
    }
    if len(query_offsets) != 1:
        raise ProtocolError("evaluation runs must share one query_offset")
    try:
        allocator_plans = validate_allocator_manifest(
            protocol.get("gold_position_allocator"),
            cells,
            candidate_seed=int(seeds["candidate_seed"]),
            query_counts={
                "image": int(query_counts["image"]),
                "speech": int(query_counts["speech"]),
            },
            query_offset=int(next(iter(query_offsets))),
        )
    except (AssignmentPlanError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid gold-position allocator: {exc}") from exc
    if len(allocator_plans) != 2 * len(cells):
        raise ProtocolError("gold-position allocator does not cover every cell/modality")
    validate_evaluation_runs(
        runs,
        cells,
        _require_mapping(
            protocol.get("gold_position_allocator"),
            "protocol.gold_position_allocator",
        ),
    )
    if protocol.get("holm_families") != holm_families(cells):
        raise ProtocolError("Holm correction families changed")
    if protocol.get("fixed_claim_decision_rules") != list(FIXED_CLAIM_DECISION_RULES):
        raise ProtocolError("fixed claim decision rules changed")
    if protocol.get("sealed_metrics_policy") != SEALED_METRICS_POLICY:
        raise ProtocolError("sealed metrics policy changed")
    if verify_git_state and protocol.get("git") != git_state():
        raise ProtocolError("git HEAD or dirty diff hash drifted")
    return protocol


def _load_checkpoint_args(value: str) -> Tuple[Dict[str, Any], Path | None]:
    source: Path | None = None
    text = value
    candidate = Path(value[1:] if value.startswith("@") else value)
    if value.startswith("@") or candidate.is_file():
        source = candidate
        args = _read_json(candidate)
    else:
        try:
            args = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"--checkpoint-args-json is not valid JSON: {exc}") from exc
        if not isinstance(args, dict):
            raise ProtocolError("--checkpoint-args-json must decode to an object")
    return args, source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, help="verify frozen inputs and protocol content, then exit")
    parser.add_argument(
        "--verify-git-state",
        action="store_true",
        help="also require current Git HEAD and dirty diff to match freeze time",
    )
    parser.add_argument("--selected-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-args-json",
        help="checkpoint args as a JSON object, a JSON file path, or @JSON_FILE",
    )
    parser.add_argument("--sealed-manifest", type=Path)
    parser.add_argument("--image-test", type=Path)
    parser.add_argument("--speech-test", type=Path)
    parser.add_argument("--evaluator-script", action="append", type=Path, default=[])
    parser.add_argument("--paired-analysis-script", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runai-project", default=os.environ.get("RUNAI_PROJECT", ""))
    parser.add_argument("--candidate-seed", type=int)
    parser.add_argument("--control-seed", type=int)
    parser.add_argument("--candidate-size", action="append", type=int, default=[])
    parser.add_argument("--candidate-protocol", action="append", default=[])
    parser.add_argument("--hard-negative-protocol")
    parser.add_argument("--image-query-count", type=int)
    parser.add_argument("--speech-query-count", type=int)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--conditional-batch-size", type=int, default=8)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--tie-epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--candidate-permutation",
        choices=["query_identity_seeded"],
        default="query_identity_seeded",
    )
    parser.add_argument(
        "--randomize-positive-position",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--protocol-name", default="sealed_evaluation_v1")
    parser.add_argument("--eval-split-name", default="sealed_test")
    parser.add_argument("--paired-test", action="append", default=[])
    parser.add_argument(
        "--evaluation-cell",
        action="append",
        default=[],
        metavar="ID:CANDIDATES:NEGATIVE_MODE:ROLE",
    )
    return parser


def parse_evaluation_cell(value: str) -> Dict[str, Any]:
    parts = value.split(":")
    if len(parts) != 4:
        raise ProtocolError("evaluation cells use ID:CANDIDATES:NEGATIVE_MODE:ROLE")
    cell_id, size, negative_mode, role = parts
    try:
        candidate_count = int(size)
    except ValueError as exc:
        raise ProtocolError(f"invalid evaluation candidate count: {size!r}") from exc
    return {
        "id": cell_id,
        "candidate_count": candidate_count,
        "negative_mode": negative_mode,
        "role": role,
    }


def _required_freeze_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    names = (
        "selected_root",
        "checkpoint",
        "checkpoint_args_json",
        "sealed_manifest",
        "image_test",
        "speech_test",
        "output",
        "candidate_seed",
        "control_seed",
        "hard_negative_protocol",
        "image_query_count",
        "speech_query_count",
    )
    missing = ["--" + name.replace("_", "-") for name in names if getattr(args, name) is None]
    if missing:
        parser.error("freeze mode requires " + ", ".join(missing))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            freeze_only_values = [
                args.selected_root,
                args.checkpoint,
                args.checkpoint_args_json,
                args.sealed_manifest,
                args.image_test,
                args.speech_test,
                args.output,
            ]
            if any(value is not None for value in freeze_only_values):
                parser.error("--verify cannot be combined with freeze-mode inputs")
            verify_protocol(args.verify, verify_git_state=bool(args.verify_git_state))
            print(json.dumps({
                "verified": str(args.verify.resolve()),
                "git_state_checked": bool(args.verify_git_state),
            }, sort_keys=True))
            return 0

        _required_freeze_args(args, parser)
        checkpoint_args, checkpoint_args_source = _load_checkpoint_args(args.checkpoint_args_json)
        config = FreezeConfig(
            selected_root=args.selected_root,
            checkpoint=args.checkpoint,
            checkpoint_args=checkpoint_args,
            checkpoint_args_source=checkpoint_args_source,
            sealed_manifest=args.sealed_manifest,
            image_test=args.image_test,
            speech_test=args.speech_test,
            evaluator_scripts=args.evaluator_script,
            paired_analysis_scripts=args.paired_analysis_script,
            output=args.output,
            runai_project=args.runai_project,
            candidate_seed=args.candidate_seed,
            control_seed=args.control_seed,
            candidate_sizes=args.candidate_size,
            candidate_protocols=args.candidate_protocol,
            hard_negative_protocol=args.hard_negative_protocol,
            image_query_count=args.image_query_count,
            speech_query_count=args.speech_query_count,
            max_length=args.max_length,
            conditional_batch_size=args.conditional_batch_size,
            paired_tests=args.paired_test or DEFAULT_PAIRED_TESTS,
            evaluation_cells=[
                parse_evaluation_cell(value) for value in args.evaluation_cell
            ],
            query_offset=args.query_offset,
            candidate_offset=args.candidate_offset,
            tie_epsilon=args.tie_epsilon,
            candidate_permutation=args.candidate_permutation,
            randomize_positive_position=args.randomize_positive_position,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            protocol_name=args.protocol_name,
            eval_split_name=args.eval_split_name,
        )
        protocol = freeze_protocol(config)
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "protocol_content_sha256": protocol["protocol_content_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (FileExistsError, ProtocolError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
