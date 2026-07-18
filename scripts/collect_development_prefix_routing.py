"""Collect real development image/audio prefix routing from a selected E3 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


SCHEMA_VERSION = 1
TOP_K = 2
FORBIDDEN_PATH_TERMS = ("sealed", "synthetic")
E3_EXPERIMENT_ID = "E3_final_multimodal_top2"
COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SPLIT_FILE_KEYS = {
    "train": ("image_train", "speech_train"),
    "train-dev": ("image_train", "speech_train", "image_dev", "speech_dev"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_forbidden_paths(paths: Iterable[Path | str]) -> None:
    for path in paths:
        lowered = str(path).lower().replace("\\", "/")
        for term in FORBIDDEN_PATH_TERMS:
            if term in lowered:
                raise ValueError(f"development-only collector rejects {term!r} path: {path}")


def verify_checkpoint_sha(checkpoint: Path, expected_sha256: str) -> str:
    expected = str(expected_sha256).strip().lower()
    if SHA256_RE.fullmatch(expected) is None:
        raise ValueError("--expected-checkpoint-sha256 must be an exact 64-hex SHA-256")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    return actual


def _resolve_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _exact_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label} must be an exact 64-hex SHA-256")
    return digest


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _stage_b_run_identity(run_provenance: Mapping[str, Any]) -> Dict[str, Any]:
    run_uuid = str(run_provenance.get("run_uuid", "")).lower()
    try:
        parsed_uuid = uuid.UUID(hex=run_uuid)
    except ValueError as exc:
        raise ValueError("Stage-B run_uuid must be a UUID4") from exc
    if parsed_uuid.version != 4 or parsed_uuid.hex != run_uuid:
        raise ValueError("Stage-B run_uuid must be a canonical UUID4 hex value")
    dataset = run_provenance.get("dataset_split_provenance")
    if not isinstance(dataset, Mapping):
        raise ValueError("Stage-B run identity lacks dataset provenance")
    dataset_sha = _canonical_json_sha256(dataset)
    if run_provenance.get("dataset_split_provenance_sha256") != dataset_sha:
        raise ValueError("Stage-B dataset provenance SHA mismatch")
    identity = {
        "schema_version": 1,
        "run_uuid": run_uuid,
        "source_commit_sha": run_provenance.get("source_commit_sha"),
        "runai_job_name": run_provenance.get("runai_job_name"),
        "runai_project": run_provenance.get("runai_project"),
        "producer_code": run_provenance.get("producer_code"),
        "dataset_split_provenance_sha256": dataset_sha,
        "sealed_evidence_used": run_provenance.get("sealed_evidence_used"),
        "synthetic_evidence_used": run_provenance.get("synthetic_evidence_used"),
    }
    if (
        identity["sealed_evidence_used"] is not False
        or identity["synthetic_evidence_used"] is not False
    ):
        raise ValueError("Stage-B run identity evidence policy is invalid")
    return identity


def verify_git_commit(repo_root: Path, value: Any, label: str) -> str:
    commit = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"{label} must be an exact 40-hex Git commit")
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "rev-parse",
            "--verify",
            f"{commit}^{{commit}}",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip().lower() != commit:
        raise ValueError(f"{label} cannot be resolved as a Git commit: {commit}")
    return commit


def load_exact_json(
    path: Path, expected_sha256: str, label: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resolved = _resolve_regular_file(path, label)
    expected_sha = _exact_sha256(expected_sha256, f"{label} SHA-256")
    actual_sha = sha256_file(resolved)
    if actual_sha != expected_sha:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected_sha} observed={actual_sha}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload, {
        "path": str(resolved),
        "sha256": actual_sha,
        "size_bytes": int(resolved.stat().st_size),
    }


def verify_source_commit(repo_root: Path, source_commit_sha: str) -> str:
    expected = str(source_commit_sha).strip().lower()
    if COMMIT_RE.fullmatch(expected) is None:
        raise ValueError("--source-commit-sha must be an exact 40- or 64-hex commit SHA")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot resolve source commit: {completed.stderr.strip()}")
    actual = completed.stdout.strip().lower()
    if actual != expected:
        raise ValueError(f"source commit mismatch: expected {expected}, observed {actual}")
    return actual


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is a blank JSONL row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(dict(value))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _strict_split_file_record(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    key: str,
    supplied_path: Path,
) -> Dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not isinstance(files.get(key), Mapping):
        raise ValueError(f"strict split manifest is missing files.{key}")
    record = dict(files[key])
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"strict split manifest files.{key}.path is missing")
    recorded_path = Path(raw_path).expanduser()
    if not recorded_path.is_absolute():
        recorded_path = manifest_path.parent / recorded_path
    recorded_path = _resolve_regular_file(recorded_path, f"files.{key}")
    supplied = _resolve_regular_file(supplied_path, f"supplied {key} manifest")
    if supplied != recorded_path:
        raise ValueError(
            f"strict split manifest source mismatch for {key}: "
            f"recorded={recorded_path} supplied={supplied}"
        )
    expected_sha = _exact_sha256(record.get("sha256"), f"files.{key}.sha256")
    actual_sha = sha256_file(recorded_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"strict split manifest SHA-256 mismatch for {key}: "
            f"expected={expected_sha} observed={actual_sha}"
        )
    return {**record, "path": str(recorded_path), "sha256": actual_sha}


def _verify_split_builder_source(
    manifest: Mapping[str, Any], repo_root: Path
) -> Dict[str, Any]:
    builder = manifest.get("builder")
    if not isinstance(builder, Mapping):
        raise ValueError("strict split manifest is missing builder provenance")
    record = dict(builder)
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("strict split manifest builder.path is missing")
    path = _resolve_regular_file(Path(raw_path).expanduser(), "split builder")
    expected_sha = _exact_sha256(record.get("sha256"), "split builder sha256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"strict split builder SHA-256 mismatch: expected={expected_sha} "
            f"observed={actual_sha}"
        )
    source_commit = str(record.get("source_commit_sha", "")).strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("strict split builder source_commit_sha must be exact")
    if record.get("source_matches_commit") is not True:
        raise ValueError("strict split builder must prove source_matches_commit=true")
    if record.get("command") != "python scripts/materialize_eval_splits.py":
        raise ValueError("strict split builder command is missing or invalid")
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "show",
            f"{source_commit}:scripts/materialize_eval_splits.py",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            "strict split builder source commit cannot be resolved: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    tracked_sha = hashlib.sha256(completed.stdout).hexdigest()
    if tracked_sha != expected_sha:
        raise ValueError(
            "strict split builder source mismatch: declared bytes do not match "
            f"{source_commit}:scripts/materialize_eval_splits.py"
        )
    return {
        **record,
        "path": str(path),
        "sha256": actual_sha,
        "source_commit_sha": source_commit,
        "source_matches_commit": True,
    }


def load_strict_split_manifest(
    path: Path,
    expected_sha256: str,
    *,
    collection_split: str,
    supplied_paths: Mapping[str, Path],
    repo_root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    if collection_split not in SPLIT_FILE_KEYS:
        raise ValueError(f"unsupported collection split: {collection_split}")
    manifest_path = _resolve_regular_file(path, "development split manifest")
    expected_sha = _exact_sha256(
        expected_sha256, "--expected-development-split-manifest-sha256"
    )
    actual_sha = sha256_file(manifest_path)
    if actual_sha != expected_sha:
        raise ValueError(
            "development split manifest SHA-256 mismatch: "
            f"expected={expected_sha} observed={actual_sha}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("strict split manifest root must be an object")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version <= 0
    ):
        raise ValueError(
            "strict split manifest schema_version must be a positive integer "
            "emitted by its verified builder"
        )
    for field, expected in (
        ("real_subset", True),
        ("synthetic_evidence_used", False),
        ("sealed_data_used", False),
    ):
        if payload.get(field) is not expected:
            raise ValueError(f"strict split manifest must declare {field}={expected!r}")
    builder = _verify_split_builder_source(payload, repo_root)
    requested = SPLIT_FILE_KEYS[collection_split]
    if set(supplied_paths) != set(requested):
        raise ValueError(
            f"supplied manifest keys do not match collection split {collection_split}: "
            f"expected={list(requested)} observed={sorted(supplied_paths)}"
        )
    records = {
        key: _strict_split_file_record(
            payload, manifest_path, key, supplied_paths[key]
        )
        for key in requested
    }
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("strict split manifest is missing counts")
    for key in requested:
        count = counts.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"strict split manifest counts.{key} must be positive")
        records[key]["rows"] = count
    payload = {
        **payload,
        "_verified": {
            "path": str(manifest_path),
            "sha256": actual_sha,
            "builder": builder,
            "collection_split": collection_split,
            "requested_files": list(requested),
            "unread_files": sorted(
                key
                for key in (payload.get("files") or {})
                if key not in requested
            ),
        },
    }
    return payload, records


def validate_strict_split_rows(
    rows: Sequence[Mapping[str, Any]], key: str
) -> None:
    modality, split = key.split("_", 1)
    expected_split_name = f"{modality}_{split}"
    for index, row in enumerate(rows, 1):
        if row.get("task") != modality:
            raise ValueError(
                f"{key} row {index} source mismatch: task={row.get('task')!r}"
            )
        source = row.get("source") or row.get("source_dataset")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{key} row {index} is missing source provenance")
        if row.get("eval_split_name") != expected_split_name:
            raise ValueError(
                f"{key} row {index} split mismatch: "
                f"eval_split_name={row.get('eval_split_name')!r}"
            )


def _media_ids(row: Mapping[str, Any], modality: str, index: int) -> set[str]:
    values: set[str] = set()
    for key in ("media_sha256", "content_sha256", "resized_content_sha256"):
        value = row.get(key)
        if value not in (None, ""):
            values.add(f"{modality}:hash:{value}")
    for key in ("uid", "utterance_id", "id", "source_id"):
        value = row.get(key)
        if value not in (None, ""):
            values.add(f"{modality}:id:{value}")
    media_key = "image_path" if modality == "image" else "audio_path"
    value = row.get(media_key)
    if value not in (None, ""):
        values.add(f"{modality}:path:{value}")
    if not values:
        raise ValueError(f"{modality} manifest row {index} has no stable media ID")
    return values


def assert_disjoint_media_ids(
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    modality: str,
) -> None:
    train_sets = [_media_ids(row, modality, index) for index, row in enumerate(train_rows)]
    dev_sets = [_media_ids(row, modality, index) for index, row in enumerate(dev_rows)]

    def flattened(rows: Sequence[set[str]]) -> set[str]:
        output: set[str] = set()
        for identities in rows:
            output.update(identities)
        return output

    # Multiple rows may describe the same media item (for example, COCO's
    # multiple captions). Leakage is defined across splits, not within one.
    train_ids = flattened(train_sets)
    dev_ids = flattened(dev_sets)
    overlap = sorted(train_ids & dev_ids)
    if overlap:
        raise ValueError(
            f"train/dev {modality} media ID overlap ({len(overlap)}): {overlap[:5]}"
        )


def select_fixed_samples(
    rows: Sequence[Dict[str, Any]], sample_count: int, label: str
) -> List[Dict[str, Any]]:
    count = int(sample_count)
    if count <= 0:
        raise ValueError("--sample-count must be positive")
    if len(rows) < count:
        raise ValueError(f"{label} has {len(rows)} rows, fewer than sample-count={count}")
    return [dict(row) for row in rows[:count]]


def resolve_media_paths(
    rows: Sequence[Dict[str, Any]],
    manifest_path: Path,
    modality: str,
    data_root: Path | None = None,
) -> None:
    key = "image_path" if modality == "image" else "audio_path"
    for index, row in enumerate(rows):
        raw = row.get(key)
        if not raw:
            raise ValueError(f"{manifest_path}:{index + 1} is missing {key}")
        reject_forbidden_paths([str(raw)])
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            candidates = []
            if data_root is not None:
                candidates.append(data_root / path)
            candidates.extend((manifest_path.parent / path, Path.cwd() / path))
            existing = {
                candidate.resolve() for candidate in candidates if candidate.is_file()
            }
            if len(existing) > 1:
                raise ValueError(f"ambiguous relative {key} path: {raw}")
            path = next(iter(existing)) if existing else candidates[0]
        path = path.resolve()
        reject_forbidden_paths([path])
        if not path.is_file():
            raise FileNotFoundError(path)
        row[key] = str(path)


def assert_top2_checkpoint_state(state: Mapping[str, Any]) -> None:
    last_row = state.get("last_row")
    if not isinstance(last_row, Mapping):
        raise ValueError("E3 checkpoint is missing last_row routing metadata")
    if str(last_row.get("experiment_id", "")) != E3_EXPERIMENT_ID:
        raise ValueError("checkpoint is not an E3 multimodal checkpoint")
    observed: List[Tuple[str, int]] = []
    for location, mapping in (
        ("last_row", last_row),
        ("args", state.get("args")),
        ("trainable_meta", state.get("trainable_meta")),
    ):
        if isinstance(mapping, Mapping) and mapping.get("top_k") is not None:
            observed.append((location, int(mapping["top_k"])))
    if not observed:
        raise ValueError("checkpoint does not declare its routing top_k")
    if any(value != TOP_K for _location, value in observed):
        raise ValueError(f"checkpoint is not Top-2: {observed}")
    for key in ("image_resampler", "audio_resampler", "args"):
        if key not in state:
            raise ValueError(f"E3 checkpoint is missing architecture/bridge state: {key}")


def validate_e3_stage_b_source(
    state: Mapping[str, Any], stage_b_path: Path, stage_b_sha256: str
) -> Dict[str, Any]:
    trainable_meta = state.get("trainable_meta")
    if not isinstance(trainable_meta, Mapping):
        raise ValueError("E3 checkpoint is missing trainable_meta")
    recorded = trainable_meta.get("stage_b_initialization")
    if not isinstance(recorded, Mapping):
        raise ValueError("E3 checkpoint is missing Stage-B source provenance")
    recorded_path_value = recorded.get("path")
    if not isinstance(recorded_path_value, str) or not recorded_path_value:
        raise ValueError("E3 checkpoint Stage-B source path is missing")
    recorded_path = Path(recorded_path_value).expanduser().resolve(strict=False)
    supplied_path = stage_b_path.resolve()
    if recorded_path != supplied_path:
        raise ValueError(
            "E3 checkpoint Stage-B source path mismatch: "
            f"recorded={recorded_path} supplied={supplied_path}"
        )
    expected_sha = _exact_sha256(stage_b_sha256, "Stage-B checkpoint SHA-256")
    recorded_sha = _exact_sha256(
        recorded.get("sha256"), "E3 checkpoint Stage-B source SHA-256"
    )
    if recorded_sha != expected_sha:
        raise ValueError(
            "E3 checkpoint Stage-B source SHA-256 mismatch: "
            f"recorded={recorded_sha} supplied={expected_sha}"
        )
    required_recorded = {
        "policy": "development_only_stage_b_top8_to_top2_initialization",
        "sealed_evidence_used": False,
        "synthetic_evidence_used": False,
        "state_restored": True,
        "final_inference_top_k": TOP_K,
    }
    for field, expected in required_recorded.items():
        if recorded.get(field) != expected:
            raise ValueError(
                f"E3 checkpoint Stage-B provenance mismatch for {field}: "
                f"expected={expected!r} observed={recorded.get(field)!r}"
            )
    last_row = state.get("last_row")
    if not isinstance(last_row, Mapping):
        raise ValueError("E3 checkpoint is missing last_row routing metadata")
    if last_row.get("stage_b_checkpoint_state_restored") is not True:
        raise ValueError("E3 checkpoint does not prove Stage-B state restoration")
    last_row_sha = _exact_sha256(
        last_row.get("source_stage_b_checkpoint_sha256"),
        "E3 last_row Stage-B source SHA-256",
    )
    if last_row_sha != expected_sha:
        raise ValueError(
            "E3 last_row Stage-B source SHA-256 mismatch: "
            f"recorded={last_row_sha} supplied={expected_sha}"
        )
    run_provenance = state.get("run_provenance")
    if not isinstance(run_provenance, Mapping):
        raise ValueError("E3 checkpoint is missing run_provenance")
    source_commit = str(run_provenance.get("source_commit_sha", "")).lower()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("E3 checkpoint source_commit_sha is missing or invalid")
    for field in ("runai_job_name", "runai_project"):
        if not isinstance(run_provenance.get(field), str) or not str(
            run_provenance[field]
        ).strip():
            raise ValueError(f"E3 checkpoint {field} is missing")
    for field in ("sealed_evidence_used", "synthetic_evidence_used"):
        if run_provenance.get(field) is not False:
            raise ValueError(f"E3 checkpoint must declare {field}=False")
    return {
        "recorded_path": str(recorded_path),
        "recorded_sha256": recorded_sha,
        "last_row_sha256": last_row_sha,
        "run_provenance": dict(run_provenance),
    }


def _verify_recorded_json(
    record: Any, label: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} record is missing")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path is missing")
    reject_forbidden_paths([raw_path])
    return load_exact_json(
        Path(raw_path), str(record.get("sha256", "")), label
    )


def _git_blob_sha256(
    repo_root: Path, source_commit: str, source_path: str, label: str
) -> str:
    if (
        not source_path
        or source_path.startswith("/")
        or ".." in Path(source_path).parts
    ):
        raise ValueError(f"{label} must be a repository-relative path")
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "show",
            f"{source_commit}:{source_path}",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"{label} cannot be resolved at source commit {source_commit}"
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def load_stage_b_companion_manifest(
    path: Path,
    expected_sha256: str,
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_state: Mapping[str, Any],
    expected_base_model: str,
    repo_root: Path,
) -> Dict[str, Any]:
    payload, companion_record = load_exact_json(
        path, expected_sha256, "Stage-B companion manifest"
    )
    required = {
        "schema_version": 1,
        "artifact_type": "development_stage_b_checkpoint_companion",
        "development_only": True,
        "sealed_evidence_used": False,
        "synthetic_evidence_used": False,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"Stage-B companion mismatch for {field}: "
                f"expected={expected!r} observed={payload.get(field)!r}"
            )
    source_commit = verify_git_commit(
        repo_root, payload.get("source_commit_sha"), "Stage-B source commit"
    )
    checkpoint_record = payload.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise ValueError("Stage-B companion checkpoint record is missing")
    recorded_path = Path(str(checkpoint_record.get("path", ""))).resolve(
        strict=False
    )
    if recorded_path != checkpoint.resolve():
        raise ValueError("Stage-B companion checkpoint path mismatch")
    expected_checkpoint_sha = _exact_sha256(
        checkpoint_sha256, "Stage-B checkpoint SHA-256"
    )
    if (
        _exact_sha256(
            checkpoint_record.get("sha256"),
            "Stage-B companion checkpoint SHA-256",
        )
        != expected_checkpoint_sha
    ):
        raise ValueError("Stage-B companion checkpoint SHA-256 mismatch")
    if checkpoint_record.get("size_bytes") != checkpoint.stat().st_size:
        raise ValueError("Stage-B companion checkpoint size mismatch")

    run_manifest, run_manifest_record = _verify_recorded_json(
        payload.get("run_manifest"), "Stage-B run manifest"
    )
    run_provenance = run_manifest.get("run_provenance")
    if not isinstance(run_provenance, Mapping):
        raise ValueError("Stage-B run manifest lacks run_provenance")
    for field in ("source_commit_sha", "runai_job_name", "runai_project"):
        value = run_provenance.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Stage-B run provenance is missing {field}")
        if payload.get(field) != value or run_manifest.get(field) != value:
            raise ValueError(f"Stage-B companion/run manifest {field} mismatch")
    if run_provenance["source_commit_sha"] != source_commit:
        raise ValueError("Stage-B run provenance source commit mismatch")
    if (
        run_provenance.get("policy")
        != "development_only_stage_b_top8_to_top2"
        or run_provenance.get("sealed_evidence_used") is not False
        or run_provenance.get("synthetic_evidence_used") is not False
    ):
        raise ValueError("Stage-B run provenance policy is invalid")
    dataset = payload.get("dataset_split_provenance")
    if (
        not isinstance(dataset, Mapping)
        or run_manifest.get("dataset_split_provenance") != dataset
        or run_provenance.get("dataset_split_provenance") != dataset
        or dataset.get("policy") != "development_only_real_manifests"
        or dataset.get("sealed_evidence_used") is not False
        or dataset.get("synthetic_evidence_used") is not False
    ):
        raise ValueError("Stage-B dataset/split provenance mismatch")
    dataset_files = dataset.get("files")
    required_splits = {
        "text_tasks": "development_calibration_source",
        "train": "train",
        "development_eval": "development_eval",
    }
    if not isinstance(dataset_files, Mapping) or not set(required_splits).issubset(
        dataset_files
    ):
        raise ValueError("Stage-B dataset/split file ledger is incomplete")
    for role, record in dataset_files.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"Stage-B dataset/split record {role} is invalid")
        if role in required_splits and record.get("split") != required_splits[role]:
            raise ValueError(f"Stage-B dataset/split role mismatch for {role}")
        if role == "text_replay" and record.get("split") != "train_replay":
            raise ValueError("Stage-B text replay split provenance mismatch")
        if role not in {*required_splits, "text_replay"}:
            raise ValueError(f"unexpected Stage-B dataset/split role: {role}")
        raw_path = str(record.get("path", ""))
        reject_forbidden_paths([raw_path])
        source_path = _resolve_regular_file(
            Path(raw_path), f"Stage-B dataset/split {role}"
        )
        source_sha = _exact_sha256(
            record.get("sha256"), f"Stage-B dataset/split {role} SHA-256"
        )
        if (
            sha256_file(source_path) != source_sha
            or record.get("size_bytes") != source_path.stat().st_size
            or type(record.get("rows")) is not int
            or record["rows"] <= 0
        ):
            raise ValueError(f"Stage-B dataset/split binding mismatch for {role}")
    run_identity = _stage_b_run_identity(run_provenance)
    if (
        payload.get("run_identity") != run_identity
        or run_manifest.get("run_identity") != run_identity
    ):
        raise ValueError("Stage-B companion/run manifest identity mismatch")
    if checkpoint_state.get("run_identity") != run_identity:
        raise ValueError("Stage-B checkpoint lacks matching internal run identity")
    internal_provenance = checkpoint_state.get("provenance")
    if not isinstance(internal_provenance, Mapping):
        raise ValueError("Stage-B checkpoint lacks internal provenance")
    for field in (
        "run_uuid",
        "source_commit_sha",
        "runai_job_name",
        "runai_project",
        "producer_code",
        "dataset_split_provenance",
        "dataset_split_provenance_sha256",
        "sealed_evidence_used",
        "synthetic_evidence_used",
    ):
        if internal_provenance.get(field) != run_provenance.get(field):
            raise ValueError(
                f"Stage-B checkpoint internal provenance mismatch for {field}"
            )
    if run_manifest.get("data_policy") != "development_only_real_manifests":
        raise ValueError("Stage-B run manifest has invalid data policy")
    run_args = run_manifest.get("args")
    if not isinstance(run_args, Mapping):
        raise ValueError("Stage-B run manifest args are missing")
    if (
        run_args.get("base_model") != expected_base_model
        or int(run_args.get("student_top_k", 0)) != TOP_K
        or int(run_manifest.get("final_inference_top_k", 0)) != TOP_K
    ):
        raise ValueError("Stage-B run manifest model/Top-2 provenance mismatch")

    metrics, metrics_record = _verify_recorded_json(
        payload.get("metrics"), "Stage-B final metrics"
    )
    checkpoint_provenance = metrics.get("checkpoint_provenance")
    if not isinstance(checkpoint_provenance, Mapping):
        raise ValueError("Stage-B final metrics lack checkpoint_provenance")
    for field in (
        "run_uuid",
        "source_commit_sha",
        "runai_job_name",
        "runai_project",
        "producer_code",
        "dataset_split_provenance",
        "dataset_split_provenance_sha256",
    ):
        if checkpoint_provenance.get(field) != run_provenance.get(field):
            raise ValueError(
                f"Stage-B final metrics run provenance mismatch for {field}"
            )
    if checkpoint_provenance.get("stage") != "B":
        raise ValueError("Stage-B final metrics stage mismatch")
    metrics_checkpoint = Path(
        str(checkpoint_provenance.get("saved_checkpoint", ""))
    ).resolve(strict=False)
    if (
        metrics_checkpoint != checkpoint.resolve()
        or checkpoint_provenance.get("saved_checkpoint_sha256")
        != expected_checkpoint_sha
        or checkpoint_provenance.get("saved_checkpoint_size_bytes")
        != checkpoint.stat().st_size
    ):
        raise ValueError("Stage-B final metrics checkpoint binding mismatch")

    code = payload.get("code")
    if not isinstance(code, Mapping):
        raise ValueError("Stage-B companion code provenance is missing")
    code_path = str(code.get("path", ""))
    if code_path != "training/distill_olmoe_top2_real.py":
        raise ValueError("Stage-B companion code path mismatch")
    code_sha = _exact_sha256(code.get("sha256"), "Stage-B producer code SHA-256")
    if run_provenance.get("producer_code") != {
        "path": code_path,
        "sha256": code_sha,
    }:
        raise ValueError("Stage-B run producer code provenance mismatch")
    if (
        _git_blob_sha256(
            repo_root, source_commit, code_path, "Stage-B producer code"
        )
        != code_sha
    ):
        raise ValueError("Stage-B producer code does not match source commit")
    return {
        **companion_record,
        "source_commit_sha": source_commit,
        "runai_job_name": run_provenance["runai_job_name"],
        "runai_project": run_provenance["runai_project"],
        "run_identity": run_identity,
        "checkpoint": dict(checkpoint_record),
        "run_manifest": run_manifest_record,
        "run_provenance": dict(run_provenance),
        "metrics": metrics_record,
        "code": {"path": code_path, "sha256": code_sha},
        "dataset_split_provenance": dict(dataset),
    }


def load_e3_checkpoint_manifest(
    path: Path,
    expected_sha256: str,
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_state: Mapping[str, Any],
    stage_b_checkpoint: Path,
    stage_b_checkpoint_sha256: str,
    split_manifest: Path,
    split_manifest_sha256: str,
    repo_root: Path,
) -> Dict[str, Any]:
    payload, manifest_record = load_exact_json(
        path, expected_sha256, "E3 checkpoint manifest"
    )
    completion = payload.get("completion")
    if not isinstance(completion, Mapping) or completion.get("status") != "completed":
        raise ValueError("E3 checkpoint manifest completion is not completed")
    completion_path = Path(
        str(completion.get("e3_checkpoint_path", ""))
    ).resolve(strict=False)
    if completion_path != checkpoint.resolve():
        raise ValueError("E3 checkpoint manifest completion path mismatch")
    expected_checkpoint_sha = _exact_sha256(
        checkpoint_sha256, "E3 checkpoint SHA-256"
    )
    if completion.get("e3_checkpoint_sha256") != expected_checkpoint_sha:
        raise ValueError("E3 checkpoint manifest completion SHA-256 mismatch")
    if completion.get("e3_checkpoint_size_bytes") != checkpoint.stat().st_size:
        raise ValueError("E3 checkpoint manifest completion size mismatch")

    run_provenance = checkpoint_state.get("run_provenance")
    manifest_run_provenance = payload.get("run_provenance")
    if (
        not isinstance(run_provenance, Mapping)
        or not isinstance(manifest_run_provenance, Mapping)
        or dict(run_provenance) != dict(manifest_run_provenance)
    ):
        raise ValueError("E3 checkpoint and manifest run_provenance disagree")
    source_commit = verify_git_commit(
        repo_root, run_provenance.get("source_commit_sha"), "E3 source commit"
    )
    for field in ("source_commit_sha", "runai_job_name", "runai_project"):
        if payload.get(field) != run_provenance.get(field):
            raise ValueError(f"E3 checkpoint manifest {field} mismatch")
    expected_split_sha = _exact_sha256(
        split_manifest_sha256, "supplied strict split manifest SHA-256"
    )
    resolved_split_manifest = _resolve_regular_file(
        split_manifest, "supplied strict split manifest"
    )
    if sha256_file(resolved_split_manifest) != expected_split_sha:
        raise ValueError("supplied strict split manifest SHA-256 mismatch")
    development_split = payload.get("development_split_provenance")
    if not isinstance(development_split, Mapping):
        raise ValueError("E3 manifest lacks development_split_provenance")
    if (
        Path(str(development_split.get("manifest_path", ""))).resolve(
            strict=False
        )
        != resolved_split_manifest
        or development_split.get("manifest_sha256") != expected_split_sha
    ):
        raise ValueError("E3 manifest strict split bytes mismatch")
    if (
        run_provenance.get("development_split_manifest_sha256")
        != expected_split_sha
    ):
        raise ValueError("E3 checkpoint run provenance strict split SHA mismatch")
    expected_split_source = {
        "path": str(resolved_split_manifest),
        "sha256": expected_split_sha,
    }
    checkpoint_args = checkpoint_state.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("E3 checkpoint args are missing")
    if (
        Path(
            str(checkpoint_args.get("development_split_manifest", ""))
        ).resolve(strict=False)
        != resolved_split_manifest
        or checkpoint_args.get("development_split_manifest_sha256")
        != expected_split_sha
        or checkpoint_args.get("development_split_manifest_source")
        != expected_split_source
    ):
        raise ValueError("E3 checkpoint args strict split source mismatch")

    expected_stage_b_sha = _exact_sha256(
        stage_b_checkpoint_sha256, "Stage-B checkpoint SHA-256"
    )
    stage_b_initialization = payload.get("stage_b_initialization")
    if not isinstance(stage_b_initialization, Mapping):
        raise ValueError("E3 checkpoint manifest lacks Stage-B initialization")
    if (
        Path(str(stage_b_initialization.get("path", ""))).resolve(strict=False)
        != stage_b_checkpoint.resolve()
        or stage_b_initialization.get("sha256") != expected_stage_b_sha
    ):
        raise ValueError("E3 checkpoint manifest Stage-B source mismatch")
    args = payload.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("E3 checkpoint manifest args are missing")
    if (
        Path(str(args.get("stage_b_checkpoint", ""))).resolve(strict=False)
        != stage_b_checkpoint.resolve()
        or args.get("stage_b_checkpoint_sha256") != expected_stage_b_sha
        or args.get("development_split_manifest_sha256") != expected_split_sha
        or args.get("development_split_manifest_source") != expected_split_source
        or Path(str(args.get("development_split_manifest", ""))).resolve(
            strict=False
        )
        != resolved_split_manifest
    ):
        raise ValueError("E3 checkpoint manifest args provenance mismatch")
    return {
        **manifest_record,
        "source_commit_sha": source_commit,
        "development_split_manifest_sha256": expected_split_sha,
        "runai_job_name": run_provenance["runai_job_name"],
        "runai_project": run_provenance["runai_project"],
        "completion": dict(completion),
        "stage_b_initialization": dict(stage_b_initialization),
    }


def _checkpoint_config(state: Mapping[str, Any]) -> SimpleNamespace:
    raw = dict(state.get("args") or {})
    defaults = {
        "base_model": "allenai/OLMoE-1B-7B-0924",
        "vision_model": "openai/clip-vit-base-patch32",
        "speech_model": "openai/whisper-base.en",
        "speech_target_space": "olmoe_text_hidden",
        "alignment_prefix_residual": False,
        "image_prefix_tokens": 50,
        "audio_prefix_tokens": 64,
        "encoder_feature_tokens": 100,
        "sample_rate": 16000,
        "capacity_factor": 4.0,
        "aux_coef": 0.01,
    }
    for key, value in defaults.items():
        raw.setdefault(key, value)
    return SimpleNamespace(**raw)


def _load_gamma(
    gamma_path: Optional[Path], checkpoint: Path, required: bool
) -> Tuple[Optional[List[float]], Optional[Path]]:
    candidate = gamma_path or checkpoint.parent.parent / "calibration" / "gamma.json"
    if not candidate.is_file():
        if required:
            raise FileNotFoundError(f"checkpoint requires calibrated gamma: {candidate}")
        return None, None
    reject_forbidden_paths([candidate])
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    values = payload.get("gamma")
    if not isinstance(values, list) or not values:
        raise ValueError(f"invalid gamma JSON: {candidate}")
    gamma = [float(value) for value in values]
    if not all(math.isfinite(value) and value > 0.0 for value in gamma):
        raise ValueError(f"gamma values must be finite and positive: {candidate}")
    return gamma, candidate.resolve()


def load_e3_wrapper(
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_manifest: Path,
    checkpoint_manifest_sha256: str,
    gamma_path: Optional[Path],
    stage_b_checkpoint: Path,
    stage_b_checkpoint_sha256: str,
    stage_b_companion_manifest: Path,
    stage_b_companion_manifest_sha256: str,
    split_manifest: Path,
    split_manifest_sha256: str,
    repo_root: Path,
) -> Tuple[Any, ...]:
    from training.olmoe_real_subset_runs import make_wrapper
    from training.olmoe_required_runs import (
        iter_olmoe_mlp_layers,
        load_encoders,
        load_model,
    )
    from training.olmoe_real_subset_runs import (
        base_model_identity,
        load_stage_b_initialization_checkpoint,
        restore_stage_b_student_initialization,
        restore_training_checkpoint,
    )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    assert_top2_checkpoint_state(state)
    config = _checkpoint_config(state)
    stage_b_state, stage_b_file_provenance = load_stage_b_initialization_checkpoint(
        str(stage_b_checkpoint), stage_b_checkpoint_sha256
    )
    if stage_b_state is None:
        raise ValueError("Stage-B checkpoint state is required")
    stage_b_companion = load_stage_b_companion_manifest(
        stage_b_companion_manifest,
        stage_b_companion_manifest_sha256,
        checkpoint=stage_b_checkpoint,
        checkpoint_sha256=stage_b_checkpoint_sha256,
        checkpoint_state=stage_b_state,
        expected_base_model=str(config.base_model),
        repo_root=repo_root,
    )
    e3_manifest_provenance = load_e3_checkpoint_manifest(
        checkpoint_manifest,
        checkpoint_manifest_sha256,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_state=state,
        stage_b_checkpoint=stage_b_checkpoint,
        stage_b_checkpoint_sha256=stage_b_checkpoint_sha256,
        split_manifest=split_manifest,
        split_manifest_sha256=split_manifest_sha256,
        repo_root=repo_root,
    )
    stage_b_linkage = validate_e3_stage_b_source(
        state, stage_b_checkpoint, stage_b_file_provenance["sha256"]
    )
    gamma_required = bool((state.get("last_row") or {}).get("gamma_applied"))
    gamma, resolved_gamma_path = _load_gamma(gamma_path, checkpoint, gamma_required)
    model, tokenizer, model_meta = load_model(
        config.base_model,
        TOP_K,
        float(config.aux_coef),
        gamma=gamma,
        capacity_factor=float(config.capacity_factor),
        dynamic_expert_bias=bool(state.get("dynamic_expert_bias")),
        pre_routing_identity_fn=lambda loaded_model: base_model_identity(
            loaded_model, config.base_model
        ),
    )
    runtime_base_model_identity = model_meta.pop(
        "pre_routing_model_identity", None
    )
    if not isinstance(runtime_base_model_identity, Mapping):
        raise ValueError("E3 pre-routing base-model identity is missing")
    device = next(model.parameters()).device
    image_processor, vision_model, speech_processor, speech_model = load_encoders(
        config.vision_model, config.speech_model, device
    )
    wrapper = make_wrapper(model, vision_model, speech_model, config).to(device)
    stage_b_restore = restore_stage_b_student_initialization(
        wrapper,
        stage_b_state,
        str(config.base_model),
        runtime_base_model_identity,
    )
    trainable_meta = state.get("trainable_meta") or {}
    selected_ids = trainable_meta.get("selected_expert_ids_by_layer")
    selection_provenance = trainable_meta.get("expert_selection_provenance")
    if trainable_meta.get("selected_expert_training"):
        if not isinstance(selected_ids, Mapping) or not selected_ids:
            raise ValueError("E3 selected-expert checkpoint is missing selected IDs")
        if not isinstance(selection_provenance, Mapping) or not selection_provenance:
            raise ValueError(
                "E3 selected-expert checkpoint is missing selection provenance"
            )
    restore_training_checkpoint(
        wrapper,
        dict(state),
        speech_model=speech_model,
        expected_selected_expert_ids=selected_ids,
        expected_selection_provenance=selection_provenance,
    )
    if trainable_meta.get("train_experts") and "experts" not in state:
        raise ValueError("checkpoint trained experts but omitted expert weights")
    if "experts" in state:
        expected_layers = {
            f"layer_{layer_index}"
            for layer_index, _mlp in iter_olmoe_mlp_layers(wrapper.lm)
        }
        if set(state["experts"]) != expected_layers:
            raise ValueError("E3 full-expert checkpoint must cover every model layer")
        for layer_index, mlp in iter_olmoe_mlp_layers(wrapper.lm):
            key = f"layer_{layer_index}"
            mlp.experts.load_state_dict(state["experts"][key])
    if trainable_meta.get("train_router_gates"):
        router_state = state.get("router_gates")
        expected_layers = {
            f"layer_{index}" for index in range(len(wrapper.lm.model.layers))
        }
        if not isinstance(router_state, Mapping) or set(router_state) != expected_layers:
            raise ValueError("E3 trainable-router checkpoint must cover every model layer")
    if trainable_meta.get("train_lm_head") and state.get("lm_output_embeddings") is None:
        raise ValueError("E3 trainable-LM-head checkpoint is missing output embeddings")
    wrapper.eval()
    stage_b_provenance = {
        **stage_b_file_provenance,
        "internal_provenance": dict(stage_b_state.get("provenance") or {}),
        "companion_manifest": stage_b_companion,
        "e3_source_linkage": stage_b_linkage,
        "restore": stage_b_restore,
    }
    e3_overlay = {
        "restoration_order": ["stage_b_student", "e3_adapter"],
        "trainable_meta": dict(trainable_meta),
        "selected_expert_selection_provenance": dict(
            state.get("selected_expert_selection_provenance") or {}
        ),
        "bridge_state_restored": True,
        "dynamic_expert_bias_restored": state.get("dynamic_expert_bias") is not None,
        "router_gates_restored": state.get("router_gates") is not None,
        "full_experts_restored": state.get("experts") is not None,
        "selected_experts_restored": state.get("selected_experts") is not None,
        "lm_output_embeddings_restored": state.get("lm_output_embeddings") is not None,
        "lm_input_embeddings_restored": state.get("lm_input_embeddings") is not None,
        "speech_encoder_state_restored": (
            state.get("speech_encoder_trainable_state") is not None
        ),
    }
    return (
        wrapper,
        image_processor,
        vision_model,
        speech_processor,
        speech_model,
        device,
        config,
        model_meta,
        resolved_gamma_path,
        stage_b_provenance,
        e3_overlay,
        dict(state.get("run_provenance") or {}),
        e3_manifest_provenance,
    )


def assert_runtime_top2(model: Any) -> None:
    values: List[Tuple[str, int]] = []
    config_value = getattr(model.config, "num_experts_per_tok", None)
    if config_value is not None:
        values.append(("config.num_experts_per_tok", int(config_value)))
    for object_name, value in (
        ("model.num_experts_per_tok", getattr(model, "num_experts_per_tok", None)),
        (
            "model.model.num_experts_per_tok",
            getattr(getattr(model, "model", None), "num_experts_per_tok", None),
        ),
    ):
        if value is not None:
            values.append((object_name, int(value)))
    if not values or any(value != TOP_K for _name, value in values):
        raise RuntimeError(f"runtime is not Top-2: {values}")
    if getattr(model.config, "output_router_logits", None) is not True:
        raise RuntimeError("runtime must enable output_router_logits")


def _router_tensors(outputs: Any) -> List[torch.Tensor]:
    raw = getattr(outputs, "router_logits", None)
    tensors = [raw] if torch.is_tensor(raw) else list(raw or [])
    if not tensors or not all(torch.is_tensor(tensor) for tensor in tensors):
        raise RuntimeError("shared OLMoE forward returned no router logits")
    return tensors


def canonical_layer_rows(
    outputs: Any,
    *,
    modality: str,
    batch_size: int,
    prefix_tokens: int,
    num_experts: int,
    top_k: int = TOP_K,
    expert_biases: Optional[Sequence[Optional[torch.Tensor]]] = None,
    normalize_topk_probs: Optional[Sequence[bool]] = None,
) -> List[Dict[str, Any]]:
    if modality not in {"image_prefix", "audio_prefix"}:
        raise ValueError(f"unsupported prefix modality: {modality}")
    if int(top_k) != TOP_K:
        raise ValueError("canonical development routing requires Top-2")
    expected_tokens = int(batch_size) * int(prefix_tokens)
    router_tensors = _router_tensors(outputs)
    if expert_biases is not None and len(expert_biases) != len(router_tensors):
        raise ValueError("expert_biases must match the number of router layers")
    if normalize_topk_probs is not None and len(normalize_topk_probs) != len(router_tensors):
        raise ValueError("normalize_topk_probs must match the number of router layers")
    rows: List[Dict[str, Any]] = []
    for layer_index, raw_tensor in enumerate(router_tensors):
        tensor = raw_tensor.detach()
        if tensor.ndim == 2:
            if tensor.shape != (expected_tokens, int(num_experts)):
                raise RuntimeError(
                    f"router layer {layer_index} shape {tuple(tensor.shape)} does not match "
                    f"({expected_tokens}, {num_experts})"
                )
            tensor = tensor.reshape(int(batch_size), int(prefix_tokens), int(num_experts))
        elif tensor.ndim == 3:
            if tensor.shape != (int(batch_size), int(prefix_tokens), int(num_experts)):
                raise RuntimeError(
                    f"router layer {layer_index} shape {tuple(tensor.shape)} is incompatible"
                )
        else:
            raise RuntimeError(f"router layer {layer_index} must be rank 2 or 3")
        selection_logits = tensor.float()
        bias = expert_biases[layer_index] if expert_biases is not None else None
        if bias is not None:
            bias = torch.as_tensor(bias, dtype=torch.float32, device=selection_logits.device)
            if bias.shape != (int(num_experts),):
                raise ValueError(
                    f"expert bias for layer {layer_index} has shape {tuple(bias.shape)}, "
                    f"expected ({num_experts},)"
                )
            if not bool(torch.isfinite(bias).all()):
                raise ValueError(f"expert bias for layer {layer_index} is non-finite")
            selection_logits = selection_logits + bias.view(1, 1, -1)
        probabilities = torch.softmax(selection_logits, dim=-1)
        selected_weights, selected_ids = torch.topk(probabilities, TOP_K, dim=-1)
        normalize = bool(normalize_topk_probs[layer_index]) if normalize_topk_probs else False
        if normalize:
            selected_weights = selected_weights / (
                selected_weights.sum(dim=-1, keepdim=True) + 1e-9
            )
        flat_ids = selected_ids.reshape(-1).detach().cpu()
        flat_weights = selected_weights.reshape(-1).detach().double().cpu()
        counts = torch.bincount(flat_ids, minlength=int(num_experts))
        score_sums = torch.zeros(int(num_experts), dtype=torch.float64)
        score_sums.scatter_add_(0, flat_ids, flat_weights)
        observed = int(counts.sum().item())
        expected = expected_tokens * TOP_K
        row = {
            "layer": int(layer_index),
            "modality": modality,
            "token_count": expected_tokens,
            "top_k": TOP_K,
            "expected_assignments_tokens_x_k": expected,
            "attempted_expert_counts": counts.tolist(),
            "gate_score_sums": score_sums.tolist(),
            "attempted_assignments": observed,
            "conservation_ok": observed == expected,
            "selection_bias_applied": bias is not None,
            "selection_accounting_source": (
                "router_logits_plus_dynamic_expert_bias_pre_capacity"
                if bias is not None
                else "router_logits_pre_capacity"
            ),
            "topk_weights_normalized": normalize,
        }
        validate_canonical_layer_row(row)
        rows.append(row)
    return rows


def validate_canonical_layer_row(row: Mapping[str, Any]) -> None:
    top_k = int(row["top_k"])
    token_count = int(row["token_count"])
    counts = torch.as_tensor(row["attempted_expert_counts"], dtype=torch.long)
    scores = torch.as_tensor(row["gate_score_sums"], dtype=torch.float64)
    if top_k != TOP_K:
        raise ValueError("routing row is not Top-2")
    if token_count <= 0 or counts.ndim != 1 or scores.shape != counts.shape:
        raise ValueError("routing row has invalid token count or expert vectors")
    if bool((counts < 0).any()) or not bool(torch.isfinite(scores).all()) or bool((scores < 0).any()):
        raise ValueError("routing row has invalid counts or gate scores")
    expected = token_count * top_k
    observed = int(counts.sum().item())
    if observed != expected or row.get("conservation_ok") is not True:
        raise ValueError(
            f"routing conservation failure: expected {expected}, observed {observed}"
        )


def aggregate_layer_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    aggregated: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for row in rows:
        validate_canonical_layer_row(row)
        key = (int(row["layer"]), str(row["modality"]))
        counts = torch.as_tensor(row["attempted_expert_counts"], dtype=torch.long)
        scores = torch.as_tensor(row["gate_score_sums"], dtype=torch.float64)
        current = aggregated.get(key)
        if current is None:
            current = {
                "layer": key[0],
                "modality": key[1],
                "token_count": 0,
                "top_k": TOP_K,
                "attempted_expert_counts": torch.zeros_like(counts),
                "gate_score_sums": torch.zeros_like(scores),
                "selection_bias_applied": bool(row.get("selection_bias_applied", False)),
                "selection_accounting_source": str(row.get("selection_accounting_source", "")),
                "topk_weights_normalized": bool(row.get("topk_weights_normalized", False)),
            }
            aggregated[key] = current
        if current["attempted_expert_counts"].shape != counts.shape:
            raise ValueError(f"layer {key[0]} changes num_experts between batches")
        for field in (
            "selection_bias_applied",
            "selection_accounting_source",
            "topk_weights_normalized",
        ):
            if current[field] != row.get(field, current[field]):
                raise ValueError(f"layer {key[0]} changes {field} between batches")
        current["token_count"] += int(row["token_count"])
        current["attempted_expert_counts"] += counts
        current["gate_score_sums"] += scores
    output: List[Dict[str, Any]] = []
    for key in sorted(aggregated):
        current = aggregated[key]
        counts = current["attempted_expert_counts"]
        token_count = int(current["token_count"])
        expected = token_count * TOP_K
        row = {
            "layer": int(current["layer"]),
            "modality": str(current["modality"]),
            "token_count": token_count,
            "top_k": TOP_K,
            "expected_assignments_tokens_x_k": expected,
            "attempted_expert_counts": counts.tolist(),
            "gate_score_sums": current["gate_score_sums"].tolist(),
            "attempted_assignments": int(counts.sum().item()),
            "conservation_ok": int(counts.sum().item()) == expected,
            "selection_bias_applied": current["selection_bias_applied"],
            "selection_accounting_source": current["selection_accounting_source"],
            "topk_weights_normalized": current["topk_weights_normalized"],
        }
        validate_canonical_layer_row(row)
        output.append(row)
    if not output:
        raise ValueError("no prefix routing rows were collected")
    return output


def build_outer_row(
    *,
    split: str,
    modality: str,
    sample_count: int,
    batch_count: int,
    layer_rows: Sequence[Mapping[str, Any]],
    num_experts: int,
    accounting_sources: Sequence[str],
    source_manifest_key: str,
    source_manifest_sha256: str,
    strict_split_manifest_sha256: str,
) -> Dict[str, Any]:
    canonical = [dict(row) for row in layer_rows]
    for row in canonical:
        validate_canonical_layer_row(row)
    if split not in {"train", "dev"}:
        raise ValueError("split must be train or dev")
    expected_source_key = (
        f"image_{split}" if modality == "image_prefix" else f"speech_{split}"
    )
    if source_manifest_key != expected_source_key:
        raise ValueError(
            f"source manifest key mismatch: expected={expected_source_key} "
            f"observed={source_manifest_key}"
        )
    source_sha = _exact_sha256(
        source_manifest_sha256, "source manifest SHA-256"
    )
    split_manifest_sha = _exact_sha256(
        strict_split_manifest_sha256, "strict split manifest SHA-256"
    )
    prefix_tokens_across_layers = sum(int(row["token_count"]) for row in canonical)
    expected = prefix_tokens_across_layers * TOP_K
    observed = sum(sum(row["attempted_expert_counts"]) for row in canonical)
    if expected <= 0 or observed != expected:
        raise ValueError(
            f"outer-row conservation failure: expected {expected}, observed {observed}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "development_real_prefix_routing",
        "development_only": True,
        "real_subset": True,
        "split": split,
        "modality": modality,
        "source_manifest_key": source_manifest_key,
        "source_manifest_sha256": source_sha,
        "strict_split_manifest_sha256": split_manifest_sha,
        "sample_count": int(sample_count),
        "batch_count": int(batch_count),
        "top_k": TOP_K,
        "num_experts": int(num_experts),
        "router_layers": len(canonical),
        "shared_olmoe_prefix_path": True,
        "prefix_routing_included": True,
        "prefix_token_count_across_layers": prefix_tokens_across_layers,
        "prefix_expected_assignments_tokens_x_layers_x_k": expected,
        "prefix_observed_assignments": observed,
        "prefix_token_k_conservation_ok": True,
        "modality_routing_accounting_sources": sorted(set(accounting_sources)),
        "modality_layer_accounting": canonical,
    }


def collect_modality(
    *,
    wrapper: Any,
    image_processor: Any,
    vision_model: Any,
    speech_processor: Any,
    speech_model: Any,
    rows: Sequence[Dict[str, Any]],
    modality: str,
    device: torch.device,
    config: Any,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[str], int]:
    from training.olmoe_real_subset_runs import (
        audio_features_from_paths,
        image_features_from_paths,
    )
    from training.olmoe_required_runs import modality_router_metrics

    mlp_layers = [
        getattr(layer, "mlp", None)
        for layer in getattr(getattr(wrapper.lm, "model", None), "layers", [])
    ]
    if not mlp_layers or any(mlp is None for mlp in mlp_layers):
        raise RuntimeError("could not resolve OLMoE MLP layers for routing accounting")
    expert_biases: List[Optional[torch.Tensor]] = []
    normalize_topk_probs: List[bool] = []
    for mlp in mlp_layers:
        enabled = bool(getattr(mlp, "dynamic_expert_bias_enabled", False))
        expert_biases.append(getattr(mlp, "expert_bias", None) if enabled else None)
        normalize_topk_probs.append(
            bool(getattr(mlp, "dynamic_expert_bias_norm_topk", False)) if enabled else False
        )

    if int(batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    prefix_name = "image_prefix" if modality == "image" else "audio_prefix"
    prefix_tokens = int(
        config.image_prefix_tokens if modality == "image" else config.audio_prefix_tokens
    )
    all_rows: List[Dict[str, Any]] = []
    accounting_sources: List[str] = []
    batches = 0
    target_dtype = wrapper.lm.get_input_embeddings().weight.dtype
    for start in range(0, len(rows), int(batch_size)):
        batch_rows = list(rows[start : start + int(batch_size)])
        if modality == "image":
            encoder_tokens = image_features_from_paths(
                image_processor,
                vision_model,
                [str(row["image_path"]) for row in batch_rows],
                device,
                int(config.encoder_feature_tokens),
            )
            prefix = wrapper.image_prefix(encoder_tokens)
        else:
            encoder_tokens = audio_features_from_paths(
                speech_processor,
                speech_model,
                [str(row["audio_path"]) for row in batch_rows],
                device,
                int(config.sample_rate),
                int(config.encoder_feature_tokens),
            )
            prefix = wrapper.audio_prefix(encoder_tokens)
        if prefix.shape[1] != prefix_tokens:
            raise RuntimeError(
                f"checkpoint resampler emitted {prefix.shape[1]} tokens, expected {prefix_tokens}"
            )
        lm_prefix = prefix.to(device=device, dtype=target_dtype)
        attention_mask = torch.ones(lm_prefix.shape[:2], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = wrapper.lm(
                inputs_embeds=lm_prefix,
                attention_mask=attention_mask,
                output_router_logits=True,
                return_dict=True,
            )
        metrics = modality_router_metrics(
            outputs,
            TOP_K,
            int(wrapper.lm.config.num_experts),
            len(batch_rows),
            prefix_tokens if modality == "image" else 0,
            prefix_tokens if modality == "speech" else 0,
            0,
        )
        conservation = metrics.get("modality_assignment_conservation", {})
        if conservation and conservation.get(prefix_name) is not True:
            raise ValueError(f"modality_router_metrics conservation failed for {prefix_name}")
        accounting_sources.append(
            str(metrics.get("modality_routing_accounting_source", "unknown"))
        )
        all_rows.extend(
            canonical_layer_rows(
                outputs,
                modality=prefix_name,
                batch_size=len(batch_rows),
                prefix_tokens=prefix_tokens,
                num_experts=int(wrapper.lm.config.num_experts),
                expert_biases=expert_biases,
                normalize_topk_probs=normalize_topk_probs,
            )
        )
        batches += 1
    return aggregate_layer_rows(all_rows), accounting_sources, batches


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--stage-b-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-stage-b-checkpoint-sha256", required=True)
    parser.add_argument("--stage-b-companion-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-stage-b-companion-manifest-sha256", required=True
    )
    parser.add_argument("--source-commit-sha", required=True)
    parser.add_argument("--gamma-json", type=Path)
    parser.add_argument("--train-image-manifest", type=Path, required=True)
    parser.add_argument("--train-speech-manifest", type=Path, required=True)
    parser.add_argument("--dev-image-manifest", type=Path)
    parser.add_argument("--dev-speech-manifest", type=Path)
    parser.add_argument("--development-split-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-development-split-manifest-sha256", required=True
    )
    parser.add_argument(
        "--collection-split", choices=sorted(SPLIT_FILE_KEYS), default="train"
    )
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("development prefix routing collection requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"collector requires exactly one visible CUDA GPU, observed {torch.cuda.device_count()}"
        )
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if args.collection_split == "train":
        if args.dev_image_manifest is not None or args.dev_speech_manifest is not None:
            raise ValueError(
                "train-only collection rejects dev manifests so they cannot be read"
            )
    elif args.dev_image_manifest is None or args.dev_speech_manifest is None:
        raise ValueError("train-dev collection requires both dev manifests")
    paths = {
        "checkpoint": args.checkpoint.resolve(),
        "checkpoint_manifest": args.checkpoint_manifest.resolve(),
        "stage_b_checkpoint": args.stage_b_checkpoint.resolve(),
        "stage_b_companion_manifest": args.stage_b_companion_manifest.resolve(),
        "split_manifest": args.development_split_manifest.resolve(),
        "image_train": args.train_image_manifest.resolve(),
        "speech_train": args.train_speech_manifest.resolve(),
        "output": args.output_dir.resolve(),
    }
    if args.collection_split == "train-dev":
        paths["image_dev"] = args.dev_image_manifest.resolve()
        paths["speech_dev"] = args.dev_speech_manifest.resolve()
    if args.gamma_json is not None:
        paths["gamma"] = args.gamma_json.resolve()
    reject_forbidden_paths(paths.values())
    if paths["output"].exists():
        raise FileExistsError(f"refusing to overwrite output directory: {paths['output']}")
    repo_root = args.repo_root.resolve(strict=True)
    source_commit = verify_source_commit(repo_root, args.source_commit_sha)
    checkpoint_sha = verify_checkpoint_sha(
        paths["checkpoint"], args.expected_checkpoint_sha256
    )
    stage_b_checkpoint_sha = verify_checkpoint_sha(
        paths["stage_b_checkpoint"], args.expected_stage_b_checkpoint_sha256
    )

    supplied_manifest_paths = {
        key: paths[key] for key in SPLIT_FILE_KEYS[args.collection_split]
    }
    split_manifest, verified_split_records = load_strict_split_manifest(
        paths["split_manifest"],
        args.expected_development_split_manifest_sha256,
        collection_split=args.collection_split,
        supplied_paths=supplied_manifest_paths,
        repo_root=repo_root,
    )
    data_root_value = split_manifest.get("data_dir")
    if not isinstance(data_root_value, str) or not data_root_value:
        raise ValueError("strict split manifest is missing data_dir")
    data_root = Path(data_root_value).resolve(strict=True)
    reject_forbidden_paths([data_root])
    if not data_root.is_dir():
        raise ValueError("strict split manifest data_dir is not a directory")
    manifests = {
        key: read_manifest(Path(record["path"]))
        for key, record in verified_split_records.items()
    }
    for key, rows in manifests.items():
        if len(rows) != int(verified_split_records[key]["rows"]):
            raise ValueError(
                f"strict split manifest count mismatch for {key}: "
                f"expected={verified_split_records[key]['rows']} observed={len(rows)}"
            )
        validate_strict_split_rows(rows, key)
        media_key = "image_path" if key.startswith("image") else "audio_path"
        reject_forbidden_paths(
            str(row[media_key]) for row in rows if row.get(media_key) not in (None, "")
        )
    if args.collection_split == "train-dev":
        assert_disjoint_media_ids(
            manifests["image_train"], manifests["image_dev"], "image"
        )
        assert_disjoint_media_ids(
            manifests["speech_train"], manifests["speech_dev"], "speech"
        )
    selected: Dict[str, List[Dict[str, Any]]] = {}
    for key, rows in manifests.items():
        selected[key] = select_fixed_samples(rows, args.sample_count, key)
        modality = "image" if key.startswith("image") else "speech"
        resolve_media_paths(selected[key], paths[key], modality, data_root)

    (
        wrapper,
        image_processor,
        vision_model,
        speech_processor,
        speech_model,
        device,
        config,
        model_meta,
        resolved_gamma_path,
        stage_b_provenance,
        e3_overlay,
        e3_run_provenance,
        e3_manifest_provenance,
    ) = load_e3_wrapper(
        paths["checkpoint"],
        checkpoint_sha,
        paths["checkpoint_manifest"],
        args.expected_checkpoint_manifest_sha256,
        paths.get("gamma"),
        paths["stage_b_checkpoint"],
        stage_b_checkpoint_sha,
        paths["stage_b_companion_manifest"],
        args.expected_stage_b_companion_manifest_sha256,
        paths["split_manifest"],
        split_manifest["_verified"]["sha256"],
        repo_root,
    )
    if device.type != "cuda":
        raise RuntimeError(f"loaded E3 runtime is not CUDA: {device}")
    assert_runtime_top2(wrapper.lm)
    num_experts = int(wrapper.lm.config.num_experts)

    collection_splits = (
        ("train",) if args.collection_split == "train" else ("train", "dev")
    )
    output_rows: Dict[str, List[Dict[str, Any]]] = {
        split: [] for split in collection_splits
    }
    for split in collection_splits:
        for modality in ("image", "speech"):
            layer_rows, sources, batch_count = collect_modality(
                wrapper=wrapper,
                image_processor=image_processor,
                vision_model=vision_model,
                speech_processor=speech_processor,
                speech_model=speech_model,
                rows=selected[f"{modality}_{split}"],
                modality=modality,
                device=device,
                config=config,
                batch_size=args.batch_size,
            )
            output_rows[split].append(
                build_outer_row(
                    split=split,
                    modality=f"{modality}_prefix" if modality == "image" else "audio_prefix",
                    sample_count=args.sample_count,
                    batch_count=batch_count,
                    layer_rows=layer_rows,
                    num_experts=num_experts,
                    accounting_sources=sources,
                    source_manifest_key=f"{modality}_{split}",
                    source_manifest_sha256=verified_split_records[
                        f"{modality}_{split}"
                    ]["sha256"],
                    strict_split_manifest_sha256=split_manifest["_verified"][
                        "sha256"
                    ],
                )
            )

    output_dir = paths["output"]
    output_dir.mkdir(parents=True)
    output_paths = {
        split: output_dir / f"{split}.jsonl" for split in collection_splits
    }
    for split, output_path in output_paths.items():
        _write_jsonl(output_path, output_rows[split])
    collector_path = Path(__file__).resolve()
    source_manifest_records = {
        key: {
            "path": str(paths[key]),
            "sha256": sha256_file(paths[key]),
            "rows": len(manifests[key]),
            "selected_samples": len(selected[key]),
            "strict_split_record": verified_split_records[key],
        }
        for key in SPLIT_FILE_KEYS[args.collection_split]
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "development_real_prefix_routing_collection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "real_subset": True,
        "sealed": False,
        "synthetic": False,
        "shared_olmoe_prefix_path": True,
        "prefix_routing_included": True,
        "sample_policy": "first_n_manifest_order",
        "collection_split": args.collection_split,
        "collection_splits": list(collection_splits),
        "dev_files_read": args.collection_split == "train-dev",
        "eval_files_read": False,
        "sample_count_per_manifest": int(args.sample_count),
        "batch_size": int(args.batch_size),
        "top_k": TOP_K,
        "num_experts": num_experts,
        "checkpoint": {
            "path": str(paths["checkpoint"]),
            "sha256": checkpoint_sha,
            "experiment_id": E3_EXPERIMENT_ID,
            "architecture_and_bridge_state_restored": True,
            "run_provenance": e3_run_provenance,
            "companion_manifest": e3_manifest_provenance,
        },
        "stage_b_checkpoint": stage_b_provenance,
        "model_state_restoration": e3_overlay,
        "strict_split_manifest": {
            **split_manifest["_verified"],
            "schema_version": split_manifest["schema_version"],
            "split_policy": split_manifest.get("split_policy"),
            "counts": {
                key: split_manifest["counts"][key]
                for key in SPLIT_FILE_KEYS[args.collection_split]
            },
            "declared_source_files": split_manifest.get("source_files"),
        },
        "source_manifests": source_manifest_records,
        "code": {
            "source_commit_sha": source_commit,
            "collector_path": str(collector_path),
            "collector_sha256": sha256_file(collector_path),
        },
        "gamma": {
            "path": str(resolved_gamma_path) if resolved_gamma_path is not None else None,
            "sha256": sha256_file(resolved_gamma_path)
            if resolved_gamma_path is not None
            else None,
        },
        "model": {
            "base_model": config.base_model,
            "vision_model": config.vision_model,
            "speech_model": config.speech_model,
            "image_prefix_tokens": int(config.image_prefix_tokens),
            "audio_prefix_tokens": int(config.audio_prefix_tokens),
            "encoder_feature_tokens": int(config.encoder_feature_tokens),
            "runtime": model_meta,
        },
        "outputs": {
            split: {
                "path": str(output_paths[split]),
                "sha256": sha256_file(output_paths[split]),
                "rows": len(output_rows[split]),
                "samples": sum(
                    row["sample_count"] for row in output_rows[split]
                ),
            }
            for split in collection_splits
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "train_rows": len(output_rows["train"]),
                "dev_rows": len(output_rows.get("dev", [])),
                "sample_count_per_manifest": int(args.sample_count),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
