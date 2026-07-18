#!/usr/bin/env python3
"""Build deterministic, development-only image and speech routing splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE_DIR = Path("data/real_subset_clean_260708b")
DEFAULT_OUTPUT_DIR = Path("data/development_routing_splits")
TRAIN_COUNT = 5_000
DEV_COUNT = 250
FORBIDDEN_PATH_MARKERS = ("sealed", "synthetic")
OUTPUT_NAMES = {
    "image_train": "image_train.jsonl",
    "speech_train": "speech_train.jsonl",
    "image_dev": "image_dev.jsonl",
    "speech_dev": "speech_dev.jsonl",
    "reconstruction_dev": "reconstruction_dev.jsonl",
    "manifest": "manifest.json",
}


class SplitBuildError(ValueError):
    """Raised when development split provenance cannot be established safely."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def reject_forbidden_path(path: Path, label: str) -> None:
    lowered_parts = [part.lower() for part in path.parts]
    marker = next(
        (marker for marker in FORBIDDEN_PATH_MARKERS if any(marker in part for part in lowered_parts)),
        None,
    )
    if marker is not None:
        raise SplitBuildError(f"{label} uses forbidden {marker} path: {path}")


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise SplitBuildError(f"source JSONL is missing or not a regular file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise SplitBuildError(f"blank JSONL row: {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SplitBuildError(f"invalid JSONL: {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise SplitBuildError(f"JSONL object required: {path}:{line_number}")
            rows.append(row)
    return rows, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "bytes": path.stat().st_size,
    }


def media_candidates(raw: Path, source_dir: Path) -> Iterable[Path]:
    if raw.is_absolute():
        yield raw
        return
    yield source_dir / raw
    for parent in source_dir.parents:
        yield parent / raw


def resolve_media(value: Any, source_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SplitBuildError(f"{label} must be a non-empty string")
    raw = Path(value)
    reject_forbidden_path(raw, label)
    for candidate in media_candidates(raw, source_dir):
        if candidate.is_file():
            resolved = candidate.resolve()
            reject_forbidden_path(resolved, label)
            return resolved
    raise SplitBuildError(f"missing media file for {label}: {value}")


def validate_id(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)) or value == "":
        raise SplitBuildError(f"{label} has missing or invalid id: {value!r}")
    return f"{type(value).__name__}:{value}"


def normalize_rows(
    rows: list[dict[str, Any]],
    *,
    modality: str,
    source_dir: Path,
) -> list[dict[str, Any]]:
    path_field = "image_path" if modality == "image" else "audio_path"
    normalized: list[dict[str, Any]] = []
    for index, original in enumerate(rows):
        label = f"{modality} row {index}"
        if original.get("task") != modality:
            raise SplitBuildError(f"{label} has task {original.get('task')!r}, expected {modality!r}")
        validate_id(original.get("id"), label)
        media_path = resolve_media(original.get(path_field), source_dir, f"{label} {path_field}")
        if modality == "speech":
            preprocess = original.get("preprocess")
            resampled_to = preprocess.get("resampled_to") if isinstance(preprocess, dict) else None
            if original.get("sample_rate") != 16_000 or resampled_to != 16_000:
                raise SplitBuildError(
                    f"{label} has non-16k speech metadata: "
                    f"sample_rate={original.get('sample_rate')!r}, resampled_to={resampled_to!r}"
                )
        row = dict(original)
        row[path_field] = str(media_path)
        normalized.append(row)
    return normalized


def uniqueness_keys(rows: list[dict[str, Any]], modality: str, split: str) -> tuple[set[str], set[str]]:
    path_field = "image_path" if modality == "image" else "audio_path"
    ids: set[str] = set()
    paths: set[str] = set()
    for index, row in enumerate(rows):
        row_id = validate_id(row.get("id"), f"{modality} {split} row {index}")
        media_path = str(row[path_field])
        if row_id in ids:
            raise SplitBuildError(f"duplicate {modality} ID in {split}: {row.get('id')!r}")
        if media_path in paths:
            raise SplitBuildError(f"duplicate {modality} path in {split}: {media_path}")
        ids.add(row_id)
        paths.add(media_path)
    return ids, paths


def validate_partitions(
    train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], modality: str
) -> None:
    train_ids, train_paths = uniqueness_keys(train_rows, modality, "train")
    dev_ids, dev_paths = uniqueness_keys(dev_rows, modality, "dev")
    overlapping_ids = train_ids & dev_ids
    overlapping_paths = train_paths & dev_paths
    if overlapping_ids or overlapping_paths:
        details = []
        if overlapping_ids:
            details.append(f"IDs={len(overlapping_ids)}")
        if overlapping_paths:
            details.append(f"paths={len(overlapping_paths)}")
        raise SplitBuildError(f"{modality} train/dev overlap: {', '.join(details)}")


def canonical_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
            for row in rows
        )
    ).encode("utf-8")


def id_range(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"first_id": rows[0]["id"], "last_id": rows[-1]["id"]}


def build_reconstruction_rows(
    image_dev: list[dict[str, Any]],
    speech_dev: list[dict[str, Any]],
    prompts_per_modality: int,
) -> list[dict[str, Any]]:
    if prompts_per_modality <= 0:
        raise SplitBuildError("reconstruction prompts per modality must be positive")
    output: list[dict[str, Any]] = []
    for modality, rows, text_field in (
        ("image", image_dev, "caption"),
        ("speech", speech_dev, "transcript"),
    ):
        for row in rows[:prompts_per_modality]:
            text = row.get(text_field)
            if not isinstance(text, str) or not text.strip():
                raise SplitBuildError(
                    f"{modality} dev row {row.get('id')!r} has no {text_field}"
                )
            output.append(
                {
                    "id": f"{modality}:{row['id']}",
                    "modality": modality,
                    "real_subset": True,
                    "source": row.get("source"),
                    "source_id": row["id"],
                    "split": "dev",
                    "text": text,
                }
            )
    return output


def output_record(path: Path, payload: bytes, rows: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_bytes(payload),
        "rows": rows,
        "bytes": len(payload),
    }


def write_all_atomic(materializations: list[tuple[Path, bytes]]) -> None:
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for destination, payload in materializations:
            with tempfile.NamedTemporaryFile(
                "wb", dir=destination.parent, prefix=destination.name + ".", delete=False
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_paths.append((Path(handle.name), destination))
        for temporary, destination in temporary_paths:
            os.replace(temporary, destination)
    finally:
        for temporary, _destination in temporary_paths:
            temporary.unlink(missing_ok=True)


def build_splits(
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    train_count: int = TRAIN_COUNT,
    dev_count: int = DEV_COUNT,
    reconstruction_prompts_per_modality: int = 16,
) -> dict[str, Any]:
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if train_count <= 0 or dev_count <= 0:
        raise SplitBuildError("train and dev counts must both be positive")
    reject_forbidden_path(source, "source directory")
    reject_forbidden_path(output, "output directory")
    if not source.is_dir() or source.is_symlink():
        raise SplitBuildError(f"source directory is missing or not a regular directory: {source}")
    if path_exists(output):
        raise SplitBuildError(f"refusing existing output directory: {output}")

    destinations = {key: output / name for key, name in OUTPUT_NAMES.items()}
    existing = [path for path in destinations.values() if path_exists(path)]
    if existing:
        raise SplitBuildError(f"refusing existing output: {existing[0]}")

    image_rows, image_source_record = load_jsonl(source / "image_captions.jsonl")
    speech_rows, speech_source_record = load_jsonl(source / "speech_transcripts.jsonl")
    expected_count = train_count + dev_count
    for modality, rows in (("image", image_rows), ("speech", speech_rows)):
        if len(rows) != expected_count:
            raise SplitBuildError(
                f"wrong {modality} source count: expected {expected_count}, got {len(rows)}"
            )

    normalized = {
        "image": normalize_rows(image_rows, modality="image", source_dir=source),
        "speech": normalize_rows(speech_rows, modality="speech", source_dir=source),
    }
    splits: dict[str, list[dict[str, Any]]] = {}
    for modality, rows in normalized.items():
        splits[f"{modality}_train"] = rows[:train_count]
        splits[f"{modality}_dev"] = rows[-dev_count:]
        validate_partitions(
            splits[f"{modality}_train"], splits[f"{modality}_dev"], modality
        )

    reconstruction_dev = build_reconstruction_rows(
        splits["image_dev"],
        splits["speech_dev"],
        reconstruction_prompts_per_modality,
    )
    materialized_rows = {**splits, "reconstruction_dev": reconstruction_dev}
    payloads = {name: canonical_jsonl(rows) for name, rows in materialized_rows.items()}
    output_files = {
        name: output_record(destinations[name], payloads[name], len(materialized_rows[name]))
        for name in materialized_rows
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "builder": "scripts/build_development_routing_splits.py",
        "source_files": {
            "image": image_source_record,
            "speech": speech_source_record,
        },
        "output_files": output_files,
        "counts": {
            "image_source": len(image_rows),
            "speech_source": len(speech_rows),
            "image_train": len(splits["image_train"]),
            "speech_train": len(splits["speech_train"]),
            "image_dev": len(splits["image_dev"]),
            "speech_dev": len(splits["speech_dev"]),
            "reconstruction_dev": len(reconstruction_dev),
        },
        "id_ranges": {name: id_range(rows) for name, rows in materialized_rows.items()},
        "split_policy": {
            "ordering": "source_jsonl_order",
            "train": f"first {train_count} rows",
            "dev": f"last {dev_count} rows",
            "source_rows_required_per_modality": expected_count,
            "train_count": train_count,
            "dev_count": dev_count,
            "reconstruction": (
                f"first up to {reconstruction_prompts_per_modality} real dev rows per modality; "
                "text copied from caption/transcript"
            ),
        },
        "semantic_rows_changed": False,
        "media_paths_normalized_to_absolute": True,
        "non_path_semantic_fields_changed": False,
        "sealed_data_used": False,
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    output.mkdir(parents=True, exist_ok=True)
    write_all_atomic(
        [(destinations[name], payloads[name]) for name in materialized_rows]
        + [(destinations["manifest"], manifest_payload)]
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reconstruction-prompts-per-modality", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_splits(
            args.source_dir,
            args.output_dir,
            reconstruction_prompts_per_modality=args.reconstruction_prompts_per_modality,
        )
    except SplitBuildError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
