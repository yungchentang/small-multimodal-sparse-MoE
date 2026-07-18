#!/usr/bin/env python3
"""Resolve manifest media paths without changing row identity or content."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class ManifestError(RuntimeError):
    """Raised when path canonicalization would violate the source contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ManifestError(f"{path}:{line_number} is blank")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ManifestError(f"{path}:{line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ManifestError(f"{path} is empty")
    return rows


def canonicalize(
    source: Path, data_root: Path, output: Path, media_key: str, hash_key: str
) -> dict[str, Any]:
    source = source.resolve(strict=True)
    data_root = data_root.resolve(strict=True)
    rows = read_rows(source)
    output_rows: list[dict[str, Any]] = []
    media_hashes: list[str] = []
    for index, source_row in enumerate(rows):
        raw = source_row.get(media_key)
        if not isinstance(raw, str) or not raw:
            raise ManifestError(f"row {index} lacks {media_key}")
        candidate = Path(raw)
        resolved = candidate.resolve(strict=True) if candidate.is_absolute() else (data_root / candidate).resolve(strict=True)
        if not resolved.is_file():
            raise ManifestError(f"row {index} media is not a file: {resolved}")
        observed = sha256_file(resolved)
        if source_row.get(hash_key) != observed:
            raise ManifestError(f"row {index} {hash_key} does not match media bytes")
        row = dict(source_row)
        row[media_key] = str(resolved)
        output_rows.append(row)
        media_hashes.append(observed)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    reread = read_rows(output)
    for index, (before, after) in enumerate(zip(rows, reread)):
        restored = dict(after)
        restored[media_key] = before[media_key]
        if restored != before:
            raise ManifestError(f"row {index} changed outside {media_key}")
    return {
        "schema_version": 1,
        "operation": "absolute_media_path_canonicalization_only",
        "source": {"path": str(source), "sha256": sha256_file(source), "rows": len(rows)},
        "output": {"path": str(output.resolve()), "sha256": sha256_file(output), "rows": len(rows)},
        "data_root": str(data_root),
        "media_key": media_key,
        "hash_key": hash_key,
        "all_media_hashes_verified": True,
        "non_path_fields_unchanged": True,
        "aggregate_media_sha256": hashlib.sha256("\n".join(media_hashes).encode("ascii")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--media-key", required=True)
    parser.add_argument("--hash-key", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (args.output.exists() or args.provenance.exists()) and not args.force:
        raise ManifestError("refusing to overwrite outputs without --force")
    result = canonicalize(
        args.source, args.data_root, args.output, args.media_key, args.hash_key
    )
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
