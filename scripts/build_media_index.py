#!/usr/bin/env python3
"""Build a content-free media index for train/development overlap auditing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    header = f"RGB:{image.width}x{image.height}\n".encode("ascii")
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve(path_value: Any, data_dir: Path) -> Path:
    path = Path(str(path_value))
    candidates = [path, data_dir / path, Path.cwd() / path]
    resolved = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
    if resolved is None:
        raise FileNotFoundError(path)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/real_subset_clean_260708b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dev-tail", type=int, default=250)
    parser.add_argument("--speech-dev-tail", type=int, default=250)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_rows = read_jsonl(data_dir / "image_captions.jsonl")
    speech_rows = read_jsonl(data_dir / "speech_transcripts.jsonl")
    output_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(image_rows):
        path = resolve(row["image_path"], data_dir)
        output_rows.append({
            "modality": "image",
            "task": "image_captioning",
            "id": row.get("id", index),
            "split_role": "development_selection" if index >= len(image_rows) - args.image_dev_tail else "train",
            "resized_content_sha256": pixel_sha256(path),
            "media_sha256": sha256_file(path),
            "source_ids": [f"legacy_image_row:{row.get('source')}:{row.get('id', index)}"],
        })
    for index, row in enumerate(speech_rows):
        path = resolve(row["audio_path"], data_dir)
        output_rows.append({
            "modality": "speech",
            "task": "speech_transcription",
            "id": row.get("id", index),
            "split_role": "development_selection" if index >= len(speech_rows) - args.speech_dev_tail else "train",
            "media_sha256": sha256_file(path),
            "source_ids": [f"legacy_speech_row:{row.get('source')}:{row.get('id', index)}"],
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "rows": len(output_rows),
        "image_rows": len(image_rows),
        "speech_rows": len(speech_rows),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
