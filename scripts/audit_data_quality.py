"""Audit real subset data quality before launching GPU training.

This intentionally checks the data directory directly. It is a preflight guard
for issues that model-level requirement audits can otherwise discover too late.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                row["_line_no"] = line_no
                rows.append(row)
    return rows


def caption_ascii_letter_stats(caption: str) -> Tuple[int, int, int, float]:
    letters = [ch for ch in caption if ch.isalpha()]
    ascii_letters = [ch for ch in letters if ch.isascii()]
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", caption)
    ratio = len(ascii_letters) / len(letters) if letters else 0.0
    return len(ascii_letters), len(letters), len(words), ratio


def word_count(text: str) -> int:
    return len(str(text).split())


def percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def first_examples(rows: Iterable[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(row)
        if len(out) >= limit:
            break
    return out


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    image_rows = load_jsonl(data_dir / "image_captions.jsonl")
    image_bad: List[Dict[str, Any]] = []
    for row in image_rows:
        caption = str(row.get("caption", ""))
        ascii_letters, letters, ascii_words, ratio = caption_ascii_letter_stats(caption)
        if ascii_letters < args.caption_min_letters or ratio < args.caption_min_ascii_ratio:
            image_bad.append(
                {
                    "line": row.get("_line_no"),
                    "id": row.get("id"),
                    "source": row.get("source"),
                    "caption": caption[:160],
                    "ascii_letters": ascii_letters,
                    "letters": letters,
                    "ascii_words": ascii_words,
                    "ascii_letter_ratio": round(ratio, 4),
                }
            )

    speech_rows = load_jsonl(data_dir / "speech_transcripts.jsonl")
    speech_word_counts = [word_count(row.get("transcript", row.get("text", ""))) for row in speech_rows]
    missing_duration: List[Dict[str, Any]] = []
    duration_bad: List[Dict[str, Any]] = []
    transcript_bad: List[Dict[str, Any]] = []
    missing_filter_meta: List[Dict[str, Any]] = []
    for row, words in zip(speech_rows, speech_word_counts):
        preprocess = row.get("preprocess") if isinstance(row.get("preprocess"), dict) else {}
        source_duration = preprocess.get("source_duration_seconds")
        if source_duration is None:
            missing_duration.append({"line": row.get("_line_no"), "id": row.get("id"), "preprocess_keys": sorted(preprocess)})
        if "max_source_audio_seconds" not in preprocess or "max_transcript_words" not in preprocess:
            missing_filter_meta.append({"line": row.get("_line_no"), "id": row.get("id"), "preprocess_keys": sorted(preprocess)})
        try:
            duration_f = float(source_duration)
        except (TypeError, ValueError):
            duration_f = None
        if duration_f is not None and args.max_source_audio_seconds > 0 and duration_f > args.max_source_audio_seconds:
            duration_bad.append({"line": row.get("_line_no"), "id": row.get("id"), "duration": duration_f})
        if args.max_transcript_words > 0 and words > args.max_transcript_words:
            transcript_bad.append(
                {
                    "line": row.get("_line_no"),
                    "id": row.get("id"),
                    "words": words,
                    "transcript": str(row.get("transcript", row.get("text", "")))[:160],
                }
            )

    checks = [
        {"name": "image_count", "passed": len(image_rows) >= args.min_image_rows, "observed": len(image_rows), "expected": args.min_image_rows},
        {"name": "image_ascii_english", "passed": len(image_bad) == 0, "failures": len(image_bad), "examples": first_examples(image_bad)},
        {"name": "speech_count", "passed": len(speech_rows) >= args.min_speech_rows, "observed": len(speech_rows), "expected": args.min_speech_rows},
        {"name": "speech_source_duration_metadata", "passed": len(missing_duration) == 0, "failures": len(missing_duration), "examples": first_examples(missing_duration)},
        {"name": "speech_filter_metadata", "passed": len(missing_filter_meta) == 0, "failures": len(missing_filter_meta), "examples": first_examples(missing_filter_meta)},
        {"name": "speech_duration_cap", "passed": len(duration_bad) == 0, "failures": len(duration_bad), "examples": first_examples(duration_bad)},
        {"name": "speech_transcript_word_cap", "passed": len(transcript_bad) == 0, "failures": len(transcript_bad), "examples": first_examples(transcript_bad)},
    ]
    stats = {
        "data_dir": str(data_dir),
        "manifest_exists": manifest_path.exists(),
        "manifest_sources": manifest.get("sources", {}),
        "speech_word_median": float(median(speech_word_counts)) if speech_word_counts else 0.0,
        "speech_word_p95": percentile(speech_word_counts, 0.95),
    }
    return {"passed": all(check["passed"] for check in checks), "checks": checks, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--min-image-rows", type=int, default=5250)
    parser.add_argument("--min-speech-rows", type=int, default=5250)
    parser.add_argument("--caption-min-ascii-ratio", type=float, default=0.85)
    parser.add_argument("--caption-min-letters", type=int, default=8)
    parser.add_argument("--max-source-audio-seconds", type=float, default=6.0)
    parser.add_argument("--max-transcript-words", type=int, default=18)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = audit(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
