"""Create a small JSONL speech-transcript manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TRANSCRIPTS = ["the speaker reads a short sentence", "sparse experts process audio prefix tokens"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/audio_subset.jsonl")
    parser.add_argument("--max-samples", type=int, default=32)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(args.max_samples):
            row = {"id": i, "audio_path": f"audio/synthetic_{i:04d}.wav", "transcript": TRANSCRIPTS[i % len(TRANSCRIPTS)], "sample_rate": 16000}
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
