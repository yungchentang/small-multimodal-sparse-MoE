"""Create a small JSONL image-caption manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CAPTIONS = ["a red square on a plain background", "a blue circle near the center"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/image_subset.jsonl")
    parser.add_argument("--max-samples", type=int, default=32)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(args.max_samples):
            row = {"id": i, "image_path": f"images/synthetic_{i:04d}.png", "caption": CAPTIONS[i % len(CAPTIONS)]}
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
