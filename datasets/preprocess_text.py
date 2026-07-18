"""Create a small JSONL text subset for smoke or full experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SAMPLES = [
    {"task": "text", "text": "The sparse MoE model routes tokens to two experts."},
    {"task": "code", "text": "def add(a, b): return a + b"},
    {"task": "reasoning", "text": "If A implies B and A is true, B is true."},
    {"task": "math", "text": "2 + 3 = 5"},
    {"task": "education", "text": "SAT analogy: up is to down as hot is to cold."},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/text_subset.jsonl")
    parser.add_argument("--max-samples", type=int, default=32)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(args.max_samples):
            row = dict(SAMPLES[i % len(SAMPLES)], id=i)
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
