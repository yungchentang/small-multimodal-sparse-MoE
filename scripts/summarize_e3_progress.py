"""Summarize E3 train_metrics.jsonl without dumping expert histograms."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def avg(rows: List[Dict[str, Any]], key: str) -> float | None:
    vals = [as_float(row.get(key)) for row in rows]
    vals = [val for val in vals if val is not None]
    return mean(vals) if vals else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_metrics", help="Path to E3 train_metrics.jsonl")
    parser.add_argument("--window", type=int, default=120)
    args = parser.parse_args()
    path = Path(args.train_metrics)
    rows: List[Dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        print(json.dumps({"path": str(path), "rows": 0}, sort_keys=True))
        return
    recent = rows[-args.window:]
    by_modality: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in recent:
        by_modality[str(row.get("modality", "unknown"))].append(row)
    summary = {
        "path": str(path),
        "rows": len(rows),
        "last_step": rows[-1].get("step"),
        "last_modality": rows[-1].get("modality"),
        "overall_recent": {
            "ce_loss": avg(recent, "ce_loss"),
            "loss": avg(recent, "loss"),
            "conditional_ranking_accuracy": avg(recent, "conditional_ranking_accuracy"),
            "capacity_overflow_ratio_mean": avg(recent, "capacity_overflow_ratio_mean"),
            "inactive_expert_ratio_mean": avg(recent, "inactive_expert_ratio_mean"),
        },
        "by_modality_recent": {
            mode: {
                "count": len(items),
                "ce_loss": avg(items, "ce_loss"),
                "conditional_ranking_accuracy": avg(items, "conditional_ranking_accuracy"),
                "capacity_overflow_ratio_mean": avg(items, "capacity_overflow_ratio_mean"),
                "inactive_expert_ratio_mean": avg(items, "inactive_expert_ratio_mean"),
            }
            for mode, items in sorted(by_modality.items())
        },
    }
    print(json.dumps(summary, sort_keys=True))
    print("\n| split | n | ce | rank_acc | overflow | inactive |")
    print("|---|---:|---:|---:|---:|---:|")
    print(
        "| recent | {n} | {ce} | {acc} | {overflow} | {inactive} |".format(
            n=len(recent),
            ce=fmt(summary["overall_recent"]["ce_loss"]),
            acc=fmt(summary["overall_recent"]["conditional_ranking_accuracy"]),
            overflow=fmt(summary["overall_recent"]["capacity_overflow_ratio_mean"]),
            inactive=fmt(summary["overall_recent"]["inactive_expert_ratio_mean"]),
        )
    )
    for mode, stats in summary["by_modality_recent"].items():
        print(
            "| {mode} | {n} | {ce} | {acc} | {overflow} | {inactive} |".format(
                mode=mode,
                n=stats["count"],
                ce=fmt(stats["ce_loss"]),
                acc=fmt(stats["conditional_ranking_accuracy"]),
                overflow=fmt(stats["capacity_overflow_ratio_mean"]),
                inactive=fmt(stats["inactive_expert_ratio_mean"]),
            )
        )


if __name__ == "__main__":
    main()
