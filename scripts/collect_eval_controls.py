"""Collect conditional prefix-control evaluation metrics into CSV/Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        img = float(metrics.get("conditional_image_to_text_r_at_1", metrics.get("image_to_text_r_at_1", 0.0)) or 0.0)
        sp = float(metrics.get("conditional_speech_to_text_r_at_1", metrics.get("speech_to_text_r_at_1", 0.0)) or 0.0)
        chance = float(metrics.get("conditional_image_chance_r_at_1", metrics.get("image_chance_r_at_1", 0.0)) or 0.0)
        image_ci_low = metrics.get("conditional_image_to_text_r_at_1_bootstrap_ci_low", metrics.get("image_to_text_r_at_1_bootstrap_ci_low"))
        image_ci_high = metrics.get("conditional_image_to_text_r_at_1_bootstrap_ci_high", metrics.get("image_to_text_r_at_1_bootstrap_ci_high"))
        speech_ci_low = metrics.get("conditional_speech_to_text_r_at_1_bootstrap_ci_low", metrics.get("speech_to_text_r_at_1_bootstrap_ci_low"))
        speech_ci_high = metrics.get("conditional_speech_to_text_r_at_1_bootstrap_ci_high", metrics.get("speech_to_text_r_at_1_bootstrap_ci_high"))
        rows.append({
            "label": path.parent.name,
            "metrics_path": str(path),
            "checkpoint": metrics.get("checkpoint", ""),
            "run_output_dir": metrics.get("run_output_dir", ""),
            "prefix_control": metrics.get("prefix_control", "unknown"),
            "negative_mode": metrics.get("negative_mode", "unknown"),
            "eval_path": metrics.get("eval_path", metrics.get("conditional_eval_path", "unknown")),
            "eval_split_name": metrics.get("eval_split_name", "unknown"),
            "query_offset": metrics.get("query_offset", ""),
            "candidate_offset": metrics.get("candidate_offset", ""),
            "image_query_start": metrics.get("image_query_start", ""),
            "image_query_end_exclusive": metrics.get("image_query_end_exclusive", ""),
            "speech_query_start": metrics.get("speech_query_start", ""),
            "speech_query_end_exclusive": metrics.get("speech_query_end_exclusive", ""),
            "candidates": int(metrics.get("conditional_candidates_per_query", metrics.get("candidate_count", 0)) or 0),
            "image_eval_count": int(metrics.get("conditional_image_eval_count", metrics.get("image_eval_count", 0)) or 0),
            "speech_eval_count": int(metrics.get("conditional_speech_eval_count", metrics.get("speech_eval_count", 0)) or 0),
            "image_r1": img,
            "speech_r1": sp,
            "chance_r1": chance,
            "image_margin": img - chance,
            "speech_margin": sp - chance,
            "image_r1_ci_low": image_ci_low if image_ci_low is not None else "",
            "image_r1_ci_high": image_ci_high if image_ci_high is not None else "",
            "speech_r1_ci_low": speech_ci_low if speech_ci_low is not None else "",
            "speech_r1_ci_high": speech_ci_high if speech_ci_high is not None else "",
            "image_r5": float(metrics.get("conditional_image_to_text_r_at_5", metrics.get("image_to_text_r_at_5", 0.0)) or 0.0),
            "speech_r5": float(metrics.get("conditional_speech_to_text_r_at_5", metrics.get("speech_to_text_r_at_5", 0.0)) or 0.0),
        })
    return rows


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = ["# Prefix-Control Conditional Evaluation", "", "| label | split | path | query offset | candidate offset | prefix | negatives | candidates | img R@1 | img CI | speech R@1 | speech CI | chance | img margin | speech margin |", "|---|---|---|---:|---:|---|---|---:|---:|---|---:|---|---:|---:|---:|"]
    for row in rows:
        img_ci = ""
        sp_ci = ""
        if row.get("image_r1_ci_low") != "" and row.get("image_r1_ci_high") != "":
            img_ci = f"[{float(row['image_r1_ci_low']):.3f}, {float(row['image_r1_ci_high']):.3f}]"
        if row.get("speech_r1_ci_low") != "" and row.get("speech_r1_ci_high") != "":
            sp_ci = f"[{float(row['speech_r1_ci_low']):.3f}, {float(row['speech_r1_ci_high']):.3f}]"
        lines.append(
            f"| {row['label']} | {row['eval_split_name']} | {row['eval_path']} | {row['query_offset']} | {row['candidate_offset']} | {row['prefix_control']} | {row['negative_mode']} | {row['candidates']} | "
            f"{row['image_r1']:.4f} | {img_ci} | {row['speech_r1']:.4f} | {sp_ci} | {row['chance_r1']:.4f} | "
            f"{row['image_margin']:.4f} | {row['speech_margin']:.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    rows = load_rows(Path(args.root))
    if not rows:
        raise RuntimeError(f"No metrics found under {args.root}")
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(Path(args.output_md), rows)
    print(json.dumps({"rows": len(rows), "csv": str(out_csv), "markdown": str(args.output_md)}, sort_keys=True))


if __name__ == "__main__":
    main()
