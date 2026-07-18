"""Compare Top-8 teacher to Top-2 calibration/distillation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


EXP_DIRS = {
    "E0": "E0_top8_teacher",
    "E1": "E1_hard_top2",
    "E2": "E2_gamma_calibrated_top2",
    "E2_CE_only": "E2_CE_only",
    "E2D_logits_kl": "E2D_logits_kl",
}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def load_run(root: Path) -> List[Dict[str, Any]]:
    manifest = read_json(root / "manifest.json")
    rows: List[Dict[str, Any]] = []
    for exp_name, dirname in EXP_DIRS.items():
        metrics = read_json(root / dirname / "metrics.json")
        if not metrics:
            summary = read_json(root / "summary.json")
            metrics = summary.get(exp_name, {}) if isinstance(summary.get(exp_name), dict) else {}
        if not metrics:
            continue
        row = {
            "run": root.name,
            "root": str(root),
            "job": manifest.get("runai_job_name", ""),
            "experiment": exp_name,
            "ppl": metrics.get("perplexity"),
            "next_token_accuracy": metrics.get("next_token_accuracy"),
            "teacher_student_kl": metrics.get("teacher_student_kl"),
            "router_kl": metrics.get("router_kl"),
            "teacher_topk_mass_on_student_topk": metrics.get("teacher_topk_mass_on_student_topk"),
            "inactive": metrics.get("inactive_expert_ratio_mean"),
            "overflow": metrics.get("capacity_overflow_ratio_mean"),
            "checkpoint": str(root / dirname / "checkpoint_final.pt") if (root / dirname / "checkpoint_final.pt").exists() else "",
        }
        rows.append(row)
    baseline = next((row for row in rows if row["experiment"] == "E1"), None)
    base_ppl = as_float(baseline.get("ppl")) if baseline else None
    base_kl = as_float(baseline.get("teacher_student_kl")) if baseline else None
    for row in rows:
        ppl = as_float(row.get("ppl"))
        kl = as_float(row.get("teacher_student_kl"))
        row["delta_ppl_vs_E1"] = (ppl - base_ppl) if ppl is not None and base_ppl is not None else ""
        row["delta_teacher_kl_vs_E1"] = (kl - base_kl) if kl is not None and base_kl is not None else ""
    return rows


def fmt(value: Any) -> str:
    num = as_float(value)
    return "" if num is None else f"{num:.4g}"


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Top-2 Distillation Comparison",
        "",
        "| run | experiment | PPL | acc | teacher KL | router KL | dPPL vs E1 | dKL vs E1 | checkpoint |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['experiment']} | {fmt(row.get('ppl'))} | {fmt(row.get('next_token_accuracy'))} | "
            f"{fmt(row.get('teacher_student_kl'))} | {fmt(row.get('router_kl'))} | "
            f"{fmt(row.get('delta_ppl_vs_E1'))} | {fmt(row.get('delta_teacher_kl_vs_E1'))} | {row.get('checkpoint','')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--output-dir", default="autoresearch/top2_distillation")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for root in args.roots:
        rows.extend(load_run(Path(root)))
    if not rows:
        raise RuntimeError("No distillation metrics found")
    fields = list(rows[0].keys())
    csv_path = out_dir / "distillation_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md_path = out_dir / "distillation_comparison.md"
    write_markdown(md_path, rows)
    print(json.dumps({"rows": len(rows), "csv": str(csv_path), "markdown": str(md_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
