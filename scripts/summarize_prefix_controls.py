"""Summarize prefix-control and encoder/no-prefix conditional eval results.

The collector gives one row per job. This script groups rows by candidate count,
split, offset, and negative mode, then reports whether the real shared-prefix path
beats chance and the non-multimodal controls. It is intentionally conservative:
missing controls are listed instead of silently treated as passing evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CONTROL_ORDER = ["zero", "random", "shuffled", "no_prefix_lm", "encoder_baseline"]


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def row_control(row: Dict[str, str]) -> str:
    path = str(row.get("eval_path", ""))
    prefix = str(row.get("prefix_control", ""))
    if "no_prefix" in path:
        return "no_prefix_lm"
    if "encoder" in path:
        return "encoder_baseline"
    return prefix or "unknown"


def group_key(row: Dict[str, str]) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("eval_split_name", "unknown")),
        str(row.get("query_offset", "")),
        str(row.get("candidate_offset", "")),
        str(row.get("candidates", "")),
        str(row.get("negative_mode", "unknown")),
    )


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_group(key: Tuple[str, str, str, str, str], rows: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    rows = list(rows)
    controls: Dict[str, Dict[str, str]] = {}
    for row in rows:
        controls[row_control(row)] = row
    real = controls.get("real")
    chance = as_float(real.get("chance_r1") if real else rows[0].get("chance_r1"), 0.0) if rows else 0.0

    missing_controls = [name for name in CONTROL_ORDER if name not in controls]
    controls_complete = not missing_controls
    out: Dict[str, Any] = {
        "eval_split_name": key[0],
        "query_offset": key[1],
        "candidate_offset": key[2],
        "candidates": int(as_float(key[3], 0.0)),
        "negative_mode": key[4],
        "chance_r1": chance,
        "available_controls": sorted(controls.keys()),
        "missing_controls": missing_controls,
        "controls_complete": controls_complete,
        "required_controls": CONTROL_ORDER,
        "has_real_prefix": real is not None,
    }
    if real is None:
        out.update({
            "image_real_r1": None,
            "speech_real_r1": None,
            "passes_image_prefix_sensitivity": False,
            "passes_speech_prefix_sensitivity": False,
            "reason": "missing real shared-prefix row",
        })
        return out

    img_real = as_float(real.get("image_r1"))
    sp_real = as_float(real.get("speech_r1"))
    out.update({
        "image_real_r1": img_real,
        "speech_real_r1": sp_real,
        "image_real_margin_vs_chance": img_real - chance,
        "speech_real_margin_vs_chance": sp_real - chance,
    })
    image_deltas: Dict[str, float] = {}
    speech_deltas: Dict[str, float] = {}
    for name in CONTROL_ORDER:
        row = controls.get(name)
        if not row:
            continue
        image_deltas[name] = img_real - as_float(row.get("image_r1"))
        speech_deltas[name] = sp_real - as_float(row.get("speech_r1"))
    out["image_delta_vs_controls"] = image_deltas
    out["speech_delta_vs_controls"] = speech_deltas
    img_min_delta = min(image_deltas.values()) if image_deltas else None
    sp_min_delta = min(speech_deltas.values()) if speech_deltas else None
    out["passes_image_prefix_sensitivity"] = bool(controls_complete and img_real > chance and img_min_delta is not None and img_min_delta > 0.0)
    out["passes_speech_prefix_sensitivity"] = bool(controls_complete and sp_real > chance and sp_min_delta is not None and sp_min_delta > 0.0)
    reasons: List[str] = []
    if not controls_complete:
        reasons.append("missing required controls: " + ", ".join(missing_controls))
    if img_real <= chance:
        reasons.append("image real prefix is not above chance")
    if sp_real <= chance:
        reasons.append("speech real prefix is not above chance")
    if img_min_delta is not None and img_min_delta <= 0.0:
        reasons.append("image real prefix does not beat every control")
    if sp_min_delta is not None and sp_min_delta <= 0.0:
        reasons.append("speech real prefix does not beat every control")
    out["reason"] = "; ".join(reasons) if reasons else "complete controls and real prefix beats chance/all controls"
    return out


def write_markdown(path: Path, summaries: List[Dict[str, Any]]) -> None:
    lines = [
        "# Prefix Sensitivity Summary",
        "",
        "| split | query offset | candidate offset | candidates | negatives | controls complete | img real | img chance delta | img min control delta | img pass | speech real | speech chance delta | speech min control delta | speech pass | missing controls |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        img_deltas = list((row.get("image_delta_vs_controls") or {}).values())
        sp_deltas = list((row.get("speech_delta_vs_controls") or {}).values())
        img_min = min(img_deltas) if img_deltas else None
        sp_min = min(sp_deltas) if sp_deltas else None
        lines.append(
            "| {split} | {offset} | {cand_offset} | {cands} | {neg} | {complete} | {img:.4f} | {img_ch:.4f} | {img_min} | {img_pass} | {sp:.4f} | {sp_ch:.4f} | {sp_min} | {sp_pass} | {missing} |".format(
                split=row.get("eval_split_name"),
                offset=row.get("query_offset"),
                cand_offset=row.get("candidate_offset"),
                cands=row.get("candidates"),
                neg=row.get("negative_mode"),
                complete="yes" if row.get("controls_complete") else "no",
                img=float(row.get("image_real_r1") or 0.0),
                img_ch=float(row.get("image_real_margin_vs_chance") or 0.0),
                img_min="" if img_min is None else f"{img_min:.4f}",
                img_pass="yes" if row.get("passes_image_prefix_sensitivity") else "no",
                sp=float(row.get("speech_real_r1") or 0.0),
                sp_ch=float(row.get("speech_real_margin_vs_chance") or 0.0),
                sp_min="" if sp_min is None else f"{sp_min:.4f}",
                sp_pass="yes" if row.get("passes_speech_prefix_sensitivity") else "no",
                missing=", ".join(row.get("missing_controls") or []),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.input_csv))
    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    summaries = [summarize_group(key, groups[key]) for key in sorted(groups)]
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"groups": summaries, "rows": len(rows)}, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(Path(args.output_md), summaries)
    print(json.dumps({"groups": len(summaries), "rows": len(rows), "output_json": str(output_json), "output_md": str(args.output_md)}, sort_keys=True))


if __name__ == "__main__":
    main()
