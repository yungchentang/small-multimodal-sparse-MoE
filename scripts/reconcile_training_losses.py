#!/usr/bin/env python3
"""Reconcile legacy OLMoE training-loss fields without mutating raw logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_EXPERIMENTS = (
    "E3_final_multimodal_top2",
    "E4_no_aux_load_balance_ablation",
    "E5_capacity_1p25_ablation",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(sum(items) / len(items)) if items else float("nan")


def summarize_rows(rows: List[Dict[str, Any]], window: int) -> Dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty rows")
    width = max(1, min(int(window), len(rows)))
    first = rows[:width]
    last = rows[-width:]
    modalities = sorted({str(row["modality"]) for row in rows})

    def block(items: List[Dict[str, Any]]) -> Dict[str, float]:
        return {
            "lm_ce_mean": mean(float(row["lm_ce_loss"]) for row in items),
            "hf_base_loss_mean": mean(float(row["hf_base_loss"]) for row in items),
            "weighted_aux_mean": mean(float(row["router_aux_loss_weighted"]) for row in items),
        }

    by_modality: Dict[str, Any] = {}
    for modality in modalities:
        subset = [row for row in rows if row["modality"] == modality]
        mod_width = max(1, min(width, len(subset)))
        by_modality[modality] = {
            "rows": len(subset),
            "first_window": block(subset[:mod_width]),
            "last_window": block(subset[-mod_width:]),
        }

    first_mean = mean(float(row["lm_ce_loss"]) for row in first)
    last_mean = mean(float(row["lm_ce_loss"]) for row in last)
    return {
        "rows": len(rows),
        "window_rows": width,
        "endpoint_first": rows[0],
        "endpoint_last": rows[-1],
        "first_window": block(first),
        "last_window": block(last),
        "lm_ce_last_minus_first_window": float(last_mean - first_mean),
        "lm_ce_nondivergent_window": bool(math.isfinite(last_mean) and last_mean <= first_mean),
        "by_modality": by_modality,
    }


def reconcile(path: Path, legacy_effective_aux_coef: float) -> List[Dict[str, Any]]:
    output = []
    for raw in read_jsonl(path):
        requested_aux_coef = float(raw.get("aux_coef", 0.0) or 0.0)
        aux_raw = float(raw.get("router_aux_loss_raw", raw.get("aux_loss", 0.0)) or 0.0)
        if raw.get("lm_ce_loss") is not None:
            effective_aux_coef = requested_aux_coef
            weighted_aux = float(raw.get("router_aux_loss_weighted", effective_aux_coef * aux_raw))
            lm_ce = float(raw["lm_ce_loss"])
            hf_base = float(raw.get("hf_reported_loss", lm_ce + weighted_aux))
            source = "explicit_logged_lm_ce"
        else:
            # Historical setter changed config/layer gates but not the CausalLM
            # instance attributes copied at model initialization.
            effective_aux_coef = float(legacy_effective_aux_coef)
            weighted_aux = effective_aux_coef * aux_raw
            hf_base = float(raw["ce_loss"])
            lm_ce = hf_base - weighted_aux
            source = "reconstructed_from_legacy_stale_model_level_aux"
        values = [lm_ce, hf_base, aux_raw, weighted_aux]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite loss at {path}:{raw.get('step')}")
        output.append({
            "step": int(raw["step"]),
            "modality": str(raw["modality"]),
            "lm_ce_loss": lm_ce,
            "hf_base_loss": hf_base,
            "router_aux_loss_raw": aux_raw,
            "router_aux_loss_weighted": weighted_aux,
            "requested_aux_coef": requested_aux_coef,
            "effective_aux_coef": effective_aux_coef,
            "total_training_loss": float(raw.get("loss", hf_base)),
            "source_semantics": source,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument(
        "--legacy-effective-aux-coef",
        type=float,
        default=0.01,
        help="Effective CausalLM aux coefficient before the model-level runtime setter fix.",
    )
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "review_repair"
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "schema_version": 1,
        "legacy_field_semantics": (
            "Historical ce_loss stored Hugging Face base loss, which includes "
            "aux_coef * raw router aux loss. Raw JSONL files are unchanged."
        ),
        "formula": "lm_ce_loss = legacy_ce_loss - effective_model_level_aux_coef * aux_loss",
        "legacy_requested_vs_effective_aux": {
            "effective_model_level_aux_coef": float(args.legacy_effective_aux_coef),
            "cause": (
                "Historical runtime setter changed config.router_aux_loss_coef but not "
                "OlmoeForCausalLM.router_aux_loss_coef copied at model initialization."
            ),
            "implication": (
                "Historical E4 requested aux=0 but still used the native model-level "
                "coefficient; it is not a valid no-aux ablation."
            ),
        },
        "experiments": {},
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "inputs": {},
        },
    }
    for experiment in experiments:
        log_path = run_root / experiment / "train_metrics.jsonl"
        rows = reconcile(log_path, args.legacy_effective_aux_coef)
        for row in rows:
            all_rows.append({"experiment": experiment, **row})
        summary["experiments"][experiment] = summarize_rows(rows, args.window)
        summary["provenance"]["inputs"][experiment] = {
            "path": str(log_path.resolve()),
            "sha256": sha256_file(log_path),
            "rows": len(rows),
        }

    csv_path = output_dir / "loss_reconciliation.csv"
    fields = list(all_rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    summary["provenance"]["curve_csv"] = {
        "path": str(csv_path.resolve()),
        "sha256": sha256_file(csv_path),
        "rows": len(all_rows),
    }
    output_path = output_dir / "loss_reconciliation.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "experiments": experiments,
        "rows": len(all_rows),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
