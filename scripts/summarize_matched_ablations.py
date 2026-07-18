#!/usr/bin/env python
"""Validate and summarize three seed roots from run_matched_ablations.py."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_matched_ablations as protocol


ARM_NAMES = [name for name, _, _ in protocol.ARM_SPECS]
PAIRED_COMPARISONS = (
    ("E4_minus_E3", "E4_noaux_cap7_300", "E3_aux_cap7_300"),
    ("E5_minus_E3", "E5_aux_cap1p25_300", "E3_aux_cap7_300"),
)
HARD_BALANCE_FIELDS = (
    "accepted_hard_assignment_entropy",
    "accepted_effective_experts",
    "accepted_active_experts",
    "accepted_inactive_expert_ratio",
    "accepted_load_cv",
    "accepted_load_gini",
    "accepted_max_to_mean_load",
)
ROUTING_COUNT_FIELDS = (
    "routing_attempted_assignments_total",
    "routing_accepted_assignments_total",
    "routing_dropped_assignments_total",
)
ROUTING_MEAN_FIELDS = (
    "routing_drop_ratio_total",
    "capacity_overflow_ratio_mean",
    "inactive_expert_ratio_mean",
    "dynamic_expert_bias_overflow_proxy",
    "dynamic_expert_bias_inactive_proxy",
    *HARD_BALANCE_FIELDS,
)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def protocol_fingerprint(pre: Mapping[str, Any]) -> str:
    arms = []
    for arm in pre["arms"]:
        normalized = dict(arm)
        normalized.pop("seed", None)
        matched_args = dict(normalized.get("matched_args", {}))
        matched_args.pop("seed", None)
        normalized["matched_args"] = matched_args
        arms.append(normalized)
    payload = {
        "protocol": pre.get("protocol"),
        "hashes": pre.get("hashes"),
        "allowed_differences": pre.get("allowed_differences"),
        "expected_data_order_sha256": pre.get("expected_data_order", {}).get("sha256"),
        "arms": arms,
    }
    return protocol.canonical_sha256(payload)


def validate_seed_root(root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pre = read_json(root / protocol.PRE_PROTOCOL)
    post = read_json(root / protocol.POST_PROTOCOL)
    errors = protocol.verify_matched_arm_contracts(pre.get("arms", []))
    errors.extend(protocol.verify_matched_arm_contracts(post.get("observed_arms", [])))
    if post.get("pre_protocol_sha256") != protocol.canonical_sha256(pre):
        errors.append("post protocol does not match pre protocol hash")
    if post.get("passed") is not True:
        errors.append(f"post protocol failed: {post.get('errors')}")
    if sorted(post.get("completed_arms", [])) != sorted(ARM_NAMES):
        errors.append("post protocol does not list exactly three completed arms")
    for arm in post.get("observed_arms", []):
        if arm.get("observed_steps") != protocol.STEPS:
            errors.append(f"{arm.get('experiment_id')}: observed_steps != {protocol.STEPS}")
        if arm.get("successful_optimizer_steps") != protocol.STEPS:
            errors.append(f"{arm.get('experiment_id')}: successful_optimizer_steps != {protocol.STEPS}")
    for name in ARM_NAMES:
        if not (root / name / "metrics.json").exists():
            errors.append(f"missing metrics for {name}")
    if errors:
        raise ValueError(f"Invalid matched protocol root {root}: " + "; ".join(errors))
    return pre, post


def step_values(rows: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    return [float(row[key]) for row in rows if key in row and numeric(row[key])]


def extract_metrics(artifact: Mapping[str, Any]) -> Dict[str, float]:
    rows = artifact.get("steps")
    if not isinstance(rows, list) or len(rows) != protocol.STEPS:
        raise ValueError(f"Expected {protocol.STEPS} training rows")
    text_eval = artifact.get("text_eval") if isinstance(artifact.get("text_eval"), dict) else {}
    retrieval = artifact.get("retrieval_eval") if isinstance(artifact.get("retrieval_eval"), dict) else {}
    result: Dict[str, float] = {}
    if numeric(text_eval.get("perplexity")):
        result["text_perplexity"] = float(text_eval["perplexity"])
    if numeric(text_eval.get("loss")):
        result["text_lm_ce"] = float(text_eval["loss"])

    lm_ce = step_values(rows, "lm_ce_loss")
    if lm_ce:
        result["train_lm_ce_mean"] = statistics.mean(lm_ce)
        result["train_lm_ce_final"] = lm_ce[-1]

    for key, value in sorted(retrieval.items()):
        if key.startswith("conditional_") and numeric(value):
            result[key] = float(value)

    for key in ROUTING_COUNT_FIELDS:
        values = step_values(rows, key)
        if values:
            short = key.removeprefix("routing_").removesuffix("_total")
            result[f"routing_{short}_sum"] = sum(values)
            result[f"routing_{short}_mean_per_step"] = statistics.mean(values)
    for key in ROUTING_MEAN_FIELDS:
        values = step_values(rows, key)
        if values:
            result[f"{key}_mean_over_steps"] = statistics.mean(values)

    required = {
        "text_perplexity",
        "text_lm_ce",
        "train_lm_ce_mean",
        "routing_attempted_assignments_sum",
        "routing_accepted_assignments_sum",
        "routing_dropped_assignments_sum",
        "capacity_overflow_ratio_mean_mean_over_steps",
        "inactive_expert_ratio_mean_mean_over_steps",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"Metrics artifact lacks required matched-ablation metrics: {missing}")
    if not any(key.startswith("conditional_") for key in result):
        raise ValueError("Metrics artifact lacks numeric conditional metrics")
    if not any(key.startswith("accepted_") for key in result):
        raise ValueError("Metrics artifact lacks hard balance metrics")
    return result


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return lines


def summarize(seed_roots: Sequence[Path], output_dir: Path) -> None:
    if len(seed_roots) != 3:
        raise ValueError("Exactly three --seed-roots are required")
    validated = [(root, *validate_seed_root(root)) for root in seed_roots]
    seeds = [int(pre["seed"]) for _, pre, _ in validated]
    if len(set(seeds)) != 3:
        raise ValueError(f"Seed roots must contain three distinct seeds; got {seeds}")
    fingerprints = {protocol_fingerprint(pre) for _, pre, _ in validated}
    if len(fingerprints) != 1:
        raise ValueError("Seed roots do not share the same code/config/data/protocol fingerprint")

    run_metrics: Dict[Tuple[int, str], Dict[str, float]] = {}
    for root, pre, _ in validated:
        seed = int(pre["seed"])
        for arm in ARM_NAMES:
            run_metrics[(seed, arm)] = extract_metrics(read_json(root / arm / "metrics.json"))

    metric_sets = {tuple(sorted(values)) for values in run_metrics.values()}
    if len(metric_sets) != 1:
        raise ValueError("Metric fields differ across seed/arm artifacts")
    metric_names = list(next(iter(metric_sets)))

    aggregate: List[Dict[str, Any]] = []
    for arm in ARM_NAMES:
        for metric in metric_names:
            values = [run_metrics[(seed, arm)][metric] for seed in seeds]
            avg, std = mean_std(values)
            aggregate.append({"arm": arm, "metric": metric, "n": len(values), "mean": avg, "std": std})

    paired: List[Dict[str, Any]] = []
    paired_summary: List[Dict[str, Any]] = []
    for comparison, minuend, subtrahend in PAIRED_COMPARISONS:
        for metric in metric_names:
            deltas = []
            for seed in seeds:
                delta = run_metrics[(seed, minuend)][metric] - run_metrics[(seed, subtrahend)][metric]
                deltas.append(delta)
                paired.append({"comparison": comparison, "metric": metric, "seed": seed, "delta": delta})
            avg, std = mean_std(deltas)
            paired_summary.append(
                {"comparison": comparison, "metric": metric, "n": len(deltas), "mean_delta": avg, "std_delta": std}
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "matched_ablation_summary.csv"
    csv_rows: List[Dict[str, Any]] = []
    for row in aggregate:
        csv_rows.append({"record_type": "aggregate", **row})
    for row in paired:
        csv_rows.append({"record_type": "paired_seed_delta", **row, "value": row["delta"]})
    for row in paired_summary:
        csv_rows.append(
            {
                "record_type": "paired_delta_summary",
                **row,
                "mean": row["mean_delta"],
                "std": row["std_delta"],
            }
        )
    fields = [
        "record_type",
        "arm",
        "comparison",
        "metric",
        "seed",
        "value",
        "n",
        "mean",
        "std",
        "delta",
        "mean_delta",
        "std_delta",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    json_payload = {
        "schema_version": 1,
        "seed_roots": [str(root.resolve()) for root in seed_roots],
        "seeds": seeds,
        "protocol_fingerprint": next(iter(fingerprints)),
        "aggregate": aggregate,
        "paired_seed_deltas": paired,
        "paired_delta_summary": paired_summary,
    }
    json_path = output_dir / "matched_ablation_summary.json"
    json_path.write_text(json.dumps(json_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        "# Matched Early-Training Ablations",
        "",
        f"Validated seeds: {', '.join(str(seed) for seed in seeds)}. Values are mean and sample standard deviation across seeds.",
        "",
        "## Arm Metrics",
        "",
    ]
    md_lines.extend(
        markdown_table(
            ["Arm", "Metric", "Mean", "Std", "N"],
            ([row["arm"], row["metric"], f"{row['mean']:.8g}", f"{row['std']:.8g}", row["n"]] for row in aggregate),
        )
    )
    md_lines.extend(["", "## Paired Seed Deltas", ""])
    md_lines.extend(
        markdown_table(
            ["Comparison", "Metric", "Mean Delta", "Std Delta", "N"],
            (
                [row["comparison"], row["metric"], f"{row['mean_delta']:.8g}", f"{row['std_delta']:.8g}", row["n"]]
                for row in paired_summary
            ),
        )
    )
    md_path = output_dir / "matched_ablation_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-roots", nargs=3, required=True, metavar=("SEED1", "SEED2", "SEED3"))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize([Path(value) for value in args.seed_roots], Path(args.output_dir))


if __name__ == "__main__":
    main()
