#!/usr/bin/env python3
"""Build the seed-42 E4/E5 continuation-sensitivity summary from raw artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping


METRICS = (
    "perplexity",
    "next_token_accuracy",
    "capacity_overflow_ratio_mean",
    "inactive_expert_ratio_mean",
)


class SummaryError(ValueError):
    """Raised when continuation artifacts are incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise SummaryError(f"{label} cannot be a symlink: {path}")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SummaryError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise SummaryError(f"{label} is not a regular file: {resolved}")
    return resolved


def read_json(path: Path, label: str) -> Dict[str, Any]:
    resolved = resolve_file(path, label)
    try:
        payload = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                SummaryError(f"{label} contains non-finite token {token}")
            ),
        )
    except SummaryError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SummaryError(f"{label} root must be an object")
    return payload


def file_record(path: Path, label: str) -> Dict[str, Any]:
    resolved = resolve_file(path, label)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise SummaryError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise SummaryError(f"{label} must be finite")
    return result


def validate_text_eval(payload: Mapping[str, Any], label: str) -> Dict[str, float]:
    values = {metric: finite_number(payload.get(metric), f"{label}.{metric}") for metric in METRICS}
    if values["perplexity"] <= 0.0:
        raise SummaryError(f"{label}.perplexity must be positive")
    for metric in (
        "next_token_accuracy",
        "capacity_overflow_ratio_mean",
        "inactive_expert_ratio_mean",
    ):
        if not 0.0 <= values[metric] <= 1.0:
            raise SummaryError(f"{label}.{metric} is outside [0,1]")
    return values


def validate_continuation(
    role: str,
    metrics: Mapping[str, Any],
    checkpoint: Path,
    source_checkpoint_sha256: str,
) -> None:
    if metrics.get("schema_version") != 1 or metrics.get("artifact_type") != "matched_ablation_final":
        raise SummaryError(f"{role} is not a matched-final artifact")
    if str(metrics.get("experiment_role", "")).upper() != role:
        raise SummaryError(f"{role} experiment_role mismatch")
    if metrics.get("source_selected_checkpoint_sha256") != source_checkpoint_sha256:
        raise SummaryError(f"{role} source checkpoint mismatch")
    record = metrics.get("checkpoint")
    if not isinstance(record, Mapping):
        raise SummaryError(f"{role} checkpoint record is missing")
    recorded_path = resolve_file(Path(str(record.get("path", ""))), f"{role} recorded checkpoint")
    if recorded_path != checkpoint:
        raise SummaryError(f"{role} checkpoint path mismatch")
    if record.get("sha256") != sha256_file(checkpoint):
        raise SummaryError(f"{role} checkpoint SHA-256 mismatch")
    steps = metrics.get("steps")
    if not isinstance(steps, list) or len(steps) != 300:
        raise SummaryError(f"{role} must contain exactly 300 optimizer steps")
    if [row.get("step") for row in steps if isinstance(row, Mapping)] != list(range(1, 301)):
        raise SummaryError(f"{role} steps must be contiguous 1..300")
    optimizer_rows = [row for row in steps if row.get("optimizer_step") is True]
    frozen_text_rows = [
        row
        for row in steps
        if row.get("optimizer_step") is False
        and row.get("modality") == "text"
        and row.get("train_router_gates") is False
        and row.get("train_experts") is False
        and row.get("train_lm_head") is False
    ]
    if not optimizer_rows or len(optimizer_rows) + len(frozen_text_rows) != len(steps):
        raise SummaryError(f"{role} contains unexplained non-optimizer rows")
    if any(row.get("modality") not in {"image", "speech"} for row in optimizer_rows):
        raise SummaryError(f"{role} optimizer rows have an unexpected modality")
    if (
        metrics.get("training_iterations") != len(steps)
        or metrics.get("optimizer_step_count") != len(optimizer_rows)
        or metrics.get("frozen_text_row_count") != len(frozen_text_rows)
    ):
        raise SummaryError(f"{role} optimizer/frozen-text accounting mismatch")


def build_summary(args: argparse.Namespace) -> Dict[str, Any]:
    e3_metrics_path = resolve_file(args.e3_metrics, "E3 metrics")
    e3_metrics = read_json(e3_metrics_path, "E3 metrics")
    source_sha = str(e3_metrics.get("checkpoint_sha256", "")).lower()
    if len(source_sha) != 64 or any(char not in "0123456789abcdef" for char in source_sha):
        raise SummaryError("E3 checkpoint SHA-256 is invalid")

    text_paths = {
        "E3_selected": resolve_file(args.e3_text_eval, "E3 text evaluation"),
        "E4_no_aux": resolve_file(args.e4_text_eval, "E4 text evaluation"),
        "E5_low_capacity": resolve_file(args.e5_text_eval, "E5 text evaluation"),
    }
    text_values = {
        arm: validate_text_eval(read_json(path, f"{arm} text evaluation"), arm)
        for arm, path in text_paths.items()
    }

    continuation_inputs = {
        "E4": (
            resolve_file(args.e4_metrics, "E4 metrics"),
            resolve_file(args.e4_checkpoint, "E4 checkpoint"),
        ),
        "E5": (
            resolve_file(args.e5_metrics, "E5 metrics"),
            resolve_file(args.e5_checkpoint, "E5 checkpoint"),
        ),
    }
    continuation_payloads: Dict[str, Dict[str, Any]] = {}
    for role, (metrics_path, checkpoint_path) in continuation_inputs.items():
        payload = read_json(metrics_path, f"{role} metrics")
        validate_continuation(role, payload, checkpoint_path, source_sha)
        continuation_payloads[role] = payload

    artifacts = {
        "e3_metrics": file_record(e3_metrics_path, "E3 metrics"),
        "e4_metrics": file_record(continuation_inputs["E4"][0], "E4 metrics"),
        "e4_checkpoint": file_record(continuation_inputs["E4"][1], "E4 checkpoint"),
        "e4_text_eval": file_record(text_paths["E4_no_aux"], "E4 text evaluation"),
        "e5_metrics": file_record(continuation_inputs["E5"][0], "E5 metrics"),
        "e5_checkpoint": file_record(continuation_inputs["E5"][1], "E5 checkpoint"),
        "e5_text_eval": file_record(text_paths["E5_low_capacity"], "E5 text evaluation"),
    }
    aggregate = []
    for arm, metrics in text_values.items():
        for metric, value in metrics.items():
            aggregate.append(
                {"arm": arm, "metric": metric, "n": 1, "mean": value, "std": 0.0}
            )

    deltas = []
    summaries = []
    baseline = text_values["E3_selected"]
    for arm in ("E4_no_aux", "E5_low_capacity"):
        comparison = f"{arm}_minus_E3_selected"
        for metric in METRICS:
            delta = text_values[arm][metric] - baseline[metric]
            deltas.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "seed": args.seed,
                    "delta": delta,
                }
            )
            summaries.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "n": 1,
                    "mean_delta": delta,
                    "std_delta": 0.0,
                }
            )

    return {
        "schema_version": 2,
        "analysis_type": "single_seed_selected_checkpoint_continuations",
        "seeds": [args.seed],
        "single_seed_only": True,
        "cross_seed_stability_claim": False,
        "training_budget_matched": False,
        "continuation_steps": 300,
        "source_selected_checkpoint_sha256": source_sha,
        "artifacts": artifacts,
        "aggregate": aggregate,
        "paired_seed_deltas": deltas,
        "paired_delta_summary": summaries,
        "limitations": [
            "One training seed cannot establish cross-seed stability or uncertainty.",
            "The 300-step E4/E5 checkpoint continuations are not training-budget matched to E3 and do not support causal claims.",
            "Standard deviations are zero by single-observation convention, not estimates of variability.",
        ],
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e3-metrics", type=Path, required=True)
    parser.add_argument("--e3-text-eval", type=Path, required=True)
    parser.add_argument("--e4-metrics", type=Path, required=True)
    parser.add_argument("--e4-checkpoint", type=Path, required=True)
    parser.add_argument("--e4-text-eval", type=Path, required=True)
    parser.add_argument("--e5-metrics", type=Path, required=True)
    parser.add_argument("--e5-checkpoint", type=Path, required=True)
    parser.add_argument("--e5-text-eval", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise SummaryError("the frozen final protocol requires seed 42")
    payload = build_summary(args)
    write_json_atomic(args.output, payload)
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(payload["aggregate"])}))


if __name__ == "__main__":
    main()
