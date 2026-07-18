#!/usr/bin/env python3
"""Validate and analyze the complete frozen sealed-evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_paired_controls as paired  # noqa: E402
from scripts.freeze_evaluation_protocol import sha256_file, verify_protocol  # noqa: E402
from scripts.protocol_v2 import (  # noqa: E402
    ProtocolV2Error,
    frozen_checkpoint_artifact,
    validate_metrics_against_protocol_v2,
)


CONTROL_PATHS = {
    "real": "real",
    "shuffled": "shuffled",
    "zero": "zero",
    "norm-matched-random": "random",
    "no-prefix": "no-prefix",
}
ANALYSIS_CONTROLS = {
    "real": "real",
    "shuffled": "shuffled",
    "zero": "zero",
    "norm-matched-random": "random",
    "no-prefix": "no_prefix",
}


class MatrixError(ValueError):
    """Raised when sealed evidence is incomplete or inconsistent."""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatrixError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatrixError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise MatrixError(f"cannot open JSONL {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MatrixError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise MatrixError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    if not rows:
        raise MatrixError(f"sealed per-query file is empty: {path}")
    return rows


def holm_adjust(records: Sequence[Tuple[str, float]], alpha: float) -> List[Dict[str, Any]]:
    if not records:
        raise MatrixError("Holm family is empty")
    for label, value in records:
        if not label or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise MatrixError(f"invalid Holm p-value: {label}={value}")
    ordered = sorted(records, key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running = 0.0
    output: List[Dict[str, Any]] = []
    for index, (label, raw) in enumerate(ordered):
        adjusted = max(running, min(1.0, (count - index) * raw))
        running = adjusted
        output.append({
            "label": label,
            "raw_p": raw,
            "adjusted_p": adjusted,
            "rank": index + 1,
            "family_size": count,
            "alpha": alpha,
            "rejected": adjusted <= alpha,
        })
    return output


def _expected_evaluator_hashes(protocol: Mapping[str, Any]) -> set[Tuple[str, str]]:
    records = protocol.get("inputs", {}).get("evaluator_scripts", [])
    return {
        (str(record.get("path", "")), str(record.get("sha256", "")))
        for record in records
        if isinstance(record, Mapping)
    }


def _condition_name(control: str) -> str:
    return ANALYSIS_CONTROLS[control]


def validate_raw_condition(
    *,
    cell: Mapping[str, Any],
    control: str,
    metrics_path: Path,
    rows_path: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    evaluator_hashes: set[Tuple[str, str]],
) -> Dict[str, Any]:
    if not metrics_path.is_file() or not rows_path.is_file():
        raise MatrixError(f"missing sealed output for {cell['id']}/{control}")
    metrics = _read_json(metrics_path)
    rows = _read_jsonl(rows_path)
    try:
        validate_metrics_against_protocol_v2(
            protocol,
            metrics,
            rows,
            cell_id=str(cell["id"]),
            control=control,
            protocol_file_sha256=protocol_sha256,
            per_query_file_sha256=sha256_file(rows_path),
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
        )
    except ProtocolV2Error as exc:
        raise MatrixError(f"{cell['id']}/{control}: {exc}") from exc
    query_counts = protocol["query_counts"]
    expected_by_modality = {
        "image": int(query_counts["image"]),
        "speech": int(query_counts["speech"]),
    }
    observed = Counter(str(row.get("modality", "")) for row in rows)
    if observed != Counter(expected_by_modality):
        raise MatrixError(
            f"{cell['id']}/{control}: modality row counts {dict(observed)} "
            f"!= {expected_by_modality}"
        )
    expected_condition = _condition_name(control)
    candidate_count = int(cell["candidate_count"])
    candidate_seed = int(protocol["seeds"]["candidate_seed"])
    allowed_protocols = set(protocol["candidate_sets"]["protocols"])
    for index, row in enumerate(rows):
        row_condition = str(row.get("condition", row.get("prefix_control", "")))
        if row_condition != expected_condition:
            raise MatrixError(
                f"{cell['id']}/{control}:{index}: condition={row_condition!r}"
            )
        row_protocol = row.get("protocol")
        if not isinstance(row_protocol, Mapping):
            raise MatrixError(f"{cell['id']}/{control}:{index}: protocol metadata missing")
        checks = (
            (row_protocol.get("manifest_sha256") == protocol_sha256, "protocol hash"),
            ("sealed" in str(row_protocol.get("eval_split_name", "")).lower(), "sealed split"),
            (int(row_protocol.get("candidate_count", -1)) == candidate_count, "candidate count"),
            (int(row_protocol.get("candidate_seed", -1)) == candidate_seed, "candidate seed"),
            (row_protocol.get("randomized_positive_position") is True, "positive randomization"),
            (str(row_protocol.get("negative_mode", "")) == str(cell["negative_mode"]), "negative mode"),
            (str(row_protocol.get("name", "")) in allowed_protocols, "protocol name"),
        )
        failed = [label for passed, label in checks if not passed]
        if failed:
            raise MatrixError(
                f"{cell['id']}/{control}:{index}: protocol mismatch: {failed}"
            )

    provenance = metrics.get("provenance")
    if not isinstance(provenance, Mapping):
        raise MatrixError(f"{cell['id']}/{control}: metrics provenance missing")
    evaluator_pair = (
        str(provenance.get("evaluator_path", "")),
        str(provenance.get("evaluator_sha256", "")),
    )
    checks = (
        (metrics.get("sealed_protocol") is True, "sealed protocol flag"),
        (metrics.get("protocol_manifest_sha256") == protocol_sha256, "metrics protocol hash"),
        (int(metrics.get("candidate_count", -1)) == candidate_count, "metrics candidate count"),
        (metrics.get("per_query_sha256") == sha256_file(rows_path), "per-query hash"),
        (int(metrics.get("per_query_rows", -1)) == len(rows), "per-query count"),
        (provenance.get("checkpoint_sha256") == checkpoint_sha256, "checkpoint hash"),
        (evaluator_pair in evaluator_hashes, "evaluator fingerprint"),
        (
            bool(metrics.get("conditional_uses_multimodal_prefix"))
            == (control != "no-prefix"),
            "shared-prefix path",
        ),
    )
    failed = [label for passed, label in checks if not passed]
    if failed:
        raise MatrixError(f"{cell['id']}/{control}: metrics mismatch: {failed}")
    return {
        "metrics": metrics,
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "per_query_path": str(rows_path.resolve()),
        "per_query_sha256": sha256_file(rows_path),
        "rows": len(rows),
    }


def _effect(report: Mapping[str, Any], control: str, modality: str) -> Mapping[str, Any]:
    key = _condition_name(control)
    return report["comparisons_vs_real"][key]["by_modality"][modality]


def build_holm_results(
    protocol: Mapping[str, Any], cell_reports: Mapping[str, Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for family in protocol["holm_families"]:
        records: List[Tuple[str, float]] = []
        for cell_id in family["cell_ids"]:
            report = cell_reports[str(cell_id)]
            for modality in family["modalities"]:
                for control in family["controls"]:
                    effect = _effect(report, str(control), str(modality))
                    if family["test"] == "paired_exact_mcnemar":
                        p_value = float(effect["mcnemar_exact"]["p_value_two_sided"])
                    elif family["test"] == "paired_sign_flip_permutation":
                        p_value = float(
                            effect["mrr_difference"]["permutation"]["p_value_two_sided"]
                        )
                    else:
                        raise MatrixError(f"unsupported Holm test: {family['test']}")
                    label = f"{cell_id}:{modality}:{control}"
                    records.append((label, p_value))
        adjusted = holm_adjust(records, float(family["alpha"]))
        expected = (
            len(family["cell_ids"])
            * len(family["modalities"])
            * len(family["controls"])
        )
        if len(adjusted) != expected:
            raise MatrixError(f"Holm family {family['id']} is incomplete")
        output.append({**dict(family), "comparisons": adjusted})
    return output


def _holm_lookup(holm_results: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    lookup: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for family in holm_results:
        for comparison in family["comparisons"]:
            lookup[(str(family["id"]), str(comparison["label"]))] = comparison
    return lookup


def build_claim_status(
    protocol_sha256: str,
    protocol: Mapping[str, Any],
    cell_reports: Mapping[str, Mapping[str, Any]],
    holm_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    primary = [cell for cell in protocol["evaluation_matrix"] if cell["role"] == "primary"]
    lookup = _holm_lookup(holm_results)
    modalities: Dict[str, Any] = {}
    for modality in ("image", "speech"):
        reasons: List[str] = []
        endpoints: List[Dict[str, Any]] = []
        for cell in primary:
            cell_id = str(cell["id"])
            report = cell_reports[cell_id]
            real = report["conditions"]["real"]["by_modality"][modality]
            chance = float(report["chance"]["by_modality"][modality]["r_at_1"])
            above_chance = float(real["r_at_1"]["wilson_95"]["low"]) > chance
            if not above_chance:
                reasons.append(f"{cell_id}: R@1 Wilson interval does not clear chance")
            control_results: Dict[str, Any] = {}
            for control in protocol["holm_families"][0]["controls"]:
                effect = _effect(report, str(control), modality)
                label = f"{cell_id}:{modality}:{control}"
                r1_holm = lookup[("primary_r1_mcnemar", label)]
                mrr_holm = lookup[("primary_mrr_permutation", label)]
                r1_low = float(effect["r_at_1_difference"]["ci_95"]["low"])
                mrr_low = float(effect["mrr_difference"]["bootstrap"]["ci_95"]["low"])
                passed = (
                    r1_low > 0.0
                    and mrr_low > 0.0
                    and bool(r1_holm["rejected"])
                    and bool(mrr_holm["rejected"])
                )
                if not passed:
                    reasons.append(f"{cell_id}: paired effect over {control} is not established")
                control_results[str(control)] = {
                    "passed": passed,
                    "r_at_1_ci_low": r1_low,
                    "mrr_ci_low": mrr_low,
                    "r_at_1_holm_adjusted_p": r1_holm["adjusted_p"],
                    "mrr_holm_adjusted_p": mrr_holm["adjusted_p"],
                }
            endpoints.append({
                "cell_id": cell_id,
                "candidate_count": int(cell["candidate_count"]),
                "negative_mode": str(cell["negative_mode"]),
                "r_at_1": float(real["r_at_1"]["rate"]),
                "chance_r_at_1": chance,
                "above_chance_wilson": above_chance,
                "controls": control_results,
            })
        supported = not reasons
        modalities[modality] = {
            "status": "supported" if supported else "not_established",
            "statement": (
                f"{modality.capitalize()} conditioning is supported on the frozen sealed "
                "hard-negative endpoint by paired controls with Holm correction."
                if supported
                else f"{modality.capitalize()} multimodal conditioning was not established "
                "by the frozen sealed paired-control protocol."
            ),
            "reasons": sorted(set(reasons)),
            "primary_endpoints": endpoints,
        }
    return {
        "schema_version": 1,
        "protocol_manifest_sha256": protocol_sha256,
        "sealed_metrics_used_for_adaptation": False,
        "decision_rule": "all frozen primary above-chance, CI-direction, and Holm gates",
        "modalities": modalities,
    }


def render_markdown(
    protocol: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    claim: Mapping[str, Any],
) -> str:
    lines = [
        "# Frozen Sealed Evaluation Matrix",
        "",
        "The hard-negative 10-way cell is the predeclared primary endpoint. All other cells are secondary diagnostics.",
        "",
        "| cell | role | candidates | negatives | modality | real R@1 | chance | Wilson low |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for cell in protocol["evaluation_matrix"]:
        report = reports[str(cell["id"])]
        for modality in ("image", "speech"):
            real = report["conditions"]["real"]["by_modality"][modality]["r_at_1"]
            chance = report["chance"]["by_modality"][modality]["r_at_1"]
            lines.append(
                f"| {cell['id']} | {cell['role']} | {cell['candidate_count']} | "
                f"{cell['negative_mode']} | {modality} | {real['rate']:.4f} | "
                f"{chance:.4f} | {real['wilson_95']['low']:.4f} |"
            )
    lines.extend(["", "## Claim Status", ""])
    for modality, record in claim["modalities"].items():
        lines.append(f"- **{modality}: {record['status']}**. {record['statement']}")
        for reason in record["reasons"]:
            lines.append(f"  - {reason}")
    lines.extend([
        "",
        "Sealed results were evaluated once after model selection and were not used for adaptation.",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output_dir.exists():
        raise MatrixError(f"refusing to overwrite output directory: {args.output_dir}")
    protocol_path = args.protocol.resolve(strict=True)
    protocol = verify_protocol(protocol_path)
    protocol_sha256 = sha256_file(protocol_path)
    artifact = frozen_checkpoint_artifact(protocol)
    checkpoint = Path(artifact["path"])
    if not checkpoint.is_file():
        raise MatrixError(f"frozen checkpoint is missing: {checkpoint}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != artifact["sha256"]:
        raise MatrixError("frozen checkpoint artifact SHA256 drifted")
    evaluator_hashes = _expected_evaluator_hashes(protocol)
    if not evaluator_hashes:
        raise MatrixError("frozen evaluator fingerprints are missing")

    validated: Dict[str, Dict[str, Any]] = {}
    cell_reports: Dict[str, Dict[str, Any]] = {}
    paired_rows_by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for cell in protocol["evaluation_matrix"]:
        cell_id = str(cell["id"])
        condition_paths: Dict[str, Path] = {}
        validated[cell_id] = {}
        for control in protocol["controls"]:
            directory = args.matrix_root / f"{cell_id}-{CONTROL_PATHS[str(control)]}"
            metrics_path = directory / "metrics.json"
            rows_path = directory / "per_query.jsonl"
            validated[cell_id][str(control)] = validate_raw_condition(
                cell=cell,
                control=str(control),
                metrics_path=metrics_path,
                rows_path=rows_path,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                checkpoint_path=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                evaluator_hashes=evaluator_hashes,
            )
            condition_paths[_condition_name(str(control))] = rows_path
        report, aligned_rows = paired.analyze(
            condition_paths,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
            seed=args.seed,
        )
        cell_reports[cell_id] = report
        paired_rows_by_cell[cell_id] = aligned_rows

    holm_results = build_holm_results(protocol, cell_reports)
    claim = build_claim_status(protocol_sha256, protocol, cell_reports, holm_results)
    args.output_dir.mkdir(parents=True)
    cells_dir = args.output_dir / "cells"
    cells_dir.mkdir()
    cell_index: Dict[str, Any] = {}
    for cell_id, report in cell_reports.items():
        cell_dir = cells_dir / cell_id
        cell_dir.mkdir()
        report_path = cell_dir / "paired_analysis.json"
        markdown_path = cell_dir / "paired_analysis.md"
        rows_path = cell_dir / "aligned_queries.jsonl"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        markdown_path.write_text(paired.render_markdown(report), encoding="utf-8")
        rows_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in paired_rows_by_cell[cell_id]),
            encoding="utf-8",
        )
        cell_index[cell_id] = {
            "paired_analysis": {"path": str(report_path.resolve()), "sha256": sha256_file(report_path)},
            "aligned_queries": {"path": str(rows_path.resolve()), "sha256": sha256_file(rows_path)},
            "raw_inputs": validated[cell_id],
        }

    claim_path = args.output_dir / "claim_status.json"
    claim_path.write_text(json.dumps(claim, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matrix = {
        "schema_version": 1,
        "protocol_manifest": {"path": str(protocol_path), "sha256": protocol_sha256},
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha256},
        "matrix_complete": True,
        "evaluation_matrix": protocol["evaluation_matrix"],
        "cells": cell_index,
        "holm_families": holm_results,
        "claim_status": {"path": str(claim_path.resolve()), "sha256": sha256_file(claim_path)},
        "sealed_metrics_used_for_adaptation": False,
        "analyzer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
    }
    matrix_path = args.output_dir / "sealed_matrix_analysis.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = args.output_dir / "sealed_matrix_analysis.md"
    markdown_path.write_text(render_markdown(protocol, cell_reports, claim), encoding="utf-8")
    return matrix


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260709)
    return parser.parse_args(argv)


def main() -> int:
    try:
        result = run(parse_args())
    except (MatrixError, paired.AnalysisError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "matrix_complete": result["matrix_complete"],
        "output": result["claim_status"]["path"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
