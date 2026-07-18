#!/usr/bin/env python3
"""Select a pre-sealed multimodal quality follow-up only if it beats the baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from scripts import select_corrected_candidate as base


EXPECTED_NAMES = {
    "baseline_cap8",
    "speech3rank49",
    "speech3rank19",
    "hard19",
    "balanced19",
    "speechlight49",
}
ALLOWED_VARIATION_KEYS = {
    "output_dir",
    "feature_cache_dir",
    "modality_cycle",
    "conditional_ranking_negatives",
    "conditional_ranking_negative_mode",
    "image_conditional_ranking_coef",
    "speech_conditional_ranking_coef",
    "speech_contrastive_coef",
}
EXPECTED_VARIATIONS = {
    "baseline_cap8": {
        "modality_cycle": "text,image,image,speech,speech",
        "conditional_ranking_negatives": 49,
        "conditional_ranking_negative_mode": "random",
        "image_conditional_ranking_coef": 1.0,
        "speech_conditional_ranking_coef": 2.5,
        "speech_contrastive_coef": 0.6,
    },
    "speech3rank49": {
        "modality_cycle": "text,image,image,speech,speech,speech",
        "conditional_ranking_negatives": 49,
        "conditional_ranking_negative_mode": "random",
        "image_conditional_ranking_coef": 1.0,
        "speech_conditional_ranking_coef": 4.0,
        "speech_contrastive_coef": 0.8,
    },
    "speech3rank19": {
        "modality_cycle": "text,image,image,speech,speech,speech",
        "conditional_ranking_negatives": 19,
        "conditional_ranking_negative_mode": "random",
        "image_conditional_ranking_coef": 1.0,
        "speech_conditional_ranking_coef": 3.5,
        "speech_contrastive_coef": 0.8,
    },
    "hard19": {
        "modality_cycle": "text,image,image,speech,speech",
        "conditional_ranking_negatives": 19,
        "conditional_ranking_negative_mode": "hard_text",
        "image_conditional_ranking_coef": 1.5,
        "speech_conditional_ranking_coef": 3.0,
        "speech_contrastive_coef": 0.6,
    },
    "balanced19": {
        "modality_cycle": "text,image,image,speech,speech",
        "conditional_ranking_negatives": 19,
        "conditional_ranking_negative_mode": "random",
        "image_conditional_ranking_coef": 1.0,
        "speech_conditional_ranking_coef": 3.5,
        "speech_contrastive_coef": 0.8,
    },
    "speechlight49": {
        "modality_cycle": "text,image,image,speech,speech",
        "conditional_ranking_negatives": 49,
        "conditional_ranking_negative_mode": "random",
        "image_conditional_ranking_coef": 1.0,
        "speech_conditional_ranking_coef": 3.0,
        "speech_contrastive_coef": 0.7,
    },
}
R1_KEYS = (
    "dev_5way_image_r1",
    "dev_5way_speech_r1",
    "dev_10way_image_r1",
    "dev_10way_speech_r1",
    "dev_h10_image_r1",
    "dev_h10_speech_r1",
)
CHANCES = (0.2, 0.2, 0.1, 0.1, 0.1, 0.1)
MAX_R1_REGRESSION = 0.02
MIN_WORST_LIFT_GAIN = 0.02
MIN_SPEECH_GAIN = 0.02
MAX_ROUTING_REGRESSION = 0.05

SELECTION_POLICY = {
    "eligibility": (
        "All corrected E3 integrity gates, PPL <= 20, aligned r5/r10/h10 "
        "development rows, and intended-variation checks must pass."
    ),
    "authoritative_order": [
        "retain baseline unless a follow-up passes the predeclared dominance gate",
        "maximize worst normalized lift across r5/r10/h10 image and speech",
        "maximize mean normalized lift",
        "minimize routing overflow, inactive ratio, and load CV",
        "ascending candidate name tie-break",
    ],
    "display_score": (
        "1000 + 100*worst_normalized_lift + 10*mean_normalized_lift "
        "+ (1-overflow) + 0.1*(1-inactive)"
    ),
    "sealed_policy": (
        "Development-only selector rejects sealed lexical/resolved paths and "
        "is executed before freezing or reading sealed evaluation metrics."
    ),
}

FOLLOWUP_CSV_FIELDS = list(base.CSV_FIELDS)
_insert_at = FOLLOWUP_CSV_FIELDS.index("above_chance_count")
FOLLOWUP_CSV_FIELDS[_insert_at:_insert_at] = [
    "dev_h10_image_r1",
    "dev_h10_speech_r1",
    "dev_h10_chance",
]
FOLLOWUP_CSV_FIELDS.insert(
    FOLLOWUP_CSV_FIELDS.index("routing_overflow"), "worst_normalized_lift"
)
_insert_at = FOLLOWUP_CSV_FIELDS.index("development_image_manifest_sha256")
FOLLOWUP_CSV_FIELDS[_insert_at:_insert_at] = [
    "dev_h10_metrics_sha256",
    "dev_h10_per_query_sha256",
    "dev_5way_evaluator_sha256",
    "dev_10way_evaluator_sha256",
    "dev_h10_evaluator_sha256",
]
del _insert_at


def parse_named_paths(values: Sequence[str], flag: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise base.SelectionError(f"{flag} must use NAME=PATH: {value!r}")
        name, raw = value.split("=", 1)
        if not name or not raw or name in result:
            raise base.SelectionError(f"invalid or duplicate {flag}: {value!r}")
        result[name] = Path(raw)
    return result


def validated_candidates(
    run_roots: Mapping[str, Path], dev_roots: Mapping[str, Path], output_dir: Path
) -> list[base.Candidate]:
    if set(run_roots) != EXPECTED_NAMES or set(dev_roots) != EXPECTED_NAMES:
        raise base.SelectionError(
            f"candidate names must be exactly {sorted(EXPECTED_NAMES)}"
        )
    candidates: list[base.Candidate] = []
    seen: set[tuple[str, str]] = set()
    for name in sorted(EXPECTED_NAMES):
        run_root = run_roots[name]
        dev_root = dev_roots[name]
        base.reject_sealed_path(run_root, f"{name} run root")
        base.reject_sealed_path(dev_root, f"{name} development root")
        run_resolved = run_root.resolve(strict=True)
        dev_resolved = dev_root.resolve(strict=True)
        if not run_resolved.is_dir() or not dev_resolved.is_dir():
            raise base.SelectionError(f"{name}: inputs must be directories")
        pair = (str(run_resolved), str(dev_resolved))
        if pair in seen:
            raise base.SelectionError(f"{name}: duplicate run/development roots")
        seen.add(pair)
        candidates.append(base.Candidate(name, run_resolved, dev_resolved))
    output_resolved = output_dir.resolve(strict=False)
    for candidate in candidates:
        if output_resolved == candidate.run_root or candidate.run_root in output_resolved.parents:
            raise base.SelectionError("output directory is inside a run root")
        if output_resolved == candidate.dev_root or candidate.dev_root in output_resolved.parents:
            raise base.SelectionError("output directory is inside a development root")
    return candidates


def validate_intended_variations(candidates: Sequence[base.Candidate]) -> None:
    baseline = next(candidate for candidate in candidates if candidate.name == "baseline_cap8")
    if baseline.manifest_args is None:
        baseline.reject("manifest", "baseline manifest args are missing")
        return
    baseline_fixed = {
        key: value
        for key, value in baseline.manifest_args.items()
        if key not in ALLOWED_VARIATION_KEYS
    }
    for candidate in candidates:
        args = candidate.manifest_args
        if args is None:
            continue
        observed = {key: args.get(key) for key in EXPECTED_VARIATIONS[candidate.name]}
        if observed != EXPECTED_VARIATIONS[candidate.name]:
            candidate.reject(
                "followup_recipe_mismatch",
                f"{candidate.name}: intended follow-up fields differ: {observed}",
            )
        fixed = {key: value for key, value in args.items() if key not in ALLOWED_VARIATION_KEYS}
        if fixed != baseline_fixed:
            candidate.reject(
                "followup_uncontrolled_difference",
                f"{candidate.name}: non-follow-up arguments differ from baseline",
            )
        if not base.numbers_close(args.get("capacity_factor"), 8.0) or not base.numbers_close(
            args.get("aux_coef"), 0.02
        ):
            candidate.reject(
                "followup_capacity_aux",
                f"{candidate.name}: capacity/aux must remain 8.0/0.02",
            )


def add_quality_components(candidate: base.Candidate) -> None:
    values = [float(candidate.metrics[key]) for key in R1_KEYS]
    lifts = [
        (value - chance) / (1.0 - chance)
        for value, chance in zip(values, CHANCES)
    ]
    worst = min(lifts)
    mean = sum(lifts) / len(lifts)
    overflow = float(candidate.metrics["routing_overflow"])
    inactive = float(candidate.metrics["routing_inactive"])
    load_cv = float(candidate.metrics["routing_load_cv"])
    candidate.selection_components = {
        "ppl_gate_passed": float(candidate.metrics["text_ppl"]) <= 20.0,
        "above_chance_count": sum(value > chance for value, chance in zip(values, CHANCES)),
        "mean_normalized_positive_lift": mean,
        "worst_normalized_lift": worst,
        "routing_overflow": overflow,
        "routing_inactive": inactive,
        "routing_load_cv": load_cv,
    }
    candidate.selection_score = (
        1000.0
        + 100.0 * worst
        + 10.0 * mean
        + (1.0 - overflow)
        + 0.1 * (1.0 - inactive)
    )


def passes_dominance(candidate: base.Candidate, baseline: base.Candidate) -> bool:
    current = candidate.selection_components or {}
    reference = baseline.selection_components or {}
    values = [float(candidate.metrics[key]) for key in R1_KEYS]
    baseline_values = [float(baseline.metrics[key]) for key in R1_KEYS]
    no_large_r1_regression = all(
        value >= baseline_value - MAX_R1_REGRESSION
        for value, baseline_value in zip(values, baseline_values)
    )
    worst_lift_gain = (
        float(current["worst_normalized_lift"])
        >= float(reference["worst_normalized_lift"]) + MIN_WORST_LIFT_GAIN
    )
    speech_gain = (
        candidate.metrics["dev_10way_speech_r1"]
        >= baseline.metrics["dev_10way_speech_r1"] + MIN_SPEECH_GAIN
        or candidate.metrics["dev_h10_speech_r1"]
        >= baseline.metrics["dev_h10_speech_r1"] + MIN_SPEECH_GAIN
    )
    routing_ok = (
        candidate.metrics["routing_overflow"]
        <= baseline.metrics["routing_overflow"] + MAX_ROUTING_REGRESSION
        and candidate.metrics["routing_inactive"]
        <= baseline.metrics["routing_inactive"] + MAX_ROUTING_REGRESSION
    )
    return bool(no_large_r1_regression and worst_lift_gain and speech_gain and routing_ok)


def ranking_key(candidate: base.Candidate) -> tuple[Any, ...]:
    components = candidate.selection_components or {}
    return (
        -float(components["worst_normalized_lift"]),
        -float(components["mean_normalized_positive_lift"]),
        float(components["routing_overflow"]),
        float(components["routing_inactive"]),
        float(components["routing_load_cv"]),
        candidate.name,
    )


def validate_h10(candidate: base.Candidate, cache: base.ArtifactCache) -> None:
    base._validate_eval_cell(
        candidate,
        10,
        cache,
        cell_name="h10",
        negative_mode="hard_text",
        metric_label="h10",
    )
    evaluator_sha = candidate.artifacts.get("dev_h10_evaluator", {}).get("sha256")
    shared_sha = candidate.artifacts.get("dev_5way_evaluator", {}).get("sha256")
    base._require(
        bool(evaluator_sha) and evaluator_sha == shared_sha,
        "development_protocol",
        f"{candidate.name}: r5/r10/h10 evaluator hashes differ",
    )


def _followup_csv_rows(
    candidates: Sequence[base.Candidate], selected_name: Optional[str]
) -> list[Dict[str, Any]]:
    rows = base._csv_rows(candidates, selected_name)
    for row, candidate in zip(rows, candidates):
        components = candidate.selection_components or {}
        row.update(
            {
                "dev_h10_image_r1": candidate.metrics.get("dev_h10_image_r1", ""),
                "dev_h10_speech_r1": candidate.metrics.get("dev_h10_speech_r1", ""),
                "dev_h10_chance": candidate.metrics.get("dev_h10_chance", ""),
                "worst_normalized_lift": components.get("worst_normalized_lift", ""),
                "dev_h10_metrics_sha256": base._artifact_hash(candidate, "dev_h10_metrics"),
                "dev_h10_per_query_sha256": base._artifact_hash(candidate, "dev_h10_per_query"),
                "dev_5way_evaluator_sha256": base._artifact_hash(candidate, "dev_5way_evaluator"),
                "dev_10way_evaluator_sha256": base._artifact_hash(candidate, "dev_10way_evaluator"),
                "dev_h10_evaluator_sha256": base._artifact_hash(candidate, "dev_h10_evaluator"),
            }
        )
    return rows


def _metric_text(metrics: Mapping[str, Any], key: str) -> str:
    value = metrics.get(key)
    return "" if value is None else f"{float(value):.9g}"


def _followup_markdown(report: Mapping[str, Any]) -> str:
    policy = report["selection_policy"]
    lines = [
        "# Final Multimodal Quality Candidate Selection",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Selected candidate: **{report.get('selected_candidate') or 'none'}**",
        "",
        "## Predeclared Development-Only Policy",
        "",
        policy["eligibility"],
        "",
    ]
    for index, item in enumerate(policy["authoritative_order"], 1):
        lines.append(f"{index}. {item}.")
    lines.extend(
        [
            "",
            f"Display score: `{policy['display_score']}`.",
            "",
            f"Sealed policy: {policy['sealed_policy']}",
            "",
            "## Candidate Metrics",
            "",
            "| candidate | valid | selected | rank | PPL | r5 image | r5 speech | r10 image | r10 speech | h10 image | h10 speech | worst lift | mean lift | overflow | inactive | load CV | score |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in report["candidates"]:
        metrics = candidate["metrics"]
        components = candidate.get("selection_components") or {}
        score = candidate.get("selection_score")
        lines.append(
            "| {name} | {valid} | {selected} | {rank} | {ppl} | {i5} | {s5} | {i10} | {s10} | {ih10} | {sh10} | {worst} | {mean} | {overflow} | {inactive} | {cv} | {score} |".format(
                name=candidate["name"],
                valid="yes" if candidate["valid"] else "no",
                selected="yes" if candidate["selected"] else "no",
                rank=candidate.get("rank") or "",
                ppl=_metric_text(metrics, "text_ppl"),
                i5=_metric_text(metrics, "dev_5way_image_r1"),
                s5=_metric_text(metrics, "dev_5way_speech_r1"),
                i10=_metric_text(metrics, "dev_10way_image_r1"),
                s10=_metric_text(metrics, "dev_10way_speech_r1"),
                ih10=_metric_text(metrics, "dev_h10_image_r1"),
                sh10=_metric_text(metrics, "dev_h10_speech_r1"),
                worst=(
                    ""
                    if components.get("worst_normalized_lift") is None
                    else f"{float(components['worst_normalized_lift']):.9g}"
                ),
                mean=(
                    ""
                    if components.get("mean_normalized_positive_lift") is None
                    else f"{float(components['mean_normalized_positive_lift']):.9g}"
                ),
                overflow=_metric_text(metrics, "routing_overflow"),
                inactive=_metric_text(metrics, "routing_inactive"),
                cv=_metric_text(metrics, "routing_load_cv"),
                score="" if score is None else f"{float(score):.9f}",
            )
        )
    lines.extend(["", "## Rejections", ""])
    rejected = [candidate for candidate in report["candidates"] if candidate["reasons"]]
    if not rejected:
        lines.append("No candidate was rejected.")
    for candidate in rejected:
        lines.extend([f"### {candidate['name']}", ""])
        for reason in candidate["reasons"]:
            lines.append(f"- `{reason['code']}`: {reason['message']}")
        lines.append("")
    lines.extend(["## Provenance", ""])
    for candidate in report["candidates"]:
        lines.extend([f"### {candidate['name']}", ""])
        for role, artifact in candidate["artifacts"].items():
            lines.append(
                f"- `{role}`: `{artifact['path']}`; SHA-256 `{artifact['sha256']}`; {artifact['size_bytes']} bytes."
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_followup_outputs(
    report: Mapping[str, Any], output_dir: Path, candidates: Sequence[base.Candidate]
) -> None:
    base.reject_sealed_path(output_dir, "output directory")
    if os.path.lexists(output_dir):
        raise base.SelectionError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    try:
        (temporary / "candidate_selection.json").write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "candidate_selection.md").write_text(
            _followup_markdown(report), encoding="utf-8"
        )
        with (temporary / "candidate_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=FOLLOWUP_CSV_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(
                _followup_csv_rows(candidates, report.get("selected_candidate"))
            )
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def select_followup(
    run_roots: Mapping[str, Path],
    dev_roots: Mapping[str, Path],
    output_dir: Path,
) -> Dict[str, Any]:
    base.reject_sealed_path(output_dir, "follow-up selection output")
    if os.path.lexists(output_dir):
        raise base.SelectionError(f"refusing to overwrite {output_dir}")
    candidates = validated_candidates(run_roots, dev_roots, output_dir)
    cache = base.ArtifactCache()
    for candidate in candidates:
        base._safe_phase(candidate, base._validate_manifest)
        if candidate.manifest_args is not None:
            base._safe_phase(candidate, base._validate_run)
        if candidate.checkpoint_sha256:
            base._safe_phase(candidate, base._validate_development, cache)
        if candidate.valid:
            base._safe_phase(candidate, validate_h10, cache)
    validate_intended_variations(candidates)

    baseline = next(candidate for candidate in candidates if candidate.name == "baseline_cap8")
    if not baseline.valid:
        raise base.NoValidCandidateError(
            f"baseline_cap8 failed integrity checks: {baseline.reasons}"
        )
    baseline_dataset = base.canonical_sha256(baseline.dataset_provenance)
    for candidate in candidates:
        if candidate.dataset_provenance is not None and base.canonical_sha256(
            candidate.dataset_provenance
        ) != baseline_dataset:
            candidate.reject(
                "dataset_provenance_mismatch",
                f"{candidate.name}: dataset provenance differs from baseline",
            )
    for cell in (5, 10, "h10"):
        expected = baseline.protocol_digests.get(cell)
        for candidate in candidates:
            if cell in candidate.protocol_digests and candidate.protocol_digests[cell] != expected:
                candidate.reject(
                    "development_protocol_mismatch",
                    f"{candidate.name}: {cell} protocol differs from baseline",
                )

    valid = [candidate for candidate in candidates if candidate.valid]
    for candidate in valid:
        add_quality_components(candidate)
    ranked = sorted(valid, key=ranking_key)
    qualifying = [
        candidate
        for candidate in ranked
        if candidate.name != baseline.name and passes_dominance(candidate, baseline)
    ]
    selected = min(qualifying, key=ranking_key) if qualifying else baseline
    ordered = [selected] + [candidate for candidate in ranked if candidate is not selected]
    for rank, candidate in enumerate(ordered, 1):
        candidate.rank = rank
    report = {
        "schema_version": 1,
        "status": "selected",
        "selected_candidate": selected.name,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid),
        "selection_policy": SELECTION_POLICY,
        "shared_validation": {
            "development_only": True,
            "sealed_metrics_used": False,
            "baseline_candidate": baseline.name,
            "dominance_gate": {
                "max_r1_regression": MAX_R1_REGRESSION,
                "min_worst_lift_gain": MIN_WORST_LIFT_GAIN,
                "min_speech_gain": MIN_SPEECH_GAIN,
                "max_routing_regression": MAX_ROUTING_REGRESSION,
            },
            "protocol_sha256": {
                str(key): value for key, value in baseline.protocol_digests.items()
            },
        },
        "candidates": [
            base._candidate_payload(candidate, candidate.name == selected.name)
            for candidate in candidates
        ],
    }
    _write_followup_outputs(report, output_dir, candidates)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--dev-eval-dir", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = select_followup(
            parse_named_paths(args.run_root, "--run-root"),
            parse_named_paths(args.dev_eval_dir, "--dev-eval-dir"),
            args.output_dir,
        )
    except base.SelectionError as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "selected_candidate": report["selected_candidate"],
                "valid_candidate_count": report["valid_candidate_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
