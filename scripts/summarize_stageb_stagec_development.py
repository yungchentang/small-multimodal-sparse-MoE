#!/usr/bin/env python
"""Summarize development-only Stage B and Stage C experiment evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STAGE_B_VARIANTS = ("E2_CE_only", "E2D_logits_kl")
STAGE_C_DIR = "E3_final_multimodal_top2"
STAGE_C_TEXT_DIR = "E3_final_multimodal_top2_text_eval"
FORBIDDEN_PATH_WORDS = ("sealed", "synthetic")
OUTPUT_STEM = "stageb_stagec_development_summary"
COMMIT_SHA_PATTERN = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

STAGE_B_METRICS = (
    "perplexity",
    "next_token_accuracy",
    "teacher_student_kl",
    "router_kl",
    "moe_reconstruction_cosine",
    "moe_reconstruction_mse",
    "moe_reconstruction_rmse",
    "capacity_overflow_ratio_mean",
    "inactive_expert_ratio_mean",
)

CSV_FIELDS = (
    "stage",
    "source_root",
    "run_id",
    "variant",
    "status",
    "validation_passed",
    "observed_steps",
    "expected_steps",
    "alignment_steps",
    "expected_alignment_steps",
    "perplexity",
    "next_token_accuracy",
    "teacher_student_kl",
    "router_kl",
    "moe_reconstruction_cosine",
    "moe_reconstruction_mse",
    "moe_reconstruction_rmse",
    "capacity_overflow_ratio_mean",
    "inactive_expert_ratio_mean",
    "conditional_image_r1",
    "conditional_image_chance_r1",
    "conditional_speech_r1",
    "conditional_speech_chance_r1",
    "embedding_image_r1",
    "embedding_image_chance_r1",
    "embedding_speech_r1",
    "embedding_speech_chance_r1",
    "checkpoint_sha256",
    "stage_b_initial_checkpoint_sha256",
    "multimodal_initial_checkpoint_sha256",
    "issue_codes",
)


class EvidenceError(ValueError):
    """An evidence input is structurally invalid."""


def reject_forbidden_path(path: Path, label: str) -> None:
    lowered = str(path).lower()
    for word in FORBIDDEN_PATH_WORDS:
        if word in lowered:
            raise EvidenceError(f"Refusing {label} containing forbidden word {word!r}: {path}")


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"Non-finite JSON value: {value}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise EvidenceError(f"Invalid JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_constant)
            except json.JSONDecodeError as error:
                raise EvidenceError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise EvidenceError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def issue(
    issues: list[dict[str, str]], code: str, message: str, path: Path, severity: str = "error"
) -> None:
    issues.append(
        {"code": code, "message": message, "path": str(path), "severity": severity}
    )


def finite_number(
    value: Any,
    label: str,
    path: Path,
    issues: list[dict[str, str]],
    *,
    required: bool = True,
) -> float | int | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issue(issues, "invalid_metric_type", f"{label} must be a real numeric value; got {value!r}", path)
        return None
    result = float(value)
    if not math.isfinite(result):
        issue(issues, "non_finite_metric", f"{label} is non-finite: {value!r}", path)
        return None
    return value


def validate_steps(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    issues: list[dict[str, str]],
    metric_fields: Sequence[str] = (),
) -> int:
    last = 0
    first_step: int | None = None
    for index, row in enumerate(rows, 1):
        step = finite_number(row.get("step"), "step", path, issues)
        if step is None:
            continue
        if int(step) != step:
            issue(
                issues, "non_contiguous_steps",
                f"Step must be an integer; got {step!r}", path,
            )
            continue
        if first_step is None:
            first_step = int(step)
        expected_step = first_step + index - 1
        if int(step) != expected_step:
            issue(
                issues, "non_contiguous_steps",
                f"Expected resumed-contiguous step {expected_step}; got {step!r}", path,
            )
        last = max(last, int(step))
        for field in metric_fields:
            if field in row:
                finite_number(row[field], field, path, issues)
    return last


def same_path(recorded: Any, actual: Path) -> bool:
    if not isinstance(recorded, str) or not recorded:
        return False
    candidate = Path(recorded)
    if candidate.is_absolute():
        return candidate.resolve(strict=False) == actual.resolve(strict=False)
    return candidate.as_posix() == actual.as_posix() or actual.as_posix().endswith(candidate.as_posix())


def real_data_checks(manifest: Mapping[str, Any], path: Path, issues: list[dict[str, str]]) -> None:
    args = manifest.get("args")
    if not isinstance(args, dict):
        issue(issues, "missing_manifest_args", "manifest.args is missing", path)
        return
    data_dir = str(args.get("data_dir", manifest.get("data_dir", "")))
    policy = str(manifest.get("data_policy", ""))
    real_named = Path(data_dir).name.startswith("real_subset")
    if not real_named and policy != "development_only_real_manifests":
        issue(issues, "non_real_data", f"Data source is not declared real development data: {data_dir!r}", path)
    serialized = json.dumps(manifest, sort_keys=True).lower()
    if '"synthetic_evidence_used": true' in serialized or '"sealed_evidence_used": true' in serialized:
        issue(issues, "non_real_data", "Manifest declares synthetic or sealed evidence use", path)


def load_manifest(run_root: Path, issues: list[dict[str, str]]) -> dict[str, Any] | None:
    path = run_root / "manifest.json"
    try:
        manifest = read_json(path)
    except (FileNotFoundError, EvidenceError) as error:
        issue(issues, "invalid_manifest", str(error), path)
        return None
    real_data_checks(manifest, path, issues)
    args = manifest.get("args")
    if isinstance(args, dict) and args.get("output_dir") is not None:
        if not same_path(args["output_dir"], run_root):
            issue(issues, "provenance_mismatch", "manifest output_dir does not match run root", path)
    return manifest


def status_for(
    progress: bool, terminal_present: bool, expected_complete: bool, issues: Sequence[Mapping[str, str]]
) -> str:
    hard_errors = [item for item in issues if item["severity"] == "error"]
    integrity_codes = {
        "invalid_metric_type",
        "non_finite_metric",
        "copied_text_metrics",
        "non_real_data",
        "provenance_mismatch",
        "checkpoint_sha_mismatch",
        "bypass_retrieval_path",
        "missing_prefix_routing_accounting",
        "invalid_jsonl",
        "non_contiguous_steps",
        "unverified_multimodal_initial_manifest",
    }
    if any(item["code"] in integrity_codes for item in hard_errors):
        return "rejected"
    if terminal_present and expected_complete:
        return "completed" if not hard_errors else "rejected"
    if progress:
        return "running"
    return "incomplete"


def validate_checkpoint(
    checkpoint: Path,
    provenance: Mapping[str, Any] | None,
    issues: list[dict[str, str]],
) -> str | None:
    if not checkpoint.is_file():
        issue(issues, "missing_checkpoint", "Final checkpoint is missing", checkpoint)
        return None
    digest = sha256_file(checkpoint)
    if provenance is None:
        issue(issues, "provenance_mismatch", "Checkpoint provenance is missing", checkpoint)
        return digest
    expected_sha = provenance.get("saved_checkpoint_sha256")
    if expected_sha is not None and expected_sha != digest:
        issue(issues, "checkpoint_sha_mismatch", f"Recorded SHA {expected_sha!r} != {digest}", checkpoint)
    saved_path = provenance.get("saved_checkpoint")
    if not same_path(saved_path, checkpoint):
        issue(issues, "provenance_mismatch", "Recorded checkpoint path does not match checkpoint", checkpoint)
    expected_size = provenance.get("saved_checkpoint_size_bytes")
    if expected_size is not None and expected_size != checkpoint.stat().st_size:
        issue(issues, "provenance_mismatch", "Recorded checkpoint size does not match checkpoint", checkpoint)
    return digest


def summarize_stage_b_run(source_root: Path, run_root: Path) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    manifest = load_manifest(run_root, issues)
    args = manifest.get("args", {}) if manifest else {}
    expected_steps = args.get("distill_steps") if isinstance(args, dict) else None
    if isinstance(expected_steps, bool) or not isinstance(expected_steps, int) or expected_steps <= 0:
        issue(issues, "invalid_expected_steps", "distill_steps must be a positive integer", run_root / "manifest.json")
        expected_steps = None

    variants: dict[str, dict[str, Any]] = {}
    for variant in STAGE_B_VARIANTS:
        variant_issues: list[dict[str, str]] = []
        variant_root = run_root / variant
        metrics_path = variant_root / "metrics.json"
        log_path = variant_root / "train_metrics.jsonl"
        checkpoint_path = variant_root / "checkpoint_final.pt"
        try:
            rows = read_jsonl(log_path)
            observed_steps = validate_steps(rows, log_path, variant_issues)
        except EvidenceError as error:
            rows = []
            observed_steps = 0
            issue(variant_issues, "invalid_jsonl", str(error), log_path)
        metrics: dict[str, Any] | None = None
        if metrics_path.is_file():
            try:
                metrics = read_json(metrics_path)
            except EvidenceError as error:
                issue(variant_issues, "invalid_metrics", str(error), metrics_path)
        values: dict[str, Any] = {}
        provenance: Mapping[str, Any] | None = None
        if metrics is not None:
            if metrics.get("experiment_id") != variant:
                issue(variant_issues, "provenance_mismatch", "experiment_id does not match variant", metrics_path)
            expected_stage = "CE_control" if variant == "E2_CE_only" else "B"
            if metrics.get("stage") != expected_stage:
                issue(variant_issues, "provenance_mismatch", f"stage must be {expected_stage!r}", metrics_path)
            for field in STAGE_B_METRICS:
                values[field] = finite_number(metrics.get(field), field, metrics_path, variant_issues)
            provenance_value = metrics.get("checkpoint_provenance")
            if isinstance(provenance_value, dict):
                provenance = provenance_value
                if provenance.get("stage") != "B" or provenance.get("development_data_only") is not True:
                    issue(variant_issues, "provenance_mismatch", "Checkpoint is not development Stage B provenance", metrics_path)
            else:
                issue(variant_issues, "provenance_mismatch", "checkpoint_provenance is missing", metrics_path)
            completed = finite_number(
                metrics.get("training_completed_steps"), "training_completed_steps", metrics_path, variant_issues
            )
            if expected_steps is not None and completed != expected_steps:
                issue(variant_issues, "provenance_mismatch", "training_completed_steps differs from manifest", metrics_path)
        digest = validate_checkpoint(checkpoint_path, provenance, variant_issues)
        expected_complete = expected_steps is not None and observed_steps >= expected_steps
        terminal_present = metrics is not None and checkpoint_path.is_file()
        variant_status = status_for(bool(rows), terminal_present, expected_complete, variant_issues)
        variants[variant] = {
            "status": variant_status,
            "validation_passed": variant_status == "completed",
            "observed_steps": observed_steps,
            "expected_steps": expected_steps,
            "metrics": values,
            "checkpoint_sha256": digest,
            "issues": variant_issues,
        }
        issues.extend(variant_issues)

    comparison: dict[str, Any] | None = None
    ce = variants[STAGE_B_VARIANTS[0]]
    distilled = variants[STAGE_B_VARIANTS[1]]
    if ce["status"] == distilled["status"] == "completed":
        deltas = {
            field: float(distilled["metrics"][field]) - float(ce["metrics"][field])
            for field in STAGE_B_METRICS
        }
        comparison = {
            "delta_e2d_minus_ce": deltas,
            "matched": True,
            "preferred_by_perplexity": min(
                STAGE_B_VARIANTS, key=lambda name: float(variants[name]["metrics"]["perplexity"])
            ),
        }
    statuses = {variant["status"] for variant in variants.values()}
    if "rejected" in statuses or any(item["code"] == "non_real_data" for item in issues):
        run_status = "rejected"
    elif statuses == {"completed"}:
        run_status = "completed"
    elif "running" in statuses:
        run_status = "running"
    else:
        run_status = "incomplete"
    return {
        "stage": "B",
        "source_root": str(source_root),
        "run_id": run_root.name,
        "status": run_status,
        "validation_passed": run_status == "completed",
        "variants": variants,
        "comparison": comparison,
        "issues": issues,
    }


def validate_initialization(
    name: str,
    manifest: Mapping[str, Any],
    args: Mapping[str, Any],
    issues: list[dict[str, str]],
    path: Path,
) -> dict[str, Any]:
    value = manifest.get(name)
    if not isinstance(value, dict):
        issue(issues, "provenance_mismatch", f"{name} is missing", path)
        return {"path": None, "sha256": None, "policy": None}
    prefix = "stage_b" if name == "stage_b_initialization" else "multimodal_initial"
    arg_path = args.get(f"{prefix}_checkpoint")
    arg_sha = args.get(f"{prefix}_checkpoint_sha256")
    if value.get("path") != arg_path or value.get("sha256") != arg_sha:
        issue(issues, "provenance_mismatch", f"{name} disagrees with manifest args", path)
    if value.get("sealed_evidence_used") is not False or value.get("synthetic_evidence_used") is not False:
        issue(issues, "non_real_data", f"{name} does not explicitly exclude sealed/synthetic evidence", path)
    if not isinstance(value.get("sha256"), str) or len(value["sha256"]) != 64:
        issue(issues, "provenance_mismatch", f"{name} SHA256 is invalid", path)
    return {"path": value.get("path"), "sha256": value.get("sha256"), "policy": value.get("policy")}


def _resolved_recorded_path(recorded: Any, relative_to: Path) -> Path | None:
    if not isinstance(recorded, str) or not recorded:
        return None
    path = Path(recorded)
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve(strict=False)


def validate_multimodal_initial_manifest(
    manifest: Mapping[str, Any],
    args: Mapping[str, Any],
    issues: list[dict[str, str]],
    stage_c_manifest_path: Path,
) -> None:
    checkpoint_value = args.get("multimodal_initial_checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        return

    code = "unverified_multimodal_initial_manifest"

    def reject(message: str, evidence_path: Path = stage_c_manifest_path) -> None:
        issue(issues, code, message, evidence_path)

    initialization = manifest.get("multimodal_initialization")
    if not isinstance(initialization, dict):
        reject("multimodal_initialization is missing")
        return

    required_strings = (
        "manifest_path",
        "manifest_sha256",
        "path",
        "sha256",
        "source_commit_sha",
        "runai_job_name",
        "runai_project",
    )
    for field in required_strings:
        if not isinstance(initialization.get(field), str) or not initialization[field]:
            reject(f"multimodal_initialization.{field} is missing")
    source_commit = initialization.get("source_commit_sha")
    if not isinstance(source_commit, str) or COMMIT_SHA_PATTERN.fullmatch(source_commit) is None:
        reject("multimodal_initialization.source_commit_sha is not a full commit SHA")
    for field in ("manifest_sha256", "sha256"):
        value = initialization.get(field)
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            reject(f"multimodal_initialization.{field} is not a 64-hex SHA256")
    if initialization.get("completion_status") != "completed":
        reject("multimodal_initialization.completion_status is not completed")
    completion_step = initialization.get("completion_step")
    if isinstance(completion_step, bool) or not isinstance(completion_step, int) or completion_step < 0:
        reject("multimodal_initialization.completion_step is invalid")
    if initialization.get("sealed_evidence_used") is not False:
        reject("multimodal initialization does not explicitly exclude sealed evidence")
    if initialization.get("synthetic_evidence_used") is not False:
        reject("multimodal initialization does not explicitly exclude synthetic evidence")

    manifest_value = initialization.get("manifest_path")
    if manifest_value != args.get("multimodal_initial_manifest"):
        reject("companion manifest path disagrees with manifest args")
    if initialization.get("path") != checkpoint_value:
        reject("companion checkpoint path disagrees with manifest args")
    if initialization.get("sha256") != args.get("multimodal_initial_checkpoint_sha256"):
        reject("companion checkpoint SHA disagrees with manifest args")

    manifest_dir = stage_c_manifest_path.parent
    companion_path = _resolved_recorded_path(manifest_value, manifest_dir)
    checkpoint_path = _resolved_recorded_path(initialization.get("path"), manifest_dir)
    for label, path in (("companion manifest", companion_path), ("multimodal checkpoint", checkpoint_path)):
        if path is None:
            reject(f"{label} path is invalid")
        elif any(word in str(path).lower() for word in FORBIDDEN_PATH_WORDS):
            reject(f"{label} path contains sealed/synthetic provenance", path)
    if companion_path is None or not companion_path.is_file():
        reject("companion manifest does not exist", companion_path or stage_c_manifest_path)
        return

    expected_manifest_sha = initialization.get("manifest_sha256")
    actual_manifest_sha = sha256_file(companion_path)
    if expected_manifest_sha != actual_manifest_sha:
        reject(
            f"companion manifest SHA {expected_manifest_sha!r} != {actual_manifest_sha}",
            companion_path,
        )
        return
    try:
        companion = read_json(companion_path)
    except EvidenceError as error:
        reject(str(error), companion_path)
        return

    companion_completion = companion.get("completion")
    if not isinstance(companion_completion, dict):
        reject("companion completion is missing", companion_path)
        companion_completion = {}
    if companion_completion.get("status") != "completed":
        reject("companion completion.status is not completed", companion_path)
    if companion_completion.get("e3_steps") != completion_step:
        reject("companion completion step disagrees with Stage C initialization", companion_path)
    if companion_completion.get("e3_checkpoint_sha256") != initialization.get("sha256"):
        reject("companion checkpoint SHA disagrees with Stage C initialization", companion_path)
    companion_checkpoint = _resolved_recorded_path(
        companion_completion.get("e3_checkpoint_path"), companion_path.parent
    )
    if checkpoint_path is None or companion_checkpoint != checkpoint_path:
        reject("companion checkpoint path disagrees with Stage C initialization", companion_path)

    companion_provenance = companion.get("run_provenance")
    if not isinstance(companion_provenance, dict):
        reject("companion run_provenance is missing", companion_path)
        companion_provenance = {}
    for field in ("source_commit_sha", "runai_job_name", "runai_project"):
        expected = initialization.get(field)
        if companion.get(field) != expected or companion_provenance.get(field) != expected:
            reject(f"companion {field} disagrees with Stage C initialization", companion_path)
    for field in ("sealed_evidence_used", "synthetic_evidence_used"):
        if companion_provenance.get(field) is not False:
            reject(f"companion run_provenance.{field} is not explicitly false", companion_path)
    serialized = json.dumps(companion, sort_keys=True).lower()
    if '"sealed_evidence_used": true' in serialized or '"synthetic_evidence_used": true' in serialized:
        reject("companion manifest declares sealed or synthetic evidence use", companion_path)

    if checkpoint_path is None or not checkpoint_path.is_file():
        reject("multimodal checkpoint does not exist", checkpoint_path or companion_path)
    elif sha256_file(checkpoint_path) != initialization.get("sha256"):
        reject("multimodal checkpoint content does not match the recorded SHA", checkpoint_path)


def retrieval_metrics(metrics: Mapping[str, Any], path: Path, issues: list[dict[str, str]]) -> dict[str, Any]:
    retrieval = metrics.get("retrieval_eval")
    if not isinstance(retrieval, dict):
        issue(issues, "missing_retrieval_metrics", "retrieval_eval is missing", path)
        return {}
    required_path_flags = {
        "conditional_uses_lm_logits": True,
        "conditional_uses_direct_encoder_pooling": False,
        "retrieval_uses_lm_hidden_states": True,
        "retrieval_uses_direct_encoder_pooling": False,
    }
    for flag, expected in required_path_flags.items():
        if retrieval.get(flag) is not expected:
            issue(
                issues,
                "bypass_retrieval_path",
                f"{flag} must be {expected!r}; got {retrieval.get(flag)!r}",
                path,
            )
    mapping = {
        "conditional_image_r1": "conditional_image_to_text_r_at_1",
        "conditional_image_chance_r1": "conditional_image_chance_r_at_1",
        "conditional_speech_r1": "conditional_speech_to_text_r_at_1",
        "conditional_speech_chance_r1": "conditional_speech_chance_r_at_1",
        "embedding_image_r1": "image_to_text_r_at_1",
        "embedding_image_chance_r1": "image_chance_r_at_1",
        "embedding_speech_r1": "speech_to_text_r_at_1",
        "embedding_speech_chance_r1": "speech_chance_r_at_1",
    }
    return {
        output: finite_number(retrieval.get(source), source, path, issues)
        for output, source in mapping.items()
    }


def summarize_stage_c_run(source_root: Path, run_root: Path) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    manifest = load_manifest(run_root, issues)
    args = manifest.get("args", {}) if manifest else {}
    if not isinstance(args, dict):
        args = {}
    expected_steps = args.get("final_steps")
    expected_alignment = args.get("alignment_pretrain_steps")
    for value, label in ((expected_steps, "final_steps"), (expected_alignment, "alignment_pretrain_steps")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            issue(issues, "invalid_expected_steps", f"{label} must be a non-negative integer", run_root / "manifest.json")
    if not isinstance(expected_steps, int) or isinstance(expected_steps, bool):
        expected_steps = None
    if not isinstance(expected_alignment, int) or isinstance(expected_alignment, bool):
        expected_alignment = None

    stage_root = run_root / STAGE_C_DIR
    train_path = stage_root / "train_metrics.jsonl"
    alignment_path = stage_root / "alignment_pretrain_metrics.jsonl"
    try:
        train_rows = read_jsonl(train_path)
        observed_steps = validate_steps(
            train_rows, train_path, issues, ("loss", "capacity_overflow_ratio_mean", "inactive_expert_ratio_mean")
        )
    except EvidenceError as error:
        train_rows, observed_steps = [], 0
        issue(issues, "invalid_jsonl", str(error), train_path)
    for row_index, row in enumerate(train_rows, 1):
        modality = row.get("modality")
        if modality not in {"image", "speech"}:
            continue
        prefix_name = "image_prefix" if modality == "image" else "audio_prefix"
        expected = row.get("prefix_expected_assignments_tokens_x_layers_x_k")
        observed = row.get("prefix_observed_assignments")
        token_counts = row.get("modality_token_counts_across_layers")
        conservation = row.get("modality_assignment_conservation")
        prefix_valid = (
            row.get("prefix_routing_included") is True
            and row.get("modality_token_k_conservation_ok") is True
            and isinstance(expected, int)
            and expected > 0
            and observed == expected
            and isinstance(token_counts, dict)
            and isinstance(token_counts.get(prefix_name), int)
            and token_counts[prefix_name] > 0
            and isinstance(conservation, dict)
            and conservation.get(prefix_name) is True
        )
        if not prefix_valid:
            issue(
                issues,
                "missing_prefix_routing_accounting",
                f"{modality} row {row_index} lacks conserved {prefix_name} routing",
                train_path,
            )
    try:
        alignment_rows = read_jsonl(alignment_path)
        alignment_steps = validate_steps(alignment_rows, alignment_path, issues, ("loss",))
    except EvidenceError as error:
        alignment_rows, alignment_steps = [], 0
        issue(issues, "invalid_jsonl", str(error), alignment_path)

    stage_b_init = validate_initialization(
        "stage_b_initialization", manifest or {}, args, issues, run_root / "manifest.json"
    )
    multimodal_init = validate_initialization(
        "multimodal_initialization", manifest or {}, args, issues, run_root / "manifest.json"
    )
    validate_multimodal_initial_manifest(
        manifest or {}, args, issues, run_root / "manifest.json"
    )
    if train_rows:
        last = train_rows[-1]
        if last.get("source_stage_b_checkpoint_sha256") != stage_b_init["sha256"]:
            issue(issues, "provenance_mismatch", "Training Stage B source SHA disagrees with manifest", train_path)
        if last.get("source_selected_checkpoint_sha256") != multimodal_init["sha256"]:
            issue(issues, "provenance_mismatch", "Training multimodal source SHA disagrees with manifest", train_path)
        if last.get("stage_b_checkpoint_state_restored") is not True or last.get("initial_checkpoint_state_restored") is not True:
            issue(issues, "provenance_mismatch", "Training did not confirm initialization restoration", train_path)

    metrics_path = stage_root / "metrics.json"
    metrics: dict[str, Any] | None = None
    retrieval: dict[str, Any] = {}
    if metrics_path.is_file():
        try:
            metrics = read_json(metrics_path)
            if metrics.get("real_subset") is not True:
                issue(issues, "non_real_data", "Stage C metrics are not marked real_subset", metrics_path)
            retrieval = retrieval_metrics(metrics, metrics_path, issues)
        except EvidenceError as error:
            issue(issues, "invalid_metrics", str(error), metrics_path)

    checkpoint_path = stage_root / "checkpoint_final.pt"
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    if checkpoint_sha is None:
        issue(issues, "missing_checkpoint", "Final checkpoint is missing", checkpoint_path)

    text_path = run_root / STAGE_C_TEXT_DIR / "metrics.json"
    text_metrics: dict[str, Any] | None = None
    text_values: dict[str, Any] = {}
    if text_path.is_file():
        try:
            text_metrics = read_json(text_path)
            if text_metrics.get("real_subset") is not True:
                issue(issues, "non_real_data", "Text metrics are not marked real_subset", text_path)
            provenance = text_metrics.get("provenance")
            if not isinstance(provenance, dict):
                issue(issues, "provenance_mismatch", "Text evaluation provenance is missing", text_path)
            else:
                if provenance.get("copied_from_e2") is not False:
                    issue(issues, "copied_text_metrics", "Text metrics were copied or do not explicitly reject copying", text_path)
                if provenance.get("source_experiment_id") != STAGE_C_DIR:
                    issue(issues, "provenance_mismatch", "Text metrics are not sourced from E3", text_path)
                if expected_steps is not None and provenance.get("source_training_steps") != expected_steps:
                    issue(issues, "provenance_mismatch", "Text source training steps differ from final_steps", text_path)
                if checkpoint_path.is_file() and provenance.get("source_checkpoint_size_bytes") != checkpoint_path.stat().st_size:
                    issue(issues, "provenance_mismatch", "Text source checkpoint size differs from E3 checkpoint", text_path)
                source_checkpoint = provenance.get("source_checkpoint")
                if source_checkpoint is not None and not same_path(source_checkpoint, checkpoint_path):
                    issue(issues, "provenance_mismatch", "Text source checkpoint path differs from E3 checkpoint", text_path)
                source_sha = provenance.get("source_checkpoint_sha256")
                if source_sha is not None and source_sha != checkpoint_sha:
                    issue(issues, "checkpoint_sha_mismatch", "Text source checkpoint SHA differs from E3 checkpoint", text_path)
            for field in ("perplexity", "next_token_accuracy", "capacity_overflow_ratio_mean", "inactive_expert_ratio_mean"):
                text_values[field] = finite_number(text_metrics.get(field), field, text_path, issues)
        except EvidenceError as error:
            issue(issues, "invalid_metrics", str(error), text_path)

    progress = bool(train_rows or alignment_rows)
    expected_complete = (
        expected_steps is not None
        and observed_steps >= expected_steps
        and expected_alignment is not None
        and alignment_steps >= expected_alignment
    )
    terminal_present = metrics is not None and text_metrics is not None and checkpoint_path.is_file()
    run_status = status_for(progress, terminal_present, expected_complete, issues)
    return {
        "stage": "C",
        "source_root": str(source_root),
        "run_id": run_root.name,
        "status": run_status,
        "validation_passed": run_status == "completed",
        "observed_steps": observed_steps,
        "expected_steps": expected_steps,
        "alignment_steps": alignment_steps,
        "expected_alignment_steps": expected_alignment,
        "text_metrics": text_values,
        "retrieval_metrics": retrieval,
        "routing_metrics": {
            "capacity_overflow_ratio_mean": text_values.get("capacity_overflow_ratio_mean"),
            "inactive_expert_ratio_mean": text_values.get("inactive_expert_ratio_mean"),
        },
        "checkpoint_sha256": checkpoint_sha,
        "stage_b_initialization": stage_b_init,
        "multimodal_initialization": multimodal_init,
        "issues": issues,
    }


def run_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)


def csv_rows(stage_b_runs: Sequence[Mapping[str, Any]], stage_c_runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in stage_b_runs:
        for variant_name in STAGE_B_VARIANTS:
            variant = run["variants"][variant_name]
            row = {field: None for field in CSV_FIELDS}
            row.update(
                {
                    "stage": "B",
                    "source_root": run["source_root"],
                    "run_id": run["run_id"],
                    "variant": variant_name,
                    "status": variant["status"],
                    "validation_passed": variant["validation_passed"],
                    "observed_steps": variant["observed_steps"],
                    "expected_steps": variant["expected_steps"],
                    "checkpoint_sha256": variant["checkpoint_sha256"],
                    "issue_codes": ";".join(sorted({item["code"] for item in variant["issues"]})),
                }
            )
            row.update(variant["metrics"])
            rows.append(row)
    for run in stage_c_runs:
        row = {field: None for field in CSV_FIELDS}
        row.update(
            {
                "stage": "C",
                "source_root": run["source_root"],
                "run_id": run["run_id"],
                "variant": STAGE_C_DIR,
                "status": run["status"],
                "validation_passed": run["validation_passed"],
                "observed_steps": run["observed_steps"],
                "expected_steps": run["expected_steps"],
                "alignment_steps": run["alignment_steps"],
                "expected_alignment_steps": run["expected_alignment_steps"],
                "checkpoint_sha256": run["checkpoint_sha256"],
                "stage_b_initial_checkpoint_sha256": run["stage_b_initialization"]["sha256"],
                "multimodal_initial_checkpoint_sha256": run["multimodal_initialization"]["sha256"],
                "issue_codes": ";".join(sorted({item["code"] for item in run["issues"]})),
            }
        )
        row.update(run["text_metrics"])
        row.update(run["retrieval_metrics"])
        rows.append(row)
    return rows


def markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Stage B / Stage C Development Summary",
        "",
        f"Overall validation passed: **{str(payload['validation_passed']).lower()}**",
        "",
        "Missing values are rendered as `NA`; they are never replaced with zero.",
        "",
        "## Stage B",
        "",
        "| Root | Run | Status | CE PPL | E2D PPL | E2D-CE PPL |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run in payload["stage_b_runs"]:
        ce = run["variants"][STAGE_B_VARIANTS[0]]["metrics"].get("perplexity")
        e2d = run["variants"][STAGE_B_VARIANTS[1]]["metrics"].get("perplexity")
        delta = None if run["comparison"] is None else run["comparison"]["delta_e2d_minus_ce"]["perplexity"]
        lines.append(
            f"| {Path(run['source_root']).name} | {run['run_id']} | {run['status']} | {_fmt(ce)} | {_fmt(e2d)} | {_fmt(delta)} |"
        )
    lines.extend(
        [
            "",
            "## Stage C",
            "",
            "| Run | Status | Steps | Align | Text PPL | Cond image/chance | Cond speech/chance | Emb image/chance | Emb speech/chance |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in payload["stage_c_runs"]:
        retrieval = run["retrieval_metrics"]
        lines.append(
            "| {run} | {status} | {steps}/{expected} | {align}/{expected_align} | {ppl} | {ci}/{cic} | {cs}/{csc} | {ei}/{eic} | {es}/{esc} |".format(
                run=run["run_id"], status=run["status"], steps=run["observed_steps"],
                expected=_fmt(run["expected_steps"]), align=run["alignment_steps"],
                expected_align=_fmt(run["expected_alignment_steps"]), ppl=_fmt(run["text_metrics"].get("perplexity")),
                ci=_fmt(retrieval.get("conditional_image_r1")), cic=_fmt(retrieval.get("conditional_image_chance_r1")),
                cs=_fmt(retrieval.get("conditional_speech_r1")), csc=_fmt(retrieval.get("conditional_speech_chance_r1")),
                ei=_fmt(retrieval.get("embedding_image_r1")), eic=_fmt(retrieval.get("embedding_image_chance_r1")),
                es=_fmt(retrieval.get("embedding_speech_r1")), esc=_fmt(retrieval.get("embedding_speech_chance_r1")),
            )
        )
    all_runs = list(payload["stage_b_runs"]) + list(payload["stage_c_runs"])
    all_issues = [(run["run_id"], item) for run in all_runs for item in run["issues"]]
    lines.extend(["", "## Validation Issues", ""])
    if not all_issues:
        lines.append("None.")
    else:
        for run_id, item in all_issues:
            lines.append(f"- `{run_id}` `{item['code']}` ({item['severity']}): {item['message']}")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def summarize(stage_b_roots: Sequence[Path], stage_c_root: Path, output_dir: Path) -> dict[str, Any]:
    for index, root in enumerate(stage_b_roots, 1):
        reject_forbidden_path(root, f"Stage B root {index}")
    reject_forbidden_path(stage_c_root, "Stage C root")
    reject_forbidden_path(output_dir, "output path")

    stage_b_runs = [
        summarize_stage_b_run(root, run_root)
        for root in stage_b_roots
        for run_root in run_directories(root)
    ]
    stage_c_runs = [summarize_stage_c_run(stage_c_root, run_root) for run_root in run_directories(stage_c_root)]
    missing_roots = [str(root) for root in (*stage_b_roots, stage_c_root) if not root.is_dir()]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "development_only": True,
        "validation_passed": bool(stage_b_runs or stage_c_runs)
        and not missing_roots
        and all(run["validation_passed"] for run in (*stage_b_runs, *stage_c_runs)),
        "missing_roots": missing_roots,
        "stage_b_roots": [str(path) for path in stage_b_roots],
        "stage_c_root": str(stage_c_root),
        "stage_b_runs": stage_b_runs,
        "stage_c_runs": stage_c_runs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{OUTPUT_STEM}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    rows = csv_rows(stage_b_runs, stage_c_runs)
    with (output_dir / f"{OUTPUT_STEM}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f"{OUTPUT_STEM}.md").write_text(markdown(payload), encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stageb-root",
        action="append",
        type=Path,
        help="Repeat for each Stage B root; overrides legacy screen/followup defaults.",
    )
    parser.add_argument("--stageb-screen-root", type=Path, default=Path("outputs/development_stageb_screen_v1"))
    parser.add_argument("--stageb-followup-root", type=Path, default=Path("outputs/development_stageb_followup_v1"))
    parser.add_argument("--stagec-root", type=Path, default=Path("outputs/development_stagec_factorized_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stage_b_roots = (
        tuple(args.stageb_root)
        if args.stageb_root
        else (args.stageb_screen_root, args.stageb_followup_root)
    )
    payload = summarize(stage_b_roots, args.stagec_root, args.output_dir)
    print(json.dumps({"validation_passed": payload["validation_passed"], "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
