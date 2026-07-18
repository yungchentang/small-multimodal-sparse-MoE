#!/usr/bin/env python3
"""Fail-closed evidence summary for the two-arm 3k MM dual promotion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence


E3_DIR = "E3_final_multimodal_top2"
E3_TEXT_DIR = "E3_final_multimodal_top2_text_eval"
OUTPUT_STEM = "mm_dual_promotion_summary"
MAIN_STEPS = 3000
ALIGNMENT_STEPS = 400
INITIALIZER_STEPS = 500
LOSS_WINDOW_RELATIVE_TOLERANCE = 0.05
EXPECTED_SOURCE_COMMIT = "4170c467d50c7dbda066e2cb3e199fb1602fdf9c"
FORBIDDEN_WORDS = ("sealed", "synthetic")

ARM_SPECS: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    (
        (
            "c_dual_seed42",
            {
                "arm": "C_DUAL",
                "kd_coefficient": 0.0,
                "runai_job_name": "sme-c-dual-s42-promote3k-260711a",
            },
        ),
        (
            "c_dual_kd025_seed42",
            {
                "arm": "C_DUAL_KD025",
                "kd_coefficient": 0.25,
                "runai_job_name": "sme-c-dual-kd025-s42-promote3k-260711a",
            },
        ),
    )
)

ABSOLUTE_METRICS = (
    "perplexity",
    "next_token_accuracy",
    "conditional_image_r1",
    "conditional_speech_r1",
    "embedding_image_r1",
    "embedding_speech_r1",
    "first_loss",
    "last_loss",
    "final_gate_entropy_mean",
    "final_inactive_expert_ratio_mean",
    "final_capacity_overflow_ratio_mean",
    "final_prefix_expected_assignments",
    "final_prefix_observed_assignments",
)


class EvidenceError(ValueError):
    """An artifact cannot be accepted as promotion evidence."""


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON constant {value!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _issue(
    issues: list[dict[str, str]], code: str, message: str, path: Path
) -> None:
    issues.append({"code": code, "message": message, "path": str(path)})


def _forbidden(value: Any) -> bool:
    lowered = str(value).lower()
    return any(word in lowered for word in FORBIDDEN_WORDS)


def _regular_file(path: Path, issues: list[dict[str, str]], label: str) -> bool:
    if not path.is_file() or path.is_symlink():
        _issue(issues, "missing_artifact", f"{label} is missing or unsafe", path)
        return False
    return True


def _load_json(
    path: Path, issues: list[dict[str, str]], label: str
) -> dict[str, Any] | None:
    if not _regular_file(path, issues, label):
        return None
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, EvidenceError) as error:
        _issue(issues, "invalid_json", f"{label}: {error}", path)
        return None
    if not isinstance(payload, dict):
        _issue(issues, "invalid_json", f"{label} must be a JSON object", path)
        return None
    return payload


def _load_jsonl(
    path: Path, issues: list[dict[str, str]], label: str
) -> list[dict[str, Any]]:
    if not _regular_file(path, issues, label):
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise EvidenceError(f"blank line at {line_number}")
                row = json.loads(line, parse_constant=_reject_constant)
                if not isinstance(row, dict):
                    raise EvidenceError(f"row {line_number} is not an object")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError, EvidenceError) as error:
        _issue(issues, "invalid_jsonl", f"{label}: {error}", path)
        return []
    return rows


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    return True


def _same_number(left: Any, right: Any) -> bool:
    return _finite(left) and _finite(right) and math.isclose(
        float(left), float(right), rel_tol=1e-12, abs_tol=1e-12
    )


def _recorded_path(value: Any, relative_to: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve(strict=False)


def _path_matches(value: Any, actual: Path, relative_to: Path) -> bool:
    recorded = _recorded_path(value, relative_to)
    return recorded == actual.resolve(strict=False) if recorded is not None else False


def _require_equal(
    issues: list[dict[str, str]],
    actual: Any,
    expected: Any,
    label: str,
    path: Path,
    code: str = "provenance_mismatch",
) -> None:
    if actual != expected:
        _issue(issues, code, f"{label}: expected {expected!r}, got {actual!r}", path)


def _validate_external_file(
    metadata: Mapping[str, Any],
    path_key: str,
    sha_key: str,
    size_key: str | None,
    issues: list[dict[str, str]],
    evidence_path: Path,
    label: str,
) -> Path | None:
    value = metadata.get(path_key)
    path = _recorded_path(value, evidence_path.parent)
    if path is None or _forbidden(value):
        _issue(
            issues,
            "invalid_evidence_path",
            f"{label} path is missing or forbidden",
            evidence_path,
        )
        return None
    if not _regular_file(path, issues, label):
        return path
    actual_sha = _sha256(path)
    _require_equal(
        issues,
        metadata.get(sha_key),
        actual_sha,
        f"{label} SHA-256",
        path,
        "checkpoint_sha_mismatch",
    )
    if size_key is not None:
        _require_equal(
            issues,
            metadata.get(size_key),
            path.stat().st_size,
            f"{label} size",
            path,
            "checkpoint_size_mismatch",
        )
    return path


def _validate_run_identity(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    run_root: Path,
    spec: Mapping[str, Any],
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    provenance = manifest.get("run_provenance")
    if not isinstance(provenance, dict):
        _issue(issues, "missing_provenance", "run_provenance is missing", manifest_path)
        return {}
    expected = {
        "source_commit_sha": EXPECTED_SOURCE_COMMIT,
        "runai_job_name": spec["runai_job_name"],
    }
    for field, value in expected.items():
        _require_equal(issues, provenance.get(field), value, field, manifest_path)
        _require_equal(issues, manifest.get(field), value, field, manifest_path)
    project = provenance.get("runai_project")
    if not isinstance(project, str) or not project.strip():
        _issue(issues, "missing_provenance", "runai_project is missing", manifest_path)
    _require_equal(issues, manifest.get("runai_project"), project, "runai_project", manifest_path)
    for field, expected_value in (
        ("final_main_steps", MAIN_STEPS),
        ("alignment_pretrain_steps", ALIGNMENT_STEPS),
        ("checkpoint_completed_step", MAIN_STEPS),
    ):
        _require_equal(
            issues, provenance.get(field), expected_value, field, manifest_path
        )
    for field in ("sealed_evidence_used", "synthetic_evidence_used"):
        _require_equal(
            issues,
            provenance.get(field),
            False,
            field,
            manifest_path,
            "non_real_data",
        )
    if not _path_matches(
        provenance.get("resolved_output_root"), run_root, manifest_path.parent
    ):
        _issue(
            issues,
            "provenance_mismatch",
            "resolved_output_root does not match arm root",
            manifest_path,
        )
    return provenance


def _validate_real_data(
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    manifest_path: Path,
    issues: list[dict[str, str]],
) -> None:
    args = manifest.get("args") if isinstance(manifest.get("args"), dict) else {}
    values = (
        manifest.get("data_dir"),
        args.get("data_dir"),
        provenance.get("resolved_data_root"),
    )
    if any(
        not isinstance(value, str) or not value or _forbidden(value)
        for value in values
    ):
        _issue(
            issues,
            "non_real_data",
            "data roots are missing or contain sealed/synthetic",
            manifest_path,
        )
        return
    resolved = [_recorded_path(value, manifest_path.parent) for value in values]
    if resolved[0] is None or any(path != resolved[0] for path in resolved):
        _issue(issues, "provenance_mismatch", "data roots disagree", manifest_path)
        return
    data_root = resolved[0]
    assert data_root is not None
    if not data_root.is_dir() or data_root.is_symlink():
        _issue(issues, "non_real_data", "real data root is missing or unsafe", data_root)
        return
    data_manifest = _load_json(
        data_root / "manifest.json", issues, "real data manifest"
    )
    if data_manifest is None or manifest.get("data_manifest") != data_manifest:
        _issue(
            issues,
            "non_real_data",
            "embedded data manifest does not match real data manifest",
            manifest_path,
        )
    serialized = json.dumps(manifest, sort_keys=True).lower()
    for field in ("sealed_evidence_used", "synthetic_evidence_used"):
        if f'"{field}": true' in serialized:
            _issue(
                issues,
                "non_real_data",
                f"manifest declares {field}",
                manifest_path,
            )


def _validate_initializer(
    initialization: Mapping[str, Any],
    args: Mapping[str, Any],
    prefix: str,
    expected_scope: str,
    manifest_path: Path,
    issues: list[dict[str, str]],
) -> str | None:
    for field in ("sealed_evidence_used", "synthetic_evidence_used"):
        _require_equal(
            issues,
            initialization.get(field),
            False,
            f"{prefix}.{field}",
            manifest_path,
            "non_real_data",
        )
    _require_equal(
        issues,
        initialization.get("scope"),
        expected_scope,
        f"{prefix}.scope",
        manifest_path,
        "wrong_initializer_scope",
    )
    _require_equal(
        issues,
        initialization.get("completion_status"),
        "completed",
        f"{prefix}.completion_status",
        manifest_path,
    )
    _require_equal(
        issues,
        initialization.get("completion_step"),
        INITIALIZER_STEPS,
        f"{prefix}.completion_step",
        manifest_path,
    )
    arg_names = {
        "multimodal_initialization": (
            "multimodal_initial_checkpoint",
            "multimodal_initial_checkpoint_sha256",
            "multimodal_initial_manifest",
        ),
        "speech_initialization": (
            "speech_initial_checkpoint",
            "speech_initial_checkpoint_sha256",
            "speech_initial_manifest",
        ),
    }
    checkpoint_arg, sha_arg, manifest_arg = arg_names[prefix]
    for arg_name, metadata_name in (
        (checkpoint_arg, "path"),
        (sha_arg, "sha256"),
        (manifest_arg, "manifest_path"),
    ):
        _require_equal(
            issues,
            args.get(arg_name),
            initialization.get(metadata_name),
            arg_name,
            manifest_path,
        )
    checkpoint = _validate_external_file(
        initialization,
        "path",
        "sha256",
        "size_bytes",
        issues,
        manifest_path,
        f"{prefix} checkpoint",
    )
    companion_path = _validate_external_file(
        initialization,
        "manifest_path",
        "manifest_sha256",
        None,
        issues,
        manifest_path,
        f"{prefix} manifest",
    )
    if companion_path is not None:
        companion = _load_json(companion_path, issues, f"{prefix} manifest")
        if companion is not None:
            completion = (
                companion.get("completion")
                if isinstance(companion.get("completion"), dict)
                else {}
            )
            companion_args = (
                companion.get("args")
                if isinstance(companion.get("args"), dict)
                else {}
            )
            companion_provenance = (
                companion.get("run_provenance")
                if isinstance(companion.get("run_provenance"), dict)
                else {}
            )
            _require_equal(
                issues,
                completion.get("status"),
                "completed",
                f"{prefix} companion status",
                companion_path,
            )
            _require_equal(
                issues,
                completion.get("e3_steps"),
                INITIALIZER_STEPS,
                f"{prefix} companion steps",
                companion_path,
            )
            if checkpoint is not None and not _path_matches(
                completion.get("e3_checkpoint_path"),
                checkpoint,
                companion_path.parent,
            ):
                _issue(
                    issues,
                    "invalid_initializer",
                    f"{prefix} companion checkpoint path mismatch",
                    companion_path,
                )
            _require_equal(
                issues,
                completion.get("e3_checkpoint_sha256"),
                initialization.get("sha256"),
                f"{prefix} companion SHA",
                companion_path,
            )
            _require_equal(
                issues,
                companion_args.get("final_steps"),
                INITIALIZER_STEPS,
                f"{prefix} companion final_steps",
                companion_path,
            )
            _require_equal(
                issues,
                companion_args.get("alignment_pretrain_steps"),
                ALIGNMENT_STEPS,
                f"{prefix} companion alignment steps",
                companion_path,
            )
            _require_equal(
                issues,
                companion_args.get("alignment_pretrain_modalities"),
                expected_scope,
                f"{prefix} companion scope",
                companion_path,
                "wrong_initializer_scope",
            )
            for field in ("source_commit_sha", "runai_job_name", "runai_project"):
                _require_equal(
                    issues,
                    companion_provenance.get(field),
                    initialization.get(field),
                    f"{prefix} companion {field}",
                    companion_path,
                )
            for field in ("sealed_evidence_used", "synthetic_evidence_used"):
                _require_equal(
                    issues,
                    companion_provenance.get(field),
                    False,
                    f"{prefix} companion {field}",
                    companion_path,
                    "non_real_data",
                )
    sha = initialization.get("sha256")
    return sha if isinstance(sha, str) else None


def _validate_initializers(
    manifest: Mapping[str, Any], manifest_path: Path, issues: list[dict[str, str]]
) -> dict[str, str | None]:
    args = manifest.get("args") if isinstance(manifest.get("args"), dict) else {}
    _require_equal(
        issues,
        args.get("image_bridge_type"),
        "linear_projector_norm",
        "image_bridge_type",
        manifest_path,
        "wrong_image_initializer",
    )
    _require_equal(
        issues,
        args.get("multimodal_initialization_scope"),
        "image",
        "multimodal_initialization_scope",
        manifest_path,
        "wrong_initializer_scope",
    )
    shas: dict[str, str | None] = {}
    for key, scope in (
        ("multimodal_initialization", "image"),
        ("speech_initialization", "speech"),
    ):
        value = manifest.get(key)
        if not isinstance(value, dict) or not value:
            _issue(issues, "invalid_initializer", f"{key} is missing", manifest_path)
            shas[key] = None
        else:
            shas[key] = _validate_initializer(
                value, args, key, scope, manifest_path, issues
            )
    stage_b = manifest.get("stage_b_initialization")
    if not isinstance(stage_b, dict) or not stage_b:
        _issue(issues, "missing_stage_b", "stage_b_initialization is missing", manifest_path)
        shas["stage_b_initialization"] = None
    else:
        for field in ("sealed_evidence_used", "synthetic_evidence_used"):
            _require_equal(
                issues,
                stage_b.get(field),
                False,
                f"stage_b.{field}",
                manifest_path,
                "non_real_data",
            )
        _validate_external_file(
            stage_b,
            "path",
            "sha256",
            "size_bytes",
            issues,
            manifest_path,
            "Stage B checkpoint",
        )
        _require_equal(
            issues,
            args.get("stage_b_checkpoint"),
            stage_b.get("path"),
            "stage_b_checkpoint",
            manifest_path,
        )
        _require_equal(
            issues,
            args.get("stage_b_checkpoint_sha256"),
            stage_b.get("sha256"),
            "stage_b_checkpoint_sha256",
            manifest_path,
        )
        sha = stage_b.get("sha256")
        shas["stage_b_initialization"] = sha if isinstance(sha, str) else None
    return shas


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    alignment_rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    train_path: Path,
    alignment_path: Path,
    initializer_shas: Mapping[str, str | None],
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    if len(rows) != MAIN_STEPS:
        _issue(
            issues,
            "wrong_main_steps",
            f"requires exactly {MAIN_STEPS} train rows; got {len(rows)}",
            train_path,
        )
    if [row.get("step") for row in rows] != list(range(1, MAIN_STEPS + 1)):
        _issue(
            issues,
            "wrong_main_steps",
            f"train steps must be exactly ordered 1..{MAIN_STEPS}",
            train_path,
        )
    if len(alignment_rows) != ALIGNMENT_STEPS:
        _issue(
            issues,
            "wrong_alignment_steps",
            f"requires exactly {ALIGNMENT_STEPS} alignment rows; got {len(alignment_rows)}",
            alignment_path,
        )
    if [row.get("step") for row in alignment_rows] != list(
        range(1, ALIGNMENT_STEPS + 1)
    ):
        _issue(
            issues,
            "wrong_alignment_steps",
            f"alignment steps must be exactly ordered 1..{ALIGNMENT_STEPS}",
            alignment_path,
        )
    if any(not _all_numbers_finite(row) for row in rows):
        _issue(
            issues,
            "non_finite_loss",
            "train rows contain non-finite numeric values",
            train_path,
        )

    ce_by_modality: dict[str, list[float]] = {"text": [], "speech": []}
    image_sha = initializer_shas.get("multimodal_initialization")
    speech_sha = initializer_shas.get("speech_initialization")
    stage_b_sha = initializer_shas.get("stage_b_initialization")
    for index, row in enumerate(rows, 1):
        for field in (
            "loss",
            "lm_ce_loss",
            "gate_entropy_mean",
            "inactive_expert_ratio_mean",
            "capacity_overflow_ratio_mean",
        ):
            if not _finite(row.get(field)):
                _issue(
                    issues,
                    "non_finite_loss",
                    f"row {index} {field} is not finite numeric",
                    train_path,
                )
        expected_modality = ("text", "speech", "speech")[(index - 1) % 3]
        modality = row.get("modality")
        if modality != expected_modality:
            _issue(
                issues,
                "wrong_main_steps",
                f"row {index} modality must follow text,speech,speech cycle",
                train_path,
            )
        expected_optimizer_step = modality == "speech"
        if row.get("optimizer_step") is not expected_optimizer_step:
            _issue(
                issues,
                "wrong_optimizer_step",
                f"row {index} optimizer_step must be {expected_optimizer_step}",
                train_path,
            )
        for field, expected in (
            ("native_top_k", 8),
            ("runtime_top_k", 2),
            ("top_k", 2),
        ):
            _require_equal(
                issues,
                row.get(field),
                expected,
                f"row {index} {field}",
                train_path,
                "wrong_runtime_top_k",
            )
        if modality in ce_by_modality and _finite(row.get("lm_ce_loss")):
            ce_by_modality[str(modality)].append(float(row["lm_ce_loss"]))
        for field, expected in (
            ("source_selected_checkpoint_sha256", image_sha),
            ("source_speech_initial_checkpoint_sha256", speech_sha),
            ("source_stage_b_checkpoint_sha256", stage_b_sha),
        ):
            _require_equal(
                issues, row.get(field), expected, f"row {index} {field}", train_path
            )
        for field in (
            "initial_checkpoint_state_restored",
            "speech_initial_checkpoint_state_restored",
            "stage_b_checkpoint_state_restored",
        ):
            _require_equal(
                issues, row.get(field), True, f"row {index} {field}", train_path
            )
        if not _same_number(
            row.get("speech_behavior_kl_coef"), spec["kd_coefficient"]
        ):
            _issue(
                issues,
                "wrong_kd_coefficient",
                f"row {index} KD coefficient mismatch",
                train_path,
            )
        if modality == "speech":
            counts = row.get("modality_token_counts_across_layers")
            conservation = row.get("modality_assignment_conservation")
            expected_prefix = row.get(
                "prefix_expected_assignments_tokens_x_layers_x_k"
            )
            observed_prefix = row.get("prefix_observed_assignments")
            valid_prefix = (
                row.get("prefix_routing_included") is True
                and row.get("modality_token_k_conservation_ok") is True
                and isinstance(counts, dict)
                and isinstance(counts.get("audio_prefix"), int)
                and counts["audio_prefix"] > 0
                and isinstance(conservation, dict)
                and conservation.get("audio_prefix") is True
                and isinstance(row.get("modality_routing_denominator"), str)
                and bool(row["modality_routing_denominator"].strip())
                and isinstance(expected_prefix, int)
                and expected_prefix > 0
                and expected_prefix == observed_prefix
            )
            if not valid_prefix:
                _issue(
                    issues,
                    "missing_prefix_path_flags",
                    f"row {index} lacks audio-token prefix routing evidence",
                    train_path,
                )
    for modality, values in ce_by_modality.items():
        if len(values) < 20:
            _issue(
                issues,
                "divergent_loss",
                f"{modality} has too few CE observations",
                train_path,
            )
            continue
        width = max(3, int(math.ceil(len(values) * 0.05)))
        first_mean = sum(values[:width]) / width
        last_mean = sum(values[-width:]) / width
        if last_mean > first_mean * (1.0 + LOSS_WINDOW_RELATIVE_TOLERANCE):
            _issue(
                issues,
                "divergent_loss",
                f"{modality} final 5% CE window exceeds initial window by more than "
                f"{LOSS_WINDOW_RELATIVE_TOLERANCE:.0%}",
                train_path,
            )
    for index, row in enumerate(alignment_rows, 1):
        if (
            row.get("modality") != "speech"
            or not _finite(row.get("loss"))
            or not _all_numbers_finite(row)
        ):
            _issue(
                issues,
                "invalid_alignment",
                f"alignment row {index} must be finite speech evidence",
                alignment_path,
            )
    last = rows[-1] if rows else {}
    return {
        "first_loss": rows[0].get("loss") if rows else None,
        "last_loss": last.get("loss"),
        "final_gate_entropy_mean": last.get("gate_entropy_mean"),
        "final_inactive_expert_ratio_mean": last.get(
            "inactive_expert_ratio_mean"
        ),
        "final_capacity_overflow_ratio_mean": last.get(
            "capacity_overflow_ratio_mean"
        ),
        "final_prefix_expected_assignments": last.get(
            "prefix_expected_assignments_tokens_x_layers_x_k"
        ),
        "final_prefix_observed_assignments": last.get(
            "prefix_observed_assignments"
        ),
    }


def _validate_final_evidence(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any] | None,
    text_metrics: Mapping[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    manifest_path: Path,
    metrics_path: Path,
    text_path: Path,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    values = {metric: None for metric in ABSOLUTE_METRICS}
    if not _regular_file(checkpoint, issues, "final checkpoint"):
        return values
    actual_sha = _sha256(checkpoint)
    actual_size = checkpoint.stat().st_size
    completion = (
        manifest.get("completion")
        if isinstance(manifest.get("completion"), dict)
        else {}
    )
    _require_equal(
        issues, completion.get("status"), "completed", "completion.status", manifest_path
    )
    _require_equal(
        issues, completion.get("e3_steps"), MAIN_STEPS, "completion.e3_steps", manifest_path
    )
    for payload, path, path_key, sha_key, size_key, label in (
        (
            completion,
            manifest_path,
            "e3_checkpoint_path",
            "e3_checkpoint_sha256",
            "e3_checkpoint_size_bytes",
            "manifest completion",
        ),
        (
            metrics or {},
            metrics_path,
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_size_bytes",
            "E3 metrics",
        ),
    ):
        if not _path_matches(payload.get(path_key), checkpoint, path.parent):
            _issue(
                issues,
                "checkpoint_path_mismatch",
                f"{label} checkpoint path mismatch",
                path,
            )
        _require_equal(
            issues,
            payload.get(sha_key),
            actual_sha,
            f"{label} checkpoint SHA",
            path,
            "checkpoint_sha_mismatch",
        )
        _require_equal(
            issues,
            payload.get(size_key),
            actual_size,
            f"{label} checkpoint size",
            path,
            "checkpoint_size_mismatch",
        )
    if metrics is None or text_metrics is None:
        return values
    _require_equal(
        issues, metrics.get("real_subset"), True, "E3 real_subset", metrics_path, "non_real_data"
    )
    _require_equal(
        issues,
        text_metrics.get("real_subset"),
        True,
        "text eval real_subset",
        text_path,
        "non_real_data",
    )
    if metrics.get("steps") != list(rows):
        _issue(
            issues,
            "metrics_ledger_mismatch",
            "E3 embedded steps differ from train_metrics.jsonl",
            metrics_path,
        )
    for field, expected in (
        ("first_loss", rows[0].get("loss") if rows else None),
        ("last_loss", rows[-1].get("loss") if rows else None),
        (
            "min_loss",
            min(
                (float(row["loss"]) for row in rows if _finite(row.get("loss"))),
                default=None,
            ),
        ),
    ):
        if not _same_number(metrics.get(field), expected):
            _issue(
                issues,
                "metrics_ledger_mismatch",
                f"E3 {field} differs from raw ledger",
                metrics_path,
            )

    provenance = (
        text_metrics.get("provenance")
        if isinstance(text_metrics.get("provenance"), dict)
        else {}
    )
    embedded_provenance = metrics.get("text_eval_provenance")
    if provenance != embedded_provenance:
        _issue(
            issues,
            "copied_text_metrics",
            "text provenance differs from E3 embedded provenance",
            text_path,
        )
    embedded_text = (
        metrics.get("text_eval") if isinstance(metrics.get("text_eval"), dict) else {}
    )
    for field in ("perplexity", "next_token_accuracy", "provenance"):
        if embedded_text.get(field) != text_metrics.get(field):
            _issue(
                issues,
                "copied_text_metrics",
                f"embedded text {field} differs from own text eval",
                text_path,
            )
    for field, expected in {
        "source_experiment_id": E3_DIR,
        "source_checkpoint_sha256": actual_sha,
        "source_checkpoint_size_bytes": actual_size,
        "source_training_steps": MAIN_STEPS,
        "source_checkpoint_saved_before_eval": True,
        "copied_from_e2": False,
    }.items():
        _require_equal(
            issues,
            provenance.get(field),
            expected,
            f"text provenance {field}",
            text_path,
            "copied_text_metrics",
        )
    if not _path_matches(
        provenance.get("source_checkpoint"), checkpoint, text_path.parent
    ):
        _issue(
            issues,
            "copied_text_metrics",
            "text metrics do not point to own checkpoint",
            text_path,
        )
    for field in ("perplexity", "next_token_accuracy"):
        value = text_metrics.get(field)
        if not _finite(value):
            _issue(
                issues,
                "invalid_quality_metric",
                f"{field} is not finite numeric",
                text_path,
            )
        else:
            values[field] = value

    retrieval = (
        metrics.get("retrieval_eval")
        if isinstance(metrics.get("retrieval_eval"), dict)
        else {}
    )
    for field, expected in {
        "retrieval_path": "shared_olmoe_prefix_hidden",
        "retrieval_uses_lm_hidden_states": True,
        "retrieval_uses_direct_encoder_pooling": False,
        "conditional_eval_path": "shared_olmoe_prefix_lm_nll",
        "conditional_uses_lm_logits": True,
        "conditional_uses_direct_encoder_pooling": False,
    }.items():
        _require_equal(
            issues,
            retrieval.get(field),
            expected,
            f"retrieval.{field}",
            metrics_path,
            "missing_prefix_path_flags",
        )
    for output_name, source_name in {
        "conditional_image_r1": "conditional_image_to_text_r_at_1",
        "conditional_speech_r1": "conditional_speech_to_text_r_at_1",
        "embedding_image_r1": "image_to_text_r_at_1",
        "embedding_speech_r1": "speech_to_text_r_at_1",
    }.items():
        value = retrieval.get(source_name)
        if not _finite(value):
            _issue(
                issues,
                "invalid_quality_metric",
                f"retrieval.{source_name} is not finite numeric",
                metrics_path,
            )
        else:
            values[output_name] = value
    for field in (
        "image_eval_count",
        "speech_eval_count",
        "conditional_image_eval_count",
        "conditional_speech_eval_count",
    ):
        if not isinstance(retrieval.get(field), int) or retrieval[field] <= 0:
            _issue(
                issues,
                "invalid_quality_metric",
                f"retrieval.{field} must be positive",
                metrics_path,
            )
    last = rows[-1] if rows else {}
    for metric_field, row_field in (
        ("final_gate_entropy_mean", "gate_entropy_mean"),
        ("final_inactive_expert_ratio_mean", "inactive_expert_ratio_mean"),
        ("final_capacity_overflow_ratio_mean", "capacity_overflow_ratio_mean"),
    ):
        if not _same_number(metrics.get(metric_field), last.get(row_field)):
            _issue(
                issues,
                "routing_metric_mismatch",
                f"{metric_field} differs from final train row",
                metrics_path,
            )
        if _finite(last.get(row_field)):
            values[metric_field] = last.get(row_field)
    values["first_loss"] = (
        rows[0].get("loss") if rows and _finite(rows[0].get("loss")) else None
    )
    values["last_loss"] = last.get("loss") if _finite(last.get("loss")) else None
    values["final_prefix_expected_assignments"] = last.get(
        "prefix_expected_assignments_tokens_x_layers_x_k"
    )
    values["final_prefix_observed_assignments"] = last.get(
        "prefix_observed_assignments"
    )
    values["checkpoint_sha256"] = actual_sha
    values["checkpoint_size_bytes"] = actual_size
    return values


def _summarize_arm(
    root: Path, name: str, spec: Mapping[str, Any]
) -> dict[str, Any]:
    run_root = root / name
    issues: list[dict[str, str]] = []
    result: dict[str, Any] = {
        "name": name,
        "arm": spec["arm"],
        "run_root": str(run_root.resolve(strict=False)),
        "validation_passed": False,
        "issues": issues,
        "expected": dict(spec),
        "source": {},
        "initialization": {},
        "observed_main_rows": 0,
        "observed_alignment_rows": 0,
        "metrics": {metric: None for metric in ABSOLUTE_METRICS},
        "deltas_vs_c_dual": {metric: None for metric in ABSOLUTE_METRICS},
    }
    if not run_root.is_dir() or run_root.is_symlink() or _forbidden(run_root):
        _issue(
            issues,
            "missing_arm",
            "arm directory is missing, unsafe, or forbidden",
            run_root,
        )
        return result
    manifest_path = run_root / "manifest.json"
    train_path = run_root / E3_DIR / "train_metrics.jsonl"
    alignment_path = run_root / E3_DIR / "alignment_pretrain_metrics.jsonl"
    metrics_path = run_root / E3_DIR / "metrics.json"
    checkpoint_path = run_root / E3_DIR / "checkpoint_final.pt"
    text_path = run_root / E3_TEXT_DIR / "metrics.json"
    manifest = _load_json(manifest_path, issues, "run manifest")
    rows = _load_jsonl(train_path, issues, "train metrics")
    alignment_rows = _load_jsonl(alignment_path, issues, "alignment metrics")
    metrics = _load_json(metrics_path, issues, "E3 metrics")
    text_metrics = _load_json(text_path, issues, "E3 text metrics")
    result["observed_main_rows"] = len(rows)
    result["observed_alignment_rows"] = len(alignment_rows)
    if manifest is None:
        return result
    args = manifest.get("args")
    if not isinstance(args, dict):
        _issue(issues, "invalid_manifest", "manifest.args is missing", manifest_path)
        args = {}
    for field, expected in (
        ("final_steps", MAIN_STEPS),
        ("alignment_pretrain_steps", ALIGNMENT_STEPS),
        ("alignment_pretrain_modalities", "speech"),
        ("modality_cycle", "text,speech,speech"),
        ("seed", 42),
    ):
        _require_equal(
            issues, args.get(field), expected, f"args.{field}", manifest_path
        )
    if not _same_number(
        args.get("speech_behavior_kl_coef"), spec["kd_coefficient"]
    ):
        _issue(
            issues,
            "wrong_kd_coefficient",
            "manifest KD coefficient mismatch",
            manifest_path,
        )
    if not _same_number(args.get("speech_behavior_kl_temperature"), 1.0):
        _issue(
            issues,
            "wrong_kd_coefficient",
            "manifest KD temperature must be 1",
            manifest_path,
        )
    provenance = _validate_run_identity(
        manifest, manifest_path, run_root, spec, issues
    )
    _validate_real_data(manifest, provenance, manifest_path, issues)
    initializer_shas = _validate_initializers(manifest, manifest_path, issues)
    row_metrics = _validate_rows(
        rows,
        alignment_rows,
        spec,
        train_path,
        alignment_path,
        initializer_shas,
        issues,
    )
    final_metrics = _validate_final_evidence(
        manifest,
        metrics,
        text_metrics,
        rows,
        checkpoint_path,
        manifest_path,
        metrics_path,
        text_path,
        issues,
    )
    final_metrics.update(
        {key: value for key, value in row_metrics.items() if final_metrics.get(key) is None}
    )
    result["metrics"] = final_metrics
    result["source"] = {
        "source_commit_sha": provenance.get("source_commit_sha"),
        "runai_job_name": provenance.get("runai_job_name"),
        "runai_project": provenance.get("runai_project"),
    }
    result["initialization"] = {
        "stage_b_checkpoint_sha256": initializer_shas.get("stage_b_initialization"),
        "image_initializer_sha256": initializer_shas.get("multimodal_initialization"),
        "speech_initializer_sha256": initializer_shas.get("speech_initialization"),
    }
    result["validation_passed"] = not issues
    return result


def _cross_arm_checks(arms: list[dict[str, Any]]) -> None:
    def add(arm: dict[str, Any], code: str, message: str) -> None:
        _issue(arm["issues"], code, message, Path(arm["run_root"]))

    for field in (
        "stage_b_checkpoint_sha256",
        "image_initializer_sha256",
        "speech_initializer_sha256",
    ):
        values = [arm["initialization"].get(field) for arm in arms]
        if any(not value for value in values) or len(set(values)) != 1:
            for arm in arms:
                add(
                    arm,
                    "initializer_mismatch",
                    f"both arms must share one non-empty {field}",
                )
    projects = [arm["source"].get("runai_project") for arm in arms]
    if any(not value for value in projects) or len(set(projects)) != 1:
        for arm in arms:
            add(arm, "source_mismatch", "both arms must share one Run:AI project")
    checkpoint_hashes = [arm["metrics"].get("checkpoint_sha256") for arm in arms]
    if any(not value for value in checkpoint_hashes):
        for arm in arms:
            add(arm, "missing_checkpoint_hash", "both final checkpoint SHAs are required")
    elif len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        for arm in arms:
            add(
                arm,
                "copied_checkpoint_hash",
                "final checkpoint SHA-256 must be unique across arms",
            )
    for arm in arms:
        arm["validation_passed"] = not arm["issues"]


def _delta(value: Any, baseline: Any) -> float | None:
    if not _finite(value) or not _finite(baseline):
        return None
    return float(value) - float(baseline)


def _add_deltas(arms: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = arms[0]["metrics"]
    for arm in arms:
        arm["deltas_vs_c_dual"] = {
            metric: _delta(arm["metrics"].get(metric), baseline.get(metric))
            for metric in ABSOLUTE_METRICS
        }
    return {
        "baseline_arm": arms[0]["arm"],
        "candidate_arm": arms[1]["arm"],
        "candidate_deltas_vs_c_dual": dict(arms[1]["deltas_vs_c_dual"]),
    }


def _write_outputs(payload: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{OUTPUT_STEM}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "name",
        "arm",
        "validation_passed",
        "source_commit_sha",
        "runai_job_name",
        "runai_project",
        "stage_b_checkpoint_sha256",
        "image_initializer_sha256",
        "speech_initializer_sha256",
        "observed_main_rows",
        "observed_alignment_rows",
    ]
    fields += list(ABSOLUTE_METRICS)
    fields += [f"delta_{metric}_vs_c_dual" for metric in ABSOLUTE_METRICS]
    fields += ["issue_codes"]
    with (output_dir / f"{OUTPUT_STEM}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in payload["arms"]:
            writer.writerow(
                {
                    "name": arm["name"],
                    "arm": arm["arm"],
                    "validation_passed": arm["validation_passed"],
                    **arm["source"],
                    **arm["initialization"],
                    "observed_main_rows": arm["observed_main_rows"],
                    "observed_alignment_rows": arm["observed_alignment_rows"],
                    **{
                        metric: arm["metrics"].get(metric)
                        for metric in ABSOLUTE_METRICS
                    },
                    **{
                        f"delta_{metric}_vs_c_dual": arm[
                            "deltas_vs_c_dual"
                        ].get(metric)
                        for metric in ABSOLUTE_METRICS
                    },
                    "issue_codes": ";".join(
                        sorted({item["code"] for item in arm["issues"]})
                    ),
                }
            )

    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "# MM Dual 3k Promotion Summary",
        "",
        f"Overall validation: **{'PASS' if payload['validation_passed'] else 'FAIL'}**",
        "",
        "All deltas are candidate minus C_DUAL.",
        "",
        "| Arm | Valid | PPL | dPPL | Accuracy | dAccuracy | Cond image R@1 | dImage | Cond speech R@1 | dSpeech | Gate entropy | Inactive | Overflow |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in payload["arms"]:
        metrics = arm["metrics"]
        deltas = arm["deltas_vs_c_dual"]
        lines.append(
            "| {arm} | {valid} | {ppl} | {dppl} | {accuracy} | {daccuracy} | "
            "{image} | {dimage} | {speech} | {dspeech} | {entropy} | {inactive} | "
            "{overflow} |".format(
                arm=arm["arm"],
                valid="PASS" if arm["validation_passed"] else "FAIL",
                ppl=fmt(metrics.get("perplexity")),
                dppl=fmt(deltas.get("perplexity")),
                accuracy=fmt(metrics.get("next_token_accuracy")),
                daccuracy=fmt(deltas.get("next_token_accuracy")),
                image=fmt(metrics.get("conditional_image_r1")),
                dimage=fmt(deltas.get("conditional_image_r1")),
                speech=fmt(metrics.get("conditional_speech_r1")),
                dspeech=fmt(deltas.get("conditional_speech_r1")),
                entropy=fmt(metrics.get("final_gate_entropy_mean")),
                inactive=fmt(metrics.get("final_inactive_expert_ratio_mean")),
                overflow=fmt(metrics.get("final_capacity_overflow_ratio_mean")),
            )
        )
    lines.extend(["", "## Validation Issues", ""])
    any_issues = False
    for arm in payload["arms"]:
        for item in arm["issues"]:
            any_issues = True
            lines.append(
                f"- `{arm['arm']}` `{item['code']}`: {item['message']} "
                f"(`{item['path']}`)"
            )
    if not any_issues:
        lines.append("None.")
    (output_dir / f"{OUTPUT_STEM}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def summarize(root: Path, output_dir: Path) -> dict[str, Any]:
    root = root.resolve(strict=False)
    output_dir = output_dir.resolve(strict=False)
    if _forbidden(root) or _forbidden(output_dir):
        raise EvidenceError("root/output path contains sealed or synthetic")
    if not root.is_dir() or root.is_symlink():
        raise EvidenceError(f"campaign root is missing or unsafe: {root}")
    arms = [_summarize_arm(root, name, spec) for name, spec in ARM_SPECS.items()]
    _cross_arm_checks(arms)
    comparison = _add_deltas(arms)
    payload = {
        "schema_version": 1,
        "campaign_root": str(root),
        "validation_passed": all(arm["validation_passed"] for arm in arms),
        "required_arms": list(ARM_SPECS),
        "expected_source_commit": EXPECTED_SOURCE_COMMIT,
        "main_steps": MAIN_STEPS,
        "alignment_pretrain_steps": ALIGNMENT_STEPS,
        "loss_window_relative_tolerance": LOSS_WINDOW_RELATIVE_TOLERANCE,
        "comparison": comparison,
        "arms": arms,
    }
    _write_outputs(payload, output_dir)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="Root containing the two exact seed-42 promotion arms"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Report destination (default: ROOT)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.root
    try:
        payload = summarize(args.root, output_dir)
    except EvidenceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "validation_passed": payload["validation_passed"],
                "output_dir": str(output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["validation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
