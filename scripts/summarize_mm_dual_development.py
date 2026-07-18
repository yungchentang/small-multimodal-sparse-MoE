#!/usr/bin/env python3
"""Fail-closed summary for the five-arm MM dual-initializer campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence


E3_DIR = "E3_final_multimodal_top2"
E3_TEXT_DIR = "E3_final_multimodal_top2_text_eval"
OUTPUT_STEM = "mm_dual_development_summary"
MAIN_STEPS = 500
ALIGNMENT_STEPS = 400
PPL_LIMIT = 13.0
LOSS_WINDOW_RELATIVE_TOLERANCE = 0.05
FORBIDDEN_WORDS = ("sealed", "synthetic")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

ARM_SPECS: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    (
        (
            "c0_seed42",
            {
                "arm": "C0",
                "image_group": "baseline",
                "image_bridge": "linear_projector",
                "speech_initializer": False,
                "kd_coefficient": 0.0,
            },
        ),
        (
            "c_image_norm_only_seed42",
            {
                "arm": "C_IMAGE_NORM_ONLY",
                "image_group": "norm",
                "image_bridge": "linear_projector_norm",
                "speech_initializer": False,
                "kd_coefficient": 0.0,
            },
        ),
        (
            "c_speech_init_only_seed42",
            {
                "arm": "C_SPEECH_INIT_ONLY",
                "image_group": "baseline",
                "image_bridge": "linear_projector",
                "speech_initializer": True,
                "kd_coefficient": 0.0,
            },
        ),
        (
            "c_dual_seed42",
            {
                "arm": "C_DUAL",
                "image_group": "norm",
                "image_bridge": "linear_projector_norm",
                "speech_initializer": True,
                "kd_coefficient": 0.0,
            },
        ),
        (
            "c_dual_kd025_seed42",
            {
                "arm": "C_DUAL_KD025",
                "image_group": "norm",
                "image_bridge": "linear_projector_norm",
                "speech_initializer": True,
                "kd_coefficient": 0.25,
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
    """An artifact cannot be accepted as campaign evidence."""


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
        _issue(issues, "invalid_evidence_path", f"{label} path is missing or forbidden", evidence_path)
        return None
    if not _regular_file(path, issues, label):
        return path
    digest = _sha256(path)
    expected_sha = metadata.get(sha_key)
    if not isinstance(expected_sha, str) or SHA256_RE.fullmatch(expected_sha) is None:
        _issue(issues, "invalid_sha256", f"{label} SHA-256 is missing or malformed", evidence_path)
    elif digest != expected_sha:
        _issue(issues, "checkpoint_sha_mismatch", f"{label} SHA-256 does not match actual file", path)
    if size_key is not None and metadata.get(size_key) != path.stat().st_size:
        _issue(issues, "checkpoint_size_mismatch", f"{label} size does not match actual file", path)
    return path


def _validate_run_identity(
    manifest: Mapping[str, Any], manifest_path: Path, run_root: Path, issues: list[dict[str, str]]
) -> dict[str, Any]:
    provenance = manifest.get("run_provenance")
    if not isinstance(provenance, dict):
        _issue(issues, "missing_provenance", "run_provenance is missing", manifest_path)
        return {}
    commit = provenance.get("source_commit_sha")
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        _issue(issues, "missing_provenance", "source commit must be full 40-hex", manifest_path)
    for field in ("runai_job_name", "runai_project"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            _issue(issues, "missing_provenance", f"{field} is missing", manifest_path)
    for field in ("source_commit_sha", "runai_job_name", "runai_project"):
        _require_equal(issues, manifest.get(field), provenance.get(field), field, manifest_path)
    _require_equal(issues, provenance.get("final_main_steps"), MAIN_STEPS, "final_main_steps", manifest_path)
    _require_equal(
        issues,
        provenance.get("alignment_pretrain_steps"),
        ALIGNMENT_STEPS,
        "alignment_pretrain_steps",
        manifest_path,
    )
    _require_equal(
        issues,
        provenance.get("checkpoint_completed_step"),
        MAIN_STEPS,
        "checkpoint_completed_step",
        manifest_path,
    )
    for field in ("sealed_evidence_used", "synthetic_evidence_used"):
        _require_equal(issues, provenance.get(field), False, field, manifest_path, "non_real_data")
    output_root = provenance.get("resolved_output_root")
    if not _path_matches(output_root, run_root, manifest_path.parent):
        _issue(issues, "provenance_mismatch", "resolved_output_root does not match arm root", manifest_path)
    return provenance


def _validate_real_data(
    manifest: Mapping[str, Any], provenance: Mapping[str, Any], manifest_path: Path, issues: list[dict[str, str]]
) -> None:
    args = manifest.get("args") if isinstance(manifest.get("args"), dict) else {}
    values = (manifest.get("data_dir"), args.get("data_dir"), provenance.get("resolved_data_root"))
    if any(not isinstance(value, str) or not value or _forbidden(value) for value in values):
        _issue(issues, "non_real_data", "data roots are missing or contain sealed/synthetic", manifest_path)
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
    data_manifest_path = data_root / "manifest.json"
    data_manifest = _load_json(data_manifest_path, issues, "real data manifest")
    embedded = manifest.get("data_manifest")
    if data_manifest is None or not isinstance(embedded, dict) or data_manifest != embedded:
        _issue(issues, "non_real_data", "embedded data manifest does not match the real data manifest", manifest_path)
    serialized = json.dumps(manifest, sort_keys=True).lower()
    if '"sealed_evidence_used": true' in serialized or '"synthetic_evidence_used": true' in serialized:
        _issue(issues, "non_real_data", "manifest declares sealed or synthetic evidence", manifest_path)


def _validate_companion_initializer(
    initialization: Mapping[str, Any],
    args: Mapping[str, Any],
    prefix: str,
    expected_scope: str,
    expected_bridge: str | None,
    manifest_path: Path,
    issues: list[dict[str, str]],
) -> None:
    required_strings = (
        "path",
        "sha256",
        "manifest_path",
        "manifest_sha256",
        "source_commit_sha",
        "runai_job_name",
        "runai_project",
    )
    for field in required_strings:
        if not isinstance(initialization.get(field), str) or not initialization[field]:
            _issue(issues, "invalid_initializer", f"{prefix}.{field} is missing", manifest_path)
    _require_equal(issues, initialization.get("scope"), expected_scope, f"{prefix}.scope", manifest_path, "wrong_initializer_scope")
    _require_equal(issues, initialization.get("completion_status"), "completed", f"{prefix}.completion_status", manifest_path)
    _require_equal(issues, initialization.get("completion_step"), MAIN_STEPS, f"{prefix}.completion_step", manifest_path)
    for field in ("sealed_evidence_used", "synthetic_evidence_used"):
        _require_equal(issues, initialization.get(field), False, f"{prefix}.{field}", manifest_path, "non_real_data")
    if not isinstance(initialization.get("source_commit_sha"), str) or COMMIT_RE.fullmatch(str(initialization.get("source_commit_sha"))) is None:
        _issue(issues, "invalid_initializer", f"{prefix}.source_commit_sha is invalid", manifest_path)

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
    _require_equal(issues, args.get(checkpoint_arg), initialization.get("path"), checkpoint_arg, manifest_path)
    _require_equal(issues, args.get(sha_arg), initialization.get("sha256"), sha_arg, manifest_path)
    _require_equal(issues, args.get(manifest_arg), initialization.get("manifest_path"), manifest_arg, manifest_path)

    checkpoint = _validate_external_file(
        initialization, "path", "sha256", "size_bytes", issues, manifest_path, prefix + " checkpoint"
    )
    companion_path = _validate_external_file(
        initialization, "manifest_path", "manifest_sha256", None, issues, manifest_path, prefix + " manifest"
    )
    if companion_path is None:
        return
    companion = _load_json(companion_path, issues, prefix + " manifest")
    if companion is None:
        return
    completion = companion.get("completion") if isinstance(companion.get("completion"), dict) else {}
    companion_args = companion.get("args") if isinstance(companion.get("args"), dict) else {}
    companion_provenance = companion.get("run_provenance") if isinstance(companion.get("run_provenance"), dict) else {}
    _require_equal(issues, completion.get("status"), "completed", f"{prefix} companion completion", companion_path)
    _require_equal(issues, completion.get("e3_steps"), MAIN_STEPS, f"{prefix} companion steps", companion_path)
    if checkpoint is not None and not _path_matches(completion.get("e3_checkpoint_path"), checkpoint, companion_path.parent):
        _issue(issues, "invalid_initializer", f"{prefix} companion checkpoint path mismatch", companion_path)
    _require_equal(issues, completion.get("e3_checkpoint_sha256"), initialization.get("sha256"), f"{prefix} companion SHA", companion_path)
    _require_equal(issues, companion_args.get("final_steps"), MAIN_STEPS, f"{prefix} companion final_steps", companion_path)
    _require_equal(issues, companion_args.get("alignment_pretrain_steps"), ALIGNMENT_STEPS, f"{prefix} companion alignment steps", companion_path)
    _require_equal(issues, companion_args.get("alignment_pretrain_modalities"), expected_scope, f"{prefix} companion scope", companion_path, "wrong_initializer_scope")
    if expected_bridge is not None:
        _require_equal(issues, companion_args.get("image_bridge_type"), expected_bridge, f"{prefix} companion image bridge", companion_path, "wrong_image_initializer")
        _require_equal(issues, companion_args.get("image_prefix_tokens"), 50, f"{prefix} companion image prefix tokens", companion_path)
    else:
        _require_equal(issues, companion_args.get("speech_unfreeze_last_blocks"), 1, "speech companion unfreeze blocks", companion_path, "wrong_initializer_scope")
        _require_equal(issues, companion_args.get("speech_unfreeze_layer_norm"), True, "speech companion layer norm", companion_path, "wrong_initializer_scope")
    for field in ("source_commit_sha", "runai_job_name", "runai_project"):
        _require_equal(issues, companion_provenance.get(field), initialization.get(field), f"{prefix} companion {field}", companion_path)
    for field in ("sealed_evidence_used", "synthetic_evidence_used"):
        _require_equal(issues, companion_provenance.get(field), False, f"{prefix} companion {field}", companion_path, "non_real_data")


def _validate_initializers(
    manifest: Mapping[str, Any], spec: Mapping[str, Any], manifest_path: Path, issues: list[dict[str, str]]
) -> tuple[str | None, str | None, str | None]:
    args = manifest.get("args") if isinstance(manifest.get("args"), dict) else {}
    _require_equal(issues, args.get("image_bridge_type"), spec["image_bridge"], "image_bridge_type", manifest_path, "wrong_image_initializer")
    _require_equal(issues, args.get("multimodal_initialization_scope"), "image", "multimodal_initialization_scope", manifest_path, "wrong_initializer_scope")
    image = manifest.get("multimodal_initialization")
    if not isinstance(image, dict) or not image:
        _issue(issues, "invalid_initializer", "multimodal_initialization is missing", manifest_path)
        image = {}
    else:
        _validate_companion_initializer(
            image, args, "multimodal_initialization", "image", str(spec["image_bridge"]), manifest_path, issues
        )

    speech = manifest.get("speech_initialization")
    expected_speech = bool(spec["speech_initializer"])
    speech_sha: str | None = None
    if expected_speech:
        if not isinstance(speech, dict) or not speech:
            _issue(issues, "missing_speech_initializer", "speech_initialization is required", manifest_path)
            speech = {}
        else:
            _validate_companion_initializer(
                speech, args, "speech_initialization", "speech", None, manifest_path, issues
            )
            speech_sha = speech.get("sha256") if isinstance(speech.get("sha256"), str) else None
    else:
        if speech not in ({}, None):
            _issue(issues, "unexpected_speech_initializer", "speech_initialization must be absent", manifest_path)
        for field in (
            "speech_initial_checkpoint",
            "speech_initial_checkpoint_sha256",
            "speech_initial_manifest",
        ):
            if args.get(field) not in (None, ""):
                _issue(issues, "unexpected_speech_initializer", f"{field} must be empty", manifest_path)
    image_sha = image.get("sha256") if isinstance(image.get("sha256"), str) else None
    stage_b = manifest.get("stage_b_initialization")
    stage_b_sha: str | None = None
    if not isinstance(stage_b, dict) or not stage_b:
        _issue(issues, "missing_stage_b", "stage_b_initialization is missing", manifest_path)
    else:
        stage_b_sha = stage_b.get("sha256") if isinstance(stage_b.get("sha256"), str) else None
        _validate_external_file(stage_b, "path", "sha256", "size_bytes", issues, manifest_path, "Stage B checkpoint")
        _require_equal(issues, args.get("stage_b_checkpoint"), stage_b.get("path"), "stage_b_checkpoint", manifest_path)
        _require_equal(issues, args.get("stage_b_checkpoint_sha256"), stage_b.get("sha256"), "stage_b_checkpoint_sha256", manifest_path)
        for field in ("sealed_evidence_used", "synthetic_evidence_used"):
            _require_equal(issues, stage_b.get(field), False, f"stage_b.{field}", manifest_path, "non_real_data")
    return image_sha, speech_sha, stage_b_sha


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    alignment_rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    train_path: Path,
    alignment_path: Path,
    image_sha: str | None,
    speech_sha: str | None,
    stage_b_sha: str | None,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    if len(rows) != MAIN_STEPS:
        _issue(issues, "wrong_main_steps", f"requires exactly {MAIN_STEPS} train rows; got {len(rows)}", train_path)
    if len(alignment_rows) != ALIGNMENT_STEPS:
        _issue(issues, "wrong_alignment_steps", f"requires exactly {ALIGNMENT_STEPS} alignment rows; got {len(alignment_rows)}", alignment_path)
    if [row.get("step") for row in rows] != list(range(1, MAIN_STEPS + 1)):
        _issue(issues, "wrong_main_steps", "train steps must be exactly ordered 1..500", train_path)
    if [row.get("step") for row in alignment_rows] != list(range(1, ALIGNMENT_STEPS + 1)):
        _issue(issues, "wrong_alignment_steps", "alignment steps must be exactly ordered 1..400", alignment_path)
    if any(not _all_numbers_finite(row) for row in rows):
        _issue(issues, "non_finite_loss", "train rows contain non-finite numeric values", train_path)
    if any(not _all_numbers_finite(row) for row in alignment_rows):
        _issue(issues, "non_finite_loss", "alignment rows contain non-finite numeric values", alignment_path)

    ce_by_modality: dict[str, list[float]] = {"text": [], "speech": []}
    prefix_seen = False
    expected_speech = bool(spec["speech_initializer"])
    expected_kd = float(spec["kd_coefficient"])
    for index, row in enumerate(rows, 1):
        for field in ("loss", "lm_ce_loss", "gate_entropy_mean", "inactive_expert_ratio_mean", "capacity_overflow_ratio_mean"):
            if not _finite(row.get(field)):
                _issue(issues, "non_finite_loss", f"row {index} {field} is not finite numeric", train_path)
        modality = row.get("modality")
        expected_modality = ("text", "speech", "speech")[(index - 1) % 3]
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
                "wrong_main_steps",
                f"row {index} optimizer_step must be {expected_optimizer_step} for {modality}",
                train_path,
            )
        for field, expected in (("native_top_k", 8), ("runtime_top_k", 2), ("top_k", 2)):
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
        _require_equal(issues, row.get("source_selected_checkpoint_sha256"), image_sha, f"row {index} image initializer SHA", train_path)
        _require_equal(issues, row.get("source_stage_b_checkpoint_sha256"), stage_b_sha, f"row {index} Stage B SHA", train_path)
        _require_equal(issues, row.get("initial_checkpoint_state_restored"), True, f"row {index} image restore flag", train_path)
        _require_equal(issues, row.get("stage_b_checkpoint_state_restored"), True, f"row {index} Stage B restore flag", train_path)
        _require_equal(issues, row.get("speech_initial_checkpoint_state_restored"), expected_speech, f"row {index} speech restore flag", train_path, "wrong_initializer_scope")
        expected_source_speech = speech_sha if expected_speech else None
        _require_equal(issues, row.get("source_speech_initial_checkpoint_sha256"), expected_source_speech, f"row {index} speech initializer SHA", train_path, "wrong_initializer_scope")
        if not _same_number(row.get("speech_behavior_kl_coef"), expected_kd):
            _issue(issues, "wrong_kd_coefficient", f"row {index} KD coefficient mismatch", train_path)
        if modality == "speech":
            counts = row.get("modality_token_counts_across_layers")
            conservation = row.get("modality_assignment_conservation")
            expected_prefix = row.get("prefix_expected_assignments_tokens_x_layers_x_k")
            observed_prefix = row.get("prefix_observed_assignments")
            prefix_ok = (
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
            if prefix_ok:
                prefix_seen = True
            else:
                _issue(issues, "missing_prefix_path_flags", f"row {index} lacks valid audio-prefix routing accounting", train_path)
    if not prefix_seen:
        _issue(issues, "missing_prefix_path_flags", "no valid audio-prefix routing row", train_path)
    for modality, values in ce_by_modality.items():
        if len(values) < 6:
            _issue(issues, "divergent_loss", f"{modality} has fewer than six CE observations", train_path)
            continue
        width = max(3, min(100, len(values) // 5))
        first_mean = sum(values[:width]) / width
        last_mean = sum(values[-width:]) / width
        if (
            not math.isfinite(first_mean)
            or not math.isfinite(last_mean)
            or last_mean > first_mean * (1.0 + LOSS_WINDOW_RELATIVE_TOLERANCE)
        ):
            _issue(
                issues,
                "divergent_loss",
                f"{modality} final CE window exceeds initial window by more than "
                f"{LOSS_WINDOW_RELATIVE_TOLERANCE:.0%}",
                train_path,
            )
    for index, row in enumerate(alignment_rows, 1):
        if row.get("modality") != "speech" or not _finite(row.get("loss")):
            _issue(issues, "invalid_alignment", f"alignment row {index} must be finite speech evidence", alignment_path)

    last = rows[-1] if rows else {}
    return {
        "first_loss": rows[0].get("loss") if rows else None,
        "last_loss": last.get("loss"),
        "final_gate_entropy_mean": last.get("gate_entropy_mean"),
        "final_inactive_expert_ratio_mean": last.get("inactive_expert_ratio_mean"),
        "final_capacity_overflow_ratio_mean": last.get("capacity_overflow_ratio_mean"),
        "final_prefix_expected_assignments": last.get("prefix_expected_assignments_tokens_x_layers_x_k"),
        "final_prefix_observed_assignments": last.get("prefix_observed_assignments"),
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
    completion = manifest.get("completion") if isinstance(manifest.get("completion"), dict) else {}
    _require_equal(issues, completion.get("status"), "completed", "completion.status", manifest_path)
    _require_equal(issues, completion.get("e3_steps"), MAIN_STEPS, "completion.e3_steps", manifest_path)
    for payload, path, path_key, sha_key, size_key, label in (
        (completion, manifest_path, "e3_checkpoint_path", "e3_checkpoint_sha256", "e3_checkpoint_size_bytes", "manifest completion"),
        (metrics or {}, metrics_path, "checkpoint_path", "checkpoint_sha256", "checkpoint_size_bytes", "E3 metrics"),
    ):
        if not _path_matches(payload.get(path_key), checkpoint, path.parent):
            _issue(issues, "checkpoint_path_mismatch", f"{label} checkpoint path mismatch", path)
        _require_equal(issues, payload.get(sha_key), actual_sha, f"{label} checkpoint SHA", path, "checkpoint_sha_mismatch")
        _require_equal(issues, payload.get(size_key), actual_size, f"{label} checkpoint size", path, "checkpoint_size_mismatch")
    if metrics is None or text_metrics is None:
        return values
    _require_equal(issues, metrics.get("real_subset"), True, "E3 real_subset", metrics_path, "non_real_data")
    _require_equal(issues, text_metrics.get("real_subset"), True, "text eval real_subset", text_path, "non_real_data")
    embedded_rows = metrics.get("steps")
    if embedded_rows != list(rows):
        _issue(issues, "metrics_ledger_mismatch", "E3 embedded steps differ from train_metrics.jsonl", metrics_path)
    for field, expected in (
        ("first_loss", rows[0].get("loss") if rows else None),
        ("last_loss", rows[-1].get("loss") if rows else None),
        ("min_loss", min((float(row["loss"]) for row in rows if _finite(row.get("loss"))), default=None)),
    ):
        if not _same_number(metrics.get(field), expected):
            _issue(issues, "metrics_ledger_mismatch", f"E3 {field} differs from raw ledger", metrics_path)

    provenance = text_metrics.get("provenance") if isinstance(text_metrics.get("provenance"), dict) else {}
    embedded_provenance = metrics.get("text_eval_provenance")
    if provenance != embedded_provenance:
        _issue(issues, "copied_text_metrics", "text provenance differs from E3 embedded provenance", text_path)
    embedded_text = metrics.get("text_eval") if isinstance(metrics.get("text_eval"), dict) else {}
    for field in ("perplexity", "next_token_accuracy", "provenance"):
        if embedded_text.get(field) != text_metrics.get(field):
            _issue(issues, "copied_text_metrics", f"embedded text {field} differs from own text eval", text_path)
    requirements = {
        "source_experiment_id": E3_DIR,
        "source_checkpoint_sha256": actual_sha,
        "source_checkpoint_size_bytes": actual_size,
        "source_training_steps": MAIN_STEPS,
        "source_checkpoint_saved_before_eval": True,
        "copied_from_e2": False,
    }
    for field, expected in requirements.items():
        _require_equal(issues, provenance.get(field), expected, f"text provenance {field}", text_path, "copied_text_metrics")
    if not _path_matches(provenance.get("source_checkpoint"), checkpoint, text_path.parent):
        _issue(issues, "copied_text_metrics", "text metrics do not point to own checkpoint", text_path)

    ppl = text_metrics.get("perplexity")
    accuracy = text_metrics.get("next_token_accuracy")
    if not _finite(ppl):
        _issue(issues, "invalid_quality_metric", "perplexity is not finite numeric", text_path)
    elif float(ppl) > PPL_LIMIT:
        _issue(issues, "ppl_above_limit", f"perplexity {ppl} exceeds {PPL_LIMIT}", text_path)
    if not _finite(accuracy):
        _issue(issues, "invalid_quality_metric", "next_token_accuracy is not finite numeric", text_path)
    values["perplexity"] = ppl if _finite(ppl) else None
    values["next_token_accuracy"] = accuracy if _finite(accuracy) else None

    retrieval = metrics.get("retrieval_eval") if isinstance(metrics.get("retrieval_eval"), dict) else {}
    path_requirements = {
        "retrieval_path": "shared_olmoe_prefix_hidden",
        "retrieval_uses_lm_hidden_states": True,
        "retrieval_uses_direct_encoder_pooling": False,
        "conditional_eval_path": "shared_olmoe_prefix_lm_nll",
        "conditional_uses_lm_logits": True,
        "conditional_uses_direct_encoder_pooling": False,
    }
    for field, expected in path_requirements.items():
        _require_equal(issues, retrieval.get(field), expected, f"retrieval.{field}", metrics_path, "missing_prefix_path_flags")
    retrieval_fields = {
        "conditional_image_r1": "conditional_image_to_text_r_at_1",
        "conditional_speech_r1": "conditional_speech_to_text_r_at_1",
        "embedding_image_r1": "image_to_text_r_at_1",
        "embedding_speech_r1": "speech_to_text_r_at_1",
    }
    for output_name, source_name in retrieval_fields.items():
        value = retrieval.get(source_name)
        if not _finite(value):
            _issue(issues, "invalid_quality_metric", f"retrieval.{source_name} is not finite numeric", metrics_path)
        else:
            values[output_name] = value
    for count_field in ("image_eval_count", "speech_eval_count", "conditional_image_eval_count", "conditional_speech_eval_count"):
        if not isinstance(retrieval.get(count_field), int) or retrieval[count_field] <= 0:
            _issue(issues, "invalid_quality_metric", f"retrieval.{count_field} must be positive", metrics_path)

    last = rows[-1] if rows else {}
    metric_row_pairs = (
        ("final_gate_entropy_mean", "gate_entropy_mean"),
        ("final_inactive_expert_ratio_mean", "inactive_expert_ratio_mean"),
        ("final_capacity_overflow_ratio_mean", "capacity_overflow_ratio_mean"),
    )
    for metric_field, row_field in metric_row_pairs:
        if not _same_number(metrics.get(metric_field), last.get(row_field)):
            _issue(issues, "routing_metric_mismatch", f"{metric_field} differs from final train row", metrics_path)
        values[metric_field] = last.get(row_field) if _finite(last.get(row_field)) else None
    values["first_loss"] = rows[0].get("loss") if rows and _finite(rows[0].get("loss")) else None
    values["last_loss"] = last.get("loss") if _finite(last.get("loss")) else None
    values["final_prefix_expected_assignments"] = last.get("prefix_expected_assignments_tokens_x_layers_x_k")
    values["final_prefix_observed_assignments"] = last.get("prefix_observed_assignments")
    values["checkpoint_sha256"] = actual_sha
    values["checkpoint_size_bytes"] = actual_size
    return values


def _summarize_arm(root: Path, name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
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
        "deltas_vs_c0": {metric: None for metric in ABSOLUTE_METRICS},
        "promotion_flags": {},
    }
    if not run_root.is_dir() or run_root.is_symlink() or _forbidden(run_root):
        _issue(issues, "missing_arm", "arm directory is missing, unsafe, or forbidden", run_root)
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
        _require_equal(issues, args.get(field), expected, f"args.{field}", manifest_path)
    if not _same_number(args.get("speech_behavior_kl_coef"), spec["kd_coefficient"]):
        _issue(issues, "wrong_kd_coefficient", "manifest KD coefficient mismatch", manifest_path)
    if not _same_number(args.get("speech_behavior_kl_temperature"), 1.0):
        _issue(issues, "wrong_kd_coefficient", "manifest KD temperature must be 1", manifest_path)

    provenance = _validate_run_identity(manifest, manifest_path, run_root, issues)
    _validate_real_data(manifest, provenance, manifest_path, issues)
    image_sha, speech_sha, stage_b_sha = _validate_initializers(manifest, spec, manifest_path, issues)
    row_metrics = _validate_rows(
        rows,
        alignment_rows,
        spec,
        train_path,
        alignment_path,
        image_sha,
        speech_sha,
        stage_b_sha,
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
    final_metrics.update({key: value for key, value in row_metrics.items() if final_metrics.get(key) is None})
    result["metrics"] = final_metrics
    result["source"] = {
        "source_commit_sha": provenance.get("source_commit_sha"),
        "runai_job_name": provenance.get("runai_job_name"),
        "runai_project": provenance.get("runai_project"),
    }
    result["initialization"] = {
        "stage_b_checkpoint_sha256": stage_b_sha,
        "image_initializer_sha256": image_sha,
        "speech_initializer_sha256": speech_sha,
    }
    result["validation_passed"] = not issues
    return result


def _cross_arm_checks(arms: list[dict[str, Any]]) -> None:
    def add(arm: dict[str, Any], code: str, message: str) -> None:
        _issue(arm["issues"], code, message, Path(arm["run_root"]))

    for field, code in (
        ("stage_b_checkpoint_sha256", "stage_b_mismatch"),
        ("source_commit_sha", "source_mismatch"),
        ("runai_project", "source_mismatch"),
    ):
        values = []
        for arm in arms:
            source = arm["initialization"] if field == "stage_b_checkpoint_sha256" else arm["source"]
            values.append(source.get(field))
        valid = [value for value in values if value is not None]
        if len(valid) != len(arms) or len(set(valid)) != 1:
            for arm in arms:
                add(arm, code, f"all arms must share one non-empty {field}")

    jobs = [arm["source"].get("runai_job_name") for arm in arms]
    if any(not job for job in jobs) or len(set(jobs)) != len(arms):
        for arm in arms:
            add(arm, "source_mismatch", "Run:AI job names must be present and unique")

    checkpoint_hashes = [arm["metrics"].get("checkpoint_sha256") for arm in arms]
    duplicates = {value for value in checkpoint_hashes if value and checkpoint_hashes.count(value) > 1}
    if duplicates:
        for arm in arms:
            if arm["metrics"].get("checkpoint_sha256") in duplicates:
                add(arm, "copied_checkpoint_hash", "final checkpoint SHA-256 is duplicated across arms")

    image_groups = {"baseline": [], "norm": []}
    for arm in arms:
        image_groups[str(arm["expected"]["image_group"])].append(
            arm["initialization"].get("image_initializer_sha256")
        )
    group_values: dict[str, str | None] = {}
    for group, values in image_groups.items():
        valid = [value for value in values if value]
        group_values[group] = valid[0] if len(valid) == len(values) and len(set(valid)) == 1 else None
        if group_values[group] is None:
            for arm in arms:
                if arm["expected"]["image_group"] == group:
                    add(arm, "wrong_image_initializer", f"{group} arms must share one image initializer")
    if group_values["baseline"] is not None and group_values["baseline"] == group_values["norm"]:
        for arm in arms:
            add(arm, "wrong_image_initializer", "baseline and norm image initializer SHA-256 must differ")

    speech_arms = [arm for arm in arms if arm["expected"]["speech_initializer"]]
    speech_hashes = [arm["initialization"].get("speech_initializer_sha256") for arm in speech_arms]
    if any(not value for value in speech_hashes) or len(set(speech_hashes)) != 1:
        for arm in speech_arms:
            add(arm, "wrong_initializer_scope", "speech-enabled arms must share one speech initializer")
    for arm in arms:
        arm["validation_passed"] = not arm["issues"]


def _delta(value: Any, baseline: Any) -> float | None:
    if not _finite(value) or not _finite(baseline):
        return None
    return float(value) - float(baseline)


def _add_deltas_and_promotions(arms: list[dict[str, Any]]) -> None:
    baseline = arms[0]["metrics"]
    for index, arm in enumerate(arms):
        deltas = {
            metric: _delta(arm["metrics"].get(metric), baseline.get(metric))
            for metric in ABSOLUTE_METRICS
        }
        arm["deltas_vs_c0"] = deltas
        ppl = arm["metrics"].get("perplexity")
        image = deltas.get("conditional_image_r1")
        speech = deltas.get("conditional_speech_r1")
        ppl_delta = deltas.get("perplexity")
        flags = {
            "reference_arm": index == 0,
            "integrity_valid": arm["validation_passed"],
            "ppl_at_most_13": _finite(ppl) and float(ppl) <= PPL_LIMIT,
            "ppl_not_worse_than_c0": ppl_delta is not None and ppl_delta <= 0.0,
            "conditional_image_not_worse_than_c0": image is not None and image >= 0.0,
            "conditional_speech_not_worse_than_c0": speech is not None and speech >= 0.0,
            "strict_metric_improvement_vs_c0": index > 0
            and any(delta is not None and delta > 0.0 for delta in (image, speech, -ppl_delta if ppl_delta is not None else None)),
        }
        flags["promote"] = bool(
            index > 0
            and flags["integrity_valid"]
            and flags["ppl_at_most_13"]
            and flags["ppl_not_worse_than_c0"]
            and flags["conditional_image_not_worse_than_c0"]
            and flags["conditional_speech_not_worse_than_c0"]
            and flags["strict_metric_improvement_vs_c0"]
        )
        arm["promotion_flags"] = flags


def _write_outputs(payload: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{OUTPUT_STEM}.json"
    csv_path = output_dir / f"{OUTPUT_STEM}.csv"
    markdown_path = output_dir / f"{OUTPUT_STEM}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
    fields += [f"delta_{metric}_vs_c0" for metric in ABSOLUTE_METRICS]
    fields += ["promote", "issue_codes"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in payload["arms"]:
            row = {
                "name": arm["name"],
                "arm": arm["arm"],
                "validation_passed": arm["validation_passed"],
                **arm["source"],
                **arm["initialization"],
                "observed_main_rows": arm["observed_main_rows"],
                "observed_alignment_rows": arm["observed_alignment_rows"],
                **{metric: arm["metrics"].get(metric) for metric in ABSOLUTE_METRICS},
                **{f"delta_{metric}_vs_c0": arm["deltas_vs_c0"].get(metric) for metric in ABSOLUTE_METRICS},
                "promote": arm["promotion_flags"].get("promote"),
                "issue_codes": ";".join(sorted({item["code"] for item in arm["issues"]})),
            }
            writer.writerow(row)

    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "# MM Dual Development Summary",
        "",
        f"Overall validation: **{'PASS' if payload['validation_passed'] else 'FAIL'}**",
        "",
        "Promotion is conservative Pareto promotion: valid evidence, PPL <= 13, no regression vs C0 in PPL or conditional image/speech R@1, and at least one strict improvement.",
        "",
        "| Arm | Valid | PPL | dPPL | Cond image R@1 | dImage | Cond speech R@1 | dSpeech | Gate entropy | Inactive | Overflow | Promote |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in payload["arms"]:
        metrics = arm["metrics"]
        deltas = arm["deltas_vs_c0"]
        lines.append(
            "| {arm} | {valid} | {ppl} | {dppl} | {image} | {dimage} | {speech} | {dspeech} | {entropy} | {inactive} | {overflow} | {promote} |".format(
                arm=arm["arm"],
                valid="PASS" if arm["validation_passed"] else "FAIL",
                ppl=fmt(metrics.get("perplexity")),
                dppl=fmt(deltas.get("perplexity")),
                image=fmt(metrics.get("conditional_image_r1")),
                dimage=fmt(deltas.get("conditional_image_r1")),
                speech=fmt(metrics.get("conditional_speech_r1")),
                dspeech=fmt(deltas.get("conditional_speech_r1")),
                entropy=fmt(metrics.get("final_gate_entropy_mean")),
                inactive=fmt(metrics.get("final_inactive_expert_ratio_mean")),
                overflow=fmt(metrics.get("final_capacity_overflow_ratio_mean")),
                promote="YES" if arm["promotion_flags"].get("promote") else "NO",
            )
        )
    lines.extend(["", "## Validation Issues", ""])
    any_issues = False
    for arm in payload["arms"]:
        for item in arm["issues"]:
            any_issues = True
            lines.append(f"- `{arm['arm']}` `{item['code']}`: {item['message']} (`{item['path']}`)")
    if not any_issues:
        lines.append("None.")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(root: Path, output_dir: Path) -> dict[str, Any]:
    root = root.resolve(strict=False)
    output_dir = output_dir.resolve(strict=False)
    if _forbidden(root) or _forbidden(output_dir):
        raise EvidenceError("root/output path contains sealed or synthetic")
    if not root.is_dir() or root.is_symlink():
        raise EvidenceError(f"campaign root is missing or unsafe: {root}")
    arms = [_summarize_arm(root, name, spec) for name, spec in ARM_SPECS.items()]
    _cross_arm_checks(arms)
    _add_deltas_and_promotions(arms)
    payload = {
        "schema_version": 1,
        "campaign_root": str(root),
        "validation_passed": all(arm["validation_passed"] for arm in arms),
        "required_arms": list(ARM_SPECS),
        "ppl_limit": PPL_LIMIT,
        "arms": arms,
    }
    _write_outputs(payload, output_dir)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Root containing the five exact seed-42 arm directories")
    parser.add_argument("--output-dir", type=Path, help="Report destination (default: ROOT)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.root
    try:
        payload = summarize(args.root, output_dir)
    except EvidenceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"validation_passed": payload["validation_passed"], "output_dir": str(output_dir.resolve())}, sort_keys=True))
    return 0 if payload["validation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
