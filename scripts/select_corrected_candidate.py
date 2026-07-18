#!/usr/bin/env python3
"""Fail-closed development-only selection for the corrected E3 sweep.

The selector intentionally reads only run artifacts and the explicitly supplied
development evaluation roots.  Sealed paths are rejected before they are
opened.  A candidate is eligible only after its training, checkpoint, built-in
evaluation, and fixed 5-way/10-way development evidence all validate.

Eligible candidates are ordered lexicographically by:

1. number of development image/speech R@1 values above exact chance (max),
2. mean normalized positive R@1 lift above chance (max),
3. final routing overflow (min),
4. final inactive-expert ratio (min),
5. final accepted-load coefficient of variation (min), and
6. candidate name (ascending deterministic tie-break).

PPL <= 20 is an eligibility gate, not a quantity that can compensate for weak
development retrieval.  ``selection_score`` is a human-readable bounded
encoding of the same components; the lexicographic key above is authoritative.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


E3_DIR = "E3_final_multimodal_top2"
E3_TEXT_DIR = "E3_final_multimodal_top2_text_eval"
REQUIRED_CANDIDATES = 5
REQUIRED_STEPS = 6000
REQUIRED_QUERIES_PER_MODALITY = 250
EXPECTED_SWEEP = frozenset(
    {
        (6.0, 0.02),
        (7.0, 0.01),
        (7.0, 0.02),
        (7.0, 0.04),
        (8.0, 0.02),
    }
)
PATH_ONLY_ARGS = frozenset({"output_dir", "feature_cache_dir"})
EXPECTED_PROTOCOL = {
    "protocol_name": "development_conditional_v2",
    "eval_split_name": "development_selection",
    "candidate_seed": 271828,
    "control_seed": 42,
    "negative_mode": "random",
    "eval_path": "shared_prefix",
    "prefix_control": "real",
    "condition": "real",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
BASE_LOSS_RECONCILIATION_ABS_TOL = 2.5e-6
BASE_LOSS_GAP_FIELD_ABS_TOL = 5.0e-7


class SelectionError(RuntimeError):
    """Base class for fail-closed selector errors."""


class ArtifactError(SelectionError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class NoValidCandidateError(SelectionError):
    """Raised after diagnostic outputs are written with no selected candidate."""


@dataclass
class Candidate:
    name: str
    run_root: Path
    dev_root: Path
    reasons: List[Dict[str, str]] = field(default_factory=list)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    non_swept_args: Optional[Dict[str, Any]] = None
    dataset_provenance: Optional[Dict[str, Any]] = None
    manifest: Optional[Dict[str, Any]] = None
    manifest_args: Optional[Dict[str, Any]] = None
    checkpoint_sha256: str = ""
    protocol_digests: Dict[Any, str] = field(default_factory=dict)
    result_digests: Dict[Any, str] = field(default_factory=dict)
    selection_score: Optional[float] = None
    selection_components: Optional[Dict[str, Any]] = None
    rank: Optional[int] = None

    @property
    def valid(self) -> bool:
        return not self.reasons

    def reject(self, code: str, message: str) -> None:
        reason = {"code": code, "message": message}
        if reason not in self.reasons:
            self.reasons.append(reason)


@dataclass
class SharedDevelopmentData:
    manifest_path: Path
    manifest_fingerprint: Dict[str, Any]
    image_path: Path
    image_fingerprint: Dict[str, Any]
    image_uids: List[str]
    speech_path: Path
    speech_fingerprint: Dict[str, Any]
    speech_uids: List[str]
    data_dir: str


class ArtifactCache:
    def __init__(self) -> None:
        self.development: Dict[Tuple[str, str], SharedDevelopmentData] = {}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value!r}")


def _object_without_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json(text: str, source: Path) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactError("invalid_json", f"{source}: invalid strict JSON: {exc}") from exc


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _stat_signature(path: Path) -> Tuple[int, int, int, int]:
    stat = path.stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _contains_sealed(path: Path | str) -> bool:
    return any("sealed" in part.lower() for part in Path(str(path)).parts)


def reject_sealed_path(path: Path | str, role: str) -> None:
    lexical = Path(str(path))
    if _contains_sealed(lexical):
        raise ArtifactError("sealed_artifact", f"{role} path is sealed and must not be inspected: {lexical}")
    try:
        resolved = lexical.resolve(strict=False)
    except OSError as exc:
        raise ArtifactError("invalid_path", f"cannot resolve {role} path {lexical}: {exc}") from exc
    if _contains_sealed(resolved):
        raise ArtifactError("sealed_artifact", f"{role} resolves into a sealed path and must not be inspected: {resolved}")


def _reject_sealed_strings(value: Any, role: str, location: str = "root") -> None:
    if isinstance(value, str):
        if "sealed" in value.lower():
            raise ArtifactError(
                "sealed_artifact",
                f"{role} contains a sealed reference at {location}; no referenced artifact was opened",
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_sealed_strings(child, role, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sealed_strings(child, role, f"{location}[{index}]")


def _regular_file(path: Path, role: str) -> Path:
    reject_sealed_path(path, role)
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ArtifactError("missing_artifact", f"missing {role}: {path}") from exc
    reject_sealed_path(resolved, role)
    if not resolved.is_file():
        raise ArtifactError("missing_artifact", f"{role} is not a regular file: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: Path, role: str) -> Dict[str, Any]:
    resolved = _regular_file(path, role)
    before = _stat_signature(resolved)
    digest = sha256_file(resolved)
    after = _stat_signature(resolved)
    if before != after:
        raise ArtifactError("artifact_changed", f"{role} changed while hashing: {resolved}")
    return {"path": str(resolved), "sha256": digest, "size_bytes": before[2]}


def load_json_artifact(path: Path, role: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resolved = _regular_file(path, role)
    before = _stat_signature(resolved)
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_json_constant,
            )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactError("invalid_json", f"{role} is not strict JSON: {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError("invalid_schema", f"{role} must contain a JSON object: {resolved}")
    digest = sha256_file(resolved)
    after = _stat_signature(resolved)
    if before != after:
        raise ArtifactError("artifact_changed", f"{role} changed while reading: {resolved}")
    return value, {"path": str(resolved), "sha256": digest, "size_bytes": before[2]}


def is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def exact_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        return None
    return int(numeric)


def numbers_close(left: Any, right: Any, rel_tol: float = 1e-5, abs_tol: float = 1e-6) -> bool:
    return is_finite_number(left) and is_finite_number(right) and math.isclose(
        float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol
    )


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ArtifactError(code, message)


def _sha256_field(value: Any, role: str) -> str:
    text = str(value or "").lower()
    if not SHA256_RE.fullmatch(text):
        raise ArtifactError("invalid_hash", f"{role} does not contain a lowercase SHA-256 digest")
    return text


def _recorded_path_matches(recorded: Any, expected: Path, bases: Sequence[Path]) -> bool:
    if not isinstance(recorded, str) or not recorded.strip():
        return False
    reject_sealed_path(recorded, "recorded provenance")
    value = Path(recorded)
    expected_resolved = expected.resolve(strict=False)
    candidates = [value] if value.is_absolute() else [base / value for base in (Path.cwd(), *bases)]
    return any(candidate.resolve(strict=False) == expected_resolved for candidate in candidates)


def _resolve_declared_file(recorded: Any, role: str, bases: Sequence[Path]) -> Path:
    if not isinstance(recorded, str) or not recorded.strip():
        raise ArtifactError("missing_provenance", f"{role} path is missing")
    reject_sealed_path(recorded, role)
    value = Path(recorded)
    candidates = [value] if value.is_absolute() else [base / value for base in (Path.cwd(), *bases)]
    existing: List[Path] = []
    for candidate in candidates:
        reject_sealed_path(candidate, role)
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if resolved.is_file() and resolved not in existing:
            existing.append(resolved)
    if len(existing) != 1:
        raise ArtifactError(
            "missing_provenance",
            f"{role} must resolve to exactly one local file, found {len(existing)} for {recorded!r}",
        )
    return existing[0]


def _normalized_modality(value: Any) -> str:
    text = str(value or "").lower()
    if "image" in text:
        return "image"
    if "speech" in text or "audio" in text:
        return "speech"
    if "text" in text:
        return "text"
    return text


def _nonnegative_int_list(value: Any, role: str) -> List[int]:
    if not isinstance(value, list) or not value:
        raise ArtifactError("routing_accounting", f"{role} must be a non-empty integer list")
    result: List[int] = []
    for item in value:
        parsed = exact_int(item)
        if parsed is None:
            raise ArtifactError("routing_accounting", f"{role} contains a non-integer count")
        result.append(parsed)
    return result


def _validate_routing_row(row: Mapping[str, Any], where: str) -> None:
    _require(row.get("capacity_enforced") is True, "capacity_compliance", f"{where}: capacity_enforced is not true")
    _require(exact_int(row.get("top_k")) == 2, "top_k", f"{where}: explicit top_k is not 2")
    _require(exact_int(row.get("runtime_top_k")) == 2, "top_k", f"{where}: runtime_top_k is not 2")
    _require(
        row.get("routing_accounting_source") == "patched_dispatch_masks_after_capacity",
        "routing_accounting",
        f"{where}: routing accounting is not sourced from patched post-capacity dispatch masks",
    )
    _require(
        row.get("routing_denominator") == "token_expert_assignments_across_layers",
        "routing_accounting",
        f"{where}: routing denominator is not explicit token-expert assignments",
    )
    attempted = exact_int(row.get("routing_attempted_assignments_total"))
    accepted = exact_int(row.get("routing_accepted_assignments_total"))
    dropped = exact_int(row.get("routing_dropped_assignments_total"))
    tokens = exact_int(row.get("routing_token_count_across_layers"))
    _require(None not in (attempted, accepted, dropped, tokens), "routing_accounting", f"{where}: missing exact routing totals")
    assert attempted is not None and accepted is not None and dropped is not None and tokens is not None
    _require(attempted == accepted + dropped, "routing_accounting", f"{where}: attempted != accepted + dropped")
    _require(attempted == tokens * 2, "routing_accounting", f"{where}: attempted assignments != tokens * Top-2")
    _require(row.get("routing_conservation_ok") is True, "routing_accounting", f"{where}: conservation flag is not true")
    _require(row.get("routing_capacity_compliant") is True, "capacity_compliance", f"{where}: capacity flag is not true")

    layers = row.get("routing_layer_accounting")
    _require(isinstance(layers, list) and bool(layers), "routing_accounting", f"{where}: layer accounting is missing")
    assert isinstance(layers, list)
    _require(exact_int(row.get("router_layers")) == len(layers), "routing_accounting", f"{where}: router layer count mismatch")
    layer_totals = [0, 0, 0]
    for index, layer in enumerate(layers):
        _require(isinstance(layer, Mapping), "routing_accounting", f"{where}: layer {index} is not an object")
        assert isinstance(layer, Mapping)
        prefix = f"{where} layer={index}"
        layer_tokens = exact_int(layer.get("token_count"))
        layer_attempted = exact_int(layer.get("attempted_assignments"))
        layer_accepted = exact_int(layer.get("accepted_assignments"))
        layer_dropped = exact_int(layer.get("dropped_assignments"))
        capacity = exact_int(layer.get("capacity_per_expert"))
        _require(
            None not in (layer_tokens, layer_attempted, layer_accepted, layer_dropped, capacity),
            "routing_accounting",
            f"{prefix}: missing exact counts",
        )
        assert layer_tokens is not None and layer_attempted is not None
        assert layer_accepted is not None and layer_dropped is not None and capacity is not None
        _require(exact_int(layer.get("top_k")) == 2, "top_k", f"{prefix}: top_k is not 2")
        _require(layer_attempted == layer_tokens * 2, "routing_accounting", f"{prefix}: attempted != tokens * 2")
        _require(layer_attempted == layer_accepted + layer_dropped, "routing_accounting", f"{prefix}: conservation failed")
        _require(layer.get("conservation_ok") is True, "routing_accounting", f"{prefix}: conservation flag is not true")
        _require(layer.get("capacity_compliant") is True, "capacity_compliance", f"{prefix}: capacity flag is not true")
        attempted_counts = _nonnegative_int_list(layer.get("attempted_expert_counts"), f"{prefix} attempted counts")
        accepted_counts = _nonnegative_int_list(layer.get("accepted_expert_counts"), f"{prefix} accepted counts")
        dropped_counts = _nonnegative_int_list(layer.get("dropped_expert_counts"), f"{prefix} dropped counts")
        _require(
            len(attempted_counts) == len(accepted_counts) == len(dropped_counts),
            "routing_accounting",
            f"{prefix}: per-expert vector lengths differ",
        )
        _require(
            all(a == b + d for a, b, d in zip(attempted_counts, accepted_counts, dropped_counts)),
            "routing_accounting",
            f"{prefix}: per-expert conservation failed",
        )
        _require(sum(attempted_counts) == layer_attempted, "routing_accounting", f"{prefix}: attempted vector sum mismatch")
        _require(sum(accepted_counts) == layer_accepted, "routing_accounting", f"{prefix}: accepted vector sum mismatch")
        _require(sum(dropped_counts) == layer_dropped, "routing_accounting", f"{prefix}: dropped vector sum mismatch")
        _require(max(accepted_counts) <= capacity, "capacity_compliance", f"{prefix}: accepted expert load exceeds capacity")
        layer_totals[0] += layer_attempted
        layer_totals[1] += layer_accepted
        layer_totals[2] += layer_dropped
    _require(layer_totals == [attempted, accepted, dropped], "routing_accounting", f"{where}: layer sums differ from aggregate totals")

    if _normalized_modality(row.get("modality")) in {"image", "speech"}:
        conservation = row.get("modality_assignment_conservation")
        _require(
            isinstance(conservation, Mapping) and bool(conservation) and all(value is True for value in conservation.values()),
            "routing_accounting",
            f"{where}: modality-prefix assignment conservation is not explicit and true",
        )


def _validate_objective_row(
    row: Mapping[str, Any], where: str, capacity_factor: float, aux_coef: float
) -> float:
    _require(row.get("optimizer_step") is True, "partial_training", f"{where}: optimizer_step is not true")
    _require(numbers_close(row.get("capacity_factor"), capacity_factor), "sweep_mismatch", f"{where}: capacity factor mismatch")
    _require(numbers_close(row.get("aux_coef"), aux_coef), "aux_coefficient", f"{where}: aux coefficient mismatch")
    required = (
        "lm_ce_loss",
        "ce_loss",
        "loss",
        "router_aux_loss_raw",
        "router_aux_loss_weighted",
        "hf_reported_loss",
        "hf_reported_loss_minus_explicit_base",
        "contrastive_coef",
        "contrastive_loss",
        "conditional_ranking_coef",
        "conditional_ranking_loss",
        "router_z_loss_coef",
        "router_z_loss",
    )
    missing = [name for name in required if not is_finite_number(row.get(name))]
    _require(not missing, "nonfinite_objective", f"{where}: missing/non-finite objective fields: {', '.join(missing)}")
    ce = float(row["lm_ce_loss"])
    _require(ce >= 0.0, "nonfinite_ce", f"{where}: LM CE is negative")
    _require(numbers_close(row.get("ce_loss"), ce), "objective_mismatch", f"{where}: CE alias mismatch")
    raw_aux = float(row["router_aux_loss_raw"])
    weighted_aux = float(row["router_aux_loss_weighted"])
    _require(
        numbers_close(weighted_aux, aux_coef * raw_aux),
        "aux_coefficient",
        f"{where}: weighted aux != configured aux_coef * raw aux",
    )
    _require(
        numbers_close(row.get("hf_reported_loss"), ce + weighted_aux),
        "aux_coefficient",
        f"{where}: HF base loss does not contain the effective weighted aux term exactly once",
    )
    reported_hf = float(row["hf_reported_loss"])
    logged_gap = float(row["hf_reported_loss_minus_explicit_base"])
    recomputed_gap = reported_hf - (ce + weighted_aux)
    _require(
        numbers_close(
            logged_gap,
            recomputed_gap,
            rel_tol=0.0,
            abs_tol=BASE_LOSS_GAP_FIELD_ABS_TOL,
        ),
        "objective_mismatch",
        f"{where}: logged HF/external base-loss gap disagrees with logged components",
    )
    _require(
        abs(logged_gap) <= BASE_LOSS_RECONCILIATION_ABS_TOL
        and abs(recomputed_gap) <= BASE_LOSS_RECONCILIATION_ABS_TOL,
        "objective_mismatch",
        f"{where}: HF/external base-loss reconciliation exceeds float32 tolerance",
    )
    equation = str(row.get("loss_equation", ""))
    _require(
        all(token in equation for token in ("lm_ce_loss", "router_aux_loss", "modality", "router_z_loss")),
        "objective_mismatch",
        f"{where}: explicit loss equation is incomplete",
    )
    expected_total = ce + weighted_aux
    for coefficient, loss in (
        ("contrastive_coef", "contrastive_loss"),
        ("conditional_ranking_coef", "conditional_ranking_loss"),
        ("router_z_loss_coef", "router_z_loss"),
    ):
        expected_total += float(row[coefficient]) * float(row[loss])
    _require(numbers_close(row.get("loss"), expected_total), "objective_mismatch", f"{where}: total loss mismatch")
    _validate_routing_row(row, where)
    return ce


def _stream_training_log(
    path: Path, expected_rows: Sequence[Mapping[str, Any]], role: str
) -> Tuple[Dict[str, Any], Optional[str]]:
    resolved = _regular_file(path, role)
    before = _stat_signature(resolved)
    digest = hashlib.sha256()
    mismatch: Optional[str] = None
    count = 0
    with resolved.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                mismatch = mismatch or f"blank line at {line_number}"
                continue
            try:
                row = _parse_json(raw.decode("utf-8"), resolved)
            except ArtifactError as exc:
                mismatch = mismatch or exc.message
                continue
            if not isinstance(row, dict):
                mismatch = mismatch or f"line {line_number} is not an object"
            elif count >= len(expected_rows):
                mismatch = mismatch or f"unexpected row at line {line_number}"
            elif row != expected_rows[count]:
                mismatch = mismatch or f"line {line_number} differs from embedded E3 step {count + 1}"
            count += 1
    after = _stat_signature(resolved)
    if before != after:
        raise ArtifactError("artifact_changed", f"{role} changed while reading: {resolved}")
    if count != len(expected_rows):
        mismatch = mismatch or f"JSONL rows={count}, embedded steps={len(expected_rows)}"
    fingerprint = {"path": str(resolved), "sha256": digest.hexdigest(), "size_bytes": before[2]}
    return fingerprint, mismatch


def _validate_ce_trends(values: Mapping[str, List[float]]) -> Dict[str, Any]:
    trends: Dict[str, Any] = {}
    for modality in ("text", "image", "speech"):
        observations = values.get(modality, [])
        if len(observations) < 6:
            raise ArtifactError("nondivergent_ce", f"{modality}: fewer than six explicit CE observations")
        width = max(3, min(100, len(observations) // 5))
        first = sum(observations[:width]) / width
        last = sum(observations[-width:]) / width
        if not math.isfinite(first) or not math.isfinite(last) or last > first:
            raise ArtifactError(
                "nondivergent_ce",
                f"{modality}: final CE window {last:.9g} exceeds initial window {first:.9g}",
            )
        trends[modality] = {
            "observations": len(observations),
            "window": width,
            "first_mean": first,
            "last_mean": last,
            "last_minus_first": last - first,
        }
    return trends


def _validate_manifest(candidate: Candidate) -> None:
    manifest_path = candidate.run_root / "manifest.json"
    manifest, fingerprint = load_json_artifact(manifest_path, f"{candidate.name} run manifest")
    _reject_sealed_strings(manifest, f"{candidate.name} run manifest")
    candidate.artifacts["run_manifest"] = fingerprint
    args = manifest.get("args")
    _require(isinstance(args, dict), "invalid_manifest", f"{candidate.name}: manifest.args is missing")
    assert isinstance(args, dict)
    _require(exact_int(args.get("final_steps")) == REQUIRED_STEPS, "partial_training", f"{candidate.name}: final_steps != {REQUIRED_STEPS}")
    _require(is_finite_number(args.get("capacity_factor")), "invalid_manifest", f"{candidate.name}: capacity_factor is missing")
    _require(is_finite_number(args.get("aux_coef")), "invalid_manifest", f"{candidate.name}: aux_coef is missing")
    capacity_factor = float(args["capacity_factor"])
    aux_coef = float(args["aux_coef"])
    _require(capacity_factor > 0 and aux_coef > 0, "invalid_manifest", f"{candidate.name}: swept values must be positive")
    candidate.metrics["capacity_factor"] = capacity_factor
    candidate.metrics["aux_coef"] = aux_coef

    if "output_dir" in args:
        _require(
            _recorded_path_matches(args["output_dir"], candidate.run_root, (candidate.run_root.parent,)),
            "invalid_manifest",
            f"{candidate.name}: args.output_dir does not identify the supplied run root",
        )
    if "feature_cache_dir" in args:
        _require(
            _recorded_path_matches(args["feature_cache_dir"], candidate.run_root / "feature_cache", (candidate.run_root, candidate.run_root.parent)),
            "invalid_manifest",
            f"{candidate.name}: args.feature_cache_dir is not run-local",
        )
    if "output_dir" in manifest:
        _require(
            _recorded_path_matches(manifest["output_dir"], candidate.run_root, (candidate.run_root.parent,)),
            "invalid_manifest",
            f"{candidate.name}: manifest.output_dir does not identify the supplied run root",
        )

    data_dir = manifest.get("data_dir")
    _require(isinstance(data_dir, str) and bool(data_dir), "dataset_provenance", f"{candidate.name}: data_dir is missing")
    _require(args.get("data_dir") == data_dir, "dataset_provenance", f"{candidate.name}: manifest/args data_dir mismatch")
    _require(isinstance(manifest.get("data_manifest"), dict) and bool(manifest["data_manifest"]), "dataset_provenance", f"{candidate.name}: data_manifest is missing")
    _require(isinstance(manifest.get("splits"), dict) and bool(manifest["splits"]), "dataset_provenance", f"{candidate.name}: split provenance is missing")

    normalized_args = copy.deepcopy(args)
    normalized_args.pop("capacity_factor", None)
    normalized_args.pop("aux_coef", None)
    for key in PATH_ONLY_ARGS:
        if key in normalized_args:
            normalized_args[key] = f"<{key}:candidate-local>"
    candidate.non_swept_args = normalized_args
    candidate.dataset_provenance = {
        key: copy.deepcopy(manifest.get(key))
        for key in ("base_model", "command_mode", "data_dir", "data_manifest", "speech_model", "splits", "vision_model")
    }
    candidate.manifest = manifest
    candidate.manifest_args = args


def _validate_built_in_retrieval(candidate: Candidate, retrieval: Any) -> Dict[str, Any]:
    _require(isinstance(retrieval, Mapping), "built_in_metrics", f"{candidate.name}: built-in retrieval metrics are missing")
    assert isinstance(retrieval, Mapping)
    required_exact = {
        "retrieval_path": "shared_olmoe_prefix_hidden",
        "retrieval_uses_lm_hidden_states": True,
        "retrieval_uses_direct_encoder_pooling": False,
        "conditional_eval_path": "shared_olmoe_prefix_lm_nll",
        "conditional_uses_lm_logits": True,
        "conditional_uses_direct_encoder_pooling": False,
        "conditional_candidates_per_query": 10,
        "conditional_image_eval_count": REQUIRED_QUERIES_PER_MODALITY,
        "conditional_speech_eval_count": REQUIRED_QUERIES_PER_MODALITY,
    }
    for key, expected in required_exact.items():
        _require(retrieval.get(key) == expected, "built_in_metrics", f"{candidate.name}: built-in {key} != {expected!r}")
    numeric_fields = (
        "image_to_text_r_at_1",
        "speech_to_text_r_at_1",
        "conditional_image_to_text_r_at_1",
        "conditional_speech_to_text_r_at_1",
        "conditional_image_chance_r_at_1",
        "conditional_speech_chance_r_at_1",
    )
    for key in numeric_fields:
        _require(is_finite_number(retrieval.get(key)), "built_in_metrics", f"{candidate.name}: built-in {key} is non-finite")
        _require(0.0 <= float(retrieval[key]) <= 1.0, "built_in_metrics", f"{candidate.name}: built-in {key} is outside [0,1]")
    _require(numbers_close(retrieval["conditional_image_chance_r_at_1"], 0.1), "built_in_metrics", f"{candidate.name}: built-in image chance mismatch")
    _require(numbers_close(retrieval["conditional_speech_chance_r_at_1"], 0.1), "built_in_metrics", f"{candidate.name}: built-in speech chance mismatch")
    return {key: retrieval[key] for key in numeric_fields}


def _validate_run(candidate: Candidate) -> None:
    assert candidate.manifest_args is not None
    args = candidate.manifest_args
    capacity_factor = float(args["capacity_factor"])
    aux_coef = float(args["aux_coef"])
    e3_path = candidate.run_root / E3_DIR / "metrics.json"
    e3, e3_fingerprint = load_json_artifact(e3_path, f"{candidate.name} final E3 metrics")
    _reject_sealed_strings(
        {key: value for key, value in e3.items() if key != "steps"},
        f"{candidate.name} final E3 metadata",
    )
    candidate.artifacts["e3_metrics"] = e3_fingerprint

    checkpoint_path = candidate.run_root / E3_DIR / "checkpoint_final.pt"
    checkpoint_fingerprint = fingerprint_file(checkpoint_path, f"{candidate.name} final checkpoint")
    candidate.artifacts["checkpoint"] = checkpoint_fingerprint
    candidate.checkpoint_sha256 = checkpoint_fingerprint["sha256"]
    _require(
        _recorded_path_matches(e3.get("checkpoint_path"), checkpoint_path, (candidate.run_root, candidate.run_root.parent)),
        "checkpoint_provenance",
        f"{candidate.name}: E3 checkpoint_path does not identify checkpoint_final.pt",
    )
    _require(
        exact_int(e3.get("checkpoint_size_bytes")) == checkpoint_fingerprint["size_bytes"],
        "checkpoint_provenance",
        f"{candidate.name}: E3 checkpoint size metadata mismatch",
    )

    meta = e3.get("meta")
    _require(isinstance(meta, Mapping), "invalid_e3", f"{candidate.name}: E3 meta is missing")
    assert isinstance(meta, Mapping)
    for key in ("top_k", "runtime_top_k"):
        _require(exact_int(meta.get(key)) == 2, "top_k", f"{candidate.name}: E3 meta.{key} is not 2")
    _require(meta.get("capacity_enforced") is True, "capacity_compliance", f"{candidate.name}: E3 capacity is not enforced")
    _require(numbers_close(meta.get("capacity_factor"), capacity_factor), "sweep_mismatch", f"{candidate.name}: E3 capacity mismatch")
    _require(numbers_close(meta.get("aux_coef"), aux_coef), "aux_coefficient", f"{candidate.name}: E3 aux mismatch")

    steps = e3.get("steps")
    _require(isinstance(steps, list), "partial_training", f"{candidate.name}: E3 steps is not a list")
    assert isinstance(steps, list)
    _require(len(steps) == REQUIRED_STEPS, "partial_training", f"{candidate.name}: E3 has {len(steps)} rows, expected {REQUIRED_STEPS}")
    values: Dict[str, List[float]] = {"text": [], "image": [], "speech": []}
    effective_aux_seen = False
    for index, row in enumerate(steps, 1):
        _require(isinstance(row, Mapping), "invalid_training_row", f"{candidate.name}: E3 row {index} is not an object")
        assert isinstance(row, Mapping)
        _require(exact_int(row.get("step")) == index, "partial_training", f"{candidate.name}: expected step {index}, found {row.get('step')!r}")
        _require(row.get("experiment_id") == E3_DIR, "invalid_training_row", f"{candidate.name}: step {index} experiment_id mismatch")
        ce = _validate_objective_row(row, f"{candidate.name} step={index}", capacity_factor, aux_coef)
        modality = _normalized_modality(row.get("modality"))
        _require(modality in values, "invalid_training_row", f"{candidate.name}: step {index} has unknown modality {row.get('modality')!r}")
        values[modality].append(ce)
        if abs(float(row["router_aux_loss_raw"])) > 0 and abs(float(row["router_aux_loss_weighted"])) > 0:
            effective_aux_seen = True
    _require(effective_aux_seen, "aux_coefficient", f"{candidate.name}: no nonzero effective auxiliary term was observed")
    trends = _validate_ce_trends(values)
    candidate.validation["ce_trends"] = trends
    candidate.validation["completed_optimizer_steps"] = len(steps)

    train_log_fingerprint, mismatch = _stream_training_log(
        candidate.run_root / E3_DIR / "train_metrics.jsonl",
        steps,
        f"{candidate.name} E3 training JSONL",
    )
    candidate.artifacts["e3_train_metrics"] = train_log_fingerprint
    _require(mismatch is None, "mismatched_metrics", f"{candidate.name}: {mismatch}")

    first = steps[0]
    last = steps[-1]
    _require(numbers_close(e3.get("first_loss"), first.get("loss")), "mismatched_metrics", f"{candidate.name}: first_loss mismatch")
    _require(numbers_close(e3.get("last_loss"), last.get("loss")), "mismatched_metrics", f"{candidate.name}: last_loss mismatch")
    _require(
        numbers_close(e3.get("min_loss"), min(float(row["loss"]) for row in steps)),
        "mismatched_metrics",
        f"{candidate.name}: min_loss mismatch",
    )
    _require(
        numbers_close(e3.get("final_capacity_overflow_ratio_mean"), last.get("capacity_overflow_ratio_mean")),
        "mismatched_metrics",
        f"{candidate.name}: final overflow mismatch",
    )
    _require(
        numbers_close(e3.get("final_inactive_expert_ratio_mean"), last.get("inactive_expert_ratio_mean")),
        "mismatched_metrics",
        f"{candidate.name}: final inactive ratio mismatch",
    )
    for key in ("capacity_overflow_ratio_mean", "inactive_expert_ratio_mean", "accepted_load_cv"):
        _require(is_finite_number(last.get(key)), "routing_metrics", f"{candidate.name}: final {key} is non-finite")
        _require(float(last[key]) >= 0.0, "routing_metrics", f"{candidate.name}: final {key} is negative")
    _require(float(last["capacity_overflow_ratio_mean"]) <= 1.0, "routing_metrics", f"{candidate.name}: final overflow exceeds 1")
    _require(float(last["inactive_expert_ratio_mean"]) <= 1.0, "routing_metrics", f"{candidate.name}: final inactive ratio exceeds 1")

    text_path = candidate.run_root / E3_TEXT_DIR / "metrics.json"
    text_metrics, text_fingerprint = load_json_artifact(text_path, f"{candidate.name} built-in E3 text metrics")
    _reject_sealed_strings(text_metrics, f"{candidate.name} built-in E3 text metrics")
    candidate.artifacts["built_in_text_metrics"] = text_fingerprint
    embedded_text = e3.get("text_eval")
    _require(isinstance(embedded_text, Mapping), "built_in_metrics", f"{candidate.name}: embedded text_eval is missing")
    independent_comparable = {key: value for key, value in text_metrics.items() if key != "expert_counts_total"}
    _require(independent_comparable == embedded_text, "mismatched_metrics", f"{candidate.name}: independent and embedded E3 text metrics differ")
    provenance = text_metrics.get("provenance")
    _require(isinstance(provenance, Mapping), "checkpoint_provenance", f"{candidate.name}: text checkpoint provenance is missing")
    assert isinstance(provenance, Mapping)
    _require(provenance == e3.get("text_eval_provenance"), "mismatched_metrics", f"{candidate.name}: text provenance copies differ")
    _require(provenance.get("source_experiment_id") == E3_DIR, "checkpoint_provenance", f"{candidate.name}: text source experiment mismatch")
    _require(exact_int(provenance.get("source_training_steps")) == REQUIRED_STEPS, "partial_training", f"{candidate.name}: text source steps != {REQUIRED_STEPS}")
    _require(provenance.get("source_checkpoint_saved_before_eval") is True, "checkpoint_provenance", f"{candidate.name}: checkpoint was not saved before text eval")
    _require(provenance.get("copied_from_e2") is False, "copied_metrics", f"{candidate.name}: text metrics declare copied_from_e2")
    _require(provenance.get("lm_trainable") is True, "checkpoint_provenance", f"{candidate.name}: corrected E3 text provenance is not trainable-LM")
    _require(
        provenance.get("model_state_source") == "in_memory_wrapper_after_training_saved_to_checkpoint",
        "checkpoint_provenance",
        f"{candidate.name}: unexpected text model_state_source",
    )
    _require(
        _recorded_path_matches(provenance.get("source_checkpoint"), checkpoint_path, (candidate.run_root, candidate.run_root.parent)),
        "checkpoint_provenance",
        f"{candidate.name}: text provenance checkpoint mismatch",
    )
    _require(
        exact_int(provenance.get("source_checkpoint_size_bytes")) == checkpoint_fingerprint["size_bytes"],
        "checkpoint_provenance",
        f"{candidate.name}: text provenance checkpoint size mismatch",
    )
    for key in ("top_k", "runtime_top_k"):
        _require(exact_int(text_metrics.get(key)) == 2, "top_k", f"{candidate.name}: text metrics {key} is not 2")
    _require(text_metrics.get("capacity_enforced") is True, "capacity_compliance", f"{candidate.name}: text metrics capacity is not enforced")
    _require(numbers_close(text_metrics.get("capacity_factor"), capacity_factor), "sweep_mismatch", f"{candidate.name}: text capacity mismatch")
    _require(numbers_close(text_metrics.get("aux_coef"), aux_coef), "aux_coefficient", f"{candidate.name}: text aux mismatch")
    ppl = text_metrics.get("perplexity")
    text_loss = text_metrics.get("loss")
    _require(is_finite_number(ppl) and float(ppl) > 0, "invalid_ppl", f"{candidate.name}: text PPL is non-finite")
    _require(is_finite_number(text_loss) and float(text_loss) >= 0, "invalid_ppl", f"{candidate.name}: text loss is non-finite")
    _require(numbers_close(math.exp(float(text_loss)), float(ppl)), "mismatched_metrics", f"{candidate.name}: PPL != exp(text loss)")
    _require(float(ppl) <= 20.0, "ppl_threshold", f"{candidate.name}: text PPL {float(ppl):.9g} exceeds 20")

    built_in = _validate_built_in_retrieval(candidate, e3.get("retrieval_eval"))
    candidate.metrics.update(
        {
            "text_ppl": float(ppl),
            "text_loss": float(text_loss),
            "routing_overflow": float(last["capacity_overflow_ratio_mean"]),
            "routing_inactive": float(last["inactive_expert_ratio_mean"]),
            "routing_load_cv": float(last["accepted_load_cv"]),
            "built_in": built_in,
        }
    )


def _uid_for_manifest_row(row: Mapping[str, Any], modality: str, index: int) -> str:
    for key in ("uid", "source_uid", "image_uid", "utterance_id", "source_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{modality}:{value}"
    return f"{modality}:{row.get('source', 'unknown')}:{row.get('id', index)}"


def _read_development_rows(path: Path, modality: str) -> Tuple[List[str], Dict[str, Any]]:
    resolved = _regular_file(path, f"development {modality} manifest")
    before = _stat_signature(resolved)
    digest = hashlib.sha256()
    uids: List[str] = []
    with resolved.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                raise ArtifactError("development_manifest", f"{resolved}: blank line {line_number}")
            row = _parse_json(raw.decode("utf-8"), resolved)
            _require(isinstance(row, Mapping), "development_manifest", f"{resolved}: line {line_number} is not an object")
            assert isinstance(row, Mapping)
            uid = _uid_for_manifest_row(row, modality, len(uids))
            _require(uid not in uids, "development_manifest", f"{resolved}: duplicate development UID {uid}")
            uids.append(uid)
    after = _stat_signature(resolved)
    if before != after:
        raise ArtifactError("artifact_changed", f"development manifest changed while reading: {resolved}")
    _require(
        len(uids) == REQUIRED_QUERIES_PER_MODALITY,
        "development_manifest",
        f"{resolved}: rows={len(uids)}, expected {REQUIRED_QUERIES_PER_MODALITY}",
    )
    return uids, {"path": str(resolved), "sha256": digest.hexdigest(), "size_bytes": before[2]}


def _load_shared_development_data(
    cache: ArtifactCache,
    image_path: Path,
    speech_path: Path,
    expected_image_hash: str,
    expected_speech_hash: str,
    candidate: Candidate,
) -> SharedDevelopmentData:
    key = (str(image_path.resolve()), str(speech_path.resolve()))
    cached = cache.development.get(key)
    if cached is not None:
        _require(cached.image_fingerprint["sha256"] == expected_image_hash, "development_manifest", "image manifest hash differs across eval cells")
        _require(cached.speech_fingerprint["sha256"] == expected_speech_hash, "development_manifest", "speech manifest hash differs across eval cells")
        return cached

    _require(image_path.parent.resolve() == speech_path.parent.resolve(), "development_manifest", "image/speech development manifests do not share a root")
    _require("test" not in image_path.name.lower() and "test" not in speech_path.name.lower(), "development_manifest", "selector accepts validation/development manifests only")
    image_uids, image_fingerprint = _read_development_rows(image_path, "image")
    speech_uids, speech_fingerprint = _read_development_rows(speech_path, "speech")
    _require(image_fingerprint["sha256"] == expected_image_hash, "development_manifest", "image manifest content hash mismatch")
    _require(speech_fingerprint["sha256"] == expected_speech_hash, "development_manifest", "speech manifest content hash mismatch")

    manifest_path = image_path.parent / "manifest.json"
    manifest, manifest_fingerprint = load_json_artifact(manifest_path, "development split manifest")
    _reject_sealed_strings(manifest, "development split manifest")
    _require(exact_int(manifest.get("val_count")) == REQUIRED_QUERIES_PER_MODALITY, "development_manifest", "development val_count mismatch")
    files = manifest.get("files")
    _require(isinstance(files, Mapping), "development_manifest", "development manifest.files is missing")
    assert isinstance(files, Mapping)
    _require(
        _recorded_path_matches(files.get("image_val"), image_path, (manifest_path.parent, manifest_path.parent.parent)),
        "development_manifest",
        "development manifest image_val path mismatch",
    )
    _require(
        _recorded_path_matches(files.get("speech_val"), speech_path, (manifest_path.parent, manifest_path.parent.parent)),
        "development_manifest",
        "development manifest speech_val path mismatch",
    )
    compatible = manifest.get("compatible_conditional_eval")
    _require(isinstance(compatible, Mapping), "development_manifest", "development compatibility record is missing")
    validation = compatible.get("validation") if isinstance(compatible, Mapping) else None
    _require(isinstance(validation, Mapping), "development_manifest", "development validation protocol is missing")
    assert isinstance(validation, Mapping)
    _require(exact_int(validation.get("CONDITIONAL_QUERIES")) == REQUIRED_QUERIES_PER_MODALITY, "development_manifest", "development query count mismatch")
    _require(exact_int(validation.get("QUERY_OFFSET")) == 0, "development_manifest", "development query offset is not zero")
    data_dir = manifest.get("data_dir")
    _require(isinstance(data_dir, str) and bool(data_dir), "development_manifest", "development data_dir is missing")
    _require(candidate.manifest_args is not None and candidate.manifest_args.get("data_dir") == data_dir, "development_manifest", f"{candidate.name}: development/training data_dir mismatch")
    shared = SharedDevelopmentData(
        manifest_path=manifest_path.resolve(),
        manifest_fingerprint=manifest_fingerprint,
        image_path=image_path.resolve(),
        image_fingerprint=image_fingerprint,
        image_uids=image_uids,
        speech_path=speech_path.resolve(),
        speech_fingerprint=speech_fingerprint,
        speech_uids=speech_uids,
        data_dir=data_dir,
    )
    cache.development[key] = shared
    return shared


def _validate_per_query(
    path: Path,
    role: str,
    candidate_count: int,
    metrics: Mapping[str, Any],
    shared: SharedDevelopmentData,
    expected_protocol: Mapping[str, Any] = EXPECTED_PROTOCOL,
) -> Tuple[Dict[str, Any], Dict[str, float], str, str]:
    resolved = _regular_file(path, role)
    before = _stat_signature(resolved)
    digest = hashlib.sha256()
    rows: List[Dict[str, Any]] = []
    with resolved.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                raise ArtifactError("per_query_mismatch", f"{role}: blank line {line_number}")
            row = _parse_json(raw.decode("utf-8"), resolved)
            _require(isinstance(row, dict), "per_query_mismatch", f"{role}: line {line_number} is not an object")
            rows.append(row)
    after = _stat_signature(resolved)
    if before != after:
        raise ArtifactError("artifact_changed", f"{role} changed while reading: {resolved}")
    fingerprint = {"path": str(resolved), "sha256": digest.hexdigest(), "size_bytes": before[2]}
    _require(len(rows) == 2 * REQUIRED_QUERIES_PER_MODALITY, "per_query_mismatch", f"{role}: expected 500 rows, found {len(rows)}")
    _require(exact_int(metrics.get("per_query_rows")) == len(rows), "per_query_mismatch", f"{role}: metrics per_query_rows mismatch")
    _require(_sha256_field(metrics.get("per_query_sha256"), role) == fingerprint["sha256"], "per_query_mismatch", f"{role}: metrics per_query_sha256 mismatch")
    _require(
        _recorded_path_matches(metrics.get("per_query_output"), resolved, (resolved.parent, resolved.parent.parent)),
        "per_query_mismatch",
        f"{role}: per_query_output path mismatch",
    )

    expected_uids = {"image": shared.image_uids, "speech": shared.speech_uids}
    counts = Counter()
    successes = Counter()
    positions: Dict[str, List[int]] = {"image": [0] * candidate_count, "speech": [0] * candidate_count}
    seen_queries: Dict[str, set[int]] = {"image": set(), "speech": set()}
    protocol_rows: List[Dict[str, Any]] = []
    result_rows: List[Dict[str, Any]] = []
    for line_number, row in enumerate(rows, 1):
        modality = str(row.get("modality", ""))
        _require(modality in expected_uids, "per_query_mismatch", f"{role}: row {line_number} has invalid modality")
        query_index = exact_int(row.get("query_index"))
        _require(query_index is not None and query_index < REQUIRED_QUERIES_PER_MODALITY, "per_query_mismatch", f"{role}: row {line_number} query_index invalid")
        assert query_index is not None
        _require(query_index not in seen_queries[modality], "per_query_mismatch", f"{role}: duplicate {modality} query index {query_index}")
        seen_queries[modality].add(query_index)
        query_uid = row.get("query_uid")
        _require(query_uid == expected_uids[modality][query_index], "per_query_mismatch", f"{role}: row {line_number} query UID disagrees with development manifest")
        candidate_ids = row.get("candidate_ids")
        scores = row.get("scores")
        _require(isinstance(candidate_ids, list) and len(candidate_ids) == candidate_count, "candidate_count", f"{role}: row {line_number} candidate count mismatch")
        _require(len(set(candidate_ids)) == candidate_count, "candidate_count", f"{role}: row {line_number} has duplicate candidate IDs")
        _require(all(value in set(expected_uids[modality]) for value in candidate_ids), "development_manifest", f"{role}: row {line_number} contains a candidate outside the fixed manifest")
        _require(isinstance(scores, list) and len(scores) == candidate_count and all(is_finite_number(value) for value in scores), "per_query_mismatch", f"{role}: row {line_number} scores invalid")
        _require(exact_int(row.get("candidate_count")) == candidate_count, "candidate_count", f"{role}: row {line_number} candidate_count field mismatch")
        gold = exact_int(row.get("gold_position"))
        predicted = exact_int(row.get("predicted_position"))
        rank = exact_int(row.get("rank"))
        _require(gold is not None and gold < candidate_count, "positive_position", f"{role}: row {line_number} gold position invalid")
        _require(predicted is not None and predicted < candidate_count, "per_query_mismatch", f"{role}: row {line_number} prediction invalid")
        _require(rank is not None and rank < candidate_count, "per_query_mismatch", f"{role}: row {line_number} rank invalid")
        assert gold is not None and predicted is not None and rank is not None
        _require(exact_int(row.get("candidate_index")) == gold, "positive_position", f"{role}: row {line_number} candidate_index mismatch")
        _require(candidate_ids[gold] == query_uid == row.get("gold_candidate_id"), "per_query_mismatch", f"{role}: row {line_number} gold candidate mismatch")
        expected_predicted = max(range(candidate_count), key=lambda index: float(scores[index]))
        expected_rank = sorted(range(candidate_count), key=lambda index: float(scores[index]), reverse=True).index(gold)
        _require(predicted == expected_predicted and row.get("predicted_candidate_id") == candidate_ids[predicted], "per_query_mismatch", f"{role}: row {line_number} prediction disagrees with scores")
        _require(rank == expected_rank, "per_query_mismatch", f"{role}: row {line_number} rank disagrees with scores")
        expected_set_hash = hashlib.sha256(
            json.dumps(candidate_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        _require(row.get("candidate_set_hash") == expected_set_hash, "per_query_mismatch", f"{role}: row {line_number} candidate-set hash mismatch")
        _require(row.get("rank_base") == 0 and row.get("score_direction") == "higher_is_better", "per_query_mismatch", f"{role}: row {line_number} rank semantics mismatch")
        for key in ("eval_split_name", "negative_mode", "eval_path", "prefix_control", "condition"):
            _require(row.get(key) == expected_protocol[key], "development_protocol", f"{role}: row {line_number} {key} mismatch")
        protocol = row.get("protocol")
        _require(isinstance(protocol, Mapping), "development_protocol", f"{role}: row {line_number} protocol is missing")
        assert isinstance(protocol, Mapping)
        _require(protocol.get("name") == expected_protocol["protocol_name"], "development_protocol", f"{role}: row {line_number} protocol name mismatch")
        _require(protocol.get("manifest_sha256") is None, "development_protocol", f"{role}: row {line_number} unexpectedly references a frozen/sealed protocol")
        _require(protocol.get("eval_split_name") == expected_protocol["eval_split_name"], "development_protocol", f"{role}: row {line_number} split mismatch")
        _require(protocol.get("negative_mode") == expected_protocol["negative_mode"], "development_protocol", f"{role}: row {line_number} negative mode mismatch")
        _require(protocol.get("hard_negative_selector") == expected_protocol["hard_negative_selector"], "development_protocol", f"{role}: row {line_number} hard-negative selector mismatch")
        _require(exact_int(protocol.get("candidate_count")) == candidate_count, "candidate_count", f"{role}: row {line_number} protocol candidate count mismatch")
        _require(exact_int(protocol.get("candidate_seed")) == expected_protocol["candidate_seed"], "development_protocol", f"{role}: row {line_number} candidate seed mismatch")
        _require(protocol.get("randomized_positive_position") is True, "positive_position", f"{role}: row {line_number} positive position is not randomized")
        _require(protocol.get("rank_base") == 0 and protocol.get("score_direction") == "higher_is_better", "development_protocol", f"{role}: row {line_number} protocol rank semantics mismatch")
        counts[modality] += 1
        successes[modality] += int(rank == 0)
        positions[modality][gold] += 1
        protocol_rows.append(
            {
                "modality": modality,
                "query_uid": query_uid,
                "query_index": query_index,
                "candidate_ids": candidate_ids,
                "candidate_set_hash": expected_set_hash,
                "gold_position": gold,
                "candidate_count": candidate_count,
                "protocol": protocol,
                "condition": row.get("condition"),
                "prefix_control": row.get("prefix_control"),
                "negative_mode": row.get("negative_mode"),
                "eval_path": row.get("eval_path"),
            }
        )
        result_rows.append(
            {
                "modality": modality,
                "query_index": query_index,
                "scores": scores,
                "rank": rank,
                "predicted_position": predicted,
            }
        )
    for modality in ("image", "speech"):
        _require(counts[modality] == REQUIRED_QUERIES_PER_MODALITY, "query_count", f"{role}: {modality} query count mismatch")
        _require(seen_queries[modality] == set(range(REQUIRED_QUERIES_PER_MODALITY)), "query_count", f"{role}: {modality} query indices are incomplete")
        _require(max(positions[modality]) - min(positions[modality]) <= 1, "positive_position", f"{role}: {modality} positive positions are imbalanced")
        metric_counts = metrics.get(f"{modality}_gold_position_counts")
        _require(metric_counts == positions[modality], "positive_position", f"{role}: {modality} position ledger mismatch")
    recalls = {
        "image": successes["image"] / REQUIRED_QUERIES_PER_MODALITY,
        "speech": successes["speech"] / REQUIRED_QUERIES_PER_MODALITY,
    }
    protocol_rows.sort(key=lambda row: (row["modality"], row["query_index"]))
    result_rows.sort(key=lambda row: (row["modality"], row["query_index"]))
    return fingerprint, recalls, canonical_sha256(protocol_rows), canonical_sha256(result_rows)


def _validate_eval_cell(
    candidate: Candidate,
    candidate_count: int,
    cache: ArtifactCache,
    *,
    cell_name: Optional[str] = None,
    negative_mode: str = "random",
    metric_label: Optional[str] = None,
) -> None:
    cell_name = cell_name or f"r{candidate_count}"
    metric_label = metric_label or f"{candidate_count}way"
    protocol_key: Any = candidate_count if cell_name == f"r{candidate_count}" else cell_name
    expected_protocol = {
        **EXPECTED_PROTOCOL,
        "negative_mode": negative_mode,
        "hard_negative_selector": (
            "lexical_jaccard_v1" if negative_mode == "hard_text" else None
        ),
    }
    cell_root = candidate.dev_root / cell_name
    reject_sealed_path(cell_root, f"{candidate.name} {cell_name} development cell")
    metrics, metrics_fingerprint = load_json_artifact(
        cell_root / "metrics.json", f"{candidate.name} {candidate_count}-way development metrics"
    )
    _reject_sealed_strings(metrics, f"{candidate.name} {candidate_count}-way development metrics")
    candidate.artifacts[f"dev_{metric_label}_metrics"] = metrics_fingerprint
    expected = {
        **expected_protocol,
        "mode": "conditional_nll_local_negatives",
        "conditional_eval_path": "shared_olmoe_prefix_lm_nll",
        "conditional_uses_lm_logits": True,
        "conditional_uses_direct_encoder_pooling": False,
        "conditional_uses_multimodal_prefix": True,
        "sealed_protocol": False,
        "image_split_source": "explicit_manifest",
        "speech_split_source": "explicit_manifest",
        "randomized_positive_position": True,
        "query_offset": 0,
        "candidate_offset": -1,
    }
    for key, value in expected.items():
        _require(metrics.get(key) == value, "development_protocol", f"{candidate.name} {candidate_count}-way: {key} != {value!r}")
    _require(metrics.get("protocol_manifest_path") is None and metrics.get("protocol_manifest_sha256") is None, "development_protocol", f"{candidate.name} {candidate_count}-way: development eval references a frozen protocol")
    for key in (
        "candidate_count",
        "speech_candidate_count",
        "conditional_candidates_per_query",
        "conditional_speech_candidates_per_query",
    ):
        _require(exact_int(metrics.get(key)) == candidate_count, "candidate_count", f"{candidate.name} {candidate_count}-way: {key} mismatch")
    for key in (
        "image_eval_count",
        "speech_eval_count",
        "conditional_image_eval_count",
        "conditional_speech_eval_count",
    ):
        _require(exact_int(metrics.get(key)) == REQUIRED_QUERIES_PER_MODALITY, "query_count", f"{candidate.name} {candidate_count}-way: {key} mismatch")
    chance = 1.0 / candidate_count
    for key in (
        "image_chance_r_at_1",
        "speech_chance_r_at_1",
        "conditional_image_chance_r_at_1",
        "conditional_speech_chance_r_at_1",
    ):
        _require(numbers_close(metrics.get(key), chance), "candidate_count", f"{candidate.name} {candidate_count}-way: {key} is not exact chance")

    provenance = metrics.get("provenance")
    _require(isinstance(provenance, Mapping), "development_provenance", f"{candidate.name} {candidate_count}-way: provenance is missing")
    assert isinstance(provenance, Mapping)
    checkpoint_hash = _sha256_field(provenance.get("checkpoint_sha256"), "development checkpoint provenance")
    _require(checkpoint_hash == candidate.checkpoint_sha256, "checkpoint_hash", f"{candidate.name} {candidate_count}-way: checkpoint hash mismatch")
    checkpoint_path = candidate.run_root / E3_DIR / "checkpoint_final.pt"
    _require(
        _recorded_path_matches(provenance.get("checkpoint_path"), checkpoint_path, (candidate.run_root, candidate.run_root.parent)),
        "checkpoint_provenance",
        f"{candidate.name} {candidate_count}-way: checkpoint path mismatch",
    )
    _require(
        _recorded_path_matches(metrics.get("checkpoint"), checkpoint_path, (candidate.run_root, candidate.run_root.parent)),
        "checkpoint_provenance",
        f"{candidate.name} {candidate_count}-way: metrics checkpoint path mismatch",
    )
    _require(
        _recorded_path_matches(metrics.get("run_output_dir"), candidate.run_root, (candidate.run_root.parent,)),
        "development_provenance",
        f"{candidate.name} {candidate_count}-way: run_output_dir mismatch",
    )
    manifest_hash = _sha256_field(provenance.get("source_run_manifest_sha256"), "development source manifest provenance")
    _require(manifest_hash == candidate.artifacts["run_manifest"]["sha256"], "copied_metrics", f"{candidate.name} {candidate_count}-way: source run manifest hash mismatch")
    _require(
        _recorded_path_matches(provenance.get("source_run_manifest_path"), candidate.run_root / "manifest.json", (candidate.run_root, candidate.run_root.parent)),
        "development_provenance",
        f"{candidate.name} {candidate_count}-way: source run manifest path mismatch",
    )
    _require(provenance.get("protocol_manifest_path") is None and provenance.get("protocol_manifest_sha256") is None, "development_protocol", f"{candidate.name} {candidate_count}-way: provenance references a frozen protocol")

    image_path = _resolve_declared_file(
        provenance.get("image_manifest_path"),
        "development image manifest",
        (candidate.dev_root, candidate.run_root, candidate.run_root.parent),
    )
    speech_path = _resolve_declared_file(
        provenance.get("speech_manifest_path"),
        "development speech manifest",
        (candidate.dev_root, candidate.run_root, candidate.run_root.parent),
    )
    image_hash = _sha256_field(provenance.get("image_manifest_sha256"), "development image manifest provenance")
    speech_hash = _sha256_field(provenance.get("speech_manifest_sha256"), "development speech manifest provenance")
    shared = _load_shared_development_data(cache, image_path, speech_path, image_hash, speech_hash, candidate)
    candidate.artifacts["development_split_manifest"] = shared.manifest_fingerprint
    candidate.artifacts["development_image_manifest"] = shared.image_fingerprint
    candidate.artifacts["development_speech_manifest"] = shared.speech_fingerprint

    evaluator_path = _resolve_declared_file(
        provenance.get("evaluator_path"),
        "development evaluator",
        (candidate.dev_root, candidate.run_root, Path.cwd()),
    )
    evaluator_fingerprint = fingerprint_file(evaluator_path, "development evaluator")
    _require(
        evaluator_fingerprint["sha256"] == _sha256_field(provenance.get("evaluator_sha256"), "development evaluator provenance"),
        "development_protocol",
        f"{candidate.name} {candidate_count}-way: evaluator hash mismatch",
    )
    candidate.artifacts[f"dev_{metric_label}_evaluator"] = evaluator_fingerprint

    meta = metrics.get("meta")
    _require(isinstance(meta, Mapping), "development_provenance", f"{candidate.name} {candidate_count}-way: checkpoint meta is missing")
    assert isinstance(meta, Mapping)
    _require(exact_int(meta.get("top_k")) == 2 and exact_int(meta.get("runtime_top_k")) == 2, "top_k", f"{candidate.name} {candidate_count}-way: checkpoint meta is not Top-2")
    _require(meta.get("capacity_enforced") is True, "capacity_compliance", f"{candidate.name} {candidate_count}-way: capacity is not enforced")
    _require(numbers_close(meta.get("capacity_factor"), candidate.metrics["capacity_factor"]), "sweep_mismatch", f"{candidate.name} {candidate_count}-way: capacity mismatch")
    _require(numbers_close(meta.get("aux_coef"), candidate.metrics["aux_coef"]), "aux_coefficient", f"{candidate.name} {candidate_count}-way: aux mismatch")

    per_query_fingerprint, recalls, protocol_digest, result_digest = _validate_per_query(
        cell_root / "per_query.jsonl",
        f"{candidate.name} {candidate_count}-way per-query metrics",
        candidate_count,
        metrics,
        shared,
        expected_protocol,
    )
    candidate.artifacts[f"dev_{metric_label}_per_query"] = per_query_fingerprint
    for modality in ("image", "speech"):
        keys = (
            f"{modality}_to_text_r_at_1",
            f"conditional_{modality}_to_text_r_at_1",
        )
        for key in keys:
            _require(numbers_close(metrics.get(key), recalls[modality]), "mismatched_metrics", f"{candidate.name} {candidate_count}-way: {key} disagrees with per-query ranks")
    candidate.metrics[f"dev_{metric_label}_image_r1"] = recalls["image"]
    candidate.metrics[f"dev_{metric_label}_speech_r1"] = recalls["speech"]
    candidate.metrics[f"dev_{metric_label}_chance"] = chance
    candidate.protocol_digests[protocol_key] = canonical_sha256(
        {
            "candidate_count": candidate_count,
            "fixed_protocol": {key: metrics.get(key) for key in sorted(expected)},
            "image_manifest": shared.image_fingerprint,
            "speech_manifest": shared.speech_fingerprint,
            "development_manifest": shared.manifest_fingerprint,
            "evaluator_sha256": evaluator_fingerprint["sha256"],
            "query_protocol_sha256": protocol_digest,
        }
    )
    candidate.result_digests[protocol_key] = result_digest


def _validate_development(candidate: Candidate, cache: ArtifactCache) -> None:
    _validate_eval_cell(candidate, 5, cache)
    _validate_eval_cell(candidate, 10, cache)
    _require(
        candidate.artifacts["dev_5way_evaluator"]["sha256"] == candidate.artifacts["dev_10way_evaluator"]["sha256"],
        "development_protocol",
        f"{candidate.name}: 5-way and 10-way evaluator hashes differ",
    )
    for role in ("development_split_manifest", "development_image_manifest", "development_speech_manifest"):
        _require(role in candidate.artifacts, "development_manifest", f"{candidate.name}: {role} provenance is missing")


def _safe_phase(candidate: Candidate, function: Any, *args: Any) -> None:
    try:
        function(candidate, *args)
    except ArtifactError as exc:
        candidate.reject(exc.code, exc.message)
    except (OSError, TypeError, KeyError, ValueError, OverflowError) as exc:
        candidate.reject("unexpected_artifact_error", f"{candidate.name}: {type(exc).__name__}: {exc}")


def _mode_digest(candidates: Sequence[Candidate], attribute: str) -> Optional[str]:
    values = [canonical_sha256(getattr(candidate, attribute)) for candidate in candidates if getattr(candidate, attribute) is not None]
    if not values:
        return None
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _cross_validate(candidates: Sequence[Candidate]) -> Dict[str, Any]:
    observed_sweep = {
        (round(float(candidate.metrics["capacity_factor"]), 12), round(float(candidate.metrics["aux_coef"]), 12))
        for candidate in candidates
        if "capacity_factor" in candidate.metrics and "aux_coef" in candidate.metrics
    }
    expected_rounded = {(round(capacity, 12), round(aux, 12)) for capacity, aux in EXPECTED_SWEEP}
    if observed_sweep != expected_rounded or len(observed_sweep) != REQUIRED_CANDIDATES:
        message = f"corrected sweep mismatch: observed={sorted(observed_sweep)}, expected={sorted(expected_rounded)}"
        for candidate in candidates:
            candidate.reject("sweep_matrix_mismatch", message)

    shared: Dict[str, Any] = {
        "expected_sweep": [
            {"capacity_factor": capacity, "aux_coef": aux}
            for capacity, aux in sorted(EXPECTED_SWEEP)
        ],
        "observed_sweep": [
            {"capacity_factor": capacity, "aux_coef": aux}
            for capacity, aux in sorted(observed_sweep)
        ],
    }
    for attribute, code in (
        ("non_swept_args", "non_swept_args_mismatch"),
        ("dataset_provenance", "dataset_provenance_mismatch"),
    ):
        mode = _mode_digest(candidates, attribute)
        shared[f"canonical_{attribute}_sha256"] = mode
        if mode is None:
            continue
        for candidate in candidates:
            value = getattr(candidate, attribute)
            if value is not None and canonical_sha256(value) != mode:
                candidate.reject(code, f"{candidate.name}: {attribute} differs from the fixed candidate set")

    for count in (5, 10):
        digests = [candidate.protocol_digests[count] for candidate in candidates if count in candidate.protocol_digests]
        if digests:
            counts = Counter(digests)
            mode = sorted(counts, key=lambda value: (-counts[value], value))[0]
            shared[f"development_{count}way_protocol_sha256"] = mode
            for candidate in candidates:
                if count in candidate.protocol_digests and candidate.protocol_digests[count] != mode:
                    candidate.reject(
                        "development_protocol_mismatch",
                        f"{candidate.name}: {count}-way query/candidate protocol differs from the fixed candidate set",
                    )

    result_groups: Dict[Tuple[str, str], List[Candidate]] = {}
    for candidate in candidates:
        if 5 in candidate.result_digests and 10 in candidate.result_digests:
            result_groups.setdefault(
                (candidate.result_digests[5], candidate.result_digests[10]), []
            ).append(candidate)
    for group in result_groups.values():
        checkpoint_hashes = {candidate.checkpoint_sha256 for candidate in group}
        if len(group) > 1 and len(checkpoint_hashes) > 1:
            names = ", ".join(sorted(candidate.name for candidate in group))
            for candidate in group:
                candidate.reject(
                    "copied_metrics",
                    f"byte-identical 5-way and 10-way per-query results appear under distinct checkpoints: {names}",
                )
    return shared


def _score(candidate: Candidate) -> None:
    values = []
    for count in (5, 10):
        chance = float(candidate.metrics[f"dev_{count}way_chance"])
        for modality in ("image", "speech"):
            value = float(candidate.metrics[f"dev_{count}way_{modality}_r1"])
            values.append((value, chance))
    above = sum(value > chance for value, chance in values)
    lifts = [max(0.0, min(1.0, (value - chance) / (1.0 - chance))) for value, chance in values]
    mean_lift = sum(lifts) / len(lifts)
    overflow = float(candidate.metrics["routing_overflow"])
    inactive = float(candidate.metrics["routing_inactive"])
    load_cv = float(candidate.metrics["routing_load_cv"])
    score = (
        10_000_000.0
        + 1_000_000.0 * above
        + 10_000.0 * mean_lift
        + 100.0 * (1.0 - min(1.0, overflow))
        + (1.0 - min(1.0, inactive))
        + 0.01 / (1.0 + load_cv)
    )
    candidate.selection_score = score
    candidate.selection_components = {
        "ppl_gate_passed": True,
        "above_chance_count": above,
        "mean_normalized_positive_lift": mean_lift,
        "routing_overflow": overflow,
        "routing_inactive": inactive,
        "routing_load_cv": load_cv,
    }


def _ranking_key(candidate: Candidate) -> Tuple[Any, ...]:
    assert candidate.selection_components is not None
    components = candidate.selection_components
    return (
        -int(components["above_chance_count"]),
        -float(components["mean_normalized_positive_lift"]),
        float(components["routing_overflow"]),
        float(components["routing_inactive"]),
        float(components["routing_load_cv"]),
        candidate.name,
    )


def _artifact_hash(candidate: Candidate, key: str) -> str:
    return str(candidate.artifacts.get(key, {}).get("sha256", ""))


def _candidate_payload(candidate: Candidate, selected: bool) -> Dict[str, Any]:
    return {
        "name": candidate.name,
        "valid": candidate.valid,
        "selected": selected,
        "rank": candidate.rank,
        "reasons": candidate.reasons,
        "run_root": str(candidate.run_root),
        "development_eval_dir": str(candidate.dev_root),
        "metrics": candidate.metrics,
        "selection_score": candidate.selection_score,
        "selection_components": candidate.selection_components,
        "validation": candidate.validation,
        "non_swept_args": candidate.non_swept_args,
        "non_swept_args_sha256": canonical_sha256(candidate.non_swept_args) if candidate.non_swept_args is not None else None,
        "dataset_provenance": candidate.dataset_provenance,
        "dataset_provenance_sha256": canonical_sha256(candidate.dataset_provenance) if candidate.dataset_provenance is not None else None,
        "development_protocol_sha256": dict(
            sorted((str(key), value) for key, value in candidate.protocol_digests.items())
        ),
        "artifacts": {key: candidate.artifacts[key] for key in sorted(candidate.artifacts)},
    }


SELECTION_POLICY = {
    "eligibility": "All integrity gates must pass, including finite non-divergent CE and text PPL <= 20.",
    "authoritative_order": [
        "maximize count of 5/10-way image/speech R@1 values strictly above exact chance",
        "maximize mean normalized positive R@1 lift above chance",
        "minimize final routing overflow",
        "minimize final inactive-expert ratio",
        "minimize final accepted-load coefficient of variation",
        "ascending candidate name tie-break",
    ],
    "display_score": "10000000 + 1000000*above_chance_count + 10000*mean_normalized_positive_lift + 100*(1-overflow) + (1-inactive) + 0.01/(1+load_cv)",
    "sealed_policy": "Reject sealed lexical/resolved paths before open; reject sealed references without following them.",
}


CSV_FIELDS = [
    "candidate",
    "valid",
    "selected",
    "rank",
    "capacity_factor",
    "aux_coef",
    "text_ppl",
    "dev_5way_image_r1",
    "dev_5way_speech_r1",
    "dev_5way_chance",
    "dev_10way_image_r1",
    "dev_10way_speech_r1",
    "dev_10way_chance",
    "above_chance_count",
    "mean_normalized_positive_lift",
    "routing_overflow",
    "routing_inactive",
    "routing_load_cv",
    "selection_score",
    "reasons",
    "run_root",
    "development_eval_dir",
    "run_manifest_sha256",
    "e3_metrics_sha256",
    "e3_train_metrics_sha256",
    "checkpoint_sha256",
    "built_in_text_metrics_sha256",
    "dev_5way_metrics_sha256",
    "dev_5way_per_query_sha256",
    "dev_10way_metrics_sha256",
    "dev_10way_per_query_sha256",
    "development_image_manifest_sha256",
    "development_speech_manifest_sha256",
    "non_swept_args_sha256",
    "dataset_provenance_sha256",
]


def _csv_rows(candidates: Sequence[Candidate], selected_name: Optional[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        components = candidate.selection_components or {}
        row = {
            "candidate": candidate.name,
            "valid": candidate.valid,
            "selected": candidate.name == selected_name,
            "rank": candidate.rank if candidate.rank is not None else "",
            "selection_score": candidate.selection_score if candidate.selection_score is not None else "",
            "above_chance_count": components.get("above_chance_count", ""),
            "mean_normalized_positive_lift": components.get("mean_normalized_positive_lift", ""),
            "reasons": "; ".join(f"{reason['code']}: {reason['message']}" for reason in candidate.reasons),
            "run_root": str(candidate.run_root),
            "development_eval_dir": str(candidate.dev_root),
            "run_manifest_sha256": _artifact_hash(candidate, "run_manifest"),
            "e3_metrics_sha256": _artifact_hash(candidate, "e3_metrics"),
            "e3_train_metrics_sha256": _artifact_hash(candidate, "e3_train_metrics"),
            "checkpoint_sha256": _artifact_hash(candidate, "checkpoint"),
            "built_in_text_metrics_sha256": _artifact_hash(candidate, "built_in_text_metrics"),
            "dev_5way_metrics_sha256": _artifact_hash(candidate, "dev_5way_metrics"),
            "dev_5way_per_query_sha256": _artifact_hash(candidate, "dev_5way_per_query"),
            "dev_10way_metrics_sha256": _artifact_hash(candidate, "dev_10way_metrics"),
            "dev_10way_per_query_sha256": _artifact_hash(candidate, "dev_10way_per_query"),
            "development_image_manifest_sha256": _artifact_hash(candidate, "development_image_manifest"),
            "development_speech_manifest_sha256": _artifact_hash(candidate, "development_speech_manifest"),
            "non_swept_args_sha256": canonical_sha256(candidate.non_swept_args) if candidate.non_swept_args is not None else "",
            "dataset_provenance_sha256": canonical_sha256(candidate.dataset_provenance) if candidate.dataset_provenance is not None else "",
        }
        for key in (
            "capacity_factor",
            "aux_coef",
            "text_ppl",
            "dev_5way_image_r1",
            "dev_5way_speech_r1",
            "dev_5way_chance",
            "dev_10way_image_r1",
            "dev_10way_speech_r1",
            "dev_10way_chance",
            "routing_overflow",
            "routing_inactive",
            "routing_load_cv",
        ):
            row[key] = candidate.metrics.get(key, "")
        rows.append(row)
    return rows


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Corrected E3 Candidate Selection",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Selected candidate: **{report.get('selected_candidate') or 'none'}**",
        "",
        "## Deterministic Policy",
        "",
        "PPL <= 20 and every integrity check are hard eligibility gates. Eligible candidates are ordered by:",
        "",
    ]
    for index, item in enumerate(SELECTION_POLICY["authoritative_order"], 1):
        lines.append(f"{index}. {item}.")
    lines.extend(
        [
            "",
            f"Display score: `{SELECTION_POLICY['display_score']}`.",
            "",
            "Sealed policy: no sealed artifact path is opened or accepted.",
            "",
            "## Candidate Metrics",
            "",
            "| candidate | valid | selected | rank | PPL | 5w image | 5w speech | 10w image | 10w speech | overflow | inactive | load CV | score |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in report["candidates"]:
        metrics = candidate["metrics"]
        fmt = lambda key: "" if metrics.get(key) is None else f"{float(metrics[key]):.9g}"
        score = "" if candidate.get("selection_score") is None else f"{float(candidate['selection_score']):.9f}"
        lines.append(
            "| {name} | {valid} | {selected} | {rank} | {ppl} | {i5} | {s5} | {i10} | {s10} | {overflow} | {inactive} | {cv} | {score} |".format(
                name=candidate["name"],
                valid="yes" if candidate["valid"] else "no",
                selected="yes" if candidate["selected"] else "no",
                rank=candidate.get("rank") or "",
                ppl=fmt("text_ppl"),
                i5=fmt("dev_5way_image_r1"),
                s5=fmt("dev_5way_speech_r1"),
                i10=fmt("dev_10way_image_r1"),
                s10=fmt("dev_10way_speech_r1"),
                overflow=fmt("routing_overflow"),
                inactive=fmt("routing_inactive"),
                cv=fmt("routing_load_cv"),
                score=score,
            )
        )
    lines.extend(["", "## Rejections", ""])
    rejected = [candidate for candidate in report["candidates"] if candidate["reasons"]]
    if not rejected:
        lines.append("No candidate was rejected.")
    else:
        for candidate in rejected:
            lines.append(f"### {candidate['name']}")
            lines.append("")
            for reason in candidate["reasons"]:
                lines.append(f"- `{reason['code']}`: {reason['message']}")
            lines.append("")
    lines.extend(["## Provenance", ""])
    for candidate in report["candidates"]:
        lines.append(f"### {candidate['name']}")
        lines.append("")
        for role, artifact in candidate["artifacts"].items():
            lines.append(
                f"- `{role}`: `{artifact['path']}`; SHA-256 `{artifact['sha256']}`; {artifact['size_bytes']} bytes."
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_outputs(report: Mapping[str, Any], output_dir: Path, candidates: Sequence[Candidate]) -> None:
    reject_sealed_path(output_dir, "output directory")
    if os.path.lexists(output_dir):
        raise SelectionError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent)))
    try:
        (temporary / "candidate_selection.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (temporary / "candidate_selection.md").write_text(_markdown(report), encoding="utf-8")
        with (temporary / "candidate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(_csv_rows(candidates, report.get("selected_candidate")))
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validated_input_paths(
    run_roots: Mapping[str, Path | str], dev_eval_dirs: Mapping[str, Path | str], output_dir: Path
) -> List[Candidate]:
    if len(run_roots) != REQUIRED_CANDIDATES or len(dev_eval_dirs) != REQUIRED_CANDIDATES:
        raise SelectionError(
            f"exactly {REQUIRED_CANDIDATES} --run-root and {REQUIRED_CANDIDATES} --dev-eval-dir mappings are required"
        )
    if set(run_roots) != set(dev_eval_dirs):
        raise SelectionError("run-root and dev-eval-dir candidate names differ")
    candidates: List[Candidate] = []
    seen_paths: set[Tuple[str, str]] = set()
    for name in sorted(run_roots):
        if not NAME_RE.fullmatch(name):
            raise SelectionError(f"invalid candidate name: {name!r}")
        run_root = Path(run_roots[name]).expanduser()
        dev_root = Path(dev_eval_dirs[name]).expanduser()
        reject_sealed_path(run_root, f"{name} run root")
        reject_sealed_path(dev_root, f"{name} development eval root")
        try:
            run_resolved = run_root.resolve(strict=True)
            dev_resolved = dev_root.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise SelectionError(f"candidate {name} has a missing input directory: {exc}") from exc
        if not run_resolved.is_dir() or not dev_resolved.is_dir():
            raise SelectionError(f"candidate {name} inputs must be directories")
        pair = (str(run_resolved), str(dev_resolved))
        if pair in seen_paths:
            raise SelectionError(f"candidate {name} reuses another candidate's run/dev roots")
        seen_paths.add(pair)
        candidates.append(Candidate(name=name, run_root=run_resolved, dev_root=dev_resolved))
    output_resolved = output_dir.resolve(strict=False)
    for candidate in candidates:
        if output_resolved == candidate.run_root or candidate.run_root in output_resolved.parents:
            raise SelectionError("output directory must not be inside a candidate run root")
        if output_resolved == candidate.dev_root or candidate.dev_root in output_resolved.parents:
            raise SelectionError("output directory must not be inside a development eval root")
    return candidates


def select_candidates(
    run_roots: Mapping[str, Path | str],
    dev_eval_dirs: Mapping[str, Path | str],
    output_dir: Path | str,
) -> Dict[str, Any]:
    """Validate, select, and write the three required output artifacts."""

    output_path = Path(output_dir).expanduser()
    reject_sealed_path(output_path, "output directory")
    if os.path.lexists(output_path):
        raise SelectionError(f"refusing to overwrite existing output directory: {output_path}")
    candidates = _validated_input_paths(run_roots, dev_eval_dirs, output_path)
    cache = ArtifactCache()
    for candidate in candidates:
        _safe_phase(candidate, _validate_manifest)
        if candidate.manifest_args is not None:
            _safe_phase(candidate, _validate_run)
        if candidate.checkpoint_sha256:
            _safe_phase(candidate, _validate_development, cache)

    shared_validation = _cross_validate(candidates)
    valid = [candidate for candidate in candidates if candidate.valid]
    for candidate in valid:
        try:
            _score(candidate)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            candidate.reject("selection_metric_missing", f"{candidate.name}: cannot compute selection score: {exc}")
    ranked = sorted((candidate for candidate in candidates if candidate.valid), key=_ranking_key)
    for index, candidate in enumerate(ranked, 1):
        candidate.rank = index
    selected = ranked[0] if ranked else None
    report: Dict[str, Any] = {
        "schema_version": 1,
        "status": "selected" if selected is not None else "no_valid_candidate",
        "selected_candidate": selected.name if selected is not None else None,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(ranked),
        "selection_policy": SELECTION_POLICY,
        "shared_validation": shared_validation,
        "candidates": [
            _candidate_payload(candidate, selected is not None and candidate.name == selected.name)
            for candidate in candidates
        ],
    }
    _write_outputs(report, output_path, candidates)
    if selected is None:
        raise NoValidCandidateError(
            f"no valid corrected E3 candidate; diagnostics written to {output_path}"
        )
    return report


def parse_named_paths(values: Sequence[str], flag: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SelectionError(f"{flag} must use NAME=PATH syntax: {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise SelectionError(f"{flag} must use non-empty NAME=PATH syntax: {value!r}")
        if name in result:
            raise SelectionError(f"duplicate {flag} candidate name: {name}")
        result[name] = Path(raw_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        "--candidate-run",
        action="append",
        default=[],
        metavar="NAME=RUN_ROOT",
        help="Corrected E3 candidate run root; repeat exactly five times.",
    )
    parser.add_argument(
        "--dev-eval-dir",
        "--dev-eval",
        action="append",
        default=[],
        metavar="NAME=DEV_EVAL_DIR",
        help="Candidate development eval root containing r5/ and r10/; repeat exactly five times.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_roots = parse_named_paths(args.run_root, "--run-root")
        dev_eval_dirs = parse_named_paths(args.dev_eval_dir, "--dev-eval-dir")
        report = select_candidates(run_roots, dev_eval_dirs, args.output_dir)
    except SelectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
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
