"""Strict, dependency-free validation for sealed protocol schema v2."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    from scripts.analyze_paired_controls import (
        AnalysisError,
        LoadedRow,
        group_aware_chance_r_at_1,
        image_group_identity,
        positive_indices_for_group,
        production_bootstrap_r_at_1_ci,
        production_retrieval_metrics,
        validate_per_query_identity_row,
    )
except ImportError:  # Direct execution from the scripts directory.
    from analyze_paired_controls import (  # type: ignore[no-redef]
        AnalysisError,
        LoadedRow,
        group_aware_chance_r_at_1,
        image_group_identity,
        positive_indices_for_group,
        production_bootstrap_r_at_1_ci,
        production_retrieval_metrics,
        validate_per_query_identity_row,
    )

try:
    from scripts.sealed_position_allocator import (
        ALLOCATOR_NAME,
        ALLOCATOR_VERSION,
        AssignmentPlanError,
        assignment_provenance,
        enforce_gold_position_assignment,
        lexical_hard_negative_indices,
        permute_candidates_for_query,
        select_local_candidate_indices,
        validate_allocator_manifest,
        validate_candidate_permutation,
        validate_executed_positions,
    )
except ImportError:  # Direct execution from the scripts directory.
    from sealed_position_allocator import (  # type: ignore[no-redef]
        ALLOCATOR_NAME,
        ALLOCATOR_VERSION,
        AssignmentPlanError,
        assignment_provenance,
        enforce_gold_position_assignment,
        lexical_hard_negative_indices,
        permute_candidates_for_query,
        select_local_candidate_indices,
        validate_allocator_manifest,
        validate_candidate_permutation,
        validate_executed_positions,
    )


SCHEMA_VERSION = 2
PROTOCOL_NAME = "sealed_evaluation_protocol"
CONTROL_RUNTIME = {
    "real": ("shared_prefix", "real"),
    "shuffled": ("shared_prefix", "shuffled"),
    "zero": ("shared_prefix", "zero"),
    "norm-matched-random": ("shared_prefix", "random"),
    "no-prefix": ("no_prefix_lm", "real"),
}
EVALUATION_RUN_FIELDS = {
    "id",
    "cell_id",
    "role",
    "negative_mode",
    "requested_candidate_count",
    "conditional_negatives",
    "conditional_candidates",
    "conditional_queries",
    "image_query_count",
    "speech_query_count",
    "image_eval_samples",
    "speech_eval_samples",
    "max_length",
    "conditional_batch_size",
    "query_offset",
    "candidate_offset",
    "tie_epsilon",
    "candidate_permutation",
    "randomize_positive_position",
    "control",
    "prefix_control",
    "eval_path",
    "candidate_seed",
    "gold_position_allocator_name",
    "gold_position_allocator_version",
    "gold_position_assignment_plans_sha256",
    "image_gold_position_assignment_id",
    "image_gold_positions_sha256",
    "speech_gold_position_assignment_id",
    "speech_gold_positions_sha256",
    "control_seed",
    "bootstrap_samples",
    "bootstrap_seed",
    "protocol_name",
    "eval_split_name",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FINAL_FEATURE_CACHE_POLICY = {
    "mode": "exclusive_write_only_recompute",
    "preexisting_root_allowed": False,
    "cache_reads_allowed": False,
    "writes": "atomic_exclusive_per_payload",
    "feature_source": "verified_frozen_media_snapshots",
}


class ProtocolV2Error(ValueError):
    pass


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolV2Error(f"{label} must be an integer >= {minimum}")
    return value


def _finite_nonnegative_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ProtocolV2Error(f"{label} must be a finite nonnegative number")
    return float(value)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_set_hash(candidate_ids: Sequence[Any]) -> str:
    payload = json.dumps(
        list(candidate_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProtocolV2Error(f"{label} must be an exact lowercase SHA256")
    return value


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ProtocolV2Error(
            f"sealed metrics contract mismatch for {label}: "
            f"expected={expected!r} observed={actual!r}"
        )


def per_query_jsonl_content(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize rows with the evaluator's exact JSONL publication semantics."""

    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )


def per_query_jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(per_query_jsonl_content(rows).encode("utf-8")).hexdigest()


def frozen_checkpoint_artifact(protocol: Mapping[str, Any]) -> Dict[str, str]:
    checkpoint = protocol.get("checkpoint")
    artifact = checkpoint.get("artifact") if isinstance(checkpoint, Mapping) else None
    if not isinstance(artifact, Mapping):
        raise ProtocolV2Error("protocol.checkpoint.artifact is missing")
    raw_path = artifact.get("path")
    if (
        artifact.get("type") != "file"
        or not isinstance(raw_path, str)
        or not raw_path
        or not Path(raw_path).is_absolute()
    ):
        raise ProtocolV2Error(
            "protocol.checkpoint.artifact must identify one absolute file path"
        )
    return {
        "path": str(Path(raw_path).resolve(strict=False)),
        "sha256": _sha256(
            artifact.get("sha256"), "protocol.checkpoint.artifact.sha256"
        ),
    }


def _frozen_image_manifest_rows(
    protocol: Mapping[str, Any], expected_count: int
) -> List[Dict[str, Any]]:
    inputs = protocol.get("inputs")
    image_input = inputs.get("image_test") if isinstance(inputs, Mapping) else None
    if not isinstance(image_input, Mapping):
        raise ProtocolV2Error("protocol.inputs.image_test is missing")
    if set(image_input) != {"path", "type", "sha256", "bytes"}:
        raise ProtocolV2Error(
            "protocol.inputs.image_test has an incomplete or extra key set"
        )
    raw_path = image_input.get("path")
    if (
        image_input.get("type") != "file"
        or not isinstance(raw_path, str)
        or not raw_path
        or not Path(raw_path).is_absolute()
    ):
        raise ProtocolV2Error(
            "protocol.inputs.image_test must identify one absolute file path"
        )
    expected_sha256 = _sha256(
        image_input.get("sha256"), "protocol.inputs.image_test.sha256"
    )
    expected_bytes = _integer(
        image_input.get("bytes"), "protocol.inputs.image_test.bytes"
    )
    path = Path(raw_path)
    try:
        resolved = path.resolve(strict=True)
        payload = resolved.read_bytes()
    except OSError as exc:
        raise ProtocolV2Error(f"cannot read frozen image manifest: {exc}") from exc
    if str(resolved) != raw_path or path.is_symlink():
        raise ProtocolV2Error("protocol.inputs.image_test path identity drifted")
    _equal(len(payload), expected_bytes, "frozen image manifest bytes")
    _equal(
        hashlib.sha256(payload).hexdigest(),
        expected_sha256,
        "frozen image manifest SHA256",
    )
    try:
        rows = [
            json.loads(line)
            for line in payload.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolV2Error("frozen image manifest is not valid JSONL") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ProtocolV2Error("frozen image manifest rows must be objects")
    if len(rows) < expected_count:
        raise ProtocolV2Error(
            "frozen image manifest has fewer rows than image_eval_samples"
        )
    return rows


def _frozen_speech_manifest_rows(
    protocol: Mapping[str, Any], expected_count: int
) -> List[Dict[str, Any]]:
    inputs = protocol.get("inputs")
    speech_input = inputs.get("speech_test") if isinstance(inputs, Mapping) else None
    if not isinstance(speech_input, Mapping):
        raise ProtocolV2Error("protocol.inputs.speech_test is missing")
    if set(speech_input) != {"path", "type", "sha256", "bytes"}:
        raise ProtocolV2Error(
            "protocol.inputs.speech_test has an incomplete or extra key set"
        )
    raw_path = speech_input.get("path")
    if (
        speech_input.get("type") != "file"
        or not isinstance(raw_path, str)
        or not raw_path
        or not Path(raw_path).is_absolute()
    ):
        raise ProtocolV2Error(
            "protocol.inputs.speech_test must identify one absolute file path"
        )
    expected_sha256 = _sha256(
        speech_input.get("sha256"), "protocol.inputs.speech_test.sha256"
    )
    expected_bytes = _integer(
        speech_input.get("bytes"), "protocol.inputs.speech_test.bytes"
    )
    path = Path(raw_path)
    try:
        resolved = path.resolve(strict=True)
        payload = resolved.read_bytes()
    except OSError as exc:
        raise ProtocolV2Error(f"cannot read frozen speech manifest: {exc}") from exc
    if str(resolved) != raw_path or path.is_symlink():
        raise ProtocolV2Error("protocol.inputs.speech_test path identity drifted")
    _equal(len(payload), expected_bytes, "frozen speech manifest bytes")
    _equal(
        hashlib.sha256(payload).hexdigest(),
        expected_sha256,
        "frozen speech manifest SHA256",
    )
    try:
        rows = [
            json.loads(line)
            for line in payload.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolV2Error("frozen speech manifest is not valid JSONL") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ProtocolV2Error("frozen speech manifest rows must be objects")
    if len(rows) < expected_count:
        raise ProtocolV2Error(
            "frozen speech manifest has fewer rows than speech_eval_samples"
        )
    return rows


def _manifest_row_index(
    value: Any, label: str, size: int, manifest_label: str = "image"
) -> int:
    index = _integer(value, label)
    if index >= size:
        raise ProtocolV2Error(f"{label} is outside frozen {manifest_label} manifest")
    return index


def _frozen_query_identity(
    row: Mapping[str, Any], modality: str, index: int
) -> str:
    for key in ("uid", "source_uid", "image_uid", "utterance_id", "source_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{modality}:{value}"
    stable_fields = {
        key: str(row[key])
        for key in (
            "source",
            "image_path",
            "audio_path",
            "caption",
            "transcript",
            "speaker_id",
        )
        if row.get(key) not in (None, "")
    }
    if stable_fields:
        payload = json.dumps(
            stable_fields,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"{modality}:sha256:{hashlib.sha256(payload).hexdigest()}"
    return f"{modality}:index:{index}"


def _allocator_contract(
    protocol: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    *,
    candidate_seed: int,
    query_counts: Mapping[str, int],
) -> Tuple[Mapping[str, Any], Dict[Tuple[str, str], Dict[str, Any]]]:
    query_offsets = {
        _integer(
            run.get("query_offset"),
            f"evaluation_runs[{index}].query_offset",
        )
        for index, run in enumerate(runs)
        if isinstance(run, Mapping)
    }
    if len(query_offsets) != 1:
        raise ProtocolV2Error("evaluation_runs must share one query_offset")
    query_offset = _integer(
        next(iter(query_offsets)), "evaluation_runs query_offset"
    )
    manifest = protocol.get("gold_position_allocator")
    if not isinstance(manifest, Mapping):
        raise ProtocolV2Error("protocol.gold_position_allocator is missing")
    try:
        plans = validate_allocator_manifest(
            manifest,
            cells,
            candidate_seed=candidate_seed,
            query_counts=query_counts,
            query_offset=query_offset,
        )
    except (AssignmentPlanError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolV2Error(f"invalid gold-position allocator: {exc}") from exc
    if len(plans) != 2 * len(cells):
        raise ProtocolV2Error(
            "gold-position allocator does not cover every cell/modality"
        )
    return manifest, plans


def _validate_row_assignment_and_replay(
    row: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    modality: str,
    row_index: int,
    run: Mapping[str, Any],
    plan: Mapping[str, Any],
    allocator_manifest: Mapping[str, Any],
) -> int:
    query_index = _manifest_row_index(
        row.get("query_index"),
        f"per-query row {row_index} {modality} query_index",
        len(manifest_rows),
        modality,
    )
    assignment_index = query_index - int(plan["query_offset"])
    if not 0 <= assignment_index < int(plan["query_count"]):
        raise ProtocolV2Error(
            f"per-query row {row_index} query_index is outside the "
            "gold-position assignment plan"
        )
    assigned_position = int(plan["positions"][assignment_index])
    gold_position = _integer(
        row.get("gold_position"),
        f"per-query row {row_index} {modality} gold_position",
    )
    _equal(
        gold_position,
        assigned_position,
        f"per-query row {row_index} frozen assigned gold position",
    )
    expected_assignment = {
        **assignment_provenance(plan, allocator_manifest),
        "assignment_index": assignment_index,
        "assigned_position": assigned_position,
    }
    _equal(
        row.get("gold_position_assignment"),
        expected_assignment,
        f"per-query row {row_index} gold_position_assignment",
    )

    raw_candidate_indices = row.get("candidate_indices")
    raw_permutation = row.get("candidate_permutation")
    if not isinstance(raw_candidate_indices, list) or not isinstance(
        raw_permutation, list
    ):
        raise ProtocolV2Error(
            f"per-query row {row_index} candidate permutation provenance is missing"
        )
    try:
        validate_candidate_permutation(
            raw_candidate_indices, raw_permutation
        )
    except AssignmentPlanError as exc:
        raise ProtocolV2Error(
            f"per-query row {row_index} invalid candidate permutation: {exc}"
        ) from exc

    eval_samples = int(run[f"{modality}_eval_samples"])
    if eval_samples > len(manifest_rows):
        raise ProtocolV2Error(
            f"frozen {modality} manifest is smaller than evaluation_run"
        )
    candidate_bank = list(manifest_rows[:eval_samples])
    local_mode = int(run["conditional_negatives"]) >= 0
    if local_mode:
        local_query_index = query_index
        if not 0 <= local_query_index < len(candidate_bank):
            raise ProtocolV2Error(
                f"per-query row {row_index} query is outside local candidate bank"
            )
        text_key = "caption" if modality == "image" else "transcript"
        try:
            texts = [str(candidate[text_key]) for candidate in candidate_bank]
        except KeyError as exc:
            raise ProtocolV2Error(
                f"frozen {modality} candidate is missing {text_key}"
            ) from exc
        group_ids = None
        if modality == "image":
            try:
                group_ids = [
                    image_group_identity(candidate)
                    for candidate in candidate_bank
                ]
            except ValueError as exc:
                raise ProtocolV2Error(
                    f"frozen image group identity is invalid: {exc}"
                ) from exc
        hard_indices: Sequence[int] = ()
        if run["negative_mode"] == "hard_text":
            try:
                hard_indices = lexical_hard_negative_indices(
                    texts,
                    int(run["conditional_negatives"]),
                    group_ids,
                )[local_query_index]
            except AssignmentPlanError as exc:
                raise ProtocolV2Error(
                    f"cannot replay hard-text candidates: {exc}"
                ) from exc
        query_seed = (
            int(run["candidate_seed"])
            + (1000003 if modality == "speech" else 0)
            + 1009 * query_index
        )
        try:
            base_candidate_indices = select_local_candidate_indices(
                len(candidate_bank),
                local_query_index,
                int(run["conditional_negatives"]),
                str(run["negative_mode"]),
                candidate_seed=query_seed,
                hard_indices=hard_indices,
                group_ids=group_ids,
            )
        except AssignmentPlanError as exc:
            raise ProtocolV2Error(
                f"cannot replay per-query row {row_index} candidates: {exc}"
            ) from exc
    else:
        candidate_start = int(run["candidate_offset"])
        if candidate_start < 0:
            candidate_start = int(run["query_offset"])
        candidate_end = candidate_start + int(run["conditional_candidates"])
        if (
            candidate_start < 0
            or candidate_end > len(candidate_bank)
            or not candidate_start <= query_index < candidate_end
        ):
            raise ProtocolV2Error(
                f"per-query row {row_index} full candidate window is invalid"
            )
        base_candidate_indices = list(range(candidate_start, candidate_end))

    query_identity = _frozen_query_identity(
        manifest_rows[query_index], modality, query_index
    )
    try:
        (
            expected_indices,
            expected_permutation,
            _,
            expected_permutation_seed,
        ) = permute_candidates_for_query(
            base_candidate_indices,
            query_index,
            int(run["control_seed"]),
            query_identity,
        )
        (
            expected_indices,
            expected_permutation,
            expected_gold_position,
        ) = enforce_gold_position_assignment(
            expected_indices,
            expected_permutation,
            query_index,
            assigned_position,
        )
    except AssignmentPlanError as exc:
        raise ProtocolV2Error(
            f"cannot replay per-query row {row_index} permutation: {exc}"
        ) from exc
    _equal(
        list(raw_candidate_indices),
        expected_indices,
        f"per-query row {row_index} deterministic candidate indices",
    )
    _equal(
        list(raw_permutation),
        expected_permutation,
        f"per-query row {row_index} deterministic candidate permutation",
    )
    _equal(
        _integer(
            row.get("candidate_permutation_seed"),
            f"per-query row {row_index} candidate_permutation_seed",
        ),
        expected_permutation_seed,
        f"per-query row {row_index} candidate_permutation_seed",
    )
    _equal(
        gold_position,
        expected_gold_position,
        f"per-query row {row_index} replayed gold position",
    )
    return gold_position


def _immutable_image_row_contract(
    row: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    row_index: int,
) -> Dict[str, Any]:
    candidate_ids = row.get("candidate_ids")
    candidate_indices = row.get("candidate_indices")
    if (
        not isinstance(candidate_ids, list)
        or not isinstance(candidate_indices, list)
        or len(candidate_indices) != len(candidate_ids)
    ):
        raise ProtocolV2Error(
            f"per-query row {row_index} candidate_indices must match candidate_ids"
        )
    query_index = _manifest_row_index(
        row.get("query_index"),
        f"per-query row {row_index} query_index",
        len(manifest_rows),
    )
    frozen_candidate_indices = [
        _manifest_row_index(
            value,
            f"per-query row {row_index} candidate_indices[{position}]",
            len(manifest_rows),
        )
        for position, value in enumerate(candidate_indices)
    ]
    gold_position = _integer(
        row.get("gold_position"), f"per-query row {row_index} gold_position"
    )
    if gold_position >= len(frozen_candidate_indices):
        raise ProtocolV2Error(
            f"per-query row {row_index} gold_position is outside candidates"
        )
    _equal(
        frozen_candidate_indices[gold_position],
        query_index,
        f"per-query row {row_index} frozen query/gold manifest index",
    )
    try:
        query_group_id = image_group_identity(manifest_rows[query_index])
        candidate_group_ids = [
            image_group_identity(manifest_rows[index])
            for index in frozen_candidate_indices
        ]
        positive_indices = positive_indices_for_group(
            candidate_group_ids, query_group_id
        )
        group_chance = group_aware_chance_r_at_1(candidate_group_ids)
    except ValueError as exc:
        raise ProtocolV2Error(
            f"per-query row {row_index} frozen image group identity is invalid: {exc}"
        ) from exc
    positive_candidate_indices = [
        frozen_candidate_indices[position] for position in positive_indices
    ]
    positive_candidate_ids = [
        candidate_ids[position] for position in positive_indices
    ]
    unique_group_count = len(set(candidate_group_ids))
    caption_chance = float(len(positive_indices) / len(candidate_group_ids))
    expected_query_uid = _frozen_query_identity(
        manifest_rows[query_index], "image", query_index
    )
    expected_candidate_ids = [
        _frozen_query_identity(manifest_rows[index], "image", index)
        for index in frozen_candidate_indices
    ]
    try:
        expected_candidate_texts = [
            str(manifest_rows[index]["caption"])
            for index in frozen_candidate_indices
        ]
    except KeyError as exc:
        raise ProtocolV2Error(
            f"per-query row {row_index} frozen image candidate has no caption"
        ) from exc
    exact_fields = {
        "query_uid": expected_query_uid,
        "query_index": query_index,
        "candidate_indices": frozen_candidate_indices,
        "candidate_ids": expected_candidate_ids,
        "candidate_texts": expected_candidate_texts,
        "candidate_count": len(expected_candidate_ids),
        "candidate_index": gold_position,
        "gold_index": gold_position,
        "gold_candidate_index": query_index,
        "gold_candidate_id": expected_candidate_ids[gold_position],
        "query_source": str(manifest_rows[query_index].get("source", "")),
        "query_image_group_id": query_group_id,
        "candidate_group_ids": candidate_group_ids,
        "positive_indices": positive_indices,
        "positive_candidate_indices": positive_candidate_indices,
        "positive_candidate_ids": positive_candidate_ids,
        "positive_count": len(positive_indices),
        "unique_candidate_group_count": unique_group_count,
        "group_aware_chance_r_at_1": group_chance,
        "caption_row_chance_r_at_1": caption_chance,
    }
    for field, expected in exact_fields.items():
        _equal(
            row.get(field),
            expected,
            f"per-query row {row_index} immutable {field}",
        )
    source_provenance = row.get("source_provenance")
    if not isinstance(source_provenance, Mapping):
        raise ProtocolV2Error(
            f"per-query row {row_index} source_provenance is missing"
        )
    for field in (
        "query_uid",
        "query_index",
        "query_source",
        "query_image_group_id",
    ):
        _equal(
            source_provenance.get(field),
            exact_fields[field],
            f"per-query row {row_index} source_provenance.{field}",
        )
    return {
        "positive_indices": positive_indices,
        "unique_group_count": unique_group_count,
        "group_chance": group_chance,
        "caption_chance": caption_chance,
    }


def _immutable_speech_row_contract(
    row: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    row_index: int,
    expected_query_index: int,
) -> None:
    candidate_ids = row.get("candidate_ids")
    candidate_indices = row.get("candidate_indices")
    candidate_texts = row.get("candidate_texts")
    if (
        not isinstance(candidate_ids, list)
        or not isinstance(candidate_indices, list)
        or not isinstance(candidate_texts, list)
        or len(candidate_indices) != len(candidate_ids)
        or len(candidate_texts) != len(candidate_ids)
    ):
        raise ProtocolV2Error(
            f"per-query row {row_index} speech candidate identities and texts must align"
        )
    query_index = _manifest_row_index(
        row.get("query_index"),
        f"per-query row {row_index} speech query_index",
        len(manifest_rows),
        "speech",
    )
    _equal(
        query_index,
        expected_query_index,
        f"per-query row {row_index} immutable speech query_index",
    )
    frozen_candidate_indices = [
        _manifest_row_index(
            value,
            f"per-query row {row_index} speech candidate_indices[{position}]",
            len(manifest_rows),
            "speech",
        )
        for position, value in enumerate(candidate_indices)
    ]
    if len(set(frozen_candidate_indices)) != len(frozen_candidate_indices):
        raise ProtocolV2Error(
            f"per-query row {row_index} speech candidate_indices must be unique"
        )
    gold_position = _integer(
        row.get("gold_position"),
        f"per-query row {row_index} speech gold_position",
    )
    if gold_position >= len(frozen_candidate_indices):
        raise ProtocolV2Error(
            f"per-query row {row_index} speech gold_position is outside candidates"
        )
    _equal(
        frozen_candidate_indices[gold_position],
        query_index,
        f"per-query row {row_index} frozen speech query/gold manifest index",
    )
    expected_query_uid = _frozen_query_identity(
        manifest_rows[query_index], "speech", query_index
    )
    expected_candidate_ids = [
        _frozen_query_identity(manifest_rows[index], "speech", index)
        for index in frozen_candidate_indices
    ]
    try:
        expected_candidate_texts = [
            str(manifest_rows[index]["transcript"])
            for index in frozen_candidate_indices
        ]
    except KeyError as exc:
        raise ProtocolV2Error(
            f"per-query row {row_index} frozen speech candidate has no transcript"
        ) from exc
    exact_fields = {
        "query_uid": expected_query_uid,
        "query_index": query_index,
        "candidate_indices": frozen_candidate_indices,
        "candidate_ids": expected_candidate_ids,
        "candidate_texts": expected_candidate_texts,
        "candidate_count": len(expected_candidate_ids),
        "candidate_index": gold_position,
        "gold_index": gold_position,
        "gold_candidate_index": query_index,
        "gold_candidate_id": expected_candidate_ids[gold_position],
        "candidate_set_hash": _candidate_set_hash(expected_candidate_ids),
        "query_source": str(manifest_rows[query_index].get("source", "")),
        "speaker_id": manifest_rows[query_index].get("speaker_id"),
    }
    for field, expected in exact_fields.items():
        _equal(
            row.get(field),
            expected,
            f"per-query row {row_index} immutable speech {field}",
        )
    source_provenance = row.get("source_provenance")
    if not isinstance(source_provenance, Mapping):
        raise ProtocolV2Error(
            f"per-query row {row_index} speech source_provenance is missing"
        )
    for field in ("query_uid", "query_index", "query_source", "speaker_id"):
        _equal(
            source_provenance.get(field),
            exact_fields[field],
            f"per-query row {row_index} immutable speech source_provenance.{field}",
        )

def validate_protocol_v2(protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("protocol") != PROTOCOL_NAME
    ):
        raise ProtocolV2Error("unsupported frozen protocol schema")
    raw_cells = protocol.get("evaluation_matrix")
    controls = protocol.get("controls")
    runs = protocol.get("evaluation_runs")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ProtocolV2Error("evaluation_matrix must be a non-empty list")
    if controls != list(CONTROL_RUNTIME):
        raise ProtocolV2Error("controls must use the exact schema-v2 order")
    if not isinstance(runs, list):
        raise ProtocolV2Error("evaluation_runs must be a list")

    cells: Dict[str, Dict[str, Any]] = {}
    for raw in raw_cells:
        if not isinstance(raw, Mapping):
            raise ProtocolV2Error("evaluation_matrix rows must be objects")
        cell = dict(raw)
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in cells:
            raise ProtocolV2Error("evaluation_matrix IDs must be unique strings")
        _integer(cell.get("candidate_count"), f"{cell_id}.candidate_count", 2)
        if cell.get("negative_mode") not in {"random", "hard_text", "full_matrix"}:
            raise ProtocolV2Error(f"{cell_id}.negative_mode is invalid")
        if cell.get("role") not in {"primary", "secondary"}:
            raise ProtocolV2Error(f"{cell_id}.role is invalid")
        cells[cell_id] = cell

    query_counts = protocol.get("query_counts")
    seeds = protocol.get("seeds")
    if not isinstance(query_counts, Mapping) or not isinstance(seeds, Mapping):
        raise ProtocolV2Error("query_counts and seeds must be objects")
    image_queries = _integer(query_counts.get("image"), "query_counts.image", 1)
    speech_queries = _integer(query_counts.get("speech"), "query_counts.speech", 1)
    candidate_seed = _integer(seeds.get("candidate_seed"), "seeds.candidate_seed")
    control_seed = _integer(seeds.get("control_seed"), "seeds.control_seed")
    allocator_manifest, allocator_plans = _allocator_contract(
        protocol,
        list(cells.values()),
        runs,
        candidate_seed=candidate_seed,
        query_counts={"image": image_queries, "speech": speech_queries},
    )

    expected_pairs = {
        (cell_id, control) for cell_id in cells for control in CONTROL_RUNTIME
    }
    observed_pairs = set()
    normalized: List[Dict[str, Any]] = []
    for index, raw in enumerate(runs):
        if not isinstance(raw, Mapping) or set(raw) != EVALUATION_RUN_FIELDS:
            raise ProtocolV2Error(
                f"evaluation_runs[{index}] has an incomplete or extra key set"
            )
        run = dict(raw)
        cell_id = run["cell_id"]
        control = run["control"]
        pair = (cell_id, control)
        if pair not in expected_pairs or pair in observed_pairs:
            raise ProtocolV2Error("evaluation_runs contain an invalid or duplicate pair")
        observed_pairs.add(pair)
        cell = cells[cell_id]
        if run["id"] != f"{cell_id}:{control}" or run["role"] != cell["role"]:
            raise ProtocolV2Error(f"evaluation run identity drift for {cell_id}/{control}")
        eval_path, prefix_control = CONTROL_RUNTIME[control]
        if run["eval_path"] != eval_path or run["prefix_control"] != prefix_control:
            raise ProtocolV2Error(f"evaluation control mapping drift for {cell_id}/{control}")
        candidate_count = int(cell["candidate_count"])
        full_matrix = (
            cell["negative_mode"] == "full_matrix"
            or candidate_count == image_queries
        )
        expected_negative_mode = (
            "random" if cell["negative_mode"] == "full_matrix" else cell["negative_mode"]
        )
        expected_values = {
            "negative_mode": expected_negative_mode,
            "requested_candidate_count": candidate_count,
            "conditional_negatives": -1 if full_matrix else candidate_count - 1,
            "conditional_candidates": candidate_count,
            "conditional_queries": image_queries,
            "image_query_count": image_queries,
            "speech_query_count": speech_queries,
            "image_eval_samples": image_queries,
            "speech_eval_samples": speech_queries,
            "candidate_seed": candidate_seed,
            "control_seed": control_seed,
            "randomize_positive_position": True,
            "candidate_permutation": "query_identity_seeded",
        }
        for modality in ("image", "speech"):
            plan = allocator_plans[(cell_id, modality)]
            expected_values.update({
                "gold_position_allocator_name": ALLOCATOR_NAME,
                "gold_position_allocator_version": ALLOCATOR_VERSION,
                "gold_position_assignment_plans_sha256": allocator_manifest[
                    "plans_sha256"
                ],
                f"{modality}_gold_position_assignment_id": plan[
                    "assignment_id"
                ],
                f"{modality}_gold_positions_sha256": plan[
                    "positions_sha256"
                ],
            })
        for field, expected in expected_values.items():
            if run[field] != expected:
                raise ProtocolV2Error(
                    f"evaluation run {cell_id}/{control} drifted field {field}"
                )
        _integer(run["max_length"], f"{cell_id}/{control}.max_length", 1)
        _integer(
            run["conditional_batch_size"],
            f"{cell_id}/{control}.conditional_batch_size",
            1,
        )
        _finite_nonnegative_number(
            run["tie_epsilon"], f"{cell_id}/{control}.tie_epsilon"
        )
        for field in (
            "query_offset",
            "candidate_offset",
            "bootstrap_samples",
            "bootstrap_seed",
        ):
            _integer(run[field], f"{cell_id}/{control}.{field}")
        if not isinstance(run["protocol_name"], str) or not run["protocol_name"]:
            raise ProtocolV2Error("evaluation run protocol_name is missing")
        if not isinstance(run["eval_split_name"], str) or not run["eval_split_name"]:
            raise ProtocolV2Error("evaluation run eval_split_name is missing")
        normalized.append(run)
    if observed_pairs != expected_pairs:
        raise ProtocolV2Error("evaluation_runs do not cover the exact matrix")
    return normalized


def validate_metrics_against_protocol_v2(
    protocol: Mapping[str, Any],
    metrics: Mapping[str, Any],
    per_query_rows: Sequence[Mapping[str, Any]],
    *,
    cell_id: str,
    control: str,
    protocol_file_sha256: str,
    per_query_file_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
) -> Dict[str, Any]:
    """Validate one sealed metrics/per-query pair against its exact frozen run."""

    runs = validate_protocol_v2(protocol)
    if control not in CONTROL_RUNTIME:
        raise ProtocolV2Error(f"unknown sealed control {control!r}")
    run_id = f"{cell_id}:{control}"
    matches = [
        run for run in runs
        if run["id"] == run_id
        and run["cell_id"] == cell_id
        and run["control"] == control
    ]
    if len(matches) != 1:
        raise ProtocolV2Error(
            f"sealed metrics do not select exactly one evaluation_run: {run_id}"
        )
    run = matches[0]
    allocator_manifest = protocol.get("gold_position_allocator")
    if not isinstance(allocator_manifest, Mapping):
        raise ProtocolV2Error("protocol.gold_position_allocator is missing")
    allocator_plans = {
        (str(plan["cell_id"]), str(plan["modality"])): plan
        for plan in allocator_manifest["plans"]
        if isinstance(plan, Mapping)
    }
    selected_plans = {
        modality: allocator_plans[(cell_id, modality)]
        for modality in ("image", "speech")
    }
    assignment_bindings = {
        modality: assignment_provenance(
            selected_plans[modality], allocator_manifest
        )
        for modality in ("image", "speech")
    }
    checkpoint_artifact = frozen_checkpoint_artifact(protocol)
    selected_checkpoint_path = str(Path(checkpoint_path).resolve(strict=False))
    _equal(
        selected_checkpoint_path,
        checkpoint_artifact["path"],
        "caller selected checkpoint path",
    )
    checkpoint_sha256 = _sha256(checkpoint_sha256, "checkpoint SHA256")
    _equal(
        checkpoint_sha256,
        checkpoint_artifact["sha256"],
        "caller selected checkpoint SHA256",
    )
    identity = metrics.get("evaluation_provenance")
    metric_args = (
        identity.get("metric_affecting_args")
        if isinstance(identity, Mapping)
        else None
    )
    provenance = metrics.get("provenance")
    if not isinstance(identity, Mapping) or not isinstance(metric_args, Mapping):
        raise ProtocolV2Error(
            "sealed metrics are missing evaluation_provenance.metric_affecting_args"
        )
    if not isinstance(provenance, Mapping):
        raise ProtocolV2Error("sealed metrics are missing provenance")
    _equal(metrics.get("sealed_protocol"), True, "sealed_protocol")
    _equal(metrics.get("evaluation_scope"), "final", "evaluation_scope")
    _equal(identity.get("evaluation_scope"), "final", "evaluation_provenance.evaluation_scope")
    _equal(metric_args.get("evaluation_scope"), "final", "metric_affecting_args.evaluation_scope")
    _equal(metrics.get("strict_control"), True, "strict_control")
    _equal(identity.get("strict_control"), True, "evaluation_provenance.strict_control")

    _equal(identity.get("frozen_evaluation_run_id"), run["id"], "frozen_evaluation_run_id")
    _equal(
        identity.get("frozen_evaluation_cell_id"),
        cell_id,
        "frozen_evaluation_cell_id",
    )
    _equal(
        identity.get("frozen_evaluation_control"),
        control,
        "frozen_evaluation_control",
    )
    _equal(
        identity.get("e3_checkpoint_path"),
        checkpoint_artifact["path"],
        "evaluation_provenance.e3_checkpoint_path",
    )
    _equal(
        metrics.get("e3_checkpoint_path"),
        checkpoint_artifact["path"],
        "metrics.e3_checkpoint_path",
    )
    metric_arg_fields = (
        "negative_mode", "conditional_negatives", "conditional_candidates",
        "conditional_queries", "image_eval_samples", "speech_eval_samples",
        "max_length", "conditional_batch_size", "query_offset",
        "candidate_offset", "tie_epsilon", "candidate_permutation",
        "randomize_positive_position", "prefix_control", "eval_path",
        "candidate_seed", "control_seed", "bootstrap_samples",
        "bootstrap_seed", "protocol_name", "eval_split_name",
    )
    for field in metric_arg_fields:
        _equal(metric_args.get(field), run[field], f"metric_affecting_args.{field}")

    expected_condition = "no_prefix" if control == "no-prefix" else run["prefix_control"]
    identity_values = {
        "negative_mode": run["negative_mode"],
        "requested_query_count": run["conditional_queries"],
        "image_query_count": run["image_query_count"],
        "speech_query_count": run["speech_query_count"],
        "requested_candidate_count": run["requested_candidate_count"],
        "image_candidate_count": run["requested_candidate_count"],
        "speech_candidate_count": run["requested_candidate_count"],
        "conditional_negatives": run["conditional_negatives"],
        "conditional_candidates": run["conditional_candidates"],
        "image_eval_samples": run["image_eval_samples"],
        "speech_eval_samples": run["speech_eval_samples"],
        "query_offset": run["query_offset"],
        "candidate_offset": run["candidate_offset"],
        "tie_epsilon": run["tie_epsilon"],
        "bootstrap_samples": run["bootstrap_samples"],
        "bootstrap_seed": run["bootstrap_seed"],
        "candidate_seed": run["candidate_seed"],
        "control_seed": run["control_seed"],
        "gold_position_allocator_name": ALLOCATOR_NAME,
        "gold_position_allocator_version": ALLOCATOR_VERSION,
        "gold_position_assignment_plans_sha256": allocator_manifest[
            "plans_sha256"
        ],
        "eval_path": run["eval_path"],
        "condition": expected_condition,
        "prefix_control": expected_condition,
        "protocol_name": run["protocol_name"],
        "eval_split_name": run["eval_split_name"],
    }
    for field, expected in identity_values.items():
        _equal(identity.get(field), expected, f"evaluation_provenance.{field}")
    for modality, expected in assignment_bindings.items():
        _equal(
            identity.get(f"{modality}_gold_position_assignment"),
            expected,
            f"evaluation_provenance.{modality}_gold_position_assignment",
        )

    top_level_values = {
        "negative_mode": run["negative_mode"],
        "candidate_count": run["requested_candidate_count"],
        "speech_candidate_count": run["requested_candidate_count"],
        "conditional_candidates_per_query": run["requested_candidate_count"],
        "conditional_speech_candidates_per_query": run["requested_candidate_count"],
        "image_eval_count": run["image_query_count"],
        "speech_eval_count": run["speech_query_count"],
        "conditional_image_eval_count": run["image_query_count"],
        "conditional_speech_eval_count": run["speech_query_count"],
        "query_offset": run["query_offset"],
        "candidate_offset": run["candidate_offset"],
        "tie_epsilon": run["tie_epsilon"],
        "candidate_seed": run["candidate_seed"],
        "control_seed": run["control_seed"],
        "gold_position_allocator_name": ALLOCATOR_NAME,
        "gold_position_allocator_version": ALLOCATOR_VERSION,
        "gold_position_assignment_plans_sha256": allocator_manifest[
            "plans_sha256"
        ],
        "candidate_permutation_policy": run["candidate_permutation"],
        "randomized_positive_position": run["randomize_positive_position"],
        "eval_path": run["eval_path"],
        "condition": expected_condition,
        "prefix_control": expected_condition,
        "protocol_name": run["protocol_name"],
        "eval_split_name": run["eval_split_name"],
        "conditional_uses_multimodal_prefix": control != "no-prefix",
    }
    for field, expected in top_level_values.items():
        _equal(metrics.get(field), expected, field)
    for modality, expected in assignment_bindings.items():
        _equal(
            metrics.get(f"{modality}_gold_position_assignment"),
            expected,
            f"{modality}_gold_position_assignment",
        )

    protocol_file_sha256 = _sha256(
        protocol_file_sha256, "protocol file SHA256"
    )
    per_query_file_sha256 = _sha256(
        per_query_file_sha256, "per-query file SHA256"
    )
    expected_counts = Counter({
        "image": run["image_query_count"],
        "speech": run["speech_query_count"],
    })
    image_manifest_rows = _frozen_image_manifest_rows(
        protocol, int(run["image_eval_samples"])
    )
    speech_manifest_rows = _frozen_speech_manifest_rows(
        protocol, int(run["speech_eval_samples"])
    )
    image_row_contracts: List[Dict[str, Any]] = []
    observed_counts: Counter[str] = Counter()
    validated_by_modality: Dict[str, List[LoadedRow]] = {
        "image": [],
        "speech": [],
    }
    executed_positions: Dict[str, List[int]] = {
        "image": [],
        "speech": [],
    }
    query_uids: set[str] = set()
    for index, row in enumerate(per_query_rows):
        if not isinstance(row, Mapping):
            raise ProtocolV2Error(f"per-query row {index} must be an object")
        candidate_ids = row.get("candidate_ids")
        if (
            not isinstance(candidate_ids, list)
            or len(candidate_ids) != run["requested_candidate_count"]
        ):
            raise ProtocolV2Error(
                f"per-query row {index} candidate cardinality disagrees with evaluation_run"
            )
        if "condition" in row:
            _equal(
                row.get("condition"),
                expected_condition,
                f"per-query row {index} condition",
            )
        try:
            validate_per_query_identity_row(
                expected_condition,
                "sealed per-query",
                index + 1,
                row,
                require_production_fields=True,
            )
        except AnalysisError as exc:
            raise ProtocolV2Error(str(exc)) from exc
        query_uid = str(row["query_uid"])
        if query_uid in query_uids:
            raise ProtocolV2Error(
                f"per-query row {index} duplicates query_uid {query_uid!r}"
            )
        query_uids.add(query_uid)
        expected_positive_indices = None
        row_modality = row.get("modality")
        if row_modality in ("image", "speech"):
            executed_positions[str(row_modality)].append(
                _validate_row_assignment_and_replay(
                    row,
                    (
                        image_manifest_rows
                        if row_modality == "image"
                        else speech_manifest_rows
                    ),
                    str(row_modality),
                    index,
                    run,
                    selected_plans[str(row_modality)],
                    allocator_manifest,
                )
            )
        if row_modality == "image":
            image_contract = _immutable_image_row_contract(
                row, image_manifest_rows, index
            )
            image_row_contracts.append(image_contract)
            expected_positive_indices = image_contract["positive_indices"]
        elif row_modality == "speech":
            _immutable_speech_row_contract(
                row,
                speech_manifest_rows,
                index,
                int(run["query_offset"]) + observed_counts["speech"],
            )
            expected_positive_indices = [
                _integer(
                    row.get("gold_position"),
                    f"per-query row {index} speech gold_position",
                )
            ]
        try:
            validated_row = validate_per_query_identity_row(
                expected_condition,
                "sealed per-query",
                index + 1,
                row,
                require_production_fields=True,
                expected_positive_indices=expected_positive_indices,
            )
        except AnalysisError as exc:
            raise ProtocolV2Error(str(exc)) from exc
        if (
            validated_row.row["modality"] == "image"
            and "positive_indices" not in row
        ):
            raise ProtocolV2Error(
                f"per-query row {index} is missing image positive_indices"
            )
        modality = str(validated_row.row["modality"])
        observed_counts[modality] += 1
        if modality not in validated_by_modality:
            raise ProtocolV2Error(
                f"per-query row {index} has unsupported modality {modality!r}"
            )
        validated_by_modality[modality].append(validated_row)
        candidate_ids = validated_row.row["candidate_ids"]
        if len(candidate_ids) != run["requested_candidate_count"]:
            raise ProtocolV2Error(
                f"per-query row {index} candidate cardinality disagrees with evaluation_run"
            )
        if row.get("candidate_set_hash") != _candidate_set_hash(candidate_ids):
            raise ProtocolV2Error(
                f"per-query row {index} candidate_set_hash disagrees with candidate_ids"
            )
        row_values = {
            "condition": expected_condition,
            "prefix_control": expected_condition,
            "eval_split_name": run["eval_split_name"],
            "negative_mode": run["negative_mode"],
            "eval_path": run["eval_path"],
        }
        for field, expected in row_values.items():
            _equal(
                row.get(field), expected,
                f"per-query row {index} {field}",
            )
        row_protocol = row.get("protocol")
        if not isinstance(row_protocol, Mapping):
            raise ProtocolV2Error(
                f"per-query row {index} protocol metadata is missing"
            )
        protocol_values = {
            "name": run["protocol_name"],
            "manifest_sha256": protocol_file_sha256,
            "eval_split_name": run["eval_split_name"],
            "negative_mode": run["negative_mode"],
            "candidate_count": run["requested_candidate_count"],
            "candidate_seed": run["candidate_seed"],
            "candidate_permutation_policy": run["candidate_permutation"],
            "randomized_positive_position": run["randomize_positive_position"],
            "gold_position_allocator_name": ALLOCATOR_NAME,
            "gold_position_allocator_version": ALLOCATOR_VERSION,
            "gold_position_assignment_plans_sha256": allocator_manifest[
                "plans_sha256"
            ],
            "tie_epsilon": run["tie_epsilon"],
        }
        for field, expected in protocol_values.items():
            _equal(
                row_protocol.get(field), expected,
                f"per-query row {index} protocol.{field}",
            )
        _equal(
            row.get("evaluation_provenance"),
            identity,
            f"per-query row {index} evaluation_provenance",
        )
    if observed_counts != expected_counts:
        raise ProtocolV2Error(
            "per-query modality cardinality disagrees with evaluation_run: "
            f"expected={dict(expected_counts)} observed={dict(observed_counts)}"
        )
    for modality, positions in executed_positions.items():
        plan = selected_plans[modality]
        try:
            validate_executed_positions(positions, plan)
        except AssignmentPlanError as exc:
            raise ProtocolV2Error(
                f"{modality} executed gold positions are invalid: {exc}"
            ) from exc
        counts = [
            positions.count(position)
            for position in range(int(plan["candidate_count"]))
        ]
        _equal(
            metrics.get(f"{modality}_gold_position_counts"),
            counts,
            f"{modality}_gold_position_counts",
        )

    aggregate_aliases = (
        "r_at_1",
        "r_at_5",
        "r_at_10",
        "strict_r_at_1",
        "mrr",
        "mean_gold_nll_margin",
        "tie_count",
        "tie_rate",
        "r_at_1_bootstrap_ci_low",
        "r_at_1_bootstrap_ci_high",
    )
    for offset, (modality, prefix) in enumerate((
        ("image", "image_to_text"),
        ("speech", "speech_to_text"),
    )):
        rows = validated_by_modality[modality]
        expected_aggregates = production_retrieval_metrics(rows, prefix)
        try:
            bootstrap = production_bootstrap_r_at_1_ci(
                rows,
                int(run["bootstrap_samples"]),
                int(run["bootstrap_seed"]) + offset,
            )
        except AnalysisError as exc:
            raise ProtocolV2Error(str(exc)) from exc
        expected_aggregates.update({
            f"{prefix}_{field}": value
            for field, value in bootstrap.items()
        })
        for field, expected in expected_aggregates.items():
            _equal(metrics.get(field), expected, field)
        for suffix in aggregate_aliases:
            field = f"{prefix}_{suffix}"
            _equal(
                metrics.get(f"conditional_{field}"),
                expected_aggregates.get(field),
                f"conditional_{field}",
            )

    image_positive_counts = [
        int(row.production_fields["positive_count"])
        for row in validated_by_modality["image"]
    ]
    _equal(
        metrics.get("image_positive_counts"),
        image_positive_counts,
        "image_positive_counts",
    )
    _equal(
        metrics.get("conditional_image_positive_counts"),
        image_positive_counts,
        "conditional_image_positive_counts",
    )
    image_unique_group_counts = [
        int(contract["unique_group_count"]) for contract in image_row_contracts
    ]
    image_group_chance = float(
        sum(float(contract["group_chance"]) for contract in image_row_contracts)
        / len(image_row_contracts)
    )
    image_caption_chance = float(
        sum(float(contract["caption_chance"]) for contract in image_row_contracts)
        / len(image_row_contracts)
    )
    chance_fields = {
        "image_unique_candidate_group_counts": image_unique_group_counts,
        "conditional_image_unique_candidate_group_counts": image_unique_group_counts,
        "image_group_aware_chance_r_at_1": image_group_chance,
        "conditional_image_group_aware_chance_r_at_1": image_group_chance,
        "image_caption_row_chance_r_at_1": image_caption_chance,
        "conditional_image_caption_row_chance_r_at_1": image_caption_chance,
        "image_chance_r_at_1": image_caption_chance,
        "conditional_image_chance_r_at_1": image_caption_chance,
        "image_legacy_gold_caption_position_chance_r_at_1": (
            1.0 / int(run["requested_candidate_count"])
        ),
        "conditional_image_legacy_gold_caption_position_chance_r_at_1": (
            1.0 / int(run["requested_candidate_count"])
        ),
        "speech_chance_r_at_1": 1.0 / int(run["requested_candidate_count"]),
        "conditional_speech_chance_r_at_1": (
            1.0 / int(run["requested_candidate_count"])
        ),
    }
    for field, expected in chance_fields.items():
        _equal(metrics.get(field), expected, field)

    _equal(metrics.get("per_query_rows"), len(per_query_rows), "per_query_rows")
    _equal(
        metrics.get("per_query_sha256"),
        per_query_file_sha256,
        "per_query exact file SHA256",
    )
    _equal(
        metrics.get("per_query_sha256"),
        per_query_jsonl_sha256(per_query_rows),
        "per_query canonical publication SHA256",
    )

    protocol_content_sha256 = _sha256(
        protocol.get("protocol_content_sha256"), "protocol content SHA256"
    )
    for container, label in (
        (metrics, "metrics"),
        (identity, "evaluation_provenance"),
        (provenance, "provenance"),
    ):
        _equal(
            container.get("protocol_manifest_sha256"),
            protocol_file_sha256,
            f"{label}.protocol_manifest_sha256",
        )
        _equal(
            container.get("protocol_content_sha256"),
            protocol_content_sha256,
            f"{label}.protocol_content_sha256",
        )
    for container, label, field in (
        (metrics, "metrics", "e3_checkpoint_sha256"),
        (identity, "evaluation_provenance", "e3_checkpoint_sha256"),
        (provenance, "provenance", "checkpoint_sha256"),
    ):
        _equal(container.get(field), checkpoint_sha256, f"{label}.{field}")
    _equal(
        provenance.get("checkpoint_path"),
        checkpoint_artifact["path"],
        "provenance.checkpoint_path",
    )
    stage_b_sha = _sha256(
        metrics.get("stage_b_checkpoint_sha256"),
        "metrics.stage_b_checkpoint_sha256",
    )
    _equal(
        identity.get("stage_b_checkpoint_sha256"),
        stage_b_sha,
        "evaluation_provenance.stage_b_checkpoint_sha256",
    )
    _equal(
        provenance.get("stage_b_checkpoint_sha256"),
        stage_b_sha,
        "provenance.stage_b_checkpoint_sha256",
    )

    evaluator_hashes = {
        (str(record.get("path", "")), str(record.get("sha256", "")))
        for record in protocol.get("inputs", {}).get("evaluator_scripts", [])
        if isinstance(record, Mapping)
    }
    evaluator_pair = (
        str(identity.get("evaluator_path", "")),
        str(identity.get("evaluator_sha256", "")),
    )
    if evaluator_pair not in evaluator_hashes:
        raise ProtocolV2Error(
            "sealed metrics evaluator hash is not frozen in the protocol"
        )
    _equal(
        (
            str(provenance.get("evaluator_path", "")),
            str(provenance.get("evaluator_sha256", "")),
        ),
        evaluator_pair,
        "provenance evaluator fingerprint",
    )

    _equal(
        identity.get("feature_cache_policy"),
        FINAL_FEATURE_CACHE_POLICY,
        "evaluation_provenance.feature_cache_policy",
    )
    for kind in ("image", "audio"):
        cache = provenance.get(f"{kind}_feature_cache")
        if not isinstance(cache, Mapping):
            raise ProtocolV2Error(f"provenance.{kind}_feature_cache is missing")
        _equal(
            cache.get("policy"),
            FINAL_FEATURE_CACHE_POLICY,
            f"provenance.{kind}_feature_cache.policy",
        )
        for identity_suffix, cache_field in (
            ("cache_identity_sha256", "base_identity_sha256"),
            ("cache_payload_set_sha256", "payload_set_sha256"),
            ("produced_features_sha256", "produced_features_sha256"),
        ):
            field = f"{kind}_{identity_suffix}"
            digest = _sha256(
                identity.get(field), f"evaluation_provenance.{field}"
            )
            _equal(
                cache.get(cache_field),
                digest,
                f"provenance.{kind}_feature_cache.{cache_field}",
            )

    identity_digest = _sha256(
        identity.get("evaluation_identity_sha256"),
        "evaluation_provenance.evaluation_identity_sha256",
    )
    identity_without_digest = dict(identity)
    identity_without_digest.pop("evaluation_identity_sha256", None)
    _equal(
        identity_digest,
        _canonical_sha256(identity_without_digest),
        "evaluation identity digest",
    )
    _equal(
        metrics.get("evaluation_identity_sha256"),
        identity_digest,
        "metrics.evaluation_identity_sha256",
    )
    _equal(
        provenance.get("evaluation_identity_sha256"),
        identity_digest,
        "provenance.evaluation_identity_sha256",
    )
    return run
