#!/usr/bin/env python3
"""Paired analysis for conditional-matching per-query JSONL outputs.

Inputs are joined on modality, query UID, candidate-set hash, and the gold
candidate identity.  The implementation intentionally uses only the Python
standard library so it can run in the evaluation image without extra packages.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


Key = Tuple[str, str, str, str]
REQUIRED_FIELDS = (
    "modality",
    "query_uid",
    "candidate_set_hash",
    "rank",
    "scores",
    "gold_position",
    "gold_candidate_id",
    "predicted_position",
    "candidate_ids",
)
CONTROL_ALIASES = {
    "no_prefix": {"no_prefix", "no_prefix_lm"},
    "no_prefix_lm": {"no_prefix", "no_prefix_lm"},
}
PROTOCOL_CONTROL_KEYS = {
    "condition",
    "control",
    "control_seed",
    "eval_path",
    "prefix_control",
    "seed",
}


class AnalysisError(ValueError):
    """Raised when inputs cannot support a valid paired analysis."""


@dataclass(frozen=True)
class LoadedRow:
    condition: str
    source: str
    line_number: int
    row: Dict[str, Any]
    key: Key
    candidate_keys: Tuple[str, ...]
    gold_key: str
    rank_one_based: int
    score_direction: str
    production_fields: Dict[str, Any]


def _identity(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"candidate identity is not JSON-serializable: {value!r}") from exc


def _control_value(row: Mapping[str, Any]) -> str:
    for field in ("control", "condition", "prefix_control"):
        value = row.get(field)
        if value is not None and str(value):
            if field == "prefix_control" and str(row.get("eval_path", "")).startswith("no_prefix"):
                return "no_prefix"
            return str(value)
    if str(row.get("eval_path", "")).startswith("no_prefix"):
        return "no_prefix"
    raise AnalysisError("row is missing control metadata (control, condition, or prefix_control)")


def _protocol_value(row: Mapping[str, Any]) -> Any:
    if "protocol" in row:
        return row["protocol"]
    fields = ("eval_split_name", "negative_mode", "eval_path", "query_offset", "candidate_offset")
    protocol = {field: row[field] for field in fields if field in row}
    if protocol:
        return protocol
    raise AnalysisError("row is missing protocol metadata")


def _common_protocol(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    return {
        key: _common_protocol(item)
        for key, item in value.items()
        if str(key) not in PROTOCOL_CONTROL_KEYS
    }


def _rank_base(row: Mapping[str, Any]) -> int:
    value = row.get("rank_base")
    protocol = row.get("protocol")
    if value is None and isinstance(protocol, dict):
        value = protocol.get("rank_base")
    base = 0 if value is None else value
    if isinstance(base, bool) or not isinstance(base, int) or base not in (0, 1):
        raise AnalysisError(f"rank_base must be 0 or 1, got {base}")
    return base


def _score_direction(row: Mapping[str, Any]) -> str:
    value = row.get("score_direction")
    protocol = row.get("protocol")
    if value is None and isinstance(protocol, dict):
        value = protocol.get("score_direction")
    direction = str(value or "higher_is_better")
    if direction not in {"higher_is_better", "lower_is_better"}:
        raise AnalysisError(f"unsupported score_direction: {direction!r}")
    return direction


def _condition_matches(expected: str, actual: str) -> bool:
    allowed = CONTROL_ALIASES.get(expected, {expected})
    return actual in allowed


def _score_order(scores: Sequence[float], direction: str) -> List[int]:
    reverse = direction == "higher_is_better"
    return sorted(range(len(scores)), key=lambda index: scores[index], reverse=reverse)


def _exact_int(value: Any, label: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisError(f"{where}: {label} must be an integer")
    return value


def _tie_epsilon(row: Mapping[str, Any], where: str) -> float:
    value = row.get("tie_epsilon")
    protocol = row.get("protocol")
    if value is None and isinstance(protocol, dict):
        value = protocol.get("tie_epsilon", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{where}: tie_epsilon must be finite and non-negative")
    epsilon = float(value)
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise AnalysisError(f"{where}: tie_epsilon must be finite and non-negative")
    return epsilon


def _candidate_set_hash(candidate_ids: Sequence[Any]) -> str:
    payload = json.dumps(
        list(candidate_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_image_group_id(kind: str, value: Any) -> str:
    payload = json.dumps(
        {"kind": str(kind), "value": value},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"image_group:{digest}"


def image_group_identity(row: Mapping[str, Any]) -> str:
    """Return the evaluator deterministic identity for one source image."""

    content_id_fields = (
        "content_sha256",
        "resized_content_sha256",
        "media_sha256",
    )
    source_id_fields = (
        "source_image_id",
        "image_id",
        "coco_image_id",
        "cocoid",
        "imgid",
        "image_uid",
        "image_key",
        "filename",
        "file_name",
        "filepath",
    )
    source_id_prefixes = (
        "coco_image:",
        "image:",
        "source_image:",
        "filename:",
        "filepath:",
    )
    for container_name, container in (("row", row), ("source", row.get("source"))):
        if not isinstance(container, Mapping):
            continue
        for field in content_id_fields:
            value = container.get(field)
            if value not in (None, ""):
                return _stable_image_group_id(
                    f"{container_name}.{field}", str(value)
                )

    source = row.get("source")
    namespace = ""
    if isinstance(source, Mapping):
        for key in ("dataset", "dataset_name", "repository", "name", "label"):
            value = source.get(key)
            if value not in (None, ""):
                namespace = str(value)
                break
    elif source not in (None, ""):
        namespace = str(source)
    if not namespace:
        for key in ("source_dataset", "dataset_name", "dataset"):
            value = row.get(key)
            if value not in (None, ""):
                namespace = str(value)
                break

    source_ids: List[str] = []
    containers = [row]
    for key in ("source_image", "source"):
        value = row.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
        elif key == "source_image" and value not in (None, ""):
            source_ids.append(f"source_image:{value}")
    for container in containers:
        for key in source_id_fields:
            value = container.get(key)
            if value not in (None, ""):
                source_ids.append(f"{key}:{value}")
        values = container.get("source_ids")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            source_ids.extend(
                str(value)
                for value in values
                if str(value).startswith(source_id_prefixes)
            )
    if source_ids:
        return _stable_image_group_id(
            "source_image_fields",
            {"namespace": namespace, "ids": sorted(set(source_ids))},
        )

    image_path = row.get("image_path")
    if image_path in (None, ""):
        raise ValueError("image row needs source-image identity fields or image_path")
    resolved_path = Path(str(image_path)).expanduser().resolve()
    return _stable_image_group_id("resolved_image_path", str(resolved_path))


def positive_indices_for_group(
    candidate_group_ids: Sequence[str], query_group_id: str
) -> List[int]:
    positives = [
        index
        for index, candidate_group_id in enumerate(candidate_group_ids)
        if str(candidate_group_id) == str(query_group_id)
    ]
    if not positives:
        raise ValueError("candidate set contains no caption from the query image group")
    return positives


def group_aware_chance_r_at_1(candidate_group_ids: Sequence[str]) -> float:
    unique_group_count = len(set(str(value) for value in candidate_group_ids))
    if unique_group_count <= 0:
        raise ValueError("candidate_group_ids must not be empty")
    return float(1.0 / unique_group_count)


def _same_derived_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_derived_value(left, right)
            for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _check_production_fields(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    required: Sequence[str],
    where: str,
) -> None:
    missing = [field for field in required if field not in row]
    if missing:
        raise AnalysisError(
            f"{where}: missing production-derived fields: {', '.join(missing)}"
        )
    for field, expected_value in expected.items():
        if (
            field in row
            and not _same_derived_value(row[field], expected_value)
        ):
            raise AnalysisError(
                f"{where}: {field} disagrees with canonical score reconstruction"
            )


def reconstruct_and_validate_per_query_row(
    condition: str,
    source: str,
    line_number: int,
    row: Any,
    *,
    require_production_fields: bool = False,
    validate_production_fields: bool = True,
    expected_positive_indices: Sequence[int] | None = None,
) -> LoadedRow:
    """Reconstruct and validate all score-derived evidence for one query."""

    where = f"{source}:{line_number}"
    if not isinstance(row, dict):
        raise AnalysisError(f"{where}: each JSONL record must be an object")
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise AnalysisError(f"{where}: missing required fields: {', '.join(missing)}")

    actual_control = _control_value(row)
    if not _condition_matches(condition, actual_control):
        raise AnalysisError(
            f"{where}: input named {condition!r} contains control metadata {actual_control!r}"
        )
    _protocol_value(row)

    modality = row["modality"]
    query_uid = row["query_uid"]
    candidate_set_hash = row["candidate_set_hash"]
    if not isinstance(modality, str) or not modality:
        raise AnalysisError(f"{where}: modality must be a non-empty string")
    if not isinstance(query_uid, str) or not query_uid:
        raise AnalysisError(f"{where}: query_uid must be a non-empty string")
    if (
        not isinstance(candidate_set_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", candidate_set_hash) is None
    ):
        raise AnalysisError(f"{where}: candidate_set_hash must be an exact SHA256")

    candidate_ids = row["candidate_ids"]
    if not isinstance(candidate_ids, list) or not candidate_ids:
        raise AnalysisError(f"{where}: candidate_ids must be a non-empty list")
    candidate_keys = tuple(_identity(value) for value in candidate_ids)
    if len(set(candidate_keys)) != len(candidate_keys):
        raise AnalysisError(f"{where}: candidate_ids contains duplicate identities")
    if candidate_set_hash != _candidate_set_hash(candidate_ids):
        raise AnalysisError(
            f"{where}: candidate_set_hash disagrees with candidate_ids"
        )

    gold_position = _exact_int(row["gold_position"], "gold_position", where)
    predicted_position = _exact_int(
        row["predicted_position"], "predicted_position", where
    )
    rank = _exact_int(row["rank"], "rank", where)
    size = len(candidate_ids)
    if not 0 <= gold_position < size:
        raise AnalysisError(
            f"{where}: gold_position {gold_position} is outside {size} candidates"
        )
    if not 0 <= predicted_position < size:
        raise AnalysisError(
            f"{where}: predicted_position {predicted_position} is outside {size} candidates"
        )
    rank_base = _rank_base(row)
    rank_one_based = rank + (1 if rank_base == 0 else 0)
    if not 1 <= rank_one_based <= size:
        raise AnalysisError(
            f"{where}: rank {rank} with rank_base={rank_base} is outside {size} candidates"
        )

    gold_key = candidate_keys[gold_position]
    if _identity(row["gold_candidate_id"]) != gold_key:
        raise AnalysisError(
            f"{where}: gold_candidate_id disagrees with candidate_ids[gold_position]"
        )

    raw_positive_positions = (
        list(expected_positive_indices)
        if expected_positive_indices is not None
        else row.get("positive_indices", [gold_position])
    )
    if expected_positive_indices is not None and row.get(
        "positive_indices", raw_positive_positions
    ) != raw_positive_positions:
        raise AnalysisError(
            f"{where}: positive_indices disagree with immutable positive identities"
        )
    if not isinstance(raw_positive_positions, list) or not raw_positive_positions:
        raise AnalysisError(f"{where}: positive_indices must be a non-empty list")
    positive_positions = [
        _exact_int(value, "positive_indices entry", where)
        for value in raw_positive_positions
    ]
    if (
        len(set(positive_positions)) != len(positive_positions)
        or min(positive_positions) < 0
        or max(positive_positions) >= size
        or gold_position not in positive_positions
    ):
        raise AnalysisError(
            f"{where}: positive_indices must be unique valid positions containing gold_position"
        )
    canonical_positive_positions = sorted(positive_positions)
    if positive_positions != canonical_positive_positions:
        raise AnalysisError(f"{where}: positive_indices must be in canonical order")
    positive_positions = canonical_positive_positions
    positive_set = set(positive_positions)
    if "positive_candidate_ids" in row:
        expected_positive_keys = [
            candidate_keys[index] for index in positive_positions
        ]
        actual_positive_ids = row["positive_candidate_ids"]
        if (
            not isinstance(actual_positive_ids, list)
            or [_identity(value) for value in actual_positive_ids]
            != expected_positive_keys
        ):
            raise AnalysisError(
                f"{where}: positive_candidate_ids disagree with positive_indices"
            )

    direction = _score_direction(row)
    scores = row["scores"]
    if not isinstance(scores, list) or len(scores) != size:
        raise AnalysisError(
            f"{where}: scores must be a finite list matching candidate_ids"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in scores
    ):
        raise AnalysisError(f"{where}: scores must contain only JSON numbers")
    numeric_scores = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in numeric_scores):
        raise AnalysisError(f"{where}: scores must be finite")
    epsilon = _tie_epsilon(row, where)

    if direction == "higher_is_better":
        best_score = max(numeric_scores)
        predicted_positions = [
            index
            for index, score in enumerate(numeric_scores)
            if score >= best_score - epsilon
        ]
        best_positive_score = max(
            numeric_scores[index] for index in positive_positions
        )
        strict_rank = sum(
            1
            for index, score in enumerate(numeric_scores)
            if index not in positive_set
            and score >= best_positive_score - epsilon
        )
    else:
        best_score = min(numeric_scores)
        predicted_positions = [
            index
            for index, score in enumerate(numeric_scores)
            if score <= best_score + epsilon
        ]
        best_positive_score = min(
            numeric_scores[index] for index in positive_positions
        )
        strict_rank = sum(
            1
            for index, score in enumerate(numeric_scores)
            if index not in positive_set
            and score <= best_positive_score + epsilon
        )
    best_positive_positions = [
        index
        for index in positive_positions
        if (
            numeric_scores[index] >= best_positive_score - epsilon
            if direction == "higher_is_better"
            else numeric_scores[index] <= best_positive_score + epsilon
        )
    ]
    negative_scores = [
        score
        for index, score in enumerate(numeric_scores)
        if index not in positive_set
    ]
    gold_tie_count = sum(
        1
        for score in negative_scores
        if abs(score - best_positive_score) <= epsilon
    )
    if negative_scores:
        best_negative_score = (
            max(negative_scores)
            if direction == "higher_is_better"
            else min(negative_scores)
        )
        margin = (
            best_positive_score - best_negative_score
            if direction == "higher_is_better"
            else best_negative_score - best_positive_score
        )
    else:
        margin = 0.0

    expected_predicted_position = predicted_positions[0]
    expected_rank = strict_rank + (0 if rank_base == 0 else 1)
    production_fields: Dict[str, Any] = {
        "candidate_index": gold_position,
        "gold_index": gold_position,
        "predicted_position": expected_predicted_position,
        "predicted_candidate_id": candidate_ids[expected_predicted_position],
        "best_candidate_indices": predicted_positions,
        "rank": expected_rank,
        "strict_rank": strict_rank,
        "strict_r_at_1": float(strict_rank == 0),
        "reciprocal_rank": float(1.0 / (strict_rank + 1)),
        "candidate_count": size,
    }
    production_required = list(production_fields)
    if direction == "higher_is_better":
        nll_scores = [-score for score in numeric_scores]
        best_positive_nll = -best_positive_score
        production_fields.update({
            "raw_nll_scores": nll_scores,
            "gold_nll": best_positive_nll,
            "gold_nll_margin": margin,
            "gold_margin": margin,
            "tie_count": gold_tie_count,
            "gold_tie_count": gold_tie_count,
            "best_tie_count": max(0, len(predicted_positions) - 1),
        })
        production_required.extend((
            "raw_nll_scores",
            "gold_nll",
            "gold_nll_margin",
            "gold_margin",
            "tie_count",
            "gold_tie_count",
            "best_tie_count",
        ))
        if "positive_indices" in row:
            production_fields.update({
                "positive_indices": positive_positions,
                "positive_candidate_ids": [
                    candidate_ids[index] for index in positive_positions
                ],
                "positive_count": len(positive_positions),
                "best_positive_nll": best_positive_nll,
                "best_positive_nll_margin": margin,
                "best_positive_indices": best_positive_positions,
            })
            production_required.extend((
                "positive_indices",
                "positive_candidate_ids",
                "positive_count",
                "best_positive_nll",
                "best_positive_nll_margin",
                "best_positive_indices",
            ))
    elif require_production_fields:
        raise AnalysisError(
            f"{where}: production-derived evidence requires higher_is_better scores"
        )

    candidate_indices = row.get("candidate_indices")
    if isinstance(candidate_indices, list) and len(candidate_indices) == size:
        production_fields["gold_candidate_index"] = candidate_indices[gold_position]
        production_required.append("gold_candidate_index")
        if "positive_indices" in row:
            production_fields["positive_candidate_indices"] = [
                candidate_indices[index] for index in positive_positions
            ]
            production_required.append("positive_candidate_indices")
    if validate_production_fields:
        _check_production_fields(
            row,
            production_fields,
            production_required if require_production_fields else (),
            where,
        )

    key: Key = (modality, query_uid, candidate_set_hash, gold_key)
    return LoadedRow(
        condition=condition,
        source=source,
        line_number=line_number,
        row=dict(row),
        key=key,
        candidate_keys=candidate_keys,
        gold_key=gold_key,
        rank_one_based=strict_rank + 1,
        score_direction=direction,
        production_fields=production_fields,
    )


def _validate_row(condition: str, source: str, line_number: int, row: Any) -> LoadedRow:
    return reconstruct_and_validate_per_query_row(
        condition, source, line_number, row
    )


def validate_per_query_identity_row(
    condition: str,
    source: str,
    line_number: int,
    row: Any,
    *,
    require_production_fields: bool = False,
    expected_positive_indices: Sequence[int] | None = None,
) -> LoadedRow:
    """Validate one row with the canonical reconstruction schema."""

    return reconstruct_and_validate_per_query_row(
        condition,
        source,
        line_number,
        row,
        require_production_fields=require_production_fields,
        expected_positive_indices=expected_positive_indices,
    )


def load_condition(condition: str, path: Path) -> Tuple[Dict[Key, LoadedRow], Dict[str, Any]]:
    records: Dict[Key, LoadedRow] = {}
    digest = hashlib.sha256()
    byte_count = 0
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise AnalysisError(f"cannot open {condition} input {path}: {exc}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AnalysisError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            loaded = _validate_row(condition, str(path), line_number, row)
            if loaded.key in records:
                first = records[loaded.key]
                raise AnalysisError(
                    f"{path}:{line_number}: duplicate paired key; first seen on line {first.line_number}"
                )
            records[loaded.key] = loaded
    if not records:
        raise AnalysisError(f"{path}: input has no records")
    return records, {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
        "rows": len(records),
    }


def align_conditions(
    conditions: Mapping[str, Mapping[Key, LoadedRow]],
) -> List[Dict[str, LoadedRow]]:
    if "real" not in conditions:
        raise AnalysisError("a real=PATH input is required")
    real_keys = set(conditions["real"])
    for condition, records in conditions.items():
        keys = set(records)
        if keys != real_keys:
            missing = sorted(real_keys - keys)[:3]
            extra = sorted(keys - real_keys)[:3]
            raise AnalysisError(
                f"{condition}: paired key mismatch: missing={len(real_keys - keys)} {missing!r}, "
                f"extra={len(keys - real_keys)} {extra!r}"
            )

    aligned: List[Dict[str, LoadedRow]] = []
    for key in sorted(real_keys):
        pair = {condition: records[key] for condition, records in conditions.items()}
        real = pair["real"]
        real_ids = set(real.candidate_keys)
        real_protocol = _common_protocol(_protocol_value(real.row))
        real_rank_base = _rank_base(real.row)
        for condition, loaded in pair.items():
            if set(loaded.candidate_keys) != real_ids:
                raise AnalysisError(
                    f"{condition} {loaded.source}:{loaded.line_number}: candidate identities differ "
                    f"for paired key {key!r}"
                )
            if loaded.score_direction != real.score_direction:
                raise AnalysisError(f"{condition}: score_direction differs for paired key {key!r}")
            if _rank_base(loaded.row) != real_rank_base:
                raise AnalysisError(f"{condition}: rank_base differs for paired key {key!r}")
            protocol = _common_protocol(_protocol_value(loaded.row))
            if protocol != real_protocol:
                raise AnalysisError(f"{condition}: protocol metadata differs for paired key {key!r}")
        aligned.append(pair)
    return aligned


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Dict[str, float]:
    if total <= 0:
        raise AnalysisError("Wilson interval requires total > 0")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return {"low": max(0.0, center - half), "high": min(1.0, center + half)}


def retrieval_metrics(ranks_one_based: Sequence[int]) -> Dict[str, Any]:
    if not ranks_one_based:
        raise AnalysisError("retrieval metrics require at least one rank")
    total = len(ranks_one_based)
    recalls: Dict[str, Any] = {}
    for cutoff in (1, 5, 10):
        count = sum(rank <= cutoff for rank in ranks_one_based)
        recalls[f"r_at_{cutoff}"] = {
            "count": count,
            "total": total,
            "rate": count / total,
            "wilson_95": wilson_interval(count, total),
        }
    return {
        "n": total,
        **recalls,
        "mrr": sum(1.0 / rank for rank in ranks_one_based) / total,
        "median_rank": float(statistics.median(ranks_one_based)),
    }


def production_retrieval_metrics(
    rows: Sequence[LoadedRow], prefix: str
) -> Dict[str, Any]:
    """Recompute the evaluator's tie-aware aggregate metrics from rows."""

    if not rows:
        raise AnalysisError("production retrieval metrics require at least one row")
    ranks = [int(row.production_fields["strict_rank"]) for row in rows]
    total = len(ranks)
    tie_count = sum(
        int(row.production_fields["gold_tie_count"]) for row in rows
    )
    tie_queries = sum(
        int(row.production_fields["gold_tie_count"]) > 0 for row in rows
    )
    output: Dict[str, Any] = {}
    for cutoff in (1, 5, 10):
        output[f"{prefix}_r_at_{cutoff}"] = float(
            sum(rank < cutoff for rank in ranks) / total
        )
    output.update({
        f"{prefix}_mean_rank": float(sum(rank + 1 for rank in ranks) / total),
        f"{prefix}_strict_r_at_1": float(
            sum(rank == 0 for rank in ranks) / total
        ),
        f"{prefix}_mrr": float(
            sum(
                float(row.production_fields["reciprocal_rank"])
                for row in rows
            )
            / total
        ),
        f"{prefix}_mean_gold_nll_margin": float(
            sum(
                float(row.production_fields["gold_nll_margin"])
                for row in rows
            )
            / total
        ),
        f"{prefix}_tie_count": int(tie_count),
        f"{prefix}_tie_query_count": int(tie_queries),
        f"{prefix}_tie_rate": float(tie_queries / total),
    })
    return output


def production_bootstrap_r_at_1_ci(
    rows: Sequence[LoadedRow], samples: int, seed: int
) -> Dict[str, float]:
    """Call the evaluator's exact query-bootstrap helper when resampling matters."""

    ranks = [int(row.production_fields["strict_rank"]) for row in rows]
    if not ranks or samples <= 0:
        return {}
    hits = [rank == 0 for rank in ranks]
    if all(hits) or not any(hits):
        value = float(all(hits))
        return {
            "r_at_1_bootstrap_ci_low": value,
            "r_at_1_bootstrap_ci_high": value,
            "r_at_1_bootstrap_samples": int(samples),
        }
    try:
        from scripts.eval_conditional_retrieval import bootstrap_r_at_1_ci
    except ModuleNotFoundError as exc:
        raise AnalysisError(
            "production bootstrap reconstruction requires evaluator dependencies"
        ) from exc
    return bootstrap_r_at_1_ci(ranks, samples, seed)


def _metrics_for_rows(rows: Sequence[LoadedRow]) -> Dict[str, Any]:
    return retrieval_metrics([row.rank_one_based for row in rows])


def chance_summary(rows: Sequence[LoadedRow]) -> Dict[str, float]:
    sizes = [len(row.candidate_keys) for row in rows]
    return {
        "r_at_1": sum(1.0 / size for size in sizes) / len(sizes),
        "r_at_5": sum(min(5, size) / size for size in sizes) / len(sizes),
        "r_at_10": sum(min(10, size) / size for size in sizes) / len(sizes),
        "mrr": sum(sum(1.0 / rank for rank in range(1, size + 1)) / size for size in sizes) / len(sizes),
    }


def positive_position_balance(rows: Sequence[LoadedRow]) -> Dict[str, Any]:
    grouped: Dict[int, List[int]] = {}
    for loaded in rows:
        grouped.setdefault(len(loaded.candidate_keys), []).append(int(loaded.row["gold_position"]))
    output: Dict[str, Any] = {"n": len(rows), "by_candidate_count": {}}
    all_balanced = True
    for size, positions in sorted(grouped.items()):
        counts = Counter(positions)
        full_counts = [counts.get(index, 0) for index in range(size)]
        difference = max(full_counts) - min(full_counts)
        balanced = difference <= 1
        all_balanced = all_balanced and balanced
        output["by_candidate_count"][str(size)] = {
            "n": len(positions),
            "expected_per_position": len(positions) / size,
            "counts": {str(index): full_counts[index] for index in range(size)},
            "max_min_count_difference": difference,
            "balanced_within_one": balanced,
        }
    output["balanced_within_one"] = all_balanced
    return output


def mcnemar_exact(real_hits: Sequence[bool], control_hits: Sequence[bool]) -> Dict[str, Any]:
    real_only = sum(real and not control for real, control in zip(real_hits, control_hits))
    control_only = sum(control and not real for real, control in zip(real_hits, control_hits))
    discordant = real_only + control_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, value) for value in range(min(real_only, control_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "real_only": real_only,
        "control_only": control_only,
        "discordant": discordant,
        "p_value_two_sided": p_value,
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = probability * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    weight = index - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def paired_bootstrap_mean_ci(
    differences: Sequence[float], samples: int, seed: int
) -> Dict[str, Any]:
    if not differences:
        raise AnalysisError("paired bootstrap requires at least one difference")
    if samples <= 0:
        raise AnalysisError("bootstrap samples must be positive")
    generator = random.Random(seed)
    count = len(differences)
    estimates = sorted(
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return {
        "estimate": sum(differences) / count,
        "ci_95": {"low": _percentile(estimates, 0.025), "high": _percentile(estimates, 0.975)},
        "samples": samples,
        "seed": seed,
    }


def _gold_margin(loaded: LoadedRow) -> float | None:
    scores = loaded.row["scores"]
    if scores is None:
        return None
    numeric = [float(value) for value in scores]
    gold_position = int(loaded.row["gold_position"])
    gold = numeric[gold_position]
    negatives = [value for index, value in enumerate(numeric) if index != gold_position]
    if not negatives:
        return 0.0
    if loaded.score_direction == "higher_is_better":
        return gold - max(negatives)
    return min(negatives) - gold


def paired_permutation_mean(differences: Sequence[float], samples: int, seed: int) -> Dict[str, Any]:
    if not differences:
        raise AnalysisError("paired permutation requires at least one difference")
    if samples <= 0:
        raise AnalysisError("permutation samples must be positive")
    observed = abs(sum(differences) / len(differences))
    exact_count = 2 ** len(differences)
    tolerance = 1e-15
    if exact_count <= samples:
        estimates = (
            abs(sum(sign * value for sign, value in zip(signs, differences)) / len(differences))
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        )
        extreme = sum(value + tolerance >= observed for value in estimates)
        return {
            "p_value_two_sided": extreme / exact_count,
            "method": "exact_sign_flip",
            "permutations": exact_count,
            "seed": None,
        }
    generator = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        estimate = abs(
            sum(value if generator.getrandbits(1) else -value for value in differences)
            / len(differences)
        )
        extreme += estimate + tolerance >= observed
    return {
        "p_value_two_sided": (extreme + 1) / (samples + 1),
        "method": "monte_carlo_sign_flip",
        "permutations": samples,
        "seed": seed,
    }


def _derive_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def compare_rows(
    real_rows: Sequence[LoadedRow],
    control_rows: Sequence[LoadedRow],
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> Dict[str, Any]:
    real_hits = [row.rank_one_based == 1 for row in real_rows]
    control_hits = [row.rank_one_based == 1 for row in control_rows]
    hit_differences = [float(real) - float(control) for real, control in zip(real_hits, control_hits)]
    reciprocal_rank_differences = [
        1.0 / real.rank_one_based - 1.0 / control.rank_one_based
        for real, control in zip(real_rows, control_rows)
    ]
    output: Dict[str, Any] = {
        "n": len(real_rows),
        "r_at_1_difference": paired_bootstrap_mean_ci(hit_differences, bootstrap_samples, seed),
        "mcnemar_exact": mcnemar_exact(real_hits, control_hits),
        "mrr_difference": {
            "bootstrap": paired_bootstrap_mean_ci(
                reciprocal_rank_differences, bootstrap_samples, seed + 3
            ),
            "permutation": paired_permutation_mean(
                reciprocal_rank_differences, permutation_samples, seed + 4
            ),
        },
    }
    real_margins = [_gold_margin(row) for row in real_rows]
    control_margins = [_gold_margin(row) for row in control_rows]
    if any(value is None for value in real_margins + control_margins):
        output["gold_margin_difference"] = {
            "available": False,
            "reason": "scores are absent for one or more paired rows",
        }
    else:
        differences = [
            float(real) - float(control)
            for real, control in zip(real_margins, control_margins)
        ]
        output["gold_margin_difference"] = {
            "available": True,
            "bootstrap": paired_bootstrap_mean_ci(differences, bootstrap_samples, seed + 1),
            "permutation": paired_permutation_mean(differences, permutation_samples, seed + 2),
        }
    return output


def _scores_by_candidate(loaded: LoadedRow) -> Dict[str, float] | None:
    if loaded.row["scores"] is None:
        return None
    return {
        candidate: float(score)
        for candidate, score in zip(loaded.candidate_keys, loaded.row["scores"])
    }


def null_subtracted_result(real: LoadedRow, no_prefix: LoadedRow) -> Dict[str, Any]:
    real_scores = _scores_by_candidate(real)
    null_scores = _scores_by_candidate(no_prefix)
    if real_scores is None or null_scores is None:
        raise AnalysisError("null subtraction requires scores in real and no_prefix rows")
    scores = [real_scores[candidate] - null_scores[candidate] for candidate in real.candidate_keys]
    order = _score_order(scores, real.score_direction)
    gold_position = real.candidate_keys.index(real.gold_key)
    predicted_position = order[0]
    rank_one_based = order.index(gold_position) + 1
    negatives = [value for index, value in enumerate(scores) if index != gold_position]
    if not negatives:
        margin = 0.0
    elif real.score_direction == "higher_is_better":
        margin = scores[gold_position] - max(negatives)
    else:
        margin = min(negatives) - scores[gold_position]
    return {
        "candidate_ids": list(real.row["candidate_ids"]),
        "scores": scores,
        "score_definition": "real_score_minus_no_prefix_score",
        "score_direction": real.score_direction,
        "gold_position": gold_position,
        "predicted_position": predicted_position,
        "rank": rank_one_based - 1,
        "rank_base": 0,
        "rank_one_based": rank_one_based,
        "gold_margin": margin,
    }


def _group_by_modality(rows: Sequence[LoadedRow]) -> Dict[str, List[LoadedRow]]:
    grouped: Dict[str, List[LoadedRow]] = {}
    for row in rows:
        grouped.setdefault(str(row.row["modality"]), []).append(row)
    return grouped


def analyze(
    input_paths: Mapping[str, Path],
    bootstrap_samples: int = 10000,
    permutation_samples: int = 10000,
    seed: int = 20260709,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if len(input_paths) < 2:
        raise AnalysisError("provide real=PATH and at least one control input")
    loaded: Dict[str, Dict[Key, LoadedRow]] = {}
    provenance_inputs: Dict[str, Any] = {}
    for condition, path in input_paths.items():
        loaded[condition], provenance_inputs[condition] = load_condition(condition, path)
    aligned = align_conditions(loaded)

    condition_metrics: Dict[str, Any] = {}
    for condition in input_paths:
        rows = [pair[condition] for pair in aligned]
        condition_metrics[condition] = {
            "overall": _metrics_for_rows(rows),
            "by_modality": {
                modality: _metrics_for_rows(modality_rows)
                for modality, modality_rows in sorted(_group_by_modality(rows).items())
            },
        }

    real_rows = [pair["real"] for pair in aligned]
    comparisons: Dict[str, Any] = {}
    for condition in input_paths:
        if condition == "real":
            continue
        condition_seed = _derive_seed(seed, condition)
        control_rows = [pair[condition] for pair in aligned]
        comparison: Dict[str, Any] = {
            "overall": compare_rows(
                real_rows,
                control_rows,
                bootstrap_samples,
                permutation_samples,
                condition_seed,
            ),
            "by_modality": {},
        }
        for modality in sorted(_group_by_modality(real_rows)):
            real_subset = [row for row in real_rows if row.row["modality"] == modality]
            control_subset = [
                pair[condition] for pair in aligned if pair["real"].row["modality"] == modality
            ]
            comparison["by_modality"][modality] = compare_rows(
                real_subset,
                control_subset,
                bootstrap_samples,
                permutation_samples,
                _derive_seed(condition_seed, modality),
            )
        comparisons[condition] = comparison

    no_prefix_name = next(
        (name for name in input_paths if name in {"no_prefix", "no_prefix_lm"}), None
    )
    null_rows: List[Dict[str, Any]] = []
    null_summary: Dict[str, Any]
    if no_prefix_name is None:
        null_summary = {"available": False, "reason": "no no_prefix input was provided"}
    else:
        missing_scores = [
            pair["real"].key
            for pair in aligned
            if pair["real"].row["scores"] is None or pair[no_prefix_name].row["scores"] is None
        ]
        if missing_scores:
            null_summary = {
                "available": False,
                "reason": f"scores are absent for {len(missing_scores)} paired rows",
            }
        else:
            null_rows = [null_subtracted_result(pair["real"], pair[no_prefix_name]) for pair in aligned]
            null_summary = {
                "available": True,
                "condition": f"real_minus_{no_prefix_name}",
                "overall": retrieval_metrics([row["rank_one_based"] for row in null_rows]),
                "by_modality": {},
            }
            for modality in sorted(_group_by_modality(real_rows)):
                ranks = [
                    null_rows[index]["rank_one_based"]
                    for index, pair in enumerate(aligned)
                    if pair["real"].row["modality"] == modality
                ]
                null_summary["by_modality"][modality] = retrieval_metrics(ranks)

    paired_output: List[Dict[str, Any]] = []
    for index, pair in enumerate(aligned):
        real = pair["real"]
        output_row: Dict[str, Any] = {
            "pair_key": {
                "modality": real.row["modality"],
                "query_uid": real.row["query_uid"],
                "candidate_set_hash": real.row["candidate_set_hash"],
                "gold_candidate_id": real.row["candidate_ids"][int(real.row["gold_position"])],
            },
            "conditions": {condition: loaded_row.row for condition, loaded_row in pair.items()},
        }
        if null_rows:
            output_row["null_subtracted"] = null_rows[index]
        paired_output.append(output_row)

    chance = {
        "overall": chance_summary(real_rows),
        "by_modality": {
            modality: chance_summary(rows)
            for modality, rows in sorted(_group_by_modality(real_rows).items())
        },
    }
    protocols = {
        condition: sorted(
            {_identity(_protocol_value(row.row)) for row in records.values()}
        )
        for condition, records in loaded.items()
    }
    report = {
        "schema_version": 1,
        "n": len(aligned),
        "conditions": condition_metrics,
        "comparisons_vs_real": comparisons,
        "null_subtracted": null_summary,
        "chance": chance,
        "randomized_positive_position_balance": positive_position_balance(real_rows),
        "consistency_checks": {
            "strict_key_alignment": True,
            "duplicate_keys_absent": True,
            "candidate_identity_sets_equal": True,
            "candidate_ids_unique": True,
            "gold_identity_consistent": True,
            "protocol_metadata_consistent": True,
            "rank_and_prediction_consistent_with_scores_when_present": True,
            "protocol_metadata_by_condition": protocols,
        },
        "seeds": {
            "base": seed,
            "bootstrap_samples": bootstrap_samples,
            "permutation_samples": permutation_samples,
            "condition_seeds": {
                condition: _derive_seed(seed, condition)
                for condition in input_paths
                if condition != "real"
            },
        },
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "tool": str(Path(__file__).resolve()),
            "python": sys.version,
            "inputs": provenance_inputs,
        },
    }
    return report, paired_output


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Paired Conditional-Matching Analysis",
        "",
        f"Paired queries: **{report['n']}**",
        "",
        "| condition | scope | n | R@1 (95% Wilson) | R@5 | R@10 | MRR | median rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, summaries in report["conditions"].items():
        scopes = [("all", summaries["overall"])] + list(summaries["by_modality"].items())
        for scope, metrics in scopes:
            r1 = metrics["r_at_1"]
            lines.append(
                f"| {condition} | {scope} | {metrics['n']} | "
                f"{r1['rate']:.4f} [{r1['wilson_95']['low']:.4f}, {r1['wilson_95']['high']:.4f}] "
                f"({r1['count']}/{r1['total']}) | {metrics['r_at_5']['rate']:.4f} | "
                f"{metrics['r_at_10']['rate']:.4f} | {metrics['mrr']:.4f} | "
                f"{metrics['median_rank']:.2f} |"
            )
    lines.extend([
        "",
        "| control | n | real-control R@1 (bootstrap 95% CI) | McNemar exact p | gold-margin delta (95% CI) | permutation p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for condition, comparison in report["comparisons_vs_real"].items():
        values = comparison["overall"]
        r1 = values["r_at_1_difference"]
        margin = values["gold_margin_difference"]
        if margin["available"]:
            bootstrap = margin["bootstrap"]
            margin_text = (
                f"{bootstrap['estimate']:.4f} [{bootstrap['ci_95']['low']:.4f}, "
                f"{bootstrap['ci_95']['high']:.4f}]"
            )
            permutation_text = f"{margin['permutation']['p_value_two_sided']:.6g}"
        else:
            margin_text = "n/a"
            permutation_text = "n/a"
        lines.append(
            f"| {condition} | {values['n']} | {r1['estimate']:.4f} "
            f"[{r1['ci_95']['low']:.4f}, {r1['ci_95']['high']:.4f}] | "
            f"{values['mcnemar_exact']['p_value_two_sided']:.6g} | {margin_text} | "
            f"{permutation_text} |"
        )
    if report["null_subtracted"]["available"]:
        metrics = report["null_subtracted"]["overall"]
        lines.extend([
            "",
            f"Null-subtracted (`{report['null_subtracted']['condition']}`) R@1: "
            f"**{metrics['r_at_1']['rate']:.4f}** ({metrics['r_at_1']['count']}/{metrics['n']}); "
            f"MRR: **{metrics['mrr']:.4f}**; median rank: **{metrics['median_rank']:.2f}**.",
        ])
    lines.append("")
    return "\n".join(lines)


def parse_named_inputs(values: Sequence[str]) -> Dict[str, Path]:
    inputs: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise AnalysisError(f"input must use NAME=PATH syntax: {value!r}")
        name, path_text = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise AnalysisError(f"invalid condition name: {name!r}")
        if name in inputs:
            raise AnalysisError(f"duplicate named input: {name}")
        if not path_text:
            raise AnalysisError(f"empty path for input {name}")
        inputs[name] = Path(path_text)
    return inputs


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", metavar="NAME=PATH")
    parser.add_argument("--input", action="append", default=[], dest="flag_inputs", metavar="NAME=PATH")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args(argv)
    try:
        input_paths = parse_named_inputs([*args.inputs, *args.flag_inputs])
        report, paired_rows = analyze(
            input_paths,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
            seed=args.seed,
        )
        output_json = Path(args.output_json)
        output_md = Path(args.output_md)
        output_jsonl = Path(args.output_jsonl)
        _write_json(output_json, report)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report), encoding="utf-8")
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for row in paired_rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
    except AnalysisError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "n": report["n"],
        "conditions": list(input_paths),
        "output_json": str(output_json),
        "output_md": str(output_md),
        "output_jsonl": str(output_jsonl),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
