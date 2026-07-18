"""Deterministic balanced gold-position plans for sealed evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ALLOCATOR_NAME = "seed_bound_balanced_gold_positions"
ALLOCATOR_VERSION = 1
SEED_DERIVATION = "sha256_canonical_sort_v1"
MODALITIES = ("image", "speech")


class AssignmentPlanError(ValueError):
    """Raised when a sealed gold-position plan is invalid or mismatched."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _integer_sequence(values: Sequence[Any], label: str) -> list[int]:
    normalized: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise AssignmentPlanError(f"{label}[{index}] must be an integer")
        normalized.append(value)
    return normalized


def deterministic_permutation(size: int, seed: int) -> list[int]:
    """Replay the evaluator's exact CPU torch.randperm contract."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise AssignmentPlanError("permutation size must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AssignmentPlanError("permutation seed must be an integer")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production has evaluator deps.
        raise AssignmentPlanError(
            "PyTorch is required to replay the frozen candidate permutation"
        ) from exc
    generator = torch.Generator()
    generator.manual_seed(seed)
    return [
        int(value)
        for value in torch.randperm(size, generator=generator).tolist()
    ]


def query_permutation_seed(control_seed: int, query_identity: str) -> int:
    if isinstance(control_seed, bool) or not isinstance(control_seed, int):
        raise AssignmentPlanError("control_seed must be an integer")
    if not isinstance(query_identity, str) or not query_identity:
        raise AssignmentPlanError("query_identity must be a non-empty string")
    payload = f"{control_seed}\0{query_identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def validate_candidate_permutation(
    candidate_indices: Sequence[Any], permutation: Sequence[Any]
) -> Tuple[list[int], list[int]]:
    candidates = _integer_sequence(candidate_indices, "candidate_indices")
    applied = _integer_sequence(permutation, "candidate_permutation")
    if len(candidates) != len(applied):
        raise AssignmentPlanError("candidate/permutation length mismatch")
    if len(set(candidates)) != len(candidates):
        raise AssignmentPlanError("candidate_indices must be unique")
    if sorted(applied) != list(range(len(candidates))):
        raise AssignmentPlanError(
            "candidate_permutation must be a bijection over candidate positions"
        )
    return candidates, applied


def permute_candidates_for_query(
    candidate_indices: Sequence[Any],
    gold_candidate_index: int,
    control_seed: int,
    query_identity: str,
) -> Tuple[list[int], list[int], int, int]:
    candidates = _integer_sequence(candidate_indices, "candidate_indices")
    if len(set(candidates)) != len(candidates):
        raise AssignmentPlanError("candidate_indices must be unique")
    if int(gold_candidate_index) not in candidates:
        raise AssignmentPlanError(
            "candidate set does not contain the gold candidate"
        )
    permutation_seed = query_permutation_seed(control_seed, query_identity)
    permutation = deterministic_permutation(len(candidates), permutation_seed)
    permuted = [candidates[position] for position in permutation]
    return (
        permuted,
        permutation,
        permuted.index(int(gold_candidate_index)),
        permutation_seed,
    )


def lexical_hard_negative_indices(
    texts: Sequence[str],
    negatives: int,
    group_ids: Sequence[str] | None = None,
) -> List[List[int]]:
    """Replay the evaluator's checkpoint-independent lexical selector."""
    if group_ids is not None and len(group_ids) != len(texts):
        raise AssignmentPlanError("texts/group_ids length mismatch")
    token_sets = [
        set(re.findall(r"[a-z0-9]+", str(text).casefold()))
        for text in texts
    ]
    result: List[List[int]] = []
    for index, target in enumerate(token_sets):
        ranked: List[Tuple[float, int, int, int]] = []
        for candidate_index, candidate in enumerate(token_sets):
            if candidate_index == index or (
                group_ids is not None
                and str(group_ids[candidate_index]) == str(group_ids[index])
            ):
                continue
            overlap = len(target & candidate)
            union = len(target | candidate)
            ranked.append((
                -(overlap / max(1, union)),
                -overlap,
                abs(len(target) - len(candidate)),
                candidate_index,
            ))
        ranked.sort()
        if group_ids is not None and len(ranked) < max(0, negatives):
            raise AssignmentPlanError(
                "not enough cross-group candidates for hard-text replay"
            )
        result.append([item[-1] for item in ranked[:max(0, negatives)]])
    return result


def select_local_candidate_indices(
    total: int,
    query_index: int,
    negatives: int,
    mode: str,
    *,
    candidate_seed: int,
    hard_indices: Sequence[int] = (),
    group_ids: Sequence[str] | None = None,
) -> list[int]:
    """Replay the evaluator's local candidate selection before permutation."""
    if total <= 0 or not 0 <= query_index < total:
        raise AssignmentPlanError("query index is outside the candidate bank")
    if negatives < 0:
        raise AssignmentPlanError("local negative count must be non-negative")
    if group_ids is not None and len(group_ids) != total:
        raise AssignmentPlanError("candidate bank/group_ids length mismatch")
    query_group_id = (
        str(group_ids[query_index]) if group_ids is not None else None
    )

    def eligible(candidate_index: int) -> bool:
        return candidate_index != query_index and (
            group_ids is None
            or str(group_ids[candidate_index]) != query_group_id
        )

    if sum(eligible(index) for index in range(total)) < negatives:
        raise AssignmentPlanError(
            "not enough eligible negatives for candidate replay"
        )
    candidates = [query_index]
    used = {query_index}
    if mode == "hard_text":
        for value in hard_indices:
            negative = int(value)
            if negative in used or not eligible(negative):
                continue
            used.add(negative)
            candidates.append(negative)
            if len(candidates) >= negatives + 1:
                break
    if mode == "random" and len(candidates) < negatives + 1:
        for negative in deterministic_permutation(max(1, total), candidate_seed):
            if negative in used or not eligible(negative):
                continue
            used.add(negative)
            candidates.append(negative)
            if len(candidates) >= negatives + 1:
                break
    cursor = 0
    while len(candidates) < negatives + 1:
        negative = (query_index + 17 + 37 * cursor) % max(1, total)
        cursor += 1
        scanned = 0
        while (negative in used or not eligible(negative)) and scanned < total:
            negative = (negative + 1) % total
            scanned += 1
        if negative in used or not eligible(negative):
            raise AssignmentPlanError(
                "not enough eligible negatives for candidate replay"
            )
        used.add(negative)
        candidates.append(negative)
    return candidates


def _plan_context(
    *,
    candidate_seed: int,
    cell_id: str,
    modality: str,
    candidate_count: int,
    query_count: int,
    query_offset: int,
) -> Dict[str, Any]:
    return {
        "allocator_name": ALLOCATOR_NAME,
        "allocator_version": ALLOCATOR_VERSION,
        "candidate_seed": int(candidate_seed),
        "cell_id": str(cell_id),
        "modality": str(modality),
        "candidate_count": int(candidate_count),
        "query_count": int(query_count),
        "query_offset": int(query_offset),
    }


def _sort_digest(context: Mapping[str, Any], purpose: str, **values: int) -> str:
    return canonical_sha256({
        "context": dict(context),
        "purpose": purpose,
        **values,
    })


def build_assignment_plan(
    *,
    candidate_seed: int,
    cell_id: str,
    modality: str,
    candidate_count: int,
    query_count: int,
    query_offset: int = 0,
) -> Dict[str, Any]:
    """Build a seed-bound assignment whose position counts differ by at most one."""
    if isinstance(candidate_seed, bool) or not isinstance(candidate_seed, int):
        raise AssignmentPlanError("candidate_seed must be an integer")
    if not str(cell_id):
        raise AssignmentPlanError("cell_id must be non-empty")
    if modality not in MODALITIES:
        raise AssignmentPlanError(f"unsupported modality: {modality!r}")
    if isinstance(candidate_count, bool) or int(candidate_count) <= 1:
        raise AssignmentPlanError("candidate_count must be greater than one")
    if isinstance(query_count, bool) or int(query_count) <= 0:
        raise AssignmentPlanError("query_count must be positive")
    if isinstance(query_offset, bool) or int(query_offset) < 0:
        raise AssignmentPlanError("query_offset must be non-negative")

    context = _plan_context(
        candidate_seed=candidate_seed,
        cell_id=cell_id,
        modality=modality,
        candidate_count=candidate_count,
        query_count=query_count,
        query_offset=query_offset,
    )
    base_count, remainder = divmod(int(query_count), int(candidate_count))
    remainder_order = sorted(
        range(int(candidate_count)),
        key=lambda position: _sort_digest(
            context, "remainder_position", position=position
        ),
    )
    counts = [base_count] * int(candidate_count)
    for position in remainder_order[:remainder]:
        counts[position] += 1

    occurrences = [
        (position, occurrence)
        for position, count in enumerate(counts)
        for occurrence in range(count)
    ]
    occurrences.sort(
        key=lambda item: _sort_digest(
            context,
            "query_assignment",
            position=item[0],
            occurrence=item[1],
        )
    )
    positions = [position for position, _ in occurrences]
    if len(positions) != int(query_count) or max(counts) - min(counts) > 1:
        raise AssertionError("allocator produced an invalid balanced plan")

    assignment_id = f"{cell_id}:{modality}"
    return {
        "assignment_id": assignment_id,
        "cell_id": str(cell_id),
        "modality": modality,
        "candidate_count": int(candidate_count),
        "query_count": int(query_count),
        "query_offset": int(query_offset),
        "candidate_seed": int(candidate_seed),
        "seed_derivation": SEED_DERIVATION,
        "seed_context_sha256": canonical_sha256(context),
        "positions": positions,
        "position_counts": counts,
        "positions_sha256": canonical_sha256(positions),
    }


def build_allocator_manifest(
    cells: Sequence[Mapping[str, Any]],
    *,
    candidate_seed: int,
    query_counts: Mapping[str, int],
    query_offset: int = 0,
) -> Dict[str, Any]:
    plans = [
        build_assignment_plan(
            candidate_seed=candidate_seed,
            cell_id=str(cell["id"]),
            modality=modality,
            candidate_count=int(cell["candidate_count"]),
            query_count=int(query_counts[modality]),
            query_offset=query_offset,
        )
        for cell in cells
        for modality in MODALITIES
    ]
    return {
        "name": ALLOCATOR_NAME,
        "version": ALLOCATOR_VERSION,
        "candidate_seed": int(candidate_seed),
        "seed_derivation": SEED_DERIVATION,
        "balance_rule": "max(position_counts)-min(position_counts)<=1",
        "plans": plans,
        "plans_sha256": canonical_sha256(plans),
    }


def validate_allocator_manifest(
    manifest: Any,
    cells: Sequence[Mapping[str, Any]],
    *,
    candidate_seed: int,
    query_counts: Mapping[str, int],
    query_offset: int = 0,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    expected = build_allocator_manifest(
        cells,
        candidate_seed=candidate_seed,
        query_counts=query_counts,
        query_offset=query_offset,
    )
    if manifest != expected:
        raise AssignmentPlanError(
            "gold-position allocator manifest disagrees with its frozen seed-bound plan"
        )
    return {
        (str(plan["cell_id"]), str(plan["modality"])): dict(plan)
        for plan in expected["plans"]
    }


def assignment_provenance(
    plan: Mapping[str, Any], manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "allocator_name": ALLOCATOR_NAME,
        "allocator_version": ALLOCATOR_VERSION,
        "candidate_seed": int(plan["candidate_seed"]),
        "seed_derivation": str(plan["seed_derivation"]),
        "seed_context_sha256": str(plan["seed_context_sha256"]),
        "assignment_id": str(plan["assignment_id"]),
        "positions_sha256": str(plan["positions_sha256"]),
        "plans_sha256": str(manifest["plans_sha256"]),
    }


def positions_for_query_indices(
    plan: Mapping[str, Any], query_indices: Sequence[int]
) -> list[int]:
    expected_indices = list(
        range(
            int(plan["query_offset"]),
            int(plan["query_offset"]) + int(plan["query_count"]),
        )
    )
    observed_indices = [int(value) for value in query_indices]
    if observed_indices != expected_indices:
        raise AssignmentPlanError(
            "actual query indices disagree with the frozen gold-position assignment plan"
        )
    positions = [int(value) for value in plan["positions"]]
    validate_executed_positions(positions, plan)
    return positions


def enforce_gold_position_assignment(
    candidate_indices: Sequence[int],
    permutation: Sequence[int],
    gold_candidate_index: int,
    assigned_position: int,
) -> Tuple[list[int], list[int], int]:
    """Move gold to its frozen position while preserving permutation provenance."""
    candidates, applied_permutation = validate_candidate_permutation(
        candidate_indices, permutation
    )
    if int(gold_candidate_index) not in candidates:
        raise AssignmentPlanError("candidate set does not contain the gold candidate")
    if not 0 <= int(assigned_position) < len(candidates):
        raise AssignmentPlanError("assigned gold position is outside the candidate set")
    current_position = candidates.index(int(gold_candidate_index))
    candidates[current_position], candidates[int(assigned_position)] = (
        candidates[int(assigned_position)],
        candidates[current_position],
    )
    applied_permutation[current_position], applied_permutation[int(assigned_position)] = (
        applied_permutation[int(assigned_position)],
        applied_permutation[current_position],
    )
    return candidates, applied_permutation, int(assigned_position)


def validate_executed_positions(
    positions: Sequence[int], plan: Mapping[str, Any]
) -> None:
    candidate_count = int(plan["candidate_count"])
    observed = [int(value) for value in positions]
    if any(value < 0 or value >= candidate_count for value in observed):
        raise AssignmentPlanError("executed gold position is outside the candidate set")
    counts = [observed.count(position) for position in range(candidate_count)]
    if (
        observed != [int(value) for value in plan["positions"]]
        or counts != [int(value) for value in plan["position_counts"]]
        or canonical_sha256(observed) != plan["positions_sha256"]
        or max(counts) - min(counts) > 1
    ):
        raise AssignmentPlanError(
            "executed gold positions disagree with the frozen balanced assignment plan"
        )
