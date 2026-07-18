from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

from scripts.analyze_paired_controls import (
    group_aware_chance_r_at_1,
    image_group_identity,
    positive_indices_for_group,
    production_bootstrap_r_at_1_ci,
    production_retrieval_metrics,
    reconstruct_and_validate_per_query_row,
    validate_per_query_identity_row,
)
from scripts.protocol_v2 import (
    FINAL_FEATURE_CACHE_POLICY,
    per_query_jsonl_sha256,
)
from scripts.sealed_position_allocator import (
    ALLOCATOR_NAME,
    ALLOCATOR_VERSION,
    assignment_provenance,
    enforce_gold_position_assignment,
    lexical_hard_negative_indices,
    permute_candidates_for_query,
    select_local_candidate_indices,
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sealed_image_manifest_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "image_path": f"/frozen/image-{index}.jpg",
            "caption": f"caption {index}",
            "media_sha256": hashlib.sha256(
                f"image-group-{index}".encode("utf-8")
            ).hexdigest(),
        }
        for index in range(count)
    ]


def sealed_image_manifest_content(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )


def sealed_speech_manifest_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"speech-{index}",
            "audio_path": f"/frozen/speech-{index}.wav",
            "transcript": f"reference transcript {index}",
            "speaker_id": f"speaker-{index // 2}",
            "source": {"source_ids": [f"utterance:{index}"]},
            "media_sha256": hashlib.sha256(
                f"speech-media-{index}".encode("utf-8")
            ).hexdigest(),
        }
        for index in range(count)
    ]


def sealed_speech_manifest_content(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )


_SEALED_RUN_ROW_CACHE: dict[str, list[dict[str, Any]]] = {}


def frozen_manifest_identity(
    row: Mapping[str, Any], modality: str, index: int
) -> str:
    """Mirror the production evaluator's stable sealed-row identity contract."""

    for key in (
        "uid",
        "source_uid",
        "image_uid",
        "utterance_id",
        "source_id",
        "id",
    ):
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


def sealed_per_query_rows_for_run(
    protocol: Mapping[str, Any],
    run: Mapping[str, Any],
    image_manifest_rows: Sequence[Mapping[str, Any]],
    speech_manifest_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build replay-exact per-query rows for one frozen matrix run."""

    allocator = protocol["gold_position_allocator"]
    cache_key = canonical_sha256({
        "cell_id": run["cell_id"],
        "negative_mode": run["negative_mode"],
        "conditional_negatives": run["conditional_negatives"],
        "conditional_candidates": run["conditional_candidates"],
        "image_query_count": run["image_query_count"],
        "speech_query_count": run["speech_query_count"],
        "image_eval_samples": run["image_eval_samples"],
        "speech_eval_samples": run["speech_eval_samples"],
        "query_offset": run["query_offset"],
        "candidate_offset": run["candidate_offset"],
        "candidate_seed": run["candidate_seed"],
        "control_seed": run["control_seed"],
        "allocator_plans_sha256": allocator["plans_sha256"],
        "image_manifest_rows": image_manifest_rows,
        "speech_manifest_rows": speech_manifest_rows,
    })
    cached = _SEALED_RUN_ROW_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)
    plans = {
        (str(plan["cell_id"]), str(plan["modality"])): plan
        for plan in allocator["plans"]
    }
    rows: list[dict[str, Any]] = []
    for modality, manifest_rows in (
        ("image", image_manifest_rows),
        ("speech", speech_manifest_rows),
    ):
        plan = plans[(str(run["cell_id"]), modality)]
        eval_samples = int(run[f"{modality}_eval_samples"])
        candidate_bank = list(manifest_rows[:eval_samples])
        text_key = "caption" if modality == "image" else "transcript"
        texts = [str(candidate[text_key]) for candidate in candidate_bank]
        group_ids = (
            [image_group_identity(candidate) for candidate in candidate_bank]
            if modality == "image"
            else None
        )
        hard_indices = (
            lexical_hard_negative_indices(
                texts, int(run["conditional_negatives"]), group_ids
            )
            if (
                int(run["conditional_negatives"]) >= 0
                and str(run["negative_mode"]) == "hard_text"
            )
            else []
        )
        query_start = int(run["query_offset"])
        query_count = int(run[f"{modality}_query_count"])
        for assignment_index, query_index in enumerate(
            range(query_start, query_start + query_count)
        ):
            if int(run["conditional_negatives"]) >= 0:
                query_seed = (
                    int(run["candidate_seed"])
                    + (1000003 if modality == "speech" else 0)
                    + 1009 * query_index
                )
                base_candidates = select_local_candidate_indices(
                    len(candidate_bank),
                    query_index,
                    int(run["conditional_negatives"]),
                    str(run["negative_mode"]),
                    candidate_seed=query_seed,
                    hard_indices=(
                        hard_indices[query_index] if hard_indices else ()
                    ),
                    group_ids=group_ids,
                )
            else:
                candidate_start = int(run["candidate_offset"])
                if candidate_start < 0:
                    candidate_start = query_start
                candidate_end = (
                    candidate_start + int(run["conditional_candidates"])
                )
                base_candidates = list(range(candidate_start, candidate_end))

            query_uid = frozen_manifest_identity(
                manifest_rows[query_index], modality, query_index
            )
            (
                candidate_indices,
                candidate_permutation,
                _gold_position,
                permutation_seed,
            ) = permute_candidates_for_query(
                base_candidates,
                query_index,
                int(run["control_seed"]),
                query_uid,
            )
            candidate_indices, candidate_permutation, gold_position = (
                enforce_gold_position_assignment(
                    candidate_indices,
                    candidate_permutation,
                    query_index,
                    int(plan["positions"][assignment_index]),
                )
            )
            candidate_ids = [
                frozen_manifest_identity(
                    manifest_rows[index], modality, index
                )
                for index in candidate_indices
            ]
            candidate_texts = [
                str(manifest_rows[index][text_key])
                for index in candidate_indices
            ]
            scores = [
                float(len(candidate_indices) - index)
                for index in range(len(candidate_indices))
            ]
            query_source = str(manifest_rows[query_index].get("source", ""))
            source_provenance: dict[str, Any] = {
                "query_uid": query_uid,
                "query_index": query_index,
                "query_source": query_source,
            }
            row: dict[str, Any] = {
                "modality": modality,
                "query_uid": query_uid,
                "query_index": query_index,
                "query_source": query_source,
                "candidate_ids": candidate_ids,
                "candidate_indices": candidate_indices,
                "candidate_permutation": candidate_permutation,
                "candidate_permutation_seed": permutation_seed,
                "gold_position_assignment": {
                    **assignment_provenance(plan, allocator),
                    "assignment_index": assignment_index,
                    "assigned_position": gold_position,
                },
                "candidate_texts": candidate_texts,
                "candidate_count": len(candidate_ids),
                "candidate_index": gold_position,
                "gold_index": gold_position,
                "gold_position": gold_position,
                "gold_candidate_index": query_index,
                "gold_candidate_id": candidate_ids[gold_position],
                "candidate_set_hash": hashlib.sha256(
                    json.dumps(
                        candidate_ids,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "predicted_position": 0,
                "rank": gold_position,
                "rank_base": 0,
                "scores": scores,
                "score_direction": "higher_is_better",
            }
            if modality == "image":
                assert group_ids is not None
                query_group_id = str(group_ids[query_index])
                candidate_group_ids = [
                    str(group_ids[index]) for index in candidate_indices
                ]
                positive_indices = positive_indices_for_group(
                    candidate_group_ids, query_group_id
                )
                row.update({
                    "query_image_group_id": query_group_id,
                    "candidate_group_ids": candidate_group_ids,
                    "positive_indices": positive_indices,
                    "positive_candidate_indices": [
                        candidate_indices[position]
                        for position in positive_indices
                    ],
                    "positive_candidate_ids": [
                        candidate_ids[position]
                        for position in positive_indices
                    ],
                    "positive_count": len(positive_indices),
                    "unique_candidate_group_count": len(
                        set(candidate_group_ids)
                    ),
                    "group_aware_chance_r_at_1": (
                        group_aware_chance_r_at_1(candidate_group_ids)
                    ),
                    "caption_row_chance_r_at_1": (
                        len(positive_indices) / len(candidate_group_ids)
                    ),
                })
                source_provenance["query_image_group_id"] = query_group_id
            else:
                speaker_id = manifest_rows[query_index].get("speaker_id")
                row["speaker_id"] = speaker_id
                source_provenance["speaker_id"] = speaker_id
            row["source_provenance"] = source_provenance
            rows.append(row)
    _SEALED_RUN_ROW_CACHE[cache_key] = copy.deepcopy(rows)
    return rows


def bind_sealed_image_group_contract(
    rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> None:
    image_query_index = 0
    for raw_row in rows:
        if raw_row.get("modality") != "image":
            continue
        if not isinstance(raw_row, dict):
            raise TypeError("fixture image row must be mutable")
        candidate_count = len(raw_row["candidate_ids"])
        if candidate_count > len(manifest_rows):
            raise ValueError("fixture image manifest is smaller than candidate set")
        query_index = image_query_index
        image_query_index += 1
        gold_position = int(raw_row["gold_position"])
        if query_index >= candidate_count or not 0 <= gold_position < candidate_count:
            raise ValueError("fixture image query/gold position is outside candidate set")
        candidate_indices = [
            index for index in range(candidate_count) if index != query_index
        ]
        candidate_indices.insert(gold_position, query_index)
        query_group_id = image_group_identity(manifest_rows[query_index])
        candidate_group_ids = [
            image_group_identity(manifest_rows[index])
            for index in candidate_indices
        ]
        candidate_ids = [
            f"image:manifest:{index}:group:{candidate_group_ids[position]}"
            for position, index in enumerate(candidate_indices)
        ]
        positive_indices = positive_indices_for_group(
            candidate_group_ids, query_group_id
        )
        raw_row.update({
            "candidate_ids": candidate_ids,
            "candidate_set_hash": hashlib.sha256(
                json.dumps(
                    candidate_ids, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "gold_candidate_id": candidate_ids[gold_position],
            "query_index": query_index,
            "candidate_indices": candidate_indices,
            "query_image_group_id": query_group_id,
            "candidate_group_ids": candidate_group_ids,
            "positive_indices": positive_indices,
            "positive_candidate_indices": [
                candidate_indices[position] for position in positive_indices
            ],
            "positive_candidate_ids": [
                candidate_ids[position] for position in positive_indices
            ],
            "positive_count": len(positive_indices),
            "unique_candidate_group_count": len(set(candidate_group_ids)),
            "group_aware_chance_r_at_1": group_aware_chance_r_at_1(
                candidate_group_ids
            ),
            "caption_row_chance_r_at_1": (
                len(positive_indices) / len(candidate_group_ids)
            ),
            "source_provenance": {
                "query_image_group_id": query_group_id,
            },
        })


def bind_sealed_speech_contract(
    rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> None:
    speech_query_index = 0
    for raw_row in rows:
        if raw_row.get("modality") != "speech":
            continue
        if not isinstance(raw_row, dict):
            raise TypeError("fixture speech row must be mutable")
        candidate_count = len(raw_row["candidate_ids"])
        if candidate_count > len(manifest_rows):
            raise ValueError("fixture speech manifest is smaller than candidate set")
        query_index = speech_query_index
        speech_query_index += 1
        gold_position = int(raw_row["gold_position"])
        if query_index >= candidate_count or not 0 <= gold_position < candidate_count:
            raise ValueError("fixture speech query/gold position is outside candidate set")
        candidate_indices = [
            index for index in range(candidate_count) if index != query_index
        ]
        candidate_indices.insert(gold_position, query_index)
        def speech_identity(index: int) -> str:
            manifest_row = manifest_rows[index]
            for key in (
                "uid",
                "source_uid",
                "image_uid",
                "utterance_id",
                "source_id",
                "id",
            ):
                value = manifest_row.get(key)
                if value not in (None, ""):
                    return f"speech:{value}"
            raise ValueError("fixture speech manifest row has no stable identity")

        candidate_ids = [speech_identity(index) for index in candidate_indices]
        candidate_texts = [
            str(manifest_rows[index]["transcript"]) for index in candidate_indices
        ]
        query_uid = speech_identity(query_index)
        query_source = str(manifest_rows[query_index].get("source", ""))
        speaker_id = manifest_rows[query_index].get("speaker_id")
        raw_row.update({
            "query_uid": query_uid,
            "query_index": query_index,
            "query_source": query_source,
            "speaker_id": speaker_id,
            "candidate_ids": candidate_ids,
            "candidate_indices": candidate_indices,
            "candidate_texts": candidate_texts,
            "candidate_count": candidate_count,
            "candidate_index": gold_position,
            "gold_index": gold_position,
            "gold_candidate_index": query_index,
            "gold_candidate_id": candidate_ids[gold_position],
            "candidate_set_hash": hashlib.sha256(
                json.dumps(
                    candidate_ids, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "source_provenance": {
                "query_uid": query_uid,
                "query_index": query_index,
                "query_source": query_source,
                "speaker_id": speaker_id,
            },
        })


def sealed_per_query_rows(
    modalities: Sequence[str],
    candidate_count: int,
    image_manifest_rows: Sequence[Mapping[str, Any]] | None = None,
    speech_manifest_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for query_index, modality in enumerate(modalities):
        candidate_ids = [
            f"{modality}:query:{query_index}:candidate:{candidate_index}"
            for candidate_index in range(candidate_count)
        ]
        scores = [float(candidate_count - index) for index in range(candidate_count)]
        rows.append({
            "modality": modality,
            "query_uid": f"{modality}:query:{query_index}",
            "candidate_ids": candidate_ids,
            "candidate_set_hash": hashlib.sha256(
                json.dumps(
                    candidate_ids, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "gold_position": 0,
            "gold_candidate_id": candidate_ids[0],
            "predicted_position": 0,
            "rank": 0,
            "rank_base": 0,
            "scores": scores,
            "score_direction": "higher_is_better",
        })
    if image_manifest_rows is not None:
        bind_sealed_image_group_contract(rows, image_manifest_rows)
    if speech_manifest_rows is not None:
        bind_sealed_speech_contract(rows, speech_manifest_rows)
    return rows


def complete_production_rows(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Populate fixture rows from the canonical production reconstruction."""

    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            raise TypeError(f"fixture row {index} must be mutable")
        if raw_row.get("modality") == "image":
            gold_position = int(raw_row["gold_position"])
            raw_row.setdefault("positive_indices", [gold_position])
            raw_row["positive_candidate_ids"] = [
                raw_row["candidate_ids"][position]
                for position in raw_row["positive_indices"]
            ]
        loaded = reconstruct_and_validate_per_query_row(
            str(raw_row["condition"]),
            "sealed metrics fixture",
            index + 1,
            raw_row,
            validate_production_fields=False,
        )
        raw_row.update(loaded.production_fields)


def complete_production_metrics(
    metrics: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    run: Mapping[str, Any],
) -> None:
    """Populate fixture aggregate fields from canonical production rows."""

    complete_production_rows(rows)
    validated_by_modality = {"image": [], "speech": []}
    for index, row in enumerate(rows):
        loaded = validate_per_query_identity_row(
            str(row["condition"]),
            "sealed metrics fixture",
            index + 1,
            row,
            require_production_fields=True,
        )
        validated_by_modality[str(row["modality"])].append(loaded)

    aliases = (
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
        aggregates = production_retrieval_metrics(
            validated_by_modality[modality], prefix
        )
        bootstrap = production_bootstrap_r_at_1_ci(
            validated_by_modality[modality],
            int(run["bootstrap_samples"]),
            int(run["bootstrap_seed"]) + offset,
        )
        aggregates.update({
            f"{prefix}_{field}": value
            for field, value in bootstrap.items()
        })
        metrics.update(aggregates)
        for suffix in aliases:
            field = f"{prefix}_{suffix}"
            metrics[f"conditional_{field}"] = aggregates.get(field)

    image_positive_counts = [
        int(row.production_fields["positive_count"])
        for row in validated_by_modality["image"]
    ]
    metrics["image_positive_counts"] = image_positive_counts
    metrics["conditional_image_positive_counts"] = image_positive_counts
    image_rows = [row for row in rows if row["modality"] == "image"]
    if image_rows and all("unique_candidate_group_count" in row for row in image_rows):
        unique_group_counts = [
            int(row["unique_candidate_group_count"]) for row in image_rows
        ]
        group_chance = sum(
            float(row["group_aware_chance_r_at_1"]) for row in image_rows
        ) / len(image_rows)
        caption_chance = sum(
            float(row["caption_row_chance_r_at_1"]) for row in image_rows
        ) / len(image_rows)
        metrics.update({
            "image_unique_candidate_group_counts": unique_group_counts,
            "conditional_image_unique_candidate_group_counts": unique_group_counts,
            "image_group_aware_chance_r_at_1": group_chance,
            "conditional_image_group_aware_chance_r_at_1": group_chance,
            "image_caption_row_chance_r_at_1": caption_chance,
            "conditional_image_caption_row_chance_r_at_1": caption_chance,
            "image_chance_r_at_1": caption_chance,
            "conditional_image_chance_r_at_1": caption_chance,
            "image_legacy_gold_caption_position_chance_r_at_1": (
                1.0 / int(run["requested_candidate_count"])
            ),
            "conditional_image_legacy_gold_caption_position_chance_r_at_1": (
                1.0 / int(run["requested_candidate_count"])
            ),
            "speech_chance_r_at_1": (
                1.0 / int(run["requested_candidate_count"])
            ),
            "conditional_speech_chance_r_at_1": (
                1.0 / int(run["requested_candidate_count"])
            ),
        })
    metrics["per_query_sha256"] = per_query_jsonl_sha256(rows)


def sealed_metrics(
    protocol: Mapping[str, Any],
    run: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol_file_sha256: str,
    checkpoint_sha256: str,
    evaluator_path: str,
    checkpoint_path: str,
    evaluator_sha256: str,
    stage_b_checkpoint_sha256: str = "e" * 64,
    checkpoint_restoration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    control = str(run["control"])
    condition = (
        "no_prefix" if control == "no-prefix" else str(run["prefix_control"])
    )
    metric_args = {
        field: run[field]
        for field in (
            "negative_mode",
            "conditional_negatives",
            "conditional_candidates",
            "conditional_queries",
            "image_eval_samples",
            "speech_eval_samples",
            "max_length",
            "conditional_batch_size",
            "query_offset",
            "candidate_offset",
            "tie_epsilon",
            "candidate_permutation",
            "randomize_positive_position",
            "prefix_control",
            "eval_path",
            "candidate_seed",
            "control_seed",
            "bootstrap_samples",
            "bootstrap_seed",
            "protocol_name",
            "eval_split_name",
        )
    }
    metric_args["evaluation_scope"] = "final"
    stage_b_sha = stage_b_checkpoint_sha256
    digests = {
        "image_cache_identity_sha256": "1" * 64,
        "audio_cache_identity_sha256": "2" * 64,
        "image_cache_payload_set_sha256": "3" * 64,
        "audio_cache_payload_set_sha256": "4" * 64,
        "image_produced_features_sha256": "5" * 64,
        "audio_produced_features_sha256": "6" * 64,
    }
    identity = {
        "frozen_evaluation_run_id": run["id"],
        "frozen_evaluation_cell_id": run["cell_id"],
        "frozen_evaluation_control": run["control"],
        "metric_affecting_args": metric_args,
        "evaluation_scope": "final",
        "strict_control": True,
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
        "eval_path": run["eval_path"],
        "condition": condition,
        "prefix_control": condition,
        "protocol_name": run["protocol_name"],
        "eval_split_name": run["eval_split_name"],
        "protocol_manifest_sha256": protocol_file_sha256,
        "protocol_content_sha256": protocol["protocol_content_sha256"],
        "e3_checkpoint_sha256": checkpoint_sha256,
        "stage_b_checkpoint_sha256": stage_b_sha,
        "e3_checkpoint_path": checkpoint_path,
        "evaluator_path": evaluator_path,
        "evaluator_sha256": evaluator_sha256,
        "feature_cache_policy": dict(FINAL_FEATURE_CACHE_POLICY),
        **digests,
    }
    if checkpoint_restoration is not None:
        identity["source_checkpoint_hashes"] = checkpoint_restoration[
            "source_checkpoint_hashes"
        ]
        identity["restoration_order"] = checkpoint_restoration[
            "restoration_order"
        ]
    allocator = protocol.get("gold_position_allocator")
    allocator_common: dict[str, Any] = {}
    allocator_bindings: dict[str, Mapping[str, Any]] = {}
    if isinstance(allocator, Mapping):
        plans = {
            (str(plan["cell_id"]), str(plan["modality"])): plan
            for plan in allocator["plans"]
        }
        allocator_common = {
            "gold_position_allocator_name": ALLOCATOR_NAME,
            "gold_position_allocator_version": ALLOCATOR_VERSION,
            "gold_position_assignment_plans_sha256": allocator["plans_sha256"],
        }
        allocator_bindings = {
            modality: assignment_provenance(
                plans[(str(run["cell_id"]), modality)], allocator
            )
            for modality in ("image", "speech")
        }
        identity.update(allocator_common)
        for modality, binding in allocator_bindings.items():
            identity[f"{modality}_gold_position_assignment"] = binding
    identity["evaluation_identity_sha256"] = canonical_sha256(identity)
    provenance = {
        "protocol_manifest_sha256": protocol_file_sha256,
        "protocol_content_sha256": protocol["protocol_content_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "stage_b_checkpoint_sha256": stage_b_sha,
        "checkpoint_path": checkpoint_path,
        "evaluator_path": evaluator_path,
        "evaluator_sha256": evaluator_sha256,
        "evaluation_identity_sha256": identity["evaluation_identity_sha256"],
    }
    if checkpoint_restoration is not None:
        provenance["checkpoint_restoration"] = checkpoint_restoration
        provenance["source_checkpoint_hashes"] = checkpoint_restoration[
            "source_checkpoint_hashes"
        ]
    for kind in ("image", "audio"):
        provenance[f"{kind}_feature_cache"] = {
            "policy": dict(FINAL_FEATURE_CACHE_POLICY),
            "base_identity_sha256": identity[
                f"{kind}_cache_identity_sha256"
            ],
            "payload_set_sha256": identity[
                f"{kind}_cache_payload_set_sha256"
            ],
            "produced_features_sha256": identity[
                f"{kind}_produced_features_sha256"
            ],
        }
    for row in rows:
        if isinstance(row, dict):
            row.update({
                "condition": condition,
                "prefix_control": condition,
                "eval_split_name": run["eval_split_name"],
                "negative_mode": run["negative_mode"],
                "eval_path": run["eval_path"],
            })
            row_protocol = dict(row.get("protocol", {}))
            row_protocol.update({
                "name": run["protocol_name"],
                "manifest_sha256": protocol_file_sha256,
                "eval_split_name": run["eval_split_name"],
                "negative_mode": run["negative_mode"],
                "candidate_count": run["requested_candidate_count"],
                "candidate_seed": run["candidate_seed"],
                "candidate_permutation_policy": run["candidate_permutation"],
                "randomized_positive_position": run[
                    "randomize_positive_position"
                ],
                "tie_epsilon": run["tie_epsilon"],
            })
            row["protocol"] = row_protocol
            row["evaluation_provenance"] = dict(identity)
    metrics = {
        "sealed_protocol": True,
        "evaluation_scope": "final",
        "strict_control": True,
        "protocol_manifest_sha256": protocol_file_sha256,
        "protocol_content_sha256": protocol["protocol_content_sha256"],
        "e3_checkpoint_sha256": checkpoint_sha256,
        "stage_b_checkpoint_sha256": stage_b_sha,
        "evaluation_identity_sha256": identity["evaluation_identity_sha256"],
        "evaluation_provenance": identity,
        "provenance": provenance,
        "negative_mode": run["negative_mode"],
        "candidate_count": run["requested_candidate_count"],
        "speech_candidate_count": run["requested_candidate_count"],
        "conditional_candidates_per_query": run["requested_candidate_count"],
        "e3_checkpoint_path": checkpoint_path,
        "conditional_speech_candidates_per_query": run[
            "requested_candidate_count"
        ],
        "image_eval_count": run["image_query_count"],
        "speech_eval_count": run["speech_query_count"],
        "conditional_image_eval_count": run["image_query_count"],
        "conditional_speech_eval_count": run["speech_query_count"],
        "query_offset": run["query_offset"],
        "candidate_offset": run["candidate_offset"],
        "tie_epsilon": run["tie_epsilon"],
        "candidate_seed": run["candidate_seed"],
        "control_seed": run["control_seed"],
        "candidate_permutation_policy": run["candidate_permutation"],
        "randomized_positive_position": run["randomize_positive_position"],
        "eval_path": run["eval_path"],
        "condition": condition,
        "prefix_control": condition,
        "protocol_name": run["protocol_name"],
        "eval_split_name": run["eval_split_name"],
        "conditional_uses_multimodal_prefix": control != "no-prefix",
        "per_query_rows": len(rows),
        "per_query_sha256": per_query_jsonl_sha256(rows),
    }
    if checkpoint_restoration is not None:
        metrics["source_checkpoint_hashes"] = checkpoint_restoration[
            "source_checkpoint_hashes"
        ]
        metrics["restoration_order"] = checkpoint_restoration[
            "restoration_order"
        ]
    metrics.update(allocator_common)
    for modality, binding in allocator_bindings.items():
        metrics[f"{modality}_gold_position_assignment"] = binding
        plan = next(
            plan
            for plan in allocator["plans"]
            if plan["cell_id"] == run["cell_id"]
            and plan["modality"] == modality
        )
        metrics[f"{modality}_gold_position_counts"] = list(
            plan["position_counts"]
        )
    for row in rows:
        if isinstance(row, dict):
            row["protocol"].update(allocator_common)
            row["evaluation_provenance"] = dict(identity)
    return metrics


def rebind_identity(metrics: dict[str, Any]) -> None:
    identity = metrics["evaluation_provenance"]
    identity.pop("evaluation_identity_sha256", None)
    digest = canonical_sha256(identity)
    identity["evaluation_identity_sha256"] = digest
    metrics["evaluation_identity_sha256"] = digest
    metrics["provenance"]["evaluation_identity_sha256"] = digest
