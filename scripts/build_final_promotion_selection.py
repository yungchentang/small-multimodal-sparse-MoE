#!/usr/bin/env python3
"""Build the development-only promotion record for the frozen final model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class SelectionError(RuntimeError):
    """Raised when an input cannot support the promotion claim."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SelectionError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise SelectionError(f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise SelectionError(f"{label} is non-finite")
    return result


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise SelectionError(f"artifact is not a file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def require_dev_metric(
    path: Path, expected_checkpoint_sha256: str, candidates: int
) -> dict[str, Any]:
    payload = read_json(path)
    checks = {
        "evaluation_scope": payload.get("evaluation_scope") == "development",
        "not_sealed": payload.get("sealed_protocol") is False,
        "shared_prefix": payload.get("conditional_uses_multimodal_prefix") is True
        and payload.get("conditional_uses_direct_encoder_pooling") is False
        and payload.get("conditional_uses_lm_logits") is True
        and payload.get("eval_path") == "shared_prefix",
        "checkpoint": payload.get("e3_checkpoint_sha256") == expected_checkpoint_sha256,
        "queries": payload.get("conditional_image_eval_count") == 137
        and payload.get("conditional_speech_eval_count") == 137,
        "candidates": payload.get("conditional_candidates_per_query") == candidates
        and payload.get("conditional_speech_candidates_per_query") == candidates,
        "real_prefix": payload.get("prefix_control") == "real",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SelectionError(f"{path} failed development checks: {failed}")
    return payload


def old_v6_row(selector_path: Path, expected_root: Path) -> dict[str, Any]:
    payload = read_json(selector_path)
    matches = [
        row
        for row in payload.get("candidates", [])
        if Path(str(row.get("run_root", ""))).resolve() == expected_root.resolve()
    ]
    if len(matches) != 1:
        raise SelectionError("frozen v6 row is missing or ambiguous")
    source = matches[0]
    return {
        "name": "v6_frozen_router_expert_baseline",
        "run_root": str(expected_root.resolve(strict=True)),
        "valid": False,
        "selected": False,
        "rank": None,
        "reasons": [
            "Frozen historical baseline only; it is not eligible for final reselection.",
            "Image was weak, speech was near chance, and routing imbalance was high.",
        ],
        "selection_score": source.get("selection_score"),
        "metrics": source.get("metrics", {}),
        "artifacts": {},
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    selected_root = args.selected_root.resolve(strict=True)
    stage_a_root = args.stage_a_root.resolve(strict=True)
    v6_root = args.v6_root.resolve(strict=True)
    provenance = read_json(args.checkpoint_provenance)
    checkpoint = Path(str(provenance.get("checkpoint_path", ""))).resolve(strict=True)
    checkpoint_sha = str(provenance.get("checkpoint_sha256", ""))
    run_manifest = Path(str(provenance.get("run_manifest_path", ""))).resolve(strict=True)
    if provenance.get("passed") is not True or checkpoint.parent.parent != selected_root:
        raise SelectionError("checkpoint provenance does not belong to selected root")
    if sha256_file(checkpoint) != checkpoint_sha:
        raise SelectionError("selected checkpoint hash mismatch")
    if sha256_file(run_manifest) != provenance.get("run_manifest_sha256"):
        raise SelectionError("selected run manifest hash mismatch")

    r5 = require_dev_metric(args.dev_controls / "5way-real-stride" / "metrics.json", checkpoint_sha, 5)
    r10 = require_dev_metric(args.dev_controls / "10way-real-stride" / "metrics.json", checkpoint_sha, 10)
    h10 = require_dev_metric(args.dev_h10_metrics, checkpoint_sha, 10)
    if h10.get("negative_mode") != "hard_text":
        raise SelectionError("h10 must use hard_text negatives")

    text_path = selected_root / "E3_final_multimodal_top2_text_eval" / "metrics.json"
    e3_path = selected_root / "E3_final_multimodal_top2" / "metrics.json"
    text = read_json(text_path)
    e3 = read_json(e3_path)
    ppl = finite(text.get("perplexity"), "text perplexity")
    overflow = finite(
        e3.get("final_cycle_capacity_overflow_ratio_mean_assignment_weighted"),
        "final-cycle overflow",
    )
    inactive = finite(
        e3.get("final_cycle_inactive_expert_ratio_mean_assignment_weighted"),
        "final-cycle inactive ratio",
    )
    load_cv = finite(text.get("accepted_load_cv"), "text accepted load CV")
    metrics = {
        "text_ppl": ppl,
        "dev_5way_image_r1": finite(r5.get("conditional_image_to_text_r_at_1"), "r5 image"),
        "dev_5way_speech_r1": finite(r5.get("conditional_speech_to_text_r_at_1"), "r5 speech"),
        "dev_10way_image_r1": finite(r10.get("conditional_image_to_text_r_at_1"), "r10 image"),
        "dev_10way_speech_r1": finite(r10.get("conditional_speech_to_text_r_at_1"), "r10 speech"),
        "dev_h10_image_r1": finite(h10.get("conditional_image_to_text_r_at_1"), "h10 image"),
        "dev_h10_speech_r1": finite(h10.get("conditional_speech_to_text_r_at_1"), "h10 speech"),
        "routing_overflow": overflow,
        "routing_inactive": inactive,
        "routing_load_cv": load_cv,
        "capacity_factor": finite(text.get("capacity_factor"), "capacity factor"),
        "aux_coef": finite(text.get("aux_coef"), "aux coefficient"),
    }
    gates = {
        "text_ppl_le_13": ppl <= 13.0,
        "overflow_le_0p15": overflow <= 0.15,
        "r5_image_above_chance": metrics["dev_5way_image_r1"] > 0.2,
        "r5_speech_above_chance": metrics["dev_5way_speech_r1"] > 0.2,
        "r10_image_above_chance": metrics["dev_10way_image_r1"] > 0.1,
        "r10_speech_above_chance": metrics["dev_10way_speech_r1"] > 0.1,
        "h10_image_above_chance": metrics["dev_h10_image_r1"] > 0.1,
        "h10_speech_above_chance": metrics["dev_h10_speech_r1"] > 0.1,
    }
    if not all(gates.values()):
        raise SelectionError(f"promotion gates failed: {[k for k, v in gates.items() if not v]}")

    chances = (0.2, 0.2, 0.1, 0.1, 0.1, 0.1)
    recalls = tuple(metrics[key] for key in (
        "dev_5way_image_r1", "dev_5way_speech_r1", "dev_10way_image_r1",
        "dev_10way_speech_r1", "dev_h10_image_r1", "dev_h10_speech_r1",
    ))
    lifts = [(value - chance) / (1.0 - chance) for value, chance in zip(recalls, chances)]
    score = 1000.0 + 100.0 * min(lifts) + 10.0 * sum(lifts) / len(lifts) + (1.0 - overflow)

    split_manifest = read_json(args.development_split_manifest)
    ledger = read_json(args.dataset_ledger)
    if ledger.get("minimum_gate", {}).get("passed") is not True:
        raise SelectionError("dataset minimum gate did not pass")
    dataset_provenance = {
        "data_root_fingerprint": ledger.get("data_root_fingerprint"),
        "counts": ledger.get("counts"),
        "dataset_ledger_sha256": sha256_file(args.dataset_ledger),
        "development_split_manifest_sha256": sha256_file(args.development_split_manifest),
        "development_split_counts": split_manifest.get("counts"),
        "selection_scope": "train_and_development_only",
        "sealed_metrics_consumed": False,
    }
    checkpoint_args = read_json(Path(str(provenance.get("checkpoint_args_path", ""))))
    non_swept = {
        key: checkpoint_args.get(key)
        for key in (
            "seed", "base_model", "vision_model", "speech_model", "capacity_factor",
            "aux_coef", "image_prefix_tokens", "audio_prefix_tokens", "image_bridge_type",
            "audio_bridge_type", "speech_unfreeze_last_blocks", "expert_selection_method",
            "expert_update_mode", "final_steps",
        )
    }
    selected_artifacts = {
        "checkpoint": artifact(checkpoint),
        "run_manifest": artifact(run_manifest),
        "e3_metrics": artifact(e3_path),
        "text_metrics": artifact(text_path),
        "dev_5way_metrics": artifact(args.dev_controls / "5way-real-stride" / "metrics.json"),
        "dev_10way_metrics": artifact(args.dev_controls / "10way-real-stride" / "metrics.json"),
        "dev_h10_metrics": artifact(args.dev_h10_metrics),
        "dataset_ledger": artifact(args.dataset_ledger),
        "development_split_manifest": artifact(args.development_split_manifest),
    }
    selected = {
        "name": "stage_b_selected_experts_seed42",
        "run_root": str(selected_root),
        "valid": True,
        "selected": True,
        "rank": 1,
        "reasons": [],
        "selection_score": score,
        "metrics": metrics,
        "validation": {
            "promotion_gates": gates,
            "checkpoint_hash_chain": True,
            "shared_olmoe_prefix_path": True,
            "single_training_seed": 42,
            "h10_role": "post-freeze reporting-only validation; not used to alter the model or protocol",
        },
        "dataset_provenance": dataset_provenance,
        "dataset_provenance_sha256": canonical_sha256(dataset_provenance),
        "non_swept_args": non_swept,
        "non_swept_args_sha256": canonical_sha256(non_swept),
        "artifacts": selected_artifacts,
    }
    stage_a_text = read_json(stage_a_root / "E3_final_multimodal_top2_text_eval" / "metrics.json")
    stage_a = {
        "name": "stage_a_alignment_checkpoint",
        "run_root": str(stage_a_root),
        "valid": False,
        "selected": False,
        "rank": None,
        "reasons": ["Required alignment-stage ablation, not a final selected-expert candidate."],
        "metrics": {"text_ppl": finite(stage_a_text.get("perplexity"), "Stage A perplexity")},
        "artifacts": {},
    }
    return {
        "schema_version": 1,
        "status": "selected",
        "selection_policy": {
            "eligibility": "Final Stage-B checkpoint must pass frozen development text, routing, r5/r10, and reporting-only h10 gates.",
            "authoritative_order": [
                "retain the pre-sealed Stage-B selection fixed at protocol freeze",
                "verify PPL and routing promotion gates",
                "verify real-prefix r5/r10 above chance on the fixed development split",
                "report h10 as post-freeze validation without model reselection",
            ],
            "display_score": "1000 + 100*worst_normalized_lift + 10*mean_normalized_lift + (1-overflow)",
            "sealed_policy": "No sealed metrics are consumed; the sealed protocol and selected checkpoint remain frozen.",
        },
        "shared_validation": {
            "development_split_manifest": artifact(args.development_split_manifest),
            "dataset_ledger": artifact(args.dataset_ledger),
            "sealed_metrics_consumed": False,
            "single_seed_only": True,
        },
        "candidate_count": 3,
        "valid_candidate_count": 1,
        "selected_candidate": selected["name"],
        "candidates": [selected, stage_a, old_v6_row(args.frozen_selector, v6_root)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--v6-root", type=Path, required=True)
    parser.add_argument("--checkpoint-provenance", type=Path, required=True)
    parser.add_argument("--dev-controls", type=Path, required=True)
    parser.add_argument("--dev-h10-metrics", type=Path, required=True)
    parser.add_argument("--development-split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-ledger", type=Path, required=True)
    parser.add_argument("--frozen-selector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise SelectionError(f"refusing to overwrite {args.output}; pass --force")
    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
