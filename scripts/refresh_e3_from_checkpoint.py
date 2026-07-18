"""Regenerate E3 evaluation artifacts from a saved checkpoint.

This script is intentionally used for provenance repair without mutating the
original training root: it mirrors an existing completed root into a new output
root, reloads the saved E3 checkpoint, re-runs E3 text and multimodal evaluation,
and rewrites only the new root's E3 evaluation artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Tuple

import torch

from scripts.eval_conditional_retrieval import load_trained_wrapper, sha256_file
from scripts.extract_checkpoint_provenance import (
    IGNORED_ARG_MISMATCHES,
    compare_args,
    validate_e3_metrics_checkpoint_identity,
)
from training.olmoe_required_runs import cleanup, save_json
from training.olmoe_real_subset_runs import (
    FeatureCache,
    evaluate_text_blocks,
    load_speech_text_tokenizer,
    load_vision_text_tokenizer,
    read_jsonl,
    retrieval_eval,
    split_tail,
)


def mirror_source(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Output root already exists: {output}")
    ignore = shutil.ignore_patterns("summary.json", "requirement_audit.json", "real_required_runs_table.csv", "dataset_counts.csv")
    shutil.copytree(source, output, ignore=ignore)


def namespace_from_checkpoint(state: Dict[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    checkpoint_args = state.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("E3 checkpoint is missing args mapping")
    raw = dict(checkpoint_args)
    required = (
        "base_model",
        "vision_model",
        "speech_model",
        "top_k",
        "capacity_factor",
        "aux_coef",
        "output_dir",
    )
    missing = [field for field in required if raw.get(field) in (None, "")]
    if missing:
        raise ValueError(
            f"E3 checkpoint is missing required refresh args: {missing}"
        )
    for field in ("base_model", "vision_model", "speech_model"):
        supplied = getattr(args, field, "")
        if supplied and supplied != raw[field]:
            raise ValueError(
                f"refresh {field} disagrees with E3 checkpoint: "
                f"cli={supplied!r} checkpoint={raw[field]!r}"
            )
    raw.update(
        {
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "feature_cache_dir": str(
                args.feature_cache_dir
                or (args.source_output_dir / "feature_cache")
            ),
            "run_output_dir": str(args.source_output_dir),
            "checkpoint": str(args.refreshed_checkpoint),
            "stage_b_checkpoint": str(args.stage_b_checkpoint or ""),
            "stage_b_checkpoint_sha256": str(args.stage_b_checkpoint_sha256),
            "evaluation_scope": "final",
        }
    )
    defaults = {
        "max_length": 512,
        "eval_batch_size": 8,
        "text_eval_blocks": 160,
        "retrieval_eval_samples": 250,
        "conditional_eval_samples": 250,
        "conditional_negatives": 4,
        "conditional_batch_size": 16,
        "image_eval_samples": 250,
        "speech_eval_samples": 250,
        "image_prefix_tokens": 50,
        "audio_prefix_tokens": 64,
        "encoder_feature_tokens": 100,
        "sample_rate": 16000,
        "speech_target_space": "olmoe_text_hidden",
        "alignment_prefix_residual": False,
    }
    for key, value in defaults.items():
        raw.setdefault(key, value)
    return SimpleNamespace(**raw)


def load_verified_e3_checkpoint(
    checkpoint: Path, expected_sha256: str
) -> Tuple[Path, bytes, Dict[str, Any]]:
    if len(expected_sha256) != 64 or any(
        char not in "0123456789abcdefABCDEF" for char in expected_sha256
    ):
        raise ValueError(
            "--checkpoint-sha256 must be an exact 64-character SHA256"
        )
    raw_path = checkpoint.expanduser()
    if raw_path.is_symlink():
        raise ValueError("E3 checkpoint cannot be a symlink")
    resolved = raw_path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("E3 checkpoint must be a regular file")
    payload = resolved.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            "E3 checkpoint SHA256 mismatch: "
            f"expected={expected_sha256.lower()} observed={actual_sha256}"
        )
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("E3 checkpoint payload must be a mapping")
    return resolved, payload, state


def resolve_refreshed_checkpoint(
    source: Path,
    output: Path,
    supplied_checkpoint: Path,
    expected_sha256: str,
    mirrored: bool,
) -> Path:
    relative_checkpoint = Path("E3_final_multimodal_top2/checkpoint_final.pt")
    expected_output_checkpoint = (output / relative_checkpoint).resolve(strict=True)
    expected_input_checkpoint = (
        (source / relative_checkpoint).resolve(strict=True)
        if mirrored
        else expected_output_checkpoint
    )
    if supplied_checkpoint != expected_input_checkpoint:
        raise ValueError(
            "E3 refresh checkpoint path must identify the exact standard artifact: "
            f"expected={expected_input_checkpoint} supplied={supplied_checkpoint}"
        )
    actual_sha256 = hashlib.sha256(expected_output_checkpoint.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            "refreshed E3 checkpoint SHA256 mismatch: "
            f"expected={expected_sha256.lower()} observed={actual_sha256}"
        )
    return expected_output_checkpoint


def checkpoint_artifact_provenance(
    checkpoint: Path, checkpoint_sha256: str
) -> Dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": int(checkpoint.stat().st_size),
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _recorded_path(value: Any, anchor: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("strict checkpoint sidecar contains an empty artifact path")
    raw = Path(value).expanduser()
    return (raw if raw.is_absolute() else anchor / raw).resolve(strict=False)


def _strict_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(
                    f"E3 train_metrics.jsonl contains blank line {line_number}"
                )
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"E3 train_metrics.jsonl row {line_number} is not an object"
                )
            rows.append(row)
    if not rows:
        raise ValueError("E3 train_metrics.jsonl contains no rows")
    return rows


def verify_original_e3_artifacts(
    checkpoint: Path,
    state: Mapping[str, Any],
    metrics_path: Path,
    training_path: Path,
) -> Dict[str, Any]:
    """Verify the immutable pre-refresh E3 evidence chain."""

    sidecar_path = checkpoint.parent / "checkpoint_provenance.json"
    if not sidecar_path.is_file():
        raise ValueError(
            "E3 refresh requires checkpoint_provenance.json for strict audit"
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict):
        raise ValueError("E3 checkpoint provenance sidecar must be an object")

    required_fields = {
        "schema_version",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "checkpoint_state_keys",
        "checkpoint_last_row_sha256",
        "checkpoint_last_row_summary",
        "metrics_path",
        "metrics_sha256",
        "metrics_step_rows",
        "checkpoint_last_row_matches_metrics_last_row",
        "checkpoint_args_path",
        "checkpoint_args_sha256",
        "run_manifest_path",
        "run_manifest_sha256",
        "ignored_path_only_arg_keys",
        "non_path_arg_mismatches",
        "passed",
    }
    missing = sorted(required_fields - set(sidecar))
    if missing:
        raise ValueError(
            f"E3 checkpoint provenance sidecar is missing strict fields: {missing}"
        )
    if sidecar["schema_version"] != 1 or sidecar["passed"] is not True:
        raise ValueError(
            "E3 checkpoint provenance sidecar is not a passing schema-v1 artifact"
        )

    checkpoint = checkpoint.resolve(strict=True)
    metrics_path = metrics_path.resolve(strict=True)
    training_path = training_path.resolve(strict=True)
    if _recorded_path(sidecar["checkpoint_path"], checkpoint.parent) != checkpoint:
        raise ValueError("E3 checkpoint provenance sidecar points to another checkpoint")
    if sidecar["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("E3 checkpoint provenance sidecar checkpoint digest mismatch")
    if sidecar["checkpoint_size_bytes"] != checkpoint.stat().st_size:
        raise ValueError("E3 checkpoint provenance sidecar checkpoint size mismatch")

    checkpoint_last_row = state.get("last_row")
    checkpoint_args = state.get("args")
    if not isinstance(checkpoint_last_row, Mapping):
        raise ValueError("E3 checkpoint does not contain a last_row mapping")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("E3 checkpoint does not contain an args mapping")
    state_keys = sorted(str(key) for key in state)
    if sidecar["checkpoint_state_keys"] != state_keys:
        raise ValueError("E3 checkpoint provenance sidecar state keys mismatch")
    if sidecar["checkpoint_last_row_sha256"] != canonical_sha256(
        checkpoint_last_row
    ):
        raise ValueError("E3 checkpoint provenance sidecar last-row digest mismatch")
    expected_summary = {
        key: checkpoint_last_row.get(key)
        for key in (
            "step",
            "modality",
            "loss",
            "lm_ce_loss",
            "router_aux_loss_raw",
            "router_aux_loss_weighted",
            "hf_reported_loss_minus_explicit_base",
        )
    }
    if sidecar["checkpoint_last_row_summary"] != expected_summary:
        raise ValueError("E3 checkpoint provenance sidecar last-row summary mismatch")

    if _recorded_path(sidecar["metrics_path"], checkpoint.parent) != metrics_path:
        raise ValueError("E3 checkpoint provenance sidecar points to other E3 metrics")
    if sidecar["metrics_sha256"] != sha256_file(metrics_path):
        raise ValueError("E3 checkpoint provenance sidecar metrics digest mismatch")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, Mapping):
        raise ValueError("original E3 metrics must be an object")
    validate_e3_metrics_checkpoint_identity(metrics, checkpoint, metrics_path)
    metrics_steps = metrics.get("steps")
    if (
        not isinstance(metrics_steps, list)
        or not metrics_steps
        or not all(isinstance(row, Mapping) for row in metrics_steps)
    ):
        raise ValueError("original E3 metrics must contain non-empty object steps")
    normalized_steps = [dict(row) for row in metrics_steps]
    if sidecar["metrics_step_rows"] != len(normalized_steps):
        raise ValueError("E3 checkpoint provenance sidecar metrics row count mismatch")
    if normalized_steps[-1] != dict(checkpoint_last_row):
        raise ValueError("E3 checkpoint last_row does not match original metrics")
    if sidecar["checkpoint_last_row_matches_metrics_last_row"] is not True:
        raise ValueError("E3 checkpoint provenance sidecar final-row flag is false")

    training_rows = _strict_jsonl_rows(training_path)
    if training_rows != normalized_steps:
        raise ValueError(
            "E3 raw train_metrics.jsonl full row sequence differs from original metrics"
        )
    training_rows_sha256 = canonical_sha256(training_rows)
    if (
        "training_rows_sha256" in sidecar
        and sidecar["training_rows_sha256"] != training_rows_sha256
    ):
        raise ValueError("E3 checkpoint provenance sidecar training-row digest mismatch")

    checkpoint_args_path = _recorded_path(
        sidecar["checkpoint_args_path"], checkpoint.parent
    )
    if not checkpoint_args_path.is_file():
        raise ValueError("E3 strict checkpoint args artifact is missing")
    if sidecar["checkpoint_args_sha256"] != sha256_file(checkpoint_args_path):
        raise ValueError("E3 strict checkpoint args digest mismatch")
    saved_args = json.loads(checkpoint_args_path.read_text(encoding="utf-8"))
    if saved_args != dict(checkpoint_args):
        raise ValueError("E3 strict checkpoint args differ from checkpoint truth")

    run_manifest_path = _recorded_path(
        sidecar["run_manifest_path"], checkpoint.parent
    )
    if not run_manifest_path.is_file():
        raise ValueError("E3 strict run manifest artifact is missing")
    if sidecar["run_manifest_sha256"] != sha256_file(run_manifest_path):
        raise ValueError("E3 strict run manifest digest mismatch")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest_args = (
        run_manifest.get("args") if isinstance(run_manifest, Mapping) else None
    )
    if not isinstance(manifest_args, Mapping):
        raise ValueError("E3 strict run manifest does not contain args")
    mismatches = compare_args(checkpoint_args, manifest_args)
    if mismatches or sidecar["non_path_arg_mismatches"] != mismatches:
        raise ValueError("E3 checkpoint and strict run manifest args mismatch")
    if sidecar["ignored_path_only_arg_keys"] != sorted(IGNORED_ARG_MISMATCHES):
        raise ValueError("E3 strict sidecar ignored-argument contract mismatch")

    return {
        "sidecar": sidecar,
        "training_rows": training_rows,
        "training_rows_sha256": training_rows_sha256,
        "checkpoint_args_path": checkpoint_args_path,
        "run_manifest_path": run_manifest_path,
    }


def load_refresh_runtime(run_args, checkpoint_bytes: bytes):
    return load_trained_wrapper(run_args, checkpoint_bytes=checkpoint_bytes)


def refresh_checkpoint_sidecar(
    checkpoint: Path,
    state: Mapping[str, Any],
    metrics_path: Path,
    training_path: Path,
    checkpoint_args_path: Path,
    run_manifest_path: Path,
) -> Dict[str, Any]:
    sidecar_path = checkpoint.parent / "checkpoint_provenance.json"
    checkpoint_last_row = state.get("last_row")
    checkpoint_args = state.get("args")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_steps = metrics.get("steps") if isinstance(metrics, Mapping) else None
    training_rows = _strict_jsonl_rows(training_path)
    if (
        not isinstance(checkpoint_last_row, Mapping)
        or not isinstance(checkpoint_args, Mapping)
        or not isinstance(metrics_steps, list)
        or not metrics_steps
        or metrics_steps[-1] != dict(checkpoint_last_row)
        or metrics_steps != training_rows
    ):
        raise ValueError(
            "E3 checkpoint, refreshed metrics, and raw training rows do not match"
        )
    saved_args = json.loads(checkpoint_args_path.read_text(encoding="utf-8"))
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest_args = manifest.get("args") if isinstance(manifest, Mapping) else None
    if saved_args != dict(checkpoint_args) or not isinstance(manifest_args, Mapping):
        raise ValueError("E3 checkpoint companion artifacts differ from checkpoint truth")
    mismatches = compare_args(checkpoint_args, manifest_args)
    if mismatches:
        raise ValueError("E3 checkpoint and run manifest args mismatch")

    sidecar = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_state_keys": sorted(str(key) for key in state),
        "checkpoint_last_row_sha256": canonical_sha256(checkpoint_last_row),
        "checkpoint_last_row_summary": {
            key: checkpoint_last_row.get(key)
            for key in (
                "step",
                "modality",
                "loss",
                "lm_ce_loss",
                "router_aux_loss_raw",
                "router_aux_loss_weighted",
                "hf_reported_loss_minus_explicit_base",
            )
        },
        "checkpoint_last_row_matches_metrics_last_row": True,
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "metrics_step_rows": len(metrics_steps),
        "metrics_steps_sha256": canonical_sha256(metrics_steps),
        "raw_training_path": str(training_path.resolve()),
        "raw_training_sha256": sha256_file(training_path),
        "raw_training_rows": len(training_rows),
        "training_rows_sha256": canonical_sha256(training_rows),
        "checkpoint_args_path": str(checkpoint_args_path.resolve()),
        "checkpoint_args_sha256": sha256_file(checkpoint_args_path),
        "run_manifest_path": str(run_manifest_path.resolve()),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "ignored_path_only_arg_keys": sorted(IGNORED_ARG_MISMATCHES),
        "non_path_arg_mismatches": mismatches,
        "passed": True,
    }
    save_json(sidecar_path, sidecar)
    return sidecar


def build_text_eval_provenance(
    checkpoint: Path,
    checkpoint_sha256: str,
    training_steps: int,
    lm_trainable: bool,
) -> Dict[str, Any]:
    checkpoint_stat = checkpoint.stat()
    text_eval_note = (
        "This final run trains selected language-model parameters, so text metrics "
        "must be read from the E3 checkpoint provenance rather than assumed equal "
        "to E2."
        if lm_trainable
        else "This final adapter run freezes the LM/router/expert weights, so "
        "text-only metrics may remain close to calibrated Top-2 while multimodal "
        "prefix modules change."
    )
    return {
        "source_experiment_id": "E3_final_multimodal_top2",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "source_training_steps": int(training_steps),
        "source_checkpoint_saved_before_eval": True,
        "model_state_source": "checkpoint_reloaded_for_e3_eval_refresh",
        "copied_from_e2": False,
        "lm_trainable": lm_trainable,
        "text_eval_note": text_eval_note,
    }


def trend_artifact(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    vals = [float(row[key]) for row in rows if key in row and row[key] is not None and math.isfinite(float(row[key]))]
    if not vals:
        return {"first": float("nan"), "last": float("nan"), "min": float("nan")}
    return {"first": vals[0], "last": vals[-1], "min": min(vals)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/real_subset_final"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--stage-b-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-b-checkpoint-sha256", required=True)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--vision-model", default="")
    parser.add_argument("--speech-model", default="")
    parser.add_argument("--mirror-source-root", action="store_true")
    args = parser.parse_args()

    source = args.source_output_dir.expanduser().resolve(strict=True)
    out_dir = args.output_dir.expanduser().resolve(strict=False)
    args.source_output_dir = source
    args.output_dir = out_dir
    supplied_checkpoint, checkpoint_bytes, state = load_verified_e3_checkpoint(
        args.checkpoint, args.checkpoint_sha256
    )
    original_root = source if args.mirror_source_root else out_dir
    original_e3_dir = original_root / "E3_final_multimodal_top2"
    expected_original_checkpoint = (
        original_e3_dir / "checkpoint_final.pt"
    ).resolve(strict=True)
    if supplied_checkpoint != expected_original_checkpoint:
        raise ValueError(
            "E3 refresh checkpoint path must identify the original standard artifact: "
            f"expected={expected_original_checkpoint} supplied={supplied_checkpoint}"
        )
    original = verify_original_e3_artifacts(
        supplied_checkpoint,
        state,
        original_e3_dir / "metrics.json",
        original_e3_dir / "train_metrics.jsonl",
    )
    rows = list(original["training_rows"])

    if args.mirror_source_root:
        mirror_source(source, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def refreshed_companion_path(path: Path) -> Path:
        if not args.mirror_source_root:
            return path
        try:
            relative = path.resolve(strict=True).relative_to(source)
        except ValueError:
            return path
        return (out_dir / relative).resolve(strict=True)

    checkpoint_args_path = refreshed_companion_path(
        original["checkpoint_args_path"]
    )
    run_manifest_path = refreshed_companion_path(original["run_manifest_path"])
    checkpoint = resolve_refreshed_checkpoint(
        source,
        out_dir,
        supplied_checkpoint,
        args.checkpoint_sha256,
        bool(args.mirror_source_root),
    )
    args.refreshed_checkpoint = checkpoint
    run_args = namespace_from_checkpoint(state, args)
    (
        wrapper,
        tokenizer,
        meta,
        image_processor,
        vision_model,
        speech_processor,
        speech_model,
        device,
    ) = load_refresh_runtime(run_args, checkpoint_bytes)

    text_blocks_eval = read_jsonl(args.data_dir / "text_blocks_eval.jsonl")
    image_rows = read_jsonl(args.data_dir / "image_captions.jsonl")
    audio_rows = read_jsonl(args.data_dir / "speech_transcripts.jsonl")
    _, image_eval = split_tail(image_rows, int(run_args.image_eval_samples))
    _, audio_eval = split_tail(audio_rows, int(run_args.speech_eval_samples))

    exp_id = "E3_final_multimodal_top2"
    checkpoint_sha256 = args.checkpoint_sha256.lower()
    checkpoint_provenance = checkpoint_artifact_provenance(
        checkpoint, checkpoint_sha256
    )
    trainable_meta = dict(state.get("trainable_meta") or {})
    lm_trainable = bool(
        trainable_meta.get("train_router_gates")
        or trainable_meta.get("train_experts")
        or trainable_meta.get("train_lm_head")
    )
    text_eval_provenance = build_text_eval_provenance(
        checkpoint, checkpoint_sha256, len(rows), lm_trainable
    )

    text_eval = evaluate_text_blocks(
        f"{exp_id}_text_eval",
        wrapper.lm,
        tokenizer,
        list(text_blocks_eval)[: int(run_args.text_eval_blocks)],
        out_dir,
        {**meta, "capacity_factor": float(run_args.capacity_factor), "aux_coef": float(run_args.aux_coef), "provenance": text_eval_provenance},
        int(run_args.max_length),
        int(run_args.eval_batch_size),
    )
    cache = FeatureCache(Path(run_args.feature_cache_dir))
    retrieval = retrieval_eval(
        wrapper,
        tokenizer,
        image_processor,
        vision_model,
        vision_text_tokenizer,
        speech_processor,
        speech_model,
        speech_text_tokenizer,
        image_eval,
        audio_eval,
        device,
        run_args,
        cache,
    )
    loss_trend = trend_artifact(rows, "loss")
    artifact = {
        "meta": {**meta, "capacity_factor": float(run_args.capacity_factor), "aux_coef": float(run_args.aux_coef), **trainable_meta},
        **checkpoint_provenance,
        "checkpoint_restoration": meta.get("checkpoint_restoration"),
        "text_eval_provenance": text_eval_provenance,
        "real_subset": True,
        "vision_model": run_args.vision_model,
        "speech_model": run_args.speech_model,
        "steps": rows,
        "first_loss": loss_trend["first"],
        "last_loss": loss_trend["last"],
        "min_loss": loss_trend["min"],
        "text_eval": {k: v for k, v in text_eval.items() if k not in {"expert_counts_total"}},
        "retrieval_eval": retrieval,
        "final_gate_entropy_mean": rows[-1].get("gate_entropy_mean"),
        "final_inactive_expert_ratio_mean": rows[-1].get("inactive_expert_ratio_mean"),
        "final_capacity_overflow_ratio_mean": rows[-1].get("capacity_overflow_ratio_mean"),
        "final_modality_expert_utilization": rows[-1].get("modality_expert_utilization"),
        "final_modality_js": {
            "image_audio": rows[-1].get("modality_js_image_audio"),
            "image_text": rows[-1].get("modality_js_image_text"),
            "audio_text": rows[-1].get("modality_js_audio_text"),
        },
    }
    metrics_path = out_dir / exp_id / "metrics.json"
    save_json(metrics_path, artifact)
    checkpoint_sidecar = refresh_checkpoint_sidecar(
        checkpoint,
        state,
        metrics_path,
        checkpoint.parent / "train_metrics.jsonl",
        checkpoint_args_path,
        run_manifest_path,
    )
    refresh = {
        "type": "checkpoint_eval_refresh",
        "source_output_dir": str(source),
        "output_dir": str(out_dir),
        **checkpoint_provenance,
        "requested_checkpoint_path": str(supplied_checkpoint),
        "stage_b_checkpoint": meta.get("checkpoint_restoration", {}).get(
            "stage_b_checkpoint"
        ),
        "restoration_order": meta.get("checkpoint_restoration", {}).get(
            "restoration_order"
        ),
        "checkpoint_sidecar_path": str(
            checkpoint.parent / "checkpoint_provenance.json"
        ),
        "checkpoint_sidecar_sha256": sha256_file(
            checkpoint.parent / "checkpoint_provenance.json"
        ),
        "checkpoint_sidecar_passed": checkpoint_sidecar.get("passed") is True,
        "numeric_training_logs_reused": True,
        "e3_text_eval_regenerated": True,
        "e3_multimodal_eval_regenerated": True,
        "provenance_note_key": "text_eval_note",
        "forbidden_stale_key_absent": "frozen_lm_text_eval_note" not in text_eval_provenance,
        "text_perplexity": text_eval.get("perplexity"),
        "text_accuracy": text_eval.get("next_token_accuracy"),
        "conditional_image_to_text_r_at_1": retrieval.get("conditional_image_to_text_r_at_1"),
        "conditional_speech_to_text_r_at_1": retrieval.get("conditional_speech_to_text_r_at_1"),
    }
    save_json(out_dir / "e3_eval_refresh_provenance.json", refresh)
    cleanup(wrapper, wrapper.lm, vision_model, speech_model)
    print(json.dumps(refresh, sort_keys=True))


if __name__ == "__main__":
    main()
