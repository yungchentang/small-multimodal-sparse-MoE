"""Real-data OLMoE E0-E6 runs for ACDL Project 18.

This runner consumes manifests produced by ``datasets.build_real_subset`` and runs
real-data evidence jobs on OLMoE-1B-7B with Top-2 capacity routing, CLIP/Whisper
projection+concatenation, text/image/speech training, evaluation, and routing logs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hf_sources import load_pretrained, resolve_model
from scripts.development_split_provenance import (
    parse_jsonl_snapshot,
    resolve_speech_audio_path,
    secure_speech_audio_snapshot,
    speech_partition_record,
    speech_source_row_identity,
    verify_builder_provenance,
    verify_speech_audio_rows,
    verify_speech_partition_derivation,
)
from model.olmoe_adapter import OLMoEMultimodalPrefixWrapper
from training.olmoe_required_runs import (
    append_jsonl,
    base_model_identity,
    calibrate_gamma,
    capture_router_dispatch,
    cleanup,
    configure_selected_full_expert_training,
    cuda_metrics,
    dynamic_expert_bias_metrics,
    dynamic_expert_bias_state_dict,
    load_dynamic_expert_bias_state,
    load_encoders,
    load_model,
    modality_router_metrics,
    router_metrics,
    save_json,
    selected_expert_anchor_loss,
    selected_expert_update_capability,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


FORBIDDEN_DEVELOPMENT_EVIDENCE_TERMS = ("sealed", "synthetic")
STAGE_A_PROVENANCE_POLICY = "development_only_stage_a_multimodal_initialization"
SPEECH_BEHAVIOR_TEACHER_PATH = (
    "frozen_olmoe_text_only_teacher_forcing_on_prompt_and_previous_transcript_tokens"
)
SPEECH_BEHAVIOR_STUDENT_PATH = (
    "shared_olmoe_audio_prefix_plus_same_teacher_forced_text_sequence"
)
SPEECH_BEHAVIOR_ALIGNMENT = "causal_shifted_supervised_transcript_label_positions_only"
SPEECH_SHARED_CONTRASTIVE_TEACHER_PATH = (
    "frozen_shared_olmoe_transcript_only_final_hidden_full_train_bank"
)
SPEECH_SHARED_CONTRASTIVE_STUDENT_PATH = (
    "shared_olmoe_audio_prefix_plus_same_teacher_forced_text_final_hidden"
)
SPEECH_SHARED_CONTRASTIVE_STUDENT_POOLING = (
    "attention_masked_mean_over_prompt_and_transcript_text_tokens"
)
SPEECH_SHARED_CONTRASTIVE_TEACHER_POOLING = (
    "attention_masked_mean_over_transcript_only_tokens"
)
SPEECH_SHARED_CONTRASTIVE_NORMALIZATION = "l2"
SPEECH_SHARED_CONTRASTIVE_OBJECTIVE = (
    "student_to_full_transcript_teacher_train_bank_infonce"
)
SPEECH_SHARED_CONTRASTIVE_ROW_IDENTITY = (
    "source_dataset_plus_utterance_id_else_row_id"
)
SPEECH_SHARED_CONTRASTIVE_POSITIVE_SELECTION = (
    "audio_train_row_index_from_modality_cursor"
)
SPEECH_SHARED_CONTRASTIVE_DUPLICATE_POLICY = (
    "exclude_same_row_identity_or_normalized_transcript_sha256_except_selected_positive"
)
STRICT_SPEECH_FEATURE_CACHE_POLICY = "strict_recompute_no_persistent_cache"
DIAGNOSTIC_SPEECH_FEATURE_CACHE_POLICY = (
    "diagnostic_persistent_cache_not_strict_evidence"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_content_digest(tensor: torch.Tensor) -> Dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    descriptor = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "nbytes": len(raw),
        "content_sha256": hashlib.sha256(raw).hexdigest(),
    }
    descriptor["payload_sha256"] = hashlib.sha256(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return descriptor


def reject_forbidden_development_path(value: str | Path, label: str) -> None:
    lowered = str(value).lower()
    for term in FORBIDDEN_DEVELOPMENT_EVIDENCE_TERMS:
        if term in lowered:
            raise ValueError(f"{label} contains forbidden {term!r} evidence path")


def _is_full_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def validate_runtime_sample_rate(args) -> int:
    raw_sample_rate = args.sample_rate
    if isinstance(raw_sample_rate, bool) or not isinstance(raw_sample_rate, int):
        raise ValueError("runtime sample rate must be an integer")
    if (
        getattr(args, "development_split_manifest", "")
        and raw_sample_rate != 16000
    ):
        raise ValueError(
            "strict development split manifest requires --sample-rate=16000"
        )
    return raw_sample_rate


def build_stage_a_run_provenance(args, checkpoint_completed_step: int) -> Dict[str, Any]:
    source_commit_sha = os.environ.get("SOURCE_COMMIT_SHA")
    runai_job_name = os.environ.get("RUNAI_JOB_NAME")
    runai_project = os.environ.get("RUNAI_PROJECT")
    if not _is_full_hex(source_commit_sha, 40):
        raise ValueError(
            "Stage A checkpoint provenance requires a full 40-character "
            "SOURCE_COMMIT_SHA"
        )
    if not isinstance(runai_job_name, str) or not runai_job_name.strip():
        raise ValueError("Stage A checkpoint provenance requires RUNAI_JOB_NAME")
    if not isinstance(runai_project, str) or not runai_project.strip():
        raise ValueError("Stage A checkpoint provenance requires RUNAI_PROJECT")

    data_root = Path(str(args.data_dir)).resolve()
    output_root = Path(str(args.output_dir)).resolve()
    reject_forbidden_development_path(data_root, "Stage A data root")
    reject_forbidden_development_path(output_root, "Stage A output root")
    final_main_steps = int(args.final_steps)
    alignment_pretrain_steps = int(args.alignment_pretrain_steps)
    completed_step = int(checkpoint_completed_step)
    if final_main_steps <= 0 or alignment_pretrain_steps < 0 or completed_step <= 0:
        raise ValueError("Stage A checkpoint provenance has invalid training steps")
    speech_cache_policy = getattr(
        args,
        "speech_feature_cache_policy",
        DIAGNOSTIC_SPEECH_FEATURE_CACHE_POLICY,
    )
    if speech_cache_policy not in {
        STRICT_SPEECH_FEATURE_CACHE_POLICY,
        DIAGNOSTIC_SPEECH_FEATURE_CACHE_POLICY,
    }:
        raise ValueError(f"invalid speech feature cache policy: {speech_cache_policy}")
    provenance = {
        "source_commit_sha": source_commit_sha.lower(),
        "runai_job_name": runai_job_name,
        "runai_project": runai_project,
        "resolved_data_root": str(data_root),
        "resolved_output_root": str(output_root),
        "final_main_steps": final_main_steps,
        "alignment_pretrain_steps": alignment_pretrain_steps,
        "checkpoint_completed_step": completed_step,
        "policy": STAGE_A_PROVENANCE_POLICY,
        "sealed_evidence_used": False,
        "synthetic_evidence_used": False,
        "runtime_sample_rate": validate_runtime_sample_rate(args),
        "speech_feature_cache_policy": speech_cache_policy,
        "speech_shared_contrastive": speech_shared_contrastive_provenance(args),
    }
    split_manifest_value = str(
        getattr(args, "development_split_manifest", "") or ""
    ).strip()
    if split_manifest_value:
        split_manifest = Path(split_manifest_value)
        reject_forbidden_development_path(
            split_manifest, "Stage A development split manifest"
        )
        if split_manifest.is_symlink() or not split_manifest.is_file():
            raise ValueError(
                "Stage A development split manifest must be a regular file"
            )
        split_sha = getattr(args, "development_split_manifest_sha256", None)
        if not _is_full_hex(split_sha, 64):
            raise ValueError("Stage A run provenance requires exact split manifest SHA")
        if sha256_file(split_manifest.resolve()) != str(split_sha).lower():
            raise ValueError("Stage A run provenance split manifest SHA mismatch")
        provenance["development_split_manifest_sha256"] = str(split_sha).lower()
    return provenance


def validate_expert_selection_request(args) -> Dict[str, Any]:
    """Validate CLI-level selected-expert policy before expensive model work."""
    mode = str(getattr(args, "expert_update_mode", "full"))
    capability = selected_expert_update_capability(mode)
    selection_path = str(getattr(args, "expert_selection_json", "") or "")
    if not selection_path:
        return {**capability, "selected_expert_training": False}
    reject_forbidden_development_path(selection_path, "expert selection JSON")
    if bool(getattr(args, "train_experts", False)):
        raise ValueError("--expert-selection-json cannot be combined with --train-experts")
    allow_router_tuning = bool(
        getattr(args, "allow_selected_expert_router_tuning", False)
    )
    if bool(getattr(args, "train_router_gates", False)) and not allow_router_tuning:
        raise ValueError("selected-expert A3 keeps router gates frozen")
    if bool(getattr(args, "train_lm_head", False)):
        raise ValueError("selected-expert A3 keeps LM head/embeddings frozen")
    learning_rate = float(getattr(args, "expert_learning_rate", 0.0))
    if not 0.0 < learning_rate <= 1e-4:
        raise ValueError("selected-expert learning rate must be in (0, 1e-4]")
    anchor_coefficient = float(getattr(args, "expert_anchor_coefficient", 0.0))
    if anchor_coefficient <= 0.0:
        raise ValueError("selected-expert weight anchoring requires a positive coefficient")
    return {
        **capability,
        "selected_expert_training": True,
        "selection_path": selection_path,
        "selection_method": str(getattr(args, "expert_selection_method", "ESFT-Gate")),
        "expert_learning_rate": learning_rate,
        "weight_anchor_coefficient": anchor_coefficient,
        "router_tuning_explicitly_enabled": allow_router_tuning,
    }


def validate_speech_behavior_kl_request(args) -> None:
    coefficient = float(getattr(args, "speech_behavior_kl_coef", 0.0))
    temperature = float(getattr(args, "speech_behavior_kl_temperature", 1.0))
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("speech behavior KL coefficient must be finite and non-negative")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("speech behavior KL temperature must be finite and positive")
    if coefficient == 0.0:
        return
    if any(
        bool(getattr(args, field, False))
        for field in ("train_router_gates", "train_experts", "train_lm_head")
    ):
        raise ValueError("speech behavior KD requires frozen router/expert/LM parameters")
    if str(getattr(args, "expert_selection_json", "") or ""):
        raise ValueError("speech behavior KD cannot update selected experts")
    if float(getattr(args, "dynamic_expert_bias_lr", 0.0)) != 0.0:
        raise ValueError("speech behavior KD requires no dynamic router-bias updates")


def speech_shared_contrastive_provenance(args) -> Dict[str, Any]:
    coefficient = float(getattr(args, "speech_shared_contrastive_coef", 0.0))
    return {
        "enabled": coefficient > 0.0,
        "coefficient": coefficient,
        "temperature": float(
            getattr(args, "speech_shared_contrastive_temperature", 0.07)
        ),
        "student_path": SPEECH_SHARED_CONTRASTIVE_STUDENT_PATH,
        "teacher_path": SPEECH_SHARED_CONTRASTIVE_TEACHER_PATH,
        "student_pooling": SPEECH_SHARED_CONTRASTIVE_STUDENT_POOLING,
        "teacher_pooling": SPEECH_SHARED_CONTRASTIVE_TEACHER_POOLING,
        "normalization": SPEECH_SHARED_CONTRASTIVE_NORMALIZATION,
        "objective": SPEECH_SHARED_CONTRASTIVE_OBJECTIVE,
        "teacher_gradient_policy": "torch_no_grad",
        "teacher_bank_partition_policy": "speech_train_rows_only",
        "teacher_bank_order": "audio_train_row_order",
        "positive_selection": SPEECH_SHARED_CONTRASTIVE_POSITIVE_SELECTION,
        "row_identity_semantics": SPEECH_SHARED_CONTRASTIVE_ROW_IDENTITY,
        "duplicate_exclusion_policy": SPEECH_SHARED_CONTRASTIVE_DUPLICATE_POLICY,
        "sealed_evidence_used": False,
        "dev_partition_used": False,
        "eval_partition_used": False,
    }


def validate_speech_shared_contrastive_request(args) -> None:
    coefficient = float(getattr(args, "speech_shared_contrastive_coef", 0.0))
    temperature = float(
        getattr(args, "speech_shared_contrastive_temperature", 0.07)
    )
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError(
            "speech shared contrastive coefficient must be finite and non-negative"
        )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "speech shared contrastive temperature must be finite and positive"
        )
    if coefficient == 0.0:
        return
    if not str(getattr(args, "development_split_manifest", "") or ""):
        raise ValueError(
            "speech shared contrastive loss requires a strict development split manifest"
        )
    speech_teacher_bank_batch_size(args)
    if any(
        bool(getattr(args, field, False))
        for field in ("train_router_gates", "train_experts", "train_lm_head")
    ):
        raise ValueError(
            "speech shared contrastive loss requires frozen router/expert/LM parameters"
        )
    if str(getattr(args, "expert_selection_json", "") or ""):
        raise ValueError("speech shared contrastive loss cannot update selected experts")
    if float(getattr(args, "dynamic_expert_bias_lr", 0.0)) != 0.0:
        raise ValueError(
            "speech shared contrastive loss requires no dynamic router-bias updates"
        )


def validate_speech_shared_split_binding(
    provenance: Optional[Dict[str, Any]], expected_rows: int
) -> Dict[str, Any]:
    if not isinstance(provenance, dict):
        raise ValueError(
            "speech shared contrastive loss requires verified development split provenance"
        )
    if (
        provenance.get("policy")
        != "manifest_train_and_dev_only_reserved_eval_split_file_unread"
        or provenance.get("strict_manifest_verified") is not True
        or provenance.get("reserved_files_opened") is not False
        or provenance.get("sealed_evidence_used") is not False
        or provenance.get("synthetic_evidence_used") is not False
        or provenance.get("manifest_hash_and_parse_same_bytes") is not True
    ):
        raise ValueError(
            "speech shared contrastive loss rejects unverified or legacy split provenance"
        )
    manifest_path = Path(str(provenance.get("manifest_path", "")))
    manifest_sha256 = str(provenance.get("manifest_sha256", "")).lower()
    source_commit_sha = str(provenance.get("source_commit_sha", "")).lower()
    expected_trusted_digest = str(
        provenance.get("expected_speech_source_sha256", "")
    ).lower()
    if (
        provenance.get("trusted_digest_verified") is not True
        or not _is_full_hex(expected_trusted_digest, 64)
    ):
        raise ValueError(
            "speech shared contrastive loss requires an externally trusted source digest"
        )
    if (
        not manifest_path.is_absolute()
        or not _is_full_hex(manifest_sha256, 64)
        or not _is_full_hex(source_commit_sha, 40)
    ):
        raise ValueError(
            "speech shared contrastive split manifest binding is incomplete"
        )
    builder = provenance.get("builder")
    if not isinstance(builder, dict):
        raise ValueError(
            "speech shared contrastive split provenance has no verified builder"
        )
    builder_path = Path(str(builder.get("path", "")))
    builder_sha256 = str(builder.get("sha256", "")).lower()
    if (
        not builder_path.is_absolute()
        or not _is_full_hex(builder_sha256, 64)
        or str(builder.get("source_commit_sha", "")).lower() != source_commit_sha
        or builder.get("source_commit_exists") is not True
        or builder.get("source_matches_commit") is not True
        or builder.get("current_bytes_match_commit") is not True
        or builder.get("command") != "python scripts/materialize_eval_splits.py"
    ):
        raise ValueError(
            "speech shared contrastive split builder provenance is incomplete"
        )
    reject_forbidden_development_path(
        manifest_path, "speech shared contrastive split manifest"
    )
    files = provenance.get("files")
    if not isinstance(files, dict):
        raise ValueError(
            "speech shared contrastive split provenance has no files"
        )
    speech_train = files.get("speech_train")
    if not isinstance(speech_train, dict):
        raise ValueError(
            "speech shared contrastive split provenance has no speech_train"
        )
    speech_train_path = Path(str(speech_train.get("path", "")))
    speech_train_sha256 = str(speech_train.get("sha256", "")).lower()
    if (
        not speech_train_path.is_absolute()
        or not _is_full_hex(speech_train_sha256, 64)
        or speech_train.get("rows") != int(expected_rows)
        or speech_train.get("read_status") != "single_snapshot_for_hash_and_rows"
        or speech_train.get("content_opened") is not True
        or speech_train.get("sha256_verified_this_run") is not True
        or speech_train.get("hash_and_rows_same_bytes") is not True
    ):
        raise ValueError(
            "speech shared contrastive teacher bank is not bound to verified speech_train"
        )
    reject_forbidden_development_path(
        speech_train_path, "speech shared contrastive speech_train file"
    )
    source_files = provenance.get("source_files")
    speech_source = (
        source_files.get("speech") if isinstance(source_files, dict) else None
    )
    expected_source_rows = sum(
        DEVELOPMENT_SPLIT_COUNTS[f"speech_{split}"]
        for split in ("train", "dev", "eval")
    )
    if not isinstance(speech_source, dict):
        raise ValueError(
            "speech shared contrastive split provenance has no speech source file"
        )
    speech_source_path = Path(str(speech_source.get("path", "")))
    speech_source_sha256 = str(speech_source.get("sha256", "")).lower()
    if (
        not speech_source_path.is_absolute()
        or not _is_full_hex(speech_source_sha256, 64)
        or speech_source.get("rows") != expected_source_rows
        or speech_source.get("read_status")
        != "single_snapshot_for_integrity_and_partition_verification"
        or speech_source.get("content_opened") is not True
        or speech_source.get("sha256_verified_this_run") is not True
        or speech_source.get("rows_verified_this_run") is not True
        or speech_source.get("content_used_for_partition_verification") is not True
        or speech_source.get("content_used_for_training") is not False
        or speech_source.get("hash_and_rows_same_bytes") is not True
        or speech_source.get("derivation_verified") is not True
        or speech_source.get("reserved_eval_split_file_opened") is not False
        or speech_source.get("raw_source_eval_rows_read_for_partition_verification") is not True
        or speech_source.get("audio_bytes_verified_this_run") is not True
        or speech_source.get("audio_rows_verified") != expected_source_rows
        or not _is_full_hex(speech_source.get("audio_row_binding_root_sha256"), 64)
        or speech_source_sha256 != expected_trusted_digest
        or not isinstance(speech_source.get("partition_commitments"), dict)
    ):
        raise ValueError(
            "speech shared contrastive teacher bank is not bound to verified speech source"
        )
    reject_forbidden_development_path(
        speech_source_path, "speech shared contrastive speech source file"
    )
    for reserved_name in DEVELOPMENT_RESERVED_FILES:
        reserved = files.get(reserved_name)
        if (
            not isinstance(reserved, dict)
            or reserved.get("read_status") != "reserved_unread"
            or reserved.get("content_opened") is not False
            or reserved.get("sha256_verified_this_run") is not False
        ):
            raise ValueError(
                "speech shared contrastive split provenance does not prove reserved eval split file unread"
            )
    return {
        "binding_verified": True,
        "development_split_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "source_commit_sha": source_commit_sha,
            "builder": dict(builder),
            "expected_speech_source_sha256": expected_trusted_digest,
            "trusted_digest_verified": True,
        },
        "speech_train_file": {
            "path": str(speech_train_path),
            "sha256": speech_train_sha256,
            "rows": int(speech_train["rows"]),
        },
        "speech_source_file": {
            "path": str(speech_source_path),
            "sha256": speech_source_sha256,
            "rows": int(speech_source["rows"]),
            "audio_bytes_verified_this_run": True,
            "audio_rows_verified": expected_source_rows,
            "audio_row_binding_root_sha256": speech_source["audio_row_binding_root_sha256"],
            "partition_commitments": dict(
                speech_source["partition_commitments"]
            ),
            "expected_speech_source_sha256": expected_trusted_digest,
            "trusted_digest_verified": True,
        },
        "reserved_eval_split_file_unread": True,
        "raw_source_eval_rows_read_for_partition_verification": True,
        "sealed_evidence_used": False,
        "dev_partition_used": False,
        "eval_partition_used": False,
    }


def load_prefix_expert_selection(
    path: str | Path,
    method: str,
    num_layers: int,
    num_experts: int,
    expected_base_model: str,
) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    """Load a development-only prefix ESFT artifact and fail closed on drift."""
    selection_path = Path(path)
    reject_forbidden_development_path(selection_path, "expert selection JSON")
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expert selection JSON must be an object")
    if payload.get("artifact_type") != "development_moe_reconstruction_and_esft_selection":
        raise ValueError("expert selection artifact_type is not the development diagnostic schema")
    if payload.get("development_only") is not True:
        raise ValueError("expert selection artifact must declare development_only=true")
    if payload.get("sealed_evidence_used") is not False:
        raise ValueError("expert selection artifact does not prove sealed_evidence_used=false")
    if payload.get("synthetic_evidence_used") is not False:
        raise ValueError("expert selection artifact does not prove synthetic_evidence_used=false")
    model_meta = payload.get("model", {})
    if model_meta.get("base_model") != expected_base_model:
        raise ValueError("expert selection base model does not match the training base model")

    selection = payload.get("esft_selection")
    if not isinstance(selection, dict):
        raise ValueError("expert selection artifact is missing esft_selection")
    if selection.get("selection_scope") != "development_train_image_audio_prefix_only":
        raise ValueError("expert selection scope is not train-only image/audio prefix")
    accounting = selection.get("routing_accounting", {})
    expected_assignments = int(accounting.get("expected_assignments_tokens_x_layers_x_k", -1))
    observed_assignments = int(accounting.get("observed_assignments", -2))
    if (
        accounting.get("conservation_ok") is not True
        or expected_assignments <= 0
        or expected_assignments != observed_assignments
    ):
        raise ValueError("expert selection routing accounting does not conserve tokens x layers x K")

    routing_provenance = payload.get("provenance", {}).get("routing", {})
    if routing_provenance.get("policy") != "development_only_real_train":
        raise ValueError("expert selection routing provenance is not development-only real train")
    if routing_provenance.get("splits") != ["train"]:
        raise ValueError("expert selection must use train routing only")
    if routing_provenance.get("sealed_evidence_used") is not False:
        raise ValueError("routing provenance does not reject sealed evidence")
    if routing_provenance.get("synthetic_evidence_used") is not False:
        raise ValueError("routing provenance does not reject synthetic evidence")
    source_files = routing_provenance.get("source_files", [])
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("expert selection routing provenance has no source file fingerprints")
    source_paths = list(routing_provenance.get("source_paths", []))
    source_paths.extend(
        item.get("path", "")
        for item in source_files
        if isinstance(item, dict)
    )
    if not source_paths:
        raise ValueError("expert selection routing provenance has no source paths")
    for source_path in source_paths:
        reject_forbidden_development_path(source_path, "routing provenance source")
    for item in source_files:
        if not isinstance(item, dict):
            raise ValueError("routing provenance source file fingerprint must be an object")
        digest = str(item.get("sha256", "")).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("routing provenance source file has an invalid sha256")

    method_rows = selection.get("methods", {}).get(method)
    if not isinstance(method_rows, dict):
        raise ValueError(f"expert selection artifact has no method {method!r}")
    expected_layers = {str(index) for index in range(int(num_layers))}
    if set(method_rows) != expected_layers:
        raise ValueError("expert selection must cover every model layer exactly once")
    declared_count = int(selection.get("selected_experts_per_layer", 0))
    selected: Dict[int, List[int]] = {}
    for layer_key in sorted(method_rows, key=int):
        row = method_rows[layer_key]
        if row.get("splits") != ["train"]:
            raise ValueError(f"selection layer {layer_key} is not based on train-only routing")
        if set(row.get("modalities", [])) != {"image_prefix", "audio_prefix"}:
            raise ValueError(f"selection layer {layer_key} is not based on image/audio prefix routing")
        if int(row.get("prefix_tokens", 0)) <= 0 or int(row.get("assignments", 0)) <= 0:
            raise ValueError(f"selection layer {layer_key} has no prefix routing support")
        expert_scores = row.get("expert_scores")
        if not isinstance(expert_scores, list) or len(expert_scores) != int(num_experts):
            raise ValueError(f"selection layer {layer_key} has incomplete expert scores")
        score_by_id: Dict[int, Dict[str, Any]] = {}
        for score in expert_scores:
            if not isinstance(score, dict):
                raise ValueError(f"selection layer {layer_key} has a malformed expert score")
            expert_id = int(score.get("expert_id", -1))
            gate_score = float(score.get("gate_score_sum", float("nan")))
            token_count = int(score.get("token_count", -1))
            token_frequency = float(score.get("token_frequency", float("nan")))
            if (
                expert_id < 0
                or expert_id >= int(num_experts)
                or not math.isfinite(gate_score)
                or gate_score < 0.0
                or token_count < 0
                or not math.isfinite(token_frequency)
                or not 0.0 <= token_frequency <= 1.0
            ):
                raise ValueError(f"selection layer {layer_key} has an invalid expert score")
            score_by_id[expert_id] = score
        if set(score_by_id) != set(range(int(num_experts))):
            raise ValueError(f"selection layer {layer_key} expert scores do not cover every expert")
        raw_selected_ids = [int(value) for value in row.get("selected_expert_ids", [])]
        expert_ids = sorted(set(raw_selected_ids))
        if len(expert_ids) != declared_count or not expert_ids:
            raise ValueError(f"selection layer {layer_key} has an inconsistent selected expert count")
        if len(expert_ids) >= int(num_experts):
            raise ValueError("selected-expert training cannot select all experts")
        if expert_ids[0] < 0 or expert_ids[-1] >= int(num_experts):
            raise ValueError(f"selection layer {layer_key} has out-of-range expert IDs")
        ranking_field = "gate_score_sum" if method == "ESFT-Gate" else "token_count"
        expected_ranking = sorted(
            range(int(num_experts)),
            key=lambda expert_id: (-float(score_by_id[expert_id][ranking_field]), expert_id),
        )[:declared_count]
        if raw_selected_ids != expected_ranking:
            raise ValueError(f"selection layer {layer_key} IDs do not match deterministic {method} scores")
        selected[int(layer_key)] = expert_ids
    provenance = {
        "selection_json": str(selection_path),
        "selection_json_sha256": sha256_file(selection_path),
        "selection_method": method,
        "selection_scope": selection["selection_scope"],
        "selected_experts_per_layer": declared_count,
        "selected_expert_ids_by_layer": {str(key): value for key, value in selected.items()},
        "routing_accounting": accounting,
        "routing_policy": routing_provenance["policy"],
        "routing_splits": ["train"],
        "routing_modalities": ["audio_prefix", "image_prefix"],
        "routing_source_files": source_files,
        "sealed_evidence_used": False,
        "synthetic_evidence_used": False,
    }
    return selected, provenance


class CombinedOptimizer:
    """Apply bridge and selected-expert optimizers as one training-step unit."""

    def __init__(self, *optimizers: torch.optim.Optimizer) -> None:
        self.optimizers = list(optimizers)
        self.param_groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()


def resolve_media_path(value: Any, data_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    project_root = data_dir.parent.parent if data_dir.parent.name == "data" else Path.cwd()
    candidates: List[Path] = []
    if path.parts and path.parts[0] == "data":
        candidates.append(project_root / path)
    candidates.extend([data_dir / path, Path.cwd() / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def absolutize_media_paths(rows: Sequence[Dict[str, Any]], data_dir: Path) -> None:
    for row in rows:
        for key in ("image_path", "audio_path"):
            value = row.get(key)
            if value and not Path(str(value)).is_absolute():
                row[key] = str(resolve_media_path(value, data_dir))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(data_dir: Path) -> Dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing real-data manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = ["text_tasks.jsonl", "image_captions.jsonl", "speech_transcripts.jsonl", "text_blocks_train.jsonl", "text_blocks_eval.jsonl"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing real-data files under {data_dir}: {missing}")
    return manifest


DEVELOPMENT_SPLIT_COUNTS = {
    "image_train": 5000,
    "image_dev": 137,
    "image_eval": 113,
    "speech_train": 5000,
    "speech_dev": 137,
    "speech_eval": 113,
}
DEVELOPMENT_SELECTED_FILES = (
    "image_train",
    "image_dev",
    "speech_train",
    "speech_dev",
)
DEVELOPMENT_RESERVED_FILES = ("image_eval", "speech_eval")


def load_development_multimodal_partitions(
    manifest_value: str | Path,
    *,
    expected_source_commit_sha: str | None = None,
    expected_data_dir: str | Path | None = None,
    expected_speech_source_sha256: str | None = None,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """Load audited train/dev partitions while keeping the reserved eval split unused."""
    manifest_path = Path(manifest_value)
    reject_forbidden_development_path(manifest_path, "development split manifest")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("development split manifest must be a regular non-symlink file")
    manifest_path = manifest_path.resolve()
    manifest_payload = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("development split manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("development split manifest root must be an object")
    manifest_data_dir_value = manifest.get("data_dir")
    if not isinstance(manifest_data_dir_value, str):
        raise ValueError("development split manifest data_dir is missing")
    manifest_data_dir = Path(manifest_data_dir_value)
    if (
        not manifest_data_dir.is_absolute()
        or manifest_data_dir.is_symlink()
        or not manifest_data_dir.is_dir()
        or manifest_data_dir.resolve() != manifest_data_dir
    ):
        raise ValueError(
            "development split manifest data_dir must be canonical"
        )
    if expected_data_dir is not None:
        runtime_data_dir = Path(expected_data_dir)
        if (
            not runtime_data_dir.is_absolute()
            or runtime_data_dir.is_symlink()
            or not runtime_data_dir.is_dir()
            or runtime_data_dir.resolve() != runtime_data_dir
        ):
            raise ValueError("runtime data_dir must be a canonical directory")
        if manifest_data_dir != runtime_data_dir:
            raise ValueError(
                "development split manifest data_dir disagrees with runtime data_dir"
            )
    if (
        manifest.get("schema_version") != 3
        or manifest.get("real_subset") is not True
        or manifest.get("synthetic_evidence_used") is not False
        or manifest.get("sealed_data_used") is not False
    ):
        raise ValueError("development split manifest has invalid real-data provenance")
    verified_builder = verify_builder_provenance(
        manifest.get("builder"),
        expected_source_commit_sha=expected_source_commit_sha,
    )
    source_commit_sha = str(verified_builder["source_commit_sha"])
    policies = manifest.get("split_policy", {})
    if policies.get("image") != "seeded_exact_image_content_disjoint_v1":
        raise ValueError("development image split is not content-group-disjoint")
    if policies.get("speech") != "explicit_source_partition":
        raise ValueError("development speech split is not explicit-source-partitioned")
    if manifest.get("speech_group_key") != ["source_dataset", "speaker_id"]:
        raise ValueError("development speech group key is invalid")
    speech_overlaps = manifest.get("speech_group_overlap")
    if not isinstance(speech_overlaps, dict) or any(
        speech_overlaps.get(key) != []
        for key in ("train_dev", "train_eval", "dev_eval")
    ):
        raise ValueError("development speech overlap provenance is invalid")

    files = manifest.get("files")
    counts = manifest.get("counts")
    if not isinstance(files, dict) or not isinstance(counts, dict):
        raise ValueError("development split manifest is missing files/counts")
    if (
        manifest.get("dev_count") != DEVELOPMENT_SPLIT_COUNTS["image_dev"]
        or manifest.get("eval_count") != DEVELOPMENT_SPLIT_COUNTS["image_eval"]
    ):
        raise ValueError("development split manifest has invalid dev/eval counts")

    rows_by_name: Dict[str, List[Dict[str, Any]]] = {}
    file_provenance: Dict[str, Dict[str, Any]] = {}
    for name in (*DEVELOPMENT_SELECTED_FILES, *DEVELOPMENT_RESERVED_FILES):
        expected_count = DEVELOPMENT_SPLIT_COUNTS[name]
        record = files.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"development split manifest is missing files.{name}")
        path = Path(str(record.get("path", "")))
        expected_sha = str(record.get("sha256", "")).lower()
        declared_rows = record.get("rows", counts.get(name))
        if (
            not path.is_absolute()
            or not _is_full_hex(expected_sha, 64)
            or counts.get(name) != expected_count
            or declared_rows != expected_count
        ):
            raise ValueError(
                f"development files.{name} has invalid manifest metadata"
            )
        if name in DEVELOPMENT_RESERVED_FILES:
            file_provenance[name] = {
                "path": str(path),
                "sha256": expected_sha,
                "rows": expected_count,
                "read_status": "reserved_unread",
                "content_opened": False,
                "sha256_verified_this_run": False,
                "metadata_source": "strict_development_split_manifest",
            }
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"development files.{name} must be an absolute regular file"
            )
        path = path.resolve()
        payload = path.read_bytes()
        actual_sha = hashlib.sha256(payload).hexdigest()
        if expected_sha != actual_sha:
            raise ValueError(f"development files.{name} SHA256 mismatch")
        rows = parse_jsonl_snapshot(
            payload, f"development files.{name}"
        )
        if len(rows) != expected_count:
            raise ValueError(
                f"development files.{name} must contain exactly {expected_count} rows"
            )
        rows_by_name[name] = rows
        file_provenance[name] = {
            "path": str(path),
            "sha256": actual_sha,
            "rows": len(rows),
            "size_bytes": len(payload),
            "read_status": "single_snapshot_for_hash_and_rows",
            "content_opened": True,
            "sha256_verified_this_run": True,
            "hash_and_rows_same_bytes": True,
            "metadata_source": "file_content_and_strict_manifest",
        }

    speech_source_file = verify_speech_partition_derivation(
        manifest,
        expected_partition_rows={
            split: DEVELOPMENT_SPLIT_COUNTS[f"speech_{split}"]
            for split in ("train", "dev", "eval")
        },
        observed_partition_rows={
            split: rows_by_name[f"speech_{split}"]
            for split in ("train", "dev")
        },
    )
    selected_speech_rows = [
        *rows_by_name["speech_train"],
        *rows_by_name["speech_dev"],
    ]
    speech_audio_verification = verify_speech_audio_rows(
        selected_speech_rows,
        data_dir=manifest_data_dir,
    )
    trusted_digest_verified = False
    expected_trusted_digest = None
    if expected_speech_source_sha256 is not None:
        if (
            not isinstance(expected_speech_source_sha256, str)
            or not _is_full_hex(expected_speech_source_sha256, 64)
        ):
            raise ValueError(
                "development speech source requires an exact trusted runtime digest"
            )
        expected_trusted_digest = expected_speech_source_sha256.lower()
        if speech_source_file.get("sha256") != expected_trusted_digest:
            raise ValueError(
                "development speech source SHA256 disagrees with trusted runtime digest"
            )
        trusted_digest_verified = True

    image_groups = {
        split: {
            image_group_id(row)
            for row in rows_by_name[f"image_{split}"]
        }
        for split in ("train", "dev")
    }
    speech_groups = {
        split: {
            (str(row.get("source_dataset", "")), str(row.get("speaker_id", "")))
            for row in rows_by_name[f"speech_{split}"]
        }
        for split in ("train", "dev")
    }
    if image_groups["train"] & image_groups["dev"]:
        raise ValueError("development image content-group overlap between train and dev")
    if speech_groups["train"] & speech_groups["dev"]:
        raise ValueError("development speech speaker overlap between train and dev")
    partition_meta = manifest.get("image_content_partition", {})
    if not isinstance(partition_meta, dict):
        raise ValueError("development image content partition metadata is missing")
    selected_image_group_counts = {
        split: len(values) for split, values in image_groups.items()
    }
    declared_group_counts = partition_meta.get("group_counts", {})
    declared_image_overlaps = partition_meta.get("pairwise_group_overlaps", {})
    expected_row_counts = {
        split: DEVELOPMENT_SPLIT_COUNTS[f"image_{split}"]
        for split in ("train", "dev", "eval")
    }
    if (
        partition_meta.get("policy")
        != "seeded_exact_image_content_disjoint_v1"
        or partition_meta.get("group_key")
        != "content_sha256_of_decoded_resized_rgb_pixels"
        or isinstance(partition_meta.get("seed"), bool)
        or not isinstance(partition_meta.get("seed"), int)
        or partition_meta.get("row_counts") != expected_row_counts
        or partition_meta.get("pairwise_group_overlap_count") != 0
        or not isinstance(declared_group_counts, dict)
        or set(declared_group_counts) != {"train", "dev", "eval"}
        or any(
            declared_group_counts.get(split) != count
            for split, count in selected_image_group_counts.items()
        )
        or not isinstance(declared_group_counts.get("eval"), int)
        or declared_group_counts["eval"] <= 0
        or partition_meta.get("source_group_count")
        != sum(declared_group_counts.values())
        or not isinstance(declared_image_overlaps, dict)
        or any(
            declared_image_overlaps.get(key) != []
            for key in ("train_dev", "train_eval", "dev_eval")
        )
    ):
        raise ValueError("development image content partition metadata mismatch")
    provenance = {
        "policy": "manifest_train_and_dev_only_reserved_eval_split_file_unread",
        "strict_manifest_verified": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_size_bytes": len(manifest_payload),
        "manifest_hash_and_parse_same_bytes": True,
        "trusted_digest_verified": trusted_digest_verified,
        "expected_speech_source_sha256": expected_trusted_digest,
        "source_commit_sha": source_commit_sha,
        "builder": verified_builder,
        "data_dir": str(manifest_data_dir),
        "source_files": {"speech": speech_source_file},
        "speech_audio_verification": speech_audio_verification,
        "files": file_provenance,
        "image_group_counts": {
            **selected_image_group_counts,
            "eval": declared_group_counts["eval"],
        },
        "speech_group_counts": {
            "train": len(speech_groups["train"]),
            "dev": len(speech_groups["dev"]),
            "eval": None,
        },
        "selection_splits": ["train", "dev"],
        "reserved_unused_split": "eval",
        "reserved_files_opened": False,
        "reserved_unused_counts": {
            "image": DEVELOPMENT_SPLIT_COUNTS["image_eval"],
            "speech": DEVELOPMENT_SPLIT_COUNTS["speech_eval"],
        },
        "sealed_evidence_used": False,
        "synthetic_evidence_used": False,
    }
    return (
        rows_by_name["image_train"],
        rows_by_name["image_dev"],
        rows_by_name["speech_train"],
        rows_by_name["speech_dev"],
        provenance,
    )


def split_tail(rows: Sequence[Dict[str, Any]], eval_count: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(rows)
    if eval_count <= 0 or len(rows) <= eval_count:
        return rows, rows
    return rows[:-eval_count], rows[-eval_count:]


def sample_cycle(rows: Sequence[Dict[str, Any]], start: int, batch_size: int) -> List[Dict[str, Any]]:
    if not rows:
        raise RuntimeError("Cannot sample from an empty row list")
    return [rows[(start + offset) % len(rows)] for offset in range(batch_size)]


def sample_cycle_indices(rows: Sequence[Dict[str, Any]], start: int, batch_size: int) -> Tuple[List[int], List[Dict[str, Any]]]:
    if not rows:
        raise RuntimeError("Cannot sample from an empty row list")
    indices = [(start + offset) % len(rows) for offset in range(batch_size)]
    return indices, [rows[idx] for idx in indices]


def sample_next_modality_batch(
    rows: Sequence[Dict[str, Any]],
    modality: str,
    cursors: Dict[str, int],
    batch_size: int,
) -> Tuple[List[int], List[Dict[str, Any]], Dict[str, Any]]:
    """Sample a contiguous cyclic batch using an independent modality cursor."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    start = int(cursors.get(modality, 0))
    indices, batch_rows = sample_cycle_indices(rows, start, batch_size)
    end = start + int(batch_size)
    cursors[modality] = end
    dataset_size = len(rows)
    return indices, batch_rows, {
        "data_cursor_policy": "independent_per_modality_contiguous_cycle",
        "data_cursor_modality": modality,
        "data_cursor_start": start,
        "data_cursor_end_exclusive": end,
        "data_cursor_start_index": start % dataset_size,
        "data_cursor_end_index_exclusive": end % dataset_size,
        "data_dataset_size": dataset_size,
        "data_rows_seen_for_modality": end,
        "data_unique_rows_covered_for_modality": min(dataset_size, end),
        "data_coverage_ratio_for_modality": float(min(dataset_size, end) / dataset_size),
        "data_completed_dataset_passes_for_modality": end // dataset_size,
    }


FINAL_ROUTING_AGGREGATE_FIELDS = {
    "gate_entropy_mean": "gate_entropy_mean_assignment_weighted",
    "inactive_expert_ratio_mean": "inactive_expert_ratio_mean_assignment_weighted",
    "capacity_overflow_ratio_mean": "capacity_overflow_ratio_mean_assignment_weighted",
    "dynamic_expert_bias_inactive_proxy": "dynamic_expert_bias_inactive_proxy_assignment_weighted",
    "dynamic_expert_bias_overflow_proxy": "dynamic_expert_bias_overflow_proxy_assignment_weighted",
}


def assignment_weighted_final_cycle_summary(
    rows: Sequence[Dict[str, Any]],
    modality_cycle: Sequence[str],
) -> Dict[str, Any]:
    """Aggregate routing summaries over the trailing complete modality cycle."""
    cycle = [str(modality) for modality in modality_cycle]
    if not cycle:
        raise ValueError("modality_cycle must not be empty")
    if len(rows) < len(cycle):
        raise ValueError(
            "final routing aggregation requires at least one complete modality cycle"
        )
    window = list(rows[-len(cycle) :])
    configured_modalities = set(cycle)
    present_modalities = {str(row.get("modality", "")) for row in window}
    missing_modalities = sorted(configured_modalities - present_modalities)
    if missing_modalities:
        raise ValueError(
            "final routing aggregation window is missing configured modalities: "
            f"{missing_modalities}"
        )

    def aggregate(selected_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        assignment_total = sum(
            int(row.get("routing_attempted_assignments_total", 0))
            for row in selected_rows
        )
        if assignment_total <= 0:
            raise ValueError(
                "final routing aggregation requires positive attempted-assignment weights"
            )
        result: Dict[str, Any] = {
            "row_count": len(selected_rows),
            "attempted_assignment_weight_total": assignment_total,
        }
        for source_name, aggregate_name in FINAL_ROUTING_AGGREGATE_FIELDS.items():
            weighted_rows = [
                (float(row[source_name]), int(row["routing_attempted_assignments_total"]))
                for row in selected_rows
                if row.get(source_name) is not None
                and int(row.get("routing_attempted_assignments_total", 0)) > 0
            ]
            metric_weight = sum(weight for _, weight in weighted_rows)
            result[aggregate_name] = (
                float(sum(value * weight for value, weight in weighted_rows) / metric_weight)
                if metric_weight > 0
                else None
            )
            result[f"{aggregate_name}_attempted_assignment_denominator"] = metric_weight
        return result

    return {
        "window_policy": "trailing_complete_modality_cycle",
        "window_row_count": len(window),
        "window_start_step": int(window[0].get("step", len(rows) - len(window) + 1)),
        "window_end_step": int(window[-1].get("step", len(rows))),
        "configured_modality_cycle": cycle,
        "modalities_present": sorted(present_modalities),
        "assignment_weight_field": "routing_attempted_assignments_total",
        "overall": aggregate(window),
        "by_modality": {
            modality: aggregate(
                [row for row in window if str(row.get("modality")) == modality]
            )
            for modality in sorted(configured_modalities)
        },
    }


def tensorize_blocks(blocks: Sequence[Dict[str, Any]], device: torch.device, max_length: int, pad_token_id: int) -> Dict[str, torch.Tensor]:
    ids_list: List[List[int]] = []
    for block in blocks:
        ids = [int(x) for x in block.get("input_ids", [])][:max_length]
        if not ids:
            ids = [pad_token_id]
        ids_list.append(ids)
    seq_len = max(len(ids) for ids in ids_list)
    input_ids = torch.full((len(ids_list), seq_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for idx, ids in enumerate(ids_list):
        input_ids[idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[idx, : len(ids)] = 1
        labels[idx, : len(ids)] = input_ids[idx, : len(ids)]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_prompt_targets(tokenizer, prompts: Sequence[str], targets: Sequence[str], device: torch.device, max_length: int) -> Dict[str, torch.Tensor]:
    encoded_ids: List[List[int]] = []
    encoded_labels: List[List[int]] = []
    for prompt, target in zip(prompts, targets):
        prompt_ids = tokenizer(str(prompt), add_special_tokens=False).input_ids
        target_ids = tokenizer(" " + str(target).strip(), add_special_tokens=False).input_ids
        if not target_ids:
            target_ids = tokenizer(str(target).strip() or " ", add_special_tokens=False).input_ids
        if len(prompt_ids) + len(target_ids) > max_length:
            keep_prompt = max(0, max_length - len(target_ids))
            prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt else []
            target_ids = target_ids[:max_length]
        ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        encoded_ids.append(ids)
        encoded_labels.append(labels)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    seq_len = max(len(ids) for ids in encoded_ids)
    input_ids = torch.full((len(encoded_ids), seq_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for idx, (ids, row_labels) in enumerate(zip(encoded_ids, encoded_labels)):
        input_ids[idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[idx, : len(ids)] = 1
        labels[idx, : len(row_labels)] = torch.tensor(row_labels, dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def reduce_sequence(states: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if max_tokens <= 0 or states.shape[1] <= max_tokens:
        return states
    chunks = torch.tensor_split(states, max_tokens, dim=1)
    return torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)


def image_features_from_paths(image_processor, vision_model, image_paths: Sequence[str], device: torch.device, max_tokens: int) -> torch.Tensor:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    batch = image_processor(images=images, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        if hasattr(vision_model, "vision_model") and "pixel_values" in batch:
            outputs = vision_model.vision_model(pixel_values=batch["pixel_values"])
        else:
            outputs = vision_model(**batch)
    return reduce_sequence(outputs.last_hidden_state.detach(), max_tokens)


def audio_file_snapshot(
    audio_path: str, expected_sha256: Optional[str], *, require_expected: bool
) -> Tuple[bytes, str]:
    if expected_sha256 is None:
        if require_expected:
            raise ValueError("strict speech row requires exact audio_sha256")
    elif not _is_full_hex(expected_sha256, 64):
        raise ValueError("speech row audio_sha256 must be exact lowercase SHA256")
    payload = Path(audio_path).read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"speech audio SHA256 mismatch: {audio_path}")
    return payload, actual_sha256


def load_audio_file(
    audio_path: str,
    expected_sha256: Optional[str] = None,
    *,
    require_expected: bool = False,
) -> Tuple[np.ndarray, int]:
    payload, _ = audio_file_snapshot(
        audio_path, expected_sha256, require_expected=require_expected
    )

    return decode_audio_payload(payload)


def decode_audio_payload(payload: bytes) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(payload), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


def speech_encoder_module(speech_model):
    encoder = getattr(speech_model, "encoder", None)
    if encoder is None and hasattr(speech_model, "model"):
        encoder = getattr(speech_model.model, "encoder", None)
    if encoder is None:
        raise RuntimeError("Speech model does not expose an encoder")
    return encoder


def audio_features_from_paths(
    speech_processor,
    speech_model,
    audio_paths: Sequence[str],
    device: torch.device,
    expected_sr: int,
    max_tokens: int,
    max_seconds: float = 0.0,
    audio_sha256s: Optional[Sequence[Optional[str]]] = None,
    require_expected_sha256: bool = False,
    audio_payloads: Optional[Sequence[bytes]] = None,
) -> torch.Tensor:
    if audio_sha256s is not None and len(audio_sha256s) != len(audio_paths):
        raise ValueError("audio paths and SHA256 commitments have different lengths")
    if audio_payloads is not None and len(audio_payloads) != len(audio_paths):
        raise ValueError("audio paths and byte snapshots have different lengths")
    waveforms: List[np.ndarray] = []
    for index, audio_path in enumerate(audio_paths):
        expected_sha256 = audio_sha256s[index] if audio_sha256s is not None else None
        if audio_payloads is None:
            audio, sr = load_audio_file(
                audio_path,
                expected_sha256,
                require_expected=require_expected_sha256,
            )
        else:
            payload = audio_payloads[index]
            if expected_sha256 is None and require_expected_sha256:
                raise ValueError("strict speech row requires exact audio_sha256")
            if expected_sha256 is not None:
                actual_sha256 = hashlib.sha256(payload).hexdigest()
                if actual_sha256 != expected_sha256:
                    raise ValueError(f"speech audio SHA256 mismatch: {audio_path}")
            audio, sr = decode_audio_payload(payload)
        if sr != expected_sr:
            try:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=expected_sr)
            except Exception:
                duration = len(audio) / max(sr, 1)
                x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
                x_new = np.linspace(0, duration, num=max(1, int(duration * expected_sr)), endpoint=False)
                audio = np.interp(x_new, x_old, audio).astype(np.float32)
        if max_seconds > 0.0:
            audio = audio[: max(1, int(float(max_seconds) * expected_sr))]
        waveforms.append(audio.astype(np.float32))
    kwargs: Dict[str, Any] = {"sampling_rate": expected_sr, "return_tensors": "pt"}
    if hasattr(speech_processor, "n_samples"):
        kwargs.update({"padding": "max_length", "max_length": int(speech_processor.n_samples), "truncation": True})
    else:
        kwargs["padding"] = True
    batch = speech_processor(waveforms, **kwargs)
    batch = {key: value.to(device) for key, value in batch.items()}
    encoder_trainable = any(param.requires_grad for param in speech_encoder_module(speech_model).parameters())
    with torch.set_grad_enabled(encoder_trainable):
        if "input_features" in batch:
            outputs = speech_encoder_module(speech_model)(input_features=batch["input_features"])
        else:
            outputs = speech_model(**batch)
    states = outputs.last_hidden_state if encoder_trainable else outputs.last_hidden_state.detach()
    return reduce_sequence(states, max_tokens)


def lm_embeddings_are_tied(model) -> bool:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        return input_embeddings is output_embeddings
    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    return (
        input_embeddings is output_embeddings
        or (
            input_weight is not None
            and input_weight is output_weight
        )
    )


def _identity_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _component_config_state_token(component) -> Tuple[Any, ...]:
    if component is None:
        return (False, None, None)
    config_payload: Dict[str, Any] = {}
    to_dict = getattr(component, "to_dict", None)
    if callable(to_dict):
        config_payload["to_dict"] = to_dict()
    component_config = getattr(component, "config", None)
    config_to_dict = getattr(component_config, "to_dict", None)
    if callable(config_to_dict):
        config_payload["config"] = config_to_dict()
    elif isinstance(component_config, dict):
        config_payload["config"] = component_config
    init_kwargs = getattr(component, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        config_payload["init_kwargs"] = init_kwargs
    for name in (
        "name_or_path",
        "model_max_length",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
    ):
        if hasattr(component, name):
            config_payload[name] = getattr(component, name)
    special_tokens_map = getattr(component, "special_tokens_map", None)
    if isinstance(special_tokens_map, dict):
        config_payload["special_tokens_map"] = special_tokens_map
    return (
        True,
        f"{type(component).__module__}.{type(component).__qualname__}",
        _identity_sha256(config_payload),
    )


def _component_config_identity_from_state_token(
    state_token: Tuple[Any, ...],
) -> Dict[str, Any]:
    present, component_class, config_sha256 = state_token
    if not present:
        return {"present": False}
    return {
        "present": True,
        "class": component_class,
        "config_sha256": config_sha256,
    }


def speech_processor_state_token(speech_processor) -> Tuple[Tuple[Any, ...], ...]:
    feature_extractor = getattr(
        speech_processor, "feature_extractor", speech_processor
    )
    return (
        _component_config_state_token(speech_processor),
        _component_config_state_token(feature_extractor),
        _component_config_state_token(getattr(speech_processor, "tokenizer", None)),
    )


_ABSENT_SPEECH_PROCESSOR_STATE_TOKEN = (
    (False, None, None),
    (False, None, None),
    (False, None, None),
)


def _speech_processor_cache_identity_from_state_token(
    state_token: Tuple[Tuple[Any, ...], ...],
) -> Dict[str, Any]:
    return {
        name: _component_config_identity_from_state_token(component_token)
        for name, component_token in zip(
            ("processor", "feature_extractor", "tokenizer"), state_token
        )
    }


def speech_processor_cache_identity(speech_processor) -> Dict[str, Any]:
    return _speech_processor_cache_identity_from_state_token(
        speech_processor_state_token(speech_processor)
    )


def _ordered_encoder_tensors(speech_model):
    encoder = speech_encoder_module(speech_model)
    for kind, named_tensors in (
        ("parameter", encoder.named_parameters),
        ("buffer", encoder.named_buffers),
    ):
        try:
            iterator = named_tensors(remove_duplicate=False)
        except TypeError:
            iterator = named_tensors()
        for name, tensor in iterator:
            yield kind, name, tensor


def encoder_weights_cache_identity(speech_model) -> Dict[str, Any]:
    tensors: List[Dict[str, Any]] = []
    for kind, name, tensor in _ordered_encoder_tensors(speech_model):
        content = tensor_content_identity(tensor)
        tensors.append(
            {
                "kind": kind,
                "name": name,
                "shape": content["shape"],
                "dtype": content["dtype"],
                "content_sha256": content["sha256"],
            }
        )
    return {
        "ordered_parameter_buffer_tensors": tensors,
        "identity_sha256": _identity_sha256(tensors),
    }


_AUDIO_FEATURE_EXTRACTION_IMPLEMENTATION_IDENTITY: Optional[Dict[str, Any]] = None


def audio_feature_extraction_implementation_identity() -> Dict[str, Any]:
    global _AUDIO_FEATURE_EXTRACTION_IMPLEMENTATION_IDENTITY
    if _AUDIO_FEATURE_EXTRACTION_IMPLEMENTATION_IDENTITY is None:
        functions = (
            audio_file_snapshot,
            load_audio_file,
            decode_audio_payload,
            audio_features_from_paths,
            reduce_sequence,
            speech_encoder_module,
        )
        sources = [
            {
                "module": function.__module__,
                "qualname": function.__qualname__,
                "source_sha256": hashlib.sha256(
                    inspect.getsource(function).encode("utf-8")
                ).hexdigest(),
            }
            for function in functions
        ]
        package_versions = {}
        for package in ("librosa", "numpy", "soundfile", "torch", "transformers"):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = None
        implementation_payload = {
            "functions": sources,
            "package_versions": package_versions,
        }
        _AUDIO_FEATURE_EXTRACTION_IMPLEMENTATION_IDENTITY = {
            "semantics": (
                "ordered_python_source_digests_for_audio_decode_resample_processor_"
                "encoder_and_sequence_reduction_plus_runtime_package_versions"
            ),
            **implementation_payload,
            "identity_sha256": _identity_sha256(implementation_payload),
        }
    return _AUDIO_FEATURE_EXTRACTION_IMPLEMENTATION_IDENTITY


def _encoder_state_token(speech_model) -> Tuple[Any, ...]:
    tensors = []
    for kind, name, tensor in _ordered_encoder_tensors(speech_model):
        tensors.append(
            (
                kind,
                name,
                id(tensor),
                int(getattr(tensor, "_version", 0)),
                tuple(tensor.shape),
                str(tensor.dtype),
                bool(getattr(tensor, "requires_grad", False)),
            )
        )
    config = getattr(speech_model, "config", None)
    config_payload = (
        config.to_dict()
        if config is not None and hasattr(config, "to_dict")
        else vars(config) if config is not None else {}
    )
    return (_identity_sha256(config_payload), *tensors)


def encoder_cache_identity(
    speech_model,
    sample_rate: int,
    max_tokens: int,
    max_seconds: float,
    speech_processor=None,
    *,
    encoder_weights_identity: Optional[Dict[str, Any]] = None,
    processor_state_token: Optional[Tuple[Tuple[Any, ...], ...]] = None,
) -> str:
    config = getattr(speech_model, "config", None)
    if config is not None and hasattr(config, "to_dict"):
        config_payload = config.to_dict()
    else:
        config_payload = {
            key: value
            for key, value in vars(config).items()
            if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict))
        } if config is not None else {}
    revision = {
        "name_or_path": getattr(config, "_name_or_path", None),
        "commit_hash": getattr(config, "_commit_hash", None),
        "revision": getattr(config, "revision", None),
    }
    trainable = any(param.requires_grad for param in speech_encoder_module(speech_model).parameters())
    payload = {
        "max_seconds": float(max_seconds),
        "sample_rate": int(sample_rate),
        "max_tokens": int(max_tokens),
        "encoder_revision": revision,
        "encoder_config": config_payload,
        "encoder_weights": (
            encoder_weights_identity
            if encoder_weights_identity is not None
            else encoder_weights_cache_identity(speech_model)
        ),
        "speech_processor": (
            _speech_processor_cache_identity_from_state_token(processor_state_token)
            if processor_state_token is not None
            else speech_processor_cache_identity(speech_processor)
        ),
        "audio_feature_extraction_implementation": (
            audio_feature_extraction_implementation_identity()
        ),
        "encoder_trainable": bool(trainable),
    }
    return _identity_sha256(payload)


def _stable_config_payload(value: Any) -> Any:
    return json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    )


def image_encoder_cache_provenance(
    image_processor,
    vision_model,
    max_tokens: int,
    identity_context: Mapping[str, Any],
) -> Dict[str, Any]:
    config = getattr(vision_model, "config", None)
    if config is not None and hasattr(config, "to_dict"):
        model_config = config.to_dict()
    elif config is not None:
        model_config = vars(config)
    else:
        model_config = {}
    if hasattr(image_processor, "to_dict"):
        preprocess_config = image_processor.to_dict()
    else:
        nested = getattr(image_processor, "image_processor", None)
        if nested is not None and hasattr(nested, "to_dict"):
            preprocess_config = nested.to_dict()
        else:
            preprocess_config = {
                key: value
                for key, value in (
                    vars(image_processor).items()
                    if hasattr(image_processor, "__dict__")
                    else ()
                )
                if not key.startswith("_")
            }
    revision = {
        "model_name_or_path": getattr(vision_model, "name_or_path", None),
        "config_name_or_path": getattr(config, "_name_or_path", None),
        "commit_hash": getattr(config, "_commit_hash", None),
        "revision": getattr(config, "revision", None),
    }
    return {
        "cache_schema_version": 2,
        "vision_encoder": {
            "class": (
                f"{vision_model.__class__.__module__}."
                f"{vision_model.__class__.__qualname__}"
            ),
            "revision": _stable_config_payload(revision),
            "config": _stable_config_payload(model_config),
        },
        "preprocess": {
            "class": (
                f"{image_processor.__class__.__module__}."
                f"{image_processor.__class__.__qualname__}"
            ),
            "config": _stable_config_payload(preprocess_config),
        },
        "encoder_feature_tokens": int(max_tokens),
        "evaluation_identity": _stable_config_payload(dict(identity_context)),
    }


def audio_encoder_cache_provenance(
    speech_processor,
    speech_model,
    sample_rate: int,
    max_tokens: int,
    max_seconds: float,
    identity_context: Mapping[str, Any],
) -> Dict[str, Any]:
    config = getattr(speech_model, "config", None)
    if config is not None and hasattr(config, "to_dict"):
        model_config = config.to_dict()
    elif config is not None:
        model_config = vars(config)
    else:
        model_config = {}
    if hasattr(speech_processor, "to_dict"):
        processor_config = speech_processor.to_dict()
    else:
        feature_extractor = getattr(speech_processor, "feature_extractor", None)
        tokenizer = getattr(speech_processor, "tokenizer", None)
        processor_config = {
            "feature_extractor": (
                feature_extractor.to_dict()
                if feature_extractor is not None
                and hasattr(feature_extractor, "to_dict")
                else vars(feature_extractor)
                if feature_extractor is not None
                and hasattr(feature_extractor, "__dict__")
                else {}
            ),
            "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        }
    revision = {
        "model_name_or_path": getattr(speech_model, "name_or_path", None),
        "config_name_or_path": getattr(config, "_name_or_path", None),
        "commit_hash": getattr(config, "_commit_hash", None),
        "revision": getattr(config, "revision", None),
    }
    return {
        "cache_schema_version": 2,
        "speech_encoder": {
            "class": (
                f"{speech_model.__class__.__module__}."
                f"{speech_model.__class__.__qualname__}"
            ),
            "revision": _stable_config_payload(revision),
            "config": _stable_config_payload(model_config),
        },
        "processor": {
            "class": (
                f"{speech_processor.__class__.__module__}."
                f"{speech_processor.__class__.__qualname__}"
            ),
            "config": _stable_config_payload(processor_config),
        },
        "preprocess": {
            "sample_rate": int(sample_rate),
            "max_seconds": float(max_seconds),
            "encoder_feature_tokens": int(max_tokens),
            "resampling": "audio_loading_resample_pad_truncate_v1",
        },
        "evaluation_identity": _stable_config_payload(dict(identity_context)),
    }


class FeatureCache:
    def __init__(
        self,
        root: Optional[Path],
        audio_encoder_changed: bool = False,
        strict_audio_integrity: bool = False,
        speech_audio_data_dir: Optional[Path] = None,
        image_cache_context: Optional[Mapping[str, Any]] = None,
        audio_cache_context: Optional[Mapping[str, Any]] = None,
        access_policy: str = "verified_read_write",
    ) -> None:
        if access_policy not in {"verified_read_write", "exclusive_write_only"}:
            raise ValueError(f"unsupported feature cache access policy: {access_policy}")
        self.root = root
        if self.root is not None:
            if access_policy == "exclusive_write_only":
                if not self.root.is_dir() or self.root.is_symlink():
                    raise ValueError(
                        "exclusive write-only feature cache root must be a new prepared directory"
                    )
            else:
                self.root.mkdir(parents=True, exist_ok=True)
        self.access_policy = access_policy
        self._audio_encoder_identity: Optional[str] = None
        self._audio_encoder_base_identity: Optional[str] = None
        self._audio_encoder_weights_identity: Optional[Dict[str, Any]] = None
        self._audio_runtime_object_key: Optional[Tuple[int, ...]] = None
        self._audio_runtime_state_token: Optional[Tuple[Any, ...]] = None
        self._audio_processor_object_key: Optional[int] = None
        self._audio_processor_state_token: Optional[
            Tuple[Tuple[Any, ...], ...]
        ] = None
        self._audio_encoder_changed = bool(audio_encoder_changed)
        self._strict_audio_integrity = bool(strict_audio_integrity)
        self._speech_audio_data_dir = speech_audio_data_dir
        if self._strict_audio_integrity:
            if self._speech_audio_data_dir is None:
                raise ValueError(
                    "strict speech cache requires canonical audio data_dir"
                )
            canonical_data_dir = self._speech_audio_data_dir.resolve(strict=True)
            if (
                not self._speech_audio_data_dir.is_absolute()
                or self._speech_audio_data_dir.is_symlink()
                or not self._speech_audio_data_dir.is_dir()
                or canonical_data_dir != self._speech_audio_data_dir
            ):
                raise ValueError(
                    "strict speech cache audio data_dir must be canonical"
                )
        self._image_cache_context = dict(image_cache_context or {})
        self._audio_cache_context = dict(
            audio_cache_context
            if audio_cache_context is not None
            else self._image_cache_context
        )
        self._payload_records: Dict[str, Dict[str, Dict[str, Any]]] = {
            "image": {},
            "audio": {},
        }
        for cache_kind, context in (
            ("image", self._image_cache_context),
            ("audio", self._audio_cache_context),
        ):
            if not context:
                continue
            for field in ("checkpoint_sha256", "evaluator_sha256"):
                value = context.get(field)
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(char not in "0123456789abcdefABCDEF" for char in value)
                ):
                    raise ValueError(
                        f"{cache_kind} cache context requires exact {field}"
                    )


    @property
    def speech_feature_cache_policy(self) -> str:
        if self._strict_audio_integrity:
            return STRICT_SPEECH_FEATURE_CACHE_POLICY
        return DIAGNOSTIC_SPEECH_FEATURE_CACHE_POLICY
    def _record_payload(
        self,
        kind: str,
        path: Path,
        metadata: Mapping[str, Any],
        digest: Mapping[str, Any],
        acquisition: str,
    ) -> None:
        self._payload_records[kind][str(path.resolve())] = {
            "cache_path": str(path.resolve()),
            "acquisition": acquisition,
            "metadata_sha256": hashlib.sha256(
                json.dumps(
                    metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
            "tensor_digest": dict(digest),
        }

    def payload_provenance(self, kind: str) -> List[Dict[str, Any]]:
        if kind not in self._payload_records:
            raise ValueError(f"unsupported feature cache kind: {kind}")
        return [
            dict(record)
            for _, record in sorted(self._payload_records[kind].items())
        ]

    def produced_payload_provenance(self, kind: str) -> List[Dict[str, Any]]:
        return [
            record
            for record in self.payload_provenance(kind)
            if record.get("acquisition") == "recomputed_and_written"
        ]

    @staticmethod
    def _atomic_save(
        payload: Mapping[str, Any], path: Path, *, exclusive: bool
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(dict(payload), handle)
                handle.flush()
                os.fsync(handle.fileno())
            if exclusive:
                try:
                    os.link(temporary_path, path)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"refusing to reuse existing feature cache payload: {path}"
                    ) from exc
            else:
                os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def verify_payloads(self, kind: str) -> List[Dict[str, Any]]:
        records = self.payload_provenance(kind)
        for record in records:
            path = Path(str(record["cache_path"]))
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"{kind} feature cache payload is not a regular file: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{kind} feature cache payload is invalid: {path}")
            features = payload.get("features")
            stored_digest = payload.get("tensor_digest")
            if (
                not isinstance(features, torch.Tensor)
                or not isinstance(stored_digest, Mapping)
                or dict(stored_digest) != tensor_content_digest(features)
                or dict(stored_digest) != record["tensor_digest"]
            ):
                raise ValueError(f"{kind} feature cache tensor digest mismatch: {path}")
        return records

    def _path(
        self, kind: str, rec: Dict[str, Any], identity: str = "legacy"
    ) -> Optional[Path]:
        if self.root is None:
            return None
        source = rec.get("source", "src")
        if isinstance(source, dict):
            source = "_".join(
                str(source.get(key, "")) for key in ("dataset", "config", "split")
            )
        safe_source = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in str(source)
        )[:80]
        raw_id = str(rec.get("id", 0))
        safe_id = (
            f"{int(raw_id):06d}"
            if raw_id.isdigit()
            else "".join(
                char if char.isalnum() or char in {"-", "_"} else "_"
                for char in raw_id
            )[:48]
        )
        if identity == "legacy":
            return self.root / kind / f"{safe_source}_{safe_id}.pt"
        return self.root / kind / identity / f"{safe_source}_{safe_id}.pt"

    @staticmethod
    def _load_audio_payload(
        path: Path, *, cache_identity: str, audio_sha256: str, row_sha256: str
    ) -> torch.Tensor:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError(f"invalid speech cache payload: {path}")
        tensor = payload.get("tensor")
        if not torch.is_tensor(tensor):
            raise ValueError(f"speech cache payload has no tensor: {path}")
        if (
            payload.get("cache_identity") != cache_identity
            or payload.get("audio_sha256") != audio_sha256
            or payload.get("row_sha256") != row_sha256
            or payload.get("tensor_identity") != tensor_content_identity(tensor)
        ):
            raise ValueError(f"speech cache payload integrity mismatch: {path}")
        return tensor

    @staticmethod
    def _save_audio_payload(
        path: Path,
        tensor: torch.Tensor,
        *,
        cache_identity: str,
        audio_sha256: str,
        row_sha256: str,
    ) -> None:
        value = tensor.detach().cpu().contiguous()
        torch.save(
            {
                "schema_version": 1,
                "cache_identity": cache_identity,
                "audio_sha256": audio_sha256,
                "row_sha256": row_sha256,
                "tensor_identity": tensor_content_identity(value),
                "tensor": value,
            },
            path,
        )

    def image_cache_provenance(
        self, image_processor, vision_model, max_tokens: int
    ) -> Dict[str, Any]:
        return image_encoder_cache_provenance(
            image_processor,
            vision_model,
            max_tokens,
            self._image_cache_context,
        )

    def image_media_provenance(
        self, records: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        provenance: List[Dict[str, Any]] = []
        for rec in records:
            media_path = (
                Path(str(rec["image_path"]))
                .expanduser()
                .resolve(strict=True)
            )
            if not media_path.is_file():
                raise FileNotFoundError(media_path)
            provenance.append({
                "canonical_path": str(media_path),
                "sha256": sha256_file(media_path),
                "size_bytes": int(media_path.stat().st_size),
            })
        return provenance

    def _image_cache_metadata(
        self,
        image_processor,
        vision_model,
        rec: Dict[str, Any],
        max_tokens: int,
    ) -> Dict[str, Any]:
        media = self.image_media_provenance([rec])[0]
        return {
            **self.image_cache_provenance(
                image_processor, vision_model, max_tokens
            ),
            "media": media,
        }

    def _image_cache_path(
        self, metadata: Mapping[str, Any]
    ) -> Optional[Path]:
        if self.root is None:
            return None
        encoded = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        identity = hashlib.sha256(encoded).hexdigest()
        return self.root / "image_v2" / identity[:2] / f"{identity}.pt"

    def image_batch(
        self,
        image_processor,
        vision_model,
        records: Sequence[Dict[str, Any]],
        device: torch.device,
        max_tokens: int,
    ) -> torch.Tensor:
        metadata = [
            self._image_cache_metadata(
                image_processor, vision_model, rec, max_tokens
            )
            for rec in records
        ]
        paths = [self._image_cache_path(item) for item in metadata]
        if self.access_policy == "verified_read_write" and all(
            path is not None and path.exists() for path in paths
        ):
            cached_features: List[torch.Tensor] = []
            for path, expected_metadata in zip(paths, metadata):
                assert path is not None
                if path.is_symlink() or not path.is_file():
                    raise ValueError(
                        f"image feature cache payload is not a regular file: {path}"
                    )
                payload = torch.load(
                    path, map_location=device, weights_only=True
                )
                if (
                    not isinstance(payload, Mapping)
                    or payload.get("metadata") != expected_metadata
                    or not isinstance(payload.get("features"), torch.Tensor)
                ):
                    raise ValueError(
                        f"image feature cache metadata mismatch: {path}"
                    )
                stored_digest = payload.get("tensor_digest")
                actual_digest = tensor_content_digest(payload["features"])
                if not isinstance(stored_digest, Mapping) or dict(stored_digest) != actual_digest:
                    raise ValueError(f"image feature cache tensor digest mismatch: {path}")
                self._record_payload(
                    "image", path, expected_metadata, actual_digest, "verified_cache_hit"
                )
                cached_features.append(payload["features"].to(device))
            return torch.cat(cached_features, dim=0)
        features = image_features_from_paths(
            image_processor,
            vision_model,
            [str(item["media"]["canonical_path"]) for item in metadata],
            device,
            max_tokens,
        )
        for feat, path, item in zip(features, paths, metadata):
            if path is not None:
                cached = feat.detach().cpu().unsqueeze(0)
                digest = tensor_content_digest(cached)
                self._atomic_save(
                    {
                        "metadata": item,
                        "features": cached,
                        "tensor_digest": digest,
                    },
                    path,
                    exclusive=self.access_policy == "exclusive_write_only",
                )
                self._record_payload(
                    "image", path, item, digest, "recomputed_and_written"
                )
        return features


    def audio_cache_provenance(
        self,
        speech_processor,
        speech_model,
        sample_rate: int,
        max_tokens: int,
        max_seconds: float,
    ) -> Dict[str, Any]:
        return audio_encoder_cache_provenance(
            speech_processor,
            speech_model,
            sample_rate,
            max_tokens,
            max_seconds,
            self._audio_cache_context,
        )

    def audio_media_provenance(
        self, records: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        provenance: List[Dict[str, Any]] = []
        for rec in records:
            media_path = (
                Path(str(rec["audio_path"]))
                .expanduser()
                .resolve(strict=True)
            )
            if not media_path.is_file():
                raise FileNotFoundError(media_path)
            provenance.append({
                "canonical_path": str(media_path),
                "sha256": sha256_file(media_path),
                "size_bytes": int(media_path.stat().st_size),
            })
        return provenance

    def _audio_cache_metadata(
        self,
        speech_processor,
        speech_model,
        rec: Dict[str, Any],
        sample_rate: int,
        max_tokens: int,
        max_seconds: float,
    ) -> Dict[str, Any]:
        media = self.audio_media_provenance([rec])[0]
        return {
            **self.audio_cache_provenance(
                speech_processor,
                speech_model,
                sample_rate,
                max_tokens,
                max_seconds,
            ),
            "media": media,
            "record_preprocess": _stable_config_payload(
                rec.get("preprocess", {})
            ),
            "record_sample_rate": rec.get("sample_rate"),
        }

    def _audio_cache_path(
        self, metadata: Mapping[str, Any]
    ) -> Optional[Path]:
        if self.root is None:
            return None
        encoded = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        identity = hashlib.sha256(encoded).hexdigest()
        return self.root / "audio_v2" / identity[:2] / f"{identity}.pt"

    def audio_batch(
        self,
        speech_processor,
        speech_model,
        records: Sequence[Dict[str, Any]],
        device: torch.device,
        sample_rate: int,
        max_tokens: int,
        max_seconds: float = 0.0,
    ) -> torch.Tensor:
        if self._strict_audio_integrity:
            if self._speech_audio_data_dir is None:
                raise RuntimeError("strict speech cache audio data_dir is missing")
            snapshots: List[bytes] = []
            audio_sha256s: List[str] = []
            runtime_audio_paths: List[str] = []
            for row_index, record in enumerate(records):
                raw_path = record.get("audio_path")
                if not isinstance(raw_path, str) or not raw_path:
                    raise ValueError(f"speech row {row_index} has no audio_path")
                declared_sha = record.get("audio_sha256")
                if not isinstance(declared_sha, str) or not _is_full_hex(
                    declared_sha, 64
                ):
                    raise ValueError(
                        "strict speech row requires exact lowercase audio_sha256"
                    )
                audio_path, payload = secure_speech_audio_snapshot(
                    raw_path,
                    data_dir=self._speech_audio_data_dir,
                    row_index=row_index,
                )
                audio_sha256 = hashlib.sha256(payload).hexdigest()
                if audio_sha256 != declared_sha:
                    raise ValueError(
                        f"speech audio SHA256 mismatch: {audio_path}"
                    )
                runtime_audio_paths.append(str(audio_path))
                snapshots.append(payload)
                audio_sha256s.append(audio_sha256)
            return audio_features_from_paths(
                speech_processor,
                speech_model,
                runtime_audio_paths,
                device,
                sample_rate,
                max_tokens,
                max_seconds,
                audio_sha256s=audio_sha256s,
                require_expected_sha256=True,
                audio_payloads=snapshots,
            )

        runtime_audio_paths = [str(record["audio_path"]) for record in records]
        encoder = speech_encoder_module(speech_model)
        runtime_object_key = (
            id(speech_model),
            id(encoder),
            int(sample_rate),
            int(max_tokens),
            hash(float(max_seconds)),
        )
        runtime_state_token = _encoder_state_token(speech_model)
        processor_object_key = id(speech_processor)
        processor_state_token = speech_processor_state_token(speech_processor)
        processor_drifted = (
            processor_object_key == self._audio_processor_object_key
            and self._audio_processor_state_token is not None
            and processor_state_token != self._audio_processor_state_token
        )
        encoder_context_changed = runtime_object_key != self._audio_runtime_object_key
        if runtime_object_key == self._audio_runtime_object_key:
            if runtime_state_token != self._audio_runtime_state_token:
                self._audio_encoder_changed = True
        else:
            encoder_weights_identity = encoder_weights_cache_identity(speech_model)
            encoder_base_identity = encoder_cache_identity(
                speech_model,
                sample_rate,
                max_tokens,
                max_seconds,
                encoder_weights_identity=encoder_weights_identity,
                processor_state_token=_ABSENT_SPEECH_PROCESSOR_STATE_TOKEN,
            )
            if (
                self._audio_encoder_base_identity is not None
                and encoder_base_identity != self._audio_encoder_base_identity
            ):
                self._audio_encoder_changed = True
            self._audio_encoder_base_identity = encoder_base_identity
            self._audio_encoder_weights_identity = encoder_weights_identity
            self._audio_runtime_object_key = runtime_object_key
            self._audio_runtime_state_token = runtime_state_token
        processor_context_changed = (
            encoder_context_changed
            or processor_object_key != self._audio_processor_object_key
            or processor_state_token != self._audio_processor_state_token
        )
        if processor_context_changed:
            if self._audio_encoder_weights_identity is None:
                raise RuntimeError("speech encoder weights identity was not initialized")
            self._audio_encoder_identity = encoder_cache_identity(
                speech_model,
                sample_rate,
                max_tokens,
                max_seconds,
                speech_processor=speech_processor,
                encoder_weights_identity=self._audio_encoder_weights_identity,
                processor_state_token=processor_state_token,
            )
            self._audio_processor_object_key = processor_object_key
            self._audio_processor_state_token = processor_state_token
        encoder_identity = self._audio_encoder_identity
        if encoder_identity is None:
            raise RuntimeError("speech cache identity was not initialized")
        encoder_trainable = any(
            param.requires_grad
            for param in speech_encoder_module(speech_model).parameters()
        )
        cache_enabled = not encoder_trainable and not self._audio_encoder_changed
        if self._audio_cache_context:
            if not cache_enabled:
                return audio_features_from_paths(
                    speech_processor,
                    speech_model,
                    runtime_audio_paths,
                    device,
                    sample_rate,
                    max_tokens,
                    max_seconds,
                )
            metadata = [
                self._audio_cache_metadata(
                    speech_processor,
                    speech_model,
                    rec,
                    sample_rate,
                    max_tokens,
                    max_seconds,
                )
                for rec in records
            ]
            paths = [self._audio_cache_path(item) for item in metadata]
            if self.access_policy == "verified_read_write" and all(
                path is not None and path.exists() for path in paths
            ):
                cached_features: List[torch.Tensor] = []
                for path, expected_metadata in zip(paths, metadata):
                    assert path is not None
                    if path.is_symlink() or not path.is_file():
                        raise ValueError(
                            f"audio feature cache payload is not a regular file: {path}"
                        )
                    payload = torch.load(
                        path, map_location=device, weights_only=True
                    )
                    if (
                        not isinstance(payload, Mapping)
                        or payload.get("metadata") != expected_metadata
                        or not isinstance(payload.get("features"), torch.Tensor)
                    ):
                        raise ValueError(
                            f"audio feature cache metadata mismatch: {path}"
                        )
                    stored_digest = payload.get("tensor_digest")
                    actual_digest = tensor_content_digest(payload["features"])
                    if (
                        not isinstance(stored_digest, Mapping)
                        or dict(stored_digest) != actual_digest
                    ):
                        raise ValueError(
                            f"audio feature cache tensor digest mismatch: {path}"
                        )
                    self._record_payload(
                        "audio",
                        path,
                        expected_metadata,
                        actual_digest,
                        "verified_cache_hit",
                    )
                    cached_features.append(payload["features"].to(device))
                return torch.cat(cached_features, dim=0)
            features = audio_features_from_paths(
                speech_processor,
                speech_model,
                [str(item["media"]["canonical_path"]) for item in metadata],
                device,
                sample_rate,
                max_tokens,
                max_seconds,
            )
            for feat, path, item in zip(features, paths, metadata):
                if path is not None:
                    cached = feat.detach().cpu().unsqueeze(0)
                    digest = tensor_content_digest(cached)
                    self._atomic_save(
                        {
                            "metadata": item,
                            "features": cached,
                            "tensor_digest": digest,
                        },
                        path,
                        exclusive=self.access_policy == "exclusive_write_only",
                    )
                    self._record_payload(
                        "audio",
                        path,
                        item,
                        digest,
                        "recomputed_and_written",
                    )
            return features

        cache_read_enabled = cache_enabled and not processor_drifted
        audio_sha256s: List[str] = []
        row_sha256s: List[str] = []
        cache_identities: List[str] = []
        for record, audio_path in zip(records, runtime_audio_paths):
            declared_sha = record.get("audio_sha256")
            if declared_sha is not None and not isinstance(declared_sha, str):
                raise ValueError("speech row audio_sha256 must be a string")
            _, audio_sha256 = audio_file_snapshot(
                audio_path,
                declared_sha,
                require_expected=self._strict_audio_integrity,
            )
            row_sha256 = canonical_identity_sha256(record)
            cache_identity = canonical_identity_sha256(
                {
                    "encoder_identity": encoder_identity,
                    "audio_sha256": audio_sha256,
                    "row_sha256": row_sha256,
                }
            )
            audio_sha256s.append(audio_sha256)
            row_sha256s.append(row_sha256)
            cache_identities.append(cache_identity)
        paths = [
            self._path("audio", record, cache_identity)
            if cache_enabled
            else None
            for record, cache_identity in zip(records, cache_identities)
        ]
        if (
            self.access_policy == "verified_read_write"
            and cache_read_enabled
            and all(path is not None and path.exists() for path in paths)
        ):
            return torch.cat(
                [
                    self._load_audio_payload(
                        path,
                        cache_identity=cache_identity,
                        audio_sha256=audio_sha256,
                        row_sha256=row_sha256,
                    ).to(device)
                    for path, cache_identity, audio_sha256, row_sha256 in zip(
                        paths, cache_identities, audio_sha256s, row_sha256s
                    )
                    if path is not None
                ],
                dim=0,
            )
        features = audio_features_from_paths(
            speech_processor,
            speech_model,
            runtime_audio_paths,
            device,
            sample_rate,
            max_tokens,
            max_seconds,
            audio_sha256s=audio_sha256s,
            require_expected_sha256=True,
        )
        for feat, path, cache_identity, audio_sha256, row_sha256 in zip(
            features, paths, cache_identities, audio_sha256s, row_sha256s
        ):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._save_audio_payload(
                    path,
                    feat.unsqueeze(0),
                    cache_identity=cache_identity,
                    audio_sha256=audio_sha256,
                    row_sha256=row_sha256,
                )
        return features


def make_wrapper(model, vision_model, speech_model, args):
    hidden_size = int(model.config.hidden_size)
    vision_cfg = getattr(vision_model.config, "vision_config", vision_model.config)
    vision_dim = int(getattr(vision_cfg, "hidden_size"))
    speech_dim_value = getattr(speech_model.config, "hidden_size", None) or getattr(speech_model.config, "d_model")
    speech_dim = int(speech_dim_value)
    image_retrieval_dim = hidden_size if str(getattr(args, "image_alignment_target", "clip_text")) == "olmoe_caption_hidden" else int(getattr(vision_model.config, "projection_dim", hidden_size))
    audio_retrieval_dim = hidden_size
    if str(getattr(args, "speech_target_space", "olmoe_text_hidden")) == "whisper_decoder_text":
        decoder = getattr(speech_model, "decoder", None)
        if decoder is None and hasattr(speech_model, "model"):
            decoder = getattr(speech_model.model, "decoder", None)
        embed_tokens = getattr(decoder, "embed_tokens", None) if decoder is not None else None
        if embed_tokens is None:
            raise RuntimeError("speech_target_space=whisper_decoder_text requires decoder token embeddings")
        audio_retrieval_dim = int(embed_tokens.embedding_dim)
    return OLMoEMultimodalPrefixWrapper(
        lm=model,
        hidden_size=hidden_size,
        image_input_dim=vision_dim,
        audio_input_dim=speech_dim,
        image_prefix_tokens=args.image_prefix_tokens,
        audio_prefix_tokens=args.audio_prefix_tokens,
        image_retrieval_dim=image_retrieval_dim,
        audio_retrieval_dim=audio_retrieval_dim,
        use_prefix_residual_alignment=args.alignment_prefix_residual,
        image_bridge_type=args.image_bridge_type,
        audio_bridge_type=args.audio_bridge_type,
        bridge_num_heads=args.bridge_num_heads,
    )


def select_speech_encoder_trainable(speech_model, last_blocks: int, layer_norm_only: bool) -> List[Tuple[str, torch.nn.Parameter]]:
    if last_blocks not in {0, 1, 2}:
        raise ValueError("speech_unfreeze_last_blocks must be 0, 1, or 2")
    for param in speech_model.parameters():
        param.requires_grad_(False)
    encoder = speech_encoder_module(speech_model)
    layers = getattr(encoder, "layers", None)
    if last_blocks > 0 and (layers is None or len(layers) < last_blocks):
        raise RuntimeError(f"Speech encoder does not expose {last_blocks} encoder blocks")
    selected_ids = set()
    if last_blocks > 0:
        for layer in list(layers)[-last_blocks:]:
            for param in layer.parameters():
                param.requires_grad_(True)
                selected_ids.add(id(param))
    if layer_norm_only:
        layer_norm = getattr(encoder, "layer_norm", None)
        if layer_norm is None:
            raise RuntimeError("Speech encoder does not expose encoder.layer_norm")
        for param in layer_norm.parameters():
            param.requires_grad_(True)
            selected_ids.add(id(param))
    return [(name, param) for name, param in encoder.named_parameters() if id(param) in selected_ids]


def configure_trainable(
    wrapper,
    train_router_gates: bool,
    train_experts: bool,
    train_lm_head: bool,
    learning_rate: float,
    router_lr: float,
    expert_lr: float,
    retrieval_lr: float,
    lm_head_lr: float,
    weight_decay: float,
    speech_model=None,
    speech_unfreeze_last_blocks: int = 0,
    speech_unfreeze_layer_norm: bool = False,
    speech_encoder_lr: float = 1e-5,
):
    for param in wrapper.parameters():
        param.requires_grad_(False)
    for param in wrapper.image_resampler.parameters():
        param.requires_grad_(True)
    for param in wrapper.audio_resampler.parameters():
        param.requires_grad_(True)
    for param in wrapper.image_retrieval_head.parameters():
        param.requires_grad_(True)
    for param in wrapper.audio_retrieval_head.parameters():
        param.requires_grad_(True)
    for param in wrapper.image_direct_retrieval_head.parameters():
        param.requires_grad_(False)
    for param in wrapper.audio_direct_retrieval_head.parameters():
        param.requires_grad_(False)
    speech_params: List[torch.nn.Parameter] = []
    speech_param_names: List[str] = []
    if speech_model is not None:
        selected = select_speech_encoder_trainable(
            speech_model,
            int(speech_unfreeze_last_blocks),
            bool(speech_unfreeze_layer_norm),
        )
        speech_param_names = [name for name, _ in selected]
        speech_params = [param for _, param in selected]
    elif speech_unfreeze_last_blocks or speech_unfreeze_layer_norm:
        raise ValueError("speech_model is required for partial Whisper unfreeze")
    gate_params: List[torch.nn.Parameter] = []
    expert_params: List[torch.nn.Parameter] = []
    lm_head_params: List[torch.nn.Parameter] = []
    if train_lm_head:
        seen_lm_head_params = set()
        for module in [wrapper.lm.get_output_embeddings(), wrapper.lm.get_input_embeddings()]:
            if module is None:
                continue
            for param in module.parameters():
                if id(param) in seen_lm_head_params:
                    continue
                seen_lm_head_params.add(id(param))
                param.requires_grad_(True)
                lm_head_params.append(param)
    if train_router_gates:
        for layer in wrapper.lm.model.layers:
            for param in layer.mlp.gate.parameters():
                param.requires_grad_(True)
                gate_params.append(param)
    if train_experts:
        for layer in wrapper.lm.model.layers:
            for param in layer.mlp.experts.parameters():
                param.requires_grad_(True)
                expert_params.append(param)
    resampler_params = list(wrapper.image_resampler.parameters()) + list(wrapper.audio_resampler.parameters())
    retrieval_params = list(wrapper.image_retrieval_head.parameters()) + list(wrapper.audio_retrieval_head.parameters())
    retrieval_lr_value = retrieval_lr if float(retrieval_lr) > 0.0 else learning_rate
    groups = [
        {"params": resampler_params, "lr": learning_rate, "weight_decay": weight_decay, "name": "prefix_resamplers"},
        {"params": retrieval_params, "lr": retrieval_lr_value, "weight_decay": weight_decay, "name": "retrieval_heads"},
    ]
    if speech_params:
        groups.append({"params": speech_params, "lr": speech_encoder_lr, "weight_decay": weight_decay, "name": "speech_encoder_partial"})
    if gate_params:
        groups.append({"params": gate_params, "lr": router_lr, "weight_decay": weight_decay, "name": "router_gates"})
    if expert_params:
        groups.append({"params": expert_params, "lr": expert_lr, "weight_decay": weight_decay, "name": "experts"})
    if lm_head_params:
        groups.append({"params": lm_head_params, "lr": lm_head_lr, "weight_decay": weight_decay, "name": "lm_head_embeddings"})
    trainable_params = sum(p.numel() for group in groups for p in group["params"] if p.requires_grad)
    optimizer_groups = [
        {
            "name": str(group.get("name", "group")),
            "lr": float(group.get("lr", learning_rate)),
            "weight_decay": float(group.get("weight_decay", weight_decay)),
            "trainable_params": int(sum(p.numel() for p in group["params"] if p.requires_grad)),
        }
        for group in groups
    ]
    optimizer = torch.optim.AdamW(groups)
    return optimizer, {
        "trainable_params": int(trainable_params),
        "train_router_gates": bool(train_router_gates),
        "train_experts": bool(train_experts),
        "train_lm_head": bool(train_lm_head),
        "speech_encoder_trainable": bool(speech_params),
        "speech_encoder_trainable_names": speech_param_names,
        "speech_unfreeze_last_blocks": int(speech_unfreeze_last_blocks),
        "speech_unfreeze_layer_norm": bool(speech_unfreeze_layer_norm),
        "optimizer_groups": optimizer_groups,
    }



def parameter_gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.requires_grad and param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().cpu())
    return float(math.sqrt(total))


def bridge_grad_norms(wrapper) -> Dict[str, float]:
    image_params = list(wrapper.image_resampler.parameters()) + list(wrapper.image_retrieval_head.parameters())
    audio_params = list(wrapper.audio_resampler.parameters()) + list(wrapper.audio_retrieval_head.parameters())
    return {
        "image_bridge_grad_norm": parameter_gradient_norm(image_params),
        "audio_bridge_grad_norm": parameter_gradient_norm(audio_params),
    }


def validate_alignment_pretrain_trainable(wrapper, speech_model=None) -> int:
    modules = [
        wrapper.image_resampler,
        wrapper.audio_resampler,
        wrapper.image_retrieval_head,
        wrapper.audio_retrieval_head,
    ]
    params = [param for module in modules for param in module.parameters() if param.requires_grad]
    if speech_model is not None:
        params.extend(param for param in speech_encoder_module(speech_model).parameters() if param.requires_grad)
    if not params:
        raise RuntimeError("alignment_pretrain_steps > 0 requires trainable bridge or speech encoder parameters")
    return int(sum(param.numel() for param in params))


def trainable_optimization_parameters(wrapper, speech_model=None) -> List[torch.nn.Parameter]:
    params = [param for param in wrapper.parameters() if param.requires_grad]
    if speech_model is not None:
        params.extend(param for param in speech_encoder_module(speech_model).parameters() if param.requires_grad)
    return params
def needs_alignment_target_bank(args) -> bool:
    return bool(
        args.alignment_pretrain_steps > 0
        or max(args.contrastive_coef, args.image_contrastive_coef, args.speech_contrastive_coef) > 0.0
    )

def parse_alignment_pretrain_modalities(value: str) -> List[str]:
    modalities = [item.strip() for item in str(value).split(",") if item.strip()]
    invalid = sorted(set(modalities) - {"image", "speech"})
    if not modalities or invalid:
        raise ValueError(
            "alignment_pretrain_modalities must contain only image and/or speech; "
            f"got {value!r}"
        )
    return modalities


def evaluate_text_blocks(exp_id: str, model, tokenizer, blocks: Sequence[Dict[str, Any]], out_dir: Path, meta: Dict[str, Any], max_length: int, batch_size: int) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    rows: List[Dict[str, Any]] = []
    loss_sum = 0.0
    token_count = 0
    correct = 0
    total_pred = 0
    by_task: Dict[str, Dict[str, float]] = {}
    router_accum: Optional[Dict[str, Any]] = None
    for start in range(0, len(blocks), batch_size):
        batch_rows = list(blocks[start:start + batch_size])
        batch = tensorize_blocks(batch_rows, device, max_length, pad_id)
        with torch.no_grad():
            outputs = model(**batch, output_router_logits=True, return_dict=True)
        shift_logits = outputs.logits[:, :-1].detach().float()
        shift_labels = batch["labels"][:, 1:]
        mask = shift_labels != -100
        batch_loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), ignore_index=-100, reduction="sum")
        ntok = int(mask.sum().item())
        loss_sum += float(batch_loss.cpu())
        token_count += ntok
        preds = shift_logits.argmax(dim=-1)
        correct += int(((preds == shift_labels) & mask).sum().item())
        total_pred += ntok
        if router_accum is None:
            router_accum = router_metrics(
                outputs,
                int(meta["top_k"]),
                int(model.config.num_experts),
                float(meta["capacity_factor"]),
                model=model,
            )
        for idx, row in enumerate(batch_rows):
            task = str(row.get("task", "text"))
            bucket = by_task.setdefault(task, {"blocks": 0, "loss_sum": 0.0, "tokens": 0})
            bucket["blocks"] += 1
            row_logits = shift_logits[idx:idx + 1]
            row_labels = shift_labels[idx:idx + 1]
            row_mask = row_labels != -100
            row_loss = F.cross_entropy(row_logits.reshape(-1, row_logits.shape[-1]), row_labels.reshape(-1), ignore_index=-100, reduction="sum")
            bucket["loss_sum"] += float(row_loss.cpu())
            bucket["tokens"] += int(row_mask.sum().item())
    loss_mean = loss_sum / max(1, token_count)
    task_metrics = {
        task: {
            "blocks": int(vals.get("blocks", 0)),
            "tokens": int(vals.get("tokens", 0)),
            "loss": float(vals.get("loss_sum", 0.0) / max(1, vals.get("tokens", 0))),
            "perplexity": float(math.exp(min(20.0, vals.get("loss_sum", 0.0) / max(1, vals.get("tokens", 0))))),
        }
        for task, vals in sorted(by_task.items())
    }
    artifact = {
        "experiment_id": exp_id,
        **meta,
        "real_subset": True,
        "eval_blocks": len(blocks),
        "eval_tokens": token_count,
        "loss": float(loss_mean),
        "perplexity": float(math.exp(min(20.0, loss_mean))),
        "next_token_accuracy": float(correct / max(1, total_pred)),
        "task_metrics": task_metrics,
        "task_block_counts": {task: {"blocks": vals["blocks"]} for task, vals in task_metrics.items()},
        **(router_accum or {}),
        **cuda_metrics(),
    }
    save_json(out_dir / exp_id / "metrics.json", artifact)
    print(json.dumps({"experiment_id": exp_id, "loss": artifact["loss"], "perplexity": artifact["perplexity"], "next_token_accuracy": artifact["next_token_accuracy"], "eval_blocks": len(blocks)}, sort_keys=True))
    return artifact


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def lm_text_embeddings(lm, tokenizer, texts: Sequence[str], device: torch.device, max_length: int, batch_size: int) -> torch.Tensor:
    vectors = []
    for start in range(0, len(texts), batch_size):
        sub = list(texts[start:start + batch_size])
        encoded = tokenizer(sub, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = lm(**encoded, output_hidden_states=True, output_router_logits=False, return_dict=True)
        vectors.append(mean_pool(outputs.hidden_states[-1].detach().float(), encoded["attention_mask"]).cpu())
    return F.normalize(torch.cat(vectors, dim=0), dim=-1)


def prefix_embeddings(lm, prefix_batches: Iterable[torch.Tensor], device: torch.device) -> torch.Tensor:
    vectors = []
    target_dtype = lm.get_input_embeddings().weight.dtype
    for prefix in prefix_batches:
        prefix = prefix.to(device=device, dtype=target_dtype)
        attention_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = lm(inputs_embeds=prefix, attention_mask=attention_mask, output_hidden_states=True, output_router_logits=True, return_dict=True)
        vectors.append(outputs.hidden_states[-1].detach().float().mean(dim=1).cpu())
    return F.normalize(torch.cat(vectors, dim=0), dim=-1)


def differentiable_prefix_embeddings(lm, prefix: torch.Tensor) -> torch.Tensor:
    target_dtype = lm.get_input_embeddings().weight.dtype
    prefix = prefix.to(dtype=target_dtype)
    attention_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)
    outputs = lm(inputs_embeds=prefix, attention_mask=attention_mask, output_hidden_states=True, output_router_logits=True, return_dict=True)
    return F.normalize(outputs.hidden_states[-1].float().mean(dim=1), dim=-1)


def symmetric_contrastive_loss(left: torch.Tensor, right: torch.Tensor, temperature: float) -> torch.Tensor:
    if left.shape[0] < 2 or right.shape[0] < 2:
        return left.new_zeros(())
    logits = (left @ right.T) / max(float(temperature), 1e-6)
    labels = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def canonical_identity_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def tensor_content_identity(tensor: torch.Tensor) -> Dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    header = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }
    raw_bytes = value.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256(
        json.dumps(
            header, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
        + raw_bytes
    ).hexdigest()
    return {
        **header,
        "bytes": len(raw_bytes),
        "sha256": digest,
        "hash_semantics": "canonical_dtype_shape_header_plus_contiguous_tensor_bytes",
    }


def _checkpoint_weight_source_identity(
    provenance: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not provenance:
        return {"applied": False}
    digest = str(provenance.get("sha256", "")).lower()
    if not _is_full_hex(digest, 64):
        raise ValueError(
            "speech shared teacher checkpoint source requires exact SHA256"
        )
    return {
        "applied": True,
        "path": str(provenance.get("path", "")),
        "sha256": digest,
        "manifest_sha256": provenance.get("manifest_sha256"),
        "source_commit_sha": provenance.get("source_commit_sha"),
    }


def speech_shared_teacher_runtime_identities(
    lm,
    tokenizer,
    base_model: str,
    *,
    multimodal_initialization: Optional[Dict[str, Any]],
    stage_b_initialization: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resolved_model = resolve_model(base_model).as_dict()
    config = getattr(lm, "config", None)
    if config is None or not hasattr(config, "to_dict"):
        raise ValueError("speech shared teacher model has no serializable config")
    model_config_sha256 = canonical_identity_sha256(config.to_dict())
    weight_sources = {
        "immutable_hf_checkpoint": resolved_model,
        "multimodal_initialization": _checkpoint_weight_source_identity(
            multimodal_initialization
        ),
        "stage_b_initialization": _checkpoint_weight_source_identity(
            stage_b_initialization
        ),
    }
    teacher_model_identity = {
        "model_class": type(lm).__qualname__,
        "config_name_or_path": str(getattr(config, "_name_or_path", "")),
        "config_sha256": model_config_sha256,
        "weights_identity_semantics": (
            "immutable_hf_revision_plus_applied_checkpoint_sha256"
        ),
        "weight_sources": weight_sources,
        "weights_identity_sha256": canonical_identity_sha256(weight_sources),
    }
    teacher_model_identity["identity_sha256"] = canonical_identity_sha256(
        teacher_model_identity
    )

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError("speech shared teacher tokenizer has no exact vocabulary")
    vocab = get_vocab()
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("speech shared teacher tokenizer vocabulary is empty")
    vocab_sha256 = canonical_identity_sha256(
        sorted((str(token), int(index)) for token, index in vocab.items())
    )
    tokenizer_config = {
        "tokenizer_class": type(tokenizer).__qualname__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "repo_id": resolved_model["repo_id"],
        "revision": resolved_model["revision"],
        "vocab_size": len(vocab),
        "vocab_sha256": vocab_sha256,
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0)),
        "padding_side": str(getattr(tokenizer, "padding_side", "")),
        "truncation_side": str(getattr(tokenizer, "truncation_side", "")),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
    }
    tokenizer_identity = {
        **tokenizer_config,
        "config_sha256": canonical_identity_sha256(tokenizer_config),
    }
    tokenizer_identity["identity_sha256"] = canonical_identity_sha256(
        tokenizer_identity
    )
    return teacher_model_identity, tokenizer_identity


def speech_shared_teacher_row_identity(
    row: Dict[str, Any], row_index: int
) -> str:
    return f"canonical_row_sha256:{speech_source_row_identity(row, row_index)}"


def speech_shared_transcript_duplicate_key(row: Dict[str, Any], row_index: int) -> str:
    transcript = str(row.get("transcript", "") or "")
    normalized = " ".join(transcript.split()).casefold()
    if not normalized:
        raise ValueError(
            f"speech teacher-bank train row {row_index} has no transcript"
        )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def speech_teacher_bank_batch_size(args) -> int:
    batch_size = int(getattr(args, "speech_teacher_bank_batch_size", 64))
    if batch_size <= 0:
        raise ValueError("speech teacher bank batch size must be positive")
    return batch_size


def build_speech_shared_teacher_bank(
    lm,
    tokenizer,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    max_length: int,
    *,
    strict_split_binding: Dict[str, Any],
    teacher_model_identity: Dict[str, Any],
    tokenizer_identity: Dict[str, Any],
    build_batch_size: int = 64,
    reusable_bank: Optional[torch.Tensor] = None,
    reusable_bank_exact: bool = False,
    reusable_bank_expected_sha256: Optional[str] = None,
) -> Tuple[torch.Tensor, List[str], List[str], Dict[str, Any]]:
    """Build a deterministic, row-aligned transcript-only OLMoE hidden bank."""
    if len(rows) < 2:
        raise ValueError("speech shared contrastive teacher bank requires >= 2 train rows")
    if (
        strict_split_binding.get("binding_verified") is not True
        or strict_split_binding.get("speech_train_file", {}).get("rows") != len(rows)
    ):
        raise ValueError(
            "speech shared contrastive teacher bank requires exact speech_train binding"
        )
    build_batch_size = int(build_batch_size)
    if build_batch_size <= 0:
        raise ValueError("speech shared contrastive teacher bank batch size must be positive")
    for label, identity in (
        ("teacher model", teacher_model_identity),
        ("tokenizer", tokenizer_identity),
    ):
        if (
            not isinstance(identity, dict)
            or not _is_full_hex(identity.get("identity_sha256"), 64)
        ):
            raise ValueError(
                f"speech shared contrastive {label} identity is incomplete"
            )
    row_identities: List[str] = []
    duplicate_keys: List[str] = []
    source_datasets = set()
    for row_index, row in enumerate(rows):
        if str(row.get("partition", "")) != "train":
            raise ValueError(
                "speech shared contrastive teacher bank accepts train partition rows only"
            )
        if "split" in row and str(row["split"]) != "train":
            raise ValueError(
                "speech shared contrastive teacher bank cannot use dev/eval split rows"
            )
        if row.get("audio_path"):
            reject_forbidden_development_path(
                str(row["audio_path"]),
                f"speech teacher-bank train row {row_index} audio path",
            )
        row_identities.append(speech_shared_teacher_row_identity(row, row_index))
        duplicate_keys.append(
            speech_shared_transcript_duplicate_key(row, row_index)
        )
        if row.get("source_dataset"):
            source_datasets.add(str(row["source_dataset"]))

    expected_train_commitment = (
        strict_split_binding.get("speech_source_file", {})
        .get("partition_commitments", {})
        .get("partitions", {})
        .get("train")
    )
    actual_train_commitment = speech_partition_record(
        rows, "train", annotated=True
    )
    if (
        not isinstance(expected_train_commitment, dict)
        or actual_train_commitment != expected_train_commitment
    ):
        raise ValueError(
            "speech shared teacher-bank train row commitments mismatch strict split binding"
        )

    reusable_bank_identity: Optional[Dict[str, Any]] = None
    if reusable_bank is not None:
        if not reusable_bank_exact:
            raise ValueError(
                "speech shared reusable teacher bank must be declared exact"
            )
        reusable_bank_identity = tensor_content_identity(reusable_bank)
        if not _is_full_hex(reusable_bank_expected_sha256, 64):
            raise ValueError(
                "speech shared reusable teacher bank requires expected SHA256"
            )
        if reusable_bank_identity["sha256"] != reusable_bank_expected_sha256:
            raise ValueError("speech shared reusable teacher bank SHA256 mismatch")
        bank = reusable_bank.detach().float().to(device)
        bank_source = "reused_row_aligned_audio_raw_bank"
    else:
        if reusable_bank_expected_sha256 is not None:
            raise ValueError(
                "speech shared reusable bank SHA256 supplied without reusable bank"
            )
        was_training = bool(lm.training)
        lm.eval()
        try:
            bank = lm_text_embeddings(
                lm,
                tokenizer,
                [str(row["transcript"]) for row in rows],
                device,
                max_length,
                batch_size=build_batch_size,
            ).to(device)
        finally:
            lm.train(was_training)
        bank_source = "explicit_transcript_only_olmoe_final_hidden"
    if bank.ndim != 2 or bank.shape[0] != len(rows):
        raise RuntimeError(
            "speech shared contrastive teacher bank is not row-aligned with audio_train"
        )
    if bank.requires_grad:
        raise RuntimeError("speech shared contrastive teacher bank must be frozen")
    bank = F.normalize(bank.detach().float(), dim=-1)
    if not bool(torch.isfinite(bank).all().item()):
        raise FloatingPointError(
            "speech shared contrastive teacher bank contains non-finite values"
        )
    bank_tensor_identity = tensor_content_identity(bank)

    row_identity_sha256 = hashlib.sha256(
        json.dumps(
            row_identities, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    transcript_key_sha256 = hashlib.sha256(
        json.dumps(
            duplicate_keys, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if len(set(row_identities)) != len(row_identities):
        raise ValueError(
            "speech teacher-bank train rows have duplicate canonical identities"
        )
    provenance = {
        "bank_source": bank_source,
        "bank_rows": len(rows),
        "bank_order": "audio_train_row_order",
        "source_scope": "speech_train_partition_only",
        "source_partitions": ["train"],
        "source_datasets": sorted(source_datasets),
        "development_split_manifest": dict(
            strict_split_binding["development_split_manifest"]
        ),
        "speech_train_file": dict(strict_split_binding["speech_train_file"]),
        "speech_source_file": dict(strict_split_binding["speech_source_file"]),
        "actual_train_partition_commitment": actual_train_commitment,
        "teacher_embedding_batch_size": build_batch_size,
        "teacher_embedding_batch_size_source": "speech_teacher_bank_batch_size",
        "eval_batch_size_independent": True,
        "teacher_model_mode": "eval",
        "teacher_model_identity": dict(teacher_model_identity),
        "tokenizer_identity": dict(tokenizer_identity),
        "bank_tensor_identity": bank_tensor_identity,
        "reusable_bank_input_identity": reusable_bank_identity,
        "row_identity_semantics": SPEECH_SHARED_CONTRASTIVE_ROW_IDENTITY,
        "row_identity_sha256": row_identity_sha256,
        "unique_row_identities": len(set(row_identities)),
        "transcript_duplicate_key_semantics": "normalized_transcript_sha256",
        "transcript_duplicate_keys_sha256": transcript_key_sha256,
        "positive_selection": SPEECH_SHARED_CONTRASTIVE_POSITIVE_SELECTION,
        "duplicate_exclusion_policy": SPEECH_SHARED_CONTRASTIVE_DUPLICATE_POLICY,
        "sealed_evidence_used": False,
        "dev_partition_used": False,
        "eval_partition_used": False,
    }
    return bank, row_identities, duplicate_keys, provenance


def speech_shared_hidden_bank_infonce_loss(
    student_hidden: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    prefix_len: int,
    teacher_bank: torch.Tensor,
    positive_indices: Sequence[int],
    bank_row_identities: Sequence[str],
    bank_duplicate_keys: Sequence[str],
    temperature: float,
) -> Tuple[torch.Tensor, int, int]:
    """Match speech-conditioned hidden states to a full frozen teacher bank."""
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError(
            "speech shared contrastive temperature must be finite and positive"
        )
    attention_mask = batch["attention_mask"]
    batch_size, text_length = attention_mask.shape
    if student_hidden.ndim != 3 or student_hidden.shape[0] != batch_size:
        raise RuntimeError(
            "speech-prefix student hidden states do not align with the batch"
        )
    expected_student_length = int(prefix_len) + int(text_length)
    if student_hidden.shape[1] < expected_student_length:
        raise RuntimeError(
            "speech-prefix student hidden states are shorter than prefix plus text"
        )
    if teacher_bank.requires_grad:
        raise RuntimeError("speech shared contrastive teacher bank must be frozen")
    if teacher_bank.ndim != 2 or teacher_bank.shape[0] < 2:
        raise RuntimeError(
            "speech shared contrastive teacher bank requires at least two rows"
        )
    bank_size = int(teacher_bank.shape[0])
    if (
        len(positive_indices) != batch_size
        or len(bank_row_identities) != bank_size
        or len(bank_duplicate_keys) != bank_size
    ):
        raise RuntimeError(
            "speech shared contrastive row-index provenance does not align"
        )
    positive_index_values = [int(index) for index in positive_indices]
    if any(index < 0 or index >= bank_size for index in positive_index_values):
        raise IndexError("speech shared contrastive positive index is out of range")

    student_vectors = F.normalize(
        mean_pool(
            student_hidden[:, int(prefix_len) : expected_student_length].float(),
            attention_mask,
        ),
        dim=-1,
    )
    teacher_vectors = F.normalize(teacher_bank.detach().float(), dim=-1)
    if student_vectors.shape[1] != teacher_vectors.shape[1]:
        raise RuntimeError(
            "speech shared contrastive student and teacher dimensions differ"
        )
    logits = (student_vectors @ teacher_vectors.T) / float(temperature)
    excluded_count = 0
    for query_index, positive_index in enumerate(positive_index_values):
        positive_identity = bank_row_identities[positive_index]
        positive_duplicate_key = bank_duplicate_keys[positive_index]
        excluded = torch.tensor(
            [
                bank_index != positive_index
                and (
                    bank_row_identities[bank_index] == positive_identity
                    or bank_duplicate_keys[bank_index] == positive_duplicate_key
                )
                for bank_index in range(bank_size)
            ],
            dtype=torch.bool,
            device=logits.device,
        )
        if int((~excluded).sum().item()) < 2:
            raise RuntimeError(
                "speech shared contrastive query has no eligible teacher-bank negative"
            )
        logits[query_index, excluded] = torch.finfo(logits.dtype).min
        excluded_count += int(excluded.sum().item())
    labels = torch.tensor(
        positive_index_values, dtype=torch.long, device=logits.device
    )
    loss = F.cross_entropy(logits, labels)
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(
            "speech shared contrastive objective produced a non-finite loss"
        )
    return loss, int(batch_size), excluded_count


def text_embedding_targets(lm, tokenizer, texts: Sequence[str], device: torch.device, max_length: int) -> torch.Tensor:
    with torch.no_grad():
        return lm_text_embeddings(lm, tokenizer, texts, device, max_length, max(1, len(texts))).to(device)


def center_normalize(vectors: torch.Tensor, center: Optional[torch.Tensor] = None) -> torch.Tensor:
    vectors = vectors.float()
    if center is not None:
        vectors = vectors - center.to(device=vectors.device, dtype=vectors.dtype)
    return F.normalize(vectors, dim=-1)


def build_text_embedding_bank(lm, tokenizer, texts: Sequence[str], device: torch.device, max_length: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    bank = lm_text_embeddings(lm, tokenizer, texts, device, max_length, batch_size).to(device)
    center = bank.mean(dim=0, keepdim=True)
    return center_normalize(bank, center), center


def load_vision_text_tokenizer(vision_model_name: str):
    from transformers import AutoTokenizer

    return load_pretrained(AutoTokenizer, vision_model_name)


def load_speech_text_tokenizer(speech_model_name: str):
    from transformers import AutoTokenizer

    return load_pretrained(AutoTokenizer, speech_model_name)


def clip_text_embeddings(vision_model, vision_tokenizer, texts: Sequence[str], device: torch.device, batch_size: int) -> torch.Tensor:
    vectors = []
    for start in range(0, len(texts), batch_size):
        sub = list(texts[start:start + batch_size])
        encoded = vision_tokenizer(sub, return_tensors="pt", padding=True, truncation=True, max_length=77)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            if hasattr(vision_model, "get_text_features"):
                outputs = vision_model.get_text_features(**encoded)
            else:
                outputs = vision_model.text_model(**encoded)
            if torch.is_tensor(outputs):
                feats = outputs
            elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                feats = outputs.text_embeds
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0]
            projection = getattr(vision_model, "text_projection", None)
            if projection is not None and feats.shape[-1] != int(getattr(vision_model.config, "projection_dim", feats.shape[-1])):
                feats = projection(feats)
        vectors.append(feats.detach().float().cpu())
    return F.normalize(torch.cat(vectors, dim=0), dim=-1)


def image_alignment_text_embeddings(
    target: str,
    lm,
    tokenizer,
    vision_model,
    vision_tokenizer,
    texts: Sequence[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    if target == "clip_text":
        return clip_text_embeddings(vision_model, vision_tokenizer, texts, device, batch_size)
    if target == "olmoe_caption_hidden":
        return lm_text_embeddings(lm, tokenizer, texts, device, max_length, batch_size)
    raise ValueError(f"Unsupported image alignment target: {target}")


def speech_text_embeddings(speech_text_tokenizer, speech_model, texts: Sequence[str], device: torch.device, batch_size: int) -> torch.Tensor:
    tokenizer = speech_text_tokenizer
    decoder = getattr(speech_model, "decoder", None)
    if decoder is None and hasattr(speech_model, "model"):
        decoder = getattr(speech_model.model, "decoder", None)
    embed_tokens = getattr(decoder, "embed_tokens", None) if decoder is not None else None
    if embed_tokens is None:
        raise RuntimeError("Speech model does not expose decoder token embeddings for transcript retrieval targets")
    vectors = []
    for start in range(0, len(texts), batch_size):
        sub = list(texts[start:start + batch_size])
        encoded = tokenizer(sub, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
        with torch.no_grad():
            hidden = embed_tokens(input_ids)
        vectors.append(mean_pool(hidden.detach().float(), attention_mask).cpu())
    return F.normalize(torch.cat(vectors, dim=0), dim=-1)


def build_centered_bank(vectors: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    bank = vectors.to(device).float()
    center = bank.mean(dim=0, keepdim=True)
    return center_normalize(bank, center), center


def sample_negative_indices(total: int, positives: Sequence[int], count: int) -> List[int]:
    if total <= len(set(positives)) or count <= 0:
        return []
    excluded = set(int(idx) for idx in positives)
    target = min(int(count), total - len(excluded))
    negatives: List[int] = []
    while len(negatives) < target:
        idx = random.randrange(total)
        if idx in excluded:
            continue
        excluded.add(idx)
        negatives.append(idx)
    return negatives


def retrieval_contrastive_loss(left: torch.Tensor, positives: torch.Tensor, negatives: Optional[torch.Tensor], temperature: float) -> torch.Tensor:
    targets = positives if negatives is None or negatives.numel() == 0 else torch.cat([positives, negatives], dim=0)
    logits = (left @ targets.T) / max(float(temperature), 1e-6)
    labels = torch.arange(left.shape[0], dtype=torch.long, device=left.device)
    return F.cross_entropy(logits, labels)


def full_bank_contrastive_loss(left: torch.Tensor, bank: torch.Tensor, positive_indices: Sequence[int], temperature: float) -> torch.Tensor:
    logits = (left @ bank.T) / max(float(temperature), 1e-6)
    labels = torch.tensor([int(idx) for idx in positive_indices], dtype=torch.long, device=left.device)
    return F.cross_entropy(logits, labels)


def resolve_override(value: float, fallback: float) -> float:
    return float(value) if float(value) >= 0.0 else float(fallback)


def resolve_positive_override(value: float, fallback: float) -> float:
    return float(value) if float(value) >= 0.0 else float(fallback)


def resolve_int_override(value: int, fallback: int) -> int:
    value_i = int(value)
    return value_i if value_i >= -1 else int(fallback)


def recall_metrics(left: torch.Tensor, right: torch.Tensor, prefix: str) -> Dict[str, float]:
    sim = left @ right.T
    ranks = torch.argsort(sim, dim=1, descending=True)
    gold = torch.arange(sim.shape[0]).unsqueeze(1)
    out: Dict[str, float] = {}
    for k in (1, 5, 10):
        kk = min(k, sim.shape[1])
        out[f"{prefix}_r_at_{k}"] = float((ranks[:, :kk] == gold).any(dim=1).float().mean().item())
    out[f"{prefix}_mean_positive_cosine"] = float(sim.diag().mean().item())
    return out


def per_example_prefix_nll(logits: torch.Tensor, labels: torch.Tensor, prefix_len: int) -> torch.Tensor:
    prefix_labels = torch.full((labels.shape[0], prefix_len), -100, dtype=torch.long, device=labels.device)
    full_labels = torch.cat([prefix_labels, labels], dim=1)
    shift_logits = logits[:, :-1].float()
    shift_labels = full_labels[:, 1:]
    flat = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    mask = shift_labels != -100
    return (flat * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def token_weighted_prefix_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prefix_len: int,
) -> Tuple[torch.Tensor, int]:
    """Return pure LM cross-entropy and its explicit supervised-token denominator."""
    prefix_labels = torch.full((labels.shape[0], prefix_len), -100, dtype=torch.long, device=labels.device)
    full_labels = torch.cat([prefix_labels, labels], dim=1)
    shift_logits = logits[:, :-1].float()
    shift_labels = full_labels[:, 1:]
    token_count = int((shift_labels != -100).sum().item())
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return loss, token_count


def speech_behavior_kl_loss(
    lm,
    student_logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    prefix_len: int,
    coefficient: float,
    temperature: float,
) -> Tuple[torch.Tensor, int]:
    """Distill text-only OLMoE behavior into the aligned speech-prefix path."""
    if float(coefficient) <= 0.0:
        return student_logits.new_zeros((), dtype=torch.float32), 0
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("speech behavior KL temperature must be finite and positive")

    labels = batch["labels"]
    with torch.no_grad():
        teacher_outputs = lm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_router_logits=False,
            return_dict=True,
        )
    teacher_logits = teacher_outputs.logits
    text_length = int(labels.shape[1])
    if teacher_logits.shape[:2] != labels.shape:
        raise RuntimeError("text-only teacher logits do not align with teacher-forced labels")
    expected_student_length = int(prefix_len) + text_length
    if student_logits.shape[1] < expected_student_length:
        raise RuntimeError("speech-prefix student logits are shorter than prefix plus text labels")

    supervised_mask = labels[:, 1:] != -100
    token_count = int(supervised_mask.sum().item())
    if token_count <= 0:
        raise RuntimeError("speech behavior KL requires supervised transcript label positions")
    teacher_selected = teacher_logits[:, :-1][supervised_mask].float()
    student_selected = student_logits[
        :, int(prefix_len) : int(prefix_len) + text_length - 1
    ][supervised_mask].float()
    scaled_teacher_log_probs = F.log_softmax(
        teacher_selected / float(temperature), dim=-1
    )
    scaled_student_log_probs = F.log_softmax(
        student_selected / float(temperature), dim=-1
    )
    loss = F.kl_div(
        scaled_student_log_probs,
        scaled_teacher_log_probs,
        reduction="batchmean",
        log_target=True,
    ) * (float(temperature) ** 2)
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("speech behavior KL produced a non-finite loss")
    return loss, token_count


def image_group_id(row: Dict[str, Any]) -> str:
    """Return a stable identity shared by every caption row for one source image."""
    for key in ("content_sha256", "resized_content_sha256", "media_sha256"):
        value = row.get(key)
        if value not in (None, ""):
            digest = str(value).lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"image row has invalid {key}")
            return f"{key}:{digest}"
    source = str(row.get("source", ""))
    for key in ("source_image_id", "image_id", "coco_image_id", "image_uid"):
        value = row.get(key)
        if value not in (None, ""):
            payload = json.dumps(
                {"source": source, "field": key, "value": value},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            return f"source_image:sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
    image_path = row.get("image_path")
    if image_path not in (None, ""):
        normalized_path = Path(os.path.normpath(str(image_path))).as_posix()
        return f"image_path:{normalized_path}"
    raise ValueError("image row has no source image identity or image_path")


def hard_text_tokens(text: str) -> set[str]:
    tokens = []
    current: List[str] = []
    for char in str(text).lower():
        if char.isalnum():
            current.append(char)
        elif current:
            token = "".join(current)
            if len(token) >= 3:
                tokens.append(token)
            current = []
    if current:
        token = "".join(current)
        if len(token) >= 3:
            tokens.append(token)
    return set(tokens)


def hard_text_indices(
    records: Sequence[Dict[str, Any]],
    key: str,
    index: int,
    negatives: int,
    pool_size: int,
    excluded_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    total = len(records)
    if total <= 1 or negatives <= 0:
        return []
    target_tokens = hard_text_tokens(str(records[index][key]))
    generator = torch.Generator()
    generator.manual_seed(131071 + int(index) * 31 + int(negatives) * 7 + int(pool_size))
    if pool_size > 0 and pool_size < total:
        pool = torch.randperm(total, generator=generator)[:pool_size].tolist()
        pool.extend([(index + delta) % total for delta in range(-8, 9) if delta != 0])
    else:
        pool = list(range(total))
    scored: List[Tuple[float, int]] = []
    seen = set(excluded_indices or ())
    seen.add(index)
    for raw_idx in pool:
        cand_idx = int(raw_idx) % total
        if cand_idx in seen:
            continue
        seen.add(cand_idx)
        cand_tokens = hard_text_tokens(str(records[cand_idx][key]))
        overlap = len(target_tokens & cand_tokens)
        union = len(target_tokens | cand_tokens)
        jaccard = float(overlap / union) if union else 0.0
        proximity = 1.0 / (1.0 + abs(cand_idx - int(index)))
        scored.append((jaccard + 0.01 * overlap + 1e-6 * proximity, cand_idx))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in scored[:negatives]]


def local_candidate_indices(
    records: Sequence[Dict[str, Any]],
    key: str,
    index: int,
    negatives: int,
    mode: str = "stride",
    hard_pool_size: int = 512,
    modality: Optional[str] = None,
) -> List[int]:
    total = len(records)
    target_negatives = int(negatives)
    candidates = [int(index)]
    excluded = {int(index)}
    if modality == "image":
        positive_group = image_group_id(records[index])
        excluded = {
            row_index
            for row_index, row in enumerate(records)
            if image_group_id(row) == positive_group
        }
        available = total - len(excluded)
        if available < target_negatives:
            raise ValueError(
                "Not enough cross-image-group negatives for conditional ranking: "
                f"requested={target_negatives}, available={available}, "
                f"positive_group={positive_group}"
            )
    used = set(excluded)
    if mode == "hard_text":
        for neg_i in hard_text_indices(
            records,
            key,
            index,
            target_negatives,
            int(hard_pool_size),
            excluded,
        ):
            if neg_i in used:
                continue
            used.add(neg_i)
            candidates.append(int(neg_i))
            if len(candidates) >= target_negatives + 1:
                return candidates
        mode = "random"
    if mode == "random":
        generator = torch.Generator()
        generator.manual_seed(65537 + int(index) * 17 + target_negatives)
        for neg in torch.randperm(max(1, total), generator=generator).tolist():
            neg_i = int(neg)
            if neg_i in used:
                continue
            used.add(neg_i)
            candidates.append(int(neg_i))
            if len(candidates) >= target_negatives + 1:
                return candidates
        return candidates
    if mode != "stride":
        raise ValueError(f"Unsupported conditional ranking negative mode: {mode}")
    stride = 37
    offset = 17
    for j in range(target_negatives):
        neg = (index + offset + stride * j) % max(1, total)
        while neg in used and len(used) < total:
            neg = (neg + 1) % total
        used.add(neg)
        candidates.append(int(neg))
    return candidates


def local_candidate_texts(
    records: Sequence[Dict[str, Any]],
    key: str,
    index: int,
    negatives: int,
    mode: str = "stride",
    hard_pool_size: int = 512,
    modality: Optional[str] = None,
) -> List[str]:
    indices = local_candidate_indices(
        records, key, index, negatives, mode, hard_pool_size, modality
    )
    return [str(records[candidate_index][key]) for candidate_index in indices]


def conditional_query_identity(row: Dict[str, Any], modality: str, index: int) -> str:
    for key in ("uid", "source_uid", "image_uid", "utterance_id", "source_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{modality}:{value}"
    stable_fields = {
        key: str(row[key])
        for key in ("source", "image_path", "audio_path", "caption", "transcript", "speaker_id")
        if row.get(key) not in (None, "")
    }
    if stable_fields:
        payload = json.dumps(stable_fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{modality}:sha256:{digest}"
    return f"{modality}:index:{index}"


def query_permutation_seed(control_seed: int, query_identity: str) -> int:
    payload = f"{int(control_seed)}\0{query_identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def permute_candidates_for_query(
    candidate_indices: Sequence[int],
    gold_candidate_index: int,
    control_seed: int,
    query_identity: str,
) -> Tuple[List[int], List[int], int, int]:
    if not candidate_indices or int(gold_candidate_index) not in candidate_indices:
        raise ValueError("candidate set must contain the explicit gold candidate index")
    permutation_seed = query_permutation_seed(control_seed, query_identity)
    generator = torch.Generator()
    generator.manual_seed(permutation_seed)
    permutation = [
        int(value)
        for value in torch.randperm(len(candidate_indices), generator=generator).tolist()
    ]
    permuted = [int(candidate_indices[position]) for position in permutation]
    gold_index = permuted.index(int(gold_candidate_index))
    return permuted, permutation, int(gold_index), int(permutation_seed)


def tie_aware_nll_evidence(
    nll_scores: Sequence[float], gold_index: int, tie_epsilon: float
) -> Dict[str, Any]:
    if not nll_scores:
        raise ValueError("nll_scores must not be empty")
    if int(gold_index) < 0 or int(gold_index) >= len(nll_scores):
        raise ValueError("gold_index is outside nll_scores")
    if float(tie_epsilon) < 0.0:
        raise ValueError("tie_epsilon must be non-negative")
    scores = [float(value) for value in nll_scores]
    gold_nll = scores[int(gold_index)]
    negative_scores = [
        value for position, value in enumerate(scores) if position != int(gold_index)
    ]
    tie_count = sum(
        1 for value in negative_scores if abs(value - gold_nll) <= float(tie_epsilon)
    )
    strict_rank = sum(
        1 for value in negative_scores if value <= gold_nll + float(tie_epsilon)
    )
    best_nll = min(scores)
    best_indices = [
        position
        for position, value in enumerate(scores)
        if value <= best_nll + float(tie_epsilon)
    ]
    margin = min(negative_scores) - gold_nll if negative_scores else 0.0
    return {
        "strict_rank": int(strict_rank),
        "strict_r_at_1": float(strict_rank == 0),
        "reciprocal_rank": float(1.0 / (strict_rank + 1)),
        "gold_nll": float(gold_nll),
        "gold_nll_margin": float(margin),
        "gold_tie_count": int(tie_count),
        "has_gold_tie": bool(tie_count > 0),
        "best_candidate_indices": best_indices,
        "best_tie_count": int(max(0, len(best_indices) - 1)),
    }


def rank_metrics(
    ranks: Sequence[int],
    prefix: str,
    evidence: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, float]:
    total = max(1, len(ranks))
    out: Dict[str, float] = {}
    for k in (1, 2, 5):
        out[f"{prefix}_r_at_{k}"] = float(sum(1 for rank in ranks if rank < k) / total)
    out[f"{prefix}_mean_rank"] = float(sum(rank + 1 for rank in ranks) / total)
    if evidence is not None:
        if len(evidence) != len(ranks):
            raise ValueError("rank/evidence length mismatch")
        tie_count = sum(int(row["gold_tie_count"]) for row in evidence)
        tie_queries = sum(bool(row["has_gold_tie"]) for row in evidence)
        out.update({
            f"{prefix}_strict_r_at_1": float(
                sum(int(rank) == 0 for rank in ranks) / total
            ),
            f"{prefix}_mrr": float(
                sum(float(row["reciprocal_rank"]) for row in evidence) / total
            ),
            f"{prefix}_mean_gold_nll_margin": float(
                sum(float(row["gold_nll_margin"]) for row in evidence) / total
            ),
            f"{prefix}_tie_count": int(tie_count),
            f"{prefix}_tie_query_count": int(tie_queries),
            f"{prefix}_tie_rate": float(tie_queries / total),
        })
    return out


def score_image_candidates(wrapper, tokenizer, image_processor, vision_model, cache: FeatureCache, row: Dict[str, Any], candidates: Sequence[str], device: torch.device, args) -> List[float]:
    scores: List[float] = []
    feat = cache.image_batch(image_processor, vision_model, [row], device, args.encoder_feature_tokens)
    for start in range(0, len(candidates), args.conditional_batch_size):
        texts = list(candidates[start:start + args.conditional_batch_size])
        batch = tokenize_prompt_targets(tokenizer, ["Caption:"] * len(texts), texts, device, args.max_length)
        feats = feat.expand(len(texts), -1, -1).contiguous()
        with torch.no_grad():
            outputs = wrapper(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], image_features=feats)
            nll = per_example_prefix_nll(outputs.logits, batch["labels"], args.image_prefix_tokens)
        scores.extend((-nll).detach().float().cpu().tolist())
    return scores


def score_speech_candidates(wrapper, tokenizer, speech_processor, speech_model, cache: FeatureCache, row: Dict[str, Any], candidates: Sequence[str], device: torch.device, args) -> List[float]:
    scores: List[float] = []
    feat = cache.audio_batch(speech_processor, speech_model, [row], device, args.sample_rate, args.encoder_feature_tokens, args.audio_max_seconds)
    for start in range(0, len(candidates), args.conditional_batch_size):
        texts = list(candidates[start:start + args.conditional_batch_size])
        batch = tokenize_prompt_targets(tokenizer, ["Transcript:"] * len(texts), texts, device, args.max_length)
        feats = feat.expand(len(texts), -1, -1).contiguous()
        with torch.no_grad():
            outputs = wrapper(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], audio_features=feats)
            nll = per_example_prefix_nll(outputs.logits, batch["labels"], args.audio_prefix_tokens)
        scores.extend((-nll).detach().float().cpu().tolist())
    return scores


def router_z_loss_from_outputs(outputs) -> torch.Tensor:
    logits = outputs.router_logits
    if logits is None:
        return outputs.logits.new_zeros(())
    tensors = list(logits) if not torch.is_tensor(logits) else [logits]
    if not tensors:
        return outputs.logits.new_zeros(())
    losses = [torch.logsumexp(tensor.float(), dim=-1).pow(2).mean() for tensor in tensors]
    return torch.stack(losses).mean()


def mean_expert_dropout_ratio(model) -> float:
    ratios = []
    for layer in getattr(model.model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "last_expert_dropout_ratio"):
            ratios.append(float(getattr(mlp, "last_expert_dropout_ratio")))
    return float(sum(ratios) / len(ratios)) if ratios else 0.0


def update_dynamic_expert_bias(
    model,
    step: int,
    args,
    capacity_factor: float,
    dispatch_snapshot: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, float]:
    """Apply aux-loss-free expert-bias balancing from the latest routing counts."""
    lr = float(getattr(args, "dynamic_expert_bias_lr", 0.0))
    interval = max(1, int(getattr(args, "dynamic_expert_bias_update_interval", 1)))
    warmup = max(0, int(getattr(args, "dynamic_expert_bias_warmup_steps", 0)))
    max_abs = max(0.0, float(getattr(args, "dynamic_expert_bias_max_abs", 2.0)))
    apply_update = lr > 0.0 and step > warmup and step % interval == 0
    errors: List[torch.Tensor] = []
    inactive: List[float] = []
    overflow: List[float] = []
    updated = 0
    snapshot_by_layer = {
        int(row["layer"]): row for row in (dispatch_snapshot or []) if "layer" in row
    }
    with torch.no_grad():
        for layer_idx, layer in enumerate(getattr(model.model, "layers", [])):
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not bool(getattr(mlp, "dynamic_expert_bias_enabled", False)):
                continue
            bias = getattr(mlp, "expert_bias", None)
            snapshot_row = snapshot_by_layer.get(int(layer_idx), {})
            counts = snapshot_row.get("attempted_expert_counts")
            if counts is None:
                counts = getattr(mlp, "last_dynamic_expert_assigned_counts", None)
            if bias is None or counts is None:
                continue
            counts_f = counts.to(device=bias.device, dtype=torch.float32)
            target = counts_f.mean().clamp_min(1.0)
            rel_error = (counts_f - target) / target
            errors.append(rel_error.detach().abs().float().cpu())
            inactive.append(float((counts_f <= 0).float().mean().detach().cpu()))
            capacity = max(1.0, math.ceil(float(capacity_factor) * float(counts_f.sum().item()) / max(1, int(counts_f.numel()))))
            overflow_assignments = torch.clamp(counts_f - float(capacity), min=0.0).sum()
            overflow.append(float((overflow_assignments / counts_f.sum().clamp_min(1.0)).detach().cpu()))
            if apply_update:
                bias.add_(-lr * rel_error)
                if max_abs > 0.0:
                    bias.clamp_(min=-max_abs, max=max_abs)
                bias.sub_(bias.mean())
                updated += 1
    mean_error = float(torch.cat(errors).mean().item()) if errors else 0.0
    return {
        "dynamic_expert_bias_lr": lr,
        "dynamic_expert_bias_update_applied": float(updated > 0),
        "dynamic_expert_bias_updated_layers": float(updated),
        "dynamic_expert_bias_mean_abs_error": mean_error,
        "dynamic_expert_bias_inactive_proxy": float(sum(inactive) / len(inactive)) if inactive else 0.0,
        "dynamic_expert_bias_overflow_proxy": float(sum(overflow) / len(overflow)) if overflow else 0.0,
        **dynamic_expert_bias_metrics(model),
    }


def conditional_ranking_objective(
    wrapper,
    tokenizer,
    features: torch.Tensor,
    batch_indices: Sequence[int],
    records: Sequence[Dict[str, Any]],
    key: str,
    prompt: str,
    modality: str,
    prefix_len: int,
    device: torch.device,
    args,
) -> Tuple[torch.Tensor, float]:
    negatives = max(1, int(args.conditional_ranking_negatives))
    candidates_per_query = negatives + 1
    texts: List[str] = []
    repeated_features: List[torch.Tensor] = []
    for local_idx, record_idx in enumerate(batch_indices):
        candidates = local_candidate_texts(
            records,
            key,
            int(record_idx),
            negatives,
            args.conditional_ranking_negative_mode,
            args.conditional_ranking_hard_pool_size,
            modality,
        )
        texts.extend(candidates)
        repeated_features.append(features[local_idx:local_idx + 1].expand(len(candidates), -1, -1))
    feature_batch = torch.cat(repeated_features, dim=0).contiguous()
    batch = tokenize_prompt_targets(tokenizer, [prompt] * len(texts), texts, device, args.max_length)
    if modality == "image":
        outputs = wrapper(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], image_features=feature_batch)
    elif modality == "speech":
        outputs = wrapper(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], audio_features=feature_batch)
    else:
        raise ValueError(f"Unsupported conditional ranking modality: {modality}")
    nll = per_example_prefix_nll(outputs.logits, batch["labels"], prefix_len).view(len(batch_indices), candidates_per_query)
    ranking_logits = (-nll) / max(float(args.conditional_ranking_temperature), 1e-6)
    labels = torch.zeros(len(batch_indices), dtype=torch.long, device=device)
    ranking_loss = F.cross_entropy(ranking_logits, labels)
    ranking_accuracy = float((ranking_logits.argmax(dim=1) == 0).float().mean().detach().cpu())
    return ranking_loss, ranking_accuracy


def conditional_matching_eval(wrapper, tokenizer, image_processor, vision_model, speech_processor, speech_model, image_records, audio_records, device: torch.device, args, cache: FeatureCache) -> Dict[str, Any]:
    image_eval = list(image_records[: min(args.conditional_eval_samples, len(image_records))])
    audio_eval = list(audio_records[: min(args.conditional_eval_samples, len(audio_records))])
    negatives = max(1, int(args.conditional_negatives))
    control_seed = int(getattr(args, "conditional_control_seed", 42))
    tie_epsilon = float(getattr(args, "conditional_tie_epsilon", 1e-8))
    permutation_policy = str(
        getattr(args, "conditional_candidate_permutation", "query_identity_seeded")
    )
    if permutation_policy != "query_identity_seeded":
        raise ValueError(f"Unsupported conditional candidate permutation: {permutation_policy}")
    image_ranks: List[int] = []
    speech_ranks: List[int] = []
    image_evidence: List[Dict[str, Any]] = []
    speech_evidence: List[Dict[str, Any]] = []
    per_query_rows: List[Dict[str, Any]] = []
    for modality, records, key, scorer, ranks, evidence_rows in (
        ("image", image_eval, "caption", score_image_candidates, image_ranks, image_evidence),
        ("speech", audio_eval, "transcript", score_speech_candidates, speech_ranks, speech_evidence),
    ):
        for idx, row in enumerate(records):
            query_identity = conditional_query_identity(row, modality, idx)
            base_indices = local_candidate_indices(
                records, key, idx, negatives, modality=modality
            )
            candidate_indices, permutation, gold_index, permutation_seed = (
                permute_candidates_for_query(
                    base_indices, idx, control_seed, query_identity
                )
            )
            candidates = [str(records[candidate_index][key]) for candidate_index in candidate_indices]
            if modality == "image":
                scores = scorer(
                    wrapper, tokenizer, image_processor, vision_model, cache,
                    row, candidates, device, args,
                )
            else:
                scores = scorer(
                    wrapper, tokenizer, speech_processor, speech_model, cache,
                    row, candidates, device, args,
                )
            nll_scores = [-float(score) for score in scores]
            evidence = tie_aware_nll_evidence(nll_scores, gold_index, tie_epsilon)
            ranks.append(int(evidence["strict_rank"]))
            evidence_rows.append(evidence)
            candidate_ids = [
                conditional_query_identity(records[candidate_index], modality, candidate_index)
                for candidate_index in candidate_indices
            ]
            per_query_rows.append({
                "modality": modality,
                "query_uid": query_identity,
                "query_index": int(idx),
                "candidate_ids": candidate_ids,
                "candidate_indices": [int(value) for value in candidate_indices],
                "candidate_permutation": permutation,
                "candidate_permutation_seed": int(permutation_seed),
                "gold_index": int(gold_index),
                "gold_candidate_index": int(idx),
                "raw_nll_scores": nll_scores,
                "gold_nll_margin": float(evidence["gold_nll_margin"]),
                "gold_tie_count": int(evidence["gold_tie_count"]),
                "strict_rank": int(evidence["strict_rank"]),
                "control_provenance": {
                    "control_seed": control_seed,
                    "prefix_control": "real",
                    "eval_path": "shared_prefix",
                    "prefix_source_uid": query_identity,
                },
                "source_provenance": {
                    "query_source": str(row.get("source", "")),
                    "query_uid": query_identity,
                    "query_index": int(idx),
                },
            })
    candidates_per_query = negatives + 1
    return {
        "conditional_eval_path": "shared_olmoe_prefix_lm_nll",
        "conditional_uses_lm_logits": True,
        "conditional_uses_direct_encoder_pooling": False,
        "conditional_candidate_permutation_policy": permutation_policy,
        "conditional_candidate_permutation_seed_source": "control_seed+stable_query_identity_sha256",
        "conditional_control_seed": control_seed,
        "conditional_tie_policy": "strict_pessimistic_epsilon",
        "conditional_tie_epsilon": tie_epsilon,
        "conditional_candidates_per_query": int(candidates_per_query),
        "conditional_image_eval_count": len(image_eval),
        "conditional_speech_eval_count": len(audio_eval),
        "conditional_image_chance_r_at_1": float(1.0 / candidates_per_query),
        "conditional_speech_chance_r_at_1": float(1.0 / candidates_per_query),
        "conditional_per_query": per_query_rows,
        **rank_metrics(image_ranks, "conditional_image_to_text", image_evidence),
        **rank_metrics(speech_ranks, "conditional_speech_to_text", speech_evidence),
    }


def retrieval_eval(wrapper, tokenizer, image_processor, vision_model, vision_text_tokenizer, speech_processor, speech_model, speech_text_tokenizer, image_records, audio_records, device, args, cache: FeatureCache) -> Dict[str, Any]:
    wrapper.eval()
    lm = wrapper.lm
    image_eval = list(image_records[: min(args.retrieval_eval_samples, len(image_records))])
    audio_eval = list(audio_records[: min(args.retrieval_eval_samples, len(audio_records))])
    image_vectors = []
    for start in range(0, len(image_eval), args.eval_batch_size):
        recs = image_eval[start:start + args.eval_batch_size]
        feats = cache.image_batch(image_processor, vision_model, recs, device, args.encoder_feature_tokens)
        with torch.no_grad():
            image_vectors.append(wrapper.image_alignment_vector(feats).detach().float().cpu())
    image_emb = F.normalize(torch.cat(image_vectors, dim=0).float(), dim=-1)
    cap_emb = image_alignment_text_embeddings(
        args.image_alignment_target,
        lm,
        tokenizer,
        vision_model,
        vision_text_tokenizer,
        [str(row["caption"]) for row in image_eval],
        device,
        args.max_length,
        args.eval_batch_size,
    )
    cap_center = cap_emb.mean(dim=0, keepdim=True)
    image_emb = center_normalize(image_emb, cap_center)
    cap_emb = center_normalize(cap_emb, cap_center)

    speech_vectors = []
    for start in range(0, len(audio_eval), args.eval_batch_size):
        recs = audio_eval[start:start + args.eval_batch_size]
        feats = cache.audio_batch(speech_processor, speech_model, recs, device, args.sample_rate, args.encoder_feature_tokens, args.audio_max_seconds)
        with torch.no_grad():
            speech_vectors.append(wrapper.audio_alignment_vector(feats).detach().float().cpu())
    speech_emb = F.normalize(torch.cat(speech_vectors, dim=0).float(), dim=-1)
    if args.speech_target_space == "whisper_decoder_text":
        transcript_emb = speech_text_embeddings(speech_text_tokenizer, speech_model, [str(row["transcript"]) for row in audio_eval], device, args.eval_batch_size)
    else:
        transcript_emb = lm_text_embeddings(lm, tokenizer, [str(row["transcript"]) for row in audio_eval], device, args.max_length, args.eval_batch_size)
    if args.speech_target_space == "whisper_decoder_text":
        speech_retrieval_centering = "none_raw_normalized"
        speech_emb = F.normalize(speech_emb.float(), dim=-1)
        transcript_emb = F.normalize(transcript_emb.float(), dim=-1)
    else:
        speech_retrieval_centering = "target_mean_centered"
        transcript_center = transcript_emb.mean(dim=0, keepdim=True)
        speech_emb = center_normalize(speech_emb, transcript_center)
        transcript_emb = center_normalize(transcript_emb, transcript_center)

    conditional = conditional_matching_eval(wrapper, tokenizer, image_processor, vision_model, speech_processor, speech_model, image_eval, audio_eval, device, args, cache)
    metrics = {
        "retrieval_path": "shared_olmoe_prefix_hidden",
        "retrieval_uses_lm_hidden_states": True,
        "retrieval_uses_direct_encoder_pooling": False,
        "retrieval_alignment_prefix_residual": bool(args.alignment_prefix_residual),
        "image_target_embedding_space": str(args.image_alignment_target),
        "speech_target_embedding_space": str(args.speech_target_space),
        "speech_retrieval_centering": speech_retrieval_centering,
        "image_prefix_tokens": int(args.image_prefix_tokens),
        "audio_prefix_tokens": int(args.audio_prefix_tokens),
        "encoder_feature_tokens": int(args.encoder_feature_tokens),
        "image_eval_count": len(image_eval),
        "speech_eval_count": len(audio_eval),
        "image_chance_r_at_1": float(1.0 / max(1, len(image_eval))),
        "speech_chance_r_at_1": float(1.0 / max(1, len(audio_eval))),
        **recall_metrics(image_emb, cap_emb, "image_to_text"),
        **recall_metrics(cap_emb, image_emb, "text_to_image"),
        **recall_metrics(speech_emb, transcript_emb, "speech_to_text"),
        **recall_metrics(transcript_emb, speech_emb, "text_to_speech"),
        **conditional,
    }
    return metrics


def train_real_multimodal(
    exp_id: str,
    args,
    text_train_blocks: Sequence[Dict[str, Any]],
    text_eval_blocks: Sequence[Dict[str, Any]],
    image_train: Sequence[Dict[str, Any]],
    image_eval: Sequence[Dict[str, Any]],
    audio_train: Sequence[Dict[str, Any]],
    audio_eval: Sequence[Dict[str, Any]],
    gamma,
    aux_coef: float,
    capacity_factor: float,
    max_steps: int,
    out_dir: Path,
    train_router_gates: bool,
    train_experts: bool,
    initial_checkpoint_state: Optional[Dict[str, Any]] = None,
    initial_checkpoint_provenance: Optional[Dict[str, Any]] = None,
    initial_checkpoint_scope: str = "both",
    speech_initial_checkpoint_state: Optional[Dict[str, Any]] = None,
    speech_initial_checkpoint_provenance: Optional[Dict[str, Any]] = None,
    stage_b_checkpoint_state: Optional[Dict[str, Any]] = None,
    stage_b_checkpoint_provenance: Optional[Dict[str, Any]] = None,
    evaluate_after_training: bool = True,
    expert_selection_path: Optional[str | Path] = None,
    expert_selection_method: str = "ESFT-Gate",
    development_split_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if max_steps <= 0:
        raise ValueError(f"{exp_id} requires max_steps > 0; got {max_steps}")
    validate_speech_shared_contrastive_request(args)
    speech_shared_contrastive_meta = speech_shared_contrastive_provenance(args)
    speech_shared_contrastive_coef = float(
        speech_shared_contrastive_meta["coefficient"]
    )
    speech_shared_contrastive_temperature = float(
        speech_shared_contrastive_meta["temperature"]
    )
    teacher_bank_build_batch_size: Optional[int] = None
    if speech_shared_contrastive_coef > 0.0:
        teacher_bank_build_batch_size = speech_teacher_bank_batch_size(args)
    speech_shared_split_binding: Dict[str, Any] = {}
    if speech_shared_contrastive_coef > 0.0:
        speech_shared_split_binding = validate_speech_shared_split_binding(
            development_split_provenance, len(audio_train)
        )
    if speech_shared_contrastive_coef > 0.0 and (
        train_router_gates or train_experts
    ):
        raise ValueError(
            "speech shared contrastive loss requires frozen router/expert/LM parameters"
        )
    run_seed = int(args.seed)
    set_seed(run_seed)
    model, tokenizer, meta = load_model(
        args.base_model,
        2,
        aux_coef,
        gamma=gamma,
        capacity_factor=capacity_factor,
        expert_dropout_prob=args.expert_dropout_prob,
        dynamic_expert_bias=bool(args.dynamic_expert_bias_lr > 0.0),
        pre_routing_identity_fn=lambda loaded_model: base_model_identity(
            loaded_model, args.base_model
        ),
    )
    runtime_base_model_identity = meta.pop("pre_routing_model_identity", None)
    if not isinstance(runtime_base_model_identity, Mapping):
        raise ValueError("E3 pre-routing base-model identity is missing")
    device = next(model.parameters()).device
    image_processor, vision_model, speech_processor, speech_model = load_encoders(args.vision_model, args.speech_model, device)
    vision_text_tokenizer = load_vision_text_tokenizer(args.vision_model)
    speech_text_tokenizer = load_speech_text_tokenizer(args.speech_model)
    wrapper = make_wrapper(model, vision_model, speech_model, args).to(device)
    validate_initialization_state_disjoint(
        initial_checkpoint_state, stage_b_checkpoint_state, initial_checkpoint_scope
    )
    validate_dual_initialization_request(
        initial_checkpoint_state,
        initial_checkpoint_scope,
        speech_initial_checkpoint_state,
    )
    stage_b_initialization = dict(stage_b_checkpoint_provenance or {})
    if stage_b_checkpoint_state is not None:
        stage_b_initialization.update(
            restore_stage_b_student_initialization(
                wrapper,
                stage_b_checkpoint_state,
                str(args.base_model),
                runtime_base_model_identity,
            )
        )
        stage_b_initialization.setdefault("state_restored", True)
    selected_expert_ids: Dict[int, List[int]] = {}
    expert_selection_provenance: Dict[str, Any] = {}
    selected_expert_anchors: Dict[int, Dict[str, Any]] = {}
    selected_expert_hook_handles: List[Any] = []
    if expert_selection_path:
        selected_expert_update_capability(str(getattr(args, "expert_update_mode", "full")))
        if train_experts:
            raise ValueError("selected-expert training cannot enable full train_experts")
        if train_router_gates and not bool(
            getattr(args, "allow_selected_expert_router_tuning", False)
        ):
            raise ValueError("selected-expert A3 requires frozen router gates")
        if bool(getattr(args, "train_lm_head", False)):
            raise ValueError("selected-expert training requires a frozen LM head")
        if float(getattr(args, "expert_anchor_coefficient", 0.0)) <= 0.0:
            raise ValueError(
                "selected-expert weight anchoring requires a positive coefficient"
            )
        selected_expert_ids, expert_selection_provenance = load_prefix_expert_selection(
            expert_selection_path,
            expert_selection_method,
            num_layers=len(model.model.layers),
            num_experts=int(model.config.num_experts),
            expected_base_model=str(args.base_model),
        )
    initialization = dict(initial_checkpoint_provenance or {})
    if initial_checkpoint_state is not None:
        initialization.update(
            restore_scoped_multimodal_checkpoint(
                wrapper,
                initial_checkpoint_state,
                initial_checkpoint_scope,
                speech_model=speech_model,
                expected_selected_expert_ids=selected_expert_ids or None,
                expected_selection_provenance=expert_selection_provenance or None,
            )
        )
        initialization.setdefault("state_restored", True)
    speech_initialization = dict(speech_initial_checkpoint_provenance or {})
    if speech_initial_checkpoint_state is not None:
        speech_initialization.update(
            restore_speech_initialization_checkpoint(
                wrapper,
                speech_initial_checkpoint_state,
                speech_model,
            )
        )
        speech_initialization.setdefault("state_restored", True)
    optimizer, trainable_meta = configure_trainable(
        wrapper,
        train_router_gates,
        train_experts,
        args.train_lm_head,
        args.learning_rate,
        args.router_learning_rate,
        args.expert_learning_rate,
        args.retrieval_head_learning_rate,
        args.lm_head_learning_rate,
        args.weight_decay,
        speech_model=speech_model,
        speech_unfreeze_last_blocks=args.speech_unfreeze_last_blocks,
        speech_unfreeze_layer_norm=args.speech_unfreeze_layer_norm,
        speech_encoder_lr=args.speech_encoder_learning_rate,
    )
    trainable_meta["stage_b_initialization"] = stage_b_initialization
    trainable_meta["multimodal_initialization"] = initialization
    trainable_meta["speech_initialization"] = speech_initialization
    if selected_expert_ids:
        selected_optimizer, selected_expert_anchors, selected_expert_hook_handles, selected_meta = (
            configure_selected_full_expert_training(
                wrapper.lm,
                selected_expert_ids,
                expert_learning_rate=float(args.expert_learning_rate),
                anchor_coefficient=float(getattr(args, "expert_anchor_coefficient", 0.0)),
            )
        )
        for layer_idx, anchor_entry in selected_expert_anchors.items():
            experts = wrapper.lm.model.layers[layer_idx].mlp.experts
            for parameter_name in ("gate_up_proj", "down_proj"):
                parameter = getattr(experts, parameter_name)
                anchor_entry[parameter_name] = anchor_entry[parameter_name].to(
                    device=parameter.device, dtype=parameter.dtype
                )
        optimizer = CombinedOptimizer(optimizer, selected_optimizer)
        logical_selected_params = 0
        packed_backing_params = 0
        for layer_idx, layer in enumerate(wrapper.lm.model.layers):
            count = len(selected_expert_ids[layer_idx])
            for parameter_name in ("gate_up_proj", "down_proj"):
                parameter = getattr(layer.mlp.experts, parameter_name)
                logical_selected_params += count * int(parameter[0].numel())
                packed_backing_params += int(parameter.numel())
        trainable_meta = {
            **trainable_meta,
            **selected_meta,
            "selected_expert_training": True,
            "train_experts": False,
            "full_expert_bank_training": False,
            "selected_full_expert_parameter_count": logical_selected_params,
            "packed_backing_parameter_count": packed_backing_params,
            "trainable_params": int(trainable_meta["trainable_params"]) + logical_selected_params,
            "expert_selection_provenance": expert_selection_provenance,
            "nonselected_update_invariant": "gradient_masked_and_absent_from_weight_decay_optimizer",
        }
        trainable_meta["optimizer_groups"] = list(trainable_meta["optimizer_groups"]) + [{
            "name": "selected_full_experts",
            "lr": float(args.expert_learning_rate),
            "weight_decay": 0.0,
            "trainable_params": logical_selected_params,
        }]
    else:
        trainable_meta["selected_expert_training"] = False
    if args.alignment_pretrain_steps > 0:
        validate_alignment_pretrain_trainable(wrapper, speech_model)
    cache_root = Path(args.feature_cache_dir) if args.feature_cache_dir else out_dir / "feature_cache"
    cache = FeatureCache(
        cache_root,
        audio_encoder_changed=bool(
            (
                initial_checkpoint_state
                and initial_checkpoint_state.get("speech_encoder_trainable_state")
            )
            or speech_initial_checkpoint_state
        ),
        strict_audio_integrity=bool(
            development_split_provenance
            and development_split_provenance.get("strict_manifest_verified") is True
        ),
        speech_audio_data_dir=(
            Path(str(development_split_provenance["data_dir"]))
            if development_split_provenance
            and development_split_provenance.get("strict_manifest_verified") is True
            else None
        ),
    )
    expected_cache_policy = getattr(
        args,
        "speech_feature_cache_policy",
        DIAGNOSTIC_SPEECH_FEATURE_CACHE_POLICY,
    )
    if cache.speech_feature_cache_policy != expected_cache_policy:
        raise ValueError(
            "speech feature cache policy disagrees with verified split provenance"
        )
    trainable_meta["speech_feature_cache_policy"] = cache.speech_feature_cache_policy
    image_bank = image_bank_center = image_raw_bank = None
    audio_bank = audio_bank_center = audio_raw_bank = None
    speech_shared_teacher_bank = None
    speech_shared_teacher_row_identities: List[str] = []
    speech_shared_teacher_duplicate_keys: List[str] = []
    speech_shared_teacher_bank_provenance: Dict[str, Any] = {}
    needs_contrastive_bank = needs_alignment_target_bank(args)
    if needs_contrastive_bank:
        wrapper.lm.eval()
        print(json.dumps({"experiment_id": exp_id, "stage": "build_text_embedding_bank", "image_texts": len(image_train), "audio_texts": len(audio_train), "contrastive_negatives": args.contrastive_negatives, "image_contrastive_negatives": args.image_contrastive_negatives, "speech_contrastive_negatives": args.speech_contrastive_negatives}, sort_keys=True))
        image_raw_bank = image_alignment_text_embeddings(
            args.image_alignment_target,
            wrapper.lm,
            tokenizer,
            vision_model,
            vision_text_tokenizer,
            [str(row["caption"]) for row in image_train],
            device,
            args.max_length,
            args.eval_batch_size,
        ).to(device)
        if args.speech_target_space == "whisper_decoder_text":
            audio_raw_bank = speech_text_embeddings(speech_text_tokenizer, speech_model, [str(row["transcript"]) for row in audio_train], device, args.eval_batch_size).to(device)
        else:
            audio_text_bank_batch_size = (
                teacher_bank_build_batch_size
                if teacher_bank_build_batch_size is not None
                else args.eval_batch_size
            )
            audio_raw_bank = lm_text_embeddings(
                wrapper.lm, tokenizer,
                [str(row["transcript"]) for row in audio_train],
                device, args.max_length, audio_text_bank_batch_size,
            ).to(device)
        image_bank, image_bank_center = build_centered_bank(image_raw_bank, device)
        audio_bank, audio_bank_center = build_centered_bank(audio_raw_bank, device)
        print(json.dumps({"experiment_id": exp_id, "stage": "built_text_embedding_bank", "image_bank": list(image_bank.shape), "audio_bank": list(audio_bank.shape)}, sort_keys=True))
    if speech_shared_contrastive_coef > 0.0:
        reusable_bank_exact = bool(
            audio_raw_bank is not None
            and args.speech_target_space == "olmoe_text_hidden"
        )
        if teacher_bank_build_batch_size is None:
            raise RuntimeError("speech teacher bank batch size was not initialized")
        teacher_model_identity, tokenizer_identity = (
            speech_shared_teacher_runtime_identities(
                wrapper.lm,
                tokenizer,
                str(args.base_model),
                multimodal_initialization=initialization,
                stage_b_initialization=stage_b_initialization,
            )
        )
        reusable_bank_expected_sha256 = (
            tensor_content_identity(audio_raw_bank)["sha256"]
            if reusable_bank_exact
            else None
        )
        (
            speech_shared_teacher_bank,
            speech_shared_teacher_row_identities,
            speech_shared_teacher_duplicate_keys,
            speech_shared_teacher_bank_provenance,
        ) = build_speech_shared_teacher_bank(
            wrapper.lm,
            tokenizer,
            audio_train,
            device,
            args.max_length,
            strict_split_binding=speech_shared_split_binding,
            teacher_model_identity=teacher_model_identity,
            tokenizer_identity=tokenizer_identity,
            build_batch_size=teacher_bank_build_batch_size,
            reusable_bank=audio_raw_bank if reusable_bank_exact else None,
            reusable_bank_exact=reusable_bank_exact,
            reusable_bank_expected_sha256=reusable_bank_expected_sha256,
        )
        speech_shared_contrastive_meta = {
            **speech_shared_contrastive_meta,
            "teacher_bank": speech_shared_teacher_bank_provenance,
        }
        trainable_meta = {
            **trainable_meta,
            "speech_shared_contrastive_provenance": (
                speech_shared_contrastive_meta
            ),
            "speech_shared_teacher_bank_provenance": (
                speech_shared_teacher_bank_provenance
            ),
        }
        print(
            json.dumps(
                {
                    "experiment_id": exp_id,
                    "stage": "built_speech_shared_teacher_bank",
                    **speech_shared_teacher_bank_provenance,
                },
                sort_keys=True,
            )
        )
    log_path = out_dir / exp_id / "train_metrics.jsonl"
    if log_path.exists():
        log_path.unlink()
    pretrain_log_path = out_dir / exp_id / "alignment_pretrain_metrics.jsonl"
    if pretrain_log_path.exists():
        pretrain_log_path.unlink()
    rows: List[Dict[str, Any]] = []
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    wrapper.train()
    speech_model.train(bool(trainable_meta.get("speech_encoder_trainable")))
    if args.alignment_pretrain_steps > 0:
        alignment_modalities = parse_alignment_pretrain_modalities(
            args.alignment_pretrain_modalities
        )
        alignment_rows_by_modality = {
            "image": image_train,
            "speech": audio_train,
        }
        alignment_data_cursors = {
            modality: 0 for modality in set(alignment_modalities)
        }
        for pre_step in range(1, args.alignment_pretrain_steps + 1):
            mode = alignment_modalities[(pre_step - 1) % len(alignment_modalities)]
            optimizer.zero_grad(set_to_none=True)
            batch_indices, batch_rows, cursor_provenance = (
                sample_next_modality_batch(
                    alignment_rows_by_modality[mode],
                    mode,
                    alignment_data_cursors,
                    args.train_batch_size,
                )
            )
            if mode == "image":
                feats = cache.image_batch(image_processor, vision_model, batch_rows, device, args.encoder_feature_tokens)
                prefix_vec = wrapper.image_alignment_vector(feats)
                raw_prefix_vec = prefix_vec
                prefix_vec = center_normalize(prefix_vec, image_bank_center)
                target_vec = image_bank[batch_indices]
                raw_target_vec = image_raw_bank[batch_indices] if image_raw_bank is not None else None
                coef = args.image_contrastive_coef if args.image_contrastive_coef > 0.0 else args.contrastive_coef
                temp = resolve_override(args.image_contrastive_temperature, args.contrastive_temperature)
                neg_count = resolve_int_override(args.image_contrastive_negatives, args.contrastive_negatives)
                center_weight = resolve_positive_override(args.image_center_positive_weight, args.center_positive_weight)
                raw_weight = resolve_positive_override(args.image_raw_positive_weight, args.raw_positive_weight)
                total_count = len(image_train)
                bank = image_bank
            else:
                feats = cache.audio_batch(speech_processor, speech_model, batch_rows, device, args.sample_rate, args.encoder_feature_tokens, args.audio_max_seconds)
                prefix_vec = wrapper.audio_alignment_vector(feats)
                raw_prefix_vec = prefix_vec
                prefix_vec = center_normalize(prefix_vec, audio_bank_center)
                target_vec = audio_bank[batch_indices]
                raw_target_vec = audio_raw_bank[batch_indices] if audio_raw_bank is not None else None
                coef = args.speech_contrastive_coef if args.speech_contrastive_coef > 0.0 else args.contrastive_coef
                temp = resolve_override(args.speech_contrastive_temperature, args.contrastive_temperature)
                neg_count = resolve_int_override(args.speech_contrastive_negatives, args.contrastive_negatives)
                center_weight = resolve_positive_override(args.speech_center_positive_weight, args.center_positive_weight)
                raw_weight = resolve_positive_override(args.speech_raw_positive_weight, args.raw_positive_weight)
                total_count = len(audio_train)
                bank = audio_bank
            if neg_count < 0 or neg_count >= total_count - len(set(batch_indices)):
                contrastive = full_bank_contrastive_loss(prefix_vec, bank, batch_indices, temp)
            else:
                neg_idx = sample_negative_indices(total_count, batch_indices, neg_count)
                negative_vec = bank[neg_idx] if neg_idx else None
                contrastive = retrieval_contrastive_loss(prefix_vec, target_vec, negative_vec, temp)
            center_cos = (prefix_vec * target_vec).sum(dim=-1).mean()
            if raw_target_vec is not None:
                raw_cos = (raw_prefix_vec * raw_target_vec).sum(dim=-1).mean()
            else:
                raw_cos = center_cos
            positive = (center_weight * (1.0 - center_cos)) + (raw_weight * (1.0 - raw_cos))
            anchor_loss = (
                selected_expert_anchor_loss(
                    wrapper.lm,
                    selected_expert_anchors,
                    coefficient=float(getattr(args, "expert_anchor_coefficient", 0.0)),
                )
                if selected_expert_anchors
                else contrastive.new_zeros(())
            )
            loss = float(coef) * (contrastive + positive) + anchor_loss
            loss.backward()
            gradient_norms = bridge_grad_norms(wrapper)
            torch.nn.utils.clip_grad_norm_(trainable_optimization_parameters(wrapper, speech_model), args.grad_clip)
            optimizer.step()
            row = {
                "experiment_id": exp_id,
                "stage": "alignment_pretrain",
                "step": pre_step,
                "modality": mode,
                **cursor_provenance,
                "loss": float(loss.detach().float().cpu()),
                "loss_equation": "contrastive_alignment + selected_expert_weight_anchor",
                "selected_expert_anchor_loss": float(anchor_loss.detach().float().cpu()),
                "selected_expert_training": bool(selected_expert_anchors),
                "contrastive_loss": float(contrastive.detach().float().cpu()),
                "contrastive_coef": float(coef),
                "contrastive_temperature": float(temp),
                "contrastive_negatives": int(neg_count),
                "center_positive_weight": float(center_weight),
                "raw_positive_weight": float(raw_weight),
                "contrastive_positive_cosine": float(center_cos.detach().float().cpu()),
                "raw_contrastive_positive_cosine": float(raw_cos.detach().float().cpu()),
                **gradient_norms,
                **cuda_metrics(),
            }
            append_jsonl(pretrain_log_path, row)
            if pre_step == 1 or pre_step % args.alignment_pretrain_log_every == 0 or pre_step == args.alignment_pretrain_steps:
                print(json.dumps(row, sort_keys=True))

    modality_cycle = [item.strip() for item in str(args.modality_cycle).split(",") if item.strip()]
    invalid_modes = sorted(set(modality_cycle) - {"text", "image", "speech"})
    if not modality_cycle or invalid_modes:
        raise ValueError(f"Invalid modality cycle {args.modality_cycle!r}; expected comma-separated text,image,speech entries")

    training_rows_by_modality = {
        "text": text_train_blocks,
        "image": image_train,
        "speech": audio_train,
    }
    modality_data_cursors = {modality: 0 for modality in set(modality_cycle)}

    for step in range(1, max_steps + 1):
        mode = modality_cycle[(step - 1) % len(modality_cycle)]
        optimizer.zero_grad(set_to_none=True)
        batch_indices, batch_rows, cursor_provenance = sample_next_modality_batch(
            training_rows_by_modality[mode],
            mode,
            modality_data_cursors,
            args.train_batch_size,
        )
        if mode == "text":
            batch = tensorize_blocks(batch_rows, device, args.max_length, pad_id)
            outputs = wrapper.lm(**batch, output_router_logits=True, return_dict=True)
            text_len = int(batch["input_ids"].shape[1])
            image_tokens = 0
            audio_tokens = 0
            task_names = sorted({str(row.get("task", "text")) for row in batch_rows})
        elif mode == "image":
            prompts = ["Caption:" for _ in batch_rows]
            targets = [str(row["caption"]) for row in batch_rows]
            batch = tokenize_prompt_targets(tokenizer, prompts, targets, device, args.max_length)
            feats = cache.image_batch(image_processor, vision_model, batch_rows, device, args.encoder_feature_tokens)
            outputs = wrapper(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], image_features=feats)
            prefix_for_contrastive = wrapper.image_prefix(feats)
            target_texts_for_contrastive = targets
            text_len = int(batch["input_ids"].shape[1])
            image_tokens = int(args.image_prefix_tokens)
            audio_tokens = 0
            task_names = ["image_caption"]
        else:
            prompts = ["Transcript:" for _ in batch_rows]
            targets = [str(row["transcript"]) for row in batch_rows]
            batch = tokenize_prompt_targets(tokenizer, prompts, targets, device, args.max_length)
            feats = cache.audio_batch(speech_processor, speech_model, batch_rows, device, args.sample_rate, args.encoder_feature_tokens, args.audio_max_seconds)
            outputs = wrapper(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                audio_features=feats,
                output_hidden_states=speech_shared_contrastive_coef > 0.0,
            )
            prefix_for_contrastive = wrapper.audio_prefix(feats)
            target_texts_for_contrastive = targets
            text_len = int(batch["input_ids"].shape[1])
            image_tokens = 0
            audio_tokens = int(args.audio_prefix_tokens)
            task_names = ["speech_transcript"]
        dispatch_snapshot = capture_router_dispatch(model)
        prefix_len = int(image_tokens + audio_tokens)
        hf_reported_loss = outputs.loss
        ce_loss, supervised_token_count = token_weighted_prefix_ce(
            outputs.logits,
            batch["labels"],
            prefix_len,
        )
        aux_loss_raw = outputs.aux_loss if outputs.aux_loss is not None else ce_loss.new_zeros(())
        aux_loss_weighted = float(aux_coef) * aux_loss_raw
        explicit_base_loss = ce_loss + aux_loss_weighted
        hf_loss_gap = hf_reported_loss - explicit_base_loss
        contrastive_loss = ce_loss.new_zeros(())
        contrastive_positive_cosine = ce_loss.new_zeros(())
        raw_contrastive_positive_cosine = ce_loss.new_zeros(())
        modality_contrastive_coef = 0.0
        contrastive_temperature = float(args.contrastive_temperature)
        contrastive_negatives = int(args.contrastive_negatives)
        center_positive_weight = float(args.center_positive_weight)
        raw_positive_weight = float(args.raw_positive_weight)
        if mode == "image":
            modality_contrastive_coef = args.image_contrastive_coef if args.image_contrastive_coef > 0.0 else args.contrastive_coef
            contrastive_temperature = resolve_override(args.image_contrastive_temperature, args.contrastive_temperature)
            contrastive_negatives = resolve_int_override(args.image_contrastive_negatives, args.contrastive_negatives)
            center_positive_weight = resolve_positive_override(args.image_center_positive_weight, args.center_positive_weight)
            raw_positive_weight = resolve_positive_override(args.image_raw_positive_weight, args.raw_positive_weight)
        elif mode == "speech":
            modality_contrastive_coef = args.speech_contrastive_coef if args.speech_contrastive_coef > 0.0 else args.contrastive_coef
            contrastive_temperature = resolve_override(args.speech_contrastive_temperature, args.contrastive_temperature)
            contrastive_negatives = resolve_int_override(args.speech_contrastive_negatives, args.contrastive_negatives)
            center_positive_weight = resolve_positive_override(args.speech_center_positive_weight, args.center_positive_weight)
            raw_positive_weight = resolve_positive_override(args.speech_raw_positive_weight, args.raw_positive_weight)
        if mode != "text" and modality_contrastive_coef > 0.0:
            if mode == "image":
                prefix_vec = wrapper.image_alignment_vector(feats)
                raw_prefix_vec = prefix_vec
                raw_target_vec = None
                if image_bank is not None and image_bank_center is not None:
                    prefix_vec = center_normalize(prefix_vec, image_bank_center)
                    target_vec = image_bank[batch_indices]
                    raw_target_vec = image_raw_bank[batch_indices] if image_raw_bank is not None else None
                    if contrastive_negatives < 0 or contrastive_negatives >= len(image_train) - len(set(batch_indices)):
                        contrastive_loss = full_bank_contrastive_loss(prefix_vec, image_bank, batch_indices, contrastive_temperature)
                    else:
                        neg_idx = sample_negative_indices(len(image_train), batch_indices, contrastive_negatives)
                        negative_vec = image_bank[neg_idx] if neg_idx else None
                        contrastive_loss = retrieval_contrastive_loss(prefix_vec, target_vec, negative_vec, contrastive_temperature)
                else:
                    target_vec = clip_text_embeddings(vision_model, vision_text_tokenizer, target_texts_for_contrastive, device, args.eval_batch_size).to(device)
                    raw_target_vec = target_vec
                    contrastive_loss = symmetric_contrastive_loss(prefix_vec, target_vec, contrastive_temperature)
            else:
                prefix_vec = wrapper.audio_alignment_vector(feats)
                raw_prefix_vec = prefix_vec
                raw_target_vec = None
                if audio_bank is not None and audio_bank_center is not None:
                    prefix_vec = center_normalize(prefix_vec, audio_bank_center)
                    target_vec = audio_bank[batch_indices]
                    raw_target_vec = audio_raw_bank[batch_indices] if audio_raw_bank is not None else None
                    if contrastive_negatives < 0 or contrastive_negatives >= len(audio_train) - len(set(batch_indices)):
                        contrastive_loss = full_bank_contrastive_loss(prefix_vec, audio_bank, batch_indices, contrastive_temperature)
                    else:
                        neg_idx = sample_negative_indices(len(audio_train), batch_indices, contrastive_negatives)
                        negative_vec = audio_bank[neg_idx] if neg_idx else None
                        contrastive_loss = retrieval_contrastive_loss(prefix_vec, target_vec, negative_vec, contrastive_temperature)
                else:
                    target_vec = text_embedding_targets(wrapper.lm, tokenizer, target_texts_for_contrastive, device, args.max_length).to(device)
                    raw_target_vec = target_vec
                    contrastive_loss = symmetric_contrastive_loss(prefix_vec, target_vec, contrastive_temperature)
            contrastive_positive_cosine = (prefix_vec * target_vec).sum(dim=-1).mean()
            raw_contrastive_positive_cosine = (raw_prefix_vec * raw_target_vec).sum(dim=-1).mean() if raw_target_vec is not None else contrastive_positive_cosine
            positive_loss = (center_positive_weight * (1.0 - contrastive_positive_cosine)) + (raw_positive_weight * (1.0 - raw_contrastive_positive_cosine))
            contrastive_loss = contrastive_loss + positive_loss
        conditional_ranking_loss = ce_loss.new_zeros(())
        conditional_ranking_accuracy = 0.0
        modality_conditional_ranking_coef = 0.0
        if mode == "image":
            modality_conditional_ranking_coef = float(args.image_conditional_ranking_coef)
        elif mode == "speech":
            modality_conditional_ranking_coef = float(args.speech_conditional_ranking_coef)
        if mode == "image" and modality_conditional_ranking_coef > 0.0:
            conditional_ranking_loss, conditional_ranking_accuracy = conditional_ranking_objective(
                wrapper, tokenizer, feats, batch_indices, image_train, "caption", "Caption:", "image", args.image_prefix_tokens, device, args
            )
        elif mode == "speech" and modality_conditional_ranking_coef > 0.0:
            conditional_ranking_loss, conditional_ranking_accuracy = conditional_ranking_objective(
                wrapper, tokenizer, feats, batch_indices, audio_train, "transcript", "Transcript:", "speech", args.audio_prefix_tokens, device, args
            )
        speech_behavior_loss = ce_loss.new_zeros((), dtype=torch.float32)
        speech_behavior_token_count = 0
        if mode == "speech":
            speech_behavior_loss, speech_behavior_token_count = speech_behavior_kl_loss(
                wrapper.lm,
                outputs.logits,
                batch,
                prefix_len,
                args.speech_behavior_kl_coef,
                args.speech_behavior_kl_temperature,
            )
        speech_shared_contrastive_loss = ce_loss.new_zeros((), dtype=torch.float32)
        speech_shared_contrastive_query_count = 0
        speech_shared_contrastive_excluded_count = 0
        speech_shared_contrastive_positive_indices: List[int] = []
        speech_shared_contrastive_positive_row_identities: List[str] = []
        if mode == "speech" and speech_shared_contrastive_coef > 0.0:
            hidden_states = getattr(outputs, "hidden_states", None)
            student_final_hidden = hidden_states[-1] if hidden_states else None
            if student_final_hidden is None or speech_shared_teacher_bank is None:
                raise RuntimeError(
                    "speech shared contrastive path is missing hidden states or teacher bank"
                )
            speech_shared_contrastive_positive_indices = [
                int(index) for index in batch_indices
            ]
            speech_shared_contrastive_positive_row_identities = [
                speech_shared_teacher_row_identities[index]
                for index in speech_shared_contrastive_positive_indices
            ]
            (
                speech_shared_contrastive_loss,
                speech_shared_contrastive_query_count,
                speech_shared_contrastive_excluded_count,
            ) = speech_shared_hidden_bank_infonce_loss(
                student_final_hidden,
                batch,
                prefix_len,
                speech_shared_teacher_bank,
                speech_shared_contrastive_positive_indices,
                speech_shared_teacher_row_identities,
                speech_shared_teacher_duplicate_keys,
                speech_shared_contrastive_temperature,
            )
        router_z_loss = router_z_loss_from_outputs(outputs)
        anchor_loss = (
            selected_expert_anchor_loss(
                wrapper.lm,
                selected_expert_anchors,
                coefficient=float(getattr(args, "expert_anchor_coefficient", 0.0)),
            )
            if selected_expert_anchors
            else explicit_base_loss.new_zeros(())
        )
        loss = (
            explicit_base_loss
            + float(modality_contrastive_coef) * contrastive_loss
            + float(modality_conditional_ranking_coef) * conditional_ranking_loss
            + float(args.speech_behavior_kl_coef) * speech_behavior_loss
            + speech_shared_contrastive_coef * speech_shared_contrastive_loss
            + float(args.router_z_loss_coef) * router_z_loss
            + anchor_loss
        )
        optimizer_step = bool(loss.requires_grad)
        if optimizer_step:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_optimization_parameters(wrapper, speech_model), args.grad_clip)
            optimizer.step()
        dynamic_bias = update_dynamic_expert_bias(
            model,
            step,
            args,
            capacity_factor,
            dispatch_snapshot=dispatch_snapshot,
        )
        routing = router_metrics(
            outputs,
            2,
            int(model.config.num_experts),
            capacity_factor,
            dispatch_snapshot=dispatch_snapshot,
        )
        row = {
            "experiment_id": exp_id,
            "step": step,
            "modality": mode,
            "tasks": task_names,
            **cursor_provenance,
            **meta,
            "capacity_factor": capacity_factor,
            "aux_coef": aux_coef,
            **trainable_meta,
            "real_subset": True,
            "vision_model": args.vision_model,
            "speech_model": args.speech_model,
            "loss": float(loss.detach().float().cpu()),
            "loss_equation": "lm_ce_loss + aux_coef * router_aux_loss_raw + modality_losses + speech_behavior_kl_coef * speech_behavior_kl_loss + speech_shared_contrastive_coef * speech_shared_contrastive_loss + router_z_loss + selected_expert_weight_anchor",
            "selected_expert_anchor_loss": float(anchor_loss.detach().float().cpu()),
            "selected_expert_anchor_coefficient": float(getattr(args, "expert_anchor_coefficient", 0.0)),
            "ce_loss": float(ce_loss.detach().float().cpu()),
            "lm_ce_loss": float(ce_loss.detach().float().cpu()),
            "supervised_token_count": int(supervised_token_count),
            "hf_reported_loss": float(hf_reported_loss.detach().float().cpu()),
            "router_aux_loss_raw": float(aux_loss_raw.detach().float().cpu()),
            "router_aux_loss_weighted": float(aux_loss_weighted.detach().float().cpu()),
            "hf_reported_loss_minus_explicit_base": float(hf_loss_gap.detach().float().cpu()),
            "contrastive_loss": float(contrastive_loss.detach().float().cpu()),
            "contrastive_coef": float(modality_contrastive_coef),
            "conditional_ranking_loss": float(conditional_ranking_loss.detach().float().cpu()),
            "conditional_ranking_coef": float(modality_conditional_ranking_coef),
            "conditional_ranking_accuracy": float(conditional_ranking_accuracy),
            "speech_behavior_kl_loss": float(speech_behavior_loss.detach().float().cpu()),
            "speech_behavior_kl_coef": float(args.speech_behavior_kl_coef),
            "speech_behavior_kl_temperature": float(args.speech_behavior_kl_temperature),
            "speech_behavior_kl_token_count": int(speech_behavior_token_count),
            "speech_behavior_teacher_path": SPEECH_BEHAVIOR_TEACHER_PATH,
            "speech_behavior_student_path": SPEECH_BEHAVIOR_STUDENT_PATH,
            "speech_behavior_alignment": SPEECH_BEHAVIOR_ALIGNMENT,
            "speech_shared_contrastive_loss": float(
                speech_shared_contrastive_loss.detach().float().cpu()
            ),
            "speech_shared_contrastive_coef": speech_shared_contrastive_coef,
            "speech_shared_contrastive_temperature": (
                speech_shared_contrastive_temperature
            ),
            "speech_shared_contrastive_query_count": int(
                speech_shared_contrastive_query_count
            ),
            "speech_shared_contrastive_positive_indices": (
                speech_shared_contrastive_positive_indices
            ),
            "speech_shared_contrastive_positive_row_identities": (
                speech_shared_contrastive_positive_row_identities
            ),
            "speech_shared_contrastive_excluded_duplicate_count": int(
                speech_shared_contrastive_excluded_count
            ),
            "speech_shared_contrastive_student_path": SPEECH_SHARED_CONTRASTIVE_STUDENT_PATH,
            "speech_shared_contrastive_teacher_path": SPEECH_SHARED_CONTRASTIVE_TEACHER_PATH,
            "speech_shared_contrastive_student_pooling": SPEECH_SHARED_CONTRASTIVE_STUDENT_POOLING,
            "speech_shared_contrastive_teacher_pooling": SPEECH_SHARED_CONTRASTIVE_TEACHER_POOLING,
            "speech_shared_contrastive_normalization": SPEECH_SHARED_CONTRASTIVE_NORMALIZATION,
            "speech_shared_contrastive_objective": SPEECH_SHARED_CONTRASTIVE_OBJECTIVE,
            "speech_shared_contrastive_teacher_bank_source": (
                speech_shared_teacher_bank_provenance.get("bank_source")
            ),
            "speech_shared_contrastive_teacher_bank_rows": int(
                speech_shared_teacher_bank_provenance.get("bank_rows", 0)
            ),
            "speech_shared_contrastive_teacher_bank_row_identity_sha256": (
                speech_shared_teacher_bank_provenance.get("row_identity_sha256")
            ),
            "router_z_loss": float(router_z_loss.detach().float().cpu()),
            "router_z_loss_coef": float(args.router_z_loss_coef),
            "expert_dropout_prob": float(args.expert_dropout_prob),
            "expert_dropout_assignment_ratio": mean_expert_dropout_ratio(model),
            **dynamic_bias,
            "conditional_ranking_negatives": int(args.conditional_ranking_negatives),
            "conditional_ranking_negative_mode": str(args.conditional_ranking_negative_mode),
            "conditional_ranking_hard_pool_size": int(args.conditional_ranking_hard_pool_size),
            "conditional_ranking_temperature": float(args.conditional_ranking_temperature),
            "base_contrastive_coef": float(args.contrastive_coef),
            "image_contrastive_coef": float(args.image_contrastive_coef),
            "speech_contrastive_coef": float(args.speech_contrastive_coef),
            "center_positive_weight": float(center_positive_weight),
            "raw_positive_weight": float(raw_positive_weight),
            "base_center_positive_weight": float(args.center_positive_weight),
            "base_raw_positive_weight": float(args.raw_positive_weight),
            "image_center_positive_weight": float(args.image_center_positive_weight),
            "image_raw_positive_weight": float(args.image_raw_positive_weight),
            "speech_center_positive_weight": float(args.speech_center_positive_weight),
            "speech_raw_positive_weight": float(args.speech_raw_positive_weight),
            "contrastive_temperature": float(contrastive_temperature),
            "base_contrastive_temperature": float(args.contrastive_temperature),
            "image_contrastive_temperature": float(args.image_contrastive_temperature),
            "speech_contrastive_temperature": float(args.speech_contrastive_temperature),
            "contrastive_negatives": int(contrastive_negatives),
            "base_contrastive_negatives": int(args.contrastive_negatives),
            "image_contrastive_negatives": int(args.image_contrastive_negatives),
            "speech_contrastive_negatives": int(args.speech_contrastive_negatives),
            "contrastive_positive_cosine": float(contrastive_positive_cosine.detach().float().cpu()),
            "raw_contrastive_positive_cosine": float(raw_contrastive_positive_cosine.detach().float().cpu()) if mode != "text" else 0.0,
            "optimizer_step": optimizer_step,
            "initial_checkpoint_state_restored": bool(initial_checkpoint_state is not None),
            "source_selected_checkpoint_sha256": initialization.get("sha256"),
            "speech_initial_checkpoint_state_restored": bool(
                speech_initial_checkpoint_state is not None
            ),
            "source_speech_initial_checkpoint_path": speech_initialization.get("path"),
            "source_speech_initial_checkpoint_sha256": speech_initialization.get("sha256"),
            "source_speech_initial_manifest_path": speech_initialization.get(
                "manifest_path"
            ),
            "source_speech_initial_manifest_sha256": speech_initialization.get(
                "manifest_sha256"
            ),
            "source_speech_initial_source_commit_sha": speech_initialization.get(
                "source_commit_sha"
            ),
            "source_speech_initial_runai_job_name": speech_initialization.get("runai_job_name"),
            "source_speech_initial_runai_project": speech_initialization.get(
                "runai_project"
            ),
            "stage_b_checkpoint_state_restored": bool(stage_b_checkpoint_state is not None),
            "source_stage_b_checkpoint_sha256": stage_b_initialization.get("sha256"),
            "aux_loss": float(aux_loss_raw.detach().float().cpu()),
            "aux_loss_semantics": "raw_hf_router_aux_loss_before_coefficient",
            **routing,
            **cuda_metrics(),
        }
        if speech_shared_teacher_bank_provenance:
            row["speech_shared_teacher_bank_provenance"] = dict(
                speech_shared_teacher_bank_provenance
            )
        if mode != "text":
            row.update(
                modality_router_metrics(
                    outputs,
                    2,
                    int(model.config.num_experts),
                    args.train_batch_size,
                    image_tokens,
                    audio_tokens,
                    text_len,
                    dispatch_snapshot=dispatch_snapshot,
                )
            )
        rows.append(row)
        append_jsonl(log_path, row)
        if step == 1 or step % args.log_every_steps == 0 or step == max_steps:
            print(json.dumps({k: row.get(k) for k in ["experiment_id", "step", "modality", "loss", "ce_loss", "contrastive_loss", "conditional_ranking_loss", "speech_behavior_kl_loss", "speech_behavior_kl_token_count", "conditional_ranking_accuracy", "contrastive_positive_cosine", "aux_loss", "gate_entropy_mean", "inactive_expert_ratio_mean", "capacity_overflow_ratio_mean", "dynamic_expert_bias_inactive_proxy", "dynamic_expert_bias_overflow_proxy", "dynamic_expert_bias_abs_max", "cuda_memory_reserved_gb"]}, sort_keys=True))
        if step % args.save_every_steps == 0:
            save_checkpoint(wrapper, out_dir / exp_id / f"checkpoint_step_{step}.pt", trainable_meta, args, rows[-1], speech_model=speech_model)

    final_cycle_routing = assignment_weighted_final_cycle_summary(rows, modality_cycle)
    final_cycle_aggregate = final_cycle_routing["overall"]
    data_cursor_provenance = {
        "policy": "independent_per_modality_contiguous_cycle",
        "configured_modality_cycle": modality_cycle,
        "by_modality": {
            modality: {
                "dataset_size": len(training_rows_by_modality[modality]),
                "final_cursor_end_exclusive": int(modality_data_cursors[modality]),
                "unique_rows_covered": min(
                    len(training_rows_by_modality[modality]),
                    int(modality_data_cursors[modality]),
                ),
                "coverage_ratio": float(
                    min(
                        len(training_rows_by_modality[modality]),
                        int(modality_data_cursors[modality]),
                    )
                    / len(training_rows_by_modality[modality])
                ),
                "completed_dataset_passes": int(modality_data_cursors[modality])
                // len(training_rows_by_modality[modality]),
            }
            for modality in sorted(modality_data_cursors)
        },
    }
    wrapper.eval()
    speech_model.eval()
    final_checkpoint_path = out_dir / exp_id / "checkpoint_final.pt"
    save_checkpoint(wrapper, final_checkpoint_path, trainable_meta, args, rows[-1], speech_model=speech_model)
    checkpoint_stat = final_checkpoint_path.stat()
    checkpoint_sha256 = sha256_file(final_checkpoint_path)
    lm_trainable = bool(
        train_router_gates or train_experts or args.train_lm_head or selected_expert_ids
    )
    text_eval_note = (
        f"This {exp_id} run trains selected language-model parameters, so text metrics must be read from the {exp_id} checkpoint provenance rather than assumed equal to E2."
        if lm_trainable
        else f"This {exp_id} adapter run freezes the LM/router/expert weights, so text-only metrics may remain close to calibrated Top-2 while multimodal prefix modules change."
    )
    text_eval_provenance = {
        "source_experiment_id": exp_id,
        "source_checkpoint": str(final_checkpoint_path),
        "source_checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_training_steps": int(len(rows)),
        "source_checkpoint_saved_before_eval": True,
        "model_state_source": "in_memory_wrapper_after_training_saved_to_checkpoint",
        "copied_from_e2": False,
        "lm_trainable": lm_trainable,
        "text_eval_note": text_eval_note,
    }
    if evaluate_after_training:
        text_eval = evaluate_text_blocks(
            f"{exp_id}_text_eval",
            wrapper.lm,
            tokenizer,
            list(text_eval_blocks)[: args.text_eval_blocks],
            out_dir,
            {**meta, "capacity_factor": capacity_factor, "aux_coef": aux_coef, "provenance": text_eval_provenance},
            args.max_length,
            args.eval_batch_size,
        )
        retrieval = retrieval_eval(wrapper, tokenizer, image_processor, vision_model, vision_text_tokenizer, speech_processor, speech_model, speech_text_tokenizer, image_eval, audio_eval, device, args, cache)
    else:
        text_eval = {"skipped": True, "reason": "feasibility_only"}
        retrieval = {"skipped": True, "reason": "feasibility_only"}
    artifact = {
        "meta": {
            **meta,
            "capacity_factor": capacity_factor,
            "aux_coef": aux_coef,
            "seed": run_seed,
            "initialization_policy": "reset_global_seed_before_each_matched_arm",
            "initial_checkpoint": initialization,
            "speech_initialization": speech_initialization,
            "stage_b_initialization": stage_b_initialization,
            **trainable_meta,
        },
        "checkpoint_path": str(final_checkpoint_path),
        "checkpoint_size_bytes": int(checkpoint_stat.st_size),
        "checkpoint_sha256": checkpoint_sha256,
        "text_eval_provenance": text_eval_provenance,
        "real_subset": True,
        "vision_model": args.vision_model,
        "speech_model": args.speech_model,
        "speech_shared_contrastive_provenance": speech_shared_contrastive_meta,
        "steps": rows,
        "data_cursor_provenance": data_cursor_provenance,
        "first_loss": rows[0]["loss"],
        "last_loss": rows[-1]["loss"],
        "min_loss": min(row["loss"] for row in rows),
        "text_eval": {k: v for k, v in text_eval.items() if k not in {"expert_counts_total"}},
        "retrieval_eval": retrieval,
        "final_routing_summary_semantics": (
            "trailing_complete_modality_cycle_attempted_assignment_weighted"
        ),
        "final_gate_entropy_mean": final_cycle_aggregate[
            "gate_entropy_mean_assignment_weighted"
        ],
        "final_inactive_expert_ratio_mean": final_cycle_aggregate[
            "inactive_expert_ratio_mean_assignment_weighted"
        ],
        "final_capacity_overflow_ratio_mean": final_cycle_aggregate[
            "capacity_overflow_ratio_mean_assignment_weighted"
        ],
        "final_cycle_gate_entropy_mean_assignment_weighted": final_cycle_aggregate[
            "gate_entropy_mean_assignment_weighted"
        ],
        "final_cycle_inactive_expert_ratio_mean_assignment_weighted": final_cycle_aggregate[
            "inactive_expert_ratio_mean_assignment_weighted"
        ],
        "final_cycle_capacity_overflow_ratio_mean_assignment_weighted": final_cycle_aggregate[
            "capacity_overflow_ratio_mean_assignment_weighted"
        ],
        "final_cycle_dynamic_expert_bias_inactive_proxy_assignment_weighted": final_cycle_aggregate[
            "dynamic_expert_bias_inactive_proxy_assignment_weighted"
        ],
        "final_cycle_dynamic_expert_bias_overflow_proxy_assignment_weighted": final_cycle_aggregate[
            "dynamic_expert_bias_overflow_proxy_assignment_weighted"
        ],
        "final_cycle_routing_aggregate_provenance": {
            key: value
            for key, value in final_cycle_routing.items()
            if key not in {"overall", "by_modality"}
        },
        "final_cycle_routing_assignment_weighted_by_modality": final_cycle_routing[
            "by_modality"
        ],
        "final_dynamic_expert_bias": {k: rows[-1].get(k) for k in rows[-1] if str(k).startswith("dynamic_expert_bias")},
        "final_modality_expert_utilization": rows[-1].get("modality_expert_utilization"),
        "final_modality_js": {
            "image_audio": rows[-1].get("modality_js_image_audio"),
            "image_text": rows[-1].get("modality_js_image_text"),
            "audio_text": rows[-1].get("modality_js_audio_text"),
        },
    }
    if speech_shared_teacher_bank_provenance:
        artifact["speech_shared_teacher_bank_provenance"] = dict(
            speech_shared_teacher_bank_provenance
        )
    save_json(out_dir / exp_id / "metrics.json", artifact)
    for handle in selected_expert_hook_handles:
        handle.remove()
    cleanup(wrapper, model, vision_model, speech_model)
    return artifact


def selected_expert_rows_state_dict(
    model,
    selected_expert_ids: Dict[int | str, Sequence[int]],
) -> Dict[str, Any]:
    """Serialize only selected full expert rows, never the full packed bank."""
    state: Dict[str, Any] = {}
    for layer_idx, layer in enumerate(model.model.layers):
        raw_ids = selected_expert_ids.get(layer_idx, selected_expert_ids.get(str(layer_idx), ()))
        expert_ids = sorted({int(value) for value in raw_ids})
        if not expert_ids:
            raise ValueError(f"selected expert checkpoint is missing layer {layer_idx}")
        experts = layer.mlp.experts
        state[f"layer_{layer_idx}"] = {
            "expert_ids": expert_ids,
            "gate_up_proj": experts.gate_up_proj.detach()[expert_ids].cpu().clone(),
            "down_proj": experts.down_proj.detach()[expert_ids].cpu().clone(),
        }
    return state


def restore_selected_expert_rows(
    model,
    state: Dict[str, Any],
    expected_selected_expert_ids: Optional[Dict[int | str, Sequence[int]]] = None,
) -> Dict[str, List[int]]:
    """Restore selected packed rows without touching any nonselected row."""
    if not isinstance(state, dict):
        raise TypeError("selected_experts checkpoint state must be a mapping")
    expected_keys = {f"layer_{index}" for index in range(len(model.model.layers))}
    if set(state) != expected_keys:
        raise ValueError("selected_experts checkpoint must cover every model layer")
    restored: Dict[str, List[int]] = {}
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.model.layers):
            key = f"layer_{layer_idx}"
            row = state[key]
            expert_ids = sorted({int(value) for value in row.get("expert_ids", [])})
            if not expert_ids:
                raise ValueError(f"selected_experts checkpoint has no IDs for {key}")
            if expected_selected_expert_ids is not None:
                expected_raw = expected_selected_expert_ids.get(
                    layer_idx, expected_selected_expert_ids.get(str(layer_idx), ())
                )
                if expert_ids != sorted({int(value) for value in expected_raw}):
                    raise ValueError(f"selected_experts checkpoint IDs do not match selection for {key}")
            experts = layer.mlp.experts
            indices = torch.tensor(expert_ids, dtype=torch.long, device=experts.gate_up_proj.device)
            for parameter_name in ("gate_up_proj", "down_proj"):
                parameter = getattr(experts, parameter_name)
                value = torch.as_tensor(row.get(parameter_name))
                expected_shape = (len(expert_ids), *parameter.shape[1:])
                if tuple(value.shape) != tuple(expected_shape):
                    raise ValueError(
                        f"selected_experts checkpoint shape mismatch for {key}.{parameter_name}"
                    )
                parameter.index_copy_(
                    0,
                    indices,
                    value.to(device=parameter.device, dtype=parameter.dtype),
                )
            restored[str(layer_idx)] = expert_ids
    return restored


def restore_training_checkpoint(
    wrapper,
    state: Dict[str, Any],
    speech_model=None,
    expected_selected_expert_ids: Optional[Dict[int | str, Sequence[int]]] = None,
    expected_selection_provenance: Optional[Dict[str, Any]] = None,
) -> None:
    """Restore the selected lightweight multimodal/LM state before an ablation."""
    if not isinstance(state, dict):
        raise TypeError("checkpoint state must be a mapping")
    for required in ("image_resampler", "audio_resampler"):
        if required not in state:
            raise ValueError(f"selected checkpoint is missing {required}")
    load_dynamic_expert_bias_state(wrapper.lm, state.get("dynamic_expert_bias"))
    wrapper.image_resampler.load_state_dict(state["image_resampler"])
    wrapper.audio_resampler.load_state_dict(state["audio_resampler"])
    selected_state = state.get("selected_experts")
    if selected_state is not None:
        if expected_selection_provenance is not None:
            checkpoint_provenance = state.get("selected_expert_selection_provenance")
            if not isinstance(checkpoint_provenance, dict):
                raise ValueError("selected-expert checkpoint is missing selection provenance")
            for key in (
                "selection_json_sha256",
                "selection_method",
                "selection_scope",
                "selected_expert_ids_by_layer",
            ):
                if checkpoint_provenance.get(key) != expected_selection_provenance.get(key):
                    raise ValueError(f"selected-expert checkpoint provenance mismatch for {key}")
        restore_selected_expert_rows(
            wrapper.lm,
            selected_state,
            expected_selected_expert_ids=expected_selected_expert_ids,
        )
    elif state.get("trainable_meta", {}).get("selected_expert_training"):
        raise ValueError("selected-expert checkpoint is missing selected_experts rows")
    for name in (
        "image_retrieval_head",
        "audio_retrieval_head",
        "image_direct_retrieval_head",
        "audio_direct_retrieval_head",
    ):
        if name in state and hasattr(wrapper, name):
            getattr(wrapper, name).load_state_dict(state[name])
    for index, layer in enumerate(wrapper.lm.model.layers):
        key = f"layer_{index}"
        if key in state.get("router_gates", {}):
            layer.mlp.gate.load_state_dict(state["router_gates"][key])
    output_embeddings = wrapper.lm.get_output_embeddings()
    if output_embeddings is not None and state.get("lm_output_embeddings") is not None:
        output_embeddings.load_state_dict(state["lm_output_embeddings"])
    input_embeddings = wrapper.lm.get_input_embeddings()
    if input_embeddings is not None and state.get("lm_input_embeddings") is not None:
        input_embeddings.load_state_dict(state["lm_input_embeddings"])
    speech_state = state.get("speech_encoder_trainable_state")
    if speech_state is not None:
        if speech_model is None:
            raise ValueError("speech_model is required to restore selected speech encoder state")
        named_params = dict(speech_encoder_module(speech_model).named_parameters())
        missing = sorted(set(speech_state) - set(named_params))
        if missing:
            raise ValueError(f"selected checkpoint speech encoder parameters are missing from model: {missing}")
        with torch.no_grad():
            for name, value in speech_state.items():
                named_params[name].copy_(value.to(device=named_params[name].device, dtype=named_params[name].dtype))


def restore_speech_initialization_checkpoint(
    wrapper,
    state: Dict[str, Any],
    speech_model,
) -> Dict[str, Any]:
    """Restore only the speech bridge and exact last1/LN encoder state."""
    if not isinstance(state, dict):
        raise TypeError("speech initialization checkpoint must be a mapping")
    audio_state = state.get("audio_resampler")
    if not isinstance(audio_state, dict):
        raise ValueError("speech initialization checkpoint is missing audio_resampler")
    speech_state = state.get("speech_encoder_trainable_state")
    if not isinstance(speech_state, dict) or not speech_state:
        raise ValueError("speech initialization checkpoint is missing speech encoder state")
    if speech_model is None:
        raise ValueError("speech_model is required for speech initialization")

    encoder = speech_encoder_module(speech_model)
    layers = getattr(encoder, "layers", None)
    layer_norm = getattr(encoder, "layer_norm", None)
    if layers is None or len(layers) < 1 or layer_norm is None:
        raise RuntimeError(
            "speech initialization requires encoder layers and final layer norm"
        )
    selected_ids = {
        id(parameter)
        for module in (layers[-1], layer_norm)
        for parameter in module.parameters()
    }
    named_params = dict(encoder.named_parameters())
    expected_names = {
        name for name, parameter in named_params.items() if id(parameter) in selected_ids
    }
    observed_names = set(speech_state)
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise ValueError(
            "speech initialization state must contain exactly last1/LN parameters: "
            f"missing={missing} unexpected={unexpected}"
        )

    wrapper.audio_resampler.load_state_dict(audio_state)
    with torch.no_grad():
        for name, value in speech_state.items():
            named_params[name].copy_(
                value.to(device=named_params[name].device, dtype=named_params[name].dtype)
            )
    return {
        "scope": "speech",
        "image_restored": False,
        "speech_restored": True,
        "speech_encoder_state_restored": True,
        "speech_encoder_state_policy": "exact_last1_and_final_layer_norm",
        "image_state_ignored": True,
        "audio_retrieval_state_ignored": True,
        "lm_router_expert_state_ignored": True,
    }



def restore_scoped_multimodal_checkpoint(
    wrapper,
    state: Dict[str, Any],
    scope: str,
    speech_model=None,
    expected_selected_expert_ids: Optional[Dict[int | str, Sequence[int]]] = None,
    expected_selection_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if scope == "both":
        restore_training_checkpoint(
            wrapper,
            state,
            speech_model=speech_model,
            expected_selected_expert_ids=expected_selected_expert_ids,
            expected_selection_provenance=expected_selection_provenance,
        )
        return {"scope": scope, "image_restored": True, "speech_restored": True}
    if scope not in {"image", "speech"}:
        raise ValueError(f"unsupported multimodal initialization scope: {scope!r}")

    if scope == "image":
        if "image_resampler" not in state:
            raise ValueError("image-scoped checkpoint is missing image_resampler")
        wrapper.image_resampler.load_state_dict(state["image_resampler"])
        names = ("image_retrieval_head", "image_direct_retrieval_head")
    else:
        if "audio_resampler" not in state:
            raise ValueError("speech-scoped checkpoint is missing audio_resampler")
        wrapper.audio_resampler.load_state_dict(state["audio_resampler"])
        names = ("audio_retrieval_head", "audio_direct_retrieval_head")
    for name in names:
        if name in state and hasattr(wrapper, name):
            getattr(wrapper, name).load_state_dict(state[name])

    speech_state = state.get("speech_encoder_trainable_state") if scope == "speech" else None
    if speech_state is not None:
        if speech_model is None:
            raise ValueError("speech_model is required for speech-scoped restoration")
        named_params = dict(speech_encoder_module(speech_model).named_parameters())
        missing = sorted(set(speech_state) - set(named_params))
        if missing:
            raise ValueError(
                f"speech-scoped checkpoint parameters are missing from model: {missing}"
            )
        with torch.no_grad():
            for name, value in speech_state.items():
                named_params[name].copy_(
                    value.to(device=named_params[name].device, dtype=named_params[name].dtype)
                )
    return {
        "scope": scope,
        "image_restored": scope == "image",
        "speech_restored": scope == "speech",
        "speech_encoder_state_restored": speech_state is not None,
        "lm_router_state_ignored": True,
    }


def _validate_stage_b_v2_contract(state: Mapping[str, Any]) -> None:
    contract = state.get("resume_contract")
    required = {
        "schema_version",
        "args",
        "trainable",
        "optimizer_groups",
        "args_sha256",
        "trainable_sha256",
        "optimizer_groups_sha256",
        "data",
        "data_sha256",
        "base_model_identity",
        "base_model_identity_sha256",
        "curriculum_plan",
        "curriculum_plan_sha256",
    }
    if not isinstance(contract, Mapping) or set(contract) != required:
        raise ValueError("Stage B v2 checkpoint is missing resume contract")
    if contract["schema_version"] != 2:
        raise ValueError("Stage B v2 resume contract schema mismatch")
    for field in (
        "args",
        "trainable",
        "optimizer_groups",
        "data",
        "base_model_identity",
        "curriculum_plan",
    ):
        value = contract[field]
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != contract[f"{field}_sha256"]:
            raise ValueError(f"Stage B v2 resume contract digest mismatch for {field}")
    checkpoint_args = state.get("args")
    checkpoint_trainable = state.get("trainable_meta")
    if state.get("curriculum_plan") != contract["curriculum_plan"]:
        raise ValueError(
            "Stage B v2 checkpoint curriculum plan disagrees with resume contract"
        )
    if (
        not isinstance(checkpoint_args, Mapping)
        or any(checkpoint_args.get(key) != value for key, value in contract["args"].items())
    ):
        raise ValueError("Stage B v2 checkpoint args disagree with resume contract")
    if (
        not isinstance(checkpoint_trainable, Mapping)
        or any(
            checkpoint_trainable.get(key) != value
            for key, value in contract["trainable"].items()
        )
    ):
        raise ValueError(
            "Stage B v2 checkpoint trainable state disagrees with resume contract"
        )
    optimizer_state = state.get("optimizer_state")
    if not isinstance(optimizer_state, Mapping) or not isinstance(
        optimizer_state.get("param_groups"), list
    ):
        raise ValueError("Stage B v2 checkpoint is missing optimizer state")
    optimizer_groups = []
    for group in optimizer_state["param_groups"]:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("Stage B v2 optimizer group state is invalid")
        parameter_names = group.get("param_names")
        if (
            not isinstance(parameter_names, list)
            or len(parameter_names) != len(group["params"])
            or any(not isinstance(name, str) or not name for name in parameter_names)
        ):
            raise ValueError(
                "Stage B v2 optimizer group parameter names are invalid"
            )
        value = {
            key: item
            for key, item in group.items()
            if key not in {"params", "param_names"}
        }
        value["parameter_count"] = len(group["params"])
        value["parameter_names"] = parameter_names
        optimizer_groups.append(value)
    if optimizer_groups != contract["optimizer_groups"]:
        raise ValueError("Stage B v2 optimizer groups disagree with resume contract")
    if not isinstance(state.get("rng_state"), Mapping):
        raise ValueError("Stage B v2 checkpoint is missing RNG state")


def restore_stage_b_student_initialization(
    wrapper,
    state: Dict[str, Any],
    expected_base_model: str,
    runtime_base_model_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Restore a provenance-checked Stage B Top-2 LM/router checkpoint into E3."""
    provenance = state.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Stage B checkpoint is missing provenance")
    required = {
        "stage": "B",
        "development_data_only": True,
        "teacher_top_k": 8,
        "final_inference_top_k": 2,
        "base_model": str(expected_base_model),
    }
    for key, expected in required.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"Stage B checkpoint provenance mismatch for {key}: "
                f"expected={expected!r} observed={provenance.get(key)!r}"
            )
    if (
        int(state.get("checkpoint_version", 0)) < 2
        or int(state.get("final_inference_top_k", 0)) != 2
    ):
        raise ValueError("Stage B checkpoint must use schema v2+ and final Top-2 inference")
    _validate_stage_b_v2_contract(state)
    checkpoint_base_model_identity = state["resume_contract"][
        "base_model_identity"
    ]
    if dict(runtime_base_model_identity) != dict(checkpoint_base_model_identity):
        raise ValueError(
            "E3 runtime pre-routing base-model identity differs from Stage-B checkpoint"
        )
    args = state.get("args")
    if not isinstance(args, dict) or int(args.get("student_top_k", 0)) != 2:
        raise ValueError("Stage B checkpoint args do not declare a Top-2 student")
    trainable = state.get("trainable_meta")
    if not isinstance(trainable, dict):
        raise ValueError("Stage B checkpoint is missing trainable_meta")
    if not any(
        bool(trainable.get(key))
        for key in ("train_router_gates", "train_lm_head", "train_gamma_scale")
    ):
        raise ValueError("Stage B checkpoint has no trainable student state")

    router_state = state.get("router_gates")
    if trainable.get("train_router_gates"):
        expected_keys = {
            f"layer_{index}" for index in range(len(wrapper.lm.model.layers))
        }
        if not isinstance(router_state, dict) or set(router_state) != expected_keys:
            raise ValueError(
                "Stage B trainable-router checkpoint must cover every router layer"
            )
    for layer_idx, layer in enumerate(wrapper.lm.model.layers):
        if isinstance(router_state, dict) and f"layer_{layer_idx}" in router_state:
            layer.mlp.gate.load_state_dict(router_state[f"layer_{layer_idx}"])

    output_embeddings = wrapper.lm.get_output_embeddings()
    input_embeddings = wrapper.lm.get_input_embeddings()
    if trainable.get("train_lm_head"):
        if state.get("lm_output_embeddings") is None:
            raise ValueError(
                "Stage B trainable-LM-head checkpoint is missing output embeddings"
            )
        embeddings_tied = state.get("lm_embeddings_tied")
        if not isinstance(embeddings_tied, bool):
            raise ValueError(
                "Stage B trainable-LM-head checkpoint is missing tied-embedding metadata"
            )
        runtime_embeddings_tied = lm_embeddings_are_tied(wrapper.lm)
        if embeddings_tied is not runtime_embeddings_tied:
            raise ValueError(
                "Stage B tied-embedding metadata disagrees with runtime model"
            )
        if not embeddings_tied and state.get("lm_input_embeddings") is None:
            raise ValueError(
                "Stage B untied trainable-LM-head checkpoint is missing input embeddings"
            )
    if output_embeddings is not None and state.get("lm_output_embeddings") is not None:
        output_embeddings.load_state_dict(state["lm_output_embeddings"])
    if input_embeddings is not None and state.get("lm_input_embeddings") is not None:
        input_embeddings.load_state_dict(state["lm_input_embeddings"])

    gamma_state = state.get("gamma_scale")
    if trainable.get("train_gamma_scale") and gamma_state is None:
        raise ValueError(
            "Stage B trainable-gamma checkpoint is missing gamma state"
        )
    if gamma_state is not None:
        if len(gamma_state) != len(wrapper.lm.model.layers):
            raise ValueError("Stage B gamma scale layer count mismatch")
        for layer, value in zip(wrapper.lm.model.layers, gamma_state):
            gamma = getattr(layer.mlp, "gamma_scale", None)
            if gamma is None:
                raise ValueError(
                    "Stage B checkpoint contains gamma but runtime layer does not"
                )
            gamma.data.copy_(
                torch.as_tensor(value, device=gamma.device, dtype=gamma.dtype)
            )
    return {
        "checkpoint_version": int(state["checkpoint_version"]),
        "completed_steps": int(state.get("completed_steps", 0)),
        "router_state_restored": bool(router_state),
        "lm_head_state_restored": state.get("lm_output_embeddings") is not None,
        "gamma_state_restored": gamma_state is not None,
        "final_inference_top_k": 2,
    }


def load_stage_b_initialization_checkpoint(
    path_value: str, expected_sha256: str
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if not path_value:
        if expected_sha256:
            raise ValueError(
                "--stage-b-checkpoint-sha256 requires --stage-b-checkpoint"
            )
        return None, {}
    reject_forbidden_development_path(path_value, "Stage B checkpoint")
    if len(expected_sha256) != 64:
        raise ValueError(
            "--stage-b-checkpoint-sha256 must be a full 64-character SHA256"
        )
    raw_path = Path(path_value).expanduser()
    if raw_path.is_symlink():
        raise ValueError("Stage B checkpoint cannot be a symlink")
    path = raw_path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("Stage B checkpoint must be a regular file")
    checkpoint_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"Stage B checkpoint SHA256 mismatch: expected={expected_sha256} "
            f"observed={actual_sha256}"
        )
    state = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state, dict):
        raise TypeError("Stage B checkpoint payload must be a mapping")
    return state, {
        "path": str(path),
        "sha256": actual_sha256,
        "size_bytes": len(checkpoint_bytes),
        "policy": "development_only_stage_b_top8_to_top2_initialization",
        "sealed_evidence_used": False,
        "synthetic_evidence_used": False,
    }


def load_multimodal_initialization_checkpoint(
    path_value: str, expected_sha256: str, manifest_path_value: str,
    expected_scope: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if not path_value:
        if expected_sha256 or manifest_path_value:
            raise ValueError(
                "--multimodal-initial-checkpoint-sha256/manifest requires "
                "--multimodal-initial-checkpoint"
            )
        return None, {}
    if not manifest_path_value:
        raise ValueError(
            "--multimodal-initial-checkpoint requires "
            "--multimodal-initial-manifest"
        )
    reject_forbidden_development_path(path_value, "multimodal checkpoint")
    reject_forbidden_development_path(manifest_path_value, "multimodal manifest")
    if not _is_full_hex(expected_sha256, 64):
        raise ValueError(
            "--multimodal-initial-checkpoint-sha256 must be a full SHA256"
        )
    path = Path(path_value).resolve(strict=True)
    manifest_path = Path(manifest_path_value).resolve(strict=True)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"multimodal checkpoint SHA256 mismatch: expected={expected_sha256} "
            f"observed={actual_sha256}"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError("multimodal checkpoint payload must be a mapping")
    missing = [
        key
        for key in ("image_resampler", "audio_resampler")
        if not isinstance(state.get(key), dict)
    ]
    if missing:
        raise ValueError(
            f"multimodal checkpoint is missing bridge states: {missing}"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("multimodal manifest is not valid structured JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("multimodal manifest must be a JSON object")
    completion = manifest.get("completion")
    if not isinstance(completion, dict) or completion.get("status") != "completed":
        raise ValueError("multimodal manifest completion.status must be completed")
    completion_path_value = completion.get("e3_checkpoint_path")
    if not isinstance(completion_path_value, str) or not completion_path_value:
        raise ValueError("multimodal manifest completion checkpoint path is missing")
    reject_forbidden_development_path(
        completion_path_value, "multimodal manifest completion checkpoint"
    )
    if Path(completion_path_value).resolve() != path:
        raise ValueError(
            "multimodal manifest completion checkpoint path does not match "
            "the exact checkpoint"
        )
    if completion.get("e3_checkpoint_sha256") != actual_sha256:
        raise ValueError("multimodal manifest completion SHA256 does not match checkpoint")

    checkpoint_provenance = state.get("run_provenance")
    manifest_provenance = manifest.get("run_provenance")
    if not isinstance(checkpoint_provenance, dict):
        raise ValueError("multimodal checkpoint is missing run_provenance")
    if not isinstance(manifest_provenance, dict):
        raise ValueError("multimodal manifest is missing run_provenance")
    provenance_fields = (
        "source_commit_sha",
        "runai_job_name",
        "runai_project",
        "resolved_data_root",
        "resolved_output_root",
        "final_main_steps",
        "alignment_pretrain_steps",
        "checkpoint_completed_step",
        "policy",
        "sealed_evidence_used",
        "synthetic_evidence_used",
    )
    if any(
        checkpoint_provenance.get(field) != manifest_provenance.get(field)
        for field in provenance_fields
    ):
        raise ValueError("multimodal checkpoint and manifest run_provenance disagree")
    provenance = checkpoint_provenance
    if not _is_full_hex(provenance.get("source_commit_sha"), 40):
        raise ValueError("multimodal provenance source commit is not full hex")
    for field in ("runai_job_name", "runai_project"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise ValueError(f"multimodal provenance is missing {field}")
    if provenance.get("policy") != STAGE_A_PROVENANCE_POLICY:
        raise ValueError("multimodal provenance policy is invalid")
    if provenance.get("sealed_evidence_used") is not False:
        raise ValueError("multimodal provenance does not reject sealed evidence")
    if provenance.get("synthetic_evidence_used") is not False:
        raise ValueError("multimodal provenance does not reject synthetic evidence")
    for field in ("resolved_data_root", "resolved_output_root"):
        value = provenance.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"multimodal provenance {field} is not an absolute path")
        reject_forbidden_development_path(value, f"multimodal provenance {field}")

    step_requirements = (
        ("final_main_steps", 1),
        ("alignment_pretrain_steps", 0),
        ("checkpoint_completed_step", 1),
    )
    for field, minimum in step_requirements:
        value = provenance.get(field)
        if type(value) is not int or value < minimum:
            raise ValueError(f"multimodal provenance {field} is invalid")

    manifest_args = manifest.get("args")
    checkpoint_args = state.get("args")
    if not isinstance(manifest_args, dict) or not isinstance(checkpoint_args, dict):
        raise ValueError("multimodal manifest/checkpoint args must be mappings")
    expected_values = {
        "final_steps": int(provenance["final_main_steps"]),
        "alignment_pretrain_steps": int(provenance["alignment_pretrain_steps"]),
    }
    for field, expected in expected_values.items():
        if manifest_args.get(field) != expected or checkpoint_args.get(field) != expected:
            raise ValueError(f"multimodal manifest/checkpoint args disagree on {field}")
    for field, provenance_field in (
        ("data_dir", "resolved_data_root"),
        ("output_dir", "resolved_output_root"),
    ):
        expected_root = Path(str(provenance[provenance_field]))
        values = (
            manifest.get(field),
            manifest_args.get(field),
            checkpoint_args.get(field),
        )
        for value in values:
            if isinstance(value, str):
                reject_forbidden_development_path(
                    value, f"multimodal manifest/checkpoint {field}"
                )
        if any(
            not isinstance(value, str) or Path(value).resolve() != expected_root
            for value in values
        ):
            raise ValueError(f"multimodal manifest/checkpoint args disagree on {field}")
    if manifest.get("source_commit_sha") != provenance["source_commit_sha"]:
        raise ValueError("multimodal manifest source commit disagrees with checkpoint")
    if manifest.get("runai_job_name") != provenance["runai_job_name"]:
        raise ValueError("multimodal manifest RunAI job disagrees with checkpoint")
    if manifest.get("runai_project") != provenance["runai_project"]:
        raise ValueError("multimodal manifest RunAI project disagrees with checkpoint")

    completed_step = int(provenance["checkpoint_completed_step"])
    if (
        completion.get("e3_steps") != int(provenance["final_main_steps"])
        or completed_step != int(provenance["final_main_steps"])
        or not isinstance(state.get("last_row"), dict)
        or state["last_row"].get("step") != completed_step
    ):
        raise ValueError("multimodal manifest/checkpoint completion steps disagree")
    if expected_scope is not None:
        if expected_scope not in {"image", "speech"}:
            raise ValueError(f"unsupported expected Stage A scope: {expected_scope!r}")
        scope_requirements: Dict[str, Any] = {
            "alignment_pretrain_modalities": expected_scope,
        }
        if expected_scope == "speech":
            scope_requirements.update(
                {
                    "speech_unfreeze_last_blocks": 1,
                    "speech_unfreeze_layer_norm": True,
                }
            )
        for field, expected in scope_requirements.items():
            if (
                manifest_args.get(field) != expected
                or checkpoint_args.get(field) != expected
            ):
                raise ValueError(
                    f"Stage A companion scope mismatch for {field}: "
                    f"expected={expected!r}"
                )
        if expected_scope == "speech":
            speech_state = state.get("speech_encoder_trainable_state")
            if not isinstance(speech_state, dict) or not speech_state:
                raise ValueError(
                    "speech-scoped Stage A checkpoint is missing speech encoder state"
                )
    manifest_sha256 = sha256_file(manifest_path)
    return state, {
        "path": str(path),
        "sha256": actual_sha256,
        "size_bytes": int(path.stat().st_size),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "source_commit_sha": provenance["source_commit_sha"],
        "runai_job_name": provenance["runai_job_name"],
        "runai_project": provenance["runai_project"],
        "completion_status": completion["status"],
        "completion_step": completed_step,
        "policy": provenance["policy"],
        "sealed_evidence_used": provenance["sealed_evidence_used"],
        "synthetic_evidence_used": provenance["synthetic_evidence_used"],
        **({"scope": expected_scope} if expected_scope is not None else {}),
    }


def load_speech_initialization_checkpoint(
    path_value: str, expected_sha256: str, manifest_path_value: str
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if not path_value:
        if expected_sha256 or manifest_path_value:
            raise ValueError(
                "--speech-initial-checkpoint-sha256/manifest requires "
                "--speech-initial-checkpoint"
            )
        return None, {}
    if not expected_sha256 or not manifest_path_value:
        raise ValueError(
            "--speech-initial-checkpoint requires an exact SHA256 and manifest"
        )
    return load_multimodal_initialization_checkpoint(
        path_value,
        expected_sha256,
        manifest_path_value,
        expected_scope="speech",
    )


def validate_dual_initialization_request(
    multimodal_state: Optional[Dict[str, Any]],
    multimodal_scope: str,
    speech_state: Optional[Dict[str, Any]],
) -> None:
    if speech_state is None:
        return
    if multimodal_state is None:
        raise ValueError("speech initializer requires an image initializer")
    if multimodal_scope != "image":
        raise ValueError(
            "dual initialization requires MULTIMODAL_INITIALIZATION_SCOPE=image"
        )


def validate_initialization_state_disjoint(
    multimodal_state: Optional[Dict[str, Any]],
    stage_b_state: Optional[Dict[str, Any]],
    multimodal_scope: str = "both",
) -> None:
    if multimodal_scope not in {"both", "image", "speech"}:
        raise ValueError(
            f"unsupported multimodal initialization scope: {multimodal_scope!r}"
        )
    if (
        multimodal_state is None
        or stage_b_state is None
        or multimodal_scope != "both"
    ):
        return
    conflicting = []
    for key in ("router_gates", "lm_output_embeddings", "lm_input_embeddings"):
        if multimodal_state.get(key) is not None and stage_b_state.get(key) is not None:
            conflicting.append(key)
    if conflicting:
        raise ValueError(
            "Stage A multimodal and Stage B student checkpoints contain overlapping "
            f"LM state: {conflicting}"
        )


def checkpoint_gamma_provenance(args) -> Dict[str, Any]:
    output_root = Path(str(args.output_dir)).expanduser().resolve(strict=True)
    gamma_path = (
        output_root / "calibration" / "gamma.json"
    ).resolve(strict=True)
    if not gamma_path.is_file() or gamma_path.parent.parent != output_root:
        raise ValueError(
            f"E3 checkpoint gamma must be {output_root / 'calibration' / 'gamma.json'}"
        )
    return {
        "path": str(gamma_path),
        "relative_path": "calibration/gamma.json",
        "output_dir": str(output_root),
        "sha256": sha256_file(gamma_path),
        "size_bytes": int(gamma_path.stat().st_size),
    }


def save_checkpoint(wrapper, path: Path, trainable_meta: Dict[str, Any], args, last_row: Dict[str, Any], speech_model=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "image_resampler": wrapper.image_resampler.state_dict(),
        "audio_resampler": wrapper.audio_resampler.state_dict(),
        "image_retrieval_head": wrapper.image_retrieval_head.state_dict(),
        "audio_retrieval_head": wrapper.audio_retrieval_head.state_dict(),
        "image_direct_retrieval_head": wrapper.image_direct_retrieval_head.state_dict(),
        "audio_direct_retrieval_head": wrapper.audio_direct_retrieval_head.state_dict(),
        "trainable_meta": trainable_meta,
        "args": vars(args),
        "last_row": last_row,
        "run_provenance": build_stage_a_run_provenance(args, last_row["step"]),
        "gamma_provenance": checkpoint_gamma_provenance(args),
    }
    teacher_bank_provenance = trainable_meta.get(
        "speech_shared_teacher_bank_provenance"
    )
    if teacher_bank_provenance:
        state["speech_shared_contrastive_provenance"] = dict(
            trainable_meta["speech_shared_contrastive_provenance"]
        )
        state["speech_shared_teacher_bank_provenance"] = dict(
            teacher_bank_provenance
        )
    dynamic_bias_state = dynamic_expert_bias_state_dict(wrapper.lm)
    if dynamic_bias_state:
        state["dynamic_expert_bias"] = dynamic_bias_state
        state["dynamic_expert_bias_meta"] = dynamic_expert_bias_metrics(wrapper.lm)
    if trainable_meta.get("train_router_gates"):
        state["router_gates"] = {f"layer_{idx}": layer.mlp.gate.state_dict() for idx, layer in enumerate(wrapper.lm.model.layers)}
    if trainable_meta.get("selected_expert_training"):
        selected_ids = trainable_meta.get("selected_expert_ids_by_layer", {})
        state["selected_experts"] = selected_expert_rows_state_dict(wrapper.lm, selected_ids)
        state["selected_expert_selection_provenance"] = trainable_meta.get(
            "expert_selection_provenance", {}
        )
    if trainable_meta.get("train_experts"):
        if getattr(args, "save_expert_weights", False):
            state["experts"] = {f"layer_{idx}": layer.mlp.experts.state_dict() for idx, layer in enumerate(wrapper.lm.model.layers)}
        else:
            state["experts_omitted"] = "Set --save-expert-weights to store full expert tensors; omitted here to keep optional ablation checkpoints portable."
    if trainable_meta.get("train_lm_head"):
        output_embeddings = wrapper.lm.get_output_embeddings()
        input_embeddings = wrapper.lm.get_input_embeddings()
        embeddings_tied = lm_embeddings_are_tied(wrapper.lm)
        state["lm_embeddings_tied"] = embeddings_tied
        state["lm_output_embeddings"] = output_embeddings.state_dict() if output_embeddings is not None else None
        if not embeddings_tied:
            state["lm_input_embeddings"] = input_embeddings.state_dict() if input_embeddings is not None else None
    speech_names = list(trainable_meta.get("speech_encoder_trainable_names", []))
    if speech_names:
        if speech_model is None:
            raise ValueError("speech_model is required to save selected speech encoder state")
        named_params = dict(speech_encoder_module(speech_model).named_parameters())
        state["speech_encoder_trainable_state"] = {
            name: named_params[name].detach().cpu().clone() for name in speech_names
        }
    torch.save(state, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/real_subset")
    parser.add_argument(
        "--development-split-manifest",
        default="",
        help=(
            "Schema-v3 content/speaker-disjoint manifest. When set, training reads "
            "only train and dev partitions and keeps its eval partition reserved."
        ),
    )
    parser.add_argument(
        "--development-speech-source-sha256",
        default="",
        help=(
            "Externally trusted SHA256 of canonical "
            "data_dir/speech_transcripts.jsonl; required with a development split."
        ),
    )
    parser.add_argument("--base-model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--vision-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--speech-model", default="openai/whisper-base.en")
    parser.add_argument("--speech-target-space", choices=["olmoe_text_hidden", "whisper_decoder_text"], default="olmoe_text_hidden")
    parser.add_argument("--image-alignment-target", choices=["clip_text", "olmoe_caption_hidden"], default="clip_text")
    parser.add_argument("--alignment-prefix-residual", action="store_true")
    bridge_choices = [
        "query_resampler",
        "linear_projector",
        "identity",
        "linear_projector_norm",
        "attention_pool",
        "temporal_resample",
        "local_pool_linear",
    ]
    parser.add_argument("--image-bridge-type", choices=bridge_choices, default="query_resampler")
    parser.add_argument("--audio-bridge-type", choices=bridge_choices, default="query_resampler")
    parser.add_argument("--bridge-num-heads", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/real_required_runs")
    parser.add_argument("--feature-cache-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--capacity-factor", type=float, default=4.0)
    parser.add_argument("--capacity-ablation-factor", type=float, default=1.25)
    parser.add_argument("--aux-coef", type=float, default=0.01)
    parser.add_argument("--router-z-loss-coef", type=float, default=0.0)
    parser.add_argument("--expert-dropout-prob", type=float, default=0.0)
    parser.add_argument("--dynamic-expert-bias-lr", type=float, default=0.0, help="Aux-loss-free expert bias update rate; 0 disables dynamic balancing.")
    parser.add_argument("--dynamic-expert-bias-update-interval", type=int, default=1)
    parser.add_argument("--dynamic-expert-bias-warmup-steps", type=int, default=0)
    parser.add_argument("--dynamic-expert-bias-max-abs", type=float, default=2.0)
    parser.add_argument("--gamma-min", type=float, default=0.25)
    parser.add_argument("--gamma-max", type=float, default=2.0)
    parser.add_argument("--final-steps", type=int, default=4000)
    parser.add_argument("--ablation-steps", type=int, default=1000)
    parser.add_argument("--capacity-ablation-steps", type=int, default=1000)
    parser.add_argument("--expert-ablation-steps", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument(
        "--speech-teacher-bank-batch-size",
        type=int,
        default=64,
        help="Transcript-teacher embedding batch size; lower explicitly on OOM.",
    )
    parser.add_argument("--modality-cycle", default="text,image,speech", help="Comma-separated training modality schedule, e.g. text,image,speech,speech.")
    parser.add_argument("--text-eval-blocks", type=int, default=250)
    parser.add_argument("--retrieval-eval-samples", type=int, default=250)
    parser.add_argument("--conditional-eval-samples", type=int, default=250)
    parser.add_argument("--conditional-negatives", type=int, default=4)
    parser.add_argument("--conditional-batch-size", type=int, default=16)
    parser.add_argument("--conditional-control-seed", type=int, default=42)
    parser.add_argument("--conditional-candidate-permutation", choices=["query_identity_seeded"], default="query_identity_seeded")
    parser.add_argument("--conditional-tie-epsilon", type=float, default=1e-8)
    parser.add_argument("--conditional-ranking-negatives", type=int, default=2)
    parser.add_argument("--conditional-ranking-negative-mode", choices=["stride", "random", "hard_text"], default="stride")
    parser.add_argument("--conditional-ranking-hard-pool-size", type=int, default=512)
    parser.add_argument("--conditional-ranking-temperature", type=float, default=1.0)
    parser.add_argument("--image-conditional-ranking-coef", type=float, default=0.0)
    parser.add_argument("--speech-conditional-ranking-coef", type=float, default=0.0)
    parser.add_argument("--speech-behavior-kl-coef", type=float, default=0.0)
    parser.add_argument("--speech-behavior-kl-temperature", type=float, default=1.0)
    parser.add_argument("--speech-shared-contrastive-coef", type=float, default=0.0)
    parser.add_argument(
        "--speech-shared-contrastive-temperature", type=float, default=0.07
    )
    parser.add_argument("--image-eval-samples", type=int, default=250)
    parser.add_argument("--speech-eval-samples", type=int, default=250)
    parser.add_argument("--image-prefix-tokens", type=int, default=50)
    parser.add_argument("--audio-prefix-tokens", type=int, default=50)
    parser.add_argument("--encoder-feature-tokens", type=int, default=50)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--audio-max-seconds", type=float, default=0.0, help="Runtime truncation after resampling; 0 preserves full input.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--router-learning-rate", type=float, default=5e-5)
    parser.add_argument("--expert-learning-rate", type=float, default=5e-5)
    parser.add_argument("--expert-selection-json", default="")
    parser.add_argument("--expert-selection-method", choices=["ESFT-Gate", "ESFT-Token"], default="ESFT-Gate")
    parser.add_argument("--expert-update-mode", choices=["full", "lora"], default="full")
    parser.add_argument("--expert-anchor-coefficient", type=float, default=0.01)
    parser.add_argument("--allow-selected-expert-router-tuning", action="store_true")
    parser.add_argument("--retrieval-head-learning-rate", type=float, default=0.0)
    parser.add_argument("--lm-head-learning-rate", type=float, default=1e-5)
    parser.add_argument("--speech-encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--speech-unfreeze-last-blocks", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--speech-unfreeze-layer-norm", action="store_true")
    parser.add_argument("--contrastive-coef", type=float, default=0.1)
    parser.add_argument("--image-contrastive-coef", type=float, default=-1.0)
    parser.add_argument("--speech-contrastive-coef", type=float, default=-1.0)
    parser.add_argument("--center-positive-weight", type=float, default=1.0)
    parser.add_argument("--raw-positive-weight", type=float, default=1.0)
    parser.add_argument("--image-center-positive-weight", type=float, default=-1.0)
    parser.add_argument("--image-raw-positive-weight", type=float, default=-1.0)
    parser.add_argument("--speech-center-positive-weight", type=float, default=-1.0)
    parser.add_argument("--speech-raw-positive-weight", type=float, default=-1.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--image-contrastive-temperature", type=float, default=-1.0)
    parser.add_argument("--speech-contrastive-temperature", type=float, default=-1.0)
    parser.add_argument("--contrastive-negatives", type=int, default=128)
    parser.add_argument("--image-contrastive-negatives", type=int, default=-2)
    parser.add_argument("--speech-contrastive-negatives", type=int, default=-2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=500)
    parser.add_argument("--alignment-pretrain-steps", type=int, default=0)
    parser.add_argument("--alignment-pretrain-log-every", type=int, default=100)
    parser.add_argument("--alignment-pretrain-modalities", default="image,speech")
    parser.add_argument("--train-router-gates", action="store_true")
    parser.add_argument("--train-experts", action="store_true")
    parser.add_argument("--train-lm-head", action="store_true")
    parser.add_argument("--stage-b-checkpoint", default="")
    parser.add_argument("--stage-b-checkpoint-sha256", default="")
    parser.add_argument("--multimodal-initial-checkpoint", default="")
    parser.add_argument("--multimodal-initial-checkpoint-sha256", default="")
    parser.add_argument("--multimodal-initial-manifest", default="")
    parser.add_argument("--speech-initial-checkpoint", default="")
    parser.add_argument("--speech-initial-checkpoint-sha256", default="")
    parser.add_argument("--speech-initial-manifest", default="")
    parser.add_argument(
        "--multimodal-initialization-scope",
        choices=("both", "image", "speech"),
        default="both",
    )
    parser.add_argument("--save-expert-weights", action="store_true", help="Store full expert tensors in trainable-expert ablation checkpoints.")
    return parser.parse_args()


def resolve_runtime_data_dir(
    value: str | Path,
    *,
    strict_development_manifest: bool,
) -> Path:
    raw_value = str(value)
    raw_path = Path(raw_value)
    if strict_development_manifest:
        resolved = raw_path.resolve()
        if (
            not raw_path.is_absolute()
            or raw_path.is_symlink()
            or not raw_path.is_dir()
            or raw_value != str(resolved)
        ):
            raise ValueError(
                "strict development --data-dir must be an absolute canonical directory"
            )
        return raw_path
    return raw_path.expanduser().resolve()


def main() -> None:
    args = parse_args()
    validate_runtime_sample_rate(args)
    validate_speech_behavior_kl_request(args)
    validate_speech_shared_contrastive_request(args)
    if args.development_split_manifest:
        if not _is_full_hex(
            args.development_speech_source_sha256, 64
        ):
            raise ValueError(
                "--development-speech-source-sha256 is required with "
                "--development-split-manifest"
            )
        args.development_speech_source_sha256 = (
            args.development_speech_source_sha256.lower()
        )
    args.data_dir = str(resolve_runtime_data_dir(
        args.data_dir,
        strict_development_manifest=bool(
            args.development_split_manifest
        ),
    ))
    args.output_dir = str(Path(args.output_dir).resolve())
    if args.development_split_manifest:
        split_manifest = Path(args.development_split_manifest)
        reject_forbidden_development_path(
            split_manifest, "Stage A development split manifest"
        )
        if split_manifest.is_symlink() or not split_manifest.is_file():
            raise ValueError(
                "Stage A development split manifest must be a regular file"
            )
        split_manifest = split_manifest.resolve()
        args.development_split_manifest = str(split_manifest)
        args.development_split_manifest_sha256 = sha256_file(split_manifest)
        args.development_split_manifest_source = {
            "path": str(split_manifest),
            "sha256": args.development_split_manifest_sha256,
        }
    else:
        args.development_split_manifest_sha256 = None
        args.development_split_manifest_source = None
    run_provenance = build_stage_a_run_provenance(args, args.final_steps)
    expert_selection_request = validate_expert_selection_request(args)
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    stage_b_checkpoint_state, stage_b_checkpoint_provenance = (
        load_stage_b_initialization_checkpoint(
            args.stage_b_checkpoint, args.stage_b_checkpoint_sha256
        )
    )
    multimodal_checkpoint_state, multimodal_checkpoint_provenance = (
        load_multimodal_initialization_checkpoint(
            args.multimodal_initial_checkpoint,
            args.multimodal_initial_checkpoint_sha256,
            args.multimodal_initial_manifest,
        )
    )
    speech_checkpoint_state, speech_checkpoint_provenance = (
        load_speech_initialization_checkpoint(
            args.speech_initial_checkpoint,
            args.speech_initial_checkpoint_sha256,
            args.speech_initial_manifest,
        )
    )
    validate_initialization_state_disjoint(
        multimodal_checkpoint_state,
        stage_b_checkpoint_state,
        args.multimodal_initialization_scope,
    )
    validate_dual_initialization_request(
        multimodal_checkpoint_state,
        args.multimodal_initialization_scope,
        speech_checkpoint_state,
    )
    data_manifest = load_manifest(data_dir)
    text_rows = read_jsonl(data_dir / "text_tasks.jsonl")
    text_train_blocks = read_jsonl(data_dir / "text_blocks_train.jsonl")
    text_eval_blocks = read_jsonl(data_dir / "text_blocks_eval.jsonl")
    development_split_provenance: Dict[str, Any]
    if args.development_split_manifest:
        if args.image_eval_samples != 137 or args.speech_eval_samples != 137:
            raise ValueError(
                "explicit development split requires image/speech eval samples = 137"
            )
        (
            image_train,
            image_eval,
            audio_train,
            audio_eval,
            development_split_provenance,
        ) = load_development_multimodal_partitions(
            args.development_split_manifest,
            expected_source_commit_sha=run_provenance["source_commit_sha"],
            expected_data_dir=data_dir,
            expected_speech_source_sha256=(
                args.development_speech_source_sha256
            ),
        )
        if (
            development_split_provenance.get("manifest_sha256")
            != run_provenance.get("development_split_manifest_sha256")
        ):
            raise ValueError(
                "E3 development split manifest/run provenance SHA mismatch"
            )
    else:
        image_rows = read_jsonl(data_dir / "image_captions.jsonl")
        audio_rows = read_jsonl(data_dir / "speech_transcripts.jsonl")
        absolutize_media_paths(image_rows, data_dir)
        absolutize_media_paths(audio_rows, data_dir)
        image_train, image_eval = split_tail(image_rows, args.image_eval_samples)
        audio_train, audio_eval = split_tail(audio_rows, args.speech_eval_samples)
        development_split_provenance = {
            "policy": "legacy_tail_split",
            "selection_splits": ["train", "tail_eval"],
            "reserved_unused_split": None,
        }
    strict_audio_integrity = bool(
        development_split_provenance.get("strict_manifest_verified") is True
    )
    args.speech_feature_cache_policy = (
        STRICT_SPEECH_FEATURE_CACHE_POLICY
        if strict_audio_integrity
        else DIAGNOSTIC_SPEECH_FEATURE_CACHE_POLICY
    )
    run_provenance = build_stage_a_run_provenance(args, args.final_steps)
    if args.speech_shared_contrastive_coef > 0.0:
        validate_speech_shared_split_binding(
            development_split_provenance, len(audio_train)
        )
    if not text_train_blocks or not text_eval_blocks or not image_train or not audio_train:
        raise RuntimeError("Real-data manifest has empty train/eval splits")
    calibration_texts = [str(row.get("text") or (str(row.get("prompt", "")) + " " + str(row.get("target", "")))) for row in text_rows[: min(128, len(text_rows))]]
    source_commit_sha = run_provenance["source_commit_sha"]
    manifest = {
        "runai_job_name": run_provenance["runai_job_name"],
        "runai_project": run_provenance["runai_project"],
        "source_commit_sha": source_commit_sha,
        "command_mode": "real-required-runs",
        "base_model": args.base_model,
        "vision_model": args.vision_model,
        "speech_model": args.speech_model,
        "output_dir": run_provenance["resolved_output_root"],
        "data_dir": run_provenance["resolved_data_root"],
        "data_manifest": data_manifest,
        "development_split_provenance": development_split_provenance,
        "conditional_evaluation_policy": {
            "candidate_permutation_policy": str(args.conditional_candidate_permutation),
            "candidate_permutation_seed_source": "control_seed+stable_query_identity_sha256",
            "control_seed": int(args.conditional_control_seed),
            "tie_policy": "strict_pessimistic_epsilon",
            "tie_epsilon": float(args.conditional_tie_epsilon),
        },
        "splits": {
            "text_train_blocks": len(text_train_blocks),
            "text_eval_blocks": len(text_eval_blocks),
            "image_train_pairs": len(image_train),
            "image_eval_pairs": len(image_eval),
            "speech_train_utterances": len(audio_train),
            "speech_eval_utterances": len(audio_eval),
        },
        "expert_selection_request": expert_selection_request,
        "stage_b_initialization": stage_b_checkpoint_provenance,
        "multimodal_initialization": {
            **multimodal_checkpoint_provenance,
            "scope": args.multimodal_initialization_scope,
        },
        "speech_initialization": speech_checkpoint_provenance,
        "run_provenance": run_provenance,
        "args": vars(args),
    }
    save_json(out_dir / "manifest.json", manifest)

    eval_blocks = text_eval_blocks[: args.text_eval_blocks]
    model, tokenizer, meta = load_model(args.base_model, 8, args.aux_coef, gamma=None, capacity_factor=args.capacity_factor)
    e0 = evaluate_text_blocks("E0_top8_teacher_baseline", model, tokenizer, eval_blocks, out_dir, meta, args.max_length, args.eval_batch_size)
    cleanup(model)

    model, tokenizer, meta = load_model(args.base_model, 2, args.aux_coef, gamma=None, capacity_factor=args.capacity_factor)
    e1 = evaluate_text_blocks("E1_hard_top2", model, tokenizer, eval_blocks, out_dir, meta, args.max_length, args.eval_batch_size)
    cleanup(model)

    gamma = calibrate_gamma(args, calibration_texts, out_dir)
    model, tokenizer, meta = load_model(args.base_model, 2, args.aux_coef, gamma=gamma, capacity_factor=args.capacity_factor)
    e2 = evaluate_text_blocks("E2_calibrated_top2", model, tokenizer, eval_blocks, out_dir, {**meta, "gamma": gamma}, args.max_length, args.eval_batch_size)
    cleanup(model)

    e3 = train_real_multimodal(
        "E3_final_multimodal_top2",
        args,
        text_train_blocks,
        text_eval_blocks,
        image_train,
        image_eval,
        audio_train,
        audio_eval,
        gamma=gamma,
        aux_coef=args.aux_coef,
        capacity_factor=args.capacity_factor,
        max_steps=args.final_steps,
        out_dir=out_dir,
        train_router_gates=args.train_router_gates,
        train_experts=args.train_experts,
        expert_selection_path=args.expert_selection_json or None,
        expert_selection_method=args.expert_selection_method,
        initial_checkpoint_state=multimodal_checkpoint_state,
        initial_checkpoint_provenance=multimodal_checkpoint_provenance,
        initial_checkpoint_scope=args.multimodal_initialization_scope,
        speech_initial_checkpoint_state=speech_checkpoint_state,
        speech_initial_checkpoint_provenance=speech_checkpoint_provenance,
        stage_b_checkpoint_state=stage_b_checkpoint_state,
        stage_b_checkpoint_provenance=stage_b_checkpoint_provenance,
        development_split_provenance=development_split_provenance,
    )
    manifest["completion"] = {
        "status": "completed",
        "e3_checkpoint_path": str(Path(e3["checkpoint_path"]).resolve()),
        "e3_checkpoint_sha256": e3["checkpoint_sha256"],
        "e3_checkpoint_size_bytes": e3["checkpoint_size_bytes"],
        "e3_steps": args.final_steps,
    }
    save_json(out_dir / "manifest.json", manifest)
    summary: Dict[str, Any] = {"manifest": manifest, "E0": e0, "E1": e1, "E2": e2, "E3": e3}
    if args.ablation_steps > 0:
        summary["E4"] = train_real_multimodal(
            "E4_no_aux_load_balance_ablation",
            args,
            text_train_blocks,
            text_eval_blocks,
            image_train,
            image_eval,
            audio_train,
            audio_eval,
            gamma=gamma,
            aux_coef=0.0,
            capacity_factor=args.capacity_factor,
            max_steps=args.ablation_steps,
            out_dir=out_dir,
            train_router_gates=args.train_router_gates,
            train_experts=False,
            initial_checkpoint_state=multimodal_checkpoint_state,
            initial_checkpoint_provenance=multimodal_checkpoint_provenance,
            initial_checkpoint_scope=args.multimodal_initialization_scope,
            speech_initial_checkpoint_state=speech_checkpoint_state,
            speech_initial_checkpoint_provenance=speech_checkpoint_provenance,
            stage_b_checkpoint_state=stage_b_checkpoint_state,
            stage_b_checkpoint_provenance=stage_b_checkpoint_provenance,
            development_split_provenance=development_split_provenance,
        )
    if args.capacity_ablation_steps > 0:
        summary["E5"] = train_real_multimodal(
            "E5_capacity_1p25_ablation",
            args,
            text_train_blocks,
            text_eval_blocks,
            image_train,
            image_eval,
            audio_train,
            audio_eval,
            gamma=gamma,
            aux_coef=args.aux_coef,
            capacity_factor=args.capacity_ablation_factor,
            max_steps=args.capacity_ablation_steps,
            out_dir=out_dir,
            train_router_gates=args.train_router_gates,
            train_experts=False,
            initial_checkpoint_state=multimodal_checkpoint_state,
            initial_checkpoint_provenance=multimodal_checkpoint_provenance,
            initial_checkpoint_scope=args.multimodal_initialization_scope,
            speech_initial_checkpoint_state=speech_checkpoint_state,
            speech_initial_checkpoint_provenance=speech_checkpoint_provenance,
            stage_b_checkpoint_state=stage_b_checkpoint_state,
            stage_b_checkpoint_provenance=stage_b_checkpoint_provenance,
            development_split_provenance=development_split_provenance,
        )
    if args.expert_ablation_steps > 0:
        summary["E6"] = train_real_multimodal(
            "E6_trainable_experts_ablation",
            args,
            text_train_blocks,
            text_eval_blocks,
            image_train,
            image_eval,
            audio_train,
            audio_eval,
            gamma=gamma,
            aux_coef=args.aux_coef,
            capacity_factor=args.capacity_factor,
            max_steps=args.expert_ablation_steps,
            out_dir=out_dir,
            train_router_gates=True,
            train_experts=True,
            initial_checkpoint_state=multimodal_checkpoint_state,
            initial_checkpoint_provenance=multimodal_checkpoint_provenance,
            initial_checkpoint_scope=args.multimodal_initialization_scope,
            speech_initial_checkpoint_state=speech_checkpoint_state,
            speech_initial_checkpoint_provenance=speech_checkpoint_provenance,
            stage_b_checkpoint_state=stage_b_checkpoint_state,
            stage_b_checkpoint_provenance=stage_b_checkpoint_provenance,
            development_split_provenance=development_split_provenance,
        )
    save_json(out_dir / "summary.json", summary)
    print(json.dumps({"summary_path": str(out_dir / "summary.json"), "experiments": sorted(k for k in summary if k.startswith("E")), "real_subset": True}, sort_keys=True))


if __name__ == "__main__":
    main()
