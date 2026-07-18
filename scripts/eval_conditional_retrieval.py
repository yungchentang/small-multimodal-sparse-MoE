"""Conditional caption/transcript matching for trained OLMoE multimodal adapters."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from scripts import build_evaluation_result_manifest as result_manifest
    from scripts.analyze_paired_controls import (
        group_aware_chance_r_at_1,
        image_group_identity,
        positive_indices_for_group,
    )
    from scripts.sealed_position_allocator import (
        ALLOCATOR_NAME,
        ALLOCATOR_VERSION,
        AssignmentPlanError,
        enforce_gold_position_assignment,
        positions_for_query_indices,
        validate_allocator_manifest,
        validate_executed_positions,
    )
except ImportError:  # Direct execution from the scripts directory.
    import build_evaluation_result_manifest as result_manifest  # type: ignore[no-redef]
    from analyze_paired_controls import (  # type: ignore[no-redef]
        group_aware_chance_r_at_1,
        image_group_identity,
        positive_indices_for_group,
    )
    from sealed_position_allocator import (  # type: ignore[no-redef]
        ALLOCATOR_NAME,
        ALLOCATOR_VERSION,
        AssignmentPlanError,
        enforce_gold_position_assignment,
        positions_for_query_indices,
        validate_allocator_manifest,
        validate_executed_positions,
    )

try:
    from scripts.freeze_evaluation_protocol import (
        ProtocolError as FrozenProtocolError,
        require_current_allocator_source_fingerprint,
    )
except ImportError:  # Direct execution from the scripts directory.
    from freeze_evaluation_protocol import (  # type: ignore[no-redef]
        ProtocolError as FrozenProtocolError,
        require_current_allocator_source_fingerprint,
    )

from training.olmoe_required_runs import (
    base_model_identity,
    load_encoders,
    load_model,
)
from training.olmoe_real_subset_runs import (
    FeatureCache,
    conditional_query_identity,
    load_stage_b_initialization_checkpoint,
    lm_embeddings_are_tied,
    permute_candidates_for_query,
    restore_stage_b_student_initialization,
    restore_training_checkpoint,
    tie_aware_nll_evidence,
    absolutize_media_paths,
    make_wrapper,
    read_jsonl,
    resolve_media_path,
    split_tail,
    tokenize_prompt_targets,
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


CHECKPOINT_PROVENANCE_FIELDS = (
    "run_provenance",
    "gamma_provenance",
    "trainable_meta",
    "last_row",
)


def checkpoint_protocol_digests(
    state: Mapping[str, Any],
) -> Dict[str, str]:
    args = state.get("args")
    if not isinstance(args, Mapping):
        raise ValueError("E3 checkpoint is missing args mapping")
    provenance = {
        field: state.get(field)
        for field in CHECKPOINT_PROVENANCE_FIELDS
    }
    return {
        "args_sha256": canonical_sha256(args),
        "provenance_sha256": canonical_sha256(provenance),
        "args_provenance_sha256": canonical_sha256({
            "args": dict(args),
            "provenance": provenance,
        }),
    }


def load_checkpoint_bound_gamma(
    run_output_dir: Path,
    checkpoint_state: Mapping[str, Any],
    evaluation_scope: str,
) -> Tuple[List[float], Dict[str, Any]]:
    checkpoint_args = checkpoint_state.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("E3 checkpoint is missing args for gamma provenance")
    recorded_output_dir = checkpoint_args.get("output_dir")
    if not recorded_output_dir:
        raise ValueError("E3 checkpoint args are missing output_dir")
    supplied_root = run_output_dir.expanduser().resolve(strict=True)
    checkpoint_root = Path(str(recorded_output_dir)).expanduser().resolve(strict=True)
    if supplied_root != checkpoint_root:
        raise ValueError(
            "RUN_OUTPUT_DIR does not match E3 checkpoint args.output_dir: "
            f"supplied={supplied_root} checkpoint={checkpoint_root}"
        )
    gamma_path = (checkpoint_root / "calibration" / "gamma.json").resolve(
        strict=True
    )
    checkpoint_gamma = checkpoint_state.get("gamma_provenance")
    checkpoint_bound = isinstance(checkpoint_gamma, Mapping)
    if not checkpoint_bound and evaluation_scope in {"development", "final"}:
        raise ValueError(
            "claim evaluation requires checkpoint-stored gamma provenance"
        )

    expected_sha256: str | None = None
    if checkpoint_bound:
        checkpoint_gamma = dict(checkpoint_gamma)
        expected_sha256 = checkpoint_gamma.get("sha256")
        expected_path = checkpoint_gamma.get("path")
        expected_relative = checkpoint_gamma.get("relative_path")
        expected_output_dir = checkpoint_gamma.get("output_dir")
        expected_size = checkpoint_gamma.get("size_bytes")
        if (
            not isinstance(expected_sha256, str)
            or SHA256_RE.fullmatch(expected_sha256) is None
        ):
            raise ValueError(
                "E3 checkpoint gamma provenance is missing exact SHA256"
            )
        if expected_relative != "calibration/gamma.json":
            raise ValueError(
                "E3 checkpoint gamma relative path identity is invalid"
            )
        if (
            not isinstance(expected_path, str)
            or Path(expected_path).expanduser().resolve(strict=True) != gamma_path
        ):
            raise ValueError(
                "E3 checkpoint gamma path identity does not match current run"
            )
        if (
            not isinstance(expected_output_dir, str)
            or Path(expected_output_dir).expanduser().resolve(strict=True)
            != checkpoint_root
        ):
            raise ValueError(
                "E3 checkpoint gamma output directory identity is invalid"
            )

    gamma_bytes = gamma_path.read_bytes()
    gamma_sha256 = _sha256_bytes(gamma_bytes)
    if checkpoint_bound and (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size != len(gamma_bytes)
    ):
        raise ValueError(
            "gamma JSON size disagrees with E3 checkpoint provenance"
        )
    if expected_sha256 is not None and gamma_sha256 != expected_sha256:
        raise ValueError(
            "gamma JSON SHA256 disagrees with E3 checkpoint: "
            f"expected={expected_sha256} observed={gamma_sha256}"
        )
    try:
        data = json.loads(gamma_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid checkpoint-bound gamma JSON: {gamma_path}"
        ) from exc
    gamma = data.get("gamma") if isinstance(data, Mapping) else None
    if not isinstance(gamma, list) or not gamma:
        raise ValueError(f"invalid checkpoint-bound gamma JSON: {gamma_path}")
    return [float(value) for value in gamma], {
        "path": str(gamma_path),
        "relative_path": "calibration/gamma.json",
        "sha256": gamma_sha256,
        "checkpoint_expected_sha256": expected_sha256,
        "size_bytes": len(gamma_bytes),
        "checkpoint_output_dir": str(checkpoint_root),
        "checkpoint_bound": checkpoint_bound,
    }


CHECKPOINT_ARCHITECTURE_DEFAULTS = {
    "base_model": "allenai/OLMoE-1B-7B-0924",
    "vision_model": "openai/clip-vit-base-patch32",
    "speech_model": "openai/whisper-base.en",
    "speech_target_space": "olmoe_text_hidden",
    "image_alignment_target": "clip_text",
    "alignment_prefix_residual": False,
    "image_bridge_type": "query_resampler",
    "audio_bridge_type": "query_resampler",
    "bridge_num_heads": 4,
    "image_prefix_tokens": 50,
    "audio_prefix_tokens": 64,
    "encoder_feature_tokens": 100,
    "sample_rate": 16000,
    "audio_max_seconds": 0.0,
}

SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


def apply_checkpoint_architecture_args(args, state: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint_args = state.get("args")
    if not isinstance(checkpoint_args, dict):
        raise ValueError("multimodal checkpoint is missing architecture args")
    for field in ("capacity_factor", "aux_coef"):
        if field in checkpoint_args and float(getattr(args, field)) != float(checkpoint_args[field]):
            raise ValueError(
                f"evaluation {field} disagrees with checkpoint: "
                f"cli={getattr(args, field)!r} checkpoint={checkpoint_args[field]!r}"
            )
    resolved: Dict[str, Any] = {}
    for field, default in CHECKPOINT_ARCHITECTURE_DEFAULTS.items():
        value = checkpoint_args.get(field, default)
        setattr(args, field, value)
        resolved[field] = value
    return resolved


def validate_eval_top_k(top_k: int, sealed_protocol: bool) -> int:
    value = int(top_k)
    if value not in (2, 4, 8):
        raise ValueError("evaluation top_k must be one of 2, 4, or 8")
    if sealed_protocol and value != 2:
        raise ValueError("sealed evaluation is frozen to final Top-2 inference")
    return value


def validate_e3_stage_b_provenance(
    state: Mapping[str, Any],
    stage_b_checkpoint: str,
    stage_b_checkpoint_sha256: str,
) -> Dict[str, Any] | None:
    """Bind an external Stage-B input to both provenance copies in an E3 checkpoint."""
    trainable_meta = state.get("trainable_meta")
    last_row = state.get("last_row")
    checkpoint_args = state.get("args")
    stage_b_initialization = (
        trainable_meta.get("stage_b_initialization")
        if isinstance(trainable_meta, Mapping)
        else None
    )
    args_declare_stage_b = bool(
        isinstance(checkpoint_args, Mapping)
        and (
            checkpoint_args.get("stage_b_checkpoint")
            or checkpoint_args.get("stage_b_checkpoint_sha256")
        )
    )
    last_row_declares_stage_b = bool(
        isinstance(last_row, Mapping)
        and (
            last_row.get("stage_b_checkpoint_state_restored")
            or last_row.get("source_stage_b_checkpoint_sha256")
        )
    )
    declares_stage_b = (
        bool(stage_b_initialization)
        or args_declare_stage_b
        or last_row_declares_stage_b
    )

    if not declares_stage_b:
        if stage_b_checkpoint or stage_b_checkpoint_sha256:
            raise ValueError(
                "Stage-B checkpoint input was supplied, but the E3 checkpoint does not "
                "declare Stage-B initialization"
            )
        return None
    if not stage_b_checkpoint or not stage_b_checkpoint_sha256:
        raise ValueError(
            "E3 checkpoint declares Stage-B initialization; both "
            "--stage-b-checkpoint and --stage-b-checkpoint-sha256 are required"
        )
    if SHA256_RE.fullmatch(stage_b_checkpoint_sha256) is None:
        raise ValueError(
            "--stage-b-checkpoint-sha256 must be an exact 64-character SHA256"
        )
    if not isinstance(stage_b_initialization, Mapping):
        raise ValueError(
            "E3 checkpoint declares Stage-B initialization but trainable_meta "
            "provenance is missing"
        )
    if stage_b_initialization.get("state_restored") is not True:
        raise ValueError("E3 trainable_meta does not verify restored Stage-B state")
    if (
        not isinstance(last_row, Mapping)
        or last_row.get("stage_b_checkpoint_state_restored") is not True
    ):
        raise ValueError("E3 last_row does not verify restored Stage-B state")

    supplied_path = Path(stage_b_checkpoint).expanduser().resolve(strict=True)
    recorded_path_value = stage_b_initialization.get("path")
    if not recorded_path_value:
        raise ValueError(
            "E3 trainable_meta Stage-B provenance is missing checkpoint path"
        )
    recorded_path = Path(str(recorded_path_value)).expanduser().resolve()
    if supplied_path != recorded_path:
        raise ValueError(
            "Stage-B checkpoint path does not match E3 trainable_meta provenance: "
            f"supplied={supplied_path} recorded={recorded_path}"
        )

    supplied_sha256 = stage_b_checkpoint_sha256.lower()
    provenance_hashes = {
        "trainable_meta": stage_b_initialization.get("sha256"),
        "last_row": last_row.get("source_stage_b_checkpoint_sha256"),
    }
    if (
        isinstance(checkpoint_args, Mapping)
        and checkpoint_args.get("stage_b_checkpoint_sha256")
    ):
        provenance_hashes["args"] = checkpoint_args.get(
            "stage_b_checkpoint_sha256"
        )
    for location, value in provenance_hashes.items():
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise ValueError(
                f"E3 {location} Stage-B SHA256 provenance is missing or malformed"
            )
        if value.lower() != supplied_sha256:
            raise ValueError(
                f"Stage-B checkpoint SHA256 does not match E3 {location} provenance: "
                f"supplied={supplied_sha256} recorded={value.lower()}"
            )
    if (
        isinstance(checkpoint_args, Mapping)
        and checkpoint_args.get("stage_b_checkpoint")
    ):
        args_path = (
            Path(str(checkpoint_args["stage_b_checkpoint"])).expanduser().resolve()
        )
        if args_path != supplied_path:
            raise ValueError(
                "Stage-B checkpoint path does not match E3 args provenance: "
                f"supplied={supplied_path} recorded={args_path}"
            )
    return {
        "path": str(supplied_path),
        "sha256": supplied_sha256,
        "declared_by_trainable_meta": True,
        "declared_by_last_row": True,
    }


def extract_e3_source_checkpoint_hashes(
    state: Mapping[str, Any],
    verified_stage_b_sha256: str | None,
) -> Dict[str, str | None]:
    trainable_meta = state.get("trainable_meta")
    trainable_meta = trainable_meta if isinstance(trainable_meta, Mapping) else {}
    roles = {
        "stage_b": "stage_b_initialization",
        "multimodal_initial": "multimodal_initialization",
        "speech_initial": "speech_initialization",
    }
    hashes: Dict[str, str | None] = {}
    for role, metadata_key in roles.items():
        metadata = trainable_meta.get(metadata_key)
        value = metadata.get("sha256") if isinstance(metadata, Mapping) else None
        if value is not None and (
            not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
        ):
            raise ValueError(
                f"E3 trainable_meta {metadata_key} SHA256 is malformed"
            )
        hashes[role] = value.lower() if isinstance(value, str) else None
    if hashes["stage_b"] != verified_stage_b_sha256:
        raise ValueError(
            "verified Stage-B hash differs from E3 source checkpoint provenance"
        )
    return hashes


def load_verified_stage_b_for_e3(
    state: Mapping[str, Any],
    stage_b_checkpoint: str,
    stage_b_checkpoint_sha256: str,
) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
    declared = validate_e3_stage_b_provenance(
        state, stage_b_checkpoint, stage_b_checkpoint_sha256
    )
    if declared is None:
        return None, {}
    stage_b_state, file_provenance = load_stage_b_initialization_checkpoint(
        stage_b_checkpoint, stage_b_checkpoint_sha256
    )
    if stage_b_state is None:
        raise ValueError("verified Stage-B checkpoint unexpectedly produced no state")
    return stage_b_state, {**file_provenance, **declared}


def validate_e3_training_checkpoint_state(
    state: Mapping[str, Any],
    layer_count: int,
) -> Tuple[Dict[int | str, Sequence[int]] | None, Dict[str, Any] | None]:
    if not isinstance(state, dict):
        raise TypeError("E3 checkpoint state must be a dict")
    trainable_meta = state.get("trainable_meta")
    if not isinstance(trainable_meta, Mapping):
        raise ValueError("E3 checkpoint is missing trainable_meta")
    required_components = (
        "image_resampler",
        "audio_resampler",
        "image_retrieval_head",
        "audio_retrieval_head",
        "image_direct_retrieval_head",
        "audio_direct_retrieval_head",
    )
    missing_components = [
        name for name in required_components if name not in state
    ]
    if missing_components:
        raise ValueError(
            f"E3 checkpoint is missing required adapter state: {missing_components}"
        )

    speech_names = {
        str(name)
        for name in trainable_meta.get("speech_encoder_trainable_names", [])
    }
    speech_state = state.get("speech_encoder_trainable_state")
    if speech_names:
        if not isinstance(speech_state, Mapping):
            raise ValueError(
                "E3 checkpoint is missing required speech encoder trainable state"
            )
        if set(speech_state) != speech_names:
            raise ValueError(
                "E3 speech encoder state does not match trainable_meta names: "
                f"missing={sorted(speech_names - set(speech_state))} "
                f"unexpected={sorted(set(speech_state) - speech_names)}"
            )
    elif speech_state is not None:
        raise ValueError(
            "E3 checkpoint contains speech encoder state without trainable_meta names"
        )

    expected_layer_keys = {
        f"layer_{index}" for index in range(int(layer_count))
    }
    router_state = state.get("router_gates")
    if trainable_meta.get("train_router_gates"):
        if not isinstance(router_state, Mapping) or set(router_state) != expected_layer_keys:
            raise ValueError(
                "E3 trainable-router checkpoint must cover every runtime layer"
            )

    checkpoint_args = state.get("args")
    dynamic_required = bool(
        isinstance(checkpoint_args, Mapping)
        and float(checkpoint_args.get("dynamic_expert_bias_lr", 0.0)) > 0.0
    )
    dynamic_state = state.get("dynamic_expert_bias")
    if dynamic_required:
        if not isinstance(dynamic_state, Mapping) or set(dynamic_state) != expected_layer_keys:
            raise ValueError(
                "E3 dynamic-expert-bias checkpoint must cover every runtime layer"
            )
    elif dynamic_state is not None and (
        not isinstance(dynamic_state, Mapping)
        or set(dynamic_state) != expected_layer_keys
    ):
        raise ValueError(
            "E3 dynamic-expert-bias state does not cover every runtime layer"
        )

    if trainable_meta.get("train_lm_head"):
        if state.get("lm_output_embeddings") is None:
            raise ValueError(
                "E3 trainable-LM-head checkpoint is missing output embeddings"
            )
        embeddings_tied = state.get("lm_embeddings_tied")
        if not isinstance(embeddings_tied, bool):
            raise ValueError(
                "E3 trainable-LM-head checkpoint is missing tied-embedding metadata"
            )
        if (
            not embeddings_tied
            and state.get("lm_input_embeddings") is None
        ):
            raise ValueError(
                "E3 untied trainable-LM-head checkpoint is missing input embeddings"
            )
    if trainable_meta.get("train_experts"):
        if "experts" not in state:
            raise ValueError("E3 trainable-expert checkpoint is missing expert state")
        raise ValueError(
            "external evaluation cannot restore full-expert checkpoints with "
            "restore_training_checkpoint"
        )

    selected_training = bool(trainable_meta.get("selected_expert_training"))
    selected_state = state.get("selected_experts")
    if selected_training:
        if not isinstance(selected_state, Mapping):
            raise ValueError(
                "E3 selected-expert checkpoint is missing selected_experts rows"
            )
        expected_selected_ids = trainable_meta.get(
            "selected_expert_ids_by_layer"
        )
        expected_provenance = trainable_meta.get("expert_selection_provenance")
        if not isinstance(expected_selected_ids, Mapping):
            raise ValueError(
                "E3 selected-expert trainable_meta is missing selected IDs"
            )
        if not isinstance(expected_provenance, Mapping):
            raise ValueError(
                "E3 selected-expert trainable_meta is missing selection provenance"
            )
        required_provenance = {
            "selection_json_sha256",
            "selection_method",
            "selection_scope",
            "selected_expert_ids_by_layer",
        }
        if not required_provenance.issubset(expected_provenance):
            raise ValueError(
                "E3 selected-expert trainable_meta has incomplete selection provenance"
            )
        return dict(expected_selected_ids), dict(expected_provenance)
    if selected_state is not None:
        raise ValueError(
            "E3 checkpoint contains selected_experts without selected-expert training metadata"
        )
    return None, None


def restore_evaluation_checkpoint_state(
    wrapper,
    e3_state: Mapping[str, Any],
    stage_b_state: Mapping[str, Any] | None,
    expected_base_model: str,
    runtime_base_model_identity: Mapping[str, Any],
    speech_model=None,
) -> Dict[str, Any]:
    """Reproduce training's base -> Stage-B student -> full E3 restore order."""
    restoration_order = ["base_model"]
    stage_b_restore: Dict[str, Any] = {}
    if stage_b_state is not None:
        stage_b_restore = restore_stage_b_student_initialization(
            wrapper,
            dict(stage_b_state),
            expected_base_model,
            runtime_base_model_identity,
        )
        restoration_order.append("stage_b_student_checkpoint")

    expected_selected_ids, expected_selection_provenance = (
        validate_e3_training_checkpoint_state(
            e3_state, len(wrapper.lm.model.layers)
        )
    )
    if e3_state.get("trainable_meta", {}).get("train_lm_head"):
        runtime_embeddings_tied = lm_embeddings_are_tied(wrapper.lm)
        if e3_state.get("lm_embeddings_tied") is not runtime_embeddings_tied:
            raise ValueError(
                "E3 tied-embedding metadata disagrees with runtime model"
            )
    restore_training_checkpoint(
        wrapper,
        dict(e3_state),
        speech_model=speech_model,
        expected_selected_expert_ids=expected_selected_ids,
        expected_selection_provenance=expected_selection_provenance,
    )
    restoration_order.append("e3_training_checkpoint")
    dynamic_state = e3_state.get("dynamic_expert_bias")
    return {
        "restoration_order": restoration_order,
        "stage_b_state_restored": stage_b_state is not None,
        "stage_b_restore": stage_b_restore,
        "e3_state_overlaid": True,
        "e3_required_state_validated": True,
        "adapter_state_restored": True,
        "retrieval_head_state_restored": True,
        "dynamic_expert_bias_state_restored": bool(dynamic_state),
        "dynamic_expert_bias_loaded_layers": (
            len(dynamic_state) if isinstance(dynamic_state, Mapping) else 0
        ),
        "router_state_restored": bool(e3_state.get("router_gates")),
        "lm_output_embeddings_restored": (
            e3_state.get("lm_output_embeddings") is not None
        ),
        "lm_input_embeddings_restored": (
            e3_state.get("lm_input_embeddings") is not None
        ),
        "speech_encoder_state_restored": (
            e3_state.get("speech_encoder_trainable_state") is not None
        ),
        "selected_expert_state_restored": (
            e3_state.get("selected_experts") is not None
        ),
        "selected_expert_provenance_validated": (
            expected_selection_provenance is not None
        ),
        "full_expert_state_restored": False,
    }


def load_trained_wrapper(args, checkpoint_bytes: bytes | None = None):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve(strict=True)
    payload = (
        checkpoint_path.read_bytes()
        if checkpoint_bytes is None
        else checkpoint_bytes
    )
    e3_checkpoint_sha256 = _sha256_bytes(payload)
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    verify_file_sha256(
        checkpoint_path, e3_checkpoint_sha256, "E3 checkpoint"
    )
    if not isinstance(state, Mapping):
        raise TypeError("E3 checkpoint payload must be a mapping")
    gamma, gamma_provenance = load_checkpoint_bound_gamma(
        Path(args.run_output_dir), state, str(args.evaluation_scope)
    )
    checkpoint_architecture = apply_checkpoint_architecture_args(args, state)
    stage_b_state, stage_b_provenance = load_verified_stage_b_for_e3(
        state,
        str(args.stage_b_checkpoint),
        str(args.stage_b_checkpoint_sha256),
    )
    dynamic_bias_state = state.get("dynamic_expert_bias")
    model, tokenizer, meta = load_model(
        args.base_model,
        validate_eval_top_k(args.top_k, False),
        args.aux_coef,
        gamma=gamma,
        capacity_factor=args.capacity_factor,
        dynamic_expert_bias=bool(dynamic_bias_state),
        pre_routing_identity_fn=lambda loaded_model: base_model_identity(
            loaded_model, args.base_model
        ),
    )
    runtime_base_model_identity = meta.pop("pre_routing_model_identity", None)
    if not isinstance(runtime_base_model_identity, Mapping):
        raise ValueError(
            "external evaluation pre-routing base-model identity is missing"
        )
    device = next(model.parameters()).device
    (
        image_processor,
        vision_model,
        speech_processor,
        speech_model,
    ) = load_encoders(args.vision_model, args.speech_model, device)
    wrapper = make_wrapper(model, vision_model, speech_model, args).to(device)
    checkpoint_restoration = restore_evaluation_checkpoint_state(
        wrapper,
        state,
        stage_b_state,
        str(args.base_model),
        runtime_base_model_identity,
        speech_model=speech_model,
    )
    source_checkpoint_hashes = extract_e3_source_checkpoint_hashes(
        state, stage_b_provenance.get("sha256")
    )
    checkpoint_restoration.update({
        "e3_checkpoint_path": str(checkpoint_path),
        "e3_checkpoint_sha256": e3_checkpoint_sha256,
        "stage_b_checkpoint": stage_b_provenance or None,
        "source_checkpoint_hashes": source_checkpoint_hashes,
    })
    meta["dynamic_expert_bias_loaded_layers"] = int(
        checkpoint_restoration["dynamic_expert_bias_loaded_layers"]
    )
    meta["checkpoint_architecture"] = checkpoint_architecture
    meta["gamma_provenance"] = gamma_provenance
    meta["checkpoint_restoration"] = checkpoint_restoration
    wrapper.eval()
    return (
        wrapper,
        tokenizer,
        meta,
        image_processor,
        vision_model,
        speech_processor,
        speech_model,
        device,
    )


def per_example_nll(logits: torch.Tensor, labels: torch.Tensor, prefix_len: int) -> torch.Tensor:
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


def recall_from_ranks(ranks: Sequence[int], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total = max(1, len(ranks))
    for k in (1, 5, 10):
        out[f"{prefix}_r_at_{k}"] = sum(1 for rank in ranks if rank < k) / total
    out[f"{prefix}_mean_rank"] = sum(r + 1 for r in ranks) / total
    return out


def tie_aware_metrics(
    ranks: Sequence[int], evidence: Sequence[Dict[str, Any]], prefix: str
) -> Dict[str, Any]:
    if len(ranks) != len(evidence):
        raise ValueError("rank/evidence length mismatch")
    total = max(1, len(ranks))
    tie_count = sum(int(row["gold_tie_count"]) for row in evidence)
    tie_queries = sum(bool(row["has_gold_tie"]) for row in evidence)
    return {
        **recall_from_ranks(ranks, prefix),
        f"{prefix}_strict_r_at_1": float(sum(int(rank) == 0 for rank in ranks) / total),
        f"{prefix}_mrr": float(
            sum(float(row["reciprocal_rank"]) for row in evidence) / total
        ),
        f"{prefix}_mean_gold_nll_margin": float(
            sum(float(row["gold_nll_margin"]) for row in evidence) / total
        ),
        f"{prefix}_tie_count": int(tie_count),
        f"{prefix}_tie_query_count": int(tie_queries),
        f"{prefix}_tie_rate": float(tie_queries / total),
    }


def bootstrap_r_at_1_ci(ranks: Sequence[int], samples: int, seed: int) -> Dict[str, float]:
    """Return a non-parametric 95% CI for R@1 over query ranks."""
    if not ranks or samples <= 0:
        return {}
    hits = torch.tensor([1.0 if int(rank) == 0 else 0.0 for rank in ranks], dtype=torch.float32)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    values: List[float] = []
    n = int(hits.numel())
    for _ in range(int(samples)):
        idx = torch.randint(0, n, (n,), generator=generator)
        values.append(float(hits[idx].mean().item()))
    values.sort()
    lo_i = max(0, min(len(values) - 1, int(0.025 * (len(values) - 1))))
    hi_i = max(0, min(len(values) - 1, int(0.975 * (len(values) - 1))))
    return {
        "r_at_1_bootstrap_ci_low": float(values[lo_i]),
        "r_at_1_bootstrap_ci_high": float(values[hi_i]),
        "r_at_1_bootstrap_samples": int(samples),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


EVALUATION_SCOPES = ("diagnostic", "development", "final")
CLAIM_EVALUATION_SCOPES = {"development", "final"}
METRIC_AFFECTING_ARG_FIELDS = (
    "evaluation_scope",
    "base_model",
    "vision_model",
    "speech_model",
    "speech_target_space",
    "alignment_prefix_residual",
    "max_length",
    "capacity_factor",
    "aux_coef",
    "top_k",
    "image_prefix_tokens",
    "audio_prefix_tokens",
    "encoder_feature_tokens",
    "sample_rate",
    "audio_max_seconds",
    "image_eval_samples",
    "speech_eval_samples",
    "conditional_queries",
    "conditional_candidates",
    "conditional_negatives",
    "conditional_batch_size",
    "negative_mode",
    "eval_path",
    "prefix_control",
    "control_seed",
    "candidate_seed",
    "candidate_permutation",
    "tie_epsilon",
    "randomize_positive_position",
    "protocol_name",
    "eval_split_name",
    "query_offset",
    "candidate_offset",
    "bootstrap_samples",
    "bootstrap_seed",
)
EVALUATION_IDENTITY_REQUIRED_FIELDS = {
    "source_commit_sha",
    "runai_job_name",
    "runai_project",
    "evaluation_scope",
    "eval_split_name",
    "strict_control",
    "condition",
    "prefix_control",
    "eval_path",
    "control_seed",
    "candidate_seed",
    "negative_mode",
    "requested_query_count",
    "image_query_count",
    "speech_query_count",
    "requested_candidate_count",
    "image_candidate_count",
    "speech_candidate_count",
    "conditional_negatives",
    "conditional_candidates",
    "query_offset",
    "candidate_offset",
    "tie_epsilon",
    "bootstrap_seed",
    "bootstrap_samples",
    "image_eval_samples",
    "speech_eval_samples",
    "metric_affecting_args",
    "checkpoint_architecture",
    "protocol_name",
    "protocol_manifest_path",
    "protocol_manifest_sha256",
    "protocol_content_sha256",
    "frozen_input_hashes",
    "image_manifest_path",
    "image_manifest_sha256",
    "speech_manifest_path",
    "speech_manifest_sha256",
    "gamma_path",
    "gamma_sha256",
    "evaluator_path",
    "evaluator_sha256",
    "image_cache_identity_sha256",
    "audio_cache_identity_sha256",
    "image_cache_payload_set_sha256",
    "audio_cache_payload_set_sha256",
    "image_produced_features_sha256",
    "audio_produced_features_sha256",
    "feature_cache_policy",
    "frozen_evaluation_run_id",
    "frozen_evaluation_cell_id",
    "frozen_evaluation_control",
    "e3_checkpoint_path",
    "source_run_manifest_sha256",
    "e3_checkpoint_sha256",
    "stage_b_checkpoint_sha256",
    "source_checkpoint_hashes",
    "restoration_order",
}


def metric_affecting_args(args) -> Dict[str, Any]:
    return {
        field: getattr(args, field)
        for field in METRIC_AFFECTING_ARG_FIELDS
    }


def validate_evaluation_scope(
    evaluation_scope: str,
    image_manifest: str,
    speech_manifest: str,
    protocol_manifest: str,
    per_query_output: str,
    feature_cache_dir: str,
) -> bool:
    if evaluation_scope not in EVALUATION_SCOPES:
        raise ValueError(
            f"unsupported evaluation scope: {evaluation_scope!r}"
        )
    claim_scope = evaluation_scope in CLAIM_EVALUATION_SCOPES
    if claim_scope and (not image_manifest or not speech_manifest):
        raise ValueError(
            "development/final evaluation requires explicit image and speech manifests"
        )
    if evaluation_scope == "final":
        missing = [
            name
            for name, value in {
                "protocol_manifest": protocol_manifest,
                "per_query_output": per_query_output,
                "feature_cache_dir": feature_cache_dir,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                f"final evaluation requires explicit artifacts: {missing}"
            )
    return claim_scope


def prepare_feature_cache_root(
    cache_root: Path,
    evaluation_scope: str,
    output_paths: Sequence[Path],
) -> Tuple[Path, Dict[str, Any]]:
    resolved = cache_root.expanduser().resolve()
    if evaluation_scope != "final":
        return resolved, {
            "mode": "verified_read_write",
            "preexisting_root_allowed": True,
            "cache_reads_allowed": True,
            "writes": "atomic_replace_or_create",
            "feature_source": "verified_media_paths",
        }
    output_parents = {
        path.expanduser().resolve().parent for path in output_paths
    }
    if len(output_parents) != 1:
        raise ValueError(
            "final metrics, per-query output, and feature cache must share one new run root"
        )
    output_root = next(iter(output_parents))
    if not path_is_within(resolved, output_root) or resolved == output_root:
        raise ValueError(
            "final feature cache must be rooted under the new evaluation output directory"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"final feature cache root must not preexist: {resolved}"
        ) from exc
    return resolved, {
        "mode": "exclusive_write_only_recompute",
        "preexisting_root_allowed": False,
        "cache_reads_allowed": False,
        "writes": "atomic_exclusive_per_payload",
        "feature_source": "verified_frozen_media_snapshots",
    }


def _verified_frozen_file(
    record: Mapping[str, Any], label: str
) -> Dict[str, Any]:
    if record.get("type") != "file":
        raise ValueError(f"frozen input {label} is not a file fingerprint")
    path_value = record.get("path")
    expected_sha256 = record.get("sha256")
    if (
        not isinstance(path_value, str)
        or not isinstance(expected_sha256, str)
        or SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise ValueError(f"frozen input {label} has incomplete identity")
    raw_path = Path(path_value).expanduser()
    if raw_path.is_symlink():
        raise ValueError(f"frozen input {label} cannot be a symlink")
    path = raw_path.resolve(strict=True)
    payload = path.read_bytes()
    actual_sha256 = _sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"frozen input {label} SHA256 mismatch: "
            f"expected={expected_sha256} observed={actual_sha256}"
        )
    expected_size = record.get("bytes")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size != len(payload)
    ):
        raise ValueError(f"frozen input {label} size mismatch")
    return {
        "label": label,
        "type": "file",
        "path": str(path),
        "sha256": actual_sha256,
        "size_bytes": len(payload),
        "bytes": payload,
    }


def _verified_frozen_directory(
    record: Mapping[str, Any], label: str
) -> Dict[str, Any]:
    path_value = record.get("path")
    expected_sha256 = record.get("sha256")
    expected_files = record.get("files")
    if (
        record.get("type") != "directory"
        or not isinstance(path_value, str)
        or not isinstance(expected_sha256, str)
        or SHA256_RE.fullmatch(expected_sha256) is None
        or not isinstance(expected_files, list)
    ):
        raise ValueError(
            f"frozen input {label} has incomplete directory identity"
        )
    raw_path = Path(path_value).expanduser()
    if raw_path.is_symlink():
        raise ValueError(f"frozen input {label} cannot be a symlink")
    path = raw_path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"frozen input {label} is not a directory")
    actual_files: List[Dict[str, Any]] = []
    for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_symlink():
            raise ValueError(
                f"frozen input {label} contains a symlink: {child}"
            )
        if not child.is_file():
            continue
        payload = child.read_bytes()
        actual_files.append({
            "relative_path": child.relative_to(path).as_posix(),
            "sha256": _sha256_bytes(payload),
            "bytes": len(payload),
        })
    if (
        actual_files != expected_files
        or canonical_sha256(actual_files) != expected_sha256
        or record.get("file_count") != len(actual_files)
    ):
        raise ValueError(
            f"frozen input {label} directory SHA256 mismatch"
        )
    return {
        "label": label,
        "type": "directory",
        "path": str(path),
        "sha256": expected_sha256,
        "file_count": len(actual_files),
    }


def load_verified_frozen_protocol(
    protocol_path: Path,
    image_manifest_path: Path,
    speech_manifest_path: Path,
    evaluator_path: Path,
    checkpoint_path: Path,
) -> Tuple[Dict[str, Any], str, Dict[str, Dict[str, Any]]]:
    raw_protocol_path = protocol_path.expanduser()
    if raw_protocol_path.is_symlink():
        raise ValueError("protocol manifest cannot be a symlink")
    resolved_protocol_path = raw_protocol_path.resolve(strict=True)
    protocol_bytes = resolved_protocol_path.read_bytes()
    protocol_file_sha256 = _sha256_bytes(protocol_bytes)
    try:
        frozen_protocol = json.loads(protocol_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen protocol is not valid JSON") from exc
    if not isinstance(frozen_protocol, dict):
        raise ValueError("frozen protocol must be a JSON object")
    stored_content_sha256 = frozen_protocol.get("protocol_content_sha256")
    unhashed_protocol = dict(frozen_protocol)
    unhashed_protocol.pop("protocol_content_sha256", None)
    if (
        not isinstance(stored_content_sha256, str)
        or SHA256_RE.fullmatch(stored_content_sha256) is None
        or canonical_sha256(unhashed_protocol) != stored_content_sha256
    ):
        raise ValueError("frozen protocol content SHA256 mismatch")
    if frozen_protocol.get("protocol") != "sealed_evaluation_protocol":
        raise ValueError("unsupported frozen protocol manifest")
    if frozen_protocol.get("schema_version") != 2:
        raise ValueError("sealed evaluation requires frozen protocol schema v2")

    inputs = frozen_protocol.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("frozen protocol is missing input fingerprints")
    try:
        require_current_allocator_source_fingerprint(
            inputs.get("evaluator_scripts")
        )
    except FrozenProtocolError as exc:
        raise ValueError(
            f"frozen protocol required source fingerprint is invalid: {exc}"
        ) from exc
    verified: Dict[str, Dict[str, Any]] = {}
    for role, value in inputs.items():
        if isinstance(value, Mapping):
            if value.get("type") == "file":
                verified[str(role)] = _verified_frozen_file(
                    value, str(role)
                )
            elif value.get("type") == "directory":
                verified[str(role)] = _verified_frozen_directory(
                    value, str(role)
                )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, Mapping) and item.get("type") == "file":
                    label = f"{role}[{index}]"
                    verified[label] = _verified_frozen_file(item, label)

    checkpoint = frozen_protocol.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("frozen protocol is missing checkpoint identity")
    else:
        selected_root_value = checkpoint.get("selected_root")
        if not isinstance(selected_root_value, str):
            raise ValueError("frozen protocol checkpoint selected_root is missing")
        selected_root = Path(selected_root_value).resolve(strict=True)
        selected_input = verified.get("selected_root")
        if (
            selected_input is None
            or selected_input.get("type") != "directory"
            or Path(str(selected_input.get("path"))) != selected_root
        ):
            raise ValueError(
                "frozen checkpoint selected_root disagrees with frozen input"
            )
        artifact = checkpoint.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError("frozen protocol is missing checkpoint artifact")
        checkpoint_record = _verified_frozen_file(
            artifact, "e3_checkpoint"
        )
        raw_checkpoint = checkpoint_path.expanduser()
        if raw_checkpoint.is_symlink():
            raise ValueError("--checkpoint cannot be a symlink")
        resolved_checkpoint = raw_checkpoint.resolve(strict=True)
        if Path(checkpoint_record["path"]) != resolved_checkpoint:
            raise ValueError(
                "--checkpoint path disagrees with frozen checkpoint artifact"
            )
        if not path_is_within(resolved_checkpoint, selected_root):
            raise ValueError(
                "frozen checkpoint artifact is outside selected_root"
            )
        try:
            checkpoint_state = torch.load(
                io.BytesIO(checkpoint_record["bytes"]),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise ValueError(
                "cannot load frozen checkpoint with weights_only=True"
            ) from exc
        if not isinstance(checkpoint_state, Mapping):
            raise ValueError("frozen checkpoint payload must be a mapping")
        digests = checkpoint_protocol_digests(checkpoint_state)
        for field, observed in digests.items():
            if (
                artifact.get(field) != observed
                or checkpoint.get(field) != observed
            ):
                raise ValueError(
                    f"frozen checkpoint {field} mismatch"
                )
        checkpoint_args = checkpoint.get("args")
        checkpoint_args_sha256 = checkpoint.get("args_sha256")
        if (
            not isinstance(checkpoint_args, Mapping)
            or canonical_sha256(checkpoint_args)
            != checkpoint_args_sha256
            or checkpoint_args != checkpoint_state.get("args")
        ):
            raise ValueError("frozen checkpoint args SHA256 mismatch")
        verified["e3_checkpoint"] = checkpoint_record
        checkpoint_input = inputs.get("checkpoint_args")
        if isinstance(checkpoint_input, Mapping):
            if checkpoint_input.get("type") == "inline-json":
                if checkpoint_input.get("sha256") != checkpoint_args_sha256:
                    raise ValueError(
                        "frozen inline checkpoint args SHA256 mismatch"
                    )
                verified["checkpoint_args"] = {
                    "label": "checkpoint_args",
                    "type": "inline-json",
                    "path": None,
                    "sha256": checkpoint_args_sha256,
                    "size_bytes": None,
                }
            elif checkpoint_input.get("type") == "file":
                input_record = verified.get("checkpoint_args")
                if input_record is None:
                    raise ValueError(
                        "frozen checkpoint args file was not verified"
                    )
                try:
                    source_args = json.loads(
                        input_record["bytes"].decode("utf-8")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "frozen checkpoint args file is invalid"
                    ) from exc
                if source_args != dict(checkpoint_args):
                    raise ValueError(
                        "frozen checkpoint args content mismatch"
                    )

    for role, expected_path in (
        ("image_test", image_manifest_path),
        ("speech_test", speech_manifest_path),
    ):
        record = verified.get(role)
        if record is None:
            raise ValueError(f"frozen protocol is missing {role} fingerprint")
        if Path(record["path"]) != expected_path.expanduser().resolve(strict=True):
            raise ValueError(f"{role} path disagrees with frozen protocol")
    if "sealed_manifest" not in verified:
        raise ValueError(
            "frozen protocol is missing sealed manifest fingerprint"
        )
    resolved_evaluator = evaluator_path.resolve(strict=True)
    evaluator_records = [
        record
        for label, record in verified.items()
        if label.startswith("evaluator_scripts[")
        and Path(record["path"]) == resolved_evaluator
    ]
    if len(evaluator_records) != 1:
        raise ValueError(
            "frozen protocol does not bind this evaluator source exactly once"
        )
    return frozen_protocol, protocol_file_sha256, verified


def verify_file_sha256(path: Path, expected_sha256: str, label: str) -> None:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} changed during evaluation: "
            f"expected={expected_sha256} observed={actual_sha256}"
        )


def validate_complete_evaluation_identity(
    identity: Mapping[str, Any],
) -> None:
    missing = sorted(EVALUATION_IDENTITY_REQUIRED_FIELDS - set(identity))
    if missing:
        raise ValueError(f"evaluation identity is incomplete: missing={missing}")
    source_commit_sha = identity.get("source_commit_sha")
    if (
        not isinstance(source_commit_sha, str)
        or COMMIT_SHA_RE.fullmatch(source_commit_sha) is None
    ):
        raise ValueError("evaluation identity has invalid source_commit_sha")
    for field in (
        "runai_job_name",
        "runai_project",
        "evaluation_scope",
        "eval_split_name",
        "condition",
        "prefix_control",
        "eval_path",
        "negative_mode",
        "protocol_name",
    ):
        value = identity.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"evaluation identity has invalid string field {field}"
            )
    evaluation_scope = identity.get("evaluation_scope")
    if evaluation_scope not in EVALUATION_SCOPES:
        raise ValueError("evaluation identity has invalid evaluation_scope")
    if not isinstance(identity.get("strict_control"), bool):
        raise ValueError("evaluation identity has invalid strict_control")
    if identity.get("strict_control") != (
        evaluation_scope in CLAIM_EVALUATION_SCOPES
    ):
        raise ValueError(
            "evaluation identity strictness disagrees with evaluation_scope"
        )

    e3_checkpoint_sha256 = identity.get("e3_checkpoint_sha256")
    if (
        not isinstance(e3_checkpoint_sha256, str)
        or SHA256_RE.fullmatch(e3_checkpoint_sha256) is None
    ):
        raise ValueError(
            "evaluation identity has invalid e3_checkpoint_sha256"
        )
    for field in (
        "stage_b_checkpoint_sha256",
        "source_run_manifest_sha256",
        "protocol_content_sha256",
    ):
        value = identity.get(field)
        if value is not None and (
            not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
        ):
            raise ValueError(f"evaluation identity has invalid {field}")
    for cache_kind in ("image", "audio"):
        for suffix in ("identity_sha256", "payload_set_sha256"):
            field = f"{cache_kind}_cache_{suffix}"
            cache_sha256 = identity.get(field)
            if (
                not isinstance(cache_sha256, str)
                or SHA256_RE.fullmatch(cache_sha256) is None
            ):
                raise ValueError(
                    f"evaluation identity has invalid {field}"
                )
    for field in (
        "image_produced_features_sha256",
        "audio_produced_features_sha256",
    ):
        value = identity.get(field)
        if (
            not isinstance(value, str)
            or SHA256_RE.fullmatch(value) is None
        ):
            raise ValueError(f"evaluation identity has invalid {field}")
    cache_policy = identity.get("feature_cache_policy")
    if (
        not isinstance(cache_policy, Mapping)
        or cache_policy.get("mode")
        not in {"verified_read_write", "exclusive_write_only_recompute"}
    ):
        raise ValueError("evaluation identity has invalid feature_cache_policy")
    if evaluation_scope == "final" and (
        cache_policy.get("mode") != "exclusive_write_only_recompute"
        or cache_policy.get("preexisting_root_allowed") is not False
        or cache_policy.get("cache_reads_allowed") is not False
        or cache_policy.get("feature_source")
        != "verified_frozen_media_snapshots"
    ):
        raise ValueError(
            "final evaluation identity has invalid feature recompute policy"
        )

    for prefix in ("gamma", "evaluator", "e3_checkpoint"):
        path_value = identity.get(f"{prefix}_path")
        sha_value = identity.get(f"{prefix}_sha256")
        if (
            not isinstance(path_value, str)
            or not Path(path_value).is_absolute()
        ):
            raise ValueError(
                f"evaluation identity has invalid {prefix}_path"
            )
        if (
            not isinstance(sha_value, str)
            or SHA256_RE.fullmatch(sha_value) is None
        ):
            raise ValueError(
                f"evaluation identity has invalid {prefix}_sha256"
            )
    for prefix in (
        "image_manifest",
        "speech_manifest",
        "protocol_manifest",
    ):
        path_value = identity.get(f"{prefix}_path")
        sha_value = identity.get(f"{prefix}_sha256")
        if (path_value is None) != (sha_value is None):
            raise ValueError(
                f"evaluation identity {prefix} path/SHA must be supplied together"
            )
        if path_value is not None:
            if (
                not isinstance(path_value, str)
                or not Path(path_value).is_absolute()
            ):
                raise ValueError(
                    f"evaluation identity has invalid {prefix}_path"
                )
            if (
                not isinstance(sha_value, str)
                or SHA256_RE.fullmatch(sha_value) is None
            ):
                raise ValueError(
                    f"evaluation identity has invalid {prefix}_sha256"
                )

    frozen_input_hashes = identity.get("frozen_input_hashes")
    if not isinstance(frozen_input_hashes, Mapping):
        raise ValueError(
            "evaluation identity has invalid frozen_input_hashes"
        )
    for label, value in frozen_input_hashes.items():
        if (
            not isinstance(label, str)
            or not isinstance(value, str)
            or SHA256_RE.fullmatch(value) is None
        ):
            raise ValueError(
                "evaluation identity has invalid frozen input hash"
            )
    if evaluation_scope == "final" and (
        identity.get("protocol_manifest_path") is None
        or identity.get("protocol_content_sha256") is None
        or not frozen_input_hashes
        or not isinstance(identity.get("frozen_evaluation_run_id"), str)
        or not identity.get("frozen_evaluation_run_id")
        or not isinstance(identity.get("frozen_evaluation_cell_id"), str)
        or not identity.get("frozen_evaluation_cell_id")
        or not isinstance(identity.get("frozen_evaluation_control"), str)
        or not identity.get("frozen_evaluation_control")
    ):
        raise ValueError(
            "final evaluation identity requires verified frozen protocol inputs"
        )
    if evaluation_scope in CLAIM_EVALUATION_SCOPES and (
        identity.get("image_manifest_path") is None
        or identity.get("speech_manifest_path") is None
    ):
        raise ValueError(
            "claim evaluation identity requires explicit image/speech manifests"
        )

    for field in (
        "control_seed",
        "candidate_seed",
        "requested_query_count",
        "image_query_count",
        "speech_query_count",
        "requested_candidate_count",
        "image_candidate_count",
        "speech_candidate_count",
        "conditional_negatives",
        "conditional_candidates",
        "query_offset",
        "candidate_offset",
        "bootstrap_seed",
        "bootstrap_samples",
        "image_eval_samples",
        "speech_eval_samples",
    ):
        value = identity.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"evaluation identity has invalid integer field {field}"
            )
    metric_args = identity.get("metric_affecting_args")
    if (
        not isinstance(metric_args, Mapping)
        or set(metric_args) != set(METRIC_AFFECTING_ARG_FIELDS)
    ):
        raise ValueError(
            "evaluation identity has incomplete metric_affecting_args"
        )
    checkpoint_architecture = identity.get("checkpoint_architecture")
    if (
        not isinstance(checkpoint_architecture, Mapping)
        or not checkpoint_architecture
    ):
        raise ValueError(
            "evaluation identity has invalid checkpoint_architecture"
        )
    tie_epsilon = identity.get("tie_epsilon")
    if (
        not isinstance(tie_epsilon, (int, float))
        or isinstance(tie_epsilon, bool)
        or not math.isfinite(float(tie_epsilon))
        or float(tie_epsilon) < 0.0
    ):
        raise ValueError("evaluation identity has invalid tie_epsilon")

    source_checkpoint_hashes = identity.get("source_checkpoint_hashes")
    if not isinstance(source_checkpoint_hashes, Mapping):
        raise ValueError(
            "evaluation identity has invalid source_checkpoint_hashes"
        )
    for name, value in source_checkpoint_hashes.items():
        if not isinstance(name, str) or (
            value is not None
            and (
                not isinstance(value, str)
                or SHA256_RE.fullmatch(value) is None
            )
        ):
            raise ValueError(
                "evaluation identity has invalid source checkpoint hash entry"
            )
    restoration_order = identity.get("restoration_order")
    if (
        not isinstance(restoration_order, list)
        or not restoration_order
        or any(
            not isinstance(item, str) or not item
            for item in restoration_order
        )
    ):
        raise ValueError(
            "evaluation identity has invalid restoration_order"
        )


def assert_output_paths_available(paths: Sequence[Path]) -> None:
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("metrics and per-query outputs must use distinct paths")
    existing = [str(path) for path in resolved if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to reuse existing evaluation output: {existing}"
        )


def atomic_write_text_exclusive(path: Path, content: str) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"refusing to reuse existing evaluation output: {destination}"
        )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to reuse existing evaluation output: {destination}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def bind_per_query_evaluation_provenance(
    rows: Sequence[Dict[str, Any]],
    evaluation_provenance: Mapping[str, Any],
) -> None:
    """Reject copied rows from another split/control before binding this evaluation."""
    validate_complete_evaluation_identity(evaluation_provenance)
    identity_sha256 = evaluation_provenance.get("evaluation_identity_sha256")
    if not isinstance(identity_sha256, str) or SHA256_RE.fullmatch(identity_sha256) is None:
        raise ValueError("evaluation provenance is missing an exact identity SHA256")
    expected_condition = evaluation_provenance.get("condition")
    expected_split = evaluation_provenance.get("eval_split_name")
    for index, row in enumerate(rows):
        if row.get("condition") != expected_condition:
            raise ValueError(
                f"per-query row {index} condition does not match evaluation identity"
            )
        if row.get("eval_split_name") != expected_split:
            raise ValueError(
                f"per-query row {index} split does not match evaluation identity"
            )
        existing = row.get("evaluation_provenance")
        if existing is not None and existing != evaluation_provenance:
            raise ValueError(
                f"per-query row {index} carries copied evaluation provenance"
            )
        row["evaluation_provenance"] = dict(evaluation_provenance)


COMMIT_SHA_RE = re.compile(r"[0-9a-fA-F]{40}")


def resolve_git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot resolve source commit: {completed.stderr.strip()}")
    head = completed.stdout.strip().lower()
    if COMMIT_SHA_RE.fullmatch(head) is None:
        raise RuntimeError(f"git HEAD is not an exact 40-hex commit SHA: {head!r}")
    return head


def load_run_environment_provenance(repo_root: Path | None = None) -> Dict[str, str | None]:
    root = repo_root or Path(__file__).resolve().parents[1]
    actual_head = resolve_git_head(root)
    source_commit_sha = os.environ.get("SOURCE_COMMIT_SHA")
    if source_commit_sha:
        if COMMIT_SHA_RE.fullmatch(source_commit_sha) is None:
            raise ValueError("SOURCE_COMMIT_SHA must be an exact 40-hex commit SHA")
        source_commit_sha = source_commit_sha.lower()
        if source_commit_sha != actual_head:
            raise ValueError(
                f"SOURCE_COMMIT_SHA does not match git HEAD: "
                f"expected {source_commit_sha}, observed {actual_head}"
            )
    else:
        source_commit_sha = actual_head
    return {
        "source_commit_sha": source_commit_sha,
        "runai_job_name": os.environ.get("RUNAI_JOB_NAME"),
        "runai_project": os.environ.get("RUNAI_PROJECT"),
    }


def load_explicit_eval_rows(
    manifest_path: Path,
    modality: str,
    data_dir: Path,
    limit: int,
    manifest_bytes: bytes | None = None,
    expected_media: Mapping[str, Mapping[str, Any]] | None = None,
    snapshot_root: Path | None = None,
) -> List[Dict[str, Any]]:
    payload = manifest_path.read_bytes() if manifest_bytes is None else manifest_bytes
    try:
        rows = [
            json.loads(line)
            for line in payload.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSONL manifest: {manifest_path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"manifest rows must be JSON objects: {manifest_path}")
    if limit > 0:
        rows = rows[: int(limit)]
    media_key = "image_path" if modality == "image" else "audio_path"
    observed_ids: set[str] = set()
    for row in rows:
        value = row.get(media_key)
        if not value:
            raise ValueError(f"Missing {media_key} in {manifest_path}")
        raw_path = Path(str(value)).expanduser()
        if raw_path.is_absolute():
            candidates = [
                (raw_path, manifest_path.parent),
                (raw_path, data_dir),
            ]
        else:
            parts = raw_path.parts
            if parts and parts[0] == "data":
                parts = parts[1:]
            if parts and parts[0] == data_dir.name:
                parts = parts[1:]
            candidates = [
                (manifest_path.parent / raw_path, manifest_path.parent),
                (data_dir.joinpath(*parts), data_dir),
            ]
        resolved = None
        for candidate, root in candidates:
            lexical = Path(os.path.abspath(candidate))
            try:
                relative = lexical.relative_to(root.resolve(strict=True))
            except (OSError, ValueError):
                continue
            current = root.resolve(strict=True)
            contains_symlink = False
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    contains_symlink = True
                    break
            if contains_symlink or not lexical.is_file():
                continue
            resolved = lexical.resolve(strict=True)
            if path_is_within(resolved, root):
                break
            resolved = None
        if resolved is None:
            raise FileNotFoundError(
                f"Cannot resolve {media_key}={value!r} from manifest/data roots "
                f"{manifest_path.parent} and {data_dir}"
            )
        payload = resolved.read_bytes()
        actual_sha256 = _sha256_bytes(payload)
        row_sha256 = row.get("media_sha256")
        if row_sha256 is not None and (
            not isinstance(row_sha256, str)
            or SHA256_RE.fullmatch(row_sha256) is None
            or row_sha256.lower() != actual_sha256
        ):
            raise ValueError(
                f"{modality} row media_sha256 does not match actual bytes: {resolved}"
            )
        if expected_media is not None and row_sha256 is None:
            raise ValueError(
                f"frozen {modality} row is missing media_sha256"
            )
        row_id = str(row.get("id", "")).strip()
        if expected_media is not None:
            expected = expected_media.get(row_id)
            if expected is None:
                raise ValueError(f"{modality} row is not frozen: {row_id!r}")
            if (
                expected.get("media_sha256") != actual_sha256
                or expected.get("media_size_bytes") != len(payload)
            ):
                raise ValueError(
                    f"{modality} media bytes disagree with frozen commitment: {row_id!r}"
                )
            observed_ids.add(row_id)
        row["_source_media_path"] = str(resolved)
        row["_verified_media_sha256"] = actual_sha256
        row["_verified_media_size_bytes"] = len(payload)
        if snapshot_root is not None:
            suffix = resolved.suffix.lower()
            snapshot = (
                snapshot_root
                / modality
                / actual_sha256[:2]
                / f"{actual_sha256}{suffix}"
            )
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if snapshot.is_symlink():
                raise ValueError(f"{modality} media snapshot cannot be a symlink")
            try:
                with snapshot.open("xb") as handle:
                    handle.write(payload)
            except FileExistsError:
                if snapshot.read_bytes() != payload:
                    raise ValueError(
                        f"{modality} media snapshot bytes mismatch: {snapshot}"
                    )
            row[media_key] = str(snapshot.resolve(strict=True))
        else:
            row[media_key] = str(resolved)
    if expected_media is not None and observed_ids != set(expected_media):
        missing = sorted(set(expected_media) - observed_ids)
        raise ValueError(f"{modality} frozen media rows were not loaded: {missing}")
    return rows


def frozen_media_commitments(
    protocol: Mapping[str, Any],
    modality: str,
) -> Dict[str, Mapping[str, Any]]:
    rows = protocol.get("sealed_rows")
    if not isinstance(rows, list):
        raise ValueError("frozen protocol is missing sealed media commitments")
    commitments: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("modality") != modality:
            continue
        row_id = str(row.get("row_id", "")).strip()
        if (
            not row_id
            or row_id in commitments
            or not isinstance(row.get("media_sha256"), str)
            or SHA256_RE.fullmatch(str(row["media_sha256"])) is None
            or not isinstance(row.get("media_size_bytes"), int)
        ):
            raise ValueError("frozen protocol has invalid sealed media commitment")
        commitments[row_id] = row
    if not commitments:
        raise ValueError(f"frozen protocol has no {modality} media commitments")
    return commitments


def evaluation_run_contract(args, requested_candidate_count: int) -> Dict[str, Any]:
    control = (
        "no-prefix"
        if args.eval_path == "no_prefix_lm"
        else "norm-matched-random"
        if args.prefix_control == "random"
        else str(args.prefix_control)
    )
    return {
        "negative_mode": str(args.negative_mode),
        "requested_candidate_count": int(requested_candidate_count),
        "conditional_negatives": int(args.conditional_negatives),
        "conditional_candidates": int(args.conditional_candidates),
        "conditional_queries": int(args.conditional_queries),
        "image_query_count": int(args.conditional_queries),
        "speech_query_count": int(args.conditional_queries),
        "image_eval_samples": int(args.image_eval_samples),
        "speech_eval_samples": int(args.speech_eval_samples),
        "max_length": int(args.max_length),
        "conditional_batch_size": int(args.conditional_batch_size),
        "query_offset": int(args.query_offset),
        "candidate_offset": int(args.candidate_offset),
        "tie_epsilon": float(args.tie_epsilon),
        "candidate_permutation": str(args.candidate_permutation),
        "randomize_positive_position": bool(args.randomize_positive_position),
        "control": control,
        "prefix_control": str(args.prefix_control),
        "eval_path": str(args.eval_path),
        "candidate_seed": int(args.candidate_seed),
        "control_seed": int(args.control_seed),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "protocol_name": str(args.protocol_name),
        "eval_split_name": str(args.eval_split_name),
    }


def select_frozen_evaluation_run(
    protocol: Mapping[str, Any],
    args,
    requested_candidate_count: int,
) -> Dict[str, Any]:
    runs = protocol.get("evaluation_runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("frozen protocol is missing exact evaluation run contracts")
    actual = evaluation_run_contract(args, requested_candidate_count)
    matches = [
        dict(run)
        for run in runs
        if isinstance(run, Mapping)
        and all(run.get(key) == value for key, value in actual.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            "evaluation args do not match exactly one frozen matrix cell: "
            f"matches={len(matches)} contract={actual}"
        )
    return matches[0]


def row_uid(row: Dict[str, Any], modality: str, index: int) -> str:
    for key in ("uid", "source_uid", "image_uid", "utterance_id", "source_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{modality}:{value}"
    return f"{modality}:{row.get('source', 'unknown')}:{row.get('id', index)}"


IMAGE_GROUP_PROTOCOL_SEMANTICS = {
    "identity": "deterministic_source_image_fields_else_resolved_image_path",
    "local_positive_semantics": "exactly_one_caption_row_from_query_image_group",
    "local_negative_semantics": "caption_rows_from_non_query_image_groups_only",
    "full_matrix_positive_semantics": "all_caption_rows_from_query_image_group",
    "full_matrix_negative_semantics": "caption_rows_from_non_query_image_groups_only",
    "strict_rank_semantics": (
        "count_non_positive_caption_rows_with_nll_at_most_best_positive_nll_plus_tie_epsilon"
    ),
    "margin_semantics": "best_negative_nll_minus_best_positive_nll",
    "group_aware_chance_semantics": "uniform_random_candidate_image_group",
    "caption_row_chance_semantics": "uniform_random_candidate_caption_row",
    "legacy_image_chance_semantics": "uniform_random_explicit_gold_caption_position",
}


def candidate_set_hash(candidate_ids: Sequence[str]) -> str:
    payload = json.dumps(list(candidate_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def path_is_within(path: Path, root: Path) -> bool:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def deterministic_permutation(size: int, seed: int) -> List[int]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return [int(value) for value in torch.randperm(int(size), generator=generator).tolist()]


def balanced_positive_position(query_index: int, candidate_count: int, seed: int) -> int:
    """Assign seeded positive positions with counts differing by at most one."""
    if int(candidate_count) <= 0:
        raise ValueError("candidate_count must be positive")
    order = deterministic_permutation(int(candidate_count), int(seed))
    return int(order[int(query_index) % int(candidate_count)])


def place_gold_at_position(
    candidates: Sequence[int], gold_index: int, position: int, seed: int
) -> Tuple[List[int], int]:
    """Randomize candidate order, then place gold at the preassigned position."""
    if not candidates or int(gold_index) not in candidates:
        raise ValueError("candidate set must contain the gold index")
    if int(position) < 0 or int(position) >= len(candidates):
        raise ValueError("gold position is outside the candidate set")
    permutation = deterministic_permutation(len(candidates), int(seed))
    ordered = [int(candidates[index]) for index in permutation]
    current = ordered.index(int(gold_index))
    ordered[current], ordered[int(position)] = ordered[int(position)], ordered[current]
    return ordered, int(position)


def gold_position_assignment_provenance(
    plan: Mapping[str, Any], allocator_manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "allocator_name": ALLOCATOR_NAME,
        "allocator_version": ALLOCATOR_VERSION,
        "candidate_seed": int(plan["candidate_seed"]),
        "seed_derivation": str(plan["seed_derivation"]),
        "seed_context_sha256": str(plan["seed_context_sha256"]),
        "assignment_id": str(plan["assignment_id"]),
        "positions_sha256": str(plan["positions_sha256"]),
        "plans_sha256": str(allocator_manifest["plans_sha256"]),
    }


def local_candidate_indices(
    records: Sequence[Dict[str, Any]],
    index: int,
    negatives: int,
    mode: str = "stride",
    hard_indices: Sequence[int] | None = None,
    *,
    candidate_seed: int = 0,
    randomize_positive_position: bool = False,
    positive_position: int | None = None,
    group_ids: Sequence[str] | None = None,
) -> Tuple[List[int], int]:
    total = len(records)
    if group_ids is not None and len(group_ids) != total:
        raise ValueError("records/group_ids length mismatch")
    query_group_id = str(group_ids[int(index)]) if group_ids is not None else None

    def eligible(candidate_index: int) -> bool:
        if int(candidate_index) == int(index):
            return False
        return group_ids is None or str(group_ids[int(candidate_index)]) != query_group_id

    eligible_negative_count = sum(
        eligible(candidate_index) for candidate_index in range(total)
    )
    if eligible_negative_count < int(negatives):
        raise RuntimeError(
            "Not enough cross-group candidates for requested local-negative evaluation: "
            f"requested={int(negatives)} available={eligible_negative_count}"
        )
    candidates = [int(index)]
    used = {int(index)}
    if mode == "hard_text" and hard_indices is not None:
        for neg in hard_indices:
            neg_i = int(neg)
            if neg_i in used or not eligible(neg_i):
                continue
            used.add(neg_i)
            candidates.append(neg_i)
            if len(candidates) >= int(negatives) + 1:
                break
    if mode == "random" and len(candidates) < int(negatives) + 1:
        generator = torch.Generator()
        generator.manual_seed(int(candidate_seed))
        for neg in torch.randperm(max(1, total), generator=generator).tolist():
            neg_i = int(neg)
            if neg_i in used or not eligible(neg_i):
                continue
            used.add(neg_i)
            candidates.append(neg_i)
            if len(candidates) >= int(negatives) + 1:
                break
    stride = 37
    offset = 17
    cursor = 0
    while len(candidates) < int(negatives) + 1:
        neg = (int(index) + offset + stride * cursor) % max(1, total)
        cursor += 1
        scanned = 0
        while (neg in used or not eligible(neg)) and scanned < total:
            neg = (neg + 1) % total
            scanned += 1
        if neg in used or not eligible(neg):
            raise RuntimeError(
                "Not enough cross-group candidates for requested local-negative evaluation"
            )
        used.add(neg)
        candidates.append(int(neg))
    if randomize_positive_position:
        if positive_position is None:
            positive_position = balanced_positive_position(index, len(candidates), candidate_seed)
        candidates, gold_position = place_gold_at_position(
            candidates, int(index), int(positive_position), int(candidate_seed)
        )
        return candidates, gold_position
    gold_position = candidates.index(int(index))
    return candidates, int(gold_position)


def local_candidate_texts(
    records: Sequence[Dict[str, Any]],
    key: str,
    index: int,
    negatives: int,
    mode: str = "stride",
    hard_indices: Sequence[int] | None = None,
    group_ids: Sequence[str] | None = None,
) -> List[str]:
    indices, _ = local_candidate_indices(
        records,
        index,
        negatives,
        mode,
        hard_indices,
        group_ids=group_ids,
    )
    return [str(records[candidate][key]) for candidate in indices]


def full_candidate_indices(
    total: int,
    gold_index: int,
    *,
    candidate_seed: int,
    randomize_positive_position: bool,
    positive_position: int | None = None,
) -> Tuple[List[int], int]:
    indices = list(range(int(total)))
    if randomize_positive_position:
        if positive_position is None:
            positive_position = balanced_positive_position(gold_index, len(indices), candidate_seed)
        indices, gold_position = place_gold_at_position(
            indices, int(gold_index), int(positive_position), int(candidate_seed)
        )
        return indices, gold_position
    return indices, int(indices.index(int(gold_index)))


def deterministic_derangement_index(index: int, size: int, seed: int) -> int:
    if size <= 1:
        return int(index)
    offset = 1 + int(seed) % (int(size) - 1)
    return int((int(index) + offset) % int(size))


def multi_positive_ranks(
    similarity: torch.Tensor,
    positive_indices: Sequence[Sequence[int]],
) -> List[int]:
    """Return best-positive zero-based rank for duplicate/multi-caption retrieval."""
    if similarity.ndim != 2 or similarity.shape[0] != len(positive_indices):
        raise ValueError("similarity/positive_indices shape mismatch")
    scores = torch.as_tensor(similarity).float().cpu()
    if not torch.isfinite(scores).all():
        raise ValueError("similarity contains non-finite values")
    ranks: List[int] = []
    for row_idx, positives in enumerate(positive_indices):
        positive_set = sorted({int(value) for value in positives})
        if (
            not positive_set
            or positive_set[0] < 0
            or positive_set[-1] >= scores.shape[1]
        ):
            raise ValueError("each query needs valid positive candidate indices")
        best_positive = scores[row_idx, positive_set].max()
        positive_lookup = set(positive_set)
        strict_rank = sum(
            1
            for index, score in enumerate(scores[row_idx].tolist())
            if index not in positive_lookup and float(score) >= float(best_positive)
        )
        ranks.append(int(strict_rank))
    return ranks


def multi_positive_nll_evidence(
    nll_scores: Sequence[float],
    positive_indices: Sequence[int],
    tie_epsilon: float,
) -> Dict[str, Any]:
    """Return strict best-positive evidence while excluding valid positives from negatives."""
    if not nll_scores:
        raise ValueError("nll_scores must not be empty")
    if float(tie_epsilon) < 0.0:
        raise ValueError("tie_epsilon must be non-negative")
    scores = [float(value) for value in nll_scores]
    positives = sorted({int(value) for value in positive_indices})
    if not positives or positives[0] < 0 or positives[-1] >= len(scores):
        raise ValueError("positive_indices must contain valid candidate positions")
    positive_set = set(positives)
    best_positive_nll = min(scores[index] for index in positives)
    best_positive_indices = [
        index
        for index in positives
        if scores[index] <= best_positive_nll + float(tie_epsilon)
    ]
    negative_scores = [
        value for index, value in enumerate(scores) if index not in positive_set
    ]
    tie_count = sum(
        1
        for value in negative_scores
        if abs(value - best_positive_nll) <= float(tie_epsilon)
    )
    strict_rank = sum(
        1
        for value in negative_scores
        if value <= best_positive_nll + float(tie_epsilon)
    )
    best_nll = min(scores)
    best_indices = [
        index
        for index, value in enumerate(scores)
        if value <= best_nll + float(tie_epsilon)
    ]
    margin = min(negative_scores) - best_positive_nll if negative_scores else 0.0
    return {
        "strict_rank": int(strict_rank),
        "strict_r_at_1": float(strict_rank == 0),
        "reciprocal_rank": float(1.0 / (strict_rank + 1)),
        "gold_nll": float(best_positive_nll),
        "best_positive_nll": float(best_positive_nll),
        "gold_nll_margin": float(margin),
        "best_positive_nll_margin": float(margin),
        "gold_tie_count": int(tie_count),
        "has_gold_tie": bool(tie_count > 0),
        "best_candidate_indices": best_indices,
        "best_tie_count": int(max(0, len(best_indices) - 1)),
        "positive_indices": positives,
        "best_positive_indices": best_positive_indices,
        "positive_count": len(positives),
    }


def apply_prefix_control(feat: torch.Tensor, control: str, seed: int) -> torch.Tensor:
    if control in {"real", "shuffled"}:
        return feat
    if control == "zero":
        return torch.zeros_like(feat)
    if control == "random":
        generator = torch.Generator(device=feat.device)
        generator.manual_seed(int(seed))
        random = torch.randn(feat.shape, device=feat.device, dtype=torch.float32, generator=generator)
        target_norm = feat.detach().float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        random = F.normalize(random, dim=-1) * target_norm
        return random.to(dtype=feat.dtype)
    raise ValueError(f"Unsupported prefix control: {control}")


def score_image_query(wrapper, tokenizer, image_processor, vision_model, cache: FeatureCache, query: Dict[str, Any], candidates: Sequence[str], device, args, feature_override: torch.Tensor | None = None) -> List[float]:
    scores: List[float] = []
    feat = feature_override if feature_override is not None else cache.image_batch(image_processor, vision_model, [query], device, args.encoder_feature_tokens)
    for start in range(0, len(candidates), args.conditional_batch_size):
        texts = list(candidates[start:start + args.conditional_batch_size])
        batch = tokenize_prompt_targets(tokenizer, ["Caption:"] * len(texts), texts, device, args.max_length)
        feats = feat.expand(len(texts), -1, -1).contiguous()
        with torch.no_grad():
            outputs = wrapper(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], image_features=feats)
            nll = per_example_nll(outputs.logits, batch["labels"], args.image_prefix_tokens)
        scores.extend((-nll).detach().float().cpu().tolist())
    return scores


def score_speech_query(wrapper, tokenizer, speech_processor, speech_model, cache: FeatureCache, query: Dict[str, Any], candidates: Sequence[str], device, args, feature_override: torch.Tensor | None = None) -> List[float]:
    scores: List[float] = []
    feat = feature_override if feature_override is not None else cache.audio_batch(
        speech_processor,
        speech_model,
        [query],
        device,
        args.sample_rate,
        args.encoder_feature_tokens,
        args.audio_max_seconds,
    )
    for start in range(0, len(candidates), args.conditional_batch_size):
        texts = list(candidates[start:start + args.conditional_batch_size])
        batch = tokenize_prompt_targets(tokenizer, ["Transcript:"] * len(texts), texts, device, args.max_length)
        feats = feat.expand(len(texts), -1, -1).contiguous()
        with torch.no_grad():
            outputs = wrapper(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], audio_features=feats)
            nll = per_example_nll(outputs.logits, batch["labels"], args.audio_prefix_tokens)
        scores.extend((-nll).detach().float().cpu().tolist())
    return scores


def score_no_prefix_query(wrapper, tokenizer, prompt: str, candidates: Sequence[str], device, args) -> List[float]:
    scores: List[float] = []
    for start in range(0, len(candidates), args.conditional_batch_size):
        texts = list(candidates[start:start + args.conditional_batch_size])
        batch = tokenize_prompt_targets(tokenizer, [prompt] * len(texts), texts, device, args.max_length)
        with torch.no_grad():
            outputs = wrapper.lm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], return_dict=True, output_router_logits=False)
            nll = per_example_nll(outputs.logits, batch["labels"], 0)
        scores.extend((-nll).detach().float().cpu().tolist())
    return scores


def rank_gold(nll_scores: Sequence[float], gold_idx: int, tie_epsilon: float = 0.0) -> int:
    return int(tie_aware_nll_evidence(nll_scores, gold_idx, tie_epsilon)["strict_rank"])


def hard_negative_indices(embeddings: torch.Tensor, negatives: int) -> List[List[int]]:
    if negatives <= 0 or embeddings.shape[0] <= 1:
        return [[] for _ in range(int(embeddings.shape[0]))]
    sim = embeddings.float() @ embeddings.float().T
    sim.fill_diagonal_(-float("inf"))
    topk = min(int(negatives), max(0, embeddings.shape[0] - 1))
    return torch.topk(sim, k=topk, dim=1).indices.cpu().tolist()


def lexical_hard_negative_indices(
    texts: Sequence[str],
    negatives: int,
    group_ids: Sequence[str] | None = None,
) -> List[List[int]]:
    """Select checkpoint-independent hard negatives by token-set Jaccard similarity."""
    if group_ids is not None and len(group_ids) != len(texts):
        raise ValueError("texts/group_ids length mismatch")
    token_sets = [set(re.findall(r"[a-z0-9]+", text.casefold())) for text in texts]
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
            jaccard = overlap / max(1, union)
            ranked.append((-jaccard, -overlap, abs(len(target) - len(candidate)), candidate_index))
        ranked.sort()
        if group_ids is not None and len(ranked) < max(0, int(negatives)):
            raise RuntimeError(
                "Not enough cross-group candidates for requested hard-text evaluation: "
                f"query={index} requested={int(negatives)} available={len(ranked)}"
            )
        result.append([item[-1] for item in ranked[: max(0, int(negatives))]])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/real_subset_final")
    parser.add_argument("--image-manifest", default="", help="Explicit immutable image-eval JSONL. Bypasses tail splitting.")
    parser.add_argument("--speech-manifest", default="", help="Explicit immutable speech-eval JSONL. Bypasses tail splitting.")
    parser.add_argument("--run-output-dir", required=True)
    parser.add_argument("--feature-cache-dir", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--stage-b-checkpoint",
        default="",
        help="Exact Stage-B student checkpoint required by Stage-B-initialized E3 checkpoints.",
    )
    parser.add_argument(
        "--stage-b-checkpoint-sha256",
        default="",
        help="Required full SHA256 for --stage-b-checkpoint.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--evaluation-scope",
        choices=EVALUATION_SCOPES,
        required=True,
        help="Explicit diagnostic, development-claim, or final-claim scope.",
    )
    parser.add_argument("--base-model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--vision-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--speech-model", default="openai/whisper-base.en")
    parser.add_argument("--speech-target-space", choices=["olmoe_text_hidden", "whisper_decoder_text"], default="olmoe_text_hidden")
    parser.add_argument("--alignment-prefix-residual", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--capacity-factor", type=float, default=4.0)
    parser.add_argument("--aux-coef", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, choices=[2, 4, 8], default=2)
    parser.add_argument("--image-prefix-tokens", type=int, default=50)
    parser.add_argument("--audio-prefix-tokens", type=int, default=50)
    parser.add_argument("--encoder-feature-tokens", type=int, default=50)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--audio-max-seconds", type=float, default=0.0)
    parser.add_argument("--image-eval-samples", type=int, default=250)
    parser.add_argument("--speech-eval-samples", type=int, default=250)
    parser.add_argument("--conditional-queries", type=int, default=250)
    parser.add_argument("--conditional-candidates", type=int, default=250)
    parser.add_argument("--conditional-negatives", type=int, default=4)
    parser.add_argument("--conditional-batch-size", type=int, default=8)
    parser.add_argument("--negative-mode", choices=["stride", "random", "hard_text"], default="stride")
    parser.add_argument("--eval-path", choices=["shared_prefix", "no_prefix_lm"], default="shared_prefix")
    parser.add_argument("--prefix-control", choices=["real", "zero", "random", "shuffled"], default="real")
    parser.add_argument("--control-seed", type=int, default=42)
    parser.add_argument("--candidate-seed", type=int, default=314159)
    parser.add_argument("--candidate-permutation", choices=["query_identity_seeded"], default="query_identity_seeded")
    parser.add_argument("--tie-epsilon", type=float, default=1e-8)
    parser.add_argument("--randomize-positive-position", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--protocol-name", default="conditional_matching_v2")
    parser.add_argument("--protocol-manifest", default="", help="Frozen protocol JSON required for sealed evaluation.")
    parser.add_argument("--eval-split-name", default="eval_tail", help="Human-readable split role, e.g. development or sealed_test.")
    parser.add_argument("--query-offset", type=int, default=0, help="Offset into the eval-tail pool for held-out query slicing.")
    parser.add_argument("--candidate-offset", type=int, default=-1, help="Full-matrix candidate window offset. Defaults to query_offset; set 0 for 250-way candidates with held-out queries.")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--per-query-output", default="", help="Optional JSONL path for per-query ranks and controls.")
    parser.add_argument(
        "--result-manifest-output",
        default="",
        help="Canonical Run:AI result manifest published after metrics and per-query outputs.",
    )
    parser.add_argument(
        "--result-role",
        default="",
        help="Final ledger role bound by --result-manifest-output.",
    )
    args = parser.parse_args()
    evaluator_path = Path(__file__).resolve()
    evaluator_sha256 = sha256_file(evaluator_path)

    output_path = Path(args.output).expanduser()
    per_query_path = (
        Path(args.per_query_output).expanduser()
        if args.per_query_output
        else None
    )
    manifest_path = (
        Path(args.result_manifest_output).expanduser()
        if args.result_manifest_output
        else None
    )
    if bool(manifest_path) != bool(args.result_role):
        raise ValueError(
            "--result-manifest-output and --result-role must be supplied together"
        )
    if manifest_path is not None and per_query_path is None:
        raise ValueError("result manifest publication requires --per-query-output")
    output_paths = [output_path]
    if per_query_path is not None:
        output_paths.append(per_query_path)
    if manifest_path is not None:
        output_paths.append(manifest_path)
    assert_output_paths_available(output_paths)
    strict_control = validate_evaluation_scope(
        str(args.evaluation_scope),
        str(args.image_manifest),
        str(args.speech_manifest),
        str(args.protocol_manifest),
        str(args.per_query_output),
        str(args.feature_cache_dir),
    )
    sealed_protocol = str(args.evaluation_scope) == "final"
    validate_eval_top_k(args.top_k, sealed_protocol)

    data_dir = Path(args.data_dir).expanduser().resolve(strict=True)
    def resolve_input_file(value: str, label: str) -> Path | None:
        if not value:
            return None
        raw = Path(value).expanduser()
        if raw.is_symlink():
            raise ValueError(f"{label} cannot be a symlink")
        resolved = raw.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"{label} must be a regular file")
        return resolved

    image_manifest_path = resolve_input_file(
        args.image_manifest, "image manifest"
    )
    speech_manifest_path = resolve_input_file(
        args.speech_manifest, "speech manifest"
    )
    protocol_manifest_path = resolve_input_file(
        args.protocol_manifest, "protocol manifest"
    )
    frozen_protocol: Dict[str, Any] | None = None
    protocol_manifest_sha256: str | None = None
    protocol_content_sha256: str | None = None
    frozen_inputs: Dict[str, Dict[str, Any]] = {}
    if protocol_manifest_path is not None:
        if image_manifest_path is None or speech_manifest_path is None:
            raise ValueError(
                "frozen protocol requires explicit image and speech manifests"
            )
        (
            frozen_protocol,
            protocol_manifest_sha256,
            frozen_inputs,
        ) = load_verified_frozen_protocol(
            protocol_manifest_path,
            image_manifest_path,
            speech_manifest_path,
            evaluator_path,
            Path(args.checkpoint),
        )
        protocol_content_sha256 = str(
            frozen_protocol["protocol_content_sha256"]
        )
        seeds = frozen_protocol.get("seeds", {})
        if int(seeds.get("candidate_seed", -1)) != int(args.candidate_seed):
            raise ValueError("candidate seed disagrees with frozen protocol")
        if int(seeds.get("control_seed", -1)) != int(args.control_seed):
            raise ValueError("control seed disagrees with frozen protocol")
        selected_root = frozen_protocol.get("checkpoint", {}).get(
            "selected_root"
        )
        if (
            not selected_root
            or Path(str(selected_root)).resolve()
            != Path(args.run_output_dir).resolve()
        ):
            raise ValueError("run output root disagrees with frozen protocol")
        if path_is_within(
            Path(args.feature_cache_dir), Path(str(selected_root))
        ):
            raise ValueError(
                "final feature cache must be outside the frozen selected root"
            )

    run_environment_provenance = load_run_environment_provenance()
    cache_root_value = (
        Path(args.feature_cache_dir)
        if args.feature_cache_dir
        else Path(args.run_output_dir) / "feature_cache"
    )
    cache_root, feature_cache_policy = prepare_feature_cache_root(
        cache_root_value,
        str(args.evaluation_scope),
        output_paths,
    )
    media_snapshot_root = (
        cache_root / "sealed_media" if frozen_protocol is not None else None
    )
    image_media_commitments = (
        frozen_media_commitments(frozen_protocol, "image")
        if frozen_protocol is not None
        else None
    )
    speech_media_commitments = (
        frozen_media_commitments(frozen_protocol, "speech")
        if frozen_protocol is not None
        else None
    )

    image_manifest_sha256: str | None = None
    speech_manifest_sha256: str | None = None
    if image_manifest_path is not None:
        image_manifest_bytes = (
            frozen_inputs["image_test"]["bytes"]
            if "image_test" in frozen_inputs
            else image_manifest_path.read_bytes()
        )
        image_manifest_sha256 = _sha256_bytes(image_manifest_bytes)
        image_eval_tail = load_explicit_eval_rows(
            image_manifest_path,
            "image",
            data_dir,
            int(args.image_eval_samples),
            manifest_bytes=image_manifest_bytes,
            expected_media=image_media_commitments,
            snapshot_root=media_snapshot_root,
        )
        image_split_source = "explicit_manifest"
    else:
        image_rows = read_jsonl(data_dir / "image_captions.jsonl")
        absolutize_media_paths(image_rows, data_dir)
        _, image_eval_tail = split_tail(
            image_rows, args.image_eval_samples
        )
        image_eval_tail = list(image_eval_tail)
        image_split_source = "legacy_development_tail"
    if speech_manifest_path is not None:
        speech_manifest_bytes = (
            frozen_inputs["speech_test"]["bytes"]
            if "speech_test" in frozen_inputs
            else speech_manifest_path.read_bytes()
        )
        speech_manifest_sha256 = _sha256_bytes(speech_manifest_bytes)
        audio_eval_tail = load_explicit_eval_rows(
            speech_manifest_path,
            "speech",
            data_dir,
            int(args.speech_eval_samples),
            manifest_bytes=speech_manifest_bytes,
            expected_media=speech_media_commitments,
            snapshot_root=media_snapshot_root,
        )
        speech_split_source = "explicit_manifest"
    else:
        audio_rows = read_jsonl(data_dir / "speech_transcripts.jsonl")
        absolutize_media_paths(audio_rows, data_dir)
        _, audio_eval_tail = split_tail(
            audio_rows, args.speech_eval_samples
        )
        audio_eval_tail = list(audio_eval_tail)
        speech_split_source = "legacy_development_tail"
    local_mode = int(args.conditional_negatives) >= 0
    requested_candidate_count = (
        int(args.conditional_negatives) + 1
        if local_mode
        else int(args.conditional_candidates)
    )
    frozen_evaluation_run: Dict[str, Any] | None = None
    if frozen_protocol is not None:
        frozen_evaluation_run = select_frozen_evaluation_run(
            frozen_protocol,
            args,
            requested_candidate_count,
        )
        frozen_sizes = {
            int(value)
            for value in frozen_protocol.get("candidate_sets", {}).get("sizes", [])
        }
        if requested_candidate_count not in frozen_sizes:
            raise ValueError(
                f"candidate count {requested_candidate_count} is not frozen: {sorted(frozen_sizes)}"
            )
        frozen_queries = frozen_protocol.get("query_counts", {})
        if int(frozen_queries.get("image", -1)) != int(args.conditional_queries):
            raise ValueError("image query count disagrees with frozen protocol")
        if int(frozen_queries.get("speech", -1)) != int(args.conditional_queries):
            raise ValueError("speech query count disagrees with frozen protocol")
        expected_paths = {
            "image_test": image_manifest_path,
            "speech_test": speech_manifest_path,
        }
        inputs = frozen_protocol.get("inputs", {})
        for role, actual_path in expected_paths.items():
            stored_path = inputs.get(role, {}).get("path")
            if actual_path is None or not stored_path or Path(str(stored_path)).resolve() != actual_path.resolve():
                raise ValueError(f"{role} path disagrees with frozen protocol")
        protocol_control = (
            "no-prefix"
            if args.eval_path == "no_prefix_lm"
            else "norm-matched-random"
            if args.prefix_control == "random"
            else str(args.prefix_control)
        )
        if protocol_control not in frozen_protocol.get("controls", []):
            raise ValueError(f"control {protocol_control!r} is not frozen")

    # Local-negative eval draws negatives from the whole eval tail. Full-matrix eval
    # defaults the candidate window to query_offset so held-out slices keep gold
    # items inside the candidate set; candidate_offset can force a wider bank, e.g.
    # 250-way candidates from 0..249 with queries starting at QUERY_OFFSET=125.
    if local_mode:
        image_candidate_start = 0
        audio_candidate_start = 0
        image_eval = image_eval_tail
        audio_eval = audio_eval_tail
        image_offset = max(0, min(int(args.query_offset), len(image_eval)))
        audio_offset = max(0, min(int(args.query_offset), len(audio_eval)))
        image_query_indices = list(range(image_offset, min(len(image_eval), image_offset + int(args.conditional_queries))))
        audio_query_indices = list(range(audio_offset, min(len(audio_eval), audio_offset + int(args.conditional_queries))))
    else:
        candidate_offset = int(args.query_offset) if int(args.candidate_offset) < 0 else int(args.candidate_offset)
        image_candidate_start = max(0, min(candidate_offset, len(image_eval_tail)))
        audio_candidate_start = max(0, min(candidate_offset, len(audio_eval_tail)))
        image_candidate_end = min(len(image_eval_tail), image_candidate_start + max(1, int(args.conditional_candidates)))
        audio_candidate_end = min(len(audio_eval_tail), audio_candidate_start + max(1, int(args.conditional_candidates)))
        image_eval = image_eval_tail[image_candidate_start:image_candidate_end]
        audio_eval = audio_eval_tail[audio_candidate_start:audio_candidate_end]
        image_query_global = list(range(max(0, int(args.query_offset)), min(len(image_eval_tail), int(args.query_offset) + int(args.conditional_queries))))
        audio_query_global = list(range(max(0, int(args.query_offset)), min(len(audio_eval_tail), int(args.query_offset) + int(args.conditional_queries))))
        image_query_indices = [idx - image_candidate_start for idx in image_query_global if image_candidate_start <= idx < image_candidate_end]
        audio_query_indices = [idx - audio_candidate_start for idx in audio_query_global if audio_candidate_start <= idx < audio_candidate_end]

    if not image_query_indices or not audio_query_indices:
        raise RuntimeError(
            "empty conditional query window: "
            f"image_queries={len(image_query_indices)} speech_queries={len(audio_query_indices)} "
            f"query_offset={args.query_offset} candidate_offset={args.candidate_offset} "
            f"conditional_candidates={args.conditional_candidates}"
        )
    if frozen_evaluation_run is not None:
        expected_image_queries = int(
            frozen_evaluation_run["image_query_count"]
        )
        expected_speech_queries = int(
            frozen_evaluation_run["speech_query_count"]
        )
        if (
            len(image_query_indices) != expected_image_queries
            or len(audio_query_indices) != expected_speech_queries
        ):
            raise ValueError(
                "actual query cardinality after slicing disagrees with frozen run: "
                f"image={len(image_query_indices)}/{expected_image_queries} "
                f"speech={len(audio_query_indices)}/{expected_speech_queries}"
            )
        if not local_mode and (
            len(image_eval) != requested_candidate_count
            or len(audio_eval) != requested_candidate_count
        ):
            raise ValueError(
                "actual candidate cardinality after slicing disagrees with frozen run"
            )

    image_assignment_plan: Dict[str, Any] | None = None
    speech_assignment_plan: Dict[str, Any] | None = None
    image_assigned_positions: List[int] | None = None
    speech_assigned_positions: List[int] | None = None
    allocator_manifest: Mapping[str, Any] | None = None
    image_assignment_provenance: Dict[str, Any] | None = None
    speech_assignment_provenance: Dict[str, Any] | None = None
    if frozen_evaluation_run is not None and frozen_protocol is not None:
        raw_cells = frozen_protocol.get("evaluation_matrix")
        query_counts = frozen_protocol.get("query_counts")
        seeds = frozen_protocol.get("seeds")
        allocator_manifest = frozen_protocol.get("gold_position_allocator")
        if (
            not isinstance(raw_cells, list)
            or not isinstance(query_counts, Mapping)
            or not isinstance(seeds, Mapping)
            or not isinstance(allocator_manifest, Mapping)
        ):
            raise ValueError("frozen protocol is missing gold-position plan inputs")
        try:
            plans = validate_allocator_manifest(
                allocator_manifest,
                raw_cells,
                candidate_seed=int(seeds["candidate_seed"]),
                query_counts={
                    "image": int(query_counts["image"]),
                    "speech": int(query_counts["speech"]),
                },
                query_offset=int(frozen_evaluation_run["query_offset"]),
            )
            cell_id = str(frozen_evaluation_run["cell_id"])
            image_assignment_plan = plans[(cell_id, "image")]
            speech_assignment_plan = plans[(cell_id, "speech")]
            image_assigned_positions = positions_for_query_indices(
                image_assignment_plan,
                [idx + image_candidate_start for idx in image_query_indices],
            )
            speech_assigned_positions = positions_for_query_indices(
                speech_assignment_plan,
                [idx + audio_candidate_start for idx in audio_query_indices],
            )
        except (AssignmentPlanError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"frozen gold-position assignment preflight failed: {exc}"
            ) from exc
        expected_run_bindings = {
            "gold_position_allocator_name": ALLOCATOR_NAME,
            "gold_position_allocator_version": ALLOCATOR_VERSION,
            "gold_position_assignment_plans_sha256": allocator_manifest[
                "plans_sha256"
            ],
            "image_gold_position_assignment_id": image_assignment_plan[
                "assignment_id"
            ],
            "image_gold_positions_sha256": image_assignment_plan[
                "positions_sha256"
            ],
            "speech_gold_position_assignment_id": speech_assignment_plan[
                "assignment_id"
            ],
            "speech_gold_positions_sha256": speech_assignment_plan[
                "positions_sha256"
            ],
        }
        if any(
            frozen_evaluation_run.get(key) != value
            for key, value in expected_run_bindings.items()
        ):
            raise ValueError(
                "frozen evaluation run disagrees with gold-position assignment plan"
            )
        image_assignment_provenance = gold_position_assignment_provenance(
            image_assignment_plan, allocator_manifest
        )
        speech_assignment_provenance = gold_position_assignment_provenance(
            speech_assignment_plan, allocator_manifest
        )

    frozen_checkpoint_bytes = (
        frozen_inputs.get("e3_checkpoint", {}).get("bytes")
        if frozen_inputs
        else None
    )
    wrapper, tokenizer, meta, image_processor, vision_model, speech_processor, speech_model, device = load_trained_wrapper(
        args,
        checkpoint_bytes=frozen_checkpoint_bytes,
    )
    loaded_restoration = meta.get("checkpoint_restoration")
    if not isinstance(loaded_restoration, Mapping):
        raise ValueError(
            "loaded model metadata is missing checkpoint restoration identity"
        )
    cache_checkpoint_sha256 = loaded_restoration.get(
        "e3_checkpoint_sha256"
    )
    if (
        not isinstance(cache_checkpoint_sha256, str)
        or SHA256_RE.fullmatch(cache_checkpoint_sha256) is None
    ):
        raise ValueError(
            "image cache requires exact loaded checkpoint identity"
        )
    cache_identity_context = {
        "checkpoint_sha256": cache_checkpoint_sha256,
        "evaluator_sha256": evaluator_sha256,
    }
    cache = FeatureCache(
        cache_root,
        image_cache_context=cache_identity_context,
        audio_cache_context=cache_identity_context,
        access_policy=(
            "exclusive_write_only"
            if sealed_protocol
            else "verified_read_write"
        ),
    )
    image_cache_provenance = cache.image_cache_provenance(
        image_processor, vision_model, int(args.encoder_feature_tokens)
    )
    image_cache_identity_sha256 = canonical_sha256(
        image_cache_provenance
    )
    image_cache_media_provenance = cache.image_media_provenance(
        image_eval
    )
    image_cache_media_set_sha256 = canonical_sha256({
        "media": image_cache_media_provenance,
    })
    audio_cache_provenance = cache.audio_cache_provenance(
        speech_processor,
        speech_model,
        int(args.sample_rate),
        int(args.encoder_feature_tokens),
        float(args.audio_max_seconds),
    )
    audio_cache_identity_sha256 = canonical_sha256(
        audio_cache_provenance
    )
    audio_cache_media_provenance = cache.audio_media_provenance(
        audio_eval
    )
    audio_cache_media_set_sha256 = canonical_sha256({
        "media": audio_cache_media_provenance,
    })

    image_candidates = [str(row["caption"]) for row in image_eval]
    speech_candidates = [str(row["transcript"]) for row in audio_eval]
    image_group_ids = [image_group_identity(row) for row in image_eval]
    image_hard_indices: List[List[int]] = [[] for _ in image_eval]
    speech_hard_indices: List[List[int]] = [[] for _ in audio_eval]
    hard_negative_selector = None
    if args.negative_mode == "hard_text" and int(args.conditional_negatives) >= 0:
        hard_negative_selector = "lexical_jaccard_v1"
        image_hard_indices = lexical_hard_negative_indices(
            image_candidates,
            int(args.conditional_negatives),
            group_ids=image_group_ids,
        )
        speech_hard_indices = lexical_hard_negative_indices(
            speech_candidates, int(args.conditional_negatives)
        )
    image_ranks: List[int] = []
    image_evidence: List[Dict[str, Any]] = []
    speech_ranks: List[int] = []
    speech_evidence: List[Dict[str, Any]] = []
    image_gold_positions: List[int] = []
    speech_gold_positions: List[int] = []
    image_positive_counts: List[int] = []
    image_unique_candidate_group_counts: List[int] = []
    image_group_aware_chances: List[float] = []
    image_caption_row_chances: List[float] = []
    per_query_rows: List[Dict[str, Any]] = []
    condition = "no_prefix" if args.eval_path == "no_prefix_lm" else str(args.prefix_control)
    for done_idx, idx in enumerate(image_query_indices):
        row = image_eval[idx]
        query_group_id = image_group_ids[idx]
        global_idx = int(idx + image_candidate_start)
        query_seed = int(args.candidate_seed) + 1009 * global_idx
        query_id = conditional_query_identity(row, "image", global_idx)
        if local_mode:
            base_candidate_indices, _ = local_candidate_indices(
                image_eval,
                idx,
                args.conditional_negatives,
                args.negative_mode,
                image_hard_indices[idx],
                candidate_seed=query_seed,
                group_ids=image_group_ids,
            )
        else:
            base_candidate_indices, _ = full_candidate_indices(
                len(image_eval),
                idx,
                candidate_seed=query_seed,
                randomize_positive_position=False,
            )
        candidate_indices, permutation, gold_index, permutation_seed = (
            permute_candidates_for_query(
                base_candidate_indices, idx, args.control_seed, query_id
            )
        )
        if image_assigned_positions is not None:
            candidate_indices, permutation, gold_index = (
                enforce_gold_position_assignment(
                    candidate_indices,
                    permutation,
                    idx,
                    image_assigned_positions[done_idx],
                )
            )
        candidate_rows = [image_eval[position] for position in candidate_indices]
        candidates = [str(candidate["caption"]) for candidate in candidate_rows]
        candidate_group_ids = [
            image_group_ids[position] for position in candidate_indices
        ]
        positive_indices = positive_indices_for_group(candidate_group_ids, query_group_id)
        if local_mode and len(positive_indices) != 1:
            raise RuntimeError(
                "local image candidate set must contain exactly one query-group caption"
            )
        candidate_record_indices = [
            int(position + image_candidate_start) for position in candidate_indices
        ]
        candidate_ids = [
            conditional_query_identity(candidate, "image", candidate_record_indices[position])
            for position, candidate in enumerate(candidate_rows)
        ]
        if args.eval_path == "no_prefix_lm":
            scores = score_no_prefix_query(wrapper, tokenizer, "Caption:", candidates, device, args)
            feature_row = None
        else:
            feature_idx = (
                deterministic_derangement_index(idx, len(image_eval), args.control_seed)
                if args.prefix_control == "shuffled"
                else idx
            )
            feature_row = image_eval[feature_idx]
            feat = cache.image_batch(image_processor, vision_model, [feature_row], device, args.encoder_feature_tokens)
            feat = apply_prefix_control(feat, args.prefix_control, args.control_seed + global_idx)
            scores = score_image_query(
                wrapper,
                tokenizer,
                image_processor,
                vision_model,
                cache,
                row,
                candidates,
                device,
                args,
                feature_override=feat,
            )
        nll_scores = [-float(score) for score in scores]
        evidence = multi_positive_nll_evidence(
            nll_scores,
            positive_indices,
            args.tie_epsilon,
        )
        rank = int(evidence["strict_rank"])
        predicted_position = int(evidence["best_candidate_indices"][0])
        prefix_source_uid = (
            conditional_query_identity(feature_row, "image", feature_idx + image_candidate_start)
            if feature_row is not None else None
        )
        image_ranks.append(rank)
        image_evidence.append(evidence)
        image_gold_positions.append(gold_index)
        image_positive_counts.append(len(positive_indices))
        image_unique_candidate_group_counts.append(len(set(candidate_group_ids)))
        image_group_aware_chances.append(
            group_aware_chance_r_at_1(candidate_group_ids)
        )
        image_caption_row_chances.append(
            float(len(positive_indices) / max(1, len(candidate_group_ids)))
        )
        per_query_rows.append({
            "modality": "image",
            "query_uid": query_id,
            "query_index": int(global_idx),
            "query_source": str(row.get("source", "")),
            "query_image_group_id": query_group_id,
            "eval_split_name": str(args.eval_split_name),
            "candidate_ids": candidate_ids,
            "candidate_indices": candidate_record_indices,
            "candidate_permutation": permutation,
            "candidate_permutation_seed": int(permutation_seed),
            "gold_position_assignment": (
                {
                    **image_assignment_provenance,
                    "assignment_index": int(done_idx),
                    "assigned_position": int(gold_index),
                }
                if image_assignment_provenance is not None
                else None
            ),
            "candidate_texts": candidates,
            "candidate_set_hash": candidate_set_hash(candidate_ids),
            "candidate_group_ids": candidate_group_ids,
            "positive_indices": positive_indices,
            "positive_candidate_indices": [
                candidate_record_indices[position] for position in positive_indices
            ],
            "positive_candidate_ids": [
                candidate_ids[position] for position in positive_indices
            ],
            "positive_count": len(positive_indices),
            "unique_candidate_group_count": len(set(candidate_group_ids)),
            "group_aware_chance_r_at_1": group_aware_chance_r_at_1(candidate_group_ids),
            "caption_row_chance_r_at_1": float(
                len(positive_indices) / max(1, len(candidate_group_ids))
            ),
            "candidate_index": int(gold_index),
            "gold_index": int(gold_index),
            "gold_position": int(gold_index),
            "gold_candidate_index": int(global_idx),
            "gold_candidate_id": candidate_ids[gold_index],
            "predicted_position": int(predicted_position),
            "predicted_candidate_id": candidate_ids[predicted_position],
            "best_candidate_indices": evidence["best_candidate_indices"],
            "rank": int(rank),
            "strict_rank": int(rank),
            "strict_r_at_1": float(evidence["strict_r_at_1"]),
            "reciprocal_rank": float(evidence["reciprocal_rank"]),
            "rank_base": 0,
            "scores": [float(score) for score in scores],
            "score_direction": "higher_is_better",
            "raw_nll_scores": nll_scores,
            "nll_direction": "lower_is_better",
            "gold_nll": float(evidence["gold_nll"]),
            "gold_nll_margin": float(evidence["gold_nll_margin"]),
            "best_positive_nll": float(evidence["best_positive_nll"]),
            "best_positive_nll_margin": float(evidence["best_positive_nll_margin"]),
            "best_positive_indices": evidence["best_positive_indices"],
            "gold_margin": float(evidence["gold_nll_margin"]),
            "tie_count": int(evidence["gold_tie_count"]),
            "gold_tie_count": int(evidence["gold_tie_count"]),
            "best_tie_count": int(evidence["best_tie_count"]),
            "candidate_count": len(candidates),
            "condition": condition,
            "prefix_control": condition,
            "prefix_source_uid": prefix_source_uid,
            "negative_mode": str(args.negative_mode),
            "eval_path": str(args.eval_path),
            "control_provenance": {
                "control_seed": int(args.control_seed),
                "condition": condition,
                "prefix_control": condition,
                "eval_path": str(args.eval_path),
                "prefix_source_uid": prefix_source_uid,
            },
            "source_provenance": {
                "query_uid": query_id,
                "query_index": int(global_idx),
                "query_source": str(row.get("source", "")),
                "split_source": image_split_source,
                "query_image_group_id": query_group_id,
                "manifest_path": str(image_manifest_path.resolve()) if image_manifest_path is not None else None,
            },
            "protocol": {
                "name": str(args.protocol_name),
                "manifest_sha256": protocol_manifest_sha256,
                "eval_split_name": str(args.eval_split_name),
                "negative_mode": str(args.negative_mode),
                "hard_negative_selector": hard_negative_selector,
                "candidate_count": len(candidates),
                "candidate_seed": int(args.candidate_seed),
                "group_aware": True,
                "group_aware_semantics": IMAGE_GROUP_PROTOCOL_SEMANTICS,
                "positive_count": len(positive_indices),
                "candidate_permutation_policy": str(args.candidate_permutation),
                "candidate_permutation_seed_source": "control_seed+stable_query_identity_sha256",
                "randomized_positive_position": True,
                "gold_position_allocator_name": (
                    ALLOCATOR_NAME if allocator_manifest is not None else None
                ),
                "gold_position_allocator_version": (
                    ALLOCATOR_VERSION if allocator_manifest is not None else None
                ),
                "gold_position_assignment_plans_sha256": (
                    allocator_manifest["plans_sha256"]
                    if allocator_manifest is not None
                    else None
                ),
                "tie_policy": "strict_pessimistic_epsilon",
                "tie_epsilon": float(args.tie_epsilon),
                "rank_base": 0,
                "score_direction": "higher_is_better",
                "nll_direction": "lower_is_better",
            },
        })
        if done_idx == 0 or (done_idx + 1) % 25 == 0:
            print(json.dumps({"stage": "image_conditional", "done": done_idx + 1, "query_index": idx, "rank": rank}, sort_keys=True))

    for done_idx, idx in enumerate(audio_query_indices):
        row = audio_eval[idx]
        global_idx = int(idx + audio_candidate_start)
        query_seed = int(args.candidate_seed) + 1000003 + 1009 * global_idx
        query_id = conditional_query_identity(row, "speech", global_idx)
        if local_mode:
            base_candidate_indices, _ = local_candidate_indices(
                audio_eval,
                idx,
                args.conditional_negatives,
                args.negative_mode,
                speech_hard_indices[idx],
                candidate_seed=query_seed,
            )
        else:
            base_candidate_indices, _ = full_candidate_indices(
                len(audio_eval),
                idx,
                candidate_seed=query_seed,
                randomize_positive_position=False,
            )
        candidate_indices, permutation, gold_index, permutation_seed = (
            permute_candidates_for_query(
                base_candidate_indices, idx, args.control_seed, query_id
            )
        )
        if speech_assigned_positions is not None:
            candidate_indices, permutation, gold_index = (
                enforce_gold_position_assignment(
                    candidate_indices,
                    permutation,
                    idx,
                    speech_assigned_positions[done_idx],
                )
            )
        candidate_rows = [audio_eval[position] for position in candidate_indices]
        candidates = [str(candidate["transcript"]) for candidate in candidate_rows]
        candidate_record_indices = [
            int(position + audio_candidate_start) for position in candidate_indices
        ]
        candidate_ids = [
            conditional_query_identity(candidate, "speech", candidate_record_indices[position])
            for position, candidate in enumerate(candidate_rows)
        ]
        if args.eval_path == "no_prefix_lm":
            scores = score_no_prefix_query(wrapper, tokenizer, "Transcript:", candidates, device, args)
            feature_row = None
        else:
            feature_idx = (
                deterministic_derangement_index(idx, len(audio_eval), args.control_seed + 100000)
                if args.prefix_control == "shuffled"
                else idx
            )
            feature_row = audio_eval[feature_idx]
            feat = cache.audio_batch(
                speech_processor,
                speech_model,
                [feature_row],
                device,
                args.sample_rate,
                args.encoder_feature_tokens,
                args.audio_max_seconds,
            )
            feat = apply_prefix_control(feat, args.prefix_control, args.control_seed + 100000 + global_idx)
            scores = score_speech_query(
                wrapper,
                tokenizer,
                speech_processor,
                speech_model,
                cache,
                row,
                candidates,
                device,
                args,
                feature_override=feat,
            )
        nll_scores = [-float(score) for score in scores]
        evidence = tie_aware_nll_evidence(nll_scores, gold_index, args.tie_epsilon)
        rank = int(evidence["strict_rank"])
        predicted_position = int(evidence["best_candidate_indices"][0])
        prefix_source_uid = (
            conditional_query_identity(feature_row, "speech", feature_idx + audio_candidate_start)
            if feature_row is not None else None
        )
        speech_ranks.append(rank)
        speech_evidence.append(evidence)
        speech_gold_positions.append(gold_index)
        per_query_rows.append({
            "modality": "speech",
            "query_uid": query_id,
            "query_index": int(global_idx),
            "query_source": str(row.get("source", "")),
            "speaker_id": row.get("speaker_id"),
            "eval_split_name": str(args.eval_split_name),
            "candidate_ids": candidate_ids,
            "candidate_indices": candidate_record_indices,
            "candidate_permutation": permutation,
            "candidate_permutation_seed": int(permutation_seed),
            "gold_position_assignment": (
                {
                    **speech_assignment_provenance,
                    "assignment_index": int(done_idx),
                    "assigned_position": int(gold_index),
                }
                if speech_assignment_provenance is not None
                else None
            ),
            "candidate_texts": candidates,
            "candidate_set_hash": candidate_set_hash(candidate_ids),
            "candidate_index": int(gold_index),
            "gold_index": int(gold_index),
            "gold_position": int(gold_index),
            "gold_candidate_index": int(global_idx),
            "gold_candidate_id": candidate_ids[gold_index],
            "predicted_position": int(predicted_position),
            "predicted_candidate_id": candidate_ids[predicted_position],
            "best_candidate_indices": evidence["best_candidate_indices"],
            "rank": int(rank),
            "strict_rank": int(rank),
            "strict_r_at_1": float(evidence["strict_r_at_1"]),
            "reciprocal_rank": float(evidence["reciprocal_rank"]),
            "rank_base": 0,
            "scores": [float(score) for score in scores],
            "score_direction": "higher_is_better",
            "raw_nll_scores": nll_scores,
            "nll_direction": "lower_is_better",
            "gold_nll": float(evidence["gold_nll"]),
            "gold_nll_margin": float(evidence["gold_nll_margin"]),
            "gold_margin": float(evidence["gold_nll_margin"]),
            "tie_count": int(evidence["gold_tie_count"]),
            "gold_tie_count": int(evidence["gold_tie_count"]),
            "best_tie_count": int(evidence["best_tie_count"]),
            "candidate_count": len(candidates),
            "condition": condition,
            "prefix_control": condition,
            "prefix_source_uid": prefix_source_uid,
            "negative_mode": str(args.negative_mode),
            "eval_path": str(args.eval_path),
            "control_provenance": {
                "control_seed": int(args.control_seed),
                "condition": condition,
                "prefix_control": condition,
                "eval_path": str(args.eval_path),
                "prefix_source_uid": prefix_source_uid,
            },
            "source_provenance": {
                "query_uid": query_id,
                "query_index": int(global_idx),
                "query_source": str(row.get("source", "")),
                "speaker_id": row.get("speaker_id"),
                "split_source": speech_split_source,
                "manifest_path": str(speech_manifest_path.resolve()) if speech_manifest_path is not None else None,
            },
            "protocol": {
                "name": str(args.protocol_name),
                "manifest_sha256": protocol_manifest_sha256,
                "eval_split_name": str(args.eval_split_name),
                "negative_mode": str(args.negative_mode),
                "hard_negative_selector": hard_negative_selector,
                "candidate_count": len(candidates),
                "candidate_seed": int(args.candidate_seed),
                "candidate_permutation_policy": str(args.candidate_permutation),
                "candidate_permutation_seed_source": "control_seed+stable_query_identity_sha256",
                "randomized_positive_position": True,
                "gold_position_allocator_name": (
                    ALLOCATOR_NAME if allocator_manifest is not None else None
                ),
                "gold_position_allocator_version": (
                    ALLOCATOR_VERSION if allocator_manifest is not None else None
                ),
                "gold_position_assignment_plans_sha256": (
                    allocator_manifest["plans_sha256"]
                    if allocator_manifest is not None
                    else None
                ),
                "tie_policy": "strict_pessimistic_epsilon",
                "tie_epsilon": float(args.tie_epsilon),
                "rank_base": 0,
                "score_direction": "higher_is_better",
                "nll_direction": "lower_is_better",
            },
        })
        if done_idx == 0 or (done_idx + 1) % 25 == 0:
            print(json.dumps({"stage": "speech_conditional", "done": done_idx + 1, "query_index": idx, "rank": rank}, sort_keys=True))

    candidate_count = int(args.conditional_negatives) + 1 if local_mode else len(image_candidates)
    speech_candidate_count = int(args.conditional_negatives) + 1 if local_mode else len(speech_candidates)
    if frozen_evaluation_run is not None:
        expected_candidates = int(
            frozen_evaluation_run["requested_candidate_count"]
        )
        actual_counts = {
            int(row["candidate_count"]) for row in per_query_rows
        }
        if (
            candidate_count != expected_candidates
            or speech_candidate_count != expected_candidates
            or actual_counts != {expected_candidates}
        ):
            raise ValueError(
                "actual per-query candidate cardinality disagrees with frozen run"
            )
    if image_assignment_plan is not None and speech_assignment_plan is not None:
        try:
            validate_executed_positions(
                image_gold_positions, image_assignment_plan
            )
            validate_executed_positions(
                speech_gold_positions, speech_assignment_plan
            )
        except AssignmentPlanError as exc:
            raise ValueError(
                f"executed gold-position assignment validation failed: {exc}"
            ) from exc
    image_position_counts = [image_gold_positions.count(position) for position in range(candidate_count)]
    speech_position_counts = [speech_gold_positions.count(position) for position in range(speech_candidate_count)]
    image_group_aware_chance = float(
        sum(image_group_aware_chances) / max(1, len(image_group_aware_chances))
    )
    image_caption_row_chance = float(
        sum(image_caption_row_chances) / max(1, len(image_caption_row_chances))
    )
    checkpoint_path = Path(args.checkpoint).resolve()
    e3_checkpoint_sha256 = sha256_file(checkpoint_path)
    source_manifest_path = Path(args.run_output_dir) / "manifest.json"
    source_manifest_sha256 = (
        sha256_file(source_manifest_path) if source_manifest_path.exists() else None
    )
    raw_restoration = meta.get("checkpoint_restoration")
    if raw_restoration is not None and not isinstance(raw_restoration, Mapping):
        raise ValueError("checkpoint restoration metadata must be a mapping")
    checkpoint_restoration = dict(raw_restoration or {
        "restoration_order": ["base_model", "e3_training_checkpoint"],
        "stage_b_state_restored": False,
        "e3_state_overlaid": True,
        "e3_checkpoint_path": str(checkpoint_path),
        "e3_checkpoint_sha256": e3_checkpoint_sha256,
        "stage_b_checkpoint": None,
        "source_checkpoint_hashes": {
            "stage_b": None,
            "multimodal_initial": None,
            "speech_initial": None,
        },
    })
    if checkpoint_restoration.get("e3_checkpoint_sha256") != e3_checkpoint_sha256:
        raise ValueError(
            "loaded E3 checkpoint hash differs from metrics checkpoint identity"
        )
    source_checkpoint_hashes = dict(
        checkpoint_restoration.get("source_checkpoint_hashes") or {}
    )
    stage_b_checkpoint_sha256 = source_checkpoint_hashes.get("stage_b")
    raw_gamma_provenance = meta.get("gamma_provenance")
    if not isinstance(raw_gamma_provenance, Mapping):
        raise ValueError("loaded model metadata is missing gamma provenance")
    if (
        str(args.evaluation_scope) in CLAIM_EVALUATION_SCOPES
        and raw_gamma_provenance.get("checkpoint_bound") is not True
    ):
        raise ValueError(
            "claim evaluation requires checkpoint-bound gamma provenance"
        )
    gamma_path_value = raw_gamma_provenance.get("path")
    gamma_sha256 = raw_gamma_provenance.get("sha256")
    if not isinstance(gamma_path_value, str):
        raise ValueError("gamma provenance is missing path")
    if not isinstance(gamma_sha256, str) or SHA256_RE.fullmatch(gamma_sha256) is None:
        raise ValueError("gamma provenance is missing exact SHA256")
    gamma_path = Path(gamma_path_value).resolve(strict=True)
    checkpoint_output_dir_value = raw_gamma_provenance.get(
        "checkpoint_output_dir"
    )
    if not isinstance(checkpoint_output_dir_value, str):
        raise ValueError(
            "gamma provenance is missing checkpoint output directory"
        )
    run_output_root = Path(args.run_output_dir).expanduser().resolve(strict=True)
    checkpoint_output_root = (
        Path(checkpoint_output_dir_value).expanduser().resolve(strict=True)
    )
    expected_gamma_path = (
        run_output_root / "calibration" / "gamma.json"
    ).resolve(strict=True)
    if checkpoint_output_root != run_output_root or gamma_path != expected_gamma_path:
        raise ValueError(
            "gamma provenance does not match the evaluated E3 run: "
            f"run={run_output_root} checkpoint={checkpoint_output_root} "
            f"gamma={gamma_path}"
        )
    verify_file_sha256(gamma_path, gamma_sha256, "gamma JSON")
    if image_manifest_path is not None and image_manifest_sha256 is not None:
        verify_file_sha256(
            image_manifest_path, image_manifest_sha256, "image manifest"
        )
    if speech_manifest_path is not None and speech_manifest_sha256 is not None:
        verify_file_sha256(
            speech_manifest_path, speech_manifest_sha256, "speech manifest"
        )
    if protocol_manifest_path is not None and protocol_manifest_sha256 is not None:
        verify_file_sha256(
            protocol_manifest_path, protocol_manifest_sha256, "protocol manifest"
        )
    for label, record in frozen_inputs.items():
        if record.get("type") != "file":
            continue
        verify_file_sha256(
            Path(record["path"]),
            str(record["sha256"]),
            f"frozen input {label}",
        )
    if (
        protocol_manifest_path is not None
        and image_manifest_path is not None
        and speech_manifest_path is not None
    ):
        _, final_protocol_sha256, _ = load_verified_frozen_protocol(
            protocol_manifest_path,
            image_manifest_path,
            speech_manifest_path,
            evaluator_path,
            checkpoint_path,
        )
        if final_protocol_sha256 != protocol_manifest_sha256:
            raise ValueError(
                "frozen protocol identity changed during evaluation"
            )
    for cache_kind, media_records in (
        ("image", image_cache_media_provenance),
        ("audio", audio_cache_media_provenance),
    ):
        for record in media_records:
            verify_file_sha256(
                Path(record["canonical_path"]),
                str(record["sha256"]),
                f"{cache_kind} cache media",
            )
    for modality, rows in (("image", image_eval), ("speech", audio_eval)):
        for row in rows:
            verify_file_sha256(
                Path(str(row["_source_media_path"])),
                str(row["_verified_media_sha256"]),
                f"{modality} source media",
            )
    image_cache_payloads = cache.verify_payloads("image")
    audio_cache_payloads = cache.verify_payloads("audio")
    image_cache_payload_set_sha256 = canonical_sha256(
        image_cache_payloads
    )
    audio_cache_payload_set_sha256 = canonical_sha256(
        audio_cache_payloads
    )
    image_produced_payloads = cache.produced_payload_provenance("image")
    audio_produced_payloads = cache.produced_payload_provenance("audio")
    if sealed_protocol and (
        image_produced_payloads != image_cache_payloads
        or audio_produced_payloads != audio_cache_payloads
    ):
        raise ValueError(
            "final evaluation observed a feature payload that was not recomputed in this run"
        )
    image_produced_features_sha256 = canonical_sha256({
        "features": [
            record["tensor_digest"] for record in image_produced_payloads
        ],
    })
    audio_produced_features_sha256 = canonical_sha256({
        "features": [
            record["tensor_digest"] for record in audio_produced_payloads
        ],
    })
    verify_file_sha256(evaluator_path, evaluator_sha256, "evaluator source")
    evaluation_identity = {
        **run_environment_provenance,
        "evaluation_scope": str(args.evaluation_scope),
        "eval_split_name": str(args.eval_split_name),
        "strict_control": bool(strict_control),
        "condition": condition,
        "prefix_control": condition,
        "eval_path": str(args.eval_path),
        "control_seed": int(args.control_seed),
        "candidate_seed": int(args.candidate_seed),
        "gold_position_allocator_name": (
            ALLOCATOR_NAME if allocator_manifest is not None else None
        ),
        "gold_position_allocator_version": (
            ALLOCATOR_VERSION if allocator_manifest is not None else None
        ),
        "gold_position_assignment_plans_sha256": (
            allocator_manifest["plans_sha256"]
            if allocator_manifest is not None
            else None
        ),
        "image_gold_position_assignment": image_assignment_provenance,
        "speech_gold_position_assignment": speech_assignment_provenance,
        "negative_mode": str(args.negative_mode),
        "requested_query_count": int(args.conditional_queries),
        "image_query_count": len(image_query_indices),
        "speech_query_count": len(audio_query_indices),
        "requested_candidate_count": int(requested_candidate_count),
        "image_candidate_count": int(candidate_count),
        "speech_candidate_count": int(speech_candidate_count),
        "conditional_negatives": int(args.conditional_negatives),
        "conditional_candidates": int(args.conditional_candidates),
        "query_offset": int(args.query_offset),
        "candidate_offset": int(args.candidate_offset),
        "tie_epsilon": float(args.tie_epsilon),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_samples": int(args.bootstrap_samples),
        "image_eval_samples": int(args.image_eval_samples),
        "speech_eval_samples": int(args.speech_eval_samples),
        "metric_affecting_args": metric_affecting_args(args),
        "checkpoint_architecture": meta.get("checkpoint_architecture"),
        "protocol_name": str(args.protocol_name),
        "protocol_manifest_path": (
            str(protocol_manifest_path) if protocol_manifest_path is not None else None
        ),
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "protocol_content_sha256": protocol_content_sha256,
        "frozen_input_hashes": {
            label: str(record["sha256"])
            for label, record in sorted(frozen_inputs.items())
        },
        "image_manifest_path": (
            str(image_manifest_path) if image_manifest_path is not None else None
        ),
        "image_manifest_sha256": image_manifest_sha256,
        "speech_manifest_path": (
            str(speech_manifest_path) if speech_manifest_path is not None else None
        ),
        "speech_manifest_sha256": speech_manifest_sha256,
        "gamma_path": str(gamma_path),
        "gamma_sha256": gamma_sha256,
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": evaluator_sha256,
        "image_cache_identity_sha256": image_cache_identity_sha256,
        "audio_cache_identity_sha256": audio_cache_identity_sha256,
        "image_cache_payload_set_sha256": image_cache_payload_set_sha256,
        "audio_cache_payload_set_sha256": audio_cache_payload_set_sha256,
        "image_produced_features_sha256": image_produced_features_sha256,
        "audio_produced_features_sha256": audio_produced_features_sha256,
        "feature_cache_policy": feature_cache_policy,
        "frozen_evaluation_run_id": (
            str(frozen_evaluation_run["id"])
            if frozen_evaluation_run is not None
            else None
        ),
        "frozen_evaluation_cell_id": (
            str(frozen_evaluation_run["cell_id"])
            if frozen_evaluation_run is not None
            else None
        ),
        "frozen_evaluation_control": (
            str(frozen_evaluation_run["control"])
            if frozen_evaluation_run is not None
            else None
        ),
        "source_run_manifest_sha256": source_manifest_sha256,
        "e3_checkpoint_sha256": e3_checkpoint_sha256,
        "e3_checkpoint_path": str(checkpoint_path),
        "stage_b_checkpoint_sha256": stage_b_checkpoint_sha256,
        "source_checkpoint_hashes": source_checkpoint_hashes,
        "restoration_order": checkpoint_restoration.get("restoration_order"),
    }
    validate_complete_evaluation_identity(evaluation_identity)
    evaluation_provenance = {
        **evaluation_identity,
        "evaluation_identity_sha256": canonical_sha256(evaluation_identity),
    }
    bind_per_query_evaluation_provenance(
        per_query_rows, evaluation_provenance
    )
    provenance = {
        **run_environment_provenance,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": e3_checkpoint_sha256,
        "checkpoint_size_bytes": int(checkpoint_path.stat().st_size),
        "checkpoint_restoration": checkpoint_restoration,
        "source_checkpoint_hashes": source_checkpoint_hashes,
        "stage_b_checkpoint_sha256": stage_b_checkpoint_sha256,
        "evaluation_identity_sha256": evaluation_provenance[
            "evaluation_identity_sha256"
        ],
        "resolved_data_dir": str(data_dir),
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": evaluator_sha256,
        "gamma_path": str(gamma_path),
        "gamma_sha256": gamma_sha256,
        "gamma_size_bytes": int(raw_gamma_provenance.get("size_bytes", 0)),
        "gamma_checkpoint_output_dir": str(checkpoint_output_root),
        "gamma_checkpoint_bound": bool(
            raw_gamma_provenance.get("checkpoint_bound")
        ),
        "gamma_checkpoint_expected_sha256": raw_gamma_provenance.get(
            "checkpoint_expected_sha256"
        ),
        "source_run_manifest_path": (
            str(source_manifest_path) if source_manifest_path.exists() else None
        ),
        "source_run_manifest_sha256": source_manifest_sha256,
        "image_manifest_path": (
            str(image_manifest_path.resolve())
            if image_manifest_path is not None
            else None
        ),
        "image_manifest_sha256": evaluation_identity["image_manifest_sha256"],
        "speech_manifest_path": (
            str(speech_manifest_path.resolve())
            if speech_manifest_path is not None
            else None
        ),
        "speech_manifest_sha256": evaluation_identity["speech_manifest_sha256"],
        "protocol_manifest_path": (
            str(protocol_manifest_path.resolve())
            if protocol_manifest_path is not None
            else None
        ),
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "protocol_content_sha256": protocol_content_sha256,
        "frozen_inputs": {
            label: {
                key: value
                for key, value in record.items()
                if key != "bytes"
            }
            for label, record in sorted(frozen_inputs.items())
        },
        "image_feature_cache": {
            "policy": feature_cache_policy,
            "base_provenance": image_cache_provenance,
            "base_identity_sha256": image_cache_identity_sha256,
            "media": image_cache_media_provenance,
            "media_set_sha256": image_cache_media_set_sha256,
            "payloads": image_cache_payloads,
            "payload_set_sha256": image_cache_payload_set_sha256,
            "produced_payloads": image_produced_payloads,
            "produced_features_sha256": image_produced_features_sha256,
        },
        "audio_feature_cache": {
            "policy": feature_cache_policy,
            "base_provenance": audio_cache_provenance,
            "base_identity_sha256": audio_cache_identity_sha256,
            "media": audio_cache_media_provenance,
            "media_set_sha256": audio_cache_media_set_sha256,
            "payloads": audio_cache_payloads,
            "payload_set_sha256": audio_cache_payload_set_sha256,
            "produced_payloads": audio_produced_payloads,
            "produced_features_sha256": audio_produced_features_sha256,
        },
        "feature_cache_dir": str(cache_root.resolve()),
    }
    metrics = {
        "mode": "conditional_nll_local_negatives" if local_mode else "conditional_nll_full_matrix",
        "conditional_eval_path": "no_prefix_lm_nll" if args.eval_path == "no_prefix_lm" else "shared_olmoe_prefix_lm_nll",
        "conditional_uses_lm_logits": True,
        "conditional_uses_direct_encoder_pooling": False,
        "conditional_uses_multimodal_prefix": args.eval_path == "shared_prefix",
        "eval_path": str(args.eval_path),
        "runtime_top_k": int(args.top_k),
        "prefix_control": condition,
        "condition": condition,
        "e3_checkpoint_sha256": e3_checkpoint_sha256,
        "e3_checkpoint_path": str(checkpoint_path),
        "stage_b_checkpoint_sha256": stage_b_checkpoint_sha256,
        "source_checkpoint_hashes": source_checkpoint_hashes,
        "restoration_order": checkpoint_restoration.get("restoration_order"),
        "evaluation_identity_sha256": evaluation_provenance[
            "evaluation_identity_sha256"
        ],
        "evaluation_provenance": evaluation_provenance,
        "protocol_name": str(args.protocol_name),
        "protocol_manifest_path": str(protocol_manifest_path.resolve()) if protocol_manifest_path is not None else None,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "protocol_content_sha256": protocol_content_sha256,
        "evaluation_scope": str(args.evaluation_scope),
        "strict_control": bool(strict_control),
        "negative_mode": str(args.negative_mode),
        "hard_negative_selector": hard_negative_selector,
        "eval_split_name": str(args.eval_split_name),
        "sealed_protocol": bool(sealed_protocol),
        "image_split_source": image_split_source,
        "speech_split_source": speech_split_source,
        "candidate_seed": int(args.candidate_seed),
        "control_seed": int(args.control_seed),
        "randomized_positive_position": True,
        "gold_position_allocator_name": (
            ALLOCATOR_NAME if allocator_manifest is not None else None
        ),
        "gold_position_allocator_version": (
            ALLOCATOR_VERSION if allocator_manifest is not None else None
        ),
        "gold_position_assignment_plans_sha256": (
            allocator_manifest["plans_sha256"]
            if allocator_manifest is not None
            else None
        ),
        "image_gold_position_assignment": image_assignment_provenance,
        "speech_gold_position_assignment": speech_assignment_provenance,
        "candidate_permutation_policy": str(args.candidate_permutation),
        "candidate_permutation_seed_source": "control_seed+stable_query_identity_sha256",
        "tie_policy": "strict_pessimistic_epsilon",
        "tie_epsilon": float(args.tie_epsilon),
        "image_gold_position_counts": image_position_counts,
        "speech_gold_position_counts": speech_position_counts,
        "image_group_aware_protocol": IMAGE_GROUP_PROTOCOL_SEMANTICS,
        "image_positive_counts": image_positive_counts,
        "image_unique_candidate_group_counts": image_unique_candidate_group_counts,
        "image_candidate_bank_unique_group_count": len(set(image_group_ids)),
        "image_group_aware_chance_r_at_1": image_group_aware_chance,
        "image_caption_row_chance_r_at_1": image_caption_row_chance,
        "query_offset": int(args.query_offset),
        "candidate_offset": int(args.candidate_offset),
        "dynamic_expert_bias_loaded_layers": int(meta.get("dynamic_expert_bias_loaded_layers", 0) or 0),
        "image_eval_tail_count": len(image_eval_tail),
        "speech_eval_tail_count": len(audio_eval_tail),
        "image_candidate_start": int(image_candidate_start),
        "image_candidate_end_exclusive": int(image_candidate_start + len(image_eval)),
        "speech_candidate_start": int(audio_candidate_start),
        "speech_candidate_end_exclusive": int(audio_candidate_start + len(audio_eval)),
        "checkpoint": str(args.checkpoint),
        "run_output_dir": str(args.run_output_dir),
        "source_commit_sha": run_environment_provenance["source_commit_sha"],
        "runai_job_name": run_environment_provenance["runai_job_name"],
        "runai_project": run_environment_provenance["runai_project"],
        "provenance": provenance,
        "image_eval_count": len(image_query_indices),
        "speech_eval_count": len(audio_query_indices),
        "image_query_start": int(image_query_indices[0] + image_candidate_start) if image_query_indices else None,
        "image_query_end_exclusive": int(image_query_indices[-1] + image_candidate_start + 1) if image_query_indices else None,
        "speech_query_start": int(audio_query_indices[0] + audio_candidate_start) if audio_query_indices else None,
        "speech_query_end_exclusive": int(audio_query_indices[-1] + audio_candidate_start + 1) if audio_query_indices else None,
        "candidate_count": candidate_count,
        "speech_candidate_count": speech_candidate_count,
        "image_chance_r_at_1": image_caption_row_chance,
        "image_legacy_gold_caption_position_chance_r_at_1": (
            1.0 / max(1, candidate_count)
        ),
        "speech_chance_r_at_1": 1.0 / max(1, speech_candidate_count),
        **tie_aware_metrics(image_ranks, image_evidence, "image_to_text"),
        **tie_aware_metrics(speech_ranks, speech_evidence, "speech_to_text"),
        **{f"image_to_text_{k}": v for k, v in bootstrap_r_at_1_ci(image_ranks, args.bootstrap_samples, args.bootstrap_seed).items()},
        **{f"speech_to_text_{k}": v for k, v in bootstrap_r_at_1_ci(speech_ranks, args.bootstrap_samples, args.bootstrap_seed + 1).items()},
        "meta": meta,
    }
    metrics.update({
        "conditional_candidates_per_query": candidate_count,
        "conditional_speech_candidates_per_query": speech_candidate_count,
        "conditional_image_eval_count": len(image_query_indices),
        "conditional_speech_eval_count": len(audio_query_indices),
        "conditional_image_chance_r_at_1": metrics["image_caption_row_chance_r_at_1"],
        "conditional_image_legacy_gold_caption_position_chance_r_at_1": metrics[
            "image_legacy_gold_caption_position_chance_r_at_1"
        ],
        "conditional_speech_chance_r_at_1": metrics["speech_chance_r_at_1"],
        "conditional_image_group_aware_chance_r_at_1": metrics["image_group_aware_chance_r_at_1"],
        "conditional_image_caption_row_chance_r_at_1": metrics["image_caption_row_chance_r_at_1"],
        "conditional_image_positive_counts": metrics["image_positive_counts"],
        "conditional_image_unique_candidate_group_counts": metrics["image_unique_candidate_group_counts"],
        "conditional_image_to_text_strict_r_at_1": metrics.get("image_to_text_strict_r_at_1"),
        "conditional_image_to_text_mrr": metrics.get("image_to_text_mrr"),
        "conditional_image_to_text_mean_gold_nll_margin": metrics.get("image_to_text_mean_gold_nll_margin"),
        "conditional_image_to_text_tie_count": metrics.get("image_to_text_tie_count"),
        "conditional_image_to_text_tie_rate": metrics.get("image_to_text_tie_rate"),
        "conditional_speech_to_text_strict_r_at_1": metrics.get("speech_to_text_strict_r_at_1"),
        "conditional_speech_to_text_mrr": metrics.get("speech_to_text_mrr"),
        "conditional_speech_to_text_mean_gold_nll_margin": metrics.get("speech_to_text_mean_gold_nll_margin"),
        "conditional_speech_to_text_tie_count": metrics.get("speech_to_text_tie_count"),
        "conditional_speech_to_text_tie_rate": metrics.get("speech_to_text_tie_rate"),
        "conditional_image_to_text_r_at_1": metrics.get("image_to_text_r_at_1"),
        "conditional_image_to_text_r_at_5": metrics.get("image_to_text_r_at_5"),
        "conditional_image_to_text_r_at_10": metrics.get("image_to_text_r_at_10"),
        "conditional_speech_to_text_r_at_1": metrics.get("speech_to_text_r_at_1"),
        "conditional_speech_to_text_r_at_5": metrics.get("speech_to_text_r_at_5"),
        "conditional_speech_to_text_r_at_10": metrics.get("speech_to_text_r_at_10"),
        "conditional_image_to_text_r_at_1_bootstrap_ci_low": metrics.get("image_to_text_r_at_1_bootstrap_ci_low"),
        "conditional_image_to_text_r_at_1_bootstrap_ci_high": metrics.get("image_to_text_r_at_1_bootstrap_ci_high"),
        "conditional_speech_to_text_r_at_1_bootstrap_ci_low": metrics.get("speech_to_text_r_at_1_bootstrap_ci_low"),
        "conditional_speech_to_text_r_at_1_bootstrap_ci_high": metrics.get("speech_to_text_r_at_1_bootstrap_ci_high"),
    })
    per_query_content: str | None = None
    if per_query_path is not None:
        per_query_content = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in per_query_rows
        )
        metrics["per_query_output"] = str(per_query_path.resolve())
        metrics["per_query_sha256"] = sha256_text(per_query_content)
        metrics["per_query_rows"] = len(per_query_rows)
    metrics["output_publication_policy"] = "atomic_exclusive_no_reuse"
    metrics_content = json.dumps(
        metrics, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    published_per_query = False
    published_metrics = False
    published_manifest = False
    manifest_digest: str | None = None
    try:
        if per_query_path is not None and per_query_content is not None:
            atomic_write_text_exclusive(per_query_path, per_query_content)
            published_per_query = True
        atomic_write_text_exclusive(output_path, metrics_content)
        published_metrics = True
        if manifest_path is not None:
            if protocol_manifest_path is None or protocol_content_sha256 is None:
                raise ValueError(
                    "result manifest publication requires a frozen protocol"
                )
            runai_job = run_environment_provenance.get("runai_job_name")
            runai_project = run_environment_provenance.get("runai_project")
            if not runai_job or not runai_project:
                raise ValueError(
                    "result manifest publication requires RUNAI_JOB_NAME and RUNAI_PROJECT"
                )
            if frozen_evaluation_run is not None:
                expected_role = f"sealed:{frozen_evaluation_run['id']}"
                if args.result_role != expected_role:
                    raise ValueError(
                        "result role disagrees with frozen evaluation run: "
                        f"expected={expected_role} observed={args.result_role}"
                    )
            runai_identity = {"job": runai_job, "project": runai_project}
            for manifest_key, environment_key in (
                ("job_uid", "RUNAI_JOB_UID"),
                ("pod", "RUNAI_POD_NAME"),
                ("pod_uid", "RUNAI_POD_UID"),
            ):
                value = os.environ.get(environment_key)
                if value:
                    runai_identity[manifest_key] = value
            checkpoint_record = result_manifest.file_record(checkpoint_path)
            manifest_payload = {
                "schema_version": result_manifest.MANIFEST_SCHEMA_VERSION,
                "manifest_type": result_manifest.MANIFEST_TYPE,
                "commit": run_environment_provenance["source_commit_sha"],
                "runai": runai_identity,
                "checkpoint": checkpoint_record,
                "protocol": {
                    "path": str(protocol_manifest_path.resolve()),
                    "file_sha256": sha256_file(protocol_manifest_path),
                    "content_sha256": protocol_content_sha256,
                },
                "evaluations": {
                    str(args.result_role): {
                        "evaluation_identity": {
                            "role": str(args.result_role),
                            "evaluation_id": evaluation_provenance[
                                "evaluation_identity_sha256"
                            ],
                            "checkpoint_sha256": checkpoint_record["sha256"],
                            "protocol_content_sha256": protocol_content_sha256,
                        },
                        "artifacts": {
                            "metrics": result_manifest.file_record(
                                output_path.resolve()
                            ),
                            "per_query": result_manifest.file_record(
                                per_query_path.resolve()
                            ),
                        },
                        "metrics_sha256": sha256_file(output_path.resolve()),
                        "per_query_sha256": sha256_file(
                            per_query_path.resolve()
                        ),
                    }
                },
            }
            result_manifest.validate_manifest_schema(manifest_payload)
            atomic_write_text_exclusive(
                manifest_path,
                result_manifest.canonical_json(manifest_payload) + "\n",
            )
            published_manifest = True
            manifest_digest = sha256_file(manifest_path.resolve())
    except Exception:
        if published_manifest and manifest_path is not None:
            manifest_path.resolve().unlink(missing_ok=True)
        if published_metrics:
            output_path.resolve().unlink(missing_ok=True)
        if published_per_query and per_query_path is not None:
            per_query_path.resolve().unlink(missing_ok=True)
        raise
    if manifest_digest is not None:
        print(result_manifest.result_manifest_log_marker(manifest_digest), flush=True)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
