"""Required ACDL Project 18 OLMoE runs.

Runs a compact but real evidence matrix on one 80GB GPU:
E0 Top-8 teacher baseline, E1 hard Top-2, E2 calibrated Top-2,
E3 multimodal Top-2 with CLIP/Whisper prefixes, and E4 no-aux ablation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import types
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn

from hf_sources import load_pretrained
from model.olmoe_adapter import OLMoEMultimodalPrefixWrapper


DEVELOPMENT_SPLITS = frozenset({"train", "dev"})
FORBIDDEN_EVIDENCE_TERMS = ("sealed", "synthetic")


class DevelopmentEvidenceError(ValueError):
    """Raised when a development-only diagnostic receives forbidden evidence."""


def choose_dtype() -> Tuple[torch.dtype, str]:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bfloat16"
    if torch.cuda.is_available():
        return torch.float16, "float16"
    return torch.float32, "float32"


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def base_model_identity(model, requested_base_model: str) -> Dict[str, Any]:
    """Fingerprint the complete loaded model before runtime routing mutation."""

    config = getattr(model, "config", None)
    config_payload = (
        config.to_dict()
        if config is not None and hasattr(config, "to_dict")
        else {
            key: value
            for key, value in vars(config).items()
            if isinstance(
                value,
                (str, int, float, bool, type(None), list, tuple, dict),
            )
        }
        if config is not None
        else {}
    )
    stable_config = json.loads(
        json.dumps(
            config_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )
    runtime_routing_fields = (
        "num_experts_per_tok",
        "router_aux_loss_coef",
        "top_k",
        "capacity_factor",
    )
    if isinstance(stable_config, dict):
        for field in runtime_routing_fields:
            stable_config.pop(field, None)
    tensor_hasher = hashlib.sha256()
    tensor_records: List[Dict[str, Any]] = []
    named_tensors = [
        ("parameter", name, tensor)
        for name, tensor in model.named_parameters()
    ] + [
        ("buffer", name, tensor)
        for name, tensor in model.named_buffers()
    ]
    for kind, name, tensor in sorted(
        named_tensors, key=lambda item: (item[1], item[0])
    ):
        detached = tensor.detach().cpu().contiguous()
        byte_view = detached.reshape(-1).view(torch.uint8)
        tensor_sha256 = hashlib.sha256(
            byte_view.numpy().tobytes()
        ).hexdigest()
        record = {
            "kind": kind,
            "name": name,
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "numel": int(detached.numel()),
            "tensor_sha256": tensor_sha256,
        }
        tensor_records.append(record)
        tensor_hasher.update(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        tensor_hasher.update(b"\n")
        del byte_view, detached
    config_sha256 = hashlib.sha256(
        json.dumps(
            stable_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "requested_base_model": str(requested_base_model),
        "model_class": (
            f"{model.__class__.__module__}.{model.__class__.__qualname__}"
        ),
        "model_name_or_path": getattr(model, "name_or_path", None),
        "config_name_or_path": getattr(config, "_name_or_path", None),
        "commit_hash": getattr(config, "_commit_hash", None),
        "revision": getattr(config, "revision", None),
        "config_sha256": config_sha256,
        "config_digest_excludes_runtime_fields": list(runtime_routing_fields),
        "loaded_tensor_identity": {
            "algorithm": "sha256_ordered_named_tensor_records_v1",
            "tensor_count": len(tensor_records),
            "parameters": sum(
                1 for record in tensor_records
                if record["kind"] == "parameter"
            ),
            "buffers": sum(
                1 for record in tensor_records
                if record["kind"] == "buffer"
            ),
            "records": tensor_records,
            "sha256": tensor_hasher.hexdigest(),
        },
    }


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def cuda_metrics() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"cuda_memory_allocated_gb": 0.0, "cuda_memory_reserved_gb": 0.0}
    return {
        "cuda_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "cuda_memory_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
    }


def cleanup(*objects: object) -> None:
    for obj in objects:
        try:
            if hasattr(obj, "to"):
                obj.to("cpu")
        except Exception:
            pass
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _forbidden_evidence_location(value: object, location: str) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_location = f"{location}.{key}"
            match = _forbidden_evidence_location(str(key), key_location)
            if match is not None:
                return match
            match = _forbidden_evidence_location(child, key_location)
            if match is not None:
                return match
        return None
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            match = _forbidden_evidence_location(child, f"{location}[{index}]")
            if match is not None:
                return match
        return None
    if isinstance(value, (str, Path)):
        lowered = str(value).lower()
        for term in FORBIDDEN_EVIDENCE_TERMS:
            if term in lowered:
                return f"{location} contains forbidden term {term!r}"
    return None


def validate_development_evidence(
    source_paths: Sequence[str | Path],
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fail closed unless rows are train/dev and contain no forbidden evidence."""
    for path in source_paths:
        match = _forbidden_evidence_location(Path(path), "source_path")
        if match is not None:
            raise DevelopmentEvidenceError(match)
    splits = set()
    for index, row in enumerate(rows):
        match = _forbidden_evidence_location(row, f"rows[{index}]")
        if match is not None:
            raise DevelopmentEvidenceError(match)
        split = str(row.get("split", row.get("data_split", ""))).strip().lower()
        if split not in DEVELOPMENT_SPLITS:
            raise DevelopmentEvidenceError(
                f"rows[{index}] must declare split=train or split=dev, got {split!r}"
            )
        if row.get("real_subset") is False:
            raise DevelopmentEvidenceError(f"rows[{index}] explicitly declares real_subset=false")
        splits.add(split)
    return {
        "policy": "development_only_real_train_dev",
        "source_paths": [str(Path(path)) for path in source_paths],
        "row_count": len(rows),
        "splits": sorted(splits),
        "forbidden_terms": list(FORBIDDEN_EVIDENCE_TERMS),
        "sealed_evidence_used": False,
        "synthetic_evidence_used": False,
    }


def _all_expert_outputs(experts: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Evaluate all packed OLMoE experts for a small diagnostic token batch."""
    num_experts = int(experts.num_experts)
    outputs: List[torch.Tensor] = []
    for expert_id in range(num_experts):
        gate, up = F.linear(hidden_states, experts.gate_up_proj[expert_id]).chunk(2, dim=-1)
        intermediate = experts.act_fn(gate) * up
        outputs.append(F.linear(intermediate, experts.down_proj[expert_id]))
    return torch.stack(outputs, dim=1)


def _weighted_expert_reconstruction(
    expert_outputs: torch.Tensor,
    probabilities: torch.Tensor,
    expert_ids: torch.Tensor,
    normalize_topk_prob: bool,
) -> torch.Tensor:
    token_ids = torch.arange(expert_ids.shape[0], device=expert_ids.device)[:, None]
    selected_outputs = expert_outputs[token_ids, expert_ids]
    selected_weights = probabilities.gather(1, expert_ids)
    if normalize_topk_prob:
        selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (selected_outputs.float() * selected_weights[..., None].float()).sum(dim=1)


@torch.no_grad()
def moe_reconstruction_diagnostics(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    native_top8_output: torch.Tensor | None = None,
    top_ks: Sequence[int] = (2, 4, 8),
    normalize_topk_prob: bool = False,
    oracle_candidate_k: int = 8,
) -> Dict[str, Any]:
    """Compare diagnostic Top-k reconstructions with a native Top-8 output.

    The oracle pair is an offline diagnostic over pairs drawn from each token's
    native Top-8 candidate set. It is never installed into model dispatch.
    """
    if hidden_states.ndim != 2 or router_logits.ndim != 2:
        raise ValueError("hidden_states and router_logits must both be rank-2")
    if hidden_states.shape[0] != router_logits.shape[0]:
        raise ValueError("hidden_states and router_logits must have the same token count")
    requested = tuple(sorted({int(value) for value in top_ks}))
    if not {2, 4, 8}.issubset(requested):
        raise ValueError("top_ks must include diagnostic Top-2/4/8")
    num_experts = int(router_logits.shape[-1])
    if requested[0] < 1 or requested[-1] > num_experts:
        raise ValueError("top_ks must be between 1 and num_experts")
    candidate_k = int(oracle_candidate_k)
    if candidate_k < 2 or candidate_k > num_experts:
        raise ValueError("oracle_candidate_k must be between 2 and num_experts")

    probabilities = torch.softmax(router_logits.float(), dim=-1)
    sorted_ids = torch.argsort(probabilities, dim=-1, descending=True, stable=True)
    expert_outputs = _all_expert_outputs(experts, hidden_states)
    top8_ids = sorted_ids[:, :8]
    reconstructed_top8 = _weighted_expert_reconstruction(
        expert_outputs, probabilities, top8_ids, bool(normalize_topk_prob)
    )
    reference = reconstructed_top8 if native_top8_output is None else native_top8_output.reshape_as(reconstructed_top8).float()

    reconstructions: Dict[str, Dict[str, float]] = {}
    for top_k in requested:
        ids = sorted_ids[:, :top_k]
        reconstructed = _weighted_expert_reconstruction(
            expert_outputs, probabilities, ids, bool(normalize_topk_prob)
        )
        mass = probabilities.gather(1, ids).sum(dim=-1)
        reconstructions[f"top_{top_k}"] = {
            "mse": float(F.mse_loss(reconstructed.float(), reference.float()).detach().cpu()),
            "cosine": float(F.cosine_similarity(reconstructed.float(), reference.float(), dim=-1).mean().detach().cpu()),
            "router_mass_coverage_mean": float(mass.mean().detach().cpu()),
            "router_mass_coverage_min": float(mass.min().detach().cpu()),
        }

    candidate_ids = sorted_ids[:, :candidate_k]
    pair_positions = torch.tensor(
        list(itertools.combinations(range(candidate_k), 2)),
        dtype=torch.long,
        device=hidden_states.device,
    )
    pair_ids = candidate_ids[:, pair_positions]
    token_ids = torch.arange(hidden_states.shape[0], device=hidden_states.device)[:, None, None]
    pair_outputs = expert_outputs[token_ids, pair_ids]
    pair_weights = probabilities.gather(1, candidate_ids)[:, pair_positions]
    if normalize_topk_prob:
        pair_weights = pair_weights / pair_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    pair_reconstructions = (pair_outputs.float() * pair_weights[..., None].float()).sum(dim=2)
    pair_mse = (pair_reconstructions - reference[:, None, :].float()).pow(2).mean(dim=-1)
    best_positions = pair_mse.argmin(dim=1)
    best_output = pair_reconstructions[torch.arange(hidden_states.shape[0], device=hidden_states.device), best_positions]
    best_pairs = pair_ids[torch.arange(hidden_states.shape[0], device=hidden_states.device), best_positions]
    histogram: Dict[str, int] = {}
    for pair in best_pairs.detach().cpu().tolist():
        key = f"{min(pair)},{max(pair)}"
        histogram[key] = histogram.get(key, 0) + 1
    oracle_mse = float(F.mse_loss(best_output, reference.float()).detach().cpu())
    router_top2_mse = float(reconstructions["top_2"]["mse"])
    return {
        "diagnostic_scope": "development_only",
        "native_reference": "captured_native_top8" if native_top8_output is not None else "manual_native_top8",
        "normalize_topk_prob": bool(normalize_topk_prob),
        "token_count": int(hidden_states.shape[0]),
        "num_experts": num_experts,
        "reconstructions": reconstructions,
        "native_top8_equivalence_mse": float(F.mse_loss(reconstructed_top8.float(), reference.float()).detach().cpu()),
        "oracle_top2": {
            "diagnostic_only": True,
            "inference_path": False,
            "candidate_scope": f"per_token_native_top_{candidate_k}_pairs",
            "mse": oracle_mse,
            "cosine": float(F.cosine_similarity(best_output, reference.float(), dim=-1).mean().detach().cpu()),
            "router_selected_top2_mse": router_top2_mse,
            "mse_improvement_over_router_top2": float(router_top2_mse - oracle_mse),
            "selected_pair_histogram": dict(sorted(histogram.items())),
        },
    }


def build_esft_selection(
    rows: Sequence[Dict[str, Any]],
    selected_experts_per_layer: int,
) -> Dict[str, Any]:
    """Build deterministic ESFT-Gate and ESFT-Token prefix-only selections."""
    validate_development_evidence([], rows)
    selected_count = int(selected_experts_per_layer)
    if selected_count <= 0:
        raise ValueError("selected_experts_per_layer must be positive")
    layers: Dict[int, Dict[str, Any]] = {}
    expected_total = 0
    observed_total = 0
    prefix_tokens_across_layers = 0
    for index, row in enumerate(rows):
        split = str(row.get("split", "")).lower()
        modality = str(row.get("modality", "")).lower()
        if split not in DEVELOPMENT_SPLITS:
            raise DevelopmentEvidenceError(f"routing row {index} is not train/dev")
        if modality not in {"image_prefix", "audio_prefix"}:
            raise DevelopmentEvidenceError(
                f"routing row {index} must be image_prefix or audio_prefix, got {modality!r}"
            )
        layer = int(row["layer"])
        top_k = int(row["top_k"])
        token_count = int(row["token_count"])
        token_counts = torch.as_tensor(row["attempted_expert_counts"], dtype=torch.long)
        gate_scores = torch.as_tensor(row["gate_score_sums"], dtype=torch.float64)
        if token_counts.ndim != 1 or gate_scores.shape != token_counts.shape:
            raise ValueError(f"routing row {index} has inconsistent expert vectors")
        if bool((token_counts < 0).any()) or not bool(torch.isfinite(gate_scores).all()) or bool((gate_scores < 0).any()):
            raise ValueError(f"routing row {index} has invalid counts or gate scores")
        expected = token_count * top_k
        observed = int(token_counts.sum().item())
        if observed != expected:
            raise ValueError(
                f"routing row {index} violates tokens x K conservation: expected {expected}, observed {observed}"
            )
        state = layers.setdefault(
            layer,
            {
                "token_counts": torch.zeros_like(token_counts),
                "gate_scores": torch.zeros_like(gate_scores),
                "prefix_tokens": 0,
                "assignments": 0,
                "splits": set(),
                "modalities": set(),
            },
        )
        if state["token_counts"].shape != token_counts.shape:
            raise ValueError(f"layer {layer} changes num_experts across routing rows")
        state["token_counts"] += token_counts
        state["gate_scores"] += gate_scores
        state["prefix_tokens"] += token_count
        state["assignments"] += observed
        state["splits"].add(split)
        state["modalities"].add(modality)
        expected_total += expected
        observed_total += observed
        prefix_tokens_across_layers += token_count

    if not layers:
        raise ValueError("no prefix routing rows were provided")
    methods: Dict[str, Dict[str, Any]] = {"ESFT-Gate": {}, "ESFT-Token": {}}
    for layer, state in sorted(layers.items()):
        num_experts = int(state["token_counts"].numel())
        if selected_count > num_experts:
            raise ValueError("selected_experts_per_layer exceeds num_experts")
        gate_rank = sorted(range(num_experts), key=lambda idx: (-float(state["gate_scores"][idx]), idx))
        token_rank = sorted(range(num_experts), key=lambda idx: (-int(state["token_counts"][idx]), idx))
        expert_rows = [
            {
                "expert_id": expert_id,
                "gate_score_sum": float(state["gate_scores"][expert_id]),
                "gate_score_per_prefix_token": float(state["gate_scores"][expert_id] / max(1, state["prefix_tokens"])),
                "token_count": int(state["token_counts"][expert_id]),
                "token_frequency": float(state["token_counts"][expert_id] / max(1, state["assignments"])),
            }
            for expert_id in range(num_experts)
        ]
        common = {
            "prefix_tokens": int(state["prefix_tokens"]),
            "assignments": int(state["assignments"]),
            "splits": sorted(state["splits"]),
            "modalities": sorted(state["modalities"]),
            "expert_scores": expert_rows,
        }
        methods["ESFT-Gate"][str(layer)] = {**common, "selected_expert_ids": gate_rank[:selected_count]}
        methods["ESFT-Token"][str(layer)] = {**common, "selected_expert_ids": token_rank[:selected_count]}
    return {
        "schema_version": 1,
        "selection_scope": "development_train_dev_image_audio_prefix_only",
        "deterministic_tie_break": "ascending_expert_id",
        "selected_experts_per_layer": selected_count,
        "methods": methods,
        "routing_accounting": {
            "denominator": "prefix_token_expert_assignments_across_layers",
            "prefix_tokens_across_layers": prefix_tokens_across_layers,
            "layer_count": len(layers),
            "expected_assignments_tokens_x_layers_x_k": expected_total,
            "observed_assignments": observed_total,
            "conservation_ok": expected_total == observed_total,
        },
    }


def selected_expert_update_capability(mode: str) -> Dict[str, Any]:
    normalized = str(mode).strip().lower()
    if normalized == "lora":
        raise RuntimeError(
            "expert LoRA is only a declared memory fallback/capacity ablation; "
            "this repository has no wired expert-LoRA training path"
        )
    if normalized != "full":
        raise ValueError("expert update mode must be 'full' or 'lora'")
    return {
        "mode": "selected_full_experts",
        "lora_fallback_declared": True,
        "lora_supported": False,
        "lora_status": "unavailable_fail_closed",
        "lora_scope": "memory_fallback_capacity_ablation_only",
    }


def configure_selected_full_expert_training(
    model: nn.Module,
    selected_expert_ids: Dict[int | str, Sequence[int]],
    expert_learning_rate: float,
    anchor_coefficient: float,
) -> Tuple[torch.optim.Optimizer, Dict[int, Dict[str, Any]], List[Any], Dict[str, Any]]:
    """Freeze the model and expose only selected packed expert rows to training."""
    if not 0.0 < float(expert_learning_rate) <= 1e-4:
        raise ValueError("expert_learning_rate must be in the low-LR range (0, 1e-4]")
    if float(anchor_coefficient) < 0.0:
        raise ValueError("anchor_coefficient must be non-negative")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    optimizer_parameters: List[nn.Parameter] = []
    anchors: Dict[int, Dict[str, Any]] = {}
    hook_handles: List[Any] = []
    selected_metadata: Dict[str, List[int]] = {}
    for layer_idx, mlp in iter_olmoe_mlp_layers(model):
        raw_ids = selected_expert_ids.get(layer_idx, selected_expert_ids.get(str(layer_idx), ()))
        expert_ids = sorted({int(value) for value in raw_ids})
        num_experts = int(mlp.experts.num_experts)
        if not expert_ids:
            raise ValueError(f"layer {layer_idx} has no selected experts")
        if expert_ids[0] < 0 or expert_ids[-1] >= num_experts:
            raise ValueError(f"layer {layer_idx} has out-of-range selected expert IDs")
        anchor_entry: Dict[str, Any] = {"expert_ids": expert_ids}
        for parameter_name in ("gate_up_proj", "down_proj"):
            parameter = getattr(mlp.experts, parameter_name, None)
            if not isinstance(parameter, nn.Parameter) or parameter.shape[0] != num_experts:
                raise RuntimeError(
                    f"layer {layer_idx} does not expose packed full-expert parameter {parameter_name}"
                )
            parameter.requires_grad_(True)
            mask = torch.zeros_like(parameter, dtype=torch.bool)
            mask[expert_ids] = True
            hook_handles.append(parameter.register_hook(lambda gradient, current_mask=mask: gradient.masked_fill(~current_mask, 0)))
            optimizer_parameters.append(parameter)
            anchor_entry[parameter_name] = parameter.detach()[expert_ids].cpu().clone()
        anchors[layer_idx] = anchor_entry
        selected_metadata[str(layer_idx)] = expert_ids
    if set(selected_metadata) != {str(layer_idx) for layer_idx, _ in iter_olmoe_mlp_layers(model)}:
        raise ValueError("selected expert IDs must cover every OLMoE layer")
    optimizer = torch.optim.AdamW(
        [{"params": optimizer_parameters, "lr": float(expert_learning_rate), "weight_decay": 0.0}],
    )
    metadata = {
        **selected_expert_update_capability("full"),
        "selected_expert_ids_by_layer": selected_metadata,
        "non_selected_experts_frozen": True,
        "packed_parameter_gradient_masking": True,
        "optimizer": "AdamW",
        "expert_learning_rate": float(expert_learning_rate),
        "weight_decay": 0.0,
        "weight_anchor_coefficient": float(anchor_coefficient),
        "weight_anchor_reference": "pre_training_selected_full_expert_weights",
        "trainable_expert_tensors": ["gate_up_proj", "down_proj"],
    }
    return optimizer, anchors, hook_handles, metadata


def selected_expert_anchor_loss(
    model: nn.Module,
    anchors: Dict[int, Dict[str, Any]],
    coefficient: float,
) -> torch.Tensor:
    """Return the selected full-expert L2 anchor term for a training objective."""
    if float(coefficient) < 0.0:
        raise ValueError("anchor coefficient must be non-negative")
    terms: List[torch.Tensor] = []
    layers = dict(iter_olmoe_mlp_layers(model))
    for layer_idx, entry in anchors.items():
        if layer_idx not in layers:
            raise ValueError(f"anchor references missing layer {layer_idx}")
        expert_ids = torch.tensor(entry["expert_ids"], dtype=torch.long)
        experts = layers[layer_idx].experts
        for parameter_name in ("gate_up_proj", "down_proj"):
            parameter = getattr(experts, parameter_name)
            indices = expert_ids.to(parameter.device)
            reference = torch.as_tensor(entry[parameter_name], device=parameter.device, dtype=parameter.dtype)
            current = parameter.index_select(0, indices)
            terms.append((current - reference).pow(2).mean().float())
    if not terms:
        raise ValueError("anchor state is empty")
    return float(coefficient) * torch.stack(terms).mean()


def capacity_experts_forward(experts: nn.Module, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor, capacity_factor: float) -> torch.Tensor:
    num_tokens, _ = hidden_states.shape
    num_experts = int(experts.num_experts)
    top_k = int(top_k_index.shape[-1])
    capacity = max(1, int(math.ceil(capacity_factor * num_tokens * top_k / num_experts)))
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor[0].item())
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue
        weights = top_k_weights[token_idx, top_k_pos].float()
        order = torch.argsort(weights, descending=True)[:capacity]
        top_k_pos = top_k_pos[order]
        token_idx = token_idx[order]
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = experts.act_fn(gate) * up
        current_hidden_states = F.linear(current_hidden_states, experts.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
    return final_hidden_states


def apply_expert_dropout_to_topk(top_k_weights: torch.Tensor, dropout_prob: float, training: bool) -> Tuple[torch.Tensor, float]:
    """Drop selected expert assignments during training while keeping one route per token."""
    prob = float(dropout_prob)
    if not training or prob <= 0.0 or top_k_weights.shape[-1] <= 1:
        return top_k_weights, 0.0
    keep = torch.rand_like(top_k_weights.float()) >= prob
    best = top_k_weights.float().argmax(dim=-1, keepdim=True)
    keep.scatter_(1, best, True)
    dropped = (~keep).float().mean().detach().cpu().item()
    return top_k_weights * keep.to(dtype=top_k_weights.dtype), float(dropped)


def mask_top_k_weights_by_expert_ids(top_k_index: torch.Tensor, top_k_weights: torch.Tensor, expert_ids: Sequence[int] | None) -> torch.Tensor:
    """Zero selected expert assignments for causal intervention analysis."""
    ids = [int(idx) for idx in (expert_ids or [])]
    if not ids:
        return top_k_weights
    masked = torch.zeros_like(top_k_weights, dtype=torch.bool)
    for expert_id in ids:
        masked |= top_k_index == int(expert_id)
    if not bool(masked.any()):
        return top_k_weights
    return top_k_weights.masked_fill(masked, 0)


def _masked_expert_counts(
    top_k_index: torch.Tensor,
    assignment_mask: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    selected = top_k_index[assignment_mask]
    if selected.numel() == 0:
        return torch.zeros(int(num_experts), dtype=torch.long, device=top_k_index.device)
    return torch.bincount(selected.reshape(-1), minlength=int(num_experts))


def _masked_expert_weight_sums(
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    assignment_mask: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    sums = torch.zeros(int(num_experts), dtype=torch.float64, device=top_k_index.device)
    selected_ids = top_k_index[assignment_mask].reshape(-1)
    selected_weights = top_k_weights[assignment_mask].reshape(-1).to(dtype=torch.float64)
    if selected_ids.numel():
        sums.scatter_add_(0, selected_ids, selected_weights)
    return sums


def capacity_mask_with_accounting(
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
    capacity_factor: float,
    randomize_ties: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Apply capacity and return integer dispatch accounting.

    The attempted mask is defined immediately before capacity enforcement. Exact
    gate-weight ties use seeded RNG jitter, avoiding token-order bias while
    remaining reproducible after torch.manual_seed.
    """
    num_tokens = int(top_k_index.shape[0])
    top_k = int(top_k_index.shape[-1])
    capacity = max(1, int(math.ceil(float(capacity_factor) * num_tokens * top_k / int(num_experts))))
    attempted_mask = top_k_weights != 0
    masked = top_k_weights.clone()
    if capacity_factor > 0:
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=int(num_experts)).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx_tensor in expert_hit:
                expert_idx = int(expert_idx_tensor[0].item())
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                active = attempted_mask[token_idx, top_k_pos]
                top_k_pos = top_k_pos[active]
                token_idx = token_idx[active]
                if token_idx.numel() <= capacity:
                    continue
                weights = top_k_weights[token_idx, top_k_pos].float()
                if randomize_ties:
                    scale = weights.detach().abs().max().clamp_min(1.0)
                    weights = weights + torch.rand_like(weights) * torch.finfo(weights.dtype).eps * scale
                keep_order = torch.argsort(weights, descending=True)
                drop_order = keep_order[capacity:]
                masked[token_idx[drop_order], top_k_pos[drop_order]] = 0
    accepted_mask = masked != 0
    dropped_mask = attempted_mask & ~accepted_mask
    attempted_counts = _masked_expert_counts(top_k_index, attempted_mask, num_experts)
    accepted_counts = _masked_expert_counts(top_k_index, accepted_mask, num_experts)
    dropped_counts = _masked_expert_counts(top_k_index, dropped_mask, num_experts)
    accounting = {
        "num_tokens": num_tokens,
        "top_k": top_k,
        "capacity_per_expert": capacity,
        "attempted_mask": attempted_mask.detach(),
        "accepted_mask": accepted_mask.detach(),
        "dropped_mask": dropped_mask.detach(),
        "attempted_expert_counts": attempted_counts.detach(),
        "accepted_expert_counts": accepted_counts.detach(),
        "dropped_expert_counts": dropped_counts.detach(),
        "attempted_assignments": int(attempted_mask.sum().item()),
        "accepted_assignments": int(accepted_mask.sum().item()),
        "dropped_assignments": int(dropped_mask.sum().item()),
        "conservation_ok": bool(torch.equal(attempted_counts, accepted_counts + dropped_counts)),
        "capacity_compliant": bool(int(accepted_counts.max().item()) <= capacity),
        "tie_break": "seeded_random_epsilon" if randomize_ties else "stable_order",
    }
    return masked, accounting


def mask_top_k_weights_by_capacity(top_k_index: torch.Tensor, top_k_weights: torch.Tensor, num_experts: int, capacity_factor: float) -> torch.Tensor:
    """Drop lowest-weight token-expert assignments over per-expert capacity."""
    masked, _ = capacity_mask_with_accounting(
        top_k_index,
        top_k_weights,
        num_experts,
        capacity_factor,
    )
    return masked


def apply_dynamic_expert_bias_to_topk(
    router_logits: torch.Tensor,
    top_k: int,
    expert_bias: torch.Tensor,
    normalize_topk_prob: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply non-gradient expert bias before Top-k selection."""
    adjusted = router_logits.float() + expert_bias.to(device=router_logits.device, dtype=torch.float32)
    probs = torch.softmax(adjusted, dim=-1)
    top_k_weights, top_k_index = torch.topk(probs, k=int(top_k), dim=-1)
    if normalize_topk_prob:
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)
    return top_k_weights.to(dtype=router_logits.dtype), top_k_index


def iter_olmoe_mlp_layers(model: nn.Module):
    """Yield layer index and OLMoE MLP modules when present."""
    for idx, layer in enumerate(getattr(getattr(model, "model", None), "layers", [])):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            yield idx, mlp


def dynamic_expert_bias_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return checkpointable dynamic expert-bias buffers."""
    state: Dict[str, torch.Tensor] = {}
    for idx, mlp in iter_olmoe_mlp_layers(model):
        bias = getattr(mlp, "expert_bias", None)
        if bias is not None and bool(getattr(mlp, "dynamic_expert_bias_enabled", False)):
            state[f"layer_{idx}"] = bias.detach().float().cpu()
    return state


def load_dynamic_expert_bias_state(model: nn.Module, state: Dict[str, Any] | None) -> int:
    """Load dynamic expert-bias buffers into patched OLMoE MLP layers."""
    if not state:
        return 0
    loaded = 0
    for idx, mlp in iter_olmoe_mlp_layers(model):
        key = f"layer_{idx}"
        if key not in state or not hasattr(mlp, "expert_bias"):
            continue
        value = torch.as_tensor(state[key], dtype=torch.float32, device=mlp.expert_bias.device)
        if value.shape != mlp.expert_bias.shape:
            raise RuntimeError(f"dynamic expert bias shape mismatch for {key}: {tuple(value.shape)} vs {tuple(mlp.expert_bias.shape)}")
        mlp.expert_bias.copy_(value)
        mlp.dynamic_expert_bias_enabled = True
        loaded += 1
    return loaded


def dynamic_expert_bias_metrics(model: nn.Module) -> Dict[str, float]:
    """Summarize dynamic expert-bias magnitudes for logs and artifacts."""
    values: List[torch.Tensor] = []
    for _, mlp in iter_olmoe_mlp_layers(model):
        bias = getattr(mlp, "expert_bias", None)
        if bias is not None and bool(getattr(mlp, "dynamic_expert_bias_enabled", False)):
            values.append(bias.detach().float().cpu())
    if not values:
        return {
            "dynamic_expert_bias_enabled": 0.0,
            "dynamic_expert_bias_layers": 0.0,
            "dynamic_expert_bias_abs_mean": 0.0,
            "dynamic_expert_bias_abs_max": 0.0,
            "dynamic_expert_bias_std": 0.0,
        }
    cat = torch.cat([value.reshape(-1) for value in values])
    return {
        "dynamic_expert_bias_enabled": 1.0,
        "dynamic_expert_bias_layers": float(len(values)),
        "dynamic_expert_bias_abs_mean": float(cat.abs().mean().item()),
        "dynamic_expert_bias_abs_max": float(cat.abs().max().item()),
        "dynamic_expert_bias_std": float(cat.std(unbiased=False).item()),
    }


def patch_moe_blocks(
    model: nn.Module,
    gamma: Sequence[float] | None,
    capacity_factor: float,
    expert_dropout_prob: float = 0.0,
    dynamic_expert_bias: bool = False,
) -> None:
    """Apply capacity masking while preserving HuggingFace OLMoE expert dispatch."""
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        block = layer.mlp
        scale = float(gamma[layer_idx]) if gamma is not None else 1.0
        block.register_buffer("gamma_scale", torch.tensor(scale, dtype=torch.float32), persistent=False)
        block.capacity_factor = float(capacity_factor)
        block.expert_dropout_prob = float(expert_dropout_prob)
        block.dynamic_expert_bias_enabled = bool(dynamic_expert_bias)
        block.dynamic_expert_bias_norm_topk = bool(getattr(model.config, "norm_topk_prob", True))
        block.last_expert_dropout_ratio = 0.0
        block.register_buffer(
            "expert_bias",
            torch.zeros(int(block.experts.num_experts), dtype=torch.float32),
            persistent=True,
        )

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            flat = hidden_states.reshape(-1, hidden_dim)
            _router_logits, top_k_weights, top_k_index = self.gate(flat)
            if bool(getattr(self, "dynamic_expert_bias_enabled", False)):
                top_k_weights, top_k_index = apply_dynamic_expert_bias_to_topk(
                    _router_logits,
                    int(top_k_index.shape[-1]),
                    self.expert_bias,
                    bool(getattr(self, "dynamic_expert_bias_norm_topk", True)),
                )
            raw_attempted_mask = torch.ones_like(top_k_weights, dtype=torch.bool)
            raw_router_weights = top_k_weights.detach()
            with torch.no_grad():
                self.last_dynamic_expert_assigned_counts = torch.bincount(
                    top_k_index.reshape(-1).detach(),
                    minlength=int(self.experts.num_experts),
                )
            top_k_weights, dropout_ratio = apply_expert_dropout_to_topk(
                top_k_weights,
                float(getattr(self, "expert_dropout_prob", 0.0)),
                bool(self.training),
            )
            self.last_expert_dropout_ratio = float(dropout_ratio)
            top_k_weights = mask_top_k_weights_by_expert_ids(
                top_k_index,
                top_k_weights,
                getattr(self, "masked_expert_ids", None),
            )
            pre_capacity_mask = top_k_weights != 0
            top_k_weights, dispatch = capacity_mask_with_accounting(
                top_k_index,
                top_k_weights,
                int(self.experts.num_experts),
                float(self.capacity_factor),
            )
            with torch.no_grad():
                pre_capacity_dropped_mask = raw_attempted_mask & ~pre_capacity_mask
                self.last_dispatch_top_k_index = top_k_index.detach()
                self.last_dispatch_router_weights = raw_router_weights
                self.last_dispatch_attempted_mask = raw_attempted_mask.detach()
                self.last_dispatch_pre_capacity_mask = pre_capacity_mask.detach()
                self.last_dispatch_accepted_mask = dispatch["accepted_mask"]
                self.last_dispatch_capacity_dropped_mask = dispatch["dropped_mask"]
                self.last_dispatch_pre_capacity_dropped_mask = pre_capacity_dropped_mask.detach()
                self.last_dispatch_attempted_expert_counts = _masked_expert_counts(
                    top_k_index, raw_attempted_mask, int(self.experts.num_experts)
                ).detach()
                self.last_dispatch_pre_capacity_expert_counts = _masked_expert_counts(
                    top_k_index, pre_capacity_mask, int(self.experts.num_experts)
                ).detach()
                self.last_dispatch_accepted_expert_counts = dispatch["accepted_expert_counts"]
                self.last_dispatch_capacity_dropped_expert_counts = dispatch["dropped_expert_counts"]
                self.last_dispatch_pre_capacity_dropped_expert_counts = _masked_expert_counts(
                    top_k_index, pre_capacity_dropped_mask, int(self.experts.num_experts)
                ).detach()
                self.last_dispatch_num_tokens = int(flat.shape[0])
                self.last_dispatch_top_k = int(top_k_index.shape[-1])
                self.last_dispatch_capacity = int(dispatch["capacity_per_expert"])
                self.last_dispatch_tie_break = str(dispatch["tie_break"])
            final_hidden_states = self.experts(flat, top_k_index, top_k_weights).reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states * self.gamma_scale.to(final_hidden_states.device, final_hidden_states.dtype)

        block.forward = types.MethodType(forward, block)


def patch_moe_output_scale(model: nn.Module, gamma: Sequence[float] | None) -> None:
    """Apply layerwise calibration without replacing native expert dispatch."""
    if gamma is None:
        return
    for layer_idx, layer in enumerate(model.model.layers):
        block = layer.mlp
        scale = float(gamma[layer_idx])
        block.register_buffer("gamma_scale", torch.tensor(scale, dtype=torch.float32), persistent=False)
        original_forward = block.forward

        def forward(self, hidden_states: torch.Tensor, *args, _original_forward=original_forward, **kwargs):
            out = _original_forward(hidden_states, *args, **kwargs)
            scale_tensor = self.gamma_scale.to(hidden_states.device, hidden_states.dtype)
            if isinstance(out, tuple):
                return (out[0] * scale_tensor, *out[1:])
            return out * scale_tensor

        block.forward = types.MethodType(forward, block)


def set_olmoe_runtime_routing(model: nn.Module, top_k: int, aux_coef: float) -> Dict[str, object]:
    """Set OLMoE routing knobs after native weight loading.

    Passing a modified config object into ``from_pretrained`` corrupted the
    Top-8 teacher sanity check in Run:AI diagnostics. Keep checkpoint loading
    native, then update the config and layer-local Top-k attributes in place.
    """
    top_k = int(top_k)
    changed: List[str] = []
    native_top_k = int(getattr(model.config, "num_experts_per_tok", top_k))
    if hasattr(model.config, "num_experts_per_tok"):
        model.config.num_experts_per_tok = top_k
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True
    if hasattr(model.config, "router_aux_loss_coef"):
        model.config.router_aux_loss_coef = float(aux_coef)

    # OLMoE copies these config values onto the CausalLM instance at init.
    # Updating config/layer gates alone leaves the auxiliary objective at the
    # checkpoint's native Top-k and coefficient.
    for object_name, runtime_object in (
        ("model", model),
        ("model.model", getattr(model, "model", None)),
    ):
        if runtime_object is None:
            continue
        if hasattr(runtime_object, "router_aux_loss_coef"):
            previous = getattr(runtime_object, "router_aux_loss_coef")
            setattr(runtime_object, "router_aux_loss_coef", float(aux_coef))
            changed.append(f"{object_name}.router_aux_loss_coef:{previous}->{float(aux_coef)}")
        if hasattr(runtime_object, "num_experts_per_tok"):
            previous = getattr(runtime_object, "num_experts_per_tok")
            setattr(runtime_object, "num_experts_per_tok", top_k)
            changed.append(f"{object_name}.num_experts_per_tok:{previous}->{top_k}")

    for layer_idx, layer in enumerate(getattr(model.model, "layers", [])):
        mlp = getattr(layer, "mlp", None)
        for obj_name, obj in [("mlp", mlp), ("mlp.gate", getattr(mlp, "gate", None) if mlp is not None else None)]:
            if obj is None:
                continue
            for attr in ("top_k", "num_experts_per_tok", "k"):
                if hasattr(obj, attr):
                    current = getattr(obj, attr)
                    if isinstance(current, int):
                        setattr(obj, attr, top_k)
                        changed.append(f"layer_{layer_idx}.{obj_name}.{attr}:{current}->{top_k}")
    return {
        "native_top_k": native_top_k,
        "runtime_top_k": top_k,
        "runtime_changed_attrs": changed[:16],
        "runtime_changed_attr_count": len(changed),
        "norm_topk_prob": getattr(model.config, "norm_topk_prob", None),
    }


def load_model(
    base_model: str,
    top_k: int,
    aux_coef: float,
    gamma: Sequence[float] | None,
    capacity_factor: float,
    apply_capacity_patch: bool | None = None,
    expert_dropout_prob: float = 0.0,
    dynamic_expert_bias: bool = False,
    pre_routing_identity_fn: Callable[[nn.Module], Any] | None = None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype, dtype_name = choose_dtype()
    tokenizer = load_pretrained(AutoTokenizer, base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_pretrained(
        AutoModelForCausalLM,
        base_model,
        torch_dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    pre_routing_model_identity = (
        pre_routing_identity_fn(model)
        if pre_routing_identity_fn is not None
        else None
    )
    routing_meta = set_olmoe_runtime_routing(model, top_k=top_k, aux_coef=aux_coef)
    if apply_capacity_patch is None:
        apply_capacity_patch = int(top_k) != int(routing_meta.get("native_top_k", top_k))
    if apply_capacity_patch:
        patch_moe_blocks(
            model,
            gamma=gamma,
            capacity_factor=capacity_factor,
            expert_dropout_prob=expert_dropout_prob,
            dynamic_expert_bias=dynamic_expert_bias,
        )
        dispatch = "native_hf_capacity_mask"
    else:
        patch_moe_output_scale(model, gamma=gamma)
        dispatch = "native_hf"
    meta = {
        "base_model": base_model,
        "dtype": dtype_name,
        "top_k": top_k,
        "aux_coef": aux_coef,
        "capacity_factor": capacity_factor,
        "dispatch": dispatch,
        "capacity_enforced": bool(apply_capacity_patch),
        "gamma_applied": gamma is not None,
        "expert_dropout_prob": float(expert_dropout_prob),
        "dynamic_expert_bias_enabled": bool(dynamic_expert_bias),
        "pre_routing_model_identity": pre_routing_model_identity,
        **routing_meta,
    }
    return model, tokenizer, meta


def capture_router_dispatch(model: nn.Module) -> List[Dict[str, Any]]:
    """Snapshot the latest patched dispatch before another forward overwrites it."""
    layers: List[Dict[str, Any]] = []
    for layer_idx, mlp in iter_olmoe_mlp_layers(model):
        top_k_index = getattr(mlp, "last_dispatch_top_k_index", None)
        if top_k_index is None:
            continue
        row: Dict[str, Any] = {
            "layer": int(layer_idx),
            "top_k_index": top_k_index.detach().cpu(),
            "router_weights": mlp.last_dispatch_router_weights.detach().cpu(),
            "attempted_mask": mlp.last_dispatch_attempted_mask.detach().cpu(),
            "pre_capacity_mask": mlp.last_dispatch_pre_capacity_mask.detach().cpu(),
            "accepted_mask": mlp.last_dispatch_accepted_mask.detach().cpu(),
            "capacity_dropped_mask": mlp.last_dispatch_capacity_dropped_mask.detach().cpu(),
            "pre_capacity_dropped_mask": mlp.last_dispatch_pre_capacity_dropped_mask.detach().cpu(),
            "attempted_expert_counts": mlp.last_dispatch_attempted_expert_counts.detach().cpu(),
            "pre_capacity_expert_counts": mlp.last_dispatch_pre_capacity_expert_counts.detach().cpu(),
            "accepted_expert_counts": mlp.last_dispatch_accepted_expert_counts.detach().cpu(),
            "capacity_dropped_expert_counts": mlp.last_dispatch_capacity_dropped_expert_counts.detach().cpu(),
            "pre_capacity_dropped_expert_counts": mlp.last_dispatch_pre_capacity_dropped_expert_counts.detach().cpu(),
            "num_tokens": int(mlp.last_dispatch_num_tokens),
            "top_k": int(mlp.last_dispatch_top_k),
            "capacity_per_expert": int(mlp.last_dispatch_capacity),
            "tie_break": str(mlp.last_dispatch_tie_break),
        }
        layers.append(row)
    return layers


def _hard_load_stats(counts: torch.Tensor) -> Dict[str, float]:
    values = counts.detach().float().cpu()
    total = float(values.sum().item())
    active = int((values > 0).sum().item())
    mean = float(values.mean().item()) if values.numel() else 0.0
    if total <= 0.0:
        return {
            "hard_assignment_entropy": 0.0,
            "effective_experts": 0.0,
            "active_experts": 0.0,
            "inactive_expert_ratio": 1.0,
            "load_cv": 0.0,
            "load_gini": 0.0,
            "max_to_mean_load": 0.0,
        }
    probs = values / total
    entropy = float((-(probs[probs > 0] * probs[probs > 0].log())).sum().item())
    cv = float(values.std(unbiased=False).item() / max(mean, 1e-12))
    sorted_values = torch.sort(values).values
    n = int(sorted_values.numel())
    positions = torch.arange(1, n + 1, dtype=torch.float32)
    gini = float(((2.0 * positions - n - 1.0) * sorted_values).sum().item() / max(n * total, 1e-12))
    return {
        "hard_assignment_entropy": entropy,
        "effective_experts": float(math.exp(entropy)),
        "active_experts": float(active),
        "inactive_expert_ratio": float(1.0 - active / max(1, n)),
        "load_cv": cv,
        "load_gini": gini,
        "max_to_mean_load": float(values.max().item() / max(mean, 1e-12)),
    }


def router_metrics(
    outputs,
    top_k: int,
    num_experts: int,
    capacity_factor: float,
    model: nn.Module | None = None,
    dispatch_snapshot: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, object]:
    logits = getattr(outputs, "router_logits", None)
    tensors = []
    if logits is not None:
        tensors = list(logits) if not torch.is_tensor(logits) else [logits]
        tensors = [tensor for tensor in tensors if torch.is_tensor(tensor)]
    entropies: List[float] = []
    for tensor in tensors:
        probs = torch.softmax(tensor.float(), dim=-1)
        entropies.append(float((-(probs * (probs + 1e-9).log()).sum(dim=-1)).mean().detach().cpu()))

    snapshot = list(dispatch_snapshot or capture_router_dispatch(model) if model is not None else dispatch_snapshot or [])
    if snapshot:
        layer_rows: List[Dict[str, Any]] = []
        accepted_total = torch.zeros(int(num_experts), dtype=torch.long)
        attempted_total = torch.zeros(int(num_experts), dtype=torch.long)
        dropped_total = torch.zeros(int(num_experts), dtype=torch.long)
        overflow_ratios: List[float] = []
        inactive_ratios: List[float] = []
        for layer in snapshot:
            attempted = torch.as_tensor(layer["attempted_expert_counts"], dtype=torch.long)
            pre_capacity = torch.as_tensor(layer["pre_capacity_expert_counts"], dtype=torch.long)
            accepted = torch.as_tensor(layer["accepted_expert_counts"], dtype=torch.long)
            capacity_dropped = torch.as_tensor(layer["capacity_dropped_expert_counts"], dtype=torch.long)
            pre_capacity_dropped = torch.as_tensor(layer["pre_capacity_dropped_expert_counts"], dtype=torch.long)
            dropped = capacity_dropped + pre_capacity_dropped
            attempted_n = int(attempted.sum().item())
            pre_capacity_n = int(pre_capacity.sum().item())
            accepted_n = int(accepted.sum().item())
            dropped_n = int(dropped.sum().item())
            capacity_dropped_n = int(capacity_dropped.sum().item())
            expected_n = int(layer["num_tokens"]) * int(layer["top_k"])
            conservation = bool(torch.equal(attempted, accepted + dropped))
            capacity = int(layer["capacity_per_expert"])
            compliant = bool(int(accepted.max().item()) <= capacity)
            hard_stats = _hard_load_stats(accepted)
            layer_rows.append({
                "layer": int(layer["layer"]),
                "token_count": int(layer["num_tokens"]),
                "top_k": int(layer["top_k"]),
                "capacity_per_expert": capacity,
                "expected_assignments_tokens_x_k": expected_n,
                "token_k_conservation_ok": attempted_n == expected_n,
                "attempted_assignments": attempted_n,
                "pre_capacity_assignments": pre_capacity_n,
                "accepted_assignments": accepted_n,
                "capacity_dropped_assignments": capacity_dropped_n,
                "pre_capacity_dropped_assignments": int(pre_capacity_dropped.sum().item()),
                "dropped_assignments": dropped_n,
                "drop_ratio": float(dropped_n / max(1, attempted_n)),
                "capacity_overflow_ratio": float(capacity_dropped_n / max(1, pre_capacity_n)),
                "conservation_ok": conservation,
                "capacity_compliant": compliant,
                "tie_break": str(layer["tie_break"]),
                "attempted_expert_counts": attempted.tolist(),
                "accepted_expert_counts": accepted.tolist(),
                "dropped_expert_counts": dropped.tolist(),
                **hard_stats,
            })
            attempted_total += attempted
            accepted_total += accepted
            dropped_total += dropped
            overflow_ratios.append(float(capacity_dropped_n / max(1, pre_capacity_n)))
            inactive_ratios.append(float(hard_stats["inactive_expert_ratio"]))
        attempted_n = int(attempted_total.sum().item())
        accepted_n = int(accepted_total.sum().item())
        dropped_n = int(dropped_total.sum().item())
        expected_n = int(sum(int(row["expected_assignments_tokens_x_k"]) for row in layer_rows))
        return {
            "router_layers": len(snapshot),
            "routing_accounting_source": "patched_dispatch_masks_after_capacity",
            "routing_denominator": "token_expert_assignments_across_layers",
            "gate_entropy_mean": float(sum(entropies) / len(entropies)) if entropies else None,
            "inactive_expert_ratio_mean": float(sum(inactive_ratios) / len(inactive_ratios)),
            "capacity_overflow_ratio_mean": float(sum(overflow_ratios) / len(overflow_ratios)),
            "routing_token_count_across_layers": int(sum(int(row["token_count"]) for row in layer_rows)),
            "routing_expected_assignments_tokens_x_layers_x_k": expected_n,
            "routing_token_k_conservation_ok": attempted_n == expected_n,
            "routing_attempted_assignments_total": attempted_n,
            "routing_accepted_assignments_total": accepted_n,
            "routing_dropped_assignments_total": dropped_n,
            "routing_drop_ratio_total": float(dropped_n / max(1, attempted_n)),
            "routing_conservation_ok": bool(
                attempted_n == expected_n
                and attempted_n == accepted_n + dropped_n
                and all(bool(row["conservation_ok"]) and bool(row["token_k_conservation_ok"]) for row in layer_rows)
            ),
            "routing_capacity_compliant": bool(all(bool(row["capacity_compliant"]) for row in layer_rows)),
            "attempted_expert_counts_total": attempted_total.tolist(),
            "accepted_expert_counts_total": accepted_total.tolist(),
            "dropped_expert_counts_total": dropped_total.tolist(),
            "expert_counts_total": accepted_total.tolist(),
            "expert_counts_total_semantics": "accepted_assignments_after_capacity",
            "routing_layer_accounting": layer_rows,
            **{f"accepted_{key}": value for key, value in _hard_load_stats(accepted_total).items()},
        }

    inactive: List[float] = []
    overflow: List[float] = []
    counts_total = torch.zeros(num_experts, dtype=torch.long)
    token_count_across_layers = 0
    expected_assignments = 0
    for tensor in tensors:
        probs = torch.softmax(tensor.float(), dim=-1)
        _, selected = torch.topk(probs, top_k, dim=-1)
        counts = torch.bincount(selected.reshape(-1).detach().cpu(), minlength=num_experts)
        counts_total += counts
        token_count_across_layers += int(selected.numel() // max(1, int(top_k)))
        expected_assignments += int(selected.numel())
        inactive.append(float((counts == 0).float().mean().item()))
        capacity = max(1, int(math.ceil(capacity_factor * selected.shape[0] * top_k / num_experts)))
        overflow_assignments = torch.clamp(counts - capacity, min=0).sum().item()
        overflow.append(float(overflow_assignments / max(1, selected.numel())))
    return {
        "router_layers": len(tensors),
        "routing_accounting_source": "router_logits_attempted_only",
        "routing_denominator": "attempted_token_expert_assignments_across_layers",
        "gate_entropy_mean": float(sum(entropies) / len(entropies)) if entropies else None,
        "inactive_expert_ratio_mean": float(sum(inactive) / len(inactive)) if inactive else None,
        "capacity_overflow_ratio_mean": float(sum(overflow) / len(overflow)) if overflow else None,
        "routing_token_count_across_layers": token_count_across_layers,
        "routing_expected_assignments_tokens_x_layers_x_k": expected_assignments,
        "routing_token_k_conservation_ok": int(counts_total.sum().item()) == expected_assignments,
        "routing_attempted_assignments_total": int(counts_total.sum().item()),
        "routing_accepted_assignments_total": int(counts_total.sum().item()),
        "routing_dropped_assignments_total": 0,
        "routing_drop_ratio_total": 0.0,
        "routing_conservation_ok": int(counts_total.sum().item()) == expected_assignments,
        "attempted_expert_counts_total": counts_total.tolist(),
        "expert_counts_total": counts_total.tolist(),
        "expert_counts_total_semantics": "attempted_assignments_before_capacity",
    }


def modality_router_metrics(
    outputs,
    top_k: int,
    num_experts: int,
    batch_size: int,
    image_prefix_tokens: int,
    audio_prefix_tokens: int,
    text_tokens: int,
    model: nn.Module | None = None,
    dispatch_snapshot: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, object]:
    total_len = int(image_prefix_tokens + audio_prefix_tokens + text_tokens)
    spans = {
        "image_prefix": (0, int(image_prefix_tokens)),
        "audio_prefix": (int(image_prefix_tokens), int(image_prefix_tokens + audio_prefix_tokens)),
        "text": (int(image_prefix_tokens + audio_prefix_tokens), total_len),
    }
    accepted_counts = {name: torch.zeros(num_experts, dtype=torch.long) for name in spans}
    attempted_counts = {name: torch.zeros(num_experts, dtype=torch.long) for name in spans}
    dropped_counts = {name: torch.zeros(num_experts, dtype=torch.long) for name in spans}
    gate_score_sums = {name: torch.zeros(num_experts, dtype=torch.float64) for name in spans}
    token_counts = {name: 0 for name in spans}
    layer_rows: List[Dict[str, Any]] = []

    snapshot = list(dispatch_snapshot or capture_router_dispatch(model) if model is not None else dispatch_snapshot or [])
    for layer in snapshot:
        selected = torch.as_tensor(layer["top_k_index"], dtype=torch.long)
        if selected.shape[0] != int(batch_size) * total_len:
            continue
        selected = selected.reshape(int(batch_size), total_len, -1)
        raw_router_weights = layer.get("router_weights")
        router_weights = (
            torch.ones_like(selected, dtype=torch.float32)
            if raw_router_weights is None
            else torch.as_tensor(raw_router_weights, dtype=torch.float32).reshape_as(selected)
        )
        attempted_mask = torch.as_tensor(layer["attempted_mask"], dtype=torch.bool).reshape_as(selected)
        accepted_mask = torch.as_tensor(layer["accepted_mask"], dtype=torch.bool).reshape_as(selected)
        dropped_mask = attempted_mask & ~accepted_mask
        for name, (start, end) in spans.items():
            end = min(end, selected.shape[1])
            if start >= end:
                continue
            ids = selected[:, start:end, :]
            attempted = attempted_mask[:, start:end, :]
            accepted = accepted_mask[:, start:end, :]
            dropped = dropped_mask[:, start:end, :]
            a_counts = _masked_expert_counts(ids, attempted, num_experts).cpu()
            k_counts = _masked_expert_counts(ids, accepted, num_experts).cpu()
            d_counts = _masked_expert_counts(ids, dropped, num_experts).cpu()
            score_sums = _masked_expert_weight_sums(ids, router_weights[:, start:end, :], attempted, num_experts).cpu()
            attempted_counts[name] += a_counts
            accepted_counts[name] += k_counts
            dropped_counts[name] += d_counts
            gate_score_sums[name] += score_sums
            tokens = int(batch_size) * int(end - start)
            token_counts[name] += tokens
            layer_rows.append({
                "layer": int(layer["layer"]),
                "modality": name,
                "token_count": tokens,
                "top_k": int(selected.shape[-1]),
                "expected_assignments_tokens_x_k": int(tokens * selected.shape[-1]),
                "attempted_expert_counts": a_counts.tolist(),
                "accepted_expert_counts": k_counts.tolist(),
                "dropped_expert_counts": d_counts.tolist(),
                "gate_score_sums": score_sums.tolist(),
                "attempted_assignments": int(a_counts.sum().item()),
                "accepted_assignments": int(k_counts.sum().item()),
                "dropped_assignments": int(d_counts.sum().item()),
                "conservation_ok": bool(torch.equal(a_counts, k_counts + d_counts)),
                "capacity_per_expert": int(layer["capacity_per_expert"]),
                "capacity_compliant": bool(int(k_counts.max().item()) <= int(layer["capacity_per_expert"])),
                **_hard_load_stats(k_counts),
            })

    accounting_source = "patched_dispatch_masks_after_capacity"
    if not snapshot or not layer_rows:
        logits = getattr(outputs, "router_logits", None)
        tensors = list(logits) if logits is not None and not torch.is_tensor(logits) else ([logits] if torch.is_tensor(logits) else [])
        for layer_idx, tensor in enumerate(tensors):
            tensor = tensor.detach()
            if tensor.ndim == 2:
                if tensor.shape[0] != batch_size * total_len:
                    continue
                tensor = tensor.reshape(batch_size, total_len, num_experts)
            elif tensor.ndim != 3:
                continue
            probs = torch.softmax(tensor.float(), dim=-1)
            selected_weights, selected = torch.topk(probs, top_k, dim=-1)
            selected = selected.detach().cpu()
            selected_weights = selected_weights.detach().cpu()
            for name, (start, end) in spans.items():
                clipped = selected[:, start:min(end, selected.shape[1]), :]
                if clipped.numel() == 0:
                    continue
                values = torch.bincount(clipped.reshape(-1), minlength=num_experts)
                weights = selected_weights[:, start:min(end, selected.shape[1]), :]
                score_sums = _masked_expert_weight_sums(
                    clipped,
                    weights,
                    torch.ones_like(clipped, dtype=torch.bool),
                    num_experts,
                ).cpu()
                attempted_counts[name] += values
                accepted_counts[name] += values
                gate_score_sums[name] += score_sums
                tokens = int(batch_size) * int(max(0, min(end, selected.shape[1]) - start))
                token_counts[name] += tokens
                layer_rows.append({
                    "layer": layer_idx,
                    "modality": name,
                    "token_count": tokens,
                    "top_k": int(top_k),
                    "expected_assignments_tokens_x_k": tokens * int(top_k),
                    "attempted_expert_counts": values.tolist(),
                    "accepted_expert_counts": values.tolist(),
                    "dropped_expert_counts": [0] * int(num_experts),
                    "gate_score_sums": score_sums.tolist(),
                    "attempted_assignments": int(values.sum().item()),
                    "accepted_assignments": int(values.sum().item()),
                    "dropped_assignments": 0,
                    "conservation_ok": int(values.sum().item()) == tokens * int(top_k),
                })
        accounting_source = "router_logits_attempted_only"

    def normalize(values: torch.Tensor) -> List[float]:
        total = float(values.sum().item())
        if total <= 0.0:
            return [0.0 for _ in range(num_experts)]
        return [float(v) / total for v in values.tolist()]

    def js_divergence(left: List[float], right: List[float]) -> float:
        eps = 1e-12
        mid = [(a + b) * 0.5 for a, b in zip(left, right)]

        def kl(p: List[float], q: List[float]) -> float:
            return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q) if pi > 0.0)

        return float(0.5 * kl(left, mid) + 0.5 * kl(right, mid))

    utilization = {name: normalize(value) for name, value in accepted_counts.items()}
    conservation = {
        name: bool(torch.equal(attempted_counts[name], accepted_counts[name] + dropped_counts[name]))
        for name in spans
    }
    expected_total = int(sum(token_counts.values()) * int(top_k))
    observed_total = int(sum(int(value.sum().item()) for value in attempted_counts.values()))
    prefix_expected = int((token_counts["image_prefix"] + token_counts["audio_prefix"]) * int(top_k))
    prefix_observed = int(
        attempted_counts["image_prefix"].sum().item() + attempted_counts["audio_prefix"].sum().item()
    )
    return {
        "modality_routing_accounting_source": accounting_source,
        "modality_routing_denominator": "token_expert_assignments_across_layers",
        "modality_token_counts_across_layers": token_counts,
        "modality_expected_assignments_tokens_x_layers_x_k": expected_total,
        "modality_observed_assignments": observed_total,
        "modality_token_k_conservation_ok": expected_total == observed_total,
        "prefix_expected_assignments_tokens_x_layers_x_k": prefix_expected,
        "prefix_observed_assignments": prefix_observed,
        "prefix_routing_included": prefix_expected > 0 and prefix_expected == prefix_observed,
        "modality_attempted_expert_counts": {name: value.tolist() for name, value in attempted_counts.items()},
        "modality_expert_counts": {name: value.tolist() for name, value in accepted_counts.items()},
        "modality_dropped_expert_counts": {name: value.tolist() for name, value in dropped_counts.items()},
        "modality_gate_score_sums": {name: value.tolist() for name, value in gate_score_sums.items()},
        "modality_assignment_conservation": conservation,
        "modality_expert_utilization": utilization,
        "modality_js_image_audio": js_divergence(utilization["image_prefix"], utilization["audio_prefix"]),
        "modality_js_image_text": js_divergence(utilization["image_prefix"], utilization["text"]),
        "modality_js_audio_text": js_divergence(utilization["audio_prefix"], utilization["text"]),
        "modality_layer_accounting": layer_rows,
    }

def encode_batch(tokenizer, texts: Sequence[str], device: torch.device, max_length: int) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    encoded["labels"] = labels
    return encoded


def run_text_eval(exp_id: str, model: nn.Module, tokenizer, texts: Sequence[str], out_dir: Path, meta: Dict[str, object], max_length: int) -> Dict[str, object]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        batch = encode_batch(tokenizer, texts, device, max_length)
        outputs = model(**batch, output_router_logits=True, return_dict=True)
    row = {
        "experiment_id": exp_id,
        **meta,
        "loss": float(outputs.loss.detach().float().cpu()),
        "aux_loss": float(outputs.aux_loss.detach().float().cpu()) if outputs.aux_loss is not None else None,
        "logits_shape": list(outputs.logits.shape),
        **router_metrics(outputs, int(meta["top_k"]), int(model.config.num_experts), float(meta["capacity_factor"])),
        **cuda_metrics(),
    }
    save_json(out_dir / exp_id / "metrics.json", row)
    print(json.dumps(row, sort_keys=True))
    return row


def collect_mlp_norms(model: nn.Module, tokenizer, texts: Sequence[str], max_length: int) -> List[float]:
    device = next(model.parameters()).device
    norms: List[List[float]] = [[] for _ in model.model.layers]
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        def hook(_module, _inputs, output, idx=layer_idx):
            norms[idx].append(float(output.detach().float().norm(dim=-1).mean().cpu()))
        handles.append(layer.mlp.register_forward_hook(hook))
    with torch.no_grad():
        batch = encode_batch(tokenizer, texts, device, max_length)
        model(**batch, output_router_logits=True, return_dict=True)
    for handle in handles:
        handle.remove()
    return [sum(values) / len(values) if values else 1.0 for values in norms]


def calibrate_gamma(args: argparse.Namespace, texts: Sequence[str], out_dir: Path) -> List[float]:
    teacher, tokenizer, _ = load_model(args.base_model, 8, args.aux_coef, gamma=None, capacity_factor=args.capacity_factor)
    teacher_norms = collect_mlp_norms(teacher, tokenizer, texts, args.max_length)
    cleanup(teacher)
    teacher = None
    tokenizer = None
    student, tokenizer, _ = load_model(args.base_model, 2, args.aux_coef, gamma=None, capacity_factor=args.capacity_factor)
    student_norms = collect_mlp_norms(student, tokenizer, texts, args.max_length)
    cleanup(student)
    student = None
    tokenizer = None
    gamma = [float(max(args.gamma_min, min(args.gamma_max, t / (s + 1e-6)))) for t, s in zip(teacher_norms, student_norms)]
    save_json(out_dir / "calibration" / "gamma.json", {"teacher_norms": teacher_norms, "student_norms": student_norms, "gamma": gamma})
    return gamma


def make_image(step: int) -> Image.Image:
    image = Image.new("RGB", (224, 224), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    color = ((step * 47) % 255, (80 + step * 31) % 255, (150 + step * 17) % 255)
    draw.rectangle([32, 32, 192, 192], fill=color)
    draw.ellipse([72, 72, 152, 152], fill=(255 - color[0], 255 - color[1], 255 - color[2]))
    return image


def make_audio(step: int, sample_rate: int, seconds: float) -> np.ndarray:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    freq = 220.0 + 20.0 * (step % 12)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def load_encoders(vision_model_name: str, speech_model_name: str, device: torch.device):
    from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModel

    image_processor = load_pretrained(AutoImageProcessor, vision_model_name)
    vision_model = load_pretrained(AutoModel, vision_model_name).to(device).eval()
    speech_processor = load_pretrained(AutoFeatureExtractor, speech_model_name)
    speech_model = load_pretrained(AutoModel, speech_model_name).to(device).eval()
    for model in (vision_model, speech_model):
        for param in model.parameters():
            param.requires_grad_(False)
    return image_processor, vision_model, speech_processor, speech_model


def image_features(image_processor, vision_model, step: int, device: torch.device) -> torch.Tensor:
    batch = image_processor(images=[make_image(step)], return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        if hasattr(vision_model, "vision_model") and "pixel_values" in batch:
            outputs = vision_model.vision_model(pixel_values=batch["pixel_values"])
        else:
            outputs = vision_model(**batch)
    return outputs.last_hidden_state.detach()


def audio_features(speech_processor, speech_model, step: int, device: torch.device, sample_rate: int) -> torch.Tensor:
    audio = make_audio(step, sample_rate=sample_rate, seconds=1.0)
    processor_kwargs = {
        "sampling_rate": sample_rate,
        "return_tensors": "pt",
    }
    if hasattr(speech_processor, "n_samples"):
        processor_kwargs.update(
            {
                "padding": "max_length",
                "max_length": int(speech_processor.n_samples),
                "truncation": True,
            }
        )
    else:
        processor_kwargs["padding"] = True
    batch = speech_processor([audio], **processor_kwargs)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        if hasattr(speech_model, "encoder") and "input_features" in batch:
            outputs = speech_model.encoder(input_features=batch["input_features"])
        else:
            outputs = speech_model(**batch)
    return outputs.last_hidden_state.detach()


def run_multimodal_train(exp_id: str, args: argparse.Namespace, gamma: Sequence[float] | None, aux_coef: float, max_steps: int, out_dir: Path) -> Dict[str, object]:
    model, tokenizer, meta = load_model(args.base_model, 2, aux_coef, gamma=gamma, capacity_factor=args.capacity_factor)
    device = next(model.parameters()).device
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    image_processor, vision_model, speech_processor, speech_model = load_encoders(args.vision_model, args.speech_model, device)
    hidden_size = int(model.config.hidden_size)
    vision_cfg = getattr(vision_model.config, "vision_config", vision_model.config)
    vision_dim = int(getattr(vision_cfg, "hidden_size"))
    speech_dim = int(getattr(speech_model.config, "hidden_size", getattr(speech_model.config, "d_model")))
    wrapper = OLMoEMultimodalPrefixWrapper(
        lm=model,
        hidden_size=hidden_size,
        image_input_dim=vision_dim,
        audio_input_dim=speech_dim,
        image_prefix_tokens=args.image_prefix_tokens,
        audio_prefix_tokens=args.audio_prefix_tokens,
    ).to(device)
    trainable = list(wrapper.image_resampler.parameters()) + list(wrapper.audio_resampler.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    log_path = out_dir / exp_id / "train_metrics.jsonl"
    if log_path.exists():
        log_path.unlink()
    rows: List[Dict[str, object]] = []
    for step in range(1, max_steps + 1):
        text = f"Image and speech example {step}: describe the colored shape and transcribe the tone."
        encoded = encode_batch(tokenizer, [text], device, args.max_length)
        img = image_features(image_processor, vision_model, step, device)
        aud = audio_features(speech_processor, speech_model, step, device, args.sample_rate)
        outputs = wrapper(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=encoded["labels"],
            image_features=img,
            audio_features=aud,
        )
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        routing = router_metrics(outputs, 2, int(model.config.num_experts), args.capacity_factor)
        modality_routing = modality_router_metrics(
            outputs,
            2,
            int(model.config.num_experts),
            int(encoded["input_ids"].shape[0]),
            int(args.image_prefix_tokens),
            int(args.audio_prefix_tokens),
            int(encoded["input_ids"].shape[1]),
        )
        row = {
            "experiment_id": exp_id,
            "step": step,
            **meta,
            "vision_model": args.vision_model,
            "speech_model": args.speech_model,
            "loss": float(loss.detach().float().cpu()),
            "aux_loss": float(outputs.aux_loss.detach().float().cpu()) if outputs.aux_loss is not None else None,
            **routing,
            **modality_routing,
            **cuda_metrics(),
        }
        rows.append(row)
        append_jsonl(log_path, row)
        print(json.dumps(row, sort_keys=True))
    artifact = {
        "meta": meta,
        "vision_model": args.vision_model,
        "speech_model": args.speech_model,
        "steps": rows,
        "first_loss": rows[0]["loss"],
        "last_loss": rows[-1]["loss"],
        "min_loss": min(row["loss"] for row in rows),
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
    save_json(out_dir / exp_id / "metrics.json", artifact)
    torch.save({"image_resampler": wrapper.image_resampler.state_dict(), "audio_resampler": wrapper.audio_resampler.state_dict(), "meta": artifact}, out_dir / exp_id / "prefix_adapters.pt")
    cleanup(wrapper, model, vision_model, speech_model)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--vision-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--speech-model", default="openai/whisper-tiny.en")
    parser.add_argument("--output-dir", default="outputs/required_runs")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--aux-coef", type=float, default=0.01)
    parser.add_argument("--gamma-min", type=float, default=0.25)
    parser.add_argument("--gamma-max", type=float, default=2.0)
    parser.add_argument("--final-steps", type=int, default=20)
    parser.add_argument("--ablation-steps", type=int, default=10)
    parser.add_argument("--image-prefix-tokens", type=int, default=4)
    parser.add_argument("--audio-prefix-tokens", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = [
        "Sparse mixture of experts routing assigns each token to a small set of experts.",
        "A Top-2 student should preserve as much of the Top-8 teacher behavior as possible.",
        "Multimodal prefixes contain image and speech information before the target text.",
    ]
    manifest = {
        "runai_job_name": os.environ.get("RUNAI_JOB_NAME"),
        "runai_project": os.environ.get("RUNAI_PROJECT"),
        "command_mode": "required-runs",
        "base_model": args.base_model,
        "vision_model": args.vision_model,
        "speech_model": args.speech_model,
        "output_dir": str(out_dir),
    }
    save_json(out_dir / "manifest.json", manifest)

    model, tokenizer, meta = load_model(args.base_model, 8, args.aux_coef, gamma=None, capacity_factor=args.capacity_factor)
    e0 = run_text_eval("E0_top8_teacher_baseline", model, tokenizer, texts, out_dir, meta, args.max_length)
    cleanup(model)
    model = None
    tokenizer = None

    model, tokenizer, meta = load_model(args.base_model, 2, args.aux_coef, gamma=None, capacity_factor=args.capacity_factor)
    e1 = run_text_eval("E1_hard_top2", model, tokenizer, texts, out_dir, meta, args.max_length)
    cleanup(model)
    model = None
    tokenizer = None

    gamma = calibrate_gamma(args, texts, out_dir)
    model, tokenizer, meta = load_model(args.base_model, 2, args.aux_coef, gamma=gamma, capacity_factor=args.capacity_factor)
    e2 = run_text_eval("E2_calibrated_top2", model, tokenizer, texts, out_dir, {**meta, "gamma": gamma}, args.max_length)
    cleanup(model)
    model = None
    tokenizer = None

    e3 = run_multimodal_train("E3_final_multimodal_top2", args, gamma=gamma, aux_coef=args.aux_coef, max_steps=args.final_steps, out_dir=out_dir)
    e4 = run_multimodal_train("E4_no_aux_ablation", args, gamma=gamma, aux_coef=0.0, max_steps=args.ablation_steps, out_dir=out_dir)

    summary = {"manifest": manifest, "E0": e0, "E1": e1, "E2": e2, "E3": e3, "E4": e4}
    save_json(out_dir / "summary.json", summary)
    print(json.dumps({"summary_path": str(out_dir / "summary.json"), "experiments": ["E0", "E1", "E2", "E3", "E4"]}, sort_keys=True))


if __name__ == "__main__":
    main()
