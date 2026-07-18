"""GPU functional audit for trained OLMoE multimodal checkpoints.

The audit intentionally uses development rows by default. It performs one
ephemeral aux-only optimizer step per modality and never writes a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn

from scripts.refresh_e3_from_checkpoint import load_checkpoint_into_wrapper
from training.olmoe_required_runs import (
    iter_olmoe_mlp_layers,
    load_encoders,
    load_model,
    save_json,
)
from training.olmoe_real_subset_runs import (
    absolutize_media_paths,
    audio_features_from_paths,
    image_features_from_paths,
    load_manifest,
    make_wrapper,
    per_example_prefix_nll,
    read_jsonl,
    split_tail,
    tokenize_prompt_targets,
)


DEFAULTS: Dict[str, Any] = {
    "base_model": "allenai/OLMoE-1B-7B-0924",
    "vision_model": "openai/clip-vit-base-patch32",
    "speech_model": "openai/whisper-base.en",
    "speech_target_space": "olmoe_text_hidden",
    "alignment_prefix_residual": False,
    "image_prefix_tokens": 50,
    "audio_prefix_tokens": 64,
    "encoder_feature_tokens": 100,
    "image_eval_samples": 250,
    "speech_eval_samples": 250,
    "sample_rate": 16000,
    "max_length": 512,
    "capacity_factor": 4.0,
    "aux_coef": 0.01,
    "learning_rate": 2e-4,
    "expert_dropout_prob": 0.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_evidence(path: Path | None) -> Dict[str, Any] | None:
    if path is None:
        return None
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def checkpoint_namespace(state: Mapping[str, Any], data_dir: Path) -> SimpleNamespace:
    raw = dict(state.get("args") or {})
    for key, value in DEFAULTS.items():
        raw.setdefault(key, value)
    raw["data_dir"] = str(data_dir)
    return SimpleNamespace(**raw)


def read_gamma(path: Path | None, state: Mapping[str, Any]) -> List[float] | None:
    if path is None or not path.exists():
        if bool((state.get("last_row") or {}).get("gamma_applied")):
            raise FileNotFoundError("checkpoint requires calibrated gamma but no gamma JSON was found")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("gamma")
    if not isinstance(values, list) or not values:
        raise ValueError(f"invalid gamma JSON: {path}")
    return [float(value) for value in values]


def restore_optional_experts(wrapper: nn.Module, state: Mapping[str, Any]) -> int:
    expert_state = state.get("experts")
    if not isinstance(expert_state, Mapping):
        return 0
    loaded = 0
    for layer_idx, mlp in iter_olmoe_mlp_layers(wrapper.lm):
        key = f"layer_{layer_idx}"
        if key in expert_state:
            mlp.experts.load_state_dict(expert_state[key])
            loaded += 1
    return loaded


def select_development_rows(
    manifest_path: Path,
    data_dir: Path,
    media_key: str,
    eval_count: int,
    explicit_manifest: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = [dict(row) for row in read_jsonl(manifest_path)]
    if len(rows) < 2:
        raise RuntimeError(f"{manifest_path} has fewer than two rows")
    absolutize_media_paths(rows, manifest_path.parent if explicit_manifest else data_dir)
    if explicit_manifest:
        candidates = rows
        policy = "explicit_manifest_tail"
        excluded_eval_rows = 0
    else:
        if eval_count <= 0 or len(rows) <= eval_count:
            raise RuntimeError(
                f"cannot establish a development-only tail for {manifest_path}: "
                f"rows={len(rows)} configured_eval_rows={eval_count}"
            )
        candidates, _ = split_tail(rows, eval_count)
        policy = "development_train_tail_excluding_configured_eval_tail"
        excluded_eval_rows = min(eval_count, len(rows)) if len(rows) > eval_count else 0
    selected: List[Dict[str, Any]] = []
    seen_media: set[str] = set()
    seen_media_hashes: set[str] = set()
    for source_index in range(len(candidates) - 1, -1, -1):
        row = dict(candidates[source_index])
        media_path = str(row.get(media_key, ""))
        if not media_path or media_path in seen_media:
            continue
        if not Path(media_path).is_file():
            raise FileNotFoundError(media_path)
        media_hash = sha256_file(Path(media_path))
        if media_hash in seen_media_hashes:
            continue
        row["_audit_source_index"] = source_index
        row["_audit_media_sha256"] = media_hash
        selected.append(row)
        seen_media.add(media_path)
        seen_media_hashes.add(media_hash)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RuntimeError(f"{manifest_path} does not contain two distinct existing media rows")
    selected.reverse()
    return selected, {
        "selection_policy": policy,
        "manifest_rows": len(rows),
        "candidate_rows": len(candidates),
        "excluded_configured_eval_tail_rows": excluded_eval_rows,
        "selected_source_indices": [int(row["_audit_source_index"]) for row in selected],
    }


def row_evidence(row: Mapping[str, Any], media_key: str, text_key: str) -> Dict[str, Any]:
    media_path = Path(str(row[media_key])).resolve()
    return {
        "id": row.get("id"),
        "source_index": int(row["_audit_source_index"]),
        "source": row.get("source"),
        "task": row.get("task"),
        "text": str(row[text_key]),
        "media_path": str(media_path),
        "media_sha256": sha256_file(media_path),
    }


def named_router_and_expert_parameters(
    lm: nn.Module,
) -> Tuple[List[Tuple[str, nn.Parameter]], List[Tuple[str, nn.Parameter]]]:
    routers: List[Tuple[str, nn.Parameter]] = []
    experts: List[Tuple[str, nn.Parameter]] = []
    for layer_idx, mlp in iter_olmoe_mlp_layers(lm):
        routers.extend(
            (f"lm.model.layers.{layer_idx}.mlp.gate.{name}", parameter)
            for name, parameter in mlp.gate.named_parameters()
        )
        experts.extend(
            (f"lm.model.layers.{layer_idx}.mlp.experts.{name}", parameter)
            for name, parameter in mlp.experts.named_parameters()
        )
    return routers, experts


def clone_parameters(parameters: Iterable[Tuple[str, nn.Parameter]]) -> Dict[str, Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in parameters
    }


def parameter_versions(parameters: Iterable[Tuple[str, nn.Parameter]]) -> Dict[str, int]:
    return {name: int(parameter._version) for name, parameter in parameters}


def parameter_probes(parameters: Iterable[Tuple[str, nn.Parameter]]) -> Dict[str, List[float]]:
    probes: Dict[str, List[float]] = {}
    for name, parameter in parameters:
        flat = parameter.detach().reshape(-1)
        if flat.numel() == 0:
            probes[name] = []
            continue
        indices = sorted({0, int(flat.numel() // 2), int(flat.numel() - 1)})
        probes[name] = [float(flat[index].float().cpu()) for index in indices]
    return probes


def direct_state_delta(
    before: Mapping[str, Tensor],
    parameters: Iterable[Tuple[str, nn.Parameter]],
) -> Dict[str, Any]:
    squared_l2 = 0.0
    max_abs = 0.0
    changed = 0
    parameter_count = 0
    numel = 0
    for name, parameter in parameters:
        parameter_count += 1
        current = parameter.detach().float().cpu()
        delta = current - before[name]
        numel += int(delta.numel())
        squared_l2 += float(delta.double().square().sum())
        local_max = float(delta.abs().max()) if delta.numel() else 0.0
        max_abs = max(max_abs, local_max)
        changed += int(local_max != 0.0)
    return {
        "parameter_count": parameter_count,
        "numel": numel,
        "changed_parameter_count": changed,
        "l2_norm": math.sqrt(squared_l2),
        "max_abs": max_abs,
    }


def version_probe_delta(
    parameters: Sequence[Tuple[str, nn.Parameter]],
    versions_before: Mapping[str, int],
    probes_before: Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    versions_after = parameter_versions(parameters)
    probes_after = parameter_probes(parameters)
    version_deltas = {
        name: versions_after[name] - versions_before[name]
        for name in versions_before
    }
    probe_max_abs = 0.0
    changed_probes = 0
    for name, before_values in probes_before.items():
        differences = [abs(after - before) for before, after in zip(before_values, probes_after[name])]
        local_max = max(differences, default=0.0)
        probe_max_abs = max(probe_max_abs, local_max)
        changed_probes += int(local_max != 0.0)
    return {
        "parameter_count": len(parameters),
        "numel": int(sum(parameter.numel() for _, parameter in parameters)),
        "changed_parameter_version_count": sum(delta != 0 for delta in version_deltas.values()),
        "max_parameter_version_delta": max(version_deltas.values(), default=0),
        "changed_parameter_probe_count": changed_probes,
        "probe_max_abs_delta": probe_max_abs,
        "evidence_semantics": "unchanged PyTorch mutation versions plus deterministic first/middle/last value probes",
    }


def grad_summary(parameters: Iterable[Tuple[str, nn.Parameter]]) -> Dict[str, Any]:
    parameter_count = 0
    grads_present = 0
    finite_grad_parameters = 0
    nonzero_grad_parameters = 0
    squared_l2 = 0.0
    max_abs = 0.0
    for _, parameter in parameters:
        parameter_count += 1
        grad = parameter.grad
        if grad is None:
            continue
        grads_present += 1
        grad_float = grad.detach().float()
        finite = bool(torch.isfinite(grad_float).all())
        finite_grad_parameters += int(finite)
        local_max = float(grad_float.abs().max()) if grad_float.numel() else 0.0
        nonzero_grad_parameters += int(local_max > 0.0)
        max_abs = max(max_abs, local_max)
        squared_l2 += float(grad_float.double().square().sum().cpu())
    return {
        "parameter_count": parameter_count,
        "grads_present": grads_present,
        "finite_grad_parameters": finite_grad_parameters,
        "nonzero_grad_parameters": nonzero_grad_parameters,
        "l2_norm": math.sqrt(squared_l2),
        "max_abs": max_abs,
    }


def tensor_delta(left: Tensor, right: Tensor) -> Dict[str, Any]:
    delta = left.detach().float() - right.detach().float()
    return {
        "shape": list(left.shape),
        "finite": bool(torch.isfinite(delta).all()),
        "l2_norm": float(delta.double().square().sum().sqrt().cpu()),
        "max_abs": float(delta.abs().max().cpu()) if delta.numel() else 0.0,
        "exactly_equal": bool(torch.equal(left.detach(), right.detach())),
    }


def tensor_grad_summary(tensor: Tensor) -> Dict[str, Any]:
    grad = tensor.grad
    if grad is None:
        return {"present": False, "finite": False, "l2_norm": 0.0, "max_abs": 0.0}
    grad_float = grad.detach().float()
    return {
        "present": True,
        "finite": bool(torch.isfinite(grad_float).all()),
        "l2_norm": float(grad_float.double().square().sum().sqrt().cpu()),
        "max_abs": float(grad_float.abs().max().cpu()) if grad_float.numel() else 0.0,
    }


def finite_scalar(value: Tensor) -> Tuple[float, bool]:
    result = float(value.detach().float().cpu())
    return result, math.isfinite(result)


def audit_modality(
    wrapper: nn.Module,
    tokenizer: Any,
    modality: str,
    features: Tensor,
    rows: Sequence[Mapping[str, Any]],
    run_args: SimpleNamespace,
    aux_coef: float,
    learning_rate: float,
    router_parameters: Sequence[Tuple[str, nn.Parameter]],
    expert_parameters: Sequence[Tuple[str, nn.Parameter]],
) -> Dict[str, Any]:
    if modality == "image":
        resampler = wrapper.image_resampler
        other_resampler = wrapper.audio_resampler
        prefix_fn = wrapper.image_prefix
        feature_key = "image_features"
        prefix_tokens = int(run_args.image_prefix_tokens)
        prompt = "Caption:"
        text_key = "caption"
    elif modality == "speech":
        resampler = wrapper.audio_resampler
        other_resampler = wrapper.image_resampler
        prefix_fn = wrapper.audio_prefix
        feature_key = "audio_features"
        prefix_tokens = int(run_args.audio_prefix_tokens)
        prompt = "Transcript:"
        text_key = "transcript"
    else:
        raise ValueError(modality)

    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in resampler.parameters():
        parameter.requires_grad_(True)

    resampler_parameters = [(f"{modality}_resampler.{name}", parameter) for name, parameter in resampler.named_parameters()]
    other_parameters = [(f"other_resampler.{name}", parameter) for name, parameter in other_resampler.named_parameters()]
    router_before = clone_parameters(router_parameters)
    resampler_before = clone_parameters(resampler_parameters)
    other_before = clone_parameters(other_parameters)
    expert_versions_before = parameter_versions(expert_parameters)
    expert_probes_before = parameter_probes(expert_parameters)

    same_text = tokenize_prompt_targets(
        tokenizer,
        [prompt],
        [str(rows[0][text_key])],
        features.device,
        int(run_args.max_length),
    )
    aux_batch = tokenize_prompt_targets(
        tokenizer,
        [prompt, prompt],
        [str(row[text_key]) for row in rows],
        features.device,
        int(run_args.max_length),
    )

    wrapper.eval()
    with torch.no_grad():
        prefixes = prefix_fn(features)
        output_left = wrapper(**same_text, **{feature_key: features[0:1]})
        output_right = wrapper(**same_text, **{feature_key: features[1:2]})
        text_length = int(same_text["input_ids"].shape[1])
        left_logits = output_left.logits[:, -text_length:]
        right_logits = output_right.logits[:, -text_length:]
        left_score = -per_example_prefix_nll(output_left.logits, same_text["labels"], prefix_tokens)[0]
        right_score = -per_example_prefix_nll(output_right.logits, same_text["labels"], prefix_tokens)[0]

    captured: Dict[str, Tensor] = {}

    def capture_prefix(_module: nn.Module, _inputs: Tuple[Any, ...], output: Tensor) -> None:
        output.retain_grad()
        captured["prefix"] = output

    hook = resampler.register_forward_hook(capture_prefix)
    optimizer = torch.optim.AdamW(resampler.parameters(), lr=learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    aux_outputs = wrapper(
        input_ids=aux_batch["input_ids"],
        attention_mask=aux_batch["attention_mask"],
        labels=None,
        **{feature_key: features},
    )
    if aux_outputs.aux_loss is None:
        hook.remove()
        raise RuntimeError(f"{modality} forward did not return router aux_loss")
    raw_aux = aux_outputs.aux_loss
    weighted_aux = raw_aux * float(aux_coef)
    weighted_aux.backward()
    hook.remove()

    prefix_grad = tensor_grad_summary(captured["prefix"])
    resampler_grads = grad_summary(resampler_parameters)
    router_grads = grad_summary(router_parameters)
    expert_grads = grad_summary(expert_parameters)
    optimizer.step()

    resampler_delta = direct_state_delta(resampler_before, resampler_parameters)
    other_delta = direct_state_delta(other_before, other_parameters)
    router_delta = direct_state_delta(router_before, router_parameters)
    expert_delta = version_probe_delta(
        expert_parameters,
        expert_versions_before,
        expert_probes_before,
    )
    raw_aux_value, raw_aux_finite = finite_scalar(raw_aux)
    weighted_aux_value, weighted_aux_finite = finite_scalar(weighted_aux)
    score_left = float(left_score.detach().float().cpu())
    score_right = float(right_score.detach().float().cpu())
    score_delta = abs(score_left - score_right)
    prefix_delta = tensor_delta(prefixes[0], prefixes[1])
    feature_delta = tensor_delta(features[0], features[1])
    logits_delta = tensor_delta(left_logits, right_logits)

    trainable_names = [name for name, parameter in wrapper.named_parameters() if parameter.requires_grad]
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    intended_ids = {id(parameter) for parameter in resampler.parameters()}
    gates = {
        "real_encoder_inputs_differ": feature_delta["finite"] and feature_delta["max_abs"] > 0.0,
        "changing_encoder_input_changes_prefix": prefix_delta["finite"] and prefix_delta["max_abs"] > 0.0,
        "changing_prefix_changes_shared_lm_logits": logits_delta["finite"] and logits_delta["max_abs"] > 0.0,
        "changing_prefix_changes_target_score": math.isfinite(score_delta) and score_delta > 0.0,
        "raw_aux_finite_nonzero": raw_aux_finite and raw_aux_value != 0.0,
        "weighted_aux_finite_nonzero": weighted_aux_finite and weighted_aux_value != 0.0,
        "prefix_tensor_grad_finite_nonzero": prefix_grad["finite"] and prefix_grad["l2_norm"] > 0.0,
        "resampler_grad_finite_nonzero": (
            resampler_grads["grads_present"] > 0
            and resampler_grads["finite_grad_parameters"] == resampler_grads["grads_present"]
            and resampler_grads["l2_norm"] > 0.0
        ),
        "router_frozen_no_gradient": (
            bool(router_parameters)
            and all(not parameter.requires_grad for _, parameter in router_parameters)
            and router_grads["grads_present"] == 0
        ),
        "experts_frozen_no_gradient": (
            bool(expert_parameters)
            and all(not parameter.requires_grad for _, parameter in expert_parameters)
            and expert_grads["grads_present"] == 0
        ),
        "optimizer_contains_only_intended_resampler": optimizer_ids == intended_ids,
        "optimizer_step_changes_intended_resampler": resampler_delta["l2_norm"] > 0.0,
        "optimizer_step_does_not_change_other_resampler": other_delta["max_abs"] == 0.0,
        "optimizer_step_does_not_change_router": router_delta["max_abs"] == 0.0,
        "optimizer_step_does_not_change_experts": (
            expert_delta["changed_parameter_version_count"] == 0
            and expert_delta["changed_parameter_probe_count"] == 0
        ),
    }
    return {
        "modality": modality,
        "aux_only_objective": "checkpoint_aux_coef * hf_router_aux_loss_raw",
        "optimizer": {
            "type": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
            "trainable_parameter_names": trainable_names,
        },
        "input_ids": {
            "shared_score_input_ids": same_text["input_ids"].detach().cpu().tolist(),
            "aux_backward_input_ids": aux_batch["input_ids"].detach().cpu().tolist(),
        },
        "encoder_feature_delta": feature_delta,
        "prefix_delta": prefix_delta,
        "shared_lm_logits_delta": logits_delta,
        "shared_target_scores": {
            "row_0": score_left,
            "row_1": score_right,
            "absolute_delta": score_delta,
        },
        "router_aux_loss_raw": raw_aux_value,
        "router_aux_loss_weighted": weighted_aux_value,
        "aux_coef": float(aux_coef),
        "grad_norms": {
            "prefix_tensor": prefix_grad,
            "intended_resampler": resampler_grads,
            "frozen_router_gates": router_grads,
            "frozen_experts": expert_grads,
        },
        "state_deltas": {
            "intended_resampler": resampler_delta,
            "other_resampler": other_delta,
            "frozen_router_gates": router_delta,
            "frozen_experts": expert_delta,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--image-manifest", type=Path, default=None)
    parser.add_argument("--speech-manifest", type=Path, default=None)
    parser.add_argument("--run-manifest", type=Path, default=None)
    parser.add_argument("--gamma-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("functional path audit requires a CUDA GPU")
    checkpoint_path = args.checkpoint.resolve()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = dict(state.get("args") or {})
    data_dir = (args.data_dir or Path(checkpoint_args.get("data_dir", "data/real_subset_final"))).resolve()
    load_manifest(data_dir)
    run_args = checkpoint_namespace(state, data_dir)
    run_root = checkpoint_path.parent.parent
    run_manifest_path = (args.run_manifest or (run_root / "manifest.json")).resolve()
    data_manifest_path = (data_dir / "manifest.json").resolve()
    image_manifest_path = (args.image_manifest or (data_dir / "image_captions.jsonl")).resolve()
    speech_manifest_path = (args.speech_manifest or (data_dir / "speech_transcripts.jsonl")).resolve()
    gamma_path = (args.gamma_json or (run_root / "calibration" / "gamma.json")).resolve()
    gamma = read_gamma(gamma_path if gamma_path.exists() else None, state)

    image_rows, image_selection = select_development_rows(
        image_manifest_path,
        data_dir,
        "image_path",
        int(run_args.image_eval_samples),
        args.image_manifest is not None,
    )
    speech_rows, speech_selection = select_development_rows(
        speech_manifest_path,
        data_dir,
        "audio_path",
        int(run_args.speech_eval_samples),
        args.speech_manifest is not None,
    )

    last_row = dict(state.get("last_row") or {})
    aux_coef = float(last_row.get("aux_coef", run_args.aux_coef))
    capacity_factor = float(last_row.get("capacity_factor", run_args.capacity_factor))
    top_k = int(last_row.get("top_k", 2))
    model, tokenizer, model_meta = load_model(
        run_args.base_model,
        top_k,
        aux_coef,
        gamma=gamma,
        capacity_factor=capacity_factor,
        expert_dropout_prob=float(getattr(run_args, "expert_dropout_prob", 0.0)),
        dynamic_expert_bias=bool(state.get("dynamic_expert_bias")),
    )
    device = next(model.parameters()).device
    image_processor, vision_model, speech_processor, speech_model = load_encoders(
        run_args.vision_model,
        run_args.speech_model,
        device,
    )
    wrapper = make_wrapper(model, vision_model, speech_model, run_args).to(device)
    load_checkpoint_into_wrapper(wrapper, state)
    loaded_expert_layers = restore_optional_experts(wrapper, state)
    wrapper.eval()

    image_features = image_features_from_paths(
        image_processor,
        vision_model,
        [str(row["image_path"]) for row in image_rows],
        device,
        int(run_args.encoder_feature_tokens),
    )
    speech_features = audio_features_from_paths(
        speech_processor,
        speech_model,
        [str(row["audio_path"]) for row in speech_rows],
        device,
        int(run_args.sample_rate),
        int(run_args.encoder_feature_tokens),
    )
    router_parameters, expert_parameters = named_router_and_expert_parameters(wrapper.lm)
    learning_rate = float(args.learning_rate or run_args.learning_rate)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    image_result = audit_modality(
        wrapper,
        tokenizer,
        "image",
        image_features,
        image_rows,
        run_args,
        aux_coef,
        learning_rate,
        router_parameters,
        expert_parameters,
    )
    load_checkpoint_into_wrapper(wrapper, state)
    restore_optional_experts(wrapper, state)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    speech_result = audit_modality(
        wrapper,
        tokenizer,
        "speech",
        speech_features,
        speech_rows,
        run_args,
        aux_coef,
        learning_rate,
        router_parameters,
        expert_parameters,
    )

    artifact = {
        "schema_version": 1,
        "audit": "trained_olmoe_multimodal_functional_paths",
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "seed": int(args.seed),
        "ephemeral_optimizer_steps": 2,
        "writes_checkpoint": False,
        "checkpoint": file_evidence(checkpoint_path),
        "manifests": {
            "run": file_evidence(run_manifest_path),
            "data": file_evidence(data_manifest_path),
            "image_rows": file_evidence(image_manifest_path),
            "speech_rows": file_evidence(speech_manifest_path),
            "gamma": file_evidence(gamma_path) if gamma_path.exists() else None,
        },
        "checkpoint_settings": {
            "base_model": run_args.base_model,
            "vision_model": run_args.vision_model,
            "speech_model": run_args.speech_model,
            "speech_target_space": run_args.speech_target_space,
            "alignment_prefix_residual": bool(run_args.alignment_prefix_residual),
            "image_prefix_tokens": int(run_args.image_prefix_tokens),
            "audio_prefix_tokens": int(run_args.audio_prefix_tokens),
            "encoder_feature_tokens": int(run_args.encoder_feature_tokens),
            "sample_rate": int(run_args.sample_rate),
            "top_k": top_k,
            "capacity_factor": capacity_factor,
            "aux_coef": aux_coef,
            "model_meta": model_meta,
            "loaded_optional_expert_layers": loaded_expert_layers,
        },
        "data_policy": {
            "sealed_test_used": False if args.image_manifest is None and args.speech_manifest is None else None,
            "note": "Defaults exclude configured eval tails; explicit manifests are caller-selected and used only for this ephemeral audit.",
            "image": image_selection,
            "speech": speech_selection,
        },
        "inputs": {
            "image": [row_evidence(row, "image_path", "caption") for row in image_rows],
            "speech": [row_evidence(row, "audio_path", "transcript") for row in speech_rows],
        },
        "image": image_result,
        "speech": speech_result,
        "passed": bool(image_result["passed"] and speech_result["passed"]),
    }
    output_path = (args.output or (checkpoint_path.parent / "functional_path_audit.json")).resolve()
    save_json(output_path, artifact)
    print(json.dumps({"output": str(output_path), "passed": artifact["passed"]}, sort_keys=True))
    if not artifact["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
