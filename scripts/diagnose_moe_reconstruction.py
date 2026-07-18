"""Development-only OLMoE reconstruction diagnostics and ESFT selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from training.olmoe_required_runs import (
    DevelopmentEvidenceError,
    build_esft_selection,
    cleanup,
    encode_batch,
    iter_olmoe_mlp_layers,
    load_dynamic_expert_bias_state,
    load_model,
    moe_reconstruction_diagnostics,
    save_json,
    selected_expert_update_capability,
    set_olmoe_runtime_routing,
    validate_development_evidence,
)


SCHEMA_VERSION = 1
EXPECTED_EXPERIMENT_ID = "E3_final_multimodal_top2"


def validate_e3_checkpoint_metadata(state: Mapping[str, Any]) -> None:
    args = state.get("args")
    last_row = state.get("last_row")
    trainable_meta = state.get("trainable_meta")
    if not isinstance(args, dict):
        raise ValueError("E3 checkpoint is missing args metadata")
    if not isinstance(last_row, dict):
        raise ValueError("E3 checkpoint is missing last_row metadata")
    if last_row.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise ValueError(
            f"checkpoint is not {EXPECTED_EXPERIMENT_ID}: "
            f"{last_row.get('experiment_id')!r}"
        )
    if last_row.get("top_k") != 2:
        raise ValueError("E3 checkpoint metadata must declare Top-2 final inference")
    for location, metadata in (("args", args), ("trainable_meta", trainable_meta)):
        if isinstance(metadata, dict) and metadata.get("top_k") is not None:
            if int(metadata["top_k"]) != 2:
                raise ValueError(f"E3 checkpoint {location} must declare Top-2")
    if not isinstance(trainable_meta, dict):
        raise ValueError("E3 checkpoint is missing trainable_meta")


def load_verified_checkpoint(
    checkpoint: Path, expected_sha256: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load only an exact, development-only E3 Top-2 checkpoint."""
    validate_development_evidence([checkpoint], [])
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(
            "expected checkpoint SHA256 must be exactly 64 hexadecimal characters"
        )
    resolved = checkpoint.expanduser().resolve(strict=True)
    validate_development_evidence([resolved], [])
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(
            f"checkpoint SHA256 mismatch: expected={expected} actual={actual}"
        )
    try:
        state = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(resolved, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("checkpoint state must be a mapping")
    validate_e3_checkpoint_metadata(state)
    return state, {
        "path": str(resolved),
        "sha256": actual,
        "size_bytes": int(resolved.stat().st_size),
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "final_inference_top_k": 2,
    }


def _require_layer_state(
    state: object, expected_keys: set[str], component: str
) -> Mapping[str, Any]:
    if not isinstance(state, dict) or set(state) != expected_keys:
        observed = sorted(state) if isinstance(state, dict) else []
        raise ValueError(
            f"checkpoint {component} must cover every model layer: "
            f"expected={sorted(expected_keys)} observed={observed}"
        )
    return state


def _restore_selected_experts(
    layers: Mapping[int, Any], selected_state: Mapping[str, Any]
) -> Dict[str, List[int]]:
    restored: Dict[str, List[int]] = {}
    with torch.no_grad():
        for layer_idx, mlp in layers.items():
            key = f"layer_{layer_idx}"
            row = selected_state[key]
            if not isinstance(row, dict):
                raise ValueError(
                    f"checkpoint selected_experts.{key} must be a mapping"
                )
            expert_ids = sorted({int(value) for value in row.get("expert_ids", [])})
            if not expert_ids:
                raise ValueError(
                    f"checkpoint selected_experts.{key} has no expert IDs"
                )
            experts = mlp.experts
            indices = torch.tensor(
                expert_ids, dtype=torch.long, device=experts.gate_up_proj.device
            )
            for parameter_name in ("gate_up_proj", "down_proj"):
                parameter = getattr(experts, parameter_name)
                value = torch.as_tensor(row.get(parameter_name))
                expected_shape = (len(expert_ids), *parameter.shape[1:])
                if tuple(value.shape) != tuple(expected_shape):
                    raise ValueError(
                        f"selected expert shape mismatch for {key}.{parameter_name}"
                    )
                parameter.index_copy_(
                    0,
                    indices,
                    value.to(device=parameter.device, dtype=parameter.dtype),
                )
            restored[str(layer_idx)] = expert_ids
    return restored


def restore_checkpoint_model_state(
    model: Any, state: Mapping[str, Any]
) -> Dict[str, Any]:
    """Restore all checkpointed LM state and fail closed on trained omissions."""
    trainable_meta = state.get("trainable_meta")
    if not isinstance(trainable_meta, dict):
        raise ValueError("E3 checkpoint is missing trainable_meta")
    last_row = state.get("last_row")
    last_row = last_row if isinstance(last_row, dict) else {}
    trainable = {
        key: bool(trainable_meta.get(key) or last_row.get(key))
        for key in (
            "train_router_gates",
            "train_experts",
            "selected_expert_training",
            "train_lm_head",
        )
    }
    layers = dict(iter_olmoe_mlp_layers(model))
    expected_keys = {f"layer_{index}" for index in layers}
    restored: List[str] = []
    details: Dict[str, Any] = {}

    router_state = state.get("router_gates")
    if trainable["train_router_gates"] and router_state is None:
        raise ValueError("checkpoint trained router gates but omitted router_gates")
    if router_state is not None:
        router_state = _require_layer_state(
            router_state, expected_keys, "router_gates"
        )
        for layer_idx, mlp in layers.items():
            mlp.gate.load_state_dict(router_state[f"layer_{layer_idx}"])
        restored.append("router_gates")

    full_experts = state.get("experts")
    selected_experts = state.get("selected_experts")
    if trainable["train_experts"] and full_experts is None:
        raise ValueError("checkpoint trained full experts but omitted experts")
    if trainable["selected_expert_training"] and selected_experts is None:
        raise ValueError(
            "checkpoint trained selected experts but omitted selected_experts"
        )
    if full_experts is not None:
        full_experts = _require_layer_state(
            full_experts, expected_keys, "experts"
        )
        for layer_idx, mlp in layers.items():
            mlp.experts.load_state_dict(full_experts[f"layer_{layer_idx}"])
        restored.append("experts")
    if selected_experts is not None:
        selected_experts = _require_layer_state(
            selected_experts, expected_keys, "selected_experts"
        )
        details["selected_expert_ids_by_layer"] = _restore_selected_experts(
            layers, selected_experts
        )
        restored.append("selected_experts")

    output_embeddings = model.get_output_embeddings()
    input_embeddings = model.get_input_embeddings()
    output_state = state.get("lm_output_embeddings")
    input_state = state.get("lm_input_embeddings")
    if trainable["train_lm_head"]:
        if output_embeddings is None or output_state is None:
            raise ValueError(
                "checkpoint trained LM head but omitted output embeddings"
            )
        if input_embeddings is not output_embeddings and input_state is None:
            raise ValueError(
                "checkpoint trained untied LM embeddings but omitted input embeddings"
            )
    if output_state is not None:
        if output_embeddings is None:
            raise ValueError(
                "checkpoint has output embeddings but runtime model does not"
            )
        output_embeddings.load_state_dict(output_state)
        restored.append("lm_output_embeddings")
    if input_state is not None:
        if input_embeddings is None:
            raise ValueError(
                "checkpoint has input embeddings but runtime model does not"
            )
        input_embeddings.load_state_dict(input_state)
        restored.append("lm_input_embeddings")

    dynamic_bias = state.get("dynamic_expert_bias")
    if dynamic_bias is not None:
        _require_layer_state(dynamic_bias, expected_keys, "dynamic_expert_bias")
        loaded = int(load_dynamic_expert_bias_state(model, dynamic_bias))
        if loaded != len(layers):
            raise ValueError(
                "checkpoint dynamic_expert_bias could not be restored "
                "for every model layer"
            )
        restored.append("dynamic_expert_bias")
        details["dynamic_expert_bias_loaded_layers"] = loaded

    return {
        "restored_components": restored,
        "restored_component_details": details,
        "model_layer_count": len(layers),
    }


def load_gamma_json(path: Path) -> Tuple[List[float], Dict[str, Any]]:
    validate_development_evidence([path], [])
    resolved = path.expanduser().resolve(strict=True)
    validate_development_evidence([resolved], [])
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    gamma = payload.get("gamma") if isinstance(payload, dict) else None
    if not isinstance(gamma, list) or not gamma:
        raise ValueError(f"invalid gamma JSON: {resolved}")
    values = [float(value) for value in gamma]
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError(f"gamma values must be finite and positive: {resolved}")
    return values, {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_routing_source(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("routing sources must use SPLIT=PATH")
    split, raw_path = value.split("=", 1)
    split = split.strip().lower()
    if split not in {"train", "dev"}:
        raise argparse.ArgumentTypeError("routing source split must be train or dev")
    path = Path(raw_path).expanduser()
    return split, path


def _canonical_prefix_rows(
    sources: Sequence[Tuple[str, Path]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    canonical: List[Dict[str, Any]] = []
    validated_outer_rows: List[Dict[str, Any]] = []
    paths: List[Path] = []
    for split, path in sources:
        paths.append(path)
        for row_index, outer in enumerate(read_jsonl(path)):
            tagged_outer = {**outer, "split": split}
            if tagged_outer.get("real_subset") is not True:
                raise DevelopmentEvidenceError(
                    f"{path}:{row_index + 1} must declare real_subset=true"
                )
            validated_outer_rows.append(tagged_outer)
            candidates = outer.get("modality_layer_accounting")
            if candidates is None and outer.get("modality") in {"image_prefix", "audio_prefix"}:
                candidates = [outer]
            if not isinstance(candidates, list):
                continue
            for child in candidates:
                if not isinstance(child, dict):
                    raise ValueError(f"{path}:{row_index + 1} contains a non-object routing row")
                modality = str(child.get("modality", ""))
                if modality not in {"image_prefix", "audio_prefix"}:
                    continue
                canonical.append(
                    {
                        **child,
                        "split": split,
                        "real_subset": True,
                        "source_row": row_index + 1,
                    }
                )
    provenance = validate_development_evidence(paths, validated_outer_rows)
    validate_development_evidence(paths, canonical)
    if not canonical:
        raise ValueError("routing sources contain no image_prefix or audio_prefix per-layer rows")
    provenance["canonical_prefix_routing_rows"] = len(canonical)
    provenance["source_files"] = [
        {"path": str(path), "sha256": sha256_file(path)} for path in paths
    ]
    return canonical, provenance


def _development_prompts(path: Path) -> Tuple[List[str], Dict[str, Any]]:
    rows = read_jsonl(path)
    validate_development_evidence([path], rows)
    prompts: List[str] = []
    for index, row in enumerate(rows):
        if row.get("real_subset") is not True:
            raise DevelopmentEvidenceError(
                f"{path}:{index + 1} must declare real_subset=true"
            )
        if str(row.get("split", "")).lower() != "dev":
            raise DevelopmentEvidenceError(
                f"reconstruction row {index} must use split=dev"
            )
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"reconstruction row {index} has no non-empty text")
        prompts.append(text)
    return prompts, {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "split": "dev",
        "real_subset": True,
    }


def _git_provenance(repo_root: Path) -> Dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "working_tree_dirty": bool(run("status", "--porcelain")),
    }


def run_layerwise_reconstruction(
    base_model: str,
    checkpoint_state: Mapping[str, Any],
    gamma: Sequence[float] | None,
    prompts: Sequence[str],
    max_length: int,
    max_diagnostic_tokens: int,
    oracle_candidate_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Any]:
    last_row = checkpoint_state["last_row"]
    aux_coef = float(last_row.get("aux_coef", 0.01))
    model, tokenizer, model_meta = load_model(
        base_model,
        top_k=2,
        aux_coef=aux_coef,
        gamma=gamma,
        capacity_factor=float(last_row.get("capacity_factor", 1.25)),
        dynamic_expert_bias=checkpoint_state.get("dynamic_expert_bias") is not None,
    )
    try:
        restoration = restore_checkpoint_model_state(model, checkpoint_state)
    except Exception:
        cleanup(model)
        raise
    model.eval()
    device = next(model.parameters()).device
    captured_inputs: Dict[int, torch.Tensor] = {}
    captured_outputs: Dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx, mlp in iter_olmoe_mlp_layers(model):
        def pre_hook(_module, inputs, idx=layer_idx):
            captured_inputs[idx] = inputs[0].detach()

        def output_hook(_module, _inputs, output, idx=layer_idx):
            value = output[0] if isinstance(output, tuple) else output
            captured_outputs[idx] = value.detach()

        handles.append(mlp.register_forward_pre_hook(pre_hook))
        handles.append(mlp.register_forward_hook(output_hook))
    try:
        batch = encode_batch(tokenizer, prompts, device, int(max_length))
        set_olmoe_runtime_routing(model, top_k=8, aux_coef=aux_coef)
        with torch.no_grad():
            outputs = model(**batch, output_router_logits=True, return_dict=True)
    finally:
        set_olmoe_runtime_routing(model, top_k=2, aux_coef=aux_coef)
        for handle in handles:
            handle.remove()

    router_logits = outputs.router_logits
    tensors = list(router_logits) if not torch.is_tensor(router_logits) else [router_logits]
    valid_positions = batch["attention_mask"].reshape(-1).bool().nonzero().flatten()
    valid_positions = valid_positions[: int(max_diagnostic_tokens)]
    if valid_positions.numel() == 0:
        raise ValueError("reconstruction prompts produced no valid tokens")
    layer_reports: List[Dict[str, Any]] = []
    layers = dict(iter_olmoe_mlp_layers(model))
    if len(tensors) != len(layers):
        raise RuntimeError(
            f"router layer count mismatch: outputs={len(tensors)} model={len(layers)}"
        )
    for layer_idx, tensor in enumerate(tensors):
        hidden = captured_inputs[layer_idx].reshape(-1, captured_inputs[layer_idx].shape[-1])
        native = captured_outputs[layer_idx].reshape(-1, captured_outputs[layer_idx].shape[-1])
        logits = tensor.reshape(-1, tensor.shape[-1])
        expert_bias = getattr(layers[layer_idx], "expert_bias", None)
        if expert_bias is not None and bool(
            getattr(layers[layer_idx], "dynamic_expert_bias_enabled", False)
        ):
            logits = logits + expert_bias.to(device=logits.device, dtype=logits.dtype)
        positions = valid_positions.to(hidden.device)
        gamma_scale = float(getattr(layers[layer_idx], "gamma_scale", 1.0))
        if not math.isfinite(gamma_scale) or gamma_scale <= 0.0:
            raise ValueError(f"layer {layer_idx} has invalid gamma scale {gamma_scale}")
        native_pre_gamma = native.index_select(0, positions) / gamma_scale
        diagnostics = moe_reconstruction_diagnostics(
            layers[layer_idx].experts,
            hidden.index_select(0, positions),
            logits.index_select(0, positions.to(logits.device)),
            native_top8_output=native_pre_gamma,
            top_ks=(2, 4, 8),
            normalize_topk_prob=bool(getattr(model.config, "norm_topk_prob", False)),
            oracle_candidate_k=int(oracle_candidate_k),
        )
        layer_reports.append(
            {
                "layer": layer_idx,
                "gamma_scale": gamma_scale,
                "reconstruction_output_space": "pre_gamma_expert_mixture",
                **diagnostics,
            }
        )
    model_meta = {
        **model_meta,
        "diagnostic_tokens": int(valid_positions.numel()),
        "prompt_count": len(prompts),
        "native_runtime_top_k": 8,
        "final_inference_top_k": 2,
        "teacher_state_source": "restored_current_e3_checkpoint",
        **restoration,
        "development_only": True,
    }
    return layer_reports, model_meta, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--routing-source",
        action="append",
        type=parse_routing_source,
        required=True,
        metavar="SPLIT=PATH",
        help="Real train/dev JSONL containing modality_layer_accounting rows.",
    )
    parser.add_argument("--reconstruction-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--gamma-json", type=Path)
    parser.add_argument("--base-model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--selected-experts-per-layer", type=int, default=2)
    parser.add_argument("--selection-method", choices=("ESFT-Gate", "ESFT-Token"), default="ESFT-Gate")
    parser.add_argument("--expert-update-mode", choices=("full", "lora"), default="full")
    parser.add_argument("--expert-learning-rate", type=float, default=1e-6)
    parser.add_argument("--anchor-coefficient", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-diagnostic-tokens", type=int, default=8)
    parser.add_argument("--oracle-candidate-k", type=int, choices=(8,), default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < float(args.expert_learning_rate) <= 1e-4:
        raise ValueError("expert-learning-rate must be in the low-LR range (0, 1e-4]")
    if float(args.anchor_coefficient) < 0.0:
        raise ValueError("anchor-coefficient must be non-negative")
    if int(args.max_diagnostic_tokens) <= 0:
        raise ValueError("max-diagnostic-tokens must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {args.output}")
    validate_development_evidence(
        [args.output, args.reconstruction_source, args.checkpoint]
        + ([args.gamma_json] if args.gamma_json is not None else [])
        + [path for _split, path in args.routing_source],
        [],
    )
    checkpoint_state, checkpoint_provenance = load_verified_checkpoint(
        args.checkpoint, args.expected_checkpoint_sha256
    )
    checkpoint_args = checkpoint_state["args"]
    checkpoint_base_model = str(checkpoint_args.get("base_model", ""))
    if not checkpoint_base_model:
        raise ValueError("E3 checkpoint args do not declare base_model")
    if args.base_model != checkpoint_base_model:
        raise ValueError(
            f"base-model does not match checkpoint: CLI={args.base_model!r} "
            f"checkpoint={checkpoint_base_model!r}"
        )
    gamma_required = bool(checkpoint_state["last_row"].get("gamma_applied"))
    resolved_checkpoint = Path(checkpoint_provenance["path"])
    gamma_path = (
        args.gamma_json
        if args.gamma_json is not None
        else resolved_checkpoint.parent.parent / "calibration" / "gamma.json"
    )
    gamma: Sequence[float] | None = None
    gamma_provenance: Dict[str, Any] = {"path": None, "sha256": None}
    if gamma_required or args.gamma_json is not None:
        if not gamma_path.is_file():
            raise FileNotFoundError(
                f"checkpoint requires calibrated gamma: {gamma_path}"
            )
        gamma, gamma_provenance = load_gamma_json(gamma_path)

    update_capability = selected_expert_update_capability(args.expert_update_mode)
    routing_rows, routing_provenance = _canonical_prefix_rows(args.routing_source)
    selection = build_esft_selection(routing_rows, args.selected_experts_per_layer)
    prompts, prompt_provenance = _development_prompts(args.reconstruction_source)
    layer_reports, model_meta, model = run_layerwise_reconstruction(
        args.base_model,
        checkpoint_state,
        gamma,
        prompts,
        args.max_length,
        args.max_diagnostic_tokens,
        args.oracle_candidate_k,
    )
    selected_by_layer = {
        layer: row["selected_expert_ids"]
        for layer, row in selection["methods"][args.selection_method].items()
    }
    model_layers = {str(layer_idx) for layer_idx, _ in iter_olmoe_mlp_layers(model)}
    if set(selected_by_layer) != model_layers:
        cleanup(model)
        raise ValueError(
            "ESFT routing rows must cover every model layer before selected-expert training"
        )
    training_plan = {
        **update_capability,
        "selection_method": args.selection_method,
        "selected_expert_ids_by_layer": selected_by_layer,
        "non_selected_experts_frozen": True,
        "expert_learning_rate": float(args.expert_learning_rate),
        "weight_decay": 0.0,
        "weight_anchor_coefficient": float(args.anchor_coefficient),
        "implementation": "configure_selected_full_expert_training",
    }
    cleanup(model)

    repo_root = Path(__file__).resolve().parents[1]
    code_files = [Path(__file__).resolve(), repo_root / "training" / "olmoe_required_runs.py"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "development_moe_reconstruction_and_esft_selection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_evidence_used": False,
        "synthetic_evidence_used": False,
        "model": model_meta,
        "reconstruction": {
            "reference": "restored_checkpoint_native_top8_layer_output",
            "top_k_values": [2, 4, 8],
            "oracle_top2_is_diagnostic_only": True,
            "oracle_top2_is_inference_path": False,
            "layers": layer_reports,
        },
        "esft_selection": selection,
        "selected_expert_training_plan": training_plan,
        "provenance": {
            "checkpoint": {
                **checkpoint_provenance,
                "restored_components": model_meta["restored_components"],
                "restored_component_details": model_meta[
                    "restored_component_details"
                ],
            },
            "gamma": gamma_provenance,
            "routing": routing_provenance,
            "reconstruction_prompts": prompt_provenance,
            "git": _git_provenance(repo_root),
            "code": [
                {"path": str(path), "sha256": sha256_file(path)} for path in code_files
            ],
        },
    }
    save_json(args.output, report)
    print(json.dumps({"output": str(args.output), "layers": len(layer_reports)}, sort_keys=True))


if __name__ == "__main__":
    main()
