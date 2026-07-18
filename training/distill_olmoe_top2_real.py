"""Real-data Top-8 -> Top-2 OLMoE distillation controls.

This runner is intentionally text-only: it tests whether a Top-2 student can
match the native Top-8 teacher better than hard Top-2 / gamma calibration under
matched data, steps, trainable parameters, and evaluation blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scripts.stage_b_checkpoint_provenance import (
    build_checkpoint_run_identity,
    build_stage_b_run_provenance,
    write_stage_b_companion,
)
from training.olmoe_real_subset_runs import (
    lm_embeddings_are_tied,
    read_jsonl,
    sample_cycle,
    tensorize_blocks,
)
from training.olmoe_required_runs import (
    base_model_identity,
    calibrate_gamma,
    cleanup,
    cuda_metrics,
    load_model,
    router_metrics,
    save_json,
    set_olmoe_runtime_routing,
)


CHECKPOINT_VERSION = 2
FINAL_STUDENT_TOP_K = 2
RESUME_ARG_FIELDS = (
    "data_dir",
    "base_model",
    "seed",
    "teacher_top_k",
    "student_top_k",
    "capacity_factor",
    "aux_coef",
    "gamma_min",
    "gamma_max",
    "max_length",
    "train_batch_size",
    "learning_rate",
    "router_learning_rate",
    "gamma_learning_rate",
    "weight_decay",
    "grad_clip",
    "distill_logit_coef",
    "distill_temperature",
    "router_distill_coef",
    "router_distill_temperature",
    "distill_hidden_coef",
    "distill_hidden_layers",
    "distill_hidden_mode",
    "moe_reconstruction_coef",
    "moe_reconstruction_layers",
    "text_replay_coef",
    "text_replay_manifest",
    "student_k_curriculum",
    "train_lm_head",
    "train_router_gates",
    "train_gamma_scale",
)
TRAINABLE_CONTRACT_FIELDS = (
    "train_lm_head",
    "train_router_gates",
    "train_gamma_scale",
    "gamma_scale_count",
    "trainable_params",
    "optimizer_groups",
    "trainable_parameter_specs",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    )


def read_bound_jsonl(path: Path, role: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = path.expanduser()
    if raw.is_symlink():
        raise ValueError(f"{role} JSONL must be a regular non-symlink file")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{role} JSONL must be a regular non-symlink file")
    payload = resolved.read_bytes()
    rows: List[Dict[str, Any]] = []
    try:
        for line in payload.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{role} JSONL rows must be objects")
            rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role} JSONL: {resolved}") from exc
    return rows, {
        "role": role,
        "canonical_path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "row_count": len(rows),
    }


def stage_b_data_contract(
    data_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "canonical_data_root": str(data_root.expanduser().resolve(strict=True)),
        "jsonl_inputs": [dict(record) for record in records],
    }


def capture_rng_state() -> Dict[str, Any]:
    numpy_state = np.random.get_state()
    cuda_states = (
        [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": cuda_states,
        "cuda_device_count": int(torch.cuda.device_count()),
    }


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "cuda_device_count",
    }
    if set(state) != required:
        raise ValueError("Stage B v2 RNG state is incomplete")
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("Stage B v2 NumPy RNG state is incomplete")
    if (
        not isinstance(state["torch_cpu"], torch.Tensor)
        or not isinstance(state["torch_cuda"], list)
        or state["cuda_device_count"] != torch.cuda.device_count()
    ):
        raise ValueError("Stage B v2 Torch RNG topology mismatch")
    random.setstate(_as_tuple(state["python"]))
    np.random.set_state((
        str(numpy_state["bit_generator"]),
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or any(
            not isinstance(value, torch.Tensor)
            for value in state["torch_cuda"]
        ):
            raise ValueError("Stage B v2 CUDA RNG state is invalid")
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["torch_cuda"]]
        )


def parse_layers(spec: str, num_hidden_states: int) -> List[int]:
    spec = str(spec).strip().lower()
    if not spec or spec == "none":
        return []
    if spec == "last":
        return [num_hidden_states - 1]
    if spec == "all":
        return list(range(num_hidden_states))
    layers: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx = num_hidden_states + idx
        if idx < 0 or idx >= num_hidden_states:
            raise ValueError(f"hidden layer index {part!r} out of range for {num_hidden_states} hidden states")
        layers.append(idx)
    return sorted(set(layers))


def make_gamma_scale_trainable(model) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for layer in model.model.layers:
        block = layer.mlp
        gamma = getattr(block, "gamma_scale", None)
        if gamma is None:
            continue
        if isinstance(gamma, torch.nn.Parameter):
            param = gamma
        else:
            init = gamma.detach().float().clone()
            if "gamma_scale" in block._buffers:
                del block._buffers["gamma_scale"]
            param = torch.nn.Parameter(init)
            block.register_parameter("gamma_scale", param)
        param.requires_grad_(True)
        params.append(param)
    return params


def gamma_scale_values(model) -> List[float]:
    values: List[float] = []
    for layer in model.model.layers:
        gamma = getattr(layer.mlp, "gamma_scale", None)
        if gamma is not None:
            values.append(float(gamma.detach().float().cpu().item()))
    return values


def set_trainable(model, train_lm_head: bool, train_router_gates: bool, train_gamma_scale: bool) -> Dict[str, Any]:
    for param in model.parameters():
        param.requires_grad_(False)
    groups: List[Dict[str, Any]] = []
    seen = set()
    lm_head_params: List[torch.nn.Parameter] = []
    if train_lm_head:
        for module in [model.get_output_embeddings(), model.get_input_embeddings()]:
            if module is None:
                continue
            for param in module.parameters():
                if id(param) in seen:
                    continue
                seen.add(id(param))
                param.requires_grad_(True)
                lm_head_params.append(param)
        if lm_head_params:
            groups.append({"name": "lm_head_embeddings", "params": lm_head_params})
    gamma_params: List[torch.nn.Parameter] = []
    if train_gamma_scale:
        gamma_params = make_gamma_scale_trainable(model)
        if gamma_params:
            groups.append({"name": "gamma_scale", "params": gamma_params})
    router_params: List[torch.nn.Parameter] = []
    if train_router_gates:
        for layer in model.model.layers:
            for param in layer.mlp.gate.parameters():
                param.requires_grad_(True)
                router_params.append(param)
        if router_params:
            groups.append({"name": "router_gates", "params": router_params})
    names_by_id: Dict[int, str] = {}
    for name, parameter in model.named_parameters():
        names_by_id.setdefault(id(parameter), name)
    parameter_specs: List[Dict[str, Any]] = []
    optimizer_groups: List[Dict[str, Any]] = []
    for group in groups:
        parameter_names: List[str] = []
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError("trainable parameter is missing a canonical model name")
            parameter_names.append(name)
            parameter_specs.append({
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
            })
        group["parameter_names"] = parameter_names
        optimizer_groups.append({
            "name": group["name"],
            "trainable_params": int(
                sum(p.numel() for p in group["params"] if p.requires_grad)
            ),
            "parameter_names": parameter_names,
        })
    trainable_params = sum(
        p.numel() for group in groups for p in group["params"] if p.requires_grad
    )
    return {
        "groups": groups,
        "train_lm_head": bool(train_lm_head),
        "train_router_gates": bool(train_router_gates),
        "train_gamma_scale": bool(train_gamma_scale),
        "gamma_scale_count": int(len(gamma_params)),
        "gamma_scale_values_initial": gamma_scale_values(model),
        "trainable_params": int(trainable_params),
        "optimizer_groups": optimizer_groups,
        "trainable_parameter_specs": parameter_specs,
    }


def masked_token_rows(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels != -100
    return logits[mask]


def logits_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    s = masked_token_rows(student_logits, labels).float()
    t = masked_token_rows(teacher_logits, labels).float()
    if s.numel() == 0:
        return student_logits.new_zeros(())
    temp = max(float(temperature), 1e-6)
    return F.kl_div(F.log_softmax(s / temp, dim=-1), F.softmax(t / temp, dim=-1), reduction="batchmean") * temp * temp


def router_kl(
    student_router: Any,
    teacher_router: Any,
    temperature: float,
    token_mask: torch.Tensor | None = None,
    expected_num_experts: int | None = None,
) -> torch.Tensor:
    s_layers = list(student_router) if not torch.is_tensor(student_router) else [student_router]
    t_layers = list(teacher_router) if not torch.is_tensor(teacher_router) else [teacher_router]
    if len(s_layers) != len(t_layers):
        raise ValueError(f"router layer count mismatch: student={len(s_layers)} teacher={len(t_layers)}")
    losses = []
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError("router distillation temperature must be positive")
    flat_mask = token_mask.reshape(-1).bool() if token_mask is not None else None
    for layer_idx, (student_logits, teacher_logits) in enumerate(zip(s_layers, t_layers)):
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"router shape mismatch at layer {layer_idx}: "
                f"student={tuple(student_logits.shape)} teacher={tuple(teacher_logits.shape)}"
            )
        if student_logits.ndim < 2:
            raise ValueError(f"router logits at layer {layer_idx} must include an expert dimension")
        num_experts = int(student_logits.shape[-1])
        if expected_num_experts is not None and num_experts != int(expected_num_experts):
            raise ValueError(
                f"router expert dimension mismatch at layer {layer_idx}: "
                f"expected={expected_num_experts} actual={num_experts}"
            )
        student_rows = student_logits.reshape(-1, num_experts).float()
        teacher_rows = teacher_logits.reshape(-1, num_experts).float()
        if flat_mask is not None:
            if flat_mask.numel() != student_rows.shape[0]:
                raise ValueError(
                    f"router token mask has {flat_mask.numel()} rows, expected {student_rows.shape[0]}"
                )
            student_rows = student_rows[flat_mask]
            teacher_rows = teacher_rows[flat_mask]
        if student_rows.numel() == 0:
            continue
        losses.append(
            F.kl_div(
                F.log_softmax(student_rows / temp, dim=-1),
                F.softmax(teacher_rows / temp, dim=-1),
                reduction="batchmean",
            )
            * temp
            * temp
        )
    if losses:
        return torch.stack(losses).mean()
    if s_layers:
        return s_layers[0].new_zeros(())
    return torch.tensor(0.0)


def hidden_match(student_hidden: Sequence[torch.Tensor], teacher_hidden: Sequence[torch.Tensor], labels: torch.Tensor, layers: Sequence[int], mode: str) -> torch.Tensor:
    if not layers:
        return labels.new_zeros((), dtype=torch.float32)
    mask = labels != -100
    losses = []
    for idx in layers:
        s = student_hidden[idx][:, :-1].float()[mask]
        t = teacher_hidden[idx][:, :-1].float()[mask]
        if s.numel() == 0:
            continue
        if mode == "cosine":
            losses.append((1.0 - F.cosine_similarity(s, t, dim=-1)).mean())
        else:
            losses.append(F.mse_loss(s, t))
    return torch.stack(losses).mean() if losses else labels.new_zeros((), dtype=torch.float32)


def text_replay_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[:-1] != labels.shape:
        raise ValueError(f"text replay shape mismatch: logits={tuple(logits.shape)} labels={tuple(labels.shape)}")
    if not bool((labels != -100).any()):
        return logits.new_zeros(())
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


class MoEOutputCapture:
    def __init__(self, model, layers: Sequence[int]):
        self.outputs: Dict[int, torch.Tensor] = {}
        self.handles = []
        model_layers = list(model.model.layers)
        for layer_idx in layers:
            if layer_idx < 0 or layer_idx >= len(model_layers):
                raise ValueError(f"MoE layer index {layer_idx} out of range for {len(model_layers)} layers")

            def capture(_module, _inputs, output, *, index=layer_idx):
                self.outputs[index] = output[0] if isinstance(output, tuple) else output

            self.handles.append(model_layers[layer_idx].mlp.register_forward_hook(capture))

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def moe_output_reconstruction(
    student_outputs: Dict[int, torch.Tensor],
    teacher_outputs: Dict[int, torch.Tensor],
    labels: torch.Tensor,
    layers: Sequence[int],
) -> tuple[torch.Tensor, Dict[str, float]]:
    if not layers:
        zero = labels.new_zeros((), dtype=torch.float32)
        return zero, {
            "moe_reconstruction_mse": 0.0,
            "moe_reconstruction_rmse": 0.0,
            "moe_reconstruction_cosine": 0.0,
        }
    mask = labels != -100
    losses: List[torch.Tensor] = []
    cosines: List[torch.Tensor] = []
    for layer_idx in layers:
        if layer_idx not in student_outputs or layer_idx not in teacher_outputs:
            raise RuntimeError(f"missing captured MoE output for layer {layer_idx}")
        student = student_outputs[layer_idx]
        teacher = teacher_outputs[layer_idx]
        if student.shape != teacher.shape or student.shape[:2] != (labels.shape[0], labels.shape[1] + 1):
            raise ValueError(
                f"MoE reconstruction shape mismatch at layer {layer_idx}: "
                f"student={tuple(student.shape)} teacher={tuple(teacher.shape)} labels={tuple(labels.shape)}"
            )
        student_rows = student[:, :-1].float()[mask]
        teacher_rows = teacher[:, :-1].float()[mask]
        if student_rows.numel() == 0:
            continue
        losses.append(F.mse_loss(student_rows, teacher_rows))
        cosines.append(F.cosine_similarity(student_rows, teacher_rows, dim=-1).mean())
    if not losses:
        zero = labels.new_zeros((), dtype=torch.float32)
        return zero, {
            "moe_reconstruction_mse": 0.0,
            "moe_reconstruction_rmse": 0.0,
            "moe_reconstruction_cosine": 0.0,
        }
    loss = torch.stack(losses).mean()
    cosine = torch.stack(cosines).mean()
    return loss, {
        "moe_reconstruction_mse": float(loss.detach().cpu()),
        "moe_reconstruction_rmse": float(loss.detach().sqrt().cpu()),
        "moe_reconstruction_cosine": float(cosine.detach().cpu()),
    }


def parse_k_curriculum(spec: str, teacher_top_k: int, final_top_k: int) -> List[int]:
    normalized = str(spec).strip().lower()
    if normalized in {"", "none", "off"}:
        return [int(final_top_k)]
    try:
        schedule = [int(value.strip()) for value in normalized.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid student K curriculum {spec!r}") from exc
    expected = [int(teacher_top_k), 4, int(final_top_k)]
    if schedule != expected or expected != [8, 4, 2]:
        raise ValueError("the optional student K curriculum must be exactly 8,4,2")
    return schedule


def build_k_curriculum_plan(
    schedule: Sequence[int], reference_target_steps: int
) -> Dict[str, Any]:
    if (
        isinstance(reference_target_steps, bool)
        or not isinstance(reference_target_steps, int)
        or reference_target_steps < len(schedule)
        or not schedule
    ):
        raise ValueError("curriculum reference target must cover every K stage")
    return {
        "schedule": [int(value) for value in schedule],
        "reference_target_steps": int(reference_target_steps),
    }


def validate_k_curriculum_plan(
    plan: Mapping[str, Any], args: Mapping[str, Any]
) -> Dict[str, Any]:
    if set(plan) != {"schedule", "reference_target_steps"}:
        raise ValueError("Stage B curriculum plan is missing or incomplete")
    expected_schedule = parse_k_curriculum(
        str(args["student_k_curriculum"]),
        int(args["teacher_top_k"]),
        int(args["student_top_k"]),
    )
    normalized = build_k_curriculum_plan(
        plan["schedule"], plan["reference_target_steps"]
    )
    if normalized["schedule"] != expected_schedule:
        raise ValueError("Stage B curriculum plan disagrees with current K contract")
    return normalized


def student_k_for_step(
    step: int, reference_target_steps: int, schedule: Sequence[int]
) -> int:
    if reference_target_steps <= 0 or step < 1 or not schedule:
        raise ValueError("step must be within a positive training schedule")
    if reference_target_steps < len(schedule):
        raise ValueError("distill steps must cover every K curriculum stage")
    if step > reference_target_steps:
        return int(schedule[-1])
    stage = min(
        len(schedule) - 1,
        (step - 1) * len(schedule) // reference_target_steps,
    )
    return int(schedule[stage])


def effective_top_k(model) -> int:
    return int(getattr(model.config, "num_experts_per_tok", -1))


def runtime_top_k_values(model) -> Dict[str, int]:
    values: Dict[str, int] = {}
    runtime_objects = [
        ("config", getattr(model, "config", None)),
        ("model", model),
        ("model.model", getattr(model, "model", None)),
    ]
    for name, runtime_object in runtime_objects:
        if runtime_object is None:
            continue
        for attr in ("top_k", "num_experts_per_tok", "k"):
            value = getattr(runtime_object, attr, None)
            if isinstance(value, int) and not isinstance(value, bool):
                values[f"{name}.{attr}"] = value
    for layer_idx, layer in enumerate(getattr(getattr(model, "model", None), "layers", [])):
        mlp = getattr(layer, "mlp", None)
        for object_name, runtime_object in (
            (f"layer_{layer_idx}.mlp", mlp),
            (f"layer_{layer_idx}.mlp.gate", getattr(mlp, "gate", None)),
        ):
            if runtime_object is None:
                continue
            for attr in ("top_k", "num_experts_per_tok", "k"):
                value = getattr(runtime_object, attr, None)
                if isinstance(value, int) and not isinstance(value, bool):
                    values[f"{object_name}.{attr}"] = value
    return values


def assert_final_top_k(model) -> None:
    values = runtime_top_k_values(model)
    mismatches = {name: value for name, value in values.items() if value != FINAL_STUDENT_TOP_K}
    if not values or mismatches:
        raise ValueError(
            f"final student inference top_k must be {FINAL_STUDENT_TOP_K}; "
            f"runtime values={values}"
        )


def enforce_final_top_k(model, aux_coef: float) -> Dict[str, Any]:
    routing_meta = set_olmoe_runtime_routing(model, FINAL_STUDENT_TOP_K, aux_coef)
    assert_final_top_k(model)
    return routing_meta


def validate_development_path(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    lowered = str(resolved).lower()
    if "sealed" in lowered or "synthetic" in lowered:
        raise ValueError(f"{label} must be a development-only real-data path, got {resolved}")
    return resolved


def validate_development_rows(rows: Sequence[Dict[str, Any]], label: str) -> None:
    for row_idx, row in enumerate(rows):
        values = [
            str(row.get(key, ""))
            for key in ("split", "partition", "source_split", "data_policy")
        ]
        normalized = [
            value.lower().replace("-", "_").replace("/", "_").split("_")
            for value in values
        ]
        if any("sealed" in value.lower() or "synthetic" in value.lower() for value in values) or any(
            "test" in segments for segments in normalized
        ):
            raise ValueError(
                f"{label} row {row_idx} is not development-only: {' '.join(values)!r}"
            )


def validate_stage_b_args(args: argparse.Namespace) -> List[int]:
    if int(args.teacher_top_k) != 8:
        raise ValueError("Stage B requires a Top-8 teacher")
    if int(args.student_top_k) != FINAL_STUDENT_TOP_K:
        raise ValueError("Stage B requires a final Top-2 student")
    if int(args.distill_steps) <= 0:
        raise ValueError("distill steps must be positive")
    if int(args.checkpoint_every_steps) < 0:
        raise ValueError("checkpoint every steps must be non-negative")
    for name in (
        "distill_logit_coef",
        "router_distill_coef",
        "distill_hidden_coef",
        "moe_reconstruction_coef",
        "text_replay_coef",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("distill_temperature", "router_distill_temperature"):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name} must be positive")
    schedule = parse_k_curriculum(args.student_k_curriculum, args.teacher_top_k, args.student_top_k)
    if int(args.distill_steps) < len(schedule):
        raise ValueError("distill steps must cover every K curriculum stage")
    if float(args.router_distill_coef) > 0.0 and not bool(args.train_router_gates):
        raise ValueError("router distillation requires --train-router-gates")
    if float(args.moe_reconstruction_coef) > 0.0 and not (
        bool(args.train_router_gates) or bool(args.train_gamma_scale)
    ):
        raise ValueError("MoE reconstruction requires trainable router gates or gamma scales")
    if float(args.text_replay_coef) > 0.0 and not str(args.text_replay_manifest).strip():
        raise ValueError("text replay requires --text-replay-manifest")
    validate_development_path(args.data_dir, "data directory")
    if str(args.text_replay_manifest).strip():
        validate_development_path(args.text_replay_manifest, "text replay manifest")
    if str(args.resume_checkpoint).strip():
        validate_development_path(args.resume_checkpoint, "resume checkpoint")
    return schedule


def teacher_student_metrics(
    student,
    teacher,
    tokenizer,
    blocks: Sequence[Dict[str, Any]],
    args,
    out_dir: Path,
    exp_id: str,
) -> Dict[str, Any]:
    device = next(student.parameters()).device
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    student.eval()
    teacher.eval()
    loss_sum = 0.0
    token_count = 0
    correct = 0
    total = 0
    kl_values: List[float] = []
    router_kl_values: List[float] = []
    reconstruction_values: Dict[str, List[float]] = {
        "moe_reconstruction_mse": [],
        "moe_reconstruction_rmse": [],
        "moe_reconstruction_cosine": [],
    }
    top1_agree = 0
    top1_total = 0
    coverage_values: List[float] = []
    router_accum: Dict[str, Any] | None = None
    moe_layers = parse_layers(args.moe_reconstruction_layers, len(student.model.layers))
    student_capture = MoEOutputCapture(student, moe_layers)
    teacher_capture = MoEOutputCapture(teacher, moe_layers)
    try:
        with torch.no_grad():
            for start in range(0, len(blocks), args.eval_batch_size):
                batch_rows = list(blocks[start:start + args.eval_batch_size])
                batch = tensorize_blocks(batch_rows, device, args.max_length, pad_id)
                teacher_capture.clear()
                student_capture.clear()
                teacher_out = teacher(**batch, output_router_logits=True, return_dict=True)
                student_out = student(**batch, output_router_logits=True, return_dict=True)
                shift_logits = student_out.logits[:, :-1].float()
                shift_labels = batch["labels"][:, 1:]
                mask = shift_labels != -100
                batch_loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                ntok = int(mask.sum().item())
                loss_sum += float(batch_loss.cpu())
                token_count += ntok
                preds = shift_logits.argmax(dim=-1)
                correct += int(((preds == shift_labels) & mask).sum().item())
                total += ntok
                kl = logits_kl(
                    student_out.logits[:, :-1],
                    teacher_out.logits[:, :-1],
                    shift_labels,
                    args.distill_temperature,
                )
                kl_values.append(float(kl.detach().cpu()))
                student_rows = masked_token_rows(student_out.logits[:, :-1], shift_labels)
                teacher_rows = masked_token_rows(teacher_out.logits[:, :-1], shift_labels)
                if student_rows.numel() > 0:
                    top1_agree += int(
                        (student_rows.argmax(dim=-1) == teacher_rows.argmax(dim=-1)).sum().item()
                    )
                    top1_total += int(student_rows.shape[0])
                token_mask = batch.get("attention_mask", batch["labels"] != -100)
                rkl = router_kl(
                    student_out.router_logits,
                    teacher_out.router_logits,
                    args.router_distill_temperature,
                    token_mask=token_mask,
                    expected_num_experts=int(student.config.num_experts),
                )
                router_kl_values.append(float(rkl.detach().cpu()))
                student_k = effective_top_k(student)
                coverage_values.extend(
                    router_mass_coverage(
                        student_out.router_logits,
                        teacher_out.router_logits,
                        effective_top_k(teacher),
                        student_k,
                    )
                )
                _, reconstruction = moe_output_reconstruction(
                    student_capture.outputs,
                    teacher_capture.outputs,
                    shift_labels,
                    moe_layers,
                )
                for name, value in reconstruction.items():
                    reconstruction_values[name].append(value)
                if router_accum is None:
                    router_accum = router_metrics(
                        student_out,
                        student_k,
                        int(student.config.num_experts),
                        args.capacity_factor,
                    )
    finally:
        student_capture.close()
        teacher_capture.close()
    loss_mean = loss_sum / max(1, token_count)
    metrics = {
        "experiment_id": exp_id,
        "eval_blocks": len(blocks),
        "eval_tokens": token_count,
        "loss": float(loss_mean),
        "perplexity": float(math.exp(min(20.0, loss_mean))),
        "next_token_accuracy": float(correct / max(1, total)),
        "teacher_student_kl": float(sum(kl_values) / max(1, len(kl_values))),
        "teacher_student_top1_agreement": float(top1_agree / max(1, top1_total)),
        "router_kl": float(sum(router_kl_values) / max(1, len(router_kl_values))),
        "router_distribution_num_experts": int(student.config.num_experts),
        "teacher_topk_mass_on_student_topk": float(
            sum(coverage_values) / max(1, len(coverage_values))
        ),
        "effective_teacher_top_k": effective_top_k(teacher),
        "effective_student_top_k": effective_top_k(student),
        "final_inference_top_k": effective_top_k(student),
        "moe_reconstruction_layers": moe_layers,
        **{
            name: float(sum(values) / max(1, len(values)))
            for name, values in reconstruction_values.items()
        },
        **(router_accum or {}),
        **cuda_metrics(),
    }
    save_json(out_dir / exp_id / "metrics.json", metrics)
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in [
                    "experiment_id",
                    "perplexity",
                    "next_token_accuracy",
                    "teacher_student_kl",
                    "router_kl",
                    "moe_reconstruction_rmse",
                    "effective_student_top_k",
                ]
            },
            sort_keys=True,
        )
    )
    return metrics

def router_mass_coverage(student_router: Any, teacher_router: Any, teacher_top_k: int, student_top_k: int) -> List[float]:
    s_layers = list(student_router) if not torch.is_tensor(student_router) else [student_router]
    t_layers = list(teacher_router) if not torch.is_tensor(teacher_router) else [teacher_router]
    vals: List[float] = []
    for s, t in zip(s_layers, t_layers):
        s_probs = torch.softmax(s.float(), dim=-1)
        t_probs = torch.softmax(t.float(), dim=-1)
        _, s_idx = torch.topk(s_probs, int(student_top_k), dim=-1)
        teacher_mass_on_student = t_probs.gather(-1, s_idx).sum(dim=-1)
        vals.append(float(teacher_mass_on_student.mean().detach().cpu()))
    return vals


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_checkpoint_provenance(
    args,
    resume_identity: Mapping[str, Any] | None,
    run_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {
        **dict(run_provenance),
        "stage": "B",
        "checkpoint_version": CHECKPOINT_VERSION,
        "base_model": str(args.base_model),
        "development_data_only": True,
        "teacher_top_k": int(args.teacher_top_k),
        "final_inference_top_k": FINAL_STUDENT_TOP_K,
        "resume_checkpoint": None,
    }
    if resume_identity is not None:
        provenance.update(
            {
                "resume_checkpoint": str(resume_identity["path"]),
                "resume_checkpoint_sha256": str(resume_identity["sha256"]),
                "resume_checkpoint_size_bytes": int(
                    resume_identity["size_bytes"]
                ),
                "resume_checkpoint_run_identity": dict(
                    resume_identity["run_identity"]
                ),
                "resume_checkpoint_source_commit_sha": str(
                    resume_identity["source_commit_sha"]
                ),
            }
        )
    return provenance


def resume_arg_contract(args: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [field for field in RESUME_ARG_FIELDS if field not in args]
    if missing:
        raise ValueError(
            f"Stage B v2 resume args are incomplete: missing={missing}"
        )
    return {field: args[field] for field in RESUME_ARG_FIELDS}


def trainable_contract(trainable: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [field for field in TRAINABLE_CONTRACT_FIELDS if field not in trainable]
    if missing:
        raise ValueError(
            f"Stage B v2 trainable contract is incomplete: missing={missing}"
        )
    return {field: trainable[field] for field in TRAINABLE_CONTRACT_FIELDS}


def optimizer_group_contract(
    optimizer: torch.optim.Optimizer,
    trainable: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    trainable_groups = trainable.get("groups")
    if (
        not isinstance(trainable_groups, list)
        or len(trainable_groups) != len(optimizer.param_groups)
    ):
        raise ValueError("optimizer groups do not match named trainable groups")
    groups: List[Dict[str, Any]] = []
    for index, (group, runtime_group, trainable_group) in enumerate(zip(
        optimizer.state_dict()["param_groups"],
        optimizer.param_groups,
        trainable_groups,
    )):
        expected_parameters = trainable_group.get("params")
        parameter_names = trainable_group.get("parameter_names")
        if (
            not isinstance(expected_parameters, list)
            or not isinstance(parameter_names, list)
            or len(parameter_names) != len(runtime_group["params"])
            or [id(value) for value in expected_parameters]
            != [id(value) for value in runtime_group["params"]]
        ):
            raise ValueError(
                f"optimizer group {index} parameter order disagrees with trainable metadata"
            )
        config = {
            key: value
            for key, value in group.items()
            if key not in {"params", "param_names"}
        }
        config["parameter_count"] = len(group["params"])
        config["parameter_names"] = list(parameter_names)
        groups.append(config)
    try:
        json.dumps(groups, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("optimizer group config is not serializable") from exc
    return groups


def build_resume_contract(
    args: Mapping[str, Any],
    trainable: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    data_contract: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    curriculum_plan: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    args_contract = resume_arg_contract(args)
    trainable_value = trainable_contract(trainable)
    optimizer_groups = optimizer_group_contract(optimizer, trainable)
    data_value = _stable_json_value(dict(data_contract))
    model_value = _stable_json_value(dict(model_identity))
    if curriculum_plan is None:
        schedule = parse_k_curriculum(
            str(args["student_k_curriculum"]),
            int(args["teacher_top_k"]),
            int(args["student_top_k"]),
        )
        curriculum_plan = build_k_curriculum_plan(
            schedule, int(args["distill_steps"])
        )
    curriculum_value = validate_k_curriculum_plan(curriculum_plan, args)
    contract = {
        "schema_version": CHECKPOINT_VERSION,
        "args": args_contract,
        "trainable": trainable_value,
        "optimizer_groups": optimizer_groups,
        "data": data_value,
        "base_model_identity": model_value,
        "curriculum_plan": curriculum_value,
    }
    contract["args_sha256"] = canonical_sha256(args_contract)
    contract["trainable_sha256"] = canonical_sha256(trainable_value)
    contract["optimizer_groups_sha256"] = canonical_sha256(optimizer_groups)
    contract["data_sha256"] = canonical_sha256(data_value)
    contract["base_model_identity_sha256"] = canonical_sha256(model_value)
    contract["curriculum_plan_sha256"] = canonical_sha256(curriculum_value)
    return contract


def validate_resume_contract(
    state: Mapping[str, Any],
    current_args: Mapping[str, Any],
    current_trainable: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    current_data_contract: Mapping[str, Any],
    current_model_identity: Mapping[str, Any],
) -> None:
    stored = state.get("resume_contract")
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
    if not isinstance(stored, Mapping) or set(stored) != required:
        raise ValueError("Stage B v2 resume contract is missing or incomplete")
    if stored["schema_version"] != CHECKPOINT_VERSION:
        raise ValueError("Stage B v2 resume contract schema mismatch")
    checkpoint_plan = state.get("curriculum_plan")
    if (
        not isinstance(checkpoint_plan, Mapping)
        or dict(checkpoint_plan) != stored["curriculum_plan"]
        or canonical_sha256(checkpoint_plan) != stored["curriculum_plan_sha256"]
    ):
        raise ValueError(
            "Stage B v2 checkpoint curriculum plan disagrees with resume contract"
        )
    expected = build_resume_contract(
        current_args,
        current_trainable,
        optimizer,
        current_data_contract,
        current_model_identity,
        curriculum_plan=checkpoint_plan,
    )
    for field in (
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
    ):
        if stored[field] != expected[field]:
            raise ValueError(
                f"Stage B v2 resume contract mismatch for {field}"
            )
    completed_steps = state.get("completed_steps")
    target_steps = current_args.get("distill_steps")
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 0
        or isinstance(target_steps, bool)
        or not isinstance(target_steps, int)
        or target_steps <= completed_steps
        or target_steps < int(stored["curriculum_plan"]["reference_target_steps"])
    ):
        raise ValueError(
            "Stage B v2 resume requires target distill_steps > completed_steps"
        )
    checkpoint_args = state.get("args")
    if (
        not isinstance(checkpoint_args, Mapping)
        or resume_arg_contract(checkpoint_args) != stored["args"]
    ):
        raise ValueError("Stage B v2 checkpoint args disagree with resume contract")
    checkpoint_trainable = state.get("trainable_meta")
    if (
        not isinstance(checkpoint_trainable, Mapping)
        or trainable_contract(checkpoint_trainable) != stored["trainable"]
    ):
        raise ValueError(
            "Stage B v2 checkpoint trainable_meta disagrees with resume contract"
        )


def validate_resume_checkpoint_provenance(
    state: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    provenance = state.get("provenance")
    required = {
        "run_uuid",
        "source_commit_sha",
        "runai_job_name",
        "runai_project",
        "producer_code",
        "dataset_split_provenance",
        "dataset_split_provenance_sha256",
        "sealed_evidence_used",
        "synthetic_evidence_used",
    }
    if not isinstance(provenance, Mapping):
        raise ValueError("Stage B v2 resume checkpoint provenance is missing")
    missing = sorted(required - set(provenance))
    if missing:
        raise ValueError(
            f"Stage B v2 resume checkpoint provenance is incomplete: missing={missing}"
        )
    source_commit = provenance["source_commit_sha"]
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise ValueError(
            "Stage B v2 resume checkpoint provenance has invalid source commit"
        )
    for field in ("runai_job_name", "runai_project"):
        if not isinstance(provenance[field], str) or not provenance[field].strip():
            raise ValueError(
                f"Stage B v2 resume checkpoint provenance has invalid {field}"
            )
    producer = provenance["producer_code"]
    if (
        not isinstance(producer, Mapping)
        or set(producer) != {"path", "sha256"}
        or not isinstance(producer["path"], str)
        or not producer["path"].strip()
        or not isinstance(producer["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", producer["sha256"]) is None
    ):
        raise ValueError(
            "Stage B v2 resume checkpoint provenance has invalid producer code"
        )
    expected_identity = build_checkpoint_run_identity(provenance)
    stored_identity = state.get("run_identity")
    if not isinstance(stored_identity, Mapping):
        raise ValueError("Stage B v2 resume checkpoint run_identity is missing")
    if dict(stored_identity) != expected_identity:
        raise ValueError(
            "Stage B v2 resume checkpoint run_identity disagrees with provenance"
        )
    return dict(provenance), expected_identity


def load_resume_checkpoint_once(
    path: Path,
    map_location: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw = path.expanduser()
    if raw.is_symlink():
        raise ValueError("Stage B resume checkpoint cannot be a symlink")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("Stage B resume checkpoint must be a regular file")
    payload = resolved.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        state = torch.load(
            io.BytesIO(payload),
            map_location=map_location,
            weights_only=False,
        )
    except Exception as exc:
        raise ValueError("cannot load Stage B resume checkpoint bytes") from exc
    if not isinstance(state, dict):
        raise ValueError("Stage B resume checkpoint payload must be a mapping")
    return state, {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": len(payload),
    }


def load_runner_resume_checkpoint(
    path: Path,
    map_location: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    state, identity = load_resume_checkpoint_once(path, map_location)
    checkpoint_version = int(state.get("checkpoint_version", 1))
    if checkpoint_version != CHECKPOINT_VERSION:
        raise ValueError(
            "Stage B real runner refuses legacy v1 resume checkpoints; "
            "resume requires a v2 checkpoint with curriculum, optimizer, "
            "data, topology, base-model, and RNG contracts"
        )
    return state, identity


def restore_student_checkpoint(
    student,
    state: Dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    current_args: Dict[str, Any] | None = None,
    current_trainable: Dict[str, Any] | None = None,
    current_data_contract: Dict[str, Any] | None = None,
    current_model_identity: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    output_embeddings = student.get_output_embeddings()
    input_embeddings = student.get_input_embeddings()
    checkpoint_version = int(state.get("checkpoint_version", 1))
    trainable = state.get("trainable_meta")
    verified_provenance: Dict[str, Any] | None = None
    verified_run_identity: Dict[str, Any] | None = None
    if checkpoint_version >= 2:
        if (
            optimizer is None
            or current_args is None
            or current_trainable is None
            or current_data_contract is None
            or current_model_identity is None
        ):
            raise ValueError(
                "Stage B v2 resume requires args, trainable, and optimizer context"
            )
        verified_provenance, verified_run_identity = (
            validate_resume_checkpoint_provenance(state)
        )
        validate_resume_contract(
            state,
            current_args,
            current_trainable,
            optimizer,
            current_data_contract,
            current_model_identity,
        )
        if state.get("optimizer_state") is None:
            raise ValueError("Stage B v2 resume checkpoint is missing optimizer state")
        if not isinstance(state.get("rng_state"), Mapping):
            raise ValueError("Stage B v2 resume checkpoint is missing RNG state")
        if not isinstance(trainable, dict):
            raise ValueError(
                "Stage B v2 resume checkpoint is missing trainable_meta"
            )
        if trainable.get("train_lm_head"):
            if state.get("lm_output_embeddings") is None:
                raise ValueError(
                    "Stage B resume checkpoint is missing output embeddings"
                )
            embeddings_tied = state.get("lm_embeddings_tied")
            if not isinstance(embeddings_tied, bool):
                raise ValueError(
                    "Stage B resume checkpoint is missing tied-embedding metadata"
                )
            runtime_embeddings_tied = lm_embeddings_are_tied(student)
            if embeddings_tied is not runtime_embeddings_tied:
                raise ValueError(
                    "Stage B resume embedding topology disagrees with runtime model"
                )
            if (
                not embeddings_tied
                and state.get("lm_input_embeddings") is None
            ):
                raise ValueError(
                    "Stage B resume checkpoint is missing input embeddings"
                )
        if trainable.get("train_router_gates"):
            router_state = state.get("router_gates")
            expected_router_keys = {
                f"layer_{index}"
                for index in range(len(student.model.layers))
            }
            if (
                not isinstance(router_state, dict)
                or set(router_state) != expected_router_keys
            ):
                raise ValueError(
                    "Stage B resume checkpoint is missing complete router state"
                )
        if (
            trainable.get("train_gamma_scale")
            and state.get("gamma_scale") is None
        ):
            raise ValueError(
                "Stage B resume checkpoint is missing trainable gamma state"
            )
    if state.get("lm_output_embeddings") is not None and output_embeddings is not None:
        output_embeddings.load_state_dict(state["lm_output_embeddings"])
    if state.get("lm_input_embeddings") is not None and input_embeddings is not None:
        input_embeddings.load_state_dict(state["lm_input_embeddings"])
    for layer_idx, layer in enumerate(student.model.layers):
        layer_state = state.get("router_gates", {}).get(f"layer_{layer_idx}")
        if layer_state is not None:
            layer.mlp.gate.load_state_dict(layer_state)
    gamma_state = state.get("gamma_scale")
    if gamma_state is not None:
        model_layers = list(student.model.layers)
        if len(gamma_state) != len(model_layers):
            raise ValueError(
                f"gamma scale checkpoint has {len(gamma_state)} layers, expected {len(model_layers)}"
            )
        for layer, value in zip(model_layers, gamma_state):
            gamma = getattr(layer.mlp, "gamma_scale", None)
            if gamma is None:
                raise ValueError("checkpoint contains gamma scales but the student has no gamma_scale")
            gamma.data.copy_(torch.as_tensor(value, device=gamma.device, dtype=gamma.dtype))
    optimizer_restored = optimizer is not None and state.get("optimizer_state") is not None
    if optimizer_restored:
        optimizer.load_state_dict(state["optimizer_state"])
    if checkpoint_version >= 2:
        restore_rng_state(state["rng_state"])
    return {
        "checkpoint_version": checkpoint_version,
        "completed_steps": int(state.get("completed_steps", 0)),
        "optimizer_state_restored": bool(optimizer_restored),
        "curriculum_plan": (
            dict(state["curriculum_plan"])
            if checkpoint_version >= 2
            else None
        ),
        "run_identity": verified_run_identity,
        "source_commit_sha": (
            verified_provenance["source_commit_sha"]
            if verified_provenance is not None
            else None
        ),
        "legacy_v1_behavior": (
            "partial_model_state_without_optimizer_or_rng_contract"
            if checkpoint_version == 1
            else None
        ),
    }


def train_student(
    exp_id: str,
    teacher,
    tokenizer,
    gamma: Sequence[float],
    train_blocks,
    eval_blocks,
    args,
    out_dir: Path,
    use_distill: bool,
    replay_blocks: Sequence[Dict[str, Any]] | None = None,
    stage_b_run_provenance: Mapping[str, Any] | None = None,
    data_contract: Mapping[str, Any] | None = None,
    expected_model_identity: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    student, _, meta = load_model(
        args.base_model,
        args.student_top_k,
        args.aux_coef,
        gamma=gamma,
        capacity_factor=args.capacity_factor,
        pre_routing_identity_fn=lambda model: base_model_identity(
            model, args.base_model
        ),
    )
    device = next(student.parameters()).device
    current_model_identity = meta.pop("pre_routing_model_identity", None)
    if not isinstance(current_model_identity, Mapping):
        raise ValueError("student pre-routing base-model identity is missing")
    if (
        expected_model_identity is not None
        and dict(current_model_identity) != dict(expected_model_identity)
    ):
        raise ValueError("student base-model identity differs from loaded teacher")
    if data_contract is None:
        raise ValueError("Stage B runner requires a bound data contract")
    trainable = set_trainable(
        student,
        args.train_lm_head,
        args.train_router_gates,
        args.train_gamma_scale,
    )
    groups = []
    for group in trainable["groups"]:
        if group["name"] == "router_gates":
            lr = args.router_learning_rate
        elif group["name"] == "gamma_scale":
            lr = args.gamma_learning_rate
        else:
            lr = args.learning_rate
        groups.append({
            "params": group["params"],
            "param_names": group["parameter_names"],
            "lr": lr,
            "weight_decay": args.weight_decay,
        })
    if not groups:
        raise RuntimeError(
            "No trainable parameters selected; enable --train-lm-head, "
            "--train-router-gates, or --train-gamma-scale"
        )
    optimizer = torch.optim.AdamW(groups)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    teacher.eval()
    student.train()
    if effective_top_k(teacher) != int(args.teacher_top_k):
        raise RuntimeError(
            f"teacher runtime top_k mismatch: expected={args.teacher_top_k} actual={effective_top_k(teacher)}"
        )

    schedule = (
        parse_k_curriculum(args.student_k_curriculum, args.teacher_top_k, args.student_top_k)
        if use_distill
        else [FINAL_STUDENT_TOP_K]
    )
    curriculum_plan = build_k_curriculum_plan(
        schedule, int(args.distill_steps)
    )
    resume_path = None
    if use_distill and str(args.resume_checkpoint).strip():
        raw_resume_path = Path(str(args.resume_checkpoint)).expanduser()
        if raw_resume_path.is_symlink():
            raise ValueError("Stage B resume checkpoint cannot be a symlink")
        resume_path = validate_development_path(args.resume_checkpoint, "resume checkpoint")
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
    if not isinstance(stage_b_run_provenance, Mapping):
        raise ValueError("Stage-B training requires verified run provenance")
    resume_identity: Dict[str, Any] | None = None
    resume_meta = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "completed_steps": 0,
        "optimizer_state_restored": False,
    }
    if resume_path is not None:
        state, resume_identity = load_runner_resume_checkpoint(
            resume_path,
            device,
        )
        resume_meta = restore_student_checkpoint(
            student,
            state,
            optimizer=optimizer,
            current_args=vars(args),
            current_trainable=trainable,
            current_data_contract=dict(data_contract),
            current_model_identity=current_model_identity,
        )
        resume_identity.update(
            {
                "run_identity": resume_meta["run_identity"],
                "source_commit_sha": resume_meta["source_commit_sha"],
            }
        )
        curriculum_plan = dict(resume_meta["curriculum_plan"])
    provenance = make_checkpoint_provenance(
        args, resume_identity, stage_b_run_provenance
    )
    provenance["resume_state"] = resume_meta
    start_step = int(resume_meta["completed_steps"])
    if start_step >= int(args.distill_steps):
        raise ValueError(
            f"resume checkpoint already completed {start_step} steps, "
            f"but --distill-steps is {args.distill_steps}"
        )

    log_path = out_dir / exp_id / "train_metrics.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and resume_path is None:
        log_path.unlink()
    hidden_layers: List[int] = []
    moe_layers = (
        parse_layers(args.moe_reconstruction_layers, len(student.model.layers))
        if use_distill and args.moe_reconstruction_coef > 0.0
        else []
    )
    teacher_capture = MoEOutputCapture(teacher, moe_layers)
    student_capture = MoEOutputCapture(student, moe_layers)
    component_sums = {
        "loss": 0.0,
        "ce_loss": 0.0,
        "distill_logit_loss": 0.0,
        "router_distill_loss": 0.0,
        "hidden_distill_loss": 0.0,
        "moe_reconstruction_loss": 0.0,
        "text_replay_loss": 0.0,
    }
    steps_run = 0
    current_k = -1
    replay_blocks = list(replay_blocks or [])
    try:
        for step in range(start_step + 1, args.distill_steps + 1):
            desired_k = student_k_for_step(
                step,
                int(curriculum_plan["reference_target_steps"]),
                curriculum_plan["schedule"],
            )
            if desired_k != current_k:
                set_olmoe_runtime_routing(student, desired_k, args.aux_coef)
                current_k = effective_top_k(student)
                if current_k != desired_k:
                    raise RuntimeError(
                        f"student runtime top_k mismatch: expected={desired_k} actual={current_k}"
                    )
            batch_rows = sample_cycle(
                train_blocks,
                step * args.train_batch_size,
                args.train_batch_size,
            )
            batch = tensorize_blocks(batch_rows, device, args.max_length, pad_id)
            need_hidden = use_distill and args.distill_hidden_coef > 0.0
            teacher_capture.clear()
            student_capture.clear()
            with torch.no_grad():
                teacher_out = teacher(
                    **batch,
                    output_router_logits=True,
                    output_hidden_states=need_hidden,
                    return_dict=True,
                )
            student_out = student(
                **batch,
                output_router_logits=True,
                output_hidden_states=need_hidden,
                return_dict=True,
            )
            ce = student_out.loss
            labels = batch["labels"][:, 1:]
            logit_loss = (
                logits_kl(
                    student_out.logits[:, :-1],
                    teacher_out.logits[:, :-1],
                    labels,
                    args.distill_temperature,
                )
                if use_distill and args.distill_logit_coef > 0.0
                else ce.new_zeros(())
            )
            token_mask = batch.get("attention_mask", batch["labels"] != -100)
            router_loss = (
                router_kl(
                    student_out.router_logits,
                    teacher_out.router_logits,
                    args.router_distill_temperature,
                    token_mask=token_mask,
                    expected_num_experts=int(student.config.num_experts),
                )
                if use_distill and args.router_distill_coef > 0.0
                else ce.new_zeros(())
            )
            if need_hidden:
                if not hidden_layers:
                    hidden_layers = parse_layers(
                        args.distill_hidden_layers,
                        len(student_out.hidden_states),
                    )
                hidden_loss = hidden_match(
                    student_out.hidden_states,
                    teacher_out.hidden_states,
                    labels,
                    hidden_layers,
                    args.distill_hidden_mode,
                )
            else:
                hidden_loss = ce.new_zeros(())
            if moe_layers:
                reconstruction_loss, reconstruction = moe_output_reconstruction(
                    student_capture.outputs,
                    teacher_capture.outputs,
                    labels,
                    moe_layers,
                )
            else:
                reconstruction_loss = ce.new_zeros(())
                reconstruction = {
                    "moe_reconstruction_mse": 0.0,
                    "moe_reconstruction_rmse": 0.0,
                    "moe_reconstruction_cosine": 0.0,
                }
            replay_loss = ce.new_zeros(())
            replay_tokens = 0
            if use_distill and args.text_replay_coef > 0.0:
                if not replay_blocks:
                    raise RuntimeError("text replay coefficient is positive but no replay rows were loaded")
                replay_rows = sample_cycle(
                    replay_blocks,
                    step * args.train_batch_size,
                    args.train_batch_size,
                )
                replay_batch = tensorize_blocks(
                    replay_rows,
                    device,
                    args.max_length,
                    pad_id,
                )
                replay_out = student(
                    **replay_batch,
                    output_router_logits=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                replay_labels = replay_batch["labels"][:, 1:]
                replay_loss = text_replay_loss(replay_out.logits[:, :-1], replay_labels)
                replay_tokens = int((replay_labels != -100).sum().item())

            loss = ce
            if use_distill:
                loss = (
                    loss
                    + float(args.distill_logit_coef) * logit_loss
                    + float(args.router_distill_coef) * router_loss
                    + float(args.distill_hidden_coef) * hidden_loss
                    + float(args.moe_reconstruction_coef) * reconstruction_loss
                    + float(args.text_replay_coef) * replay_loss
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in student.parameters() if parameter.requires_grad],
                args.grad_clip,
            )
            optimizer.step()
            row = {
                "experiment_id": exp_id,
                "stage": "B" if use_distill else "CE_control",
                "step": step,
                "loss": float(loss.detach().float().cpu()),
                "ce_loss": float(ce.detach().float().cpu()),
                "distill_logit_loss": float(logit_loss.detach().float().cpu()),
                "router_distill_loss": float(router_loss.detach().float().cpu()),
                "hidden_distill_loss": float(hidden_loss.detach().float().cpu()),
                "moe_reconstruction_loss": float(reconstruction_loss.detach().float().cpu()),
                "text_replay_loss": float(replay_loss.detach().float().cpu()),
                "text_replay_tokens": replay_tokens,
                "distill_logit_coef": float(args.distill_logit_coef if use_distill else 0.0),
                "router_distill_coef": float(args.router_distill_coef if use_distill else 0.0),
                "distill_hidden_coef": float(args.distill_hidden_coef if use_distill else 0.0),
                "moe_reconstruction_coef": float(
                    args.moe_reconstruction_coef if use_distill else 0.0
                ),
                "text_replay_coef": float(args.text_replay_coef if use_distill else 0.0),
                "effective_teacher_top_k": effective_top_k(teacher),
                "effective_student_top_k": current_k,
                "student_k_schedule": schedule,
                "router_distribution_num_experts": int(student.config.num_experts),
                "distill_hidden_layers_resolved": hidden_layers,
                "moe_reconstruction_layers": moe_layers,
                "gamma_learning_rate": float(args.gamma_learning_rate),
                "gamma_scale_values": gamma_scale_values(student),
                "checkpoint_provenance": provenance,
                **reconstruction,
                **meta,
                **{key: value for key, value in trainable.items() if key != "groups"},
                **cuda_metrics(),
            }
            for name in component_sums:
                component_sums[name] += float(row[name])
            steps_run += 1
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if (
                use_distill
                and int(args.checkpoint_every_steps) > 0
                and step % int(args.checkpoint_every_steps) == 0
                and step < int(args.distill_steps)
            ):
                periodic_path = (
                    out_dir
                    / exp_id
                    / f"checkpoint_step_{step:08d}.pt"
                )
                save_student_checkpoint(
                    student,
                    periodic_path,
                    trainable,
                    vars(args),
                    {"last_train_row": row},
                    provenance=provenance,
                    optimizer=optimizer,
                    completed_steps=step,
                    data_contract=dict(data_contract),
                    model_identity=current_model_identity,
                    curriculum_plan=curriculum_plan,
                    allow_intermediate_routing=True,
                )
            if step == start_step + 1 or step % args.log_every_steps == 0 or step == args.distill_steps:
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in [
                                "experiment_id",
                                "step",
                                "loss",
                                "ce_loss",
                                "distill_logit_loss",
                                "router_distill_loss",
                                "moe_reconstruction_loss",
                                "text_replay_loss",
                                "effective_student_top_k",
                                "cuda_memory_reserved_gb",
                            ]
                        },
                        sort_keys=True,
                    )
                )
    finally:
        teacher_capture.close()
        student_capture.close()

    final_routing_meta = enforce_final_top_k(student, args.aux_coef)
    metrics = teacher_student_metrics(
        student,
        teacher,
        tokenizer,
        eval_blocks[: args.text_eval_blocks],
        args,
        out_dir,
        exp_id,
    )
    metrics.update(
        {
            "stage": "B" if use_distill else "CE_control",
            "training_component_loss_mean": {
                name: value / max(1, steps_run) for name, value in component_sums.items()
            },
            "training_steps_this_run": steps_run,
            "training_completed_steps": int(args.distill_steps),
            "effective_teacher_top_k": effective_top_k(teacher),
            "effective_student_top_k": effective_top_k(student),
            "final_inference_top_k": FINAL_STUDENT_TOP_K,
            "student_k_schedule": schedule,
            "final_routing_meta": final_routing_meta,
            "checkpoint_provenance": provenance,
        }
    )
    checkpoint_path = out_dir / exp_id / "checkpoint_final.pt"
    save_student_checkpoint(
        student,
        checkpoint_path,
        trainable,
        vars(args),
        metrics,
        provenance=provenance,
        optimizer=optimizer,
        completed_steps=args.distill_steps,
        data_contract=dict(data_contract),
        model_identity=current_model_identity,
        curriculum_plan=curriculum_plan,
    )
    provenance.update(
        {
            "saved_checkpoint": str(checkpoint_path.resolve()),
            "saved_checkpoint_sha256": sha256_file(checkpoint_path),
            "saved_checkpoint_size_bytes": checkpoint_path.stat().st_size,
        }
    )
    metrics["checkpoint_provenance"] = provenance
    save_json(out_dir / exp_id / "metrics.json", metrics)
    cleanup(student)
    return {
        "metrics": metrics,
        "trainable": {key: value for key, value in trainable.items() if key != "groups"},
        "checkpoint_provenance": provenance,
    }


def save_student_checkpoint(
    student,
    path: Path,
    trainable: Dict[str, Any],
    args_dict: Dict[str, Any],
    metrics: Dict[str, Any],
    provenance: Dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    completed_steps: int | None = None,
    data_contract: Mapping[str, Any] | None = None,
    model_identity: Mapping[str, Any] | None = None,
    curriculum_plan: Mapping[str, Any] | None = None,
    allow_intermediate_routing: bool = False,
) -> None:
    if int(args_dict.get("student_top_k", FINAL_STUDENT_TOP_K)) != FINAL_STUDENT_TOP_K:
        raise ValueError("checkpoint args must declare a final Top-2 student")
    runtime_top_k = effective_top_k(student)
    if not allow_intermediate_routing:
        try:
            assert_final_top_k(student)
        except ValueError as exc:
            raise ValueError(
                "refusing to checkpoint a student whose final inference top_k is not 2"
            ) from exc
    if not isinstance(provenance, Mapping):
        raise ValueError("Stage-B checkpoint requires verified run provenance")
    checkpoint_provenance = dict(provenance)
    run_identity = build_checkpoint_run_identity(checkpoint_provenance)
    if optimizer is None:
        raise ValueError(
            "Stage B v2 checkpoint requires optimizer state and config"
        )
    if data_contract is None or model_identity is None:
        raise ValueError(
            "Stage B v2 checkpoint requires data and base-model identity contracts"
        )
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 0
    ):
        raise ValueError("Stage B v2 checkpoint requires non-negative completed_steps")
    resume_contract = build_resume_contract(
        args_dict,
        trainable,
        optimizer,
        data_contract,
        model_identity,
        curriculum_plan=curriculum_plan,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "trainable_meta": {key: value for key, value in trainable.items() if key != "groups"},
        "args": dict(args_dict),
        "metrics": metrics,
        "provenance": checkpoint_provenance,
        "run_identity": run_identity,
        "final_inference_top_k": FINAL_STUDENT_TOP_K,
        "checkpoint_runtime_top_k": int(runtime_top_k),
        "completed_steps": int(completed_steps),
        "resume_contract": resume_contract,
        "curriculum_plan": dict(resume_contract["curriculum_plan"]),
        "rng_state": capture_rng_state(),
    }
    state["optimizer_state"] = optimizer.state_dict()
    if trainable.get("train_lm_head"):
        output_embeddings = student.get_output_embeddings()
        input_embeddings = student.get_input_embeddings()
        embeddings_tied = lm_embeddings_are_tied(student)
        state["lm_embeddings_tied"] = embeddings_tied
        state["lm_output_embeddings"] = (
            output_embeddings.state_dict() if output_embeddings is not None else None
        )
        if not embeddings_tied:
            state["lm_input_embeddings"] = (
                input_embeddings.state_dict() if input_embeddings is not None else None
            )
    if trainable.get("train_router_gates"):
        state["router_gates"] = {
            f"layer_{idx}": layer.mlp.gate.state_dict()
            for idx, layer in enumerate(student.model.layers)
        }
    if trainable.get("train_gamma_scale"):
        state["gamma_scale"] = gamma_scale_values(student)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite Stage B checkpoint: {path}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/real_subset_clean_260708b")
    parser.add_argument("--output-dir", default="outputs/top2_distill_real")
    parser.add_argument("--base-model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-top-k", type=int, default=8)
    parser.add_argument("--student-top-k", type=int, default=2)
    parser.add_argument("--capacity-factor", type=float, default=6.0)
    parser.add_argument("--aux-coef", type=float, default=0.01)
    parser.add_argument("--gamma-min", type=float, default=0.25)
    parser.add_argument("--gamma-max", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--text-eval-blocks", type=int, default=160)
    parser.add_argument("--distill-steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--router-learning-rate", type=float, default=1e-5)
    parser.add_argument("--gamma-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--distill-logit-coef", type=float, default=0.1)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--router-distill-coef", type=float, default=0.0)
    parser.add_argument("--router-distill-temperature", type=float, default=2.0)
    parser.add_argument("--distill-hidden-coef", type=float, default=0.0)
    parser.add_argument("--distill-hidden-layers", default="last")
    parser.add_argument("--distill-hidden-mode", choices=["mse", "cosine"], default="cosine")
    parser.add_argument("--moe-reconstruction-coef", type=float, default=0.0)
    parser.add_argument("--moe-reconstruction-layers", default="all")
    parser.add_argument("--text-replay-coef", type=float, default=0.0)
    parser.add_argument("--text-replay-manifest", default="")
    parser.add_argument("--student-k-curriculum", default="none")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--train-lm-head", action="store_true")
    parser.add_argument("--train-router-gates", action="store_true")
    parser.add_argument("--train-gamma-scale", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = validate_stage_b_args(args)
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = validate_development_path(args.data_dir, "data directory")
    text_rows, text_record = read_bound_jsonl(
        data_dir / "text_tasks.jsonl", "text_tasks"
    )
    train_blocks, train_record = read_bound_jsonl(
        data_dir / "text_blocks_train.jsonl", "train"
    )
    eval_blocks, eval_record = read_bound_jsonl(
        data_dir / "text_blocks_eval.jsonl", "development_eval"
    )
    if not train_blocks or not eval_blocks:
        raise RuntimeError(f"Missing text blocks under {data_dir}")
    validate_development_rows(text_rows, "text task manifest")
    validate_development_rows(train_blocks, "training manifest")
    validate_development_rows(eval_blocks, "development evaluation manifest")

    replay_blocks: List[Dict[str, Any]] = []
    replay_path: Path | None = None
    data_records = [text_record, train_record, eval_record]
    if str(args.text_replay_manifest).strip():
        replay_path = validate_development_path(
            args.text_replay_manifest,
            "text replay manifest",
        )
        replay_blocks, replay_record = read_bound_jsonl(
            replay_path, "text_replay"
        )
        data_records.append(replay_record)
        validate_development_rows(replay_blocks, "text replay manifest")
    if args.text_replay_coef > 0.0 and not replay_blocks:
        raise RuntimeError("text replay manifest is empty")

    calibration_texts = [
        str(
            row.get("text")
            or (str(row.get("prompt", "")) + " " + str(row.get("target", "")))
        )
        for row in text_rows[: min(128, len(text_rows))]
    ]
    repo_root = Path(__file__).resolve().parents[1]
    source_files: Dict[str, Dict[str, Any]] = {
        "text_tasks": {
            "path": data_dir / "text_tasks.jsonl",
            "rows": len(text_rows),
            "split": "development_calibration_source",
        },
        "train": {
            "path": data_dir / "text_blocks_train.jsonl",
            "rows": len(train_blocks),
            "split": "train",
        },
        "development_eval": {
            "path": data_dir / "text_blocks_eval.jsonl",
            "rows": len(eval_blocks),
            "split": "development_eval",
        },
    }
    if replay_path is not None:
        source_files["text_replay"] = {
            "path": replay_path,
            "rows": len(replay_blocks),
            "split": "train_replay",
        }
    stage_b_run_provenance = build_stage_b_run_provenance(
        repo_root=repo_root,
        data_dir=data_dir,
        source_files=source_files,
    )
    stage_b_run_identity = build_checkpoint_run_identity(
        stage_b_run_provenance
    )
    manifest_path = out_dir / "manifest.json"
    data_contract = stage_b_data_contract(data_dir, data_records)
    save_json(
        manifest_path,
        {
            "args": vars(args),
            "data_policy": "development_only_real_manifests",
            "development_only": True,
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
            "source_commit_sha": stage_b_run_provenance["source_commit_sha"],
            "runai_job_name": stage_b_run_provenance["runai_job_name"],
            "runai_project": stage_b_run_provenance["runai_project"],
            "run_identity": stage_b_run_identity,
            "run_provenance": stage_b_run_provenance,
            "dataset_split_provenance": stage_b_run_provenance[
                "dataset_split_provenance"
            ],
            "data_dir": str(data_dir),
            "text_task_manifest": str(data_dir / "text_tasks.jsonl"),
            "train_manifest": str(data_dir / "text_blocks_train.jsonl"),
            "development_eval_manifest": str(data_dir / "text_blocks_eval.jsonl"),
            "text_replay_manifest": str(replay_path) if replay_path is not None else None,
            "resume_data_contract": data_contract,
            "train_blocks": len(train_blocks),
            "eval_blocks": len(eval_blocks),
            "replay_blocks": len(replay_blocks),
            "student_k_schedule": schedule,
            "final_inference_top_k": FINAL_STUDENT_TOP_K,
        },
    )

    teacher, tokenizer, teacher_meta = load_model(
        args.base_model,
        args.teacher_top_k,
        args.aux_coef,
        gamma=None,
        capacity_factor=args.capacity_factor,
        pre_routing_identity_fn=lambda model: base_model_identity(
            model, args.base_model
        ),
    )
    teacher.eval()
    loaded_base_model_identity = teacher_meta.pop(
        "pre_routing_model_identity", None
    )
    if not isinstance(loaded_base_model_identity, Mapping):
        raise ValueError("teacher pre-routing base-model identity is missing")
    if effective_top_k(teacher) != 8:
        raise RuntimeError(f"loaded teacher is not Top-8: actual={effective_top_k(teacher)}")
    e0 = teacher_student_metrics(
        teacher,
        teacher,
        tokenizer,
        eval_blocks[: args.text_eval_blocks],
        args,
        out_dir,
        "E0_top8_teacher",
    )

    hard_student, _, hard_meta = load_model(
        args.base_model,
        args.student_top_k,
        args.aux_coef,
        gamma=None,
        capacity_factor=args.capacity_factor,
    )
    enforce_final_top_k(hard_student, args.aux_coef)
    e1 = teacher_student_metrics(
        hard_student,
        teacher,
        tokenizer,
        eval_blocks[: args.text_eval_blocks],
        args,
        out_dir,
        "E1_hard_top2",
    )
    cleanup(hard_student)

    gamma = calibrate_gamma(args, calibration_texts, out_dir)
    calibrated, _, calibrated_meta = load_model(
        args.base_model,
        args.student_top_k,
        args.aux_coef,
        gamma=gamma,
        capacity_factor=args.capacity_factor,
    )
    enforce_final_top_k(calibrated, args.aux_coef)
    e2 = teacher_student_metrics(
        calibrated,
        teacher,
        tokenizer,
        eval_blocks[: args.text_eval_blocks],
        args,
        out_dir,
        "E2_gamma_calibrated_top2",
    )
    cleanup(calibrated)

    e2_ce = train_student(
        "E2_CE_only",
        teacher,
        tokenizer,
        gamma,
        train_blocks,
        eval_blocks,
        args,
        out_dir,
        use_distill=False,
        replay_blocks=replay_blocks,
        stage_b_run_provenance=stage_b_run_provenance,
        data_contract=data_contract,
        expected_model_identity=loaded_base_model_identity,
    )
    e2d_kl = train_student(
        "E2D_logits_kl",
        teacher,
        tokenizer,
        gamma,
        train_blocks,
        eval_blocks,
        args,
        out_dir,
        use_distill=True,
        replay_blocks=replay_blocks,
        stage_b_run_provenance=stage_b_run_provenance,
        data_contract=data_contract,
        expected_model_identity=loaded_base_model_identity,
    )
    companion_path = out_dir / "E2D_logits_kl" / "stage_b_companion_manifest.json"
    companion = write_stage_b_companion(
        output_path=companion_path,
        checkpoint_path=out_dir / "E2D_logits_kl" / "checkpoint_final.pt",
        metrics_path=out_dir / "E2D_logits_kl" / "metrics.json",
        run_manifest_path=manifest_path,
        repo_root=repo_root,
    )
    summary = {
        "manifest": {
            "args": vars(args),
            "data_policy": "development_only_real_manifests",
            "teacher_meta": teacher_meta,
            "hard_meta": hard_meta,
            "calibrated_meta": calibrated_meta,
            "gamma": gamma,
            "student_k_schedule": schedule,
            "final_inference_top_k": FINAL_STUDENT_TOP_K,
        },
        "E0": e0,
        "E1": e1,
        "E2": e2,
        "E2_CE_only": e2_ce,
        "E2D_logits_kl": e2d_kl,
        "stage_b_companion": {
            "path": str(companion_path.resolve()),
            "sha256": sha256_file(companion_path),
            "source_commit_sha": companion["source_commit_sha"],
        },
    }
    save_json(out_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "summary_path": str(out_dir / "summary.json"),
                "stage_b_companion_path": str(companion_path.resolve()),
                "experiments": [
                    "E0",
                    "E1",
                    "E2",
                    "E2_CE_only",
                    "E2D_logits_kl",
                ],
                "final_inference_top_k": FINAL_STUDENT_TOP_K,
            },
            sort_keys=True,
        )
    )
    cleanup(teacher)


if __name__ == "__main__":
    main()
