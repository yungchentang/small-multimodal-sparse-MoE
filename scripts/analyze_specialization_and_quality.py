"""Analyze modality-specific routing specialization and qualitative generations.

This script reloads the final multimodal E3 checkpoint and runs real eval records
through the shared OLMoE prefix path. It produces routing distributions, top
experts, overlap/distance statistics, layer-by-expert heatmaps, and qualitative
image/speech prefix generations. The outputs are evidence artifacts, not audit
shortcuts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from scripts.eval_conditional_retrieval import load_trained_wrapper
from scripts.eval_representation_funnel import _bind_runtime_base_identity
from training.olmoe_required_runs import iter_olmoe_mlp_layers, save_json
from training.olmoe_real_subset_runs import (
    FeatureCache,
    absolutize_media_paths,
    audio_features_from_paths,
    image_features_from_paths,
    read_jsonl,
    per_example_prefix_nll,
    split_tail,
    tensorize_blocks,
    tokenize_prompt_targets,
)

MODALITIES = ["text", "image_prefix", "audio_prefix"]


def _load_explicit_development_manifest(
    path_value: str,
    *,
    data_dir: Path,
    modality: str,
    expected_rows: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_path = Path(path_value).expanduser()
    if raw_path.is_symlink():
        raise ValueError(f"development {modality} manifest must not be a symlink")
    path = raw_path.resolve(strict=True)
    lowered_parts = {part.lower() for part in path.parts}
    if any("sealed" in part or "synthetic" in part for part in lowered_parts):
        raise ValueError(f"development {modality} manifest cannot use sealed/synthetic data")
    if not path.is_file():
        raise ValueError(f"development {modality} manifest is not a regular file: {path}")
    rows = [dict(row) for row in read_jsonl(path)]
    if len(rows) != int(expected_rows):
        raise ValueError(
            f"development {modality} manifest rows={len(rows)} != expected={expected_rows}"
        )
    expected_role = f"{modality}_dev"
    if any(str(row.get("task")) != modality for row in rows):
        raise ValueError(f"development {modality} manifest has a mismatched task")
    if any(str(row.get("eval_split_name")) != expected_role for row in rows):
        raise ValueError(
            f"development {modality} manifest must contain only {expected_role} rows"
        )
    row_ids = [str(row.get("id")) for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(f"development {modality} manifest contains duplicate row ids")
    absolutize_media_paths(rows, data_dir)
    media_key = "image_path" if modality == "image" else "audio_path"
    missing = [
        str(row.get("id"))
        for row in rows
        if not Path(str(row.get(media_key, ""))).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"development {modality} manifest has missing media for ids={missing[:3]}"
        )
    return rows, {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "split_role": expected_role,
    }


def load_specialization_rows(
    args: argparse.Namespace, data_dir: Path
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    image_manifest = str(getattr(args, "image_manifest", "") or "")
    speech_manifest = str(getattr(args, "speech_manifest", "") or "")
    if bool(image_manifest) != bool(speech_manifest):
        raise ValueError("image and speech development manifests must be provided together")
    if image_manifest:
        image_rows, image_provenance = _load_explicit_development_manifest(
            image_manifest,
            data_dir=data_dir,
            modality="image",
            expected_rows=int(args.image_eval_count),
        )
        speech_rows, speech_provenance = _load_explicit_development_manifest(
            speech_manifest,
            data_dir=data_dir,
            modality="speech",
            expected_rows=int(args.speech_eval_count),
        )
        return image_rows, speech_rows, {
            "policy": "explicit_frozen_development_manifests_v1",
            "development_only": True,
            "sealed_test_used": False,
            "synthetic_data_used": False,
            "image": image_provenance,
            "speech": speech_provenance,
        }

    image_rows = read_jsonl(data_dir / "image_captions.jsonl")
    speech_rows = read_jsonl(data_dir / "speech_transcripts.jsonl")
    absolutize_media_paths(image_rows, data_dir)
    absolutize_media_paths(speech_rows, data_dir)
    _, image_eval = split_tail(image_rows, args.image_eval_count)
    _, speech_eval = split_tail(speech_rows, args.speech_eval_count)
    return image_eval, speech_eval, {
        "policy": "legacy_canonical_tail",
        "development_only": False,
        "sealed_test_used": False,
        "synthetic_data_used": False,
        "image": {
            "path": str((data_dir / "image_captions.jsonl").resolve()),
            "rows": len(image_eval),
        },
        "speech": {
            "path": str((data_dir / "speech_transcripts.jsonl").resolve()),
            "rows": len(speech_eval),
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(values: Sequence[float]) -> List[float]:
    total = float(sum(values))
    if total <= 0.0:
        return [0.0 for _ in values]
    return [float(v) / total for v in values]


def js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    eps = 1e-12
    mid = [(float(a) + float(b)) * 0.5 for a, b in zip(left, right)]

    def kl(p: Sequence[float], q: Sequence[float]) -> float:
        return sum(float(pi) * math.log((float(pi) + eps) / (float(qi) + eps)) for pi, qi in zip(p, q) if float(pi) > 0.0)

    return float(0.5 * kl(left, mid) + 0.5 * kl(right, mid))


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    la = math.sqrt(sum(float(a) * float(a) for a in left))
    lb = math.sqrt(sum(float(b) * float(b) for b in right))
    if la <= 0.0 or lb <= 0.0:
        return 1.0
    return float(1.0 - dot / (la * lb))


def top_experts(dist: Sequence[float], k: int) -> List[Dict[str, Any]]:
    order = sorted(range(len(dist)), key=lambda idx: float(dist[idx]), reverse=True)[:k]
    return [{"expert": int(idx), "share": float(dist[idx])} for idx in order]


def layerwise_span_counts(outputs, spans: Dict[str, Tuple[int, int]], batch_size: int, seq_len: int, num_experts: int, top_k: int, expert_biases: Sequence[Any] | None = None) -> Dict[str, List[List[int]]]:
    logits = outputs.router_logits
    tensors = list(logits) if not torch.is_tensor(logits) else [logits]
    counts = {name: torch.zeros((len(tensors), num_experts), dtype=torch.long) for name in spans}
    for layer_idx, tensor in enumerate(tensors):
        tensor = tensor.detach()
        if tensor.ndim == 2:
            if tensor.shape[0] != batch_size * seq_len:
                raise ValueError("router logits do not match batch_size * sequence_length")
            tensor = tensor.reshape(batch_size, seq_len, num_experts)
        elif tensor.ndim != 3:
            raise ValueError(f"unsupported router-logit rank: {tensor.ndim}")
        if tuple(tensor.shape) != (batch_size, seq_len, num_experts):
            raise ValueError(
                f"router-logit shape {tuple(tensor.shape)} != "
                f"{(batch_size, seq_len, num_experts)}"
            )
        adjusted = tensor.float()
        if expert_biases is not None and layer_idx < len(expert_biases) and expert_biases[layer_idx] is not None:
            adjusted = adjusted + expert_biases[layer_idx].to(device=tensor.device, dtype=torch.float32)
        probs = torch.softmax(adjusted, dim=-1)
        _, selected = torch.topk(probs, top_k, dim=-1)
        selected = selected.detach().cpu()
        for name, (start, end) in spans.items():
            if start >= selected.shape[1]:
                continue
            clipped = selected[:, start:min(end, selected.shape[1]), :]
            if clipped.numel() == 0:
                continue
            counts[name][layer_idx] += torch.bincount(clipped.reshape(-1), minlength=num_experts)
    return {name: value.tolist() for name, value in counts.items()}


def add_counts(accum: Dict[str, torch.Tensor], batch_counts: Dict[str, List[List[int]]]) -> None:
    for name, matrix in batch_counts.items():
        tensor = torch.tensor(matrix, dtype=torch.long)
        if name not in accum:
            accum[name] = torch.zeros_like(tensor)
        accum[name] += tensor


def svg_heatmap(path: Path, matrix: Sequence[Sequence[float]], title: str, x_label: str, y_label: str, row_labels: Sequence[str] | None = None, cell_w: int = 10, cell_h: int = 18) -> None:
    rows = len(matrix)
    cols = max((len(row) for row in matrix), default=0)
    left = 78
    top = 52
    width = left + cols * cell_w + 90
    height = top + rows * cell_h + 70
    max_v = max((float(v) for row in matrix for v in row), default=1.0) or 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="20" y="24" font-family="Arial" font-size="15" font-weight="700">{html.escape(title)}</text>')
    parts.append(f'<text x="{left + cols * cell_w / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="11">{html.escape(x_label)}</text>')
    parts.append(f'<text x="18" y="{top + rows * cell_h / 2:.1f}" transform="rotate(-90 18 {top + rows * cell_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="11">{html.escape(y_label)}</text>')
    for r, row in enumerate(matrix):
        label = row_labels[r] if row_labels and r < len(row_labels) else str(r)
        parts.append(f'<text x="{left - 8}" y="{top + r * cell_h + cell_h * 0.68:.1f}" text-anchor="end" font-family="Arial" font-size="8">{html.escape(label)}</text>')
        for c, value in enumerate(row):
            intensity = min(1.0, math.sqrt(max(0.0, float(value)) / max_v))
            red = int(245 - 205 * intensity)
            green = int(248 - 120 * intensity)
            blue = int(255 - 20 * intensity)
            parts.append(f'<rect x="{left + c * cell_w}" y="{top + r * cell_h}" width="{cell_w}" height="{cell_h}" fill="rgb({red},{green},{blue})"/>')
    for c in range(0, cols, 8):
        parts.append(f'<text x="{left + c * cell_w + cell_w/2}" y="{top - 8}" text-anchor="middle" font-family="Arial" font-size="7">{c}</text>')
    parts.append(f'<text x="{left + cols * cell_w + 12}" y="{top + 10}" font-family="Arial" font-size="9">max={max_v:.4g}</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_bars(path: Path, dists: Dict[str, Sequence[float]]) -> None:
    cols = 64
    width = 860
    height = 330
    left = 58
    top = 42
    plot_w = 760
    row_h = 78
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append('<text x="20" y="24" font-family="Arial" font-size="15" font-weight="700">Modality-specific expert allocation distributions</text>')
    colors = {"text": "#2b6cb0", "image_prefix": "#c05621", "audio_prefix": "#2f855a"}
    for ri, (name, dist) in enumerate(dists.items()):
        y0 = top + ri * row_h
        max_v = max(dist) or 1.0
        parts.append(f'<text x="{left - 8}" y="{y0 + 35}" text-anchor="end" font-family="Arial" font-size="10">{html.escape(name)}</text>')
        parts.append(f'<line x1="{left}" y1="{y0 + 48}" x2="{left + plot_w}" y2="{y0 + 48}" stroke="#333" stroke-width="0.7"/>')
        for c, value in enumerate(dist):
            bar_h = 42 * (float(value) / max_v)
            x = left + c * (plot_w / cols)
            y = y0 + 48 - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{plot_w / cols - 1:.1f}" height="{bar_h:.1f}" fill="{colors.get(name, "#555")}"/>')
        parts.append(f'<text x="{left + plot_w + 8}" y="{y0 + 35}" font-family="Arial" font-size="9">max={max_v:.3f}</text>')
    for c in range(0, cols, 8):
        x = left + c * (plot_w / cols)
        parts.append(f'<text x="{x:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="8">{c}</text>')
    parts.append(f'<text x="{left + plot_w/2}" y="{height - 7}" text-anchor="middle" font-family="Arial" font-size="10">expert id (0-63)</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def decode_new_tokens(tokenizer, prompt_ids: Sequence[int], generated: Sequence[int]) -> str:
    decoded = unicodedata.normalize(
        "NFKC", tokenizer.decode(list(generated), skip_special_tokens=True)
    )
    return " ".join(decoded.split())


def greedy_generate_from_prefix(wrapper, tokenizer, prefix: torch.Tensor, prompt: str, max_new_tokens: int) -> str:
    device = prefix.device
    target_dtype = wrapper.lm.get_input_embeddings().weight.dtype
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    generated: List[int] = []
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        ids = torch.tensor([prompt_ids + generated], dtype=torch.long, device=device)
        token_embeds = wrapper.lm.get_input_embeddings()(ids)
        inputs_embeds = torch.cat([prefix.to(dtype=target_dtype), token_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = wrapper.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True, output_router_logits=False)
        next_id = int(outputs.logits[0, -1].argmax().detach().cpu())
        if eos_id is not None and next_id == int(eos_id):
            break
        generated.append(next_id)
    return decode_new_tokens(tokenizer, prompt_ids, generated)


def token_jaccard(left: str, right: str) -> float:
    import re

    lset = set(re.findall(r"[a-z0-9]+", left.lower()))
    rset = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not lset or not rset:
        return 0.0
    return float(len(lset & rset) / len(lset | rset))


def text_readability(text: str) -> Dict[str, Any]:
    value = str(text)
    characters = len(value)
    printable = sum(character.isprintable() or character in "\n\t" for character in value)
    controls = sum(
        not character.isprintable() and character not in "\n\t" for character in value
    )
    replacements = value.count(chr(0xFFFD))
    markers = (
        chr(0xC3),
        chr(0xC2),
        chr(0xE2) + chr(0x20AC),
        chr(0xF0) + chr(0x178),
    )
    mojibake_markers = sum(value.count(marker) for marker in markers)
    words = len(value.split())
    printable_ratio = printable / max(1, characters)
    readable = (
        characters > 0
        and words > 0
        and replacements == 0
        and controls == 0
        and mojibake_markers == 0
        and printable_ratio >= 0.98
    )
    return {
        "readable": readable,
        "characters": characters,
        "words": words,
        "printable_ratio": printable_ratio,
        "replacement_characters": replacements,
        "control_characters": controls,
        "mojibake_markers": mojibake_markers,
    }


def set_masked_experts(model, expert_ids: Sequence[int] | None) -> None:
    ids = [int(idx) for idx in (expert_ids or [])]
    for layer in getattr(model.model, "layers", []):
        setattr(layer.mlp, "masked_expert_ids", ids)


def clear_masked_experts(model) -> None:
    set_masked_experts(model, [])


def dynamic_expert_biases_by_layer(model) -> List[Any]:
    biases: List[Any] = []
    for _, mlp in iter_olmoe_mlp_layers(model):
        bias = getattr(mlp, "expert_bias", None)
        if bias is not None and bool(getattr(mlp, "dynamic_expert_bias_enabled", False)):
            biases.append(bias.detach().float())
        else:
            biases.append(None)
    return biases


def text_block_loss(lm, tokenizer, rows: Sequence[Dict[str, Any]], device: torch.device, args: argparse.Namespace) -> float:
    if not rows:
        return 0.0
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    losses: List[float] = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        batch = tensorize_blocks(batch_rows, device, args.max_length, pad_id)
        with torch.no_grad():
            outputs = lm(**batch, return_dict=True, output_router_logits=False)
        losses.append(float(outputs.loss.detach().float().cpu()))
    return float(sum(losses) / max(1, len(losses)))


def modality_gold_nll(wrapper, tokenizer, image_processor, vision_model, speech_processor, speech_model, rows: Sequence[Dict[str, Any]], modality: str, device: torch.device, args: argparse.Namespace, cache: FeatureCache) -> float:
    if not rows:
        return 0.0
    values: List[float] = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        if modality == "image":
            batch = tokenize_prompt_targets(tokenizer, ["Caption:"] * len(batch_rows), [str(row["caption"]) for row in batch_rows], device, args.max_length)
            feats = cache.image_batch(image_processor, vision_model, batch_rows, device, args.encoder_feature_tokens)
            with torch.no_grad():
                outputs = wrapper(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], image_features=feats)
            nll = per_example_prefix_nll(outputs.logits, batch["labels"], args.image_prefix_tokens)
        elif modality == "speech":
            batch = tokenize_prompt_targets(tokenizer, ["Transcript:"] * len(batch_rows), [str(row["transcript"]) for row in batch_rows], device, args.max_length)
            feats = cache.audio_batch(speech_processor, speech_model, batch_rows, device, args.sample_rate, args.encoder_feature_tokens)
            with torch.no_grad():
                outputs = wrapper(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], audio_features=feats)
            nll = per_example_prefix_nll(outputs.logits, batch["labels"], args.audio_prefix_tokens)
        else:
            raise ValueError(f"Unsupported modality for gold NLL: {modality}")
        values.extend(float(v) for v in nll.detach().float().cpu().tolist())
    return float(sum(values) / max(1, len(values)))


def intervention_analysis(wrapper, tokenizer, image_processor, vision_model, speech_processor, speech_model, text_eval: Sequence[Dict[str, Any]], image_eval: Sequence[Dict[str, Any]], audio_eval: Sequence[Dict[str, Any]], top: Dict[str, List[Dict[str, Any]]], device: torch.device, args: argparse.Namespace, cache: FeatureCache) -> Dict[str, Any]:
    image_rows = list(image_eval[: args.intervention_examples])
    audio_rows = list(audio_eval[: args.intervention_examples])
    text_rows = list(text_eval[: args.intervention_text_blocks])

    def measure() -> Dict[str, float]:
        return {
            "text_loss": text_block_loss(wrapper.lm, tokenizer, text_rows, device, args),
            "image_gold_nll": modality_gold_nll(wrapper, tokenizer, image_processor, vision_model, speech_processor, speech_model, image_rows, "image", device, args, cache),
            "speech_gold_nll": modality_gold_nll(wrapper, tokenizer, image_processor, vision_model, speech_processor, speech_model, audio_rows, "speech", device, args, cache),
        }

    clear_masked_experts(wrapper.lm)
    baseline = measure()
    masks: Dict[str, Any] = {}
    for modality in MODALITIES:
        experts = [int(item["expert"]) for item in top.get(modality, [])[: args.intervention_top_experts]]
        set_masked_experts(wrapper.lm, experts)
        masked = measure()
        masks[f"mask_{modality}_top_experts"] = {
            "masked_experts": experts,
            "metrics": masked,
            "delta_vs_baseline": {key: float(masked[key] - baseline[key]) for key in baseline},
        }
    clear_masked_experts(wrapper.lm)
    return {
        "method": "zero_selected_topk_expert_assignments_in_patched_olmoe_forward",
        "examples": {"text_blocks": len(text_rows), "image": len(image_rows), "speech": len(audio_rows)},
        "top_experts_per_mask": int(args.intervention_top_experts),
        "baseline": baseline,
        "masks": masks,
    }


def qualitative_examples(wrapper, tokenizer, image_processor, vision_model, speech_processor, speech_model, image_eval: Sequence[Dict[str, Any]], audio_eval: Sequence[Dict[str, Any]], device: torch.device, args: argparse.Namespace, cache: FeatureCache) -> Dict[str, Any]:
    examples: Dict[str, Any] = {"image": [], "speech": []}
    for idx, row in enumerate(image_eval[: args.qualitative_examples]):
        feats = cache.image_batch(image_processor, vision_model, [row], device, args.encoder_feature_tokens)
        prefix = wrapper.image_prefix(feats)
        generated = greedy_generate_from_prefix(wrapper, tokenizer, prefix, "Caption:", args.max_new_tokens)
        reference = str(row.get("caption", ""))
        jac = token_jaccard(generated, reference)
        readability = text_readability(generated)
        examples["image"].append({
            "index": idx,
            "image_path": str(row.get("image_path", "")),
            "reference": reference,
            "generated": generated,
            "lexical_jaccard": jac,
            "readability": readability,
            "outcome": (
                "overlap_present"
                if jac >= args.qualitative_success_jaccard
                else "low_overlap"
            ),
        })
    for idx, row in enumerate(audio_eval[: args.qualitative_examples]):
        feats = cache.audio_batch(speech_processor, speech_model, [row], device, args.sample_rate, args.encoder_feature_tokens)
        prefix = wrapper.audio_prefix(feats)
        generated = greedy_generate_from_prefix(wrapper, tokenizer, prefix, "Transcript:", args.max_new_tokens)
        reference = str(row.get("transcript", ""))
        jac = token_jaccard(generated, reference)
        readability = text_readability(generated)
        examples["speech"].append({
            "index": idx,
            "audio_path": str(row.get("audio_path", "")),
            "reference": reference,
            "generated": generated,
            "lexical_jaccard": jac,
            "readability": readability,
            "outcome": (
                "overlap_present"
                if jac >= args.qualitative_success_jaccard
                else "low_overlap"
            ),
        })
    return examples


def write_qualitative_md(path: Path, examples: Dict[str, Any]) -> None:
    lines = ["# Qualitative Prefix Generations", "", "Generated with greedy decoding from real image/audio prefixes through the shared OLMoE path. These examples are diagnostic and include failures.", ""]
    for kind in ["image", "speech"]:
        lines.extend([f"## {kind.title()}", "", "| # | Outcome | Readable | Jaccard | Reference | Generated |", "|---:|---|---|---:|---|---|"])
        for item in examples.get(kind, []):
            ref = str(item.get("reference", "")).replace("|", "/")[:220]
            gen = str(item.get("generated", "")).replace("|", "/")[:220]
            readable = bool(item.get("readability", {}).get("readable"))
            lines.append(f"| {item.get('index')} | {item.get('outcome')} | {readable} | {float(item.get('lexical_jaccard', 0.0)):.3f} | {ref} | {gen} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    run_output_dir = Path(args.run_output_dir)
    out_dir = Path(args.output_dir or run_output_dir)
    analysis_dir = out_dir / "routing_specialization"
    fig_dir = out_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("E3 checkpoint payload must be a mapping")
    state_args = dict(state.get("args") or {})
    for name, expected in (
        ("capacity_factor", args.capacity_factor),
        ("aux_coef", args.aux_coef),
    ):
        if name in state_args and float(state_args[name]) != float(expected):
            raise ValueError(
                f"checkpoint {name}={state_args[name]} disagrees with evaluator {expected}"
            )
    args.evaluation_scope = "development"
    args.top_k = 2
    (
        wrapper,
        tokenizer,
        meta,
        image_processor,
        vision_model,
        speech_processor,
        speech_model,
        device,
    ) = load_trained_wrapper(args)
    checkpoint_restoration = _bind_runtime_base_identity(
        meta, args.stage_b_checkpoint, args.stage_b_checkpoint_sha256
    )
    if checkpoint_restoration.get("restoration_order") != [
        "base_model",
        "stage_b_student_checkpoint",
        "e3_training_checkpoint",
    ]:
        raise ValueError("unexpected E3 checkpoint restoration order")
    model = wrapper.lm
    loaded_dynamic_bias_layers = int(
        checkpoint_restoration["dynamic_expert_bias_loaded_layers"]
    )
    expert_biases = dynamic_expert_biases_by_layer(wrapper.lm)
    wrapper.eval()
    cache = FeatureCache(Path(args.feature_cache_dir) if args.feature_cache_dir else None)

    data_dir = Path(args.data_dir)
    text_eval = read_jsonl(data_dir / "text_blocks_eval.jsonl")[: args.text_batches * args.batch_size]
    image_eval_all, audio_eval_all, input_manifests = load_specialization_rows(
        args, data_dir
    )
    image_eval = image_eval_all[: args.modality_batches * args.batch_size]
    audio_eval = audio_eval_all[: args.modality_batches * args.batch_size]

    num_experts = int(model.config.num_experts)
    top_k = 2
    layer_counts: Dict[str, torch.Tensor] = {}
    routed_token_counts = {"text": 0, "image_prefix": 0, "audio_prefix": 0}

    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    for start in range(0, len(text_eval), args.batch_size):
        rows = text_eval[start:start + args.batch_size]
        batch = tensorize_blocks(rows, device, args.max_length, pad_id)
        with torch.no_grad():
            outputs = wrapper.lm(**batch, output_router_logits=True, return_dict=True)
        seq_len = int(batch["input_ids"].shape[1])
        routed_token_counts["text"] += len(rows) * seq_len
        batch_counts = layerwise_span_counts(outputs, {"text": (0, seq_len)}, len(rows), seq_len, num_experts, top_k, expert_biases)
        add_counts(layer_counts, batch_counts)

    for start in range(0, len(image_eval), args.batch_size):
        rows = image_eval[start:start + args.batch_size]
        batch = tokenize_prompt_targets(tokenizer, ["Caption:"] * len(rows), [str(row["caption"]) for row in rows], device, args.max_length)
        feats = cache.image_batch(image_processor, vision_model, rows, device, args.encoder_feature_tokens)
        with torch.no_grad():
            outputs = wrapper(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], image_features=feats)
        seq_len = int(args.image_prefix_tokens + batch["input_ids"].shape[1])
        spans = {"image_prefix": (0, args.image_prefix_tokens)}
        routed_token_counts["image_prefix"] += len(rows) * int(args.image_prefix_tokens)
        batch_counts = layerwise_span_counts(outputs, spans, len(rows), seq_len, num_experts, top_k, expert_biases)
        add_counts(layer_counts, batch_counts)

    for start in range(0, len(audio_eval), args.batch_size):
        rows = audio_eval[start:start + args.batch_size]
        batch = tokenize_prompt_targets(tokenizer, ["Transcript:"] * len(rows), [str(row["transcript"]) for row in rows], device, args.max_length)
        feats = cache.audio_batch(speech_processor, speech_model, rows, device, args.sample_rate, args.encoder_feature_tokens)
        with torch.no_grad():
            outputs = wrapper(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], audio_features=feats)
        seq_len = int(args.audio_prefix_tokens + batch["input_ids"].shape[1])
        spans = {"audio_prefix": (0, args.audio_prefix_tokens)}
        routed_token_counts["audio_prefix"] += len(rows) * int(args.audio_prefix_tokens)
        batch_counts = layerwise_span_counts(outputs, spans, len(rows), seq_len, num_experts, top_k, expert_biases)
        add_counts(layer_counts, batch_counts)

    layer_counts_lists = {name: tensor.tolist() for name, tensor in layer_counts.items()}
    expert_counts = {name: tensor.sum(dim=0).tolist() for name, tensor in layer_counts.items()}
    router_layers = int(next(iter(layer_counts.values())).shape[0]) if layer_counts else 0
    expected_assignments = {
        name: int(tokens * router_layers * top_k)
        for name, tokens in routed_token_counts.items()
    }
    observed_assignments = {
        name: int(sum(values)) for name, values in expert_counts.items()
    }
    routing_conservation = {
        name: observed_assignments.get(name) == expected
        for name, expected in expected_assignments.items()
    }
    if not all(routing_conservation.values()):
        raise RuntimeError(
            f"routing assignment conservation failed: observed={observed_assignments} "
            f"expected={expected_assignments}"
        )
    distributions = {name: normalize(values) for name, values in expert_counts.items()}
    top = {name: top_experts(distributions[name], args.top_experts) for name in MODALITIES if name in distributions}
    js: Dict[str, float] = {}
    cosine: Dict[str, float] = {}
    overlap: Dict[str, Any] = {}
    for i, left in enumerate(MODALITIES):
        for right in MODALITIES[i + 1:]:
            if left not in distributions or right not in distributions:
                continue
            key = f"{left}_vs_{right}"
            js[key] = js_divergence(distributions[left], distributions[right])
            cosine[key] = cosine_distance(distributions[left], distributions[right])
            left_top = {item["expert"] for item in top[left]}
            right_top = {item["expert"] for item in top[right]}
            inter = sorted(left_top & right_top)
            union = sorted(left_top | right_top)
            overlap[key] = {"intersection": inter, "union": union, "jaccard": float(len(inter) / max(1, len(union)))}

    mean_js = sum(js.values()) / max(1, len(js))
    mean_overlap = sum(item["jaccard"] for item in overlap.values()) / max(1, len(overlap))
    conclusion = (
        "Frozen-router allocation distributions differ across text, image-prefix, and audio-prefix tokens. This is descriptive modality association, not evidence that experts learned or causally specialize by modality."
        if mean_js >= 0.05 or mean_overlap <= 0.6
        else "Frozen-router allocation distributions are similar across modalities. The evidence supports shared-expert usage and no learned or causal specialization claim."
    )

    # Save CSVs.
    with (analysis_dir / "modality_expert_distributions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expert"] + MODALITIES)
        for expert in range(num_experts):
            writer.writerow([expert] + [distributions.get(name, [0.0] * num_experts)[expert] for name in MODALITIES])
    with (analysis_dir / "top_experts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["modality", "rank", "expert", "share"])
        for modality, rows in top.items():
            for rank, item in enumerate(rows, start=1):
                writer.writerow([modality, rank, item["expert"], item["share"]])
    for modality, matrix in layer_counts_lists.items():
        with (analysis_dir / f"layer_expert_counts_{modality}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["layer"] + [f"expert_{idx}" for idx in range(num_experts)])
            for layer_idx, row in enumerate(matrix):
                writer.writerow([layer_idx] + row)

    svg_bars(fig_dir / "fig_modality_expert_distributions.svg", {name: distributions[name] for name in MODALITIES if name in distributions})
    # Rows as modality x experts.
    svg_heatmap(
        fig_dir / "fig_modality_expert_heatmap_64.svg",
        [distributions.get(name, [0.0] * num_experts) for name in MODALITIES],
        "Full 64-expert modality allocation heatmap",
        "expert id",
        "modality",
        MODALITIES,
        cell_w=11,
        cell_h=34,
    )
    for modality in MODALITIES:
        if modality in layer_counts_lists:
            norm_rows = [normalize(row) for row in layer_counts_lists[modality]]
            svg_heatmap(
                fig_dir / f"fig_layer_expert_heatmap_{modality}.svg",
                norm_rows,
                f"Layer x expert routing frequency: {modality}",
                "expert id",
                "layer",
                [str(i) for i in range(len(norm_rows))],
                cell_w=9,
                cell_h=16,
            )

    intervention = intervention_analysis(
        wrapper, tokenizer, image_processor, vision_model, speech_processor, speech_model,
        text_eval, image_eval_all, audio_eval_all, top, device, args, cache
    )
    save_json(analysis_dir / "expert_intervention.json", intervention)

    qualitative = qualitative_examples(wrapper, tokenizer, image_processor, vision_model, speech_processor, speech_model, image_eval_all, audio_eval_all, device, args, cache)
    save_json(analysis_dir / "qualitative_generations.json", qualitative)
    write_qualitative_md(analysis_dir / "qualitative_generations.md", qualitative)

    metrics = {
        "analysis_type": "checkpoint_layerwise_modality_routing_and_qualitative_generation",
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "source_checkpoint_sha256": sha256_file(Path(args.checkpoint).resolve()),
        "stage_b_checkpoint_sha256": checkpoint_restoration.get(
            "source_checkpoint_hashes", {}
        ).get("stage_b"),
        "checkpoint_restoration": checkpoint_restoration,
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "run_output_dir": str(run_output_dir),
        "data_dir": str(data_dir),
        "evidence_scope": (
            "development_only"
            if input_manifests["development_only"]
            else "legacy_canonical_tail"
        ),
        "sealed_test_used": False,
        "input_manifests": input_manifests,
        "num_experts": num_experts,
        "router_layers": router_layers,
        "dynamic_expert_bias_loaded_layers": int(loaded_dynamic_bias_layers),
        "routing_selection_source": "router_logits_plus_dynamic_expert_bias" if loaded_dynamic_bias_layers else "router_logits",
        "counts": observed_assignments,
        "routing_accounting": {
            "top_k": top_k,
            "assignment_stage": "attempted_pre_capacity_top2_from_router_logits",
            "denominator": "routed_tokens_x_router_layers_x_top_k",
            "routed_token_counts": routed_token_counts,
            "expected_assignments": expected_assignments,
            "observed_assignments": observed_assignments,
            "conservation": routing_conservation,
            "all_conserved": all(routing_conservation.values()),
            "image_audio_counts_include_prefix_tokens": True,
            "capacity_drops_included": False,
            "image_span": [0, int(args.image_prefix_tokens)],
            "audio_span": [0, int(args.audio_prefix_tokens)],
        },
        "expert_counts": expert_counts,
        "distributions": distributions,
        "top_experts": top,
        "overlap": overlap,
        "js_divergence": js,
        "cosine_distance": cosine,
        "mean_pairwise_js": mean_js,
        "mean_top_overlap_jaccard": mean_overlap,
        "specialization_conclusion": conclusion,
        "layer_expert_counts": layer_counts_lists,
        "figures": {
            "modality_distributions": str(fig_dir / "fig_modality_expert_distributions.svg"),
            "modality_heatmap_64": str(fig_dir / "fig_modality_expert_heatmap_64.svg"),
            "layer_text": str(fig_dir / "fig_layer_expert_heatmap_text.svg"),
            "layer_image": str(fig_dir / "fig_layer_expert_heatmap_image_prefix.svg"),
            "layer_audio": str(fig_dir / "fig_layer_expert_heatmap_audio_prefix.svg"),
        },
        "expert_intervention": {
            "json": str(analysis_dir / "expert_intervention.json"),
            "method": intervention.get("method"),
            "baseline": intervention.get("baseline"),
            "mask_summaries": {name: data.get("delta_vs_baseline", {}) for name, data in intervention.get("masks", {}).items()},
        },
        "qualitative": {
            "json": str(analysis_dir / "qualitative_generations.json"),
            "markdown": str(analysis_dir / "qualitative_generations.md"),
            "image_overlap_present": sum(
                1 for item in qualitative.get("image", [])
                if item.get("outcome") == "overlap_present"
            ),
            "speech_overlap_present": sum(
                1 for item in qualitative.get("speech", [])
                if item.get("outcome") == "overlap_present"
            ),
            "image_readable": sum(
                1 for item in qualitative.get("image", [])
                if item.get("readability", {}).get("readable")
            ),
            "speech_readable": sum(
                1 for item in qualitative.get("speech", [])
                if item.get("readability", {}).get("readable")
            ),
            "image_examples": len(qualitative.get("image", [])),
            "speech_examples": len(qualitative.get("speech", [])),
            "all_readable": all(
                item.get("readability", {}).get("readable")
                for modality in ("image", "speech")
                for item in qualitative.get(modality, [])
            ),
        },
    }
    save_json(analysis_dir / "metrics.json", metrics)
    print(json.dumps({
        "metrics": str(analysis_dir / "metrics.json"),
        "mean_pairwise_js": mean_js,
        "mean_top_overlap_jaccard": mean_overlap,
        "conclusion": conclusion,
        "qualitative": metrics["qualitative"],
    }, sort_keys=True))
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-output-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--data-dir", default="data/real_subset_final")
    parser.add_argument("--image-manifest", default="")
    parser.add_argument("--speech-manifest", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint-sha256", required=True)
    parser.add_argument("--feature-cache-dir", default="")
    parser.add_argument("--base-model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--vision-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--speech-model", default="openai/whisper-base.en")
    parser.add_argument("--speech-target-space", default="olmoe_text_hidden")
    parser.add_argument("--alignment-prefix-residual", action="store_true")
    parser.add_argument("--capacity-factor", type=float, default=4.0)
    parser.add_argument("--aux-coef", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--text-batches", type=int, default=8)
    parser.add_argument("--modality-batches", type=int, default=8)
    parser.add_argument("--image-eval-count", type=int, default=250)
    parser.add_argument("--speech-eval-count", type=int, default=250)
    parser.add_argument("--image-prefix-tokens", type=int, default=50)
    parser.add_argument("--audio-prefix-tokens", type=int, default=64)
    parser.add_argument("--encoder-feature-tokens", type=int, default=100)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--top-experts", type=int, default=10)
    parser.add_argument("--qualitative-examples", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=28)
    parser.add_argument("--qualitative-success-jaccard", type=float, default=0.08)
    parser.add_argument("--intervention-top-experts", type=int, default=5)
    parser.add_argument("--intervention-examples", type=int, default=24)
    parser.add_argument("--intervention-text-blocks", type=int, default=16)
    return parser.parse_args(argv)


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
