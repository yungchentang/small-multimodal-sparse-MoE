"""OLMoE Top-k probe and short multimodal prefix training."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from hf_sources import load_pretrained
from model.olmoe_adapter import OLMoEMultimodalPrefixWrapper


def choose_dtype() -> Tuple[torch.dtype, str]:
    if not torch.cuda.is_available():
        return torch.float32, "float32"
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bfloat16"
    return torch.float16, "float16"


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def first_device(model) -> torch.device:
    return next(model.parameters()).device


def load_model(base_model: str, top_k: int):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dtype, dtype_name = choose_dtype()
    cfg = load_pretrained(AutoConfig, base_model)
    if hasattr(cfg, "num_experts_per_tok"):
        cfg.num_experts_per_tok = top_k
    if hasattr(cfg, "output_router_logits"):
        cfg.output_router_logits = True
    if hasattr(cfg, "router_aux_loss_coef"):
        cfg.router_aux_loss_coef = 0.01
    if hasattr(cfg, "norm_topk_prob"):
        cfg.norm_topk_prob = True

    tokenizer = load_pretrained(AutoTokenizer, base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_pretrained(
        AutoModelForCausalLM,
        base_model,
        config=cfg,
        dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer, {"dtype": dtype_name, "top_k": top_k, "base_model": base_model}


def router_summary(outputs) -> Dict[str, object]:
    router_logits = getattr(outputs, "router_logits", None)
    if router_logits is None and hasattr(outputs, "keys") and "router_logits" in outputs.keys():
        router_logits = outputs["router_logits"]
    if router_logits is None:
        return {"router_layers": 0}
    if torch.is_tensor(router_logits):
        tensors = [router_logits]
    else:
        tensors = list(router_logits)
    shapes = [list(t.shape) for t in tensors if torch.is_tensor(t)]
    entropies = []
    for t in tensors:
        if torch.is_tensor(t):
            probs = torch.softmax(t.float(), dim=-1)
            entropies.append(float((-(probs * (probs + 1e-9).log()).sum(dim=-1)).mean().detach().cpu()))
    return {
        "router_layers": len(shapes),
        "router_shapes": shapes[:4],
        "gate_entropy_mean": float(sum(entropies) / len(entropies)) if entropies else None,
    }


def run_forward(model, tokenizer, text: str) -> Dict[str, object]:
    device = first_device(model)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    labels = inputs["input_ids"].clone()
    with torch.no_grad():
        outputs = model(**inputs, labels=labels, output_router_logits=True, return_dict=True)
    result = {
        "loss": float(outputs.loss.detach().float().cpu()) if outputs.loss is not None else None,
        "logits_shape": list(outputs.logits.shape),
        "device": str(device),
        "cuda_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3) if torch.cuda.is_available() else 0.0,
        "cuda_memory_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3) if torch.cuda.is_available() else 0.0,
    }
    result.update(router_summary(outputs))
    aux_loss = getattr(outputs, "aux_loss", None)
    if aux_loss is not None:
        result["aux_loss"] = float(aux_loss.detach().float().cpu())
    return result


def run_probe(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    all_results = {}
    for top_k in [int(x) for x in args.top_k_list.split(",") if x.strip()]:
        model, tokenizer, meta = load_model(args.base_model, top_k)
        result = {**meta, **run_forward(model, tokenizer, args.prompt)}
        all_results[f"top{top_k}"] = result
        save_json(out / f"probe_top{top_k}.json", result)
        print(json.dumps({f"top{top_k}": result}, sort_keys=True))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json(out / "probe_summary.json", all_results)


def synthetic_features(step: int, prefix_tokens: int, input_dim: int, device: torch.device) -> torch.Tensor:
    values = torch.zeros(1, prefix_tokens, input_dim, device=device)
    values[:, :, step % input_dim] = 1.0
    values[:, :, (step + 5) % input_dim] = 0.5
    return values


def run_train(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    model, tokenizer, meta = load_model(args.base_model, args.top_k)
    device = first_device(model)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    hidden_size = int(getattr(model.config, "hidden_size"))
    wrapper = OLMoEMultimodalPrefixWrapper(
        lm=model,
        hidden_size=hidden_size,
        image_input_dim=args.feature_dim,
        audio_input_dim=args.feature_dim,
        image_prefix_tokens=args.image_prefix_tokens,
        audio_prefix_tokens=args.audio_prefix_tokens,
    )
    wrapper.image_resampler.to(device)
    wrapper.audio_resampler.to(device)
    trainable = list(wrapper.image_resampler.parameters()) + list(wrapper.audio_resampler.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    rows: List[Dict[str, object]] = []
    for step in range(1, args.max_steps + 1):
        text = "Caption and transcript: sparse Top-2 experts process multimodal prefix tokens."
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        labels = encoded["input_ids"].clone()
        image = synthetic_features(step, args.encoder_tokens, args.feature_dim, device)
        audio = synthetic_features(step + 11, args.encoder_tokens, args.feature_dim, device)
        outputs = wrapper(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            labels=labels,
            image_features=image,
            audio_features=audio,
        )
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        row = {
            "step": step,
            "loss": float(loss.detach().float().cpu()),
            "cuda_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3) if torch.cuda.is_available() else 0.0,
            "cuda_memory_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3) if torch.cuda.is_available() else 0.0,
        }
        row.update(router_summary(outputs))
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    save_json(out / "short_train_metrics.json", {"meta": meta, "steps": rows})
    torch.save(
        {
            "image_resampler": wrapper.image_resampler.state_dict(),
            "audio_resampler": wrapper.audio_resampler.state_dict(),
            "meta": meta,
        },
        out / "prefix_adapters.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["probe", "train"], required=True)
    parser.add_argument("--base-model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--top-k-list", default="8,2")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/olmoe_probe")
    parser.add_argument("--prompt", default="Sparse Top-2 MoE routing should preserve useful language modeling behavior.")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--encoder-tokens", type=int, default=4)
    parser.add_argument("--image-prefix-tokens", type=int, default=2)
    parser.add_argument("--audio-prefix-tokens", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "probe":
        run_probe(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
