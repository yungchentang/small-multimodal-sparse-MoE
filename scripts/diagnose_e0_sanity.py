"""Diagnose E0 OLMoE Top-8 teacher evaluation sanity.

This script intentionally compares the native HuggingFace OLMoE path against the
project's patched capacity path on the same packed real-data blocks. It checks
loss shifting, masking, decoding, per-task losses, and whether the patched Top-8
teacher is numerically close to the native Top-8 teacher.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from hf_sources import load_pretrained
from training.olmoe_required_runs import cleanup, load_model as load_patched_model, router_metrics, save_json
from training.olmoe_real_subset_runs import read_jsonl, tensorize_blocks


def choose_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_native_model(base_model: str, aux_coef: float):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = load_pretrained(AutoConfig, base_model)
    native_top_k = int(getattr(cfg, "num_experts_per_tok", 8))
    cfg.output_router_logits = True
    if hasattr(cfg, "router_aux_loss_coef"):
        cfg.router_aux_loss_coef = float(aux_coef)
    tokenizer = load_pretrained(AutoTokenizer, base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_pretrained(
        AutoModelForCausalLM,
        base_model,
        config=cfg,
        torch_dtype=choose_dtype(),
        device_map={"": 0} if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    return model, tokenizer, {"top_k": native_top_k, "config_class": cfg.__class__.__name__}


def runtime_snapshot(model) -> Dict[str, Any]:
    cfg = model.config
    snapshot: Dict[str, Any] = {
        "config_num_experts": getattr(cfg, "num_experts", None),
        "config_num_experts_per_tok": getattr(cfg, "num_experts_per_tok", None),
        "config_output_router_logits": getattr(cfg, "output_router_logits", None),
        "config_router_aux_loss_coef": getattr(cfg, "router_aux_loss_coef", None),
        "config_norm_topk_prob": getattr(cfg, "norm_topk_prob", None),
    }
    layers = getattr(getattr(model, "model", None), "layers", [])
    if layers:
        mlp = getattr(layers[0], "mlp", None)
        gate = getattr(mlp, "gate", None) if mlp is not None else None
        for prefix, obj in [("layer0_mlp", mlp), ("layer0_gate", gate)]:
            if obj is None:
                continue
            attrs: Dict[str, Any] = {"class": obj.__class__.__name__}
            for attr in ("top_k", "num_experts_per_tok", "k", "num_experts", "norm_topk_prob"):
                if hasattr(obj, attr):
                    value = getattr(obj, attr)
                    if isinstance(value, (int, float, bool, str)) or value is None:
                        attrs[attr] = value
                    else:
                        attrs[attr] = str(type(value))
            snapshot[prefix] = attrs
    return snapshot


def cross_entropy_views(outputs, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    logits = outputs.logits.detach().float()
    labels = batch["labels"]
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    shift_mask = shift_labels != -100
    shifted_sum = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    shifted_mean = shifted_sum / shift_mask.sum().clamp_min(1)
    # Control: this should usually be much worse; if it is better, alignment is suspect.
    unshifted_sum = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    unshifted_mean = unshifted_sum / (labels != -100).sum().clamp_min(1)
    preds = shift_logits.argmax(dim=-1)
    correct = ((preds == shift_labels) & shift_mask).sum()
    return {
        "manual_shifted_loss": float(shifted_mean.detach().cpu()),
        "manual_shifted_ppl": float(math.exp(min(20.0, float(shifted_mean.detach().cpu())))),
        "manual_unshifted_control_loss": float(unshifted_mean.detach().cpu()),
        "manual_unshifted_control_ppl": float(math.exp(min(20.0, float(unshifted_mean.detach().cpu())))),
        "manual_next_token_accuracy": float((correct.float() / shift_mask.sum().clamp_min(1)).detach().cpu()),
        "eval_tokens": int(shift_mask.sum().item()),
    }


def evaluate_model(name: str, model, tokenizer, blocks: Sequence[Dict[str, Any]], max_length: int, batch_size: int, capacity_factor: float, top_k: int) -> Dict[str, Any]:
    device = next(model.parameters()).device
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    model.eval()
    totals = defaultdict(float)
    counts = defaultdict(int)
    per_task = defaultdict(lambda: defaultdict(float))
    router_first = None
    first_alignment: Dict[str, Any] = {}
    with torch.no_grad():
        for start in range(0, len(blocks), batch_size):
            batch_rows = list(blocks[start:start + batch_size])
            batch = tensorize_blocks(batch_rows, device, max_length, pad_id)
            outputs = model(**batch, output_router_logits=True, return_dict=True)
            views = cross_entropy_views(outputs, batch)
            ntok = views["eval_tokens"]
            totals["loss_sum"] += views["manual_shifted_loss"] * ntok
            totals["correct_sum"] += views["manual_next_token_accuracy"] * ntok
            totals["token_count"] += ntok
            totals["unshifted_loss_sum"] += views["manual_unshifted_control_loss"] * int((batch["labels"] != -100).sum().item())
            counts["label_tokens"] += int((batch["labels"] != -100).sum().item())
            if hasattr(outputs, "loss") and outputs.loss is not None:
                # HF CausalLM loss is already shifted internally. It may include aux loss if enabled.
                totals["hf_loss_sum"] += float(outputs.loss.detach().float().cpu()) * ntok
            for row in batch_rows:
                task = str(row.get("task", "unknown"))
                per_task[task]["blocks"] += 1
            # Compute per-row shifted CE for task attribution.
            logits = outputs.logits.detach().float()
            labels = batch["labels"]
            for i, row in enumerate(batch_rows):
                sl = logits[i:i + 1, :-1]
                lab = labels[i:i + 1, 1:]
                mask = lab != -100
                loss_sum = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), lab.reshape(-1), ignore_index=-100, reduction="sum")
                task = str(row.get("task", "unknown"))
                per_task[task]["loss_sum"] += float(loss_sum.detach().cpu())
                per_task[task]["tokens"] += int(mask.sum().item())
            if router_first is None:
                router_first = router_metrics(outputs, top_k, int(model.config.num_experts), capacity_factor)
            if not first_alignment:
                row0 = batch_rows[0]
                ids = batch["input_ids"][0].detach().cpu().tolist()
                labels0 = batch["labels"][0].detach().cpu().tolist()
                preds = outputs.logits[0, :-1].argmax(dim=-1).detach().cpu().tolist()
                pairs = []
                for pos in range(min(24, len(preds))):
                    target_id = labels0[pos + 1]
                    pairs.append({
                        "pos": pos,
                        "context_tail": tokenizer.decode(ids[max(0, pos - 8):pos + 1], skip_special_tokens=False),
                        "target_id": int(target_id),
                        "target": tokenizer.decode([target_id], skip_special_tokens=False) if target_id >= 0 else "<masked>",
                        "pred_id": int(preds[pos]),
                        "pred": tokenizer.decode([preds[pos]], skip_special_tokens=False),
                    })
                first_alignment = {
                    "task": row0.get("task"),
                    "source": row0.get("source"),
                    "length": row0.get("length"),
                    "decoded_first_900_chars": tokenizer.decode(ids[:max_length], skip_special_tokens=False)[:900],
                    "first_24_next_token_pairs": pairs,
                }
    token_count = max(1, int(totals["token_count"]))
    loss = totals["loss_sum"] / token_count
    out = {
        "name": name,
        "eval_blocks": len(blocks),
        "eval_tokens": token_count,
        "manual_shifted_loss": loss,
        "manual_shifted_ppl": float(math.exp(min(20.0, loss))),
        "manual_next_token_accuracy": totals["correct_sum"] / token_count,
        "manual_unshifted_control_loss": totals["unshifted_loss_sum"] / max(1, counts["label_tokens"]),
        "hf_loss_mean": totals["hf_loss_sum"] / token_count if totals.get("hf_loss_sum") else None,
        "router_first_batch": router_first,
        "per_task": {
            task: {
                "blocks": int(vals["blocks"]),
                "tokens": int(vals["tokens"]),
                "loss": vals["loss_sum"] / max(1, vals["tokens"]),
                "ppl": float(math.exp(min(20.0, vals["loss_sum"] / max(1, vals["tokens"])))),
            }
            for task, vals in sorted(per_task.items())
        },
        "first_alignment": first_alignment,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/real_subset_final")
    parser.add_argument("--output-dir", default="outputs/e0_sanity")
    parser.add_argument("--base-model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--blocks", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--aux-coef", type=float, default=0.0)
    parser.add_argument("--capacity-factor", type=float, default=4.0)
    parser.add_argument("--skip-patched", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = read_jsonl(Path(args.data_dir) / "text_blocks_eval.jsonl")[: args.blocks]
    if not blocks:
        raise RuntimeError("No eval blocks found")

    result: Dict[str, Any] = {"args": vars(args), "block_count": len(blocks)}
    native, tokenizer, native_meta = load_native_model(args.base_model, args.aux_coef)
    result["tokenizer"] = {
        "name_or_path": getattr(tokenizer, "name_or_path", ""),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "native_top_k": native_meta["top_k"],
        "model_config_class": native_meta["config_class"],
    }
    result["native_runtime"] = runtime_snapshot(native)
    result["native_top8"] = evaluate_model("native_top8", native, tokenizer, blocks, args.max_length, args.batch_size, args.capacity_factor, native_meta["top_k"])
    cleanup(native)

    if not args.skip_patched:
        project_top8, tokenizer2, meta8 = load_patched_model(args.base_model, native_meta["top_k"], args.aux_coef, gamma=None, capacity_factor=args.capacity_factor)
        result["project_top8_runtime"] = runtime_snapshot(project_top8)
        result["project_top8"] = evaluate_model("project_top8", project_top8, tokenizer2, blocks, args.max_length, args.batch_size, args.capacity_factor, native_meta["top_k"])
        result["project_top8_meta"] = meta8
        cleanup(project_top8)

        project_top2, tokenizer3, meta2 = load_patched_model(args.base_model, 2, args.aux_coef, gamma=None, capacity_factor=args.capacity_factor)
        result["project_top2_runtime"] = runtime_snapshot(project_top2)
        result["project_top2"] = evaluate_model("project_top2", project_top2, tokenizer3, blocks, args.max_length, args.batch_size, args.capacity_factor, 2)
        result["project_top2_meta"] = meta2
        cleanup(project_top2)

    if "project_top8" in result:
        result["comparison"] = {
            "project_top8_minus_native_loss": result["project_top8"]["manual_shifted_loss"] - result["native_top8"]["manual_shifted_loss"],
            "project_top8_over_native_ppl": result["project_top8"]["manual_shifted_ppl"] / max(1e-12, result["native_top8"]["manual_shifted_ppl"]),
            "project_top2_minus_native_loss": result["project_top2"]["manual_shifted_loss"] - result["native_top8"]["manual_shifted_loss"],
            "project_top2_over_native_ppl": result["project_top2"]["manual_shifted_ppl"] / max(1e-12, result["native_top8"]["manual_shifted_ppl"]),
        }
    save_json(out_dir / "e0_sanity.json", result)
    print(json.dumps({
        "output": str(out_dir / "e0_sanity.json"),
        "native_loss": result["native_top8"]["manual_shifted_loss"],
        "native_ppl": result["native_top8"]["manual_shifted_ppl"],
        "project_top8_loss": result.get("project_top8", {}).get("manual_shifted_loss"),
        "project_top8_ppl": result.get("project_top8", {}).get("manual_shifted_ppl"),
        "project_top2_loss": result.get("project_top2", {}).get("manual_shifted_loss"),
        "project_top2_ppl": result.get("project_top2", {}).get("manual_shifted_ppl"),
        "comparison": result.get("comparison"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
