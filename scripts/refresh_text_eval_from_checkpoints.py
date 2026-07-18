"""Regenerate text-eval provenance for selected multimodal experiments.

This mirrors a source evidence root into a fresh output root, reloads saved
experiment checkpoints, and re-runs held-out text evaluation through OLMoE. It is
used to refresh stale explanatory provenance without mutating old generated
artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import torch

from training.olmoe_required_runs import cleanup, load_model, save_json
from training.olmoe_real_subset_runs import evaluate_text_blocks, read_jsonl

EXP_DIR_TO_SUMMARY_KEY = {
    "E3_final_multimodal_top2": "E3",
    "E4_no_aux_load_balance_ablation": "E4",
    "E5_capacity_1p25_ablation": "E5",
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_pretty(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_gamma(root: Path) -> List[float]:
    data = load_json(root / "calibration" / "gamma.json")
    return [float(x) for x in data["gamma"]]


def mirror_source(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Output root already exists: {output}")
    ignore = shutil.ignore_patterns("summary.json", "requirement_audit.json", "real_required_runs_table.csv", "dataset_counts.csv")
    shutil.copytree(source, output, ignore=ignore)


def finite_loss(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, float]:
    vals = []
    for row in rows:
        try:
            val = float(row.get(key, row.get("loss")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    if not vals:
        return {"first": float("nan"), "last": float("nan"), "min": float("nan")}
    return {"first": vals[0], "last": vals[-1], "min": min(vals)}


def load_lm_state(model, state: Dict[str, Any]) -> None:
    if state.get("lm_output_embeddings") is not None:
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            output_embeddings.load_state_dict(state["lm_output_embeddings"])
    if state.get("lm_input_embeddings") is not None:
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None:
            input_embeddings.load_state_dict(state["lm_input_embeddings"])
    load_dynamic_expert_bias_state(model, state.get("dynamic_expert_bias"))
    if "router_gates" in state:
        for idx, layer in enumerate(model.model.layers):
            key = f"layer_{idx}"
            if key in state["router_gates"]:
                layer.mlp.gate.load_state_dict(state["router_gates"][key])


def refresh_experiment(out_dir: Path, source_dir: Path, data_dir: Path, exp_dir: str, base_model: str, text_eval_blocks: int, eval_batch_size: int, max_length: int) -> Dict[str, Any]:
    exp_path = out_dir / exp_dir
    metrics_path = exp_path / "metrics.json"
    old_metrics = load_json(metrics_path)
    old_meta = old_metrics.get("meta", {}) if isinstance(old_metrics.get("meta"), dict) else {}
    capacity_factor = float(old_meta.get("capacity_factor", old_metrics.get("final_capacity_overflow_ratio_mean", 4.0)) if old_meta.get("capacity_factor", "") != "" else 4.0)
    aux_coef = float(old_meta.get("aux_coef", 0.01) if old_meta.get("aux_coef", "") != "" else 0.01)
    checkpoint = exp_path / "checkpoint_final.pt"
    if not checkpoint.exists():
        checkpoint = source_dir / exp_dir / "checkpoint_final.pt"
    state = torch.load(checkpoint, map_location="cpu")
    gamma = read_gamma(out_dir if (out_dir / "calibration" / "gamma.json").exists() else source_dir)
    model, tokenizer, meta = load_model(
        base_model,
        2,
        aux_coef,
        gamma=gamma,
        capacity_factor=capacity_factor,
        dynamic_expert_bias=bool(state.get("dynamic_expert_bias")),
    )
    load_lm_state(model, state)
    model.eval()

    rows = read_jsonl(out_dir / exp_dir / "train_metrics.jsonl")
    if not rows:
        rows = read_jsonl(source_dir / exp_dir / "train_metrics.jsonl")
    trainable_meta = dict(state.get("trainable_meta") or old_meta)
    lm_trainable = bool(trainable_meta.get("train_router_gates") or trainable_meta.get("train_experts") or trainable_meta.get("train_lm_head"))
    text_eval_note = (
        "This run trains selected language-model parameters, so text metrics must be read from the experiment checkpoint provenance rather than assumed equal to E2."
        if lm_trainable
        else "This adapter run freezes the LM/router/expert weights, so text-only metrics may remain close to calibrated Top-2 while multimodal prefix modules change."
    )
    provenance = {
        "source_experiment_id": exp_dir,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_size_bytes": int(checkpoint.stat().st_size),
        "source_training_steps": int(len(rows)),
        "source_checkpoint_saved_before_eval": True,
        "model_state_source": "checkpoint_reloaded_for_text_eval_refresh",
        "copied_from_e2": False,
        "lm_trainable": lm_trainable,
        "text_eval_note": text_eval_note,
    }
    blocks = read_jsonl(data_dir / "text_blocks_eval.jsonl")[: int(text_eval_blocks)]
    text_eval = evaluate_text_blocks(
        f"{exp_dir}_text_eval",
        model,
        tokenizer,
        blocks,
        out_dir,
        {**meta, "capacity_factor": capacity_factor, "aux_coef": aux_coef, "provenance": provenance},
        int(max_length),
        int(eval_batch_size),
    )
    trend = finite_loss(rows, "loss")
    old_metrics["text_eval_provenance"] = provenance
    old_metrics["text_eval"] = {k: v for k, v in text_eval.items() if k not in {"expert_counts_total"}}
    old_metrics["checkpoint_path"] = str(checkpoint)
    old_metrics["checkpoint_size_bytes"] = int(checkpoint.stat().st_size)
    old_metrics["first_loss"] = trend["first"]
    old_metrics["last_loss"] = trend["last"]
    old_metrics["min_loss"] = trend["min"]
    save_json_pretty(metrics_path, old_metrics)
    cleanup(model)
    return {
        "experiment_dir": exp_dir,
        "checkpoint": str(checkpoint),
        "text_perplexity": text_eval.get("perplexity"),
        "text_accuracy": text_eval.get("next_token_accuracy"),
        "provenance_note_key": "text_eval_note",
        "forbidden_stale_key_absent": "frozen_lm_text_eval_note" not in provenance,
    }


def rebuild_summary(source_dir: Path, out_dir: Path, refreshed: List[Dict[str, Any]]) -> None:
    summary = load_json(source_dir / "summary.json")
    for item in refreshed:
        exp_dir = str(item["experiment_dir"])
        key = EXP_DIR_TO_SUMMARY_KEY.get(exp_dir)
        if key:
            summary[key] = load_json(out_dir / exp_dir / "metrics.json")
    summary["text_eval_refresh_provenance"] = refreshed
    save_json_pretty(out_dir / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/real_subset_final"))
    parser.add_argument("--experiments", default="E4_no_aux_load_balance_ablation,E5_capacity_1p25_ablation")
    parser.add_argument("--base-model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--text-eval-blocks", type=int, default=160)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--mirror-source-root", action="store_true")
    args = parser.parse_args()

    if args.mirror_source_root:
        mirror_source(args.source_output_dir, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    refreshed = []
    for exp_dir in [part.strip() for part in args.experiments.split(",") if part.strip()]:
        refreshed.append(refresh_experiment(args.output_dir, args.source_output_dir, args.data_dir, exp_dir, args.base_model, args.text_eval_blocks, args.eval_batch_size, args.max_length))
    rebuild_summary(args.source_output_dir, args.output_dir, refreshed)
    provenance = {
        "type": "text_eval_checkpoint_refresh",
        "source_output_dir": str(args.source_output_dir),
        "output_dir": str(args.output_dir),
        "experiments": refreshed,
    }
    save_json(args.output_dir / "text_eval_refresh_provenance.json", provenance)
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
