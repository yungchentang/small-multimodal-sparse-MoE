"""Evaluate a checkpoint on synthetic text, image, speech, and MoE metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from evaluation.metrics import aggregate_router_metrics, perplexity, safe_mean
from training.common import TASKS, append_jsonl, build_model, load_config, make_synthetic_batch, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config_smoke.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate_task(model, cfg: Dict[str, object], task: str, device: torch.device, output_dir: Path) -> Dict[str, object]:
    eval_cfg = cfg.get("evaluation", {})
    num_batches = int(eval_cfg.get("num_batches", 3))
    batch_size = int(eval_cfg.get("batch_size", 2))
    seq_len = int(cfg.get("model", {}).get("eval_seq_len", cfg.get("model", {}).get("train_seq_len", 64)))
    losses: List[float] = []
    router_rows: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for step in range(1, num_batches + 1):
            batch = make_synthetic_batch(cfg, task, batch_size, seq_len, 1000 + step, device)
            outputs = model(**batch)
            losses.append(float(outputs["lm_loss"].detach().cpu()))
            for layer_metrics in outputs["router_metrics"]:
                row = {"eval_task": task, "eval_step": step, **layer_metrics}
                router_rows.append(row)
                append_jsonl(output_dir / "eval_router_metrics.jsonl", row)
    mean_loss = safe_mean(losses)
    metrics = {
        "task": task,
        "loss": mean_loss,
        "perplexity": perplexity(mean_loss),
        "num_batches": num_batches,
        **aggregate_router_metrics(router_rows),
    }
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/smoke"))
    output_dir.mkdir(parents=True, exist_ok=True)
    router_log = output_dir / "eval_router_metrics.jsonl"
    if router_log.exists():
        router_log.unlink()
    device = torch.device(args.device)
    model = build_model(cfg, device)
    checkpoint_path = args.checkpoint or output_dir / "checkpoint.pt"
    if Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"], strict=True)

    all_metrics = {task: evaluate_task(model, cfg, task, device, output_dir) for task in TASKS}
    text_metrics = {task: all_metrics[task] for task in ["text", "code", "reasoning", "math", "education"]}
    image_metrics = all_metrics["image"]
    audio_metrics = all_metrics["speech"]
    moe_rows = []
    if router_log.exists():
        with router_log.open(encoding="utf-8") as handle:
            moe_rows = [json.loads(line) for line in handle if line.strip()]
    moe_metrics = aggregate_router_metrics(moe_rows)
    save_json(output_dir / "metrics_text.json", text_metrics)
    save_json(output_dir / "metrics_image.json", image_metrics)
    save_json(output_dir / "metrics_audio.json", audio_metrics)
    save_json(output_dir / "metrics_moe.json", moe_metrics)
    save_json(output_dir / "metrics_all.json", all_metrics)
    print(json.dumps({"output_dir": str(output_dir), "tasks": list(all_metrics)}, sort_keys=True))


if __name__ == "__main__":
    main()
