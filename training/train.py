"""Train the compact multimodal Top-2 sparse MoE model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.common import (
    TASKS,
    append_jsonl,
    build_model,
    load_config,
    make_synthetic_batch,
    save_json,
    set_seed,
    summarize_router_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config_smoke.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/smoke"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = build_model(cfg, device)
    train_cfg = cfg.get("training", {})
    max_steps = int(train_cfg.get("max_steps", 12))
    batch_size = int(train_cfg.get("micro_batch_size", 2))
    seq_len = int(cfg.get("model", {}).get("train_seq_len", 64))
    lr = float(train_cfg.get("learning_rate", train_cfg.get("learning_rate_projection", 2e-4)))
    log_every = int(train_cfg.get("log_every_steps", 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 0.0)))

    train_log = output_dir / "train_metrics.jsonl"
    router_log = output_dir / "router_metrics.jsonl"
    if train_log.exists():
        train_log.unlink()
    if router_log.exists():
        router_log.unlink()

    model.train()
    last_loss = None
    for step in range(1, max_steps + 1):
        task = TASKS[(step - 1) % len(TASKS)]
        batch = make_synthetic_batch(cfg, task, batch_size, seq_len, step, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
        optimizer.step()
        last_loss = float(loss.detach().cpu())

        if step % log_every == 0 or step == max_steps:
            router_summary = summarize_router_metrics(outputs["router_metrics"])
            row = {
                "step": step,
                "task": task,
                "loss": last_loss,
                "lm_loss": float(outputs["lm_loss"].detach().cpu()),
                "aux_loss": float(outputs["aux_loss"].detach().cpu()),
                **router_summary,
            }
            append_jsonl(train_log, row)
            for layer_metrics in outputs["router_metrics"]:
                append_jsonl(router_log, {"step": step, "task": task, **layer_metrics})
            print(json.dumps(row, sort_keys=True))

    checkpoint = {
        "model_state": model.state_dict(),
        "config": cfg,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    save_json(
        output_dir / "metrics_train.json",
        {"max_steps": max_steps, "last_loss": last_loss, "device": str(device), "seed": seed},
    )
    save_json(output_dir / "resolved_config.json", cfg)
    print(f"saved {output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
