"""Short text-only distillation smoke for the Top-2 student."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model.modeling_moe import copy_moe_weights
from training.calibrate_top2 import with_top_k
from training.common import append_jsonl, build_model, load_config, make_synthetic_batch, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config_smoke.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/smoke"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    teacher_top_k = min(int(cfg.get("moe", {}).get("teacher_top_k", 8)), int(cfg.get("moe", {}).get("num_experts", 8)))
    student_top_k = int(cfg.get("moe", {}).get("final_top_k", 2))
    teacher = build_model(with_top_k(cfg, teacher_top_k), device)
    student = build_model(with_top_k(cfg, student_top_k), device)
    copy_moe_weights(teacher, student)
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(cfg.get("distillation", {}).get("learning_rate", 1e-4)))
    steps = int(cfg.get("distillation", {}).get("max_steps", 4))
    temperature = float(cfg.get("loss", {}).get("distill_temperature", 2.0))
    seq_len = int(cfg.get("model", {}).get("train_seq_len", 64))
    for step in range(1, steps + 1):
        batch = make_synthetic_batch(cfg, "text", 2, seq_len, step, device)
        with torch.no_grad():
            teacher_logits = teacher(**batch)["logits"]
        outputs = student(**batch)
        student_logits = outputs["logits"]
        kl = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature * temperature
        loss = outputs["loss"] + 0.1 * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        append_jsonl(output_dir / "distill_metrics.jsonl", {"step": step, "loss": float(loss.detach().cpu()), "kl": float(kl.detach().cpu())})
    torch.save({"model_state": student.state_dict(), "config": cfg}, output_dir / "distilled_checkpoint.pt")
    save_json(output_dir / "metrics_distill.json", {"steps": steps})


if __name__ == "__main__":
    main()
