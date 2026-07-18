"""Calibrate Top-2 gamma against a same-weight Top-8 teacher in the smoke model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.modeling_moe import copy_moe_weights
from training.common import build_model, load_config, make_synthetic_batch, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config_smoke.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def with_top_k(cfg, top_k: int):
    copied = json.loads(json.dumps(cfg))
    copied.setdefault("moe", {})["final_top_k"] = top_k
    copied["moe"]["top_k"] = top_k
    return copied


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/smoke"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    num_experts = int(cfg.get("moe", {}).get("num_experts", 8))
    teacher_top_k = min(int(cfg.get("moe", {}).get("teacher_top_k", 8)), num_experts)
    student_top_k = int(cfg.get("moe", {}).get("final_top_k", 2))
    teacher = build_model(with_top_k(cfg, teacher_top_k), device)
    student = build_model(with_top_k(cfg, student_top_k), device)
    copy_moe_weights(teacher, student)
    teacher.eval()
    student.eval()
    ratios = [[] for _ in range(len(student.blocks))]
    batches = int(cfg.get("calibration", {}).get("num_batches", 4))
    seq_len = int(cfg.get("model", {}).get("eval_seq_len", 64))
    batch_size = int(cfg.get("calibration", {}).get("batch_size", 2))
    with torch.no_grad():
        for step in range(1, batches + 1):
            batch = make_synthetic_batch(cfg, "text", batch_size, seq_len, step, device)
            teacher_out = teacher(**batch)
            student_out = student(**batch)
            for layer, (tm, sm) in enumerate(zip(teacher_out["router_metrics"], student_out["router_metrics"])):
                denom = float(sm.get("moe_output_norm_mean", 0.0)) + 1e-6
                ratios[layer].append(float(tm.get("moe_output_norm_mean", 0.0)) / denom)
    gamma = []
    for layer, values in enumerate(ratios):
        value = torch.tensor(values).median().clamp(0.25, 2.0).item() if values else 1.0
        student.blocks[layer].moe.set_gamma(value)
        gamma.append(value)
    save_json(output_dir / "gamma_top2.json", {"gamma": gamma, "teacher_top_k": teacher_top_k, "student_top_k": student_top_k})
    save_json(output_dir / "metrics_top2_calibration.json", {"ratios": ratios, "gamma": gamma})
    print(json.dumps({"gamma": gamma}, sort_keys=True))


if __name__ == "__main__":
    main()
