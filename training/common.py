"""Shared config, synthetic data, and logging helpers."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from model.multimodal_model import SmallMoEConfig, SmallMultimodalMoEModel


TASKS = ["text", "code", "reasoning", "math", "education", "image", "speech"]


def load_config(path: str) -> Dict[str, object]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def save_json(path: str | Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: str | Path, row: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_text(text: str, vocab_size: int, max_len: int) -> List[int]:
    ids = [1]
    for char in text:
        ids.append(4 + (ord(char) % max(1, vocab_size - 4)))
    ids.append(2)
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


def synthetic_prompt(task: str, index: int) -> str:
    examples = {
        "text": [
            "The sparse router assigns each token to two experts.",
            "A calibrated MoE keeps output norms stable.",
        ],
        "code": [
            "def add(a, b): return a + b",
            "for i in range(3): print(i)",
        ],
        "reasoning": [
            "If all cats sleep and Luna is a cat, Luna sleeps.",
            "A is left of B and B is left of C, so A is left of C.",
        ],
        "math": [
            "Question: 2 plus 3 equals 5.",
            "Question: 7 minus 4 equals 3.",
        ],
        "education": [
            "SAT analogy: hot is to cold as up is to down.",
            "The correct option is B because the premise entails it.",
        ],
        "image": [
            "Image caption: a red square on a plain background.",
            "Image caption: a blue circle near the center.",
        ],
        "speech": [
            "Transcript: the speaker reads a short sentence.",
            "Transcript: sparse experts process audio prefix tokens.",
        ],
    }
    choices = examples[task]
    return choices[index % len(choices)]


def make_feature_tensor(
    task: str,
    batch_size: int,
    num_tokens: int,
    feature_dim: int,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    features = torch.zeros(batch_size, num_tokens, feature_dim, device=device)
    for row in range(batch_size):
        slot = (step + row) % feature_dim
        features[row, :, slot] = 1.0
        features[row, :, (slot + 3) % feature_dim] = 0.5
    if task == "speech":
        features = torch.sin(features + torch.linspace(0, 1, num_tokens, device=device).view(1, -1, 1))
    return features


def make_synthetic_batch(
    cfg: Dict[str, object],
    task: str,
    batch_size: int,
    seq_len: int,
    step: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model_cfg = SmallMoEConfig.from_dict(cfg)
    tokens = [
        encode_text(synthetic_prompt(task, step + i), model_cfg.vocab_size, seq_len)
        for i in range(batch_size)
    ]
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[input_ids == 0] = -100
    batch: Dict[str, torch.Tensor] = {"input_ids": input_ids, "labels": labels}
    if task == "image":
        batch["image_features"] = make_feature_tensor(
            task,
            batch_size,
            int(cfg.get("vision", {}).get("encoder_tokens", model_cfg.image_prefix_tokens)),
            model_cfg.image_input_dim,
            step,
            device,
        )
    if task == "speech":
        batch["audio_features"] = make_feature_tensor(
            task,
            batch_size,
            int(cfg.get("speech", {}).get("encoder_tokens", model_cfg.audio_prefix_tokens)),
            model_cfg.audio_input_dim,
            step,
            device,
        )
    return batch


def build_model(cfg: Dict[str, object], device: torch.device) -> SmallMultimodalMoEModel:
    backend = str(cfg.get("model", {}).get("backend", "compact_moe"))
    if backend not in {"compact_moe", "small_moe", "smoke"}:
        raise ValueError(
            f"training.train supports the compact reproducibility backend, got {backend!r}. "
            "Use model/olmoe_adapter.py when staging the full OLMoE run."
        )
    model = SmallMultimodalMoEModel(SmallMoEConfig.from_dict(cfg))
    model.to(device)
    return model


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def summarize_router_metrics(metrics: List[Dict[str, object]]) -> Dict[str, float]:
    return {
        "gate_entropy": mean_or_zero(float(m.get("entropy", 0.0)) for m in metrics),
        "inactive_expert_ratio": mean_or_zero(float(m.get("inactive_ratio", 0.0)) for m in metrics),
        "overflow_ratio": mean_or_zero(float(m.get("overflow_ratio", 0.0)) for m in metrics),
    }


def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(min(20.0, loss)))
