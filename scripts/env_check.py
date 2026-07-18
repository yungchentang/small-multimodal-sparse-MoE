"""Cluster environment check for the full OLMoE experiment path."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
from pathlib import Path
from typing import Dict

from hf_sources import load_pretrained


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    out_dir = Path(os.environ.get("OUT", "outputs/env_check"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "modules": {},
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }

    required = ["torch", "transformers", "datasets", "accelerate", "peft", "PIL", "yaml"]
    optional = ["soundfile", "librosa"]
    for name in required + optional:
        result["modules"][name] = module_available(name)

    errors = []
    if module_available("torch"):
        import torch

        result["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    else:
        errors.append("torch missing")

    if module_available("transformers"):
        import transformers
        from transformers import AutoConfig, AutoTokenizer

        result["transformers_version"] = transformers.__version__
        base_model = os.environ.get("BASE_MODEL", "allenai/OLMoE-1B-7B-0924")
        result["base_model"] = base_model
        try:
            cfg = load_pretrained(AutoConfig, base_model)
            result["olmoe_config"] = {
                "model_type": getattr(cfg, "model_type", None),
                "hidden_size": getattr(cfg, "hidden_size", None),
                "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
                "num_experts_per_tok": getattr(cfg, "num_experts_per_tok", None),
                "num_local_experts": getattr(cfg, "num_local_experts", None),
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"AutoConfig failed: {type(exc).__name__}: {exc}")
        try:
            tokenizer = load_pretrained(AutoTokenizer, base_model)
            result["tokenizer"] = {"vocab_size": len(tokenizer), "class": tokenizer.__class__.__name__}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"AutoTokenizer failed: {type(exc).__name__}: {exc}")
    else:
        errors.append("transformers missing")

    result["errors"] = errors
    (out_dir / "env_check.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if errors or any(not result["modules"].get(name, False) for name in required):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
