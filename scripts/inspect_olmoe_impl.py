"""Inspect the installed HuggingFace OLMoE implementation inside the Run:AI image."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/inspect_olmoe_impl")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import transformers
    import transformers.models.olmoe.modeling_olmoe as modeling

    names = [
        "OlmoeSparseMoeBlock",
        "OlmoeTopKRouter",
        "OlmoeExperts",
        "OlmoeDecoderLayer",
        "OlmoeModel",
    ]
    meta: Dict[str, Any] = {"transformers_version": transformers.__version__, "modeling_file": inspect.getfile(modeling), "classes": {}}
    text_parts = [f"transformers={transformers.__version__}", f"file={inspect.getfile(modeling)}"]
    for name in names:
        if not hasattr(modeling, name):
            continue
        obj = getattr(modeling, name)
        try:
            source = inspect.getsource(obj)
        except Exception as exc:
            source = f"<source unavailable: {exc!r}>"
        meta["classes"][name] = {
            "module": getattr(obj, "__module__", ""),
            "signature": str(inspect.signature(obj)) if callable(obj) else "",
            "source_chars": len(source),
        }
        text_parts.append(f"\n===== {name} =====\n{source}")
    (out_dir / "olmoe_impl.txt").write_text("\n".join(text_parts), encoding="utf-8")
    (out_dir / "olmoe_impl_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(out_dir / "olmoe_impl.txt"), **meta}, sort_keys=True))


if __name__ == "__main__":
    main()
