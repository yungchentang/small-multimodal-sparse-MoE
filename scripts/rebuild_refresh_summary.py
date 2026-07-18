"""Rebuild summary.json for a checkpoint-refreshed E3 output root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_summary = load_json(args.source_output_dir / "summary.json")
    refreshed_e3 = load_json(args.output_dir / "E3_final_multimodal_top2" / "metrics.json")
    summary = dict(source_summary)
    summary["E3"] = refreshed_e3
    refresh_path = args.output_dir / "e3_eval_refresh_provenance.json"
    if refresh_path.exists():
        summary["E3_refresh_provenance"] = load_json(refresh_path)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "E3_source": "refreshed_checkpoint_eval"}, sort_keys=True))


if __name__ == "__main__":
    main()
