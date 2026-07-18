#!/usr/bin/env python3
"""Extract and cross-check checkpoint metadata without loading the model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping


IGNORED_ARG_MISMATCHES = {"output_dir", "feature_cache_dir"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_jsonl_objects(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank raw training row {line_number}")
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"raw training row {line_number} is not an object")
            rows.append(dict(row))
    if not rows:
        raise ValueError("raw training JSONL has no rows")
    return rows


def compare_args(
    checkpoint_args: Mapping[str, Any], manifest_args: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    mismatches: Dict[str, Dict[str, Any]] = {}
    for key in sorted(set(checkpoint_args) | set(manifest_args)):
        if key in IGNORED_ARG_MISMATCHES:
            continue
        checkpoint_value = checkpoint_args.get(key, "<missing>")
        manifest_value = manifest_args.get(key, "<missing>")
        if checkpoint_value != manifest_value:
            mismatches[key] = {
                "checkpoint": checkpoint_value,
                "manifest": manifest_value,
            }
    return mismatches


def _recorded_path_matches(
    value: Any, checkpoint: Path, metrics_path: Path
) -> bool:
    if not isinstance(value, (str, Path)) or not str(value):
        return False
    raw = Path(value).expanduser()
    candidates = [raw.resolve(strict=False)]
    if not raw.is_absolute():
        candidates.append((metrics_path.parent / raw).resolve(strict=False))
    return checkpoint in candidates


def validate_e3_metrics_checkpoint_identity(
    metrics: Mapping[str, Any], checkpoint_path: Path, metrics_path: Path
) -> Dict[str, Any]:
    checkpoint = checkpoint_path.resolve(strict=True)
    metrics_file = metrics_path.resolve(strict=False)
    checkpoint_sha256 = sha256_file(checkpoint)
    bindings = [
        (
            "E3 metrics",
            metrics.get("checkpoint_path"),
            metrics.get("checkpoint_sha256"),
        )
    ]
    text_provenance = metrics.get("text_eval_provenance")
    if not isinstance(text_provenance, Mapping):
        raise ValueError("E3 metrics are missing text_eval_provenance")
    bindings.append(
        (
            "E3 embedded text provenance",
            text_provenance.get("source_checkpoint"),
            text_provenance.get("source_checkpoint_sha256"),
        )
    )
    for label, recorded_path, recorded_sha256 in bindings:
        if not _recorded_path_matches(recorded_path, checkpoint, metrics_file):
            raise ValueError(f"{label} checkpoint path mismatch")
        if recorded_sha256 != checkpoint_sha256:
            raise ValueError(f"{label} checkpoint SHA-256 mismatch")
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch

    checkpoint = args.checkpoint.resolve(strict=True)
    run_manifest = args.run_manifest.resolve(strict=True)
    try:
        state = torch.load(checkpoint, map_location="meta", weights_only=False)
    except (TypeError, RuntimeError):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(state, dict) or not isinstance(state.get("args"), dict):
        raise ValueError("checkpoint does not contain an args mapping")
    checkpoint_args = dict(state["args"])
    checkpoint_last_row = state.get("last_row")
    if not isinstance(checkpoint_last_row, dict):
        raise ValueError("checkpoint does not contain a last_row mapping")
    metrics_path = checkpoint.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validate_e3_metrics_checkpoint_identity(metrics, checkpoint, metrics_path)
    metrics_steps = metrics.get("steps")
    if (
        not isinstance(metrics_steps, list)
        or not metrics_steps
        or not all(isinstance(row, Mapping) for row in metrics_steps)
    ):
        raise ValueError("E3 metrics does not contain non-empty steps")
    metrics_steps = [dict(row) for row in metrics_steps]
    raw_training_path = checkpoint.parent / "train_metrics.jsonl"
    raw_training_rows = load_jsonl_objects(raw_training_path)
    if metrics_steps != raw_training_rows:
        raise ValueError("E3 metrics steps do not equal raw training JSONL rows")
    metrics_last_row = metrics_steps[-1]
    if checkpoint_last_row != metrics_last_row:
        raise ValueError("checkpoint last_row does not equal E3 metrics final row")

    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    manifest_args = manifest.get("args")
    if not isinstance(manifest_args, dict):
        raise ValueError("run manifest does not contain an args mapping")
    mismatches = compare_args(checkpoint_args, manifest_args)
    if mismatches:
        raise ValueError(f"checkpoint/manifest argument mismatches: {mismatches}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    args_output = args.output_dir / "checkpoint_args.json"
    provenance_output = args.output_dir / "checkpoint_provenance.json"
    args_output.write_text(
        json.dumps(checkpoint_args, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_state_keys": sorted(str(key) for key in state),
        "checkpoint_last_row_sha256": hashlib.sha256(
            json.dumps(checkpoint_last_row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "checkpoint_last_row_summary": {
            key: checkpoint_last_row.get(key)
            for key in (
                "step",
                "modality",
                "loss",
                "lm_ce_loss",
                "router_aux_loss_raw",
                "router_aux_loss_weighted",
                "hf_reported_loss_minus_explicit_base",
            )
        },
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "metrics_step_rows": len(metrics_steps),
        "metrics_steps_sha256": canonical_sha256(metrics_steps),
        "raw_training_path": str(raw_training_path.resolve()),
        "raw_training_sha256": sha256_file(raw_training_path),
        "raw_training_rows": len(raw_training_rows),
        "training_rows_sha256": canonical_sha256(raw_training_rows),
        "checkpoint_last_row_matches_metrics_last_row": True,
        "checkpoint_args_path": str(args_output.resolve()),
        "checkpoint_args_sha256": sha256_file(args_output),
        "run_manifest_path": str(run_manifest),
        "run_manifest_sha256": sha256_file(run_manifest),
        "ignored_path_only_arg_keys": sorted(IGNORED_ARG_MISMATCHES),
        "non_path_arg_mismatches": mismatches,
        "passed": not mismatches,
    }
    provenance_output.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(provenance_output), "passed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
