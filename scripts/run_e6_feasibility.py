#!/usr/bin/env python3
"""Run exactly three expert-training steps from the selected E3 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_missing_ablations as missing


EXPERIMENT_ID = "E6_trainable_experts_ablation"
FEASIBILITY_STEPS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> Dict[str, Any]:
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not isinstance(state.get("args"), dict):
        raise ValueError("selected checkpoint must contain an args mapping")
    return state


def validate_steps(metrics: Dict[str, Any], source_sha256: str) -> list[Dict[str, Any]]:
    steps = metrics.get("steps")
    if not isinstance(steps, list) or len(steps) != FEASIBILITY_STEPS:
        raise ValueError("E6 must contain exactly three optimizer rows")
    if [row.get("step") for row in steps] != [1, 2, 3]:
        raise ValueError("E6 step identities must be exactly 1, 2, 3")
    if [row.get("modality") for row in steps] != ["text", "image", "speech"]:
        raise ValueError("E6 must exercise text, image, and speech in order")
    for row in steps:
        if row.get("optimizer_step") is not True:
            raise ValueError("every E6 row must complete an optimizer step")
        if row.get("initial_checkpoint_state_restored") is not True:
            raise ValueError("every E6 row must prove selected checkpoint restoration")
        if row.get("source_selected_checkpoint_sha256") != source_sha256:
            raise ValueError("E6 row source checkpoint hash mismatch")
    return steps


def validate_source_manifest(manifest: Dict[str, Any]) -> int:
    manifest_args = manifest.get("args")
    completion = manifest.get("completion")
    if not isinstance(manifest_args, dict) or not isinstance(completion, dict):
        raise ValueError("selected E3 source manifest is missing completion metadata")
    final_steps = int(manifest_args.get("final_steps", 0))
    if (
        final_steps <= 0
        or completion.get("status") != "completed"
        or int(completion.get("e3_steps", 0)) != final_steps
    ):
        raise ValueError("selected E3 source must be a completed positive-step root")
    return final_steps


def build_args(source_root: Path, output_root: Path, cli: argparse.Namespace) -> SimpleNamespace:
    overrides = SimpleNamespace(
        data_dir=cli.data_dir,
        base_model=None,
        vision_model=None,
        speech_model=None,
        feature_cache_dir=cli.feature_cache_dir,
        ablation_steps=None,
        capacity_ablation_steps=None,
        text_eval_blocks=cli.text_eval_blocks,
        retrieval_eval_samples=cli.retrieval_eval_samples,
        conditional_eval_samples=cli.conditional_eval_samples,
        conditional_batch_size=cli.conditional_batch_size,
    )
    args = missing.make_train_args(source_root, output_root, overrides)
    args.expert_ablation_steps = FEASIBILITY_STEPS
    args.modality_cycle = "text,image,speech"
    args.train_batch_size = 1
    args.eval_batch_size = 1
    args.contrastive_coef = 0.0
    args.image_contrastive_coef = 0.0
    args.speech_contrastive_coef = 0.0
    args.image_conditional_ranking_coef = 0.0
    args.speech_conditional_ranking_coef = 0.0
    args.train_router_gates = True
    args.train_experts = True
    args.save_expert_weights = False
    args.output_dir = str(output_root)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--text-eval-blocks", type=int, default=160)
    parser.add_argument("--retrieval-eval-samples", type=int, default=250)
    parser.add_argument("--conditional-eval-samples", type=int, default=250)
    parser.add_argument("--conditional-batch-size", type=int, default=16)
    cli = parser.parse_args()

    source_root = cli.source_output_dir.resolve(strict=True)
    output_root = cli.output_dir.resolve(strict=False)
    checkpoint = (cli.checkpoint or source_root / "E3_final_multimodal_top2" / "checkpoint_final.pt").resolve(strict=True)
    manifest_path = (source_root / "manifest.json").resolve(strict=True)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite E6 output: {output_root}")

    source_sha256 = sha256_file(checkpoint)
    if source_sha256 != cli.expected_checkpoint_sha256.lower():
        raise ValueError("selected checkpoint SHA-256 does not match the expected selector value")
    manifest = missing.read_json(manifest_path)
    source_steps = validate_source_manifest(manifest)

    state = load_checkpoint(checkpoint)
    output_root.mkdir(parents=True, exist_ok=False)
    args = build_args(source_root, output_root, cli)
    data_manifest, text_train, text_eval, image_train, image_eval, audio_train, audio_eval = missing.load_splits(args)
    gamma = missing.materialize_gamma(source_root, output_root)

    from training import olmoe_real_subset_runs as real

    initialization = {
        "path": str(checkpoint),
        "sha256": source_sha256,
        "size_bytes": checkpoint.stat().st_size,
        "source_experiment_id": "E3_final_multimodal_top2",
        "source_training_steps": source_steps,
        "state_restored": True,
    }
    metrics = real.train_real_multimodal(
        EXPERIMENT_ID,
        args,
        text_train,
        text_eval,
        image_train,
        image_eval,
        audio_train,
        audio_eval,
        gamma=gamma,
        aux_coef=float(args.aux_coef),
        capacity_factor=float(args.capacity_factor),
        max_steps=FEASIBILITY_STEPS,
        out_dir=output_root,
        train_router_gates=True,
        train_experts=True,
        initial_checkpoint_state=state,
        initial_checkpoint_provenance=initialization,
        development_split_provenance=args.development_split_provenance,
        evaluate_after_training=False,
    )
    steps = validate_steps(metrics, source_sha256)
    raw_metrics = output_root / EXPERIMENT_ID / "metrics.json"
    e6_checkpoint = output_root / EXPERIMENT_ID / "checkpoint_final.pt"
    artifact = {
        "schema_version": 2,
        "experiment_id": "E6_expert_training_feasibility",
        "scope": "feasibility_only",
        "feasibility_only": True,
        "full_ablation": False,
        "source_selected_root": str(source_root),
        "source_selected_checkpoint": str(checkpoint),
        "source_selected_checkpoint_sha256": source_sha256,
        "source_selected_checkpoint_size_bytes": checkpoint.stat().st_size,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "runai_job_name": os.environ.get("RUNAI_JOB_NAME"),
        "runai_project": os.environ.get("RUNAI_PROJECT"),
        "initialization": initialization,
        "optimizer": "AdamW",
        "objective_scope": "lm_ce_plus_router_aux_only",
        "post_training_evaluation_skipped": True,
        "modality_cycle": "text,image,speech",
        "train_router_gates": True,
        "train_experts": True,
        "train_lm_head": bool(args.train_lm_head),
        "trainable_params": metrics.get("meta", {}).get("trainable_params"),
        "optimizer_groups": metrics.get("meta", {}).get("optimizer_groups"),
        "data_manifest_sha256": hashlib.sha256(
            json.dumps(data_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "raw_metrics_path": str(raw_metrics.resolve(strict=True)),
        "raw_metrics_sha256": sha256_file(raw_metrics),
        "e6_checkpoint_path": str(e6_checkpoint.resolve(strict=True)),
        "e6_checkpoint_sha256": sha256_file(e6_checkpoint),
        "steps": steps,
    }
    output_path = output_root / "E6_expert_training_feasibility.json"
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "steps": len(steps), "scope": "feasibility_only"}, sort_keys=True))


if __name__ == "__main__":
    main()
