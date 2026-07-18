#!/usr/bin/env python
"""Run missing E4/E5 real-data ablations for an existing final root.

This entrypoint intentionally reuses ``training.olmoe_real_subset_runs`` for
preprocessing, CLIP/Whisper loading, multimodal prefix training, and evaluation.
It only avoids rerunning completed E0-E3 artifacts when a candidate sweep was
configured with zero ablation steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_REAL_MODULE = None


def get_real_module():
    global _REAL_MODULE
    if _REAL_MODULE is None:
        from training import olmoe_real_subset_runs as real_module

        _REAL_MODULE = real_module
    return _REAL_MODULE


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


DEFAULTS: Dict[str, Any] = {
    "data_dir": "data/real_subset_clean_260708b",
    "base_model": "allenai/OLMoE-1B-7B-0924",
    "vision_model": "openai/clip-vit-base-patch32",
    "speech_model": "openai/whisper-base.en",
    "speech_target_space": "olmoe_text_hidden",
    "alignment_prefix_residual": False,
    "output_dir": "outputs/real_required_runs",
    "feature_cache_dir": "",
    "seed": 42,
    "max_length": 512,
    "capacity_factor": 4.0,
    "capacity_ablation_factor": 1.25,
    "aux_coef": 0.01,
    "router_z_loss_coef": 0.0,
    "expert_dropout_prob": 0.0,
    "dynamic_expert_bias_lr": 0.0,
    "dynamic_expert_bias_update_interval": 1,
    "dynamic_expert_bias_warmup_steps": 0,
    "dynamic_expert_bias_max_abs": 2.0,
    "gamma_min": 0.25,
    "gamma_max": 2.0,
    "final_steps": 0,
    "ablation_steps": 300,
    "capacity_ablation_steps": 300,
    "expert_ablation_steps": 0,
    "train_batch_size": 4,
    "eval_batch_size": 8,
    "modality_cycle": "text,image,speech",
    "text_eval_blocks": 160,
    "retrieval_eval_samples": 250,
    "conditional_eval_samples": 250,
    "conditional_negatives": 4,
    "conditional_batch_size": 16,
    "conditional_ranking_negatives": 2,
    "conditional_ranking_negative_mode": "stride",
    "conditional_ranking_hard_pool_size": 512,
    "conditional_ranking_temperature": 1.0,
    "image_conditional_ranking_coef": 0.0,
    "speech_conditional_ranking_coef": 0.0,
    "image_eval_samples": 250,
    "speech_eval_samples": 250,
    "image_prefix_tokens": 50,
    "audio_prefix_tokens": 64,
    "encoder_feature_tokens": 100,
    "sample_rate": 16000,
    "learning_rate": 2e-4,
    "router_learning_rate": 5e-5,
    "expert_learning_rate": 5e-5,
    "retrieval_head_learning_rate": 0.0,
    "lm_head_learning_rate": 1e-5,
    "contrastive_coef": 0.1,
    "image_contrastive_coef": -1.0,
    "speech_contrastive_coef": -1.0,
    "center_positive_weight": 1.0,
    "raw_positive_weight": 1.0,
    "image_center_positive_weight": -1.0,
    "image_raw_positive_weight": -1.0,
    "speech_center_positive_weight": -1.0,
    "speech_raw_positive_weight": -1.0,
    "contrastive_temperature": 0.07,
    "image_contrastive_temperature": -1.0,
    "speech_contrastive_temperature": -1.0,
    "contrastive_negatives": 128,
    "image_contrastive_negatives": -2,
    "speech_contrastive_negatives": -2,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "log_every_steps": 20,
    "save_every_steps": 500,
    "alignment_pretrain_steps": 0,
    "alignment_pretrain_log_every": 100,
    "train_router_gates": False,
    "train_experts": False,
    "train_lm_head": True,
    "save_expert_weights": False,
}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_checkpoint(path: Path) -> Dict[str, Any]:
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not isinstance(state.get("args"), dict):
        raise ValueError("selected E3 checkpoint must contain an args mapping")
    return state


def write_matched_final_artifact(
    role: str,
    experiment_id: str,
    metrics: Dict[str, Any],
    out_dir: Path,
    source_checkpoint: Path,
    source_checkpoint_sha256: str,
) -> Path:
    steps = metrics.get("steps")
    if not isinstance(steps, list) or len(steps) != 300:
        raise ValueError(f"{role} final continuation must contain exactly 300 steps")
    if [row.get("step") for row in steps] != list(range(1, 301)):
        raise ValueError(f"{role} final continuation steps must be contiguous 1..300")
    optimizer_rows = []
    frozen_text_rows = []
    for row in steps:
        if row.get("optimizer_step") is True:
            optimizer_rows.append(row)
        elif (
            row.get("optimizer_step") is False
            and row.get("modality") == "text"
            and row.get("train_router_gates") is False
            and row.get("train_experts") is False
            and row.get("train_lm_head") is False
        ):
            frozen_text_rows.append(row)
        else:
            raise ValueError(f"{role} contains an unexplained non-optimizer row")
        if row.get("initial_checkpoint_state_restored") is not True:
            raise ValueError(f"{role} did not restore the selected checkpoint")
        if row.get("source_selected_checkpoint_sha256") != source_checkpoint_sha256:
            raise ValueError(f"{role} source checkpoint hash mismatch")
    if not optimizer_rows:
        raise ValueError(f"{role} contains no optimizer updates")
    if any(row.get("modality") not in {"image", "speech"} for row in optimizer_rows):
        raise ValueError(f"{role} optimizer rows have an unexpected modality")
    if len(optimizer_rows) + len(frozen_text_rows) != len(steps):
        raise ValueError(f"{role} optimizer/frozen-text accounting mismatch")

    experiment_dir = out_dir / experiment_id
    raw_metrics_path = (experiment_dir / "metrics.json").resolve(strict=True)
    checkpoint_path = Path(str(metrics["checkpoint_path"])).resolve(strict=True)
    artifact = {
        **metrics,
        "schema_version": 1,
        "artifact_type": "matched_ablation_final",
        "experiment_role": role,
        "protocol": "selected_checkpoint_continuation_300",
        "claim_scope": "continuation_sensitivity_only",
        "training_iterations": len(steps),
        "optimizer_step_count": len(optimizer_rows),
        "frozen_text_row_count": len(frozen_text_rows),
        "frozen_text_row_semantics": (
            "retention-only forward rows with frozen router, experts, and LM head; "
            "not counted as optimizer updates"
        ),
        "source_selected_checkpoint": str(source_checkpoint),
        "source_selected_checkpoint_sha256": source_checkpoint_sha256,
        "capacity_factor": float(metrics.get("meta", {}).get("capacity_factor")),
        "raw_metrics": {
            "path": str(raw_metrics_path),
            "sha256": sha256_file(raw_metrics_path),
            "size_bytes": raw_metrics_path.stat().st_size,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
    }
    output = experiment_dir / "matched_final_metrics.json"
    save_json(output, artifact)
    return output


def coerce_like(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return str(value)


def make_train_args(source_root: Path, out_dir: Path, cli_args: argparse.Namespace) -> SimpleNamespace:
    source_manifest = read_json(source_root / "manifest.json")
    source_args = source_manifest.get("args", {}) if isinstance(source_manifest.get("args"), dict) else {}
    merged = dict(DEFAULTS)
    for key, value in source_args.items():
        merged[key] = (
            coerce_like(value, DEFAULTS[key]) if key in DEFAULTS else value
        )
    merged["output_dir"] = str(out_dir)
    overrides = {
        "data_dir": cli_args.data_dir,
        "base_model": cli_args.base_model,
        "vision_model": cli_args.vision_model,
        "speech_model": cli_args.speech_model,
        "feature_cache_dir": cli_args.feature_cache_dir,
        "ablation_steps": cli_args.ablation_steps,
        "capacity_ablation_steps": cli_args.capacity_ablation_steps,
        "text_eval_blocks": cli_args.text_eval_blocks,
        "retrieval_eval_samples": cli_args.retrieval_eval_samples,
        "conditional_eval_samples": cli_args.conditional_eval_samples,
        "conditional_batch_size": cli_args.conditional_batch_size,
    }
    for key, value in overrides.items():
        if value not in (None, ""):
            merged[key] = coerce_like(value, DEFAULTS[key])
    if not merged.get("feature_cache_dir"):
        merged["feature_cache_dir"] = str(out_dir / "feature_cache")
    return SimpleNamespace(**merged)


def load_splits(args: SimpleNamespace):
    real = get_real_module()
    data_dir = Path(args.data_dir)
    data_manifest = real.load_manifest(data_dir)
    text_train_blocks = real.read_jsonl(data_dir / "text_blocks_train.jsonl")
    text_eval_blocks = real.read_jsonl(data_dir / "text_blocks_eval.jsonl")
    development_manifest = str(getattr(args, "development_split_manifest", "") or "")
    split_provenance: Dict[str, Any] = {
        "strict_manifest_verified": False,
        "reserved_files_opened": None,
    }
    if development_manifest:
        if int(args.image_eval_samples) != 137 or int(args.speech_eval_samples) != 137:
            raise ValueError(
                "explicit development split requires image/speech eval samples = 137"
            )
        image_train, image_eval, audio_train, audio_eval, split_provenance = (
            real.load_development_multimodal_partitions(
                development_manifest,
                expected_data_dir=data_dir,
                expected_speech_source_sha256=str(
                    getattr(args, "development_speech_source_sha256", "") or ""
                ),
            )
        )
        if split_provenance.get("reserved_files_opened") is not False:
            raise ValueError("development ablation opened the reserved eval split")
    else:
        image_rows = real.read_jsonl(data_dir / "image_captions.jsonl")
        audio_rows = real.read_jsonl(data_dir / "speech_transcripts.jsonl")
        real.absolutize_media_paths(image_rows, data_dir)
        real.absolutize_media_paths(audio_rows, data_dir)
        image_train, image_eval = real.split_tail(
            image_rows, int(args.image_eval_samples)
        )
        audio_train, audio_eval = real.split_tail(
            audio_rows, int(args.speech_eval_samples)
        )
    args.development_split_provenance = split_provenance
    if not text_train_blocks or not text_eval_blocks or not image_train or not audio_train:
        raise RuntimeError("Real-data manifest has empty train/eval splits")
    return data_manifest, text_train_blocks, text_eval_blocks, image_train, image_eval, audio_train, audio_eval


def materialize_gamma(source_root: Path, out_dir: Path) -> List[float]:
    raw_source = source_root / "calibration" / "gamma.json"
    if raw_source.is_symlink():
        raise ValueError("source gamma JSON must not be a symlink")
    gamma_path = raw_source.resolve(strict=True)
    payload = gamma_path.read_bytes()
    data = json.loads(payload)
    gamma = data.get("gamma")
    if not isinstance(gamma, list) or not gamma:
        raise FileNotFoundError(f"Missing gamma list in {gamma_path}")
    target = out_dir / "calibration" / "gamma.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or target.read_bytes() != payload:
            raise ValueError("ablation gamma JSON disagrees with selected source")
    else:
        temporary = target.with_suffix(".json.tmp")
        if temporary.exists():
            raise FileExistsError(f"refusing stale gamma temporary file: {temporary}")
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    if sha256_file(target) != sha256_file(gamma_path):
        raise ValueError("materialized gamma hash mismatch")
    return [float(x) for x in gamma]


def update_summary(out_dir: Path, source_root: Path, metrics: Dict[str, Dict[str, Any]], args: SimpleNamespace, data_manifest: Dict[str, Any]) -> None:
    summary_path = out_dir / "summary.json"
    summary = read_json(summary_path)
    if not summary:
        source_summary = read_json(source_root / "summary.json")
        summary = {k: v for k, v in source_summary.items() if k in {"manifest", "E0", "E1", "E2", "E3"}}
    manifest = summary.get("manifest") or read_json(source_root / "manifest.json")
    if isinstance(manifest, dict):
        manifest = dict(manifest)
        manifest.setdefault("data_manifest", data_manifest)
        summary["manifest"] = manifest
    for key, value in metrics.items():
        summary[key] = value
    provenance = {
        "type": "ablation_only_completion",
        "source_output_dir": str(source_root),
        "output_dir": str(out_dir),
        "runai_job_name": os.environ.get("RUNAI_JOB_NAME"),
        "runai_project": os.environ.get("RUNAI_PROJECT"),
        "reused_completed_experiments": [key for key in ["E0", "E1", "E2", "E3"] if key in summary],
        "added_experiments": sorted(metrics),
        "data_dir": str(args.data_dir),
        "capacity_factor": float(args.capacity_factor),
        "capacity_ablation_factor": float(args.capacity_ablation_factor),
        "aux_coef": float(args.aux_coef),
        "ablation_steps": int(args.ablation_steps),
        "capacity_ablation_steps": int(args.capacity_ablation_steps),
    }
    summary["ablation_only_provenance"] = provenance
    save_json(summary_path, summary)
    save_json(out_dir / "ablation_only_provenance.json", provenance)
    print(json.dumps({"summary_path": str(summary_path), **provenance}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output-dir", default="", help="Existing root with E0-E3 and calibration/gamma.json. Defaults to --output-dir.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiments", default="E4,E5", help="Comma-separated subset of E4,E5.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--vision-model", default=None)
    parser.add_argument("--speech-model", default=None)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--ablation-steps", type=int, default=None)
    parser.add_argument("--capacity-ablation-steps", type=int, default=None)
    parser.add_argument("--text-eval-blocks", type=int, default=None)
    parser.add_argument("--retrieval-eval-samples", type=int, default=None)
    parser.add_argument("--conditional-eval-samples", type=int, default=None)
    parser.add_argument("--conditional-batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    out_dir = Path(cli_args.output_dir)
    source_root = Path(cli_args.source_output_dir) if cli_args.source_output_dir else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_args = make_train_args(source_root, out_dir, cli_args)
    real = get_real_module()
    real.set_seed(int(train_args.seed))
    data_manifest, text_train_blocks, text_eval_blocks, image_train, image_eval, audio_train, audio_eval = load_splits(train_args)
    gamma = materialize_gamma(source_root, out_dir)
    selected_checkpoint = (source_root / "E3_final_multimodal_top2" / "checkpoint_final.pt").resolve(strict=True)
    selected_checkpoint_sha256 = sha256_file(selected_checkpoint)
    selected_state = load_selected_checkpoint(selected_checkpoint)
    source_steps = int(selected_state["args"].get("final_steps", 0))
    source_metrics = read_json(source_root / "E3_final_multimodal_top2" / "metrics.json")
    source_rows = source_metrics.get("steps")
    if (
        source_steps <= 0
        or not isinstance(source_rows, list)
        or len(source_rows) != source_steps
        or source_rows[-1].get("step") != source_steps
        or selected_state.get("last_row") != source_rows[-1]
        or source_metrics.get("checkpoint_sha256") != selected_checkpoint_sha256
    ):
        raise ValueError("selected E3 checkpoint is not bound to a completed source run")
    initialization = {
        "path": str(selected_checkpoint),
        "sha256": selected_checkpoint_sha256,
        "size_bytes": selected_checkpoint.stat().st_size,
        "source_experiment_id": "E3_final_multimodal_top2",
        "source_training_steps": source_steps,
        "state_restored": True,
    }
    requested = {item.strip().upper() for item in str(cli_args.experiments).split(",") if item.strip()}
    unknown = requested - {"E4", "E5"}
    if unknown:
        raise ValueError(f"Unsupported ablation experiments: {sorted(unknown)}")
    metrics: Dict[str, Dict[str, Any]] = {}
    if "E4" in requested:
        metric_path = out_dir / "E4_no_aux_load_balance_ablation" / "metrics.json"
        if cli_args.skip_existing and metric_path.exists():
            metrics["E4"] = read_json(metric_path)
            write_matched_final_artifact(
                "E4",
                "E4_no_aux_load_balance_ablation",
                metrics["E4"],
                out_dir,
                selected_checkpoint,
                selected_checkpoint_sha256,
            )
        else:
            if int(train_args.ablation_steps) <= 0:
                raise ValueError("E4 requested but ablation_steps <= 0")
            metrics["E4"] = real.train_real_multimodal(
                "E4_no_aux_load_balance_ablation",
                train_args,
                text_train_blocks,
                text_eval_blocks,
                image_train,
                image_eval,
                audio_train,
                audio_eval,
                gamma=gamma,
                aux_coef=0.0,
                capacity_factor=float(train_args.capacity_factor),
                max_steps=int(train_args.ablation_steps),
                out_dir=out_dir,
                train_router_gates=bool(train_args.train_router_gates),
                train_experts=False,
                development_split_provenance=train_args.development_split_provenance,
                initial_checkpoint_state=selected_state,
                initial_checkpoint_provenance=initialization,
            )
            write_matched_final_artifact(
                "E4",
                "E4_no_aux_load_balance_ablation",
                metrics["E4"],
                out_dir,
                selected_checkpoint,
                selected_checkpoint_sha256,
            )
    if "E5" in requested:
        metric_path = out_dir / "E5_capacity_1p25_ablation" / "metrics.json"
        if cli_args.skip_existing and metric_path.exists():
            metrics["E5"] = read_json(metric_path)
            write_matched_final_artifact(
                "E5",
                "E5_capacity_1p25_ablation",
                metrics["E5"],
                out_dir,
                selected_checkpoint,
                selected_checkpoint_sha256,
            )
        else:
            if int(train_args.capacity_ablation_steps) <= 0:
                raise ValueError("E5 requested but capacity_ablation_steps <= 0")
            metrics["E5"] = real.train_real_multimodal(
                "E5_capacity_1p25_ablation",
                train_args,
                text_train_blocks,
                text_eval_blocks,
                image_train,
                image_eval,
                audio_train,
                audio_eval,
                gamma=gamma,
                aux_coef=float(train_args.aux_coef),
                capacity_factor=float(train_args.capacity_ablation_factor),
                max_steps=int(train_args.capacity_ablation_steps),
                out_dir=out_dir,
                train_router_gates=bool(train_args.train_router_gates),
                train_experts=False,
                development_split_provenance=train_args.development_split_provenance,
                initial_checkpoint_state=selected_state,
                initial_checkpoint_provenance=initialization,
            )
            write_matched_final_artifact(
                "E5",
                "E5_capacity_1p25_ablation",
                metrics["E5"],
                out_dir,
                selected_checkpoint,
                selected_checkpoint_sha256,
            )
    update_summary(out_dir, source_root, metrics, train_args, data_manifest)


if __name__ == "__main__":
    main()
