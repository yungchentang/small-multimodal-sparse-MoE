"""Development-only multimodal representation diagnostics.

This entry point never accepts or reads sealed manifests. It reports retention
and representation geometry for a trained bridge checkpoint using real
development rows only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F

from scripts import eval_representation_funnel as funnel


FORBIDDEN_TERMS = ("sealed", "synthetic")


def _assert_development_artifact(path: Path) -> None:
    lowered = str(path.resolve(strict=False)).lower()
    if any(term in lowered for term in FORBIDDEN_TERMS):
        raise ValueError(f"development diagnostics reject sealed/synthetic path: {path}")


def _metrics_only(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if not key.endswith("_ranks")
    }


def direct_development_retrieval(features: funnel.RetrievalFeatures, stage: str) -> Dict[str, Any]:
    media = features.stages[stage].float()
    targets = features.targets.float()
    if media.shape[-1] != targets.shape[-1]:
        return {
            "available": False,
            "reason": "not_comparable_dimension",
            "media_dimension": int(media.shape[-1]),
            "target_dimension": int(targets.shape[-1]),
        }
    media = F.normalize(media, dim=-1)
    targets = F.normalize(targets, dim=-1)
    return {
        "available": True,
        **_metrics_only(
            funnel.evaluate_similarity(media @ targets.T, features.media_to_text)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gamma-json", type=Path)
    parser.add_argument("--image-dev-manifest", type=Path, required=True)
    parser.add_argument("--speech-dev-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("representation extraction requires one CUDA GPU")
    for path in (args.image_dev_manifest, args.speech_dev_manifest, args.output_dir):
        _assert_development_artifact(path)
    checkpoint = args.checkpoint.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "development_representation_diagnostics.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = (args.cache_dir or output_dir / "cache").resolve()

    manifests = {
        "image": args.image_dev_manifest.resolve(strict=True),
        "speech": args.speech_dev_manifest.resolve(strict=True),
    }
    records = {modality: funnel._read_records(path) for modality, path in manifests.items()}
    for modality, media_key in (("image", "image_path"), ("speech", "audio_path")):
        funnel._resolve_media_paths(records[modality], manifests[modality], media_key)

    (
        wrapper,
        tokenizer,
        image_processor,
        vision_model,
        speech_processor,
        speech_model,
        device,
        config,
        model_meta,
    ) = funnel.load_selected_wrapper(checkpoint, args.gamma_json)
    checkpoint_sha256 = funnel.sha256_file(checkpoint)
    code_path = Path(__file__).resolve()
    code_sha256 = funnel.sha256_file(code_path)
    gamma_candidate = (
        args.gamma_json.resolve()
        if args.gamma_json is not None
        else checkpoint.parent.parent / "calibration" / "gamma.json"
    )
    gamma_sha256 = funnel.sha256_file(gamma_candidate) if gamma_candidate.is_file() else None

    modality_reports: Dict[str, Any] = {}
    caches: Dict[str, Any] = {}
    for modality in ("image", "speech"):
        features = funnel.extract_or_load_features(
            wrapper=wrapper,
            tokenizer=tokenizer,
            image_processor=image_processor,
            vision_model=vision_model,
            speech_processor=speech_processor,
            speech_model=speech_model,
            rows=records[modality],
            manifest_path=manifests[modality],
            modality=modality,
            split_name="development",
            device=device,
            batch_size=args.batch_size,
            cache_root=cache_root,
            checkpoint_sha256=checkpoint_sha256,
            gamma_sha256=gamma_sha256,
            code_sha256=code_sha256,
            config=config,
        )
        stages: Dict[str, Any] = {}
        for stage_name in funnel.STAGES:
            prepared = funnel.prepare_stage(features, stage_name)
            stages[stage_name] = {
                **prepared.report,
                "direct_development_retrieval": direct_development_retrieval(
                    features, stage_name
                ),
            }
        modality_reports[modality] = {
            "media_count": len(features.media_ids),
            "target_count": len(features.text_ids),
            "pooling": features.pooling,
            "stages": stages,
        }
        caches[modality] = {
            "path": str(features.cache_path),
            "sha256": features.cache_sha256,
        }

    report = {
        "artifact_type": "development_only_representation_diagnostics",
        "development_only": True,
        "sealed_manifests_read": False,
        "synthetic_data_used": False,
        "used_for_sealed_claims": False,
        "checkpoint_config": {
            "base_model": config.base_model,
            "vision_model": config.vision_model,
            "speech_model": config.speech_model,
            "image_bridge_type": config.image_bridge_type,
            "audio_bridge_type": config.audio_bridge_type,
            "image_prefix_tokens": int(config.image_prefix_tokens),
            "audio_prefix_tokens": int(config.audio_prefix_tokens),
            "encoder_feature_tokens": int(config.encoder_feature_tokens),
            "model_load_meta": model_meta,
        },
        "modalities": modality_reports,
        "provenance": {
            "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
            "gamma": {
                "path": str(gamma_candidate) if gamma_candidate.is_file() else None,
                "sha256": gamma_sha256,
            },
            "manifests": {
                modality: {"path": str(path), "sha256": funnel.sha256_file(path)}
                for modality, path in manifests.items()
            },
            "evaluator_code": {"path": str(code_path), "sha256": code_sha256},
            "cached_features": caches,
        },
    }
    funnel._json_dump(report_path, report)
    print(json.dumps({"report": str(report_path), "sha256": funnel.sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
