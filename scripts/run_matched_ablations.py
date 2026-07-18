#!/usr/bin/env python
"""Run the three-arm, 300-step matched early-training ablation protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_missing_ablations as missing


STEPS = 300
PRE_PROTOCOL = "matched_ablation_protocol.pre.json"
POST_PROTOCOL = "matched_ablation_protocol.post.json"
INITIALIZATION_POLICY = "reset_global_seed_before_each_matched_arm"
ARM_SPECS: Tuple[Tuple[str, float, float], ...] = (
    ("E3_aux_cap7_300", 0.02, 7.0),
    ("E4_noaux_cap7_300", 0.0, 7.0),
    ("E5_aux_cap1p25_300", 0.02, 1.25),
)
PAIR_ALLOWED_DIFFERENCES: Dict[Tuple[str, str], Set[str]] = {
    ("E3_aux_cap7_300", "E4_noaux_cap7_300"): {"experiment_id", "aux_coef"},
    ("E3_aux_cap7_300", "E5_aux_cap1p25_300"): {"experiment_id", "capacity_factor"},
    ("E4_noaux_cap7_300", "E5_aux_cap1p25_300"): {
        "experiment_id",
        "aux_coef",
        "capacity_factor",
    },
}

ARG_EXCLUSIONS = {
    "output_dir",
    "final_steps",
    "ablation_steps",
    "capacity_ablation_steps",
    "expert_ablation_steps",
    "capacity_factor",
    "capacity_ablation_factor",
    "aux_coef",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: Iterable[Path]) -> Dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in paths}


def _different_paths(reference: Any, candidate: Any, prefix: str = "") -> List[str]:
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        paths: List[str] = []
        for key in sorted(set(reference) | set(candidate)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in reference or key not in candidate:
                paths.append(path)
            else:
                paths.extend(_different_paths(reference[key], candidate[key], path))
        return paths
    return [] if reference == candidate else [prefix or "<root>"]


def compare_arm_protocols(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    allowed_fields: Iterable[str],
) -> List[str]:
    """Return forbidden differing field paths between two arm contracts."""
    allowed = set(allowed_fields)
    return [
        path
        for path in _different_paths(reference, candidate)
        if not any(path == field or path.startswith(field + ".") for field in allowed)
    ]


def verify_matched_arm_contracts(arms: Sequence[Mapping[str, Any]]) -> List[str]:
    by_name = {str(arm["experiment_id"]): arm for arm in arms}
    errors: List[str] = []
    expected_names = {name for name, _, _ in ARM_SPECS}
    if set(by_name) != expected_names:
        return [f"arm set differs: expected={sorted(expected_names)} actual={sorted(by_name)}"]
    for (left_name, right_name), allowed in PAIR_ALLOWED_DIFFERENCES.items():
        forbidden = compare_arm_protocols(by_name[left_name], by_name[right_name], allowed)
        if forbidden:
            errors.append(f"{left_name} vs {right_name} forbidden differences: {forbidden}")
    return errors


def filtered_args(args: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in sorted(args.items()) if key not in ARG_EXCLUSIONS}


def load_checkpoint_args(path: Path) -> Dict[str, Any]:
    """Read checkpoint metadata without materializing checkpoint tensors."""
    import torch

    try:
        state = torch.load(path, map_location="meta", weights_only=False)
    except (TypeError, RuntimeError):
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    args = state.get("args") if isinstance(state, dict) else None
    if not isinstance(args, dict):
        raise ValueError(f"Checkpoint has no args mapping: {path}")
    return dict(args)


def make_args(source_root: Path, output_root: Path, seed: int) -> SimpleNamespace:
    overrides = SimpleNamespace(
        data_dir=None,
        base_model=None,
        vision_model=None,
        speech_model=None,
        feature_cache_dir=None,
        ablation_steps=None,
        capacity_ablation_steps=None,
        text_eval_blocks=None,
        retrieval_eval_samples=None,
        conditional_eval_samples=None,
        conditional_batch_size=None,
    )
    args = missing.make_train_args(source_root, output_root, overrides)
    args.seed = int(seed)
    args.output_dir = str(output_root)
    args.capacity_factor = 7.0
    args.capacity_ablation_factor = 1.25
    args.final_steps = STEPS
    args.ablation_steps = STEPS
    args.capacity_ablation_steps = STEPS
    return args


def expected_data_order(
    args: SimpleNamespace,
    text_train: Sequence[Any],
    image_train: Sequence[Any],
    audio_train: Sequence[Any],
) -> Dict[str, Any]:
    cycle = [item.strip() for item in str(args.modality_cycle).split(",") if item.strip()]
    lengths = {"text": len(text_train), "image": len(image_train), "speech": len(audio_train)}
    rows = []
    for step in range(1, STEPS + 1):
        modality = cycle[(step - 1) % len(cycle)]
        size = lengths[modality]
        indices = [(step * int(args.train_batch_size) + offset) % size for offset in range(int(args.train_batch_size))]
        rows.append({"step": step, "modality": modality, "indices": indices})
    return {
        "algorithm": "sample_cycle(start=step*train_batch_size)",
        "steps": rows,
        "sha256": canonical_sha256(rows),
    }


def arm_contract(
    name: str,
    aux_coef: float,
    capacity_factor: float,
    args: SimpleNamespace,
    hashes: Mapping[str, Any],
    order_hash: str,
) -> Dict[str, Any]:
    return {
        "experiment_id": name,
        "aux_coef": float(aux_coef),
        "capacity_factor": float(capacity_factor),
        "max_steps": STEPS,
        "seed": int(args.seed),
        "initialization_policy": INITIALIZATION_POLICY,
        "initialization_source": "fresh base_model load; source checkpoint is configuration/provenance only",
        "expected_data_order_sha256": order_hash,
        "source_checkpoint_sha256": hashes["source_checkpoint"],
        "gamma_sha256": hashes["gamma"],
        "matched_args": filtered_args(vars(args)),
        "train_router_gates": bool(args.train_router_gates),
        "train_experts": bool(args.train_experts),
        "train_lm_head": bool(args.train_lm_head),
    }


def build_pre_protocol(source_root: Path, output_root: Path, seed: int) -> Tuple[Dict[str, Any], SimpleNamespace, Tuple[Any, ...], List[float]]:
    manifest_path = source_root / "manifest.json"
    gamma_path = source_root / "calibration" / "gamma.json"
    checkpoint_path = source_root / "E3_final_multimodal_top2" / "checkpoint_final.pt"
    for path in (manifest_path, gamma_path, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(path)

    source_manifest = missing.read_json(manifest_path)
    manifest_args = source_manifest.get("args")
    if not isinstance(manifest_args, dict):
        raise ValueError(f"Missing args mapping in {manifest_path}")
    checkpoint_args = load_checkpoint_args(checkpoint_path)
    common_keys = sorted(set(manifest_args) & set(checkpoint_args) - {"output_dir"})
    checkpoint_mismatches = [key for key in common_keys if manifest_args[key] != checkpoint_args[key]]

    args = make_args(source_root, output_root, seed)
    if float(manifest_args.get("capacity_factor", -1.0)) != 7.0:
        raise ValueError("Selected source root must use capacity_factor=7.0")
    source_aux = float(manifest_args.get("aux_coef", -1.0))
    if source_aux <= 0.0:
        raise ValueError("Selected source root must provide a positive aux_coef")

    splits = missing.load_splits(args)
    data_manifest, text_train, text_eval, image_train, image_eval, audio_train, audio_eval = splits
    gamma = missing.load_gamma(source_root)
    order = expected_data_order(args, text_train, image_train, audio_train)

    data_dir = Path(args.data_dir)
    data_paths = [
        data_dir / "manifest.json",
        data_dir / "text_tasks.jsonl",
        data_dir / "text_blocks_train.jsonl",
        data_dir / "text_blocks_eval.jsonl",
        data_dir / "image_captions.jsonl",
        data_dir / "speech_transcripts.jsonl",
    ]
    code_paths = [
        Path(__file__),
        REPO_ROOT / "scripts" / "run_missing_ablations.py",
        REPO_ROOT / "training" / "olmoe_real_subset_runs.py",
        REPO_ROOT / "training" / "olmoe_required_runs.py",
        REPO_ROOT / "model" / "olmoe_adapter.py",
    ]
    hashes: Dict[str, Any] = {
        "code": hash_files(code_paths),
        "config": hash_files([manifest_path, gamma_path]),
        "data_manifests": hash_files(data_paths),
        "source_checkpoint": sha256_file(checkpoint_path),
        "gamma": canonical_sha256(gamma),
        "source_manifest_args": canonical_sha256(manifest_args),
        "source_checkpoint_args": canonical_sha256(checkpoint_args),
        "embedded_data_manifest": canonical_sha256(data_manifest),
    }
    arms = [
        arm_contract(name, source_aux if aux > 0.0 else 0.0, capacity, args, hashes, order["sha256"])
        for name, aux, capacity in ARM_SPECS
    ]
    protocol = {
        "schema_version": 1,
        "protocol": "matched_early_training_aux_capacity_300",
        "created_at": utc_now(),
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_manifest_checkpoint_arg_mismatches": checkpoint_mismatches,
        "hashes": hashes,
        "seed": int(seed),
        "runai_job_name": os.environ.get("RUNAI_JOB_NAME"),
        "runai_project": os.environ.get("RUNAI_PROJECT"),
        "allowed_differences": {
            f"{left}_vs_{right}": sorted(fields)
            for (left, right), fields in PAIR_ALLOWED_DIFFERENCES.items()
        },
        "expected_data_order": order,
        "arms": arms,
    }
    return protocol, args, splits, gamma


def observed_contract(expected: Mapping[str, Any], metrics: Mapping[str, Any], checkpoint_args: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    contract = copy.deepcopy(dict(expected))
    errors: List[str] = []
    rows = metrics.get("steps")
    if not isinstance(rows, list):
        return contract, [f"{expected['experiment_id']}: metrics.steps is not a list"]
    meta = metrics.get("meta") if isinstance(metrics.get("meta"), dict) else {}
    successful = sum(row.get("optimizer_step") is True for row in rows if isinstance(row, dict))
    step_numbers = [row.get("step") for row in rows if isinstance(row, dict)]
    modalities = [row.get("modality") for row in rows if isinstance(row, dict)]
    first = rows[0] if rows else {}
    contract.update(
        {
            "observed_steps": len(rows),
            "successful_optimizer_steps": successful,
            "observed_step_sequence_sha256": canonical_sha256(step_numbers),
            "observed_modality_sequence_sha256": canonical_sha256(modalities),
            "optimizer_groups": first.get("optimizer_groups"),
            "observed_trainable_mask": {
                "train_router_gates": first.get("train_router_gates"),
                "train_experts": first.get("train_experts"),
                "train_lm_head": first.get("train_lm_head"),
                "trainable_params": first.get("trainable_params"),
            },
            "observed_checkpoint_args": filtered_args(checkpoint_args),
        }
    )
    expected_modalities = [row["modality"] for row in expected.get("expected_data_order", {}).get("steps", [])]
    checks = {
        "step count": len(rows) == STEPS,
        "successful optimizer steps": successful == STEPS,
        "step sequence": step_numbers == list(range(1, STEPS + 1)),
        "modality sequence": not expected_modalities or modalities == expected_modalities,
        "seed": meta.get("seed") == expected["seed"],
        "initialization policy": meta.get("initialization_policy") == INITIALIZATION_POLICY,
        "aux coefficient": meta.get("aux_coef") == expected["aux_coef"],
        "capacity factor": meta.get("capacity_factor") == expected["capacity_factor"],
        "checkpoint args": filtered_args(checkpoint_args) == expected["matched_args"],
        "text evaluator checkpoint steps": metrics.get("text_eval_provenance", {}).get("source_training_steps") == STEPS,
    }
    errors.extend(f"{expected['experiment_id']}: failed {name}" for name, passed in checks.items() if not passed)
    return contract, errors


def run_protocol(source_root: Path, output_root: Path, seed: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    reserved = [output_root / name for name, _, _ in ARM_SPECS] + [output_root / PRE_PROTOCOL, output_root / POST_PROTOCOL]
    existing = [str(path) for path in reserved if path.exists()]
    if existing:
        raise FileExistsError(f"Matched protocol requires fresh outputs; already exist: {existing}")

    pre, args, splits, gamma = build_pre_protocol(source_root, output_root, seed)
    # Embed order metadata in each contract so post-run modality checks are self-contained.
    for arm in pre["arms"]:
        arm["expected_data_order"] = pre["expected_data_order"]
    missing.save_json(output_root / PRE_PROTOCOL, pre)
    pre_hash = canonical_sha256(pre)
    if pre["source_manifest_checkpoint_arg_mismatches"]:
        raise ValueError(
            "Source manifest/checkpoint args differ: "
            + ", ".join(pre["source_manifest_checkpoint_arg_mismatches"])
        )
    initial_errors = verify_matched_arm_contracts(pre["arms"])
    if initial_errors:
        raise ValueError("; ".join(initial_errors))

    _, text_train, text_eval, image_train, image_eval, audio_train, audio_eval = splits
    real = missing.get_real_module()
    metrics_by_name: Dict[str, Dict[str, Any]] = {}
    observed: List[Dict[str, Any]] = []
    errors: List[str] = []
    try:
        for arm in pre["arms"]:
            name = str(arm["experiment_id"])
            metrics_by_name[name] = real.train_real_multimodal(
                name,
                args,
                text_train,
                text_eval,
                image_train,
                image_eval,
                audio_train,
                audio_eval,
                gamma=gamma,
                aux_coef=float(arm["aux_coef"]),
                capacity_factor=float(arm["capacity_factor"]),
                max_steps=STEPS,
                out_dir=output_root,
                train_router_gates=bool(args.train_router_gates),
                train_experts=bool(args.train_experts),
            )
        for arm in pre["arms"]:
            metrics = metrics_by_name[str(arm["experiment_id"])]
            checkpoint_args = load_checkpoint_args(Path(str(metrics["checkpoint_path"])))
            contract, arm_errors = observed_contract(arm, metrics, checkpoint_args)
            observed.append(contract)
            errors.extend(arm_errors)
        errors.extend(verify_matched_arm_contracts(observed))
    except Exception as exc:
        errors.append(f"training_or_verification_exception: {type(exc).__name__}: {exc}")
        post = {
            "schema_version": 1,
            "created_at": utc_now(),
            "pre_protocol_sha256": pre_hash,
            "passed": False,
            "errors": errors,
            "observed_arms": observed,
            "completed_arms": sorted(metrics_by_name),
        }
        missing.save_json(output_root / POST_PROTOCOL, post)
        raise

    post = {
        "schema_version": 1,
        "created_at": utc_now(),
        "pre_protocol_sha256": pre_hash,
        "passed": not errors,
        "errors": errors,
        "observed_arms": observed,
        "completed_arms": sorted(metrics_by_name),
    }
    missing.save_json(output_root / POST_PROTOCOL, post)
    if errors:
        raise RuntimeError("Matched protocol verification failed: " + "; ".join(errors))
    print(json.dumps({"post_protocol": str(output_root / POST_PROTOCOL), "passed": True}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Selected root containing manifest, calibration gamma, and E3 checkpoint.")
    parser.add_argument("--output-root", required=True, help="Fresh seed-specific output root.")
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_protocol(Path(args.source_root), Path(args.output_root), args.seed)


if __name__ == "__main__":
    main()
