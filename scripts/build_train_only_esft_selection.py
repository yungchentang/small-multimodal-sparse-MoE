#!/usr/bin/env python3
"""Build a leakage-free ESFT selection artifact from real train routing only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


FORBIDDEN_TERMS = ("sealed", "synthetic")
MODALITIES = {"image_prefix", "audio_prefix"}
TRAIN_SOURCE_KEYS = {
    "image_prefix": "image_train",
    "audio_prefix": "speech_train",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_forbidden(value: str | Path) -> None:
    lowered = str(value).lower()
    for term in FORBIDDEN_TERMS:
        if term in lowered:
            raise ValueError(f"forbidden {term!r} evidence path: {value}")


def git_provenance(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", *args],
            cwd=repo_root,
            text=True,
        ).strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "working_tree_dirty": bool(run("status", "--porcelain")),
    }


def _exact_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{label} must be an exact lowercase SHA-256")
    return digest


def _regular_file(path: Path, label: str) -> Path:
    reject_forbidden(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.resolve()


def _verify_git_commit(repo_root: Path, value: Any, label: str) -> str:
    commit = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"{label} must be an exact Git commit")
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "rev-parse",
            "--verify",
            f"{commit}^{{commit}}",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip().lower() != commit:
        raise ValueError(f"{label} cannot be resolved: {commit}")
    return commit


def _git_blob_sha256(repo_root: Path, commit: str, path: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "show",
            f"{commit}:{path}",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"collector source cannot be resolved at {commit}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _load_exact_json(
    path: Path, expected_sha256: str, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _regular_file(path, label)
    expected = _exact_sha256(expected_sha256, f"{label} SHA-256")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected} observed={actual}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload, {
        "path": str(resolved),
        "sha256": actual,
        "size_bytes": int(resolved.stat().st_size),
    }


def read_train_routing(
    path: Path, *, trusted_source: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    if not isinstance(trusted_source, Mapping):
        raise ValueError(
            "bare routing rows are not trusted; use a verified collector manifest"
        )
    reject_forbidden(path)
    strict_split_sha = _exact_sha256(
        trusted_source.get("strict_split_manifest_sha256"),
        "trusted strict split manifest SHA-256",
    )
    source_files = trusted_source.get("source_files")
    if not isinstance(source_files, Mapping):
        raise ValueError("trusted collector source files are missing")
    canonical: list[dict[str, Any]] = []
    observed_sources: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is a blank JSONL row")
            outer = json.loads(line)
            if not isinstance(outer, dict) or outer.get("real_subset") is not True:
                raise ValueError(f"{path}:{line_number} is not a real-subset routing row")
            if str(outer.get("split", "")).lower() != "train":
                raise ValueError(f"{path}:{line_number} is not split=train")
            modality = str(outer.get("modality", ""))
            expected_source_key = TRAIN_SOURCE_KEYS.get(modality)
            if expected_source_key is None:
                raise ValueError(f"{path}:{line_number} has invalid outer modality")
            if outer.get("source_manifest_key") != expected_source_key:
                raise ValueError(
                    f"{path}:{line_number} source manifest is not trusted train data"
                )
            expected_source_sha = _exact_sha256(
                source_files.get(expected_source_key),
                f"trusted {expected_source_key} SHA-256",
            )
            if (
                outer.get("source_manifest_sha256") != expected_source_sha
                or outer.get("strict_split_manifest_sha256") != strict_split_sha
            ):
                raise ValueError(
                    f"{path}:{line_number} source/split provenance mismatch"
                )
            children = outer.get("modality_layer_accounting")
            if not isinstance(children, list):
                raise ValueError(f"{path}:{line_number} has no per-layer accounting")
            for child in children:
                if not isinstance(child, dict) or child.get("modality") not in MODALITIES:
                    raise ValueError(f"{path}:{line_number} has invalid prefix modality")
                if child["modality"] != modality:
                    raise ValueError(
                        f"{path}:{line_number} outer/child modality mismatch"
                    )
                canonical.append({**child, "split": "train", "real_subset": True})
            observed_sources.add(expected_source_key)
    if not canonical:
        raise ValueError(f"{path} contains no prefix routing rows")
    if observed_sources != {"image_train", "speech_train"}:
        raise ValueError("routing source must contain image_train and speech_train")
    return canonical


def _verify_file_record(record: Any, label: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} record is missing")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path is missing")
    path = _regular_file(Path(raw_path), label)
    expected = _exact_sha256(record.get("sha256"), f"{label} SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected} observed={actual}"
        )
    return {
        "path": str(path),
        "sha256": actual,
        "size_bytes": int(path.stat().st_size),
    }


def load_verified_collector_routing(
    collector_manifest_path: Path,
    expected_collector_manifest_sha256: str,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reject_forbidden(collector_manifest_path)
    manifest, manifest_record = _load_exact_json(
        collector_manifest_path,
        expected_collector_manifest_sha256,
        "collector manifest",
    )
    required = {
        "artifact_type": "development_real_prefix_routing_collection",
        "development_only": True,
        "real_subset": True,
        "sealed": False,
        "synthetic": False,
        "collection_split": "train",
        "dev_files_read": False,
        "eval_files_read": False,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"collector manifest mismatch for {field}: "
                f"expected={expected!r} observed={manifest.get(field)!r}"
            )
    if manifest.get("collection_splits") != ["train"]:
        raise ValueError("collector manifest must contain only the train split")

    strict = manifest.get("strict_split_manifest")
    if not isinstance(strict, Mapping):
        raise ValueError("collector manifest lacks strict split provenance")
    strict_manifest, strict_record = _load_exact_json(
        Path(str(strict.get("path", ""))),
        str(strict.get("sha256", "")),
        "strict split manifest",
    )
    files = strict_manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("strict split manifest files are missing")
    schema_version = strict_manifest.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version <= 0
        or strict.get("schema_version") != schema_version
    ):
        raise ValueError("collector/strict split schema provenance mismatch")
    requested = ["image_train", "speech_train"]
    if strict.get("collection_split") != "train" or strict.get(
        "requested_files"
    ) != requested:
        raise ValueError("collector did not verify exactly the train source files")
    unread = strict.get("unread_files")
    expected_unread = sorted(set(files) - set(requested))
    if unread != expected_unread:
        raise ValueError("collector unread split-file ledger is incomplete")
    for key in ("image_dev", "image_eval", "speech_dev", "speech_eval"):
        if key in files and key not in unread:
            raise ValueError(f"collector does not prove {key} was unread")
    source_file_shas = {
        key: _exact_sha256(
            files.get(key, {}).get("sha256")
            if isinstance(files.get(key), Mapping)
            else None,
            f"strict split files.{key}.sha256",
        )
        for key in requested
    }
    source_manifests = manifest.get("source_manifests")
    if not isinstance(source_manifests, Mapping):
        raise ValueError("collector source_manifests are missing")
    for key in requested:
        file_record = files.get(key)
        if not isinstance(file_record, Mapping):
            raise ValueError(f"strict split files.{key} is missing")
        reject_forbidden(str(file_record.get("path", "")))
        source_record = source_manifests.get(key)
        if not isinstance(source_record, Mapping):
            raise ValueError(f"collector source_manifests.{key} is missing")
        if source_record.get("sha256") != source_file_shas[key]:
            raise ValueError(f"collector source_manifests.{key} SHA mismatch")
        strict_source_record = source_record.get("strict_split_record")
        if (
            not isinstance(strict_source_record, Mapping)
            or strict_source_record.get("path") != file_record.get("path")
            or strict_source_record.get("sha256") != source_file_shas[key]
        ):
            raise ValueError(
                f"collector source_manifests.{key} strict record mismatch"
            )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"train"}:
        raise ValueError("collector outputs must contain exactly train")
    train_record = _verify_file_record(outputs["train"], "collector train routing")

    code = manifest.get("code")
    if not isinstance(code, Mapping):
        raise ValueError("collector code provenance is missing")
    collector_commit = _verify_git_commit(
        repo_root, code.get("source_commit_sha"), "collector source commit"
    )
    collector_path = _regular_file(
        Path(str(code.get("collector_path", ""))), "collector source"
    )
    collector_sha = _exact_sha256(
        code.get("collector_sha256"), "collector source SHA-256"
    )
    if sha256_file(collector_path) != collector_sha:
        raise ValueError("collector source file SHA-256 mismatch")
    if (
        _git_blob_sha256(
            repo_root,
            collector_commit,
            "scripts/collect_development_prefix_routing.py",
        )
        != collector_sha
    ):
        raise ValueError("collector source bytes do not match source commit")

    checkpoint = manifest.get("checkpoint")
    stage_b = manifest.get("stage_b_checkpoint")
    if not isinstance(checkpoint, Mapping) or not isinstance(stage_b, Mapping):
        raise ValueError("collector checkpoint provenance is incomplete")
    checkpoint_record = _verify_file_record(checkpoint, "E3 checkpoint")
    stage_b_record = _verify_file_record(stage_b, "Stage-B checkpoint")
    run_provenance = checkpoint.get("run_provenance")
    checkpoint_companion = checkpoint.get("companion_manifest")
    stage_b_companion = stage_b.get("companion_manifest")
    if (
        not isinstance(run_provenance, Mapping)
        or not isinstance(checkpoint_companion, Mapping)
        or not isinstance(stage_b_companion, Mapping)
    ):
        raise ValueError("collector checkpoint companion provenance is missing")
    e3_commit = _verify_git_commit(
        repo_root, run_provenance.get("source_commit_sha"), "E3 source commit"
    )
    if checkpoint_companion.get("source_commit_sha") != e3_commit:
        raise ValueError("E3 companion source commit mismatch")
    for field in ("runai_job_name", "runai_project"):
        value = run_provenance.get(field)
        if (
            not isinstance(value, str)
            or not value.strip()
            or checkpoint_companion.get(field) != value
        ):
            raise ValueError(f"E3 {field} provenance mismatch")
    if (
        checkpoint_companion.get("development_split_manifest_sha256")
        != strict_record["sha256"]
    ):
        raise ValueError("E3 companion strict split SHA mismatch")
    stage_b_commit = _verify_git_commit(
        repo_root,
        stage_b_companion.get("source_commit_sha"),
        "Stage-B source commit",
    )
    stage_b_run = {}
    for field in ("runai_job_name", "runai_project"):
        value = stage_b_companion.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Stage-B companion is missing {field}")
        stage_b_run[field] = value
    checkpoint_companion_record = _verify_file_record(
        checkpoint_companion, "E3 companion manifest"
    )
    stage_b_companion_record = _verify_file_record(
        stage_b_companion, "Stage-B companion manifest"
    )
    restoration = manifest.get("model_state_restoration")
    if (
        not isinstance(restoration, Mapping)
        or restoration.get("restoration_order")
        != ["stage_b_student", "e3_adapter"]
    ):
        raise ValueError("collector model restoration order is invalid")

    trusted_source = {
        "strict_split_manifest_sha256": strict_record["sha256"],
        "source_files": source_file_shas,
    }
    rows = read_train_routing(
        Path(train_record["path"]), trusted_source=trusted_source
    )
    provenance = {
        "collector_manifest": manifest_record,
        "collector_source_commit_sha": collector_commit,
        "strict_split_manifest": strict_record,
        "strict_split_schema_version": strict_manifest.get("schema_version"),
        "train_routing": train_record,
        "source_manifest_sha256_by_key": source_file_shas,
        "e3_checkpoint": checkpoint_record,
        "e3_source_commit_sha": e3_commit,
        "runai_job_name": run_provenance["runai_job_name"],
        "runai_project": run_provenance["runai_project"],
        "stage_b_checkpoint": stage_b_record,
        "stage_b_source_commit_sha": stage_b_commit,
        "stage_b_runai_job_name": stage_b_run["runai_job_name"],
        "stage_b_runai_project": stage_b_run["runai_project"],
        "source_paths": [
            manifest_record["path"],
            train_record["path"],
            strict_record["path"],
            checkpoint_record["path"],
            checkpoint_companion_record["path"],
            stage_b_record["path"],
            stage_b_companion_record["path"],
        ],
        "source_files": [
            manifest_record,
            train_record,
            strict_record,
            checkpoint_record,
            checkpoint_companion_record,
            stage_b_record,
            stage_b_companion_record,
        ],
        "splits": ["train"],
        "dev_files_read": False,
        "eval_files_read": False,
    }
    return rows, provenance


def build_selection(rows: list[dict[str, Any]], selected_count: int) -> dict[str, Any]:
    if selected_count <= 0:
        raise ValueError("selected experts per layer must be positive")
    layers: dict[int, dict[str, Any]] = {}
    expected_total = observed_total = prefix_tokens_total = 0
    for index, row in enumerate(rows):
        layer = int(row["layer"])
        top_k = int(row["top_k"])
        token_count = int(row["token_count"])
        counts = [int(value) for value in row["attempted_expert_counts"]]
        scores = [float(value) for value in row["gate_score_sums"]]
        if (
            len(counts) != len(scores)
            or not counts
            or any(value < 0 for value in counts)
            or any(not math.isfinite(value) or value < 0.0 for value in scores)
        ):
            raise ValueError(f"routing row {index} has invalid expert vectors")
        expected = token_count * top_k
        observed = sum(counts)
        if expected != observed:
            raise ValueError(
                f"routing row {index} violates tokens x K: {expected} != {observed}"
            )
        state = layers.setdefault(
            layer,
            {
                "counts": [0] * len(counts),
                "scores": [0.0] * len(scores),
                "prefix_tokens": 0,
                "assignments": 0,
                "modalities": set(),
            },
        )
        if len(state["counts"]) != len(counts):
            raise ValueError(f"layer {layer} changes expert count")
        state["counts"] = [a + b for a, b in zip(state["counts"], counts)]
        state["scores"] = [a + b for a, b in zip(state["scores"], scores)]
        state["prefix_tokens"] += token_count
        state["assignments"] += observed
        state["modalities"].add(str(row["modality"]))
        expected_total += expected
        observed_total += observed
        prefix_tokens_total += token_count

    methods: dict[str, dict[str, Any]] = {"ESFT-Gate": {}, "ESFT-Token": {}}
    for layer, state in sorted(layers.items()):
        if state["modalities"] != MODALITIES:
            raise ValueError(f"layer {layer} lacks image/audio train routing")
        num_experts = len(state["counts"])
        if selected_count >= num_experts:
            raise ValueError("selection must keep at least one expert frozen")
        gate_rank = sorted(
            range(num_experts), key=lambda idx: (-state["scores"][idx], idx)
        )
        token_rank = sorted(
            range(num_experts), key=lambda idx: (-state["counts"][idx], idx)
        )
        expert_scores = [
            {
                "expert_id": expert_id,
                "gate_score_sum": state["scores"][expert_id],
                "gate_score_per_prefix_token": state["scores"][expert_id]
                / max(1, state["prefix_tokens"]),
                "token_count": state["counts"][expert_id],
                "token_frequency": state["counts"][expert_id]
                / max(1, state["assignments"]),
            }
            for expert_id in range(num_experts)
        ]
        common = {
            "splits": ["train"],
            "modalities": sorted(MODALITIES),
            "prefix_tokens": state["prefix_tokens"],
            "assignments": state["assignments"],
            "expert_scores": expert_scores,
        }
        methods["ESFT-Gate"][str(layer)] = {
            **common,
            "selected_expert_ids": gate_rank[:selected_count],
        }
        methods["ESFT-Token"][str(layer)] = {
            **common,
            "selected_expert_ids": token_rank[:selected_count],
        }
    return {
        "schema_version": 2,
        "selection_scope": "development_train_image_audio_prefix_only",
        "deterministic_tie_break": "ascending_expert_id",
        "selected_experts_per_layer": selected_count,
        "methods": methods,
        "routing_accounting": {
            "denominator": "train_prefix_token_expert_assignments_across_layers",
            "prefix_tokens_across_layers": prefix_tokens_total,
            "layer_count": len(layers),
            "expected_assignments_tokens_x_layers_x_k": expected_total,
            "observed_assignments": observed_total,
            "conservation_ok": expected_total == observed_total,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-collector-manifest-sha256", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--selected-experts-per-layer", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_forbidden(args.output)
    if args.output.exists():
        raise FileExistsError(f"refusing overwrite: {args.output}")
    repo_root = Path(__file__).resolve().parents[1]
    rows, routing_provenance = load_verified_collector_routing(
        args.collector_manifest,
        args.expected_collector_manifest_sha256,
        repo_root,
    )
    selection = build_selection(rows, args.selected_experts_per_layer)
    selected = {
        layer: row["selected_expert_ids"]
        for layer, row in selection["methods"]["ESFT-Gate"].items()
    }
    report = {
        "schema_version": 2,
        "artifact_type": "development_moe_reconstruction_and_esft_selection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "sealed_evidence_used": False,
        "synthetic_evidence_used": False,
        "model": {"base_model": args.base_model},
        "reconstruction": {
            "status": "not_run",
            "reason": "train_only_selection_artifact",
        },
        "esft_selection": selection,
        "selected_expert_training_plan": {
            "mode": "selected_full_experts",
            "selection_method": "ESFT-Gate",
            "selected_expert_ids_by_layer": selected,
            "non_selected_experts_frozen": True,
            "expert_learning_rate": 1e-6,
            "weight_anchor_coefficient": 0.01,
            "weight_decay": 0.0,
        },
        "provenance": {
            "git": git_provenance(Path(__file__).resolve().parents[1]),
            "routing": {
                **routing_provenance,
                "policy": "development_only_real_train",
                "splits": ["train"],
                "row_count": len(rows),
                "sealed_evidence_used": False,
                "synthetic_evidence_used": False,
            },
            "code": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "layers": len(selected)}))


if __name__ == "__main__":
    main()
