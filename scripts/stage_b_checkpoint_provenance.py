"""Produce fail-closed provenance for fresh development-only Stage-B runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "development_stage_b_checkpoint_companion"
PRODUCER_PATH = "training/distill_olmoe_top2_real.py"
POLICY = "development_only_stage_b_top8_to_top2"
DATA_POLICY = "development_only_real_manifests"
FORBIDDEN_TERMS = ("sealed", "synthetic")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def build_checkpoint_run_identity(run_provenance: Mapping[str, Any]) -> dict[str, Any]:
    run_uuid = str(run_provenance.get("run_uuid", "")).lower()
    try:
        parsed_uuid = uuid.UUID(hex=run_uuid)
    except ValueError as exc:
        raise ValueError("Stage-B run_uuid must be a cryptographic UUID4") from exc
    if parsed_uuid.version != 4 or parsed_uuid.hex != run_uuid:
        raise ValueError("Stage-B run_uuid must be a canonical UUID4 hex value")
    dataset = run_provenance.get("dataset_split_provenance")
    if not isinstance(dataset, Mapping):
        raise ValueError("Stage-B run identity lacks dataset provenance")
    dataset_sha = canonical_json_sha256(dataset)
    if run_provenance.get("dataset_split_provenance_sha256") != dataset_sha:
        raise ValueError("Stage-B run identity dataset provenance SHA mismatch")
    identity = {
        "schema_version": 1,
        "run_uuid": run_uuid,
        "source_commit_sha": run_provenance.get("source_commit_sha"),
        "runai_job_name": run_provenance.get("runai_job_name"),
        "runai_project": run_provenance.get("runai_project"),
        "producer_code": run_provenance.get("producer_code"),
        "dataset_split_provenance_sha256": dataset_sha,
        "sealed_evidence_used": run_provenance.get("sealed_evidence_used"),
        "synthetic_evidence_used": run_provenance.get("synthetic_evidence_used"),
    }
    if (
        identity["sealed_evidence_used"] is not False
        or identity["synthetic_evidence_used"] is not False
    ):
        raise ValueError("Stage-B run identity evidence policy is invalid")
    return identity


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_forbidden_path(path: Path, label: str) -> None:
    for term in FORBIDDEN_TERMS:
        if term in str(path).lower():
            raise ValueError(f"{label} contains forbidden {term!r} path")


def regular_file(path: Path, label: str) -> Path:
    reject_forbidden_path(path, label)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.resolve()


def exact_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{label} must be an exact lowercase SHA-256")
    return digest


def _git(repo_root: Path, *args: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={repo_root}", *args],
        cwd=repo_root,
        text=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"Git provenance command failed: {' '.join(args)}")
    return completed.stdout


def verify_source_commit(repo_root: Path, value: Any) -> str:
    commit = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Stage-B source commit must be an exact 40-hex commit")
    resolved = str(_git(repo_root, "rev-parse", "--verify", f"{commit}^{{commit}}")).strip().lower()
    head = str(_git(repo_root, "rev-parse", "--verify", "HEAD")).strip().lower()
    if resolved != commit or head != commit:
        raise ValueError(
            "Stage-B source commit must resolve and equal the producer checkout HEAD"
        )
    return commit


def producer_blob_sha256(repo_root: Path, source_commit: str) -> str:
    content = _git(repo_root, "show", f"{source_commit}:{PRODUCER_PATH}", text=False)
    if not isinstance(content, bytes):
        raise TypeError("Stage-B producer Git blob must be bytes")
    digest = hashlib.sha256(content).hexdigest()
    current = regular_file(repo_root / PRODUCER_PATH, "Stage-B producer source")
    if sha256_file(current) != digest:
        raise ValueError("Stage-B producer source bytes do not match source commit")
    return digest


def file_record(path: Path, *, rows: int, split: str, label: str) -> dict[str, Any]:
    resolved = regular_file(path, label)
    if type(rows) is not int or rows <= 0:
        raise ValueError(f"{label} row count must be positive")
    if not split:
        raise ValueError(f"{label} split label is missing")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "rows": rows,
        "split": split,
    }


def build_stage_b_run_provenance(
    *,
    repo_root: Path,
    data_dir: Path,
    source_files: Mapping[str, Mapping[str, Any]],
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ if environment is None else environment
    source_commit = verify_source_commit(repo_root, env.get("SOURCE_COMMIT_SHA"))
    job = str(env.get("RUNAI_JOB_NAME", "")).strip()
    project = str(env.get("RUNAI_PROJECT", "")).strip()
    if not job or not project:
        raise ValueError("Stage-B provenance requires RUNAI_JOB_NAME and RUNAI_PROJECT")
    if data_dir.is_symlink() or not data_dir.is_dir():
        raise ValueError("Stage-B data directory must be a regular directory")
    data_root = data_dir.resolve()
    reject_forbidden_path(data_root, "Stage-B data directory")
    if not {"text_tasks", "train", "development_eval"}.issubset(source_files):
        raise ValueError("Stage-B dataset provenance is missing required split roles")
    records = {
        role: file_record(
            Path(str(raw.get("path", ""))),
            rows=raw.get("rows"),
            split=str(raw.get("split", "")),
            label=f"Stage-B dataset {role}",
        )
        for role, raw in source_files.items()
    }
    dataset = {
        "policy": DATA_POLICY,
        "data_dir": str(data_root),
        "files": records,
        "sealed_evidence_used": False,
        "synthetic_evidence_used": False,
    }
    dataset_sha = canonical_json_sha256(dataset)
    return {
        "run_uuid": uuid.uuid4().hex,
        "source_commit_sha": source_commit,
        "runai_job_name": job,
        "runai_project": project,
        "policy": POLICY,
        "resolved_data_root": str(data_root),
        "producer_code": {
            "path": PRODUCER_PATH,
            "sha256": producer_blob_sha256(repo_root, source_commit),
        },
        "dataset_split_provenance": dataset,
        "dataset_split_provenance_sha256": dataset_sha,
        "sealed_evidence_used": False,
        "synthetic_evidence_used": False,
    }


def load_json(path: Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = regular_file(path, label)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload, resolved


def verify_dataset_provenance(dataset: Any) -> dict[str, Any]:
    if not isinstance(dataset, Mapping):
        raise ValueError("Stage-B dataset/split provenance is missing")
    if (
        dataset.get("policy") != DATA_POLICY
        or dataset.get("sealed_evidence_used") is not False
        or dataset.get("synthetic_evidence_used") is not False
    ):
        raise ValueError("Stage-B dataset/split policy is invalid")
    files = dataset.get("files")
    if not isinstance(files, Mapping) or not {
        "text_tasks", "train", "development_eval"
    }.issubset(files):
        raise ValueError("Stage-B dataset/split file ledger is incomplete")
    for role, record in files.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"Stage-B dataset role {role} is invalid")
        resolved = regular_file(
            Path(str(record.get("path", ""))), f"Stage-B dataset {role}"
        )
        expected_sha = exact_sha256(
            record.get("sha256"), f"Stage-B dataset {role} SHA-256"
        )
        if sha256_file(resolved) != expected_sha:
            raise ValueError(f"Stage-B dataset {role} SHA-256 mismatch")
        if record.get("size_bytes") != resolved.stat().st_size:
            raise ValueError(f"Stage-B dataset {role} size mismatch")
        if type(record.get("rows")) is not int or record["rows"] <= 0:
            raise ValueError(f"Stage-B dataset {role} rows are invalid")
        if not isinstance(record.get("split"), str) or not record["split"]:
            raise ValueError(f"Stage-B dataset {role} split is invalid")
    return dict(dataset)


def write_stage_b_companion(
    *,
    output_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    run_manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    reject_forbidden_path(output_path, "Stage-B companion output")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Stage-B companion: {output_path}")
    manifest, manifest_path = load_json(run_manifest_path, "Stage-B run manifest")
    metrics, metrics_path = load_json(metrics_path, "Stage-B final metrics")
    checkpoint = regular_file(checkpoint_path, "Stage-B checkpoint")
    try:
        import torch

        checkpoint_state = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ValueError("Stage-B checkpoint cannot be safely loaded") from exc
    if not isinstance(checkpoint_state, Mapping):
        raise ValueError("Stage-B checkpoint payload must be a mapping")
    run = manifest.get("run_provenance")
    if not isinstance(run, Mapping):
        raise ValueError("Stage-B run manifest lacks run_provenance")
    source_commit = verify_source_commit(repo_root, run.get("source_commit_sha"))
    for field in ("runai_job_name", "runai_project"):
        if not isinstance(run.get(field), str) or not run[field].strip():
            raise ValueError(f"Stage-B run provenance is missing {field}")
        if manifest.get(field) != run[field]:
            raise ValueError(f"Stage-B run manifest {field} mismatch")
    if manifest.get("source_commit_sha") != source_commit:
        raise ValueError("Stage-B run manifest source commit mismatch")
    if (
        run.get("policy") != POLICY
        or run.get("sealed_evidence_used") is not False
        or run.get("synthetic_evidence_used") is not False
    ):
        raise ValueError("Stage-B run provenance policy is invalid")
    dataset = verify_dataset_provenance(run.get("dataset_split_provenance"))
    if manifest.get("dataset_split_provenance") != dataset:
        raise ValueError("Stage-B run manifest dataset/split provenance mismatch")
    run_identity = build_checkpoint_run_identity(run)
    if manifest.get("run_identity") != run_identity:
        raise ValueError("Stage-B run manifest identity mismatch")
    if checkpoint_state.get("run_identity") != run_identity:
        raise ValueError("Stage-B checkpoint lacks matching internal run identity")
    internal_provenance = checkpoint_state.get("provenance")
    if not isinstance(internal_provenance, Mapping):
        raise ValueError("Stage-B checkpoint lacks internal provenance")
    for field in (
        "run_uuid",
        "source_commit_sha",
        "runai_job_name",
        "runai_project",
        "producer_code",
        "dataset_split_provenance",
        "dataset_split_provenance_sha256",
        "sealed_evidence_used",
        "synthetic_evidence_used",
    ):
        if internal_provenance.get(field) != run.get(field):
            raise ValueError(
                f"Stage-B checkpoint internal provenance mismatch for {field}"
            )
    code_sha = producer_blob_sha256(repo_root, source_commit)
    if run.get("producer_code") != {"path": PRODUCER_PATH, "sha256": code_sha}:
        raise ValueError("Stage-B run producer provenance mismatch")
    checkpoint_sha = sha256_file(checkpoint)
    recorded = metrics.get("checkpoint_provenance")
    if isinstance(recorded, Mapping):
        for field in (
            "run_uuid",
            "source_commit_sha",
            "runai_job_name",
            "runai_project",
            "producer_code",
            "dataset_split_provenance",
            "dataset_split_provenance_sha256",
        ):
            if recorded.get(field) != run.get(field):
                raise ValueError(
                    f"Stage-B metrics run provenance mismatch for {field}"
                )
    if not isinstance(recorded, Mapping) or (
        recorded.get("stage") != "B"
        or Path(str(recorded.get("saved_checkpoint", ""))).resolve() != checkpoint
        or recorded.get("saved_checkpoint_sha256") != checkpoint_sha
        or recorded.get("saved_checkpoint_size_bytes") != checkpoint.stat().st_size
    ):
        raise ValueError("Stage-B metrics do not bind the final checkpoint")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "development_only": True,
        "sealed_evidence_used": False,
        "synthetic_evidence_used": False,
        "source_commit_sha": source_commit,
        "runai_job_name": run["runai_job_name"],
        "runai_project": run["runai_project"],
        "run_identity": run_identity,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "size_bytes": int(checkpoint.stat().st_size),
        },
        "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
        "run_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "code": {"path": PRODUCER_PATH, "sha256": code_sha},
        "dataset_split_provenance": dataset,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
