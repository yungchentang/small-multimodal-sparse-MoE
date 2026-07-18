#!/usr/bin/env python3
"""Build and validate the schema-v3 final ACDL Project 18 evidence bundle.

The bundle is an immutable index, not a metric summary.  Every required final
artifact is bound to one selected root and checkpoint by its resolved path,
SHA-256, byte count, and explicit root/checkpoint identity fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


BUNDLE_SCHEMA_VERSION = 3
BUNDLE_TYPE = "acdl_project18_final_evidence"
SHA256_LENGTH = 64

REQUIRED_ARTIFACT_ROLES = (
    "candidate_selection",
    "e3_metrics",
    "e3_training_jsonl",
    "e3_checkpoint_provenance",
    "e3_text_eval",
    "dataset_ledger",
    "e4_metrics",
    "e4_training_jsonl",
    "e4_checkpoint",
    "e4_text_eval",
    "e5_metrics",
    "e5_training_jsonl",
    "e5_checkpoint",
    "e5_text_eval",
    "e6_feasibility",
    "e6_checkpoint",
    "matched_ablation_summary",
    "frozen_protocol",
    "sealed_matrix",
    "routing_specialization",
    "representation_funnel",
    "runai_success_ledger",
    "failure_ledger",
)

# These files are outputs of the selected experiment root.  Analysis indices,
# frozen protocol files, and provenance reports may live outside that root but
# still carry the same root/checkpoint bindings in their bundle records.
SELECTED_ROOT_ARTIFACT_ROLES = frozenset(
    {
        "e3_metrics",
        "e3_training_jsonl",
        "e3_text_eval",
    }
)

CLI_ROLE_FLAGS = {
    "candidate_selection": "candidate-selection",
    "e3_metrics": "e3-metrics",
    "e3_training_jsonl": "e3-training-jsonl",
    "e3_checkpoint_provenance": "e3-checkpoint-provenance",
    "e3_text_eval": "e3-text-eval",
    "dataset_ledger": "dataset-ledger",
    "e4_metrics": "e4-metrics",
    "e4_training_jsonl": "e4-training-jsonl",
    "e4_checkpoint": "e4-checkpoint",
    "e4_text_eval": "e4-text-eval",
    "e5_metrics": "e5-metrics",
    "e5_training_jsonl": "e5-training-jsonl",
    "e5_checkpoint": "e5-checkpoint",
    "e5_text_eval": "e5-text-eval",
    "e6_feasibility": "e6-feasibility",
    "e6_checkpoint": "e6-checkpoint",
    "matched_ablation_summary": "matched-ablation-summary",
    "frozen_protocol": "frozen-protocol",
    "sealed_matrix": "sealed-matrix",
    "routing_specialization": "routing-specialization",
    "representation_funnel": "representation-funnel",
    "runai_success_ledger": "runai-success-ledger",
    "failure_ledger": "failure-ledger",
}


class EvidenceBundleError(ValueError):
    """Raised when final evidence is incomplete, stale, or cross-root."""


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceBundleError(f"bundle value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceBundleError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _resolved_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise EvidenceBundleError(f"{label} cannot be a symlink: {path}")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise EvidenceBundleError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise EvidenceBundleError(f"{label} must be a regular file: {resolved}")
    return resolved


def _resolved_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise EvidenceBundleError(f"{label} cannot be a symlink: {path}")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise EvidenceBundleError(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise EvidenceBundleError(f"{label} must be a directory: {resolved}")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def file_record(path: Path) -> Dict[str, Any]:
    resolved = _resolved_file(path, "evidence artifact")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def directory_record(path: Path) -> Dict[str, Any]:
    root = _resolved_directory(path, "selected root")
    files = []
    for child in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_symlink():
            raise EvidenceBundleError(f"selected root contains a symlink: {child}")
        if child.is_file():
            files.append(
                {
                    "relative_path": child.relative_to(root).as_posix(),
                    "sha256": sha256_file(child),
                    "bytes": child.stat().st_size,
                }
            )
    if not files:
        raise EvidenceBundleError("selected root contains no files")
    return {
        "path": str(root),
        "type": "directory",
        "sha256": canonical_sha256(files),
        "file_count": len(files),
        "files": files,
    }


def _read_bundle(path: Path) -> Dict[str, Any]:
    resolved = _resolved_file(path, "evidence bundle")

    def reject_constant(token: str) -> None:
        raise EvidenceBundleError(f"evidence bundle contains non-finite token {token!r}")

    try:
        payload = json.loads(
            resolved.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except EvidenceBundleError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceBundleError(f"cannot read evidence bundle {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceBundleError("evidence bundle root must be an object")
    return payload


def _verify_file_record(record: Any, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise EvidenceBundleError(f"{label} record must be an object")
    path = _resolved_file(Path(str(record.get("path", ""))), label)
    expected_hash = str(record.get("sha256", "")).lower()
    if len(expected_hash) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in expected_hash
    ):
        raise EvidenceBundleError(f"{label} has an invalid SHA-256")
    if sha256_file(path) != expected_hash:
        raise EvidenceBundleError(f"{label} SHA-256 mismatch")
    if record.get("bytes") != path.stat().st_size:
        raise EvidenceBundleError(f"{label} byte count mismatch")
    return path


def build_bundle(
    selected_root: Path,
    selected_checkpoint: Path,
    artifacts: Mapping[str, Path],
) -> Dict[str, Any]:
    root = _resolved_directory(selected_root, "selected root")
    checkpoint = _resolved_file(selected_checkpoint, "selected checkpoint")
    if not _is_within(checkpoint, root):
        raise EvidenceBundleError("selected checkpoint is outside selected root")
    if set(artifacts) != set(REQUIRED_ARTIFACT_ROLES):
        missing = sorted(set(REQUIRED_ARTIFACT_ROLES) - set(artifacts))
        extra = sorted(set(artifacts) - set(REQUIRED_ARTIFACT_ROLES))
        raise EvidenceBundleError(
            f"artifact role set is incomplete: missing={missing}, extra={extra}"
        )

    root_fingerprint = directory_record(root)
    checkpoint_record = file_record(checkpoint)
    records: Dict[str, Any] = {}
    for role in REQUIRED_ARTIFACT_ROLES:
        record = file_record(artifacts[role])
        artifact_path = Path(record["path"])
        expected_scope = (
            "selected_root" if role in SELECTED_ROOT_ARTIFACT_ROLES else "external"
        )
        if expected_scope == "selected_root" and not _is_within(artifact_path, root):
            raise EvidenceBundleError(f"{role} is outside selected root")
        if expected_scope == "external" and _is_within(artifact_path, root):
            raise EvidenceBundleError(f"{role} must be external to selected root")
        records[role] = {
            **record,
            "root_scope": expected_scope,
            "selected_root_sha256": root_fingerprint["sha256"],
            "selected_checkpoint_sha256": checkpoint_record["sha256"],
        }

    bundle: Dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_type": BUNDLE_TYPE,
        "selected_root": root_fingerprint,
        "selected_checkpoint": checkpoint_record,
        "artifacts": records,
    }
    bundle["bundle_content_sha256"] = canonical_sha256(bundle)
    return bundle


def validate_bundle(
    path: Path,
    *,
    expected_selected_root: Path | None = None,
    expected_selected_checkpoint: Path | None = None,
    expected_artifacts: Mapping[str, Path] | None = None,
) -> Dict[str, Any]:
    payload = _read_bundle(path)
    stored_content_hash = payload.get("bundle_content_sha256")
    content = dict(payload)
    content.pop("bundle_content_sha256", None)
    if stored_content_hash != canonical_sha256(content):
        raise EvidenceBundleError("evidence bundle content hash mismatch")
    if (
        payload.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or payload.get("bundle_type") != BUNDLE_TYPE
    ):
        raise EvidenceBundleError("unsupported final evidence bundle schema")

    selected_root_record = payload.get("selected_root")
    if not isinstance(selected_root_record, Mapping):
        raise EvidenceBundleError("selected_root record is missing")
    root = _resolved_directory(
        Path(str(selected_root_record.get("path", ""))), "bundle selected root"
    )
    actual_root_record = directory_record(root)
    if dict(selected_root_record) != actual_root_record:
        raise EvidenceBundleError("selected root fingerprint/hash drift detected")
    if expected_selected_root is not None and root != _resolved_directory(
        expected_selected_root, "expected selected root"
    ):
        raise EvidenceBundleError("bundle selected root disagrees with final-build input")

    checkpoint_record = payload.get("selected_checkpoint")
    checkpoint = _verify_file_record(checkpoint_record, "bundle selected checkpoint")
    if not _is_within(checkpoint, root):
        raise EvidenceBundleError("bundle selected checkpoint is outside selected root")
    if expected_selected_checkpoint is not None and checkpoint != _resolved_file(
        expected_selected_checkpoint, "expected selected checkpoint"
    ):
        raise EvidenceBundleError(
            "bundle selected checkpoint disagrees with final-build provenance"
        )

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        REQUIRED_ARTIFACT_ROLES
    ):
        missing = sorted(
            set(REQUIRED_ARTIFACT_ROLES)
            - (set(artifacts) if isinstance(artifacts, Mapping) else set())
        )
        extra = sorted(
            (set(artifacts) if isinstance(artifacts, Mapping) else set())
            - set(REQUIRED_ARTIFACT_ROLES)
        )
        raise EvidenceBundleError(
            f"bundle artifact role set is incomplete: missing={missing}, extra={extra}"
        )
    if expected_artifacts is not None and set(expected_artifacts) - set(artifacts):
        raise EvidenceBundleError("expected artifact role is absent from evidence bundle")

    resolved_artifacts: Dict[str, Path] = {}
    root_sha = str(selected_root_record["sha256"])
    checkpoint_sha = str(checkpoint_record["sha256"])
    for role in REQUIRED_ARTIFACT_ROLES:
        record = artifacts[role]
        artifact_path = _verify_file_record(record, f"bundle artifact {role}")
        expected_scope = (
            "selected_root" if role in SELECTED_ROOT_ARTIFACT_ROLES else "external"
        )
        if record.get("root_scope") != expected_scope:
            raise EvidenceBundleError(f"bundle artifact {role} has incorrect root_scope")
        if record.get("selected_root_sha256") != root_sha:
            raise EvidenceBundleError(f"bundle artifact {role} has a cross-root binding")
        if record.get("selected_checkpoint_sha256") != checkpoint_sha:
            raise EvidenceBundleError(
                f"bundle artifact {role} has a cross-checkpoint binding"
            )
        if expected_scope == "selected_root" and not _is_within(artifact_path, root):
            raise EvidenceBundleError(f"bundle artifact {role} escaped selected root")
        if expected_scope == "external" and _is_within(artifact_path, root):
            raise EvidenceBundleError(
                f"bundle artifact {role} is not external to selected root"
            )
        if expected_artifacts is not None and role in expected_artifacts:
            expected = _resolved_file(expected_artifacts[role], f"expected {role}")
            if artifact_path != expected:
                raise EvidenceBundleError(
                    f"bundle artifact {role} disagrees with final-build input"
                )
        resolved_artifacts[role] = artifact_path

    payload["_path"] = _resolved_file(path, "evidence bundle")
    payload["_sha256"] = sha256_file(payload["_path"])
    payload["_selected_root"] = root
    payload["_selected_checkpoint"] = checkpoint
    payload["_artifact_paths"] = resolved_artifacts
    return payload


def write_bundle(path: Path, bundle: Mapping[str, Any]) -> Path:
    destination = path.expanduser().resolve(strict=False)
    if destination.exists():
        raise EvidenceBundleError(f"refusing to overwrite evidence bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(bundle, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    for role, flag in CLI_ROLE_FLAGS.items():
        parser.add_argument(f"--{flag}", dest=role, type=Path, required=True)
    parser.add_argument(
        "--output-evidence-bundle",
        "--output",
        dest="output_evidence_bundle",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifacts = {role: getattr(args, role) for role in REQUIRED_ARTIFACT_ROLES}
    bundle = build_bundle(args.selected_root, args.selected_checkpoint, artifacts)
    output = write_bundle(args.output_evidence_bundle, bundle)
    validate_bundle(
        output,
        expected_selected_root=args.selected_root,
        expected_selected_checkpoint=args.selected_checkpoint,
        expected_artifacts=artifacts,
    )
    print(
        json.dumps(
            {
                "evidence_bundle": str(output),
                "sha256": sha256_file(output),
                "schema_version": BUNDLE_SCHEMA_VERSION,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
