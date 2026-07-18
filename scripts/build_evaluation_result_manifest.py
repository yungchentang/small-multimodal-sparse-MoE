#!/usr/bin/env python3
"""Build a canonical manifest for one completed Run:AI evaluation job.

The input spec names concrete files.  The output replaces those paths with
immutable file records and binds every evaluation to the exact checkpoint and
frozen protocol used by the job.  A successful job must print the emitted log
marker before termination so the captured Run:AI log commits to this manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


SPEC_SCHEMA_VERSION = 1
SPEC_TYPE = "runai_evaluation_result_manifest_spec"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_TYPE = "runai_evaluation_result_manifest"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUNAI_IDENTITY_KEYS = frozenset({"project", "job", "job_uid", "pod", "pod_uid"})
OPTIONAL_RUNAI_IDENTITY_KEYS = frozenset({"job_uid", "pod", "pod_uid"})
LOG_MARKER_PREFIX = "RUNAI_RESULT_MANIFEST_SHA256="


class ResultManifestError(ValueError):
    """Raised when a result manifest cannot be produced or validated."""


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
        raise ResultManifestError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ResultManifestError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResultManifestError(f"JSON object has duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ResultManifestError(f"{label} must be a regular non-symlink file: {path}")

    def reject_constant(token: str) -> None:
        raise ResultManifestError(
            f"{label} contains non-standard numeric token {token!r}"
        )

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except ResultManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultManifestError(f"cannot read {label} from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResultManifestError(f"{label} root must be a JSON object")
    canonical_json(payload)
    return payload


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultManifestError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResultManifestError(f"{label} must be a non-empty string")
    return value.strip()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ResultManifestError(
            f"{label} has unexpected key set: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _resolve_file(raw_path: Any, label: str, anchor: Path) -> Path:
    value = (
        raw_path.expanduser()
        if isinstance(raw_path, Path)
        else Path(_text(raw_path, label)).expanduser()
    )
    candidate = value if value.is_absolute() else anchor / value
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ResultManifestError(f"{label} does not exist: {candidate}") from exc
    if candidate.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise ResultManifestError(f"{label} must be a regular non-symlink file: {candidate}")
    return resolved


def file_record(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ResultManifestError(f"artifact must be a regular non-symlink file: {path}")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def result_manifest_log_marker(digest: str) -> str:
    if SHA256_RE.fullmatch(digest) is None:
        raise ResultManifestError("result manifest digest must be an exact lowercase SHA256")
    return f"{LOG_MARKER_PREFIX}{digest}"


def read_canonical_manifest(path: Path) -> Dict[str, Any]:
    """Read a manifest and reject any non-canonical byte representation."""
    payload = _read_json(path, "evaluation result manifest")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ResultManifestError(f"cannot read evaluation result manifest: {exc}") from exc
    expected = (canonical_json(payload) + "\n").encode("utf-8")
    if raw != expected:
        raise ResultManifestError(
            "evaluation result manifest must use exact canonical JSON bytes"
        )
    validate_manifest_schema(payload)
    return payload


def validate_manifest_schema(payload: Mapping[str, Any]) -> None:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "manifest_type",
            "commit",
            "runai",
            "checkpoint",
            "protocol",
            "evaluations",
        },
        "evaluation result manifest",
    )
    if (
        payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or payload.get("manifest_type") != MANIFEST_TYPE
    ):
        raise ResultManifestError("unsupported evaluation result manifest schema")
    if COMMIT_RE.fullmatch(str(payload.get("commit", ""))) is None:
        raise ResultManifestError("evaluation result manifest commit must be a full lowercase Git SHA")

    runai = _mapping(payload.get("runai"), "evaluation result manifest.runai")
    if not {"project", "job"}.issubset(runai) or not set(runai).issubset(RUNAI_IDENTITY_KEYS):
        raise ResultManifestError("evaluation result manifest.runai has invalid identity keys")
    for key in runai:
        _text(runai[key], f"evaluation result manifest.runai.{key}")

    checkpoint = _mapping(payload.get("checkpoint"), "evaluation result manifest.checkpoint")
    _require_exact_keys(checkpoint, {"path", "sha256", "bytes"}, "evaluation result manifest.checkpoint")
    protocol = _mapping(payload.get("protocol"), "evaluation result manifest.protocol")
    _require_exact_keys(
        protocol,
        {"path", "file_sha256", "content_sha256"},
        "evaluation result manifest.protocol",
    )
    for label, digest in (
        ("checkpoint.sha256", checkpoint.get("sha256")),
        ("protocol.file_sha256", protocol.get("file_sha256")),
        ("protocol.content_sha256", protocol.get("content_sha256")),
    ):
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ResultManifestError(f"evaluation result manifest.{label} is invalid")

    evaluations = _mapping(payload.get("evaluations"), "evaluation result manifest.evaluations")
    if not evaluations:
        raise ResultManifestError("evaluation result manifest.evaluations must be non-empty")
    for role, raw_evaluation in evaluations.items():
        _text(role, "evaluation result manifest evaluation role")
        evaluation = _mapping(raw_evaluation, f"evaluation result manifest.evaluations.{role}")
        _require_exact_keys(
            evaluation,
            {"evaluation_identity", "artifacts", "metrics_sha256", "per_query_sha256"},
            f"evaluation result manifest.evaluations.{role}",
        )
        identity = _mapping(
            evaluation.get("evaluation_identity"),
            f"evaluation result manifest.evaluations.{role}.evaluation_identity",
        )
        _require_exact_keys(
            identity,
            {"role", "evaluation_id", "checkpoint_sha256", "protocol_content_sha256"},
            f"evaluation result manifest.evaluations.{role}.evaluation_identity",
        )
        if identity.get("role") != role:
            raise ResultManifestError(f"evaluation result manifest role identity mismatch for {role}")
        _text(identity.get("evaluation_id"), f"evaluation result manifest.evaluations.{role}.evaluation_id")
        if identity.get("checkpoint_sha256") != checkpoint.get("sha256"):
            raise ResultManifestError(f"evaluation result manifest checkpoint identity mismatch for {role}")
        if identity.get("protocol_content_sha256") != protocol.get("content_sha256"):
            raise ResultManifestError(f"evaluation result manifest protocol identity mismatch for {role}")
        artifacts = _mapping(
            evaluation.get("artifacts"),
            f"evaluation result manifest.evaluations.{role}.artifacts",
        )
        if len(artifacts) < 2:
            raise ResultManifestError(f"evaluation result manifest {role} needs metrics and per-query artifacts")
        artifact_hashes = set()
        for artifact_name, raw_record in artifacts.items():
            _text(artifact_name, f"evaluation result manifest.evaluations.{role} artifact name")
            record = _mapping(raw_record, f"evaluation result manifest.evaluations.{role}.artifacts.{artifact_name}")
            _require_exact_keys(record, {"path", "sha256", "bytes"}, f"evaluation result manifest.evaluations.{role}.artifacts.{artifact_name}")
            digest = record.get("sha256")
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                raise ResultManifestError(f"evaluation result manifest {role}/{artifact_name} SHA256 is invalid")
            artifact_hashes.add(digest)
        for field in ("metrics_sha256", "per_query_sha256"):
            digest = evaluation.get(field)
            if not isinstance(digest, str) or digest not in artifact_hashes:
                raise ResultManifestError(f"evaluation result manifest {role} {field} is not an artifact SHA256")
        if evaluation["metrics_sha256"] == evaluation["per_query_sha256"]:
            raise ResultManifestError(f"evaluation result manifest {role} metrics and per-query evidence must be distinct")


def build_manifest(spec_path: Path, output_path: Path) -> Dict[str, Any]:
    """Materialize one canonical result manifest from an explicit spec."""
    if output_path.exists() or output_path.is_symlink():
        raise ResultManifestError(f"refusing to overwrite existing manifest: {output_path}")
    spec_file = _resolve_file(spec_path, "evaluation result manifest spec", Path.cwd())
    spec = _read_json(spec_file, "evaluation result manifest spec")
    _require_exact_keys(
        spec,
        {"schema_version", "spec_type", "commit", "runai", "checkpoint", "protocol", "evaluations"},
        "evaluation result manifest spec",
    )
    if spec.get("schema_version") != SPEC_SCHEMA_VERSION or spec.get("spec_type") != SPEC_TYPE:
        raise ResultManifestError("unsupported evaluation result manifest spec schema")
    commit = str(spec.get("commit", ""))
    if COMMIT_RE.fullmatch(commit) is None:
        raise ResultManifestError("evaluation result manifest spec commit must be a full lowercase Git SHA")

    raw_runai = _mapping(spec.get("runai"), "evaluation result manifest spec.runai")
    if not {"project", "job"}.issubset(raw_runai) or not set(raw_runai).issubset(RUNAI_IDENTITY_KEYS):
        raise ResultManifestError("evaluation result manifest spec.runai has invalid identity keys")
    runai = {key: _text(raw_runai[key], f"evaluation result manifest spec.runai.{key}") for key in sorted(raw_runai)}

    checkpoint_path = _resolve_file(spec.get("checkpoint"), "evaluation result manifest spec.checkpoint", spec_file.parent)
    checkpoint = file_record(checkpoint_path)
    protocol_path = _resolve_file(spec.get("protocol"), "evaluation result manifest spec.protocol", spec_file.parent)
    protocol_payload = _read_json(protocol_path, "frozen protocol")
    protocol_content_sha256 = protocol_payload.get("protocol_content_sha256")
    unhashed_protocol = dict(protocol_payload)
    unhashed_protocol.pop("protocol_content_sha256", None)
    if (
        not isinstance(protocol_content_sha256, str)
        or SHA256_RE.fullmatch(protocol_content_sha256) is None
        or canonical_sha256(unhashed_protocol) != protocol_content_sha256
    ):
        raise ResultManifestError("frozen protocol content hash mismatch")
    raw_protocol_checkpoint = _mapping(protocol_payload.get("checkpoint"), "frozen protocol.checkpoint")
    raw_protocol_artifact = _mapping(raw_protocol_checkpoint.get("artifact"), "frozen protocol.checkpoint.artifact")
    if (
        raw_protocol_artifact.get("path") != checkpoint["path"]
        or raw_protocol_artifact.get("sha256") != checkpoint["sha256"]
    ):
        raise ResultManifestError("manifest checkpoint does not match frozen protocol checkpoint")
    protocol = {
        "path": str(protocol_path),
        "file_sha256": sha256_file(protocol_path),
        "content_sha256": protocol_content_sha256,
    }

    raw_evaluations = _mapping(spec.get("evaluations"), "evaluation result manifest spec.evaluations")
    if not raw_evaluations:
        raise ResultManifestError("evaluation result manifest spec.evaluations must be non-empty")
    evaluations: Dict[str, Any] = {}
    for role in sorted(raw_evaluations):
        _text(role, "evaluation result manifest spec evaluation role")
        raw_evaluation = _mapping(raw_evaluations[role], f"evaluation result manifest spec.evaluations.{role}")
        _require_exact_keys(
            raw_evaluation,
            {"evaluation_id", "artifacts", "metrics_artifact", "per_query_artifact"},
            f"evaluation result manifest spec.evaluations.{role}",
        )
        raw_artifacts = _mapping(raw_evaluation.get("artifacts"), f"evaluation result manifest spec.evaluations.{role}.artifacts")
        artifacts: Dict[str, Any] = {}
        for name in sorted(raw_artifacts):
            _text(name, f"evaluation result manifest spec.evaluations.{role} artifact name")
            artifact_path = _resolve_file(raw_artifacts[name], f"evaluation result manifest spec.evaluations.{role}.artifacts.{name}", spec_file.parent)
            artifacts[name] = file_record(artifact_path)
        metrics_name = _text(raw_evaluation.get("metrics_artifact"), f"evaluation result manifest spec.evaluations.{role}.metrics_artifact")
        per_query_name = _text(raw_evaluation.get("per_query_artifact"), f"evaluation result manifest spec.evaluations.{role}.per_query_artifact")
        if metrics_name == per_query_name or metrics_name not in artifacts or per_query_name not in artifacts:
            raise ResultManifestError(f"evaluation result manifest spec {role} must select distinct metrics and per-query artifacts")
        evaluations[role] = {
            "evaluation_identity": {
                "role": role,
                "evaluation_id": _text(raw_evaluation.get("evaluation_id"), f"evaluation result manifest spec.evaluations.{role}.evaluation_id"),
                "checkpoint_sha256": checkpoint["sha256"],
                "protocol_content_sha256": protocol_content_sha256,
            },
            "artifacts": artifacts,
            "metrics_sha256": artifacts[metrics_name]["sha256"],
            "per_query_sha256": artifacts[per_query_name]["sha256"],
        }

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "commit": commit,
        "runai": runai,
        "checkpoint": checkpoint,
        "protocol": protocol,
        "evaluations": evaluations,
    }
    validate_manifest_schema(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(manifest) + "\n")
    except FileExistsError as exc:
        raise ResultManifestError(f"refusing to overwrite existing manifest: {output_path}") from exc
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    build_manifest(args.spec, args.output)
    digest = sha256_file(args.output)
    summary = {
        "manifest": str(args.output.resolve()),
        "sha256": digest,
        "log_marker": result_manifest_log_marker(digest),
    }
    print(canonical_json(summary))
    print(summary["log_marker"], flush=True)


if __name__ == "__main__":
    main()
