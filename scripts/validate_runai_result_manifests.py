"""Bind canonical evaluation manifests to captured immutable Run:AI logs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    from scripts import build_evaluation_result_manifest as manifest_lib
except ImportError:  # Direct execution from the scripts directory.
    import build_evaluation_result_manifest as manifest_lib  # type: ignore[no-redef]


LOG_DIGEST_RE = re.compile(
    rf"^{re.escape(manifest_lib.LOG_MARKER_PREFIX)}([0-9a-f]{{64}})$",
    re.MULTILINE,
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise manifest_lib.ResultManifestError(f"{label} must be an object")
    return value


def _current_file_record(raw: Mapping[str, Any], label: str) -> Dict[str, Any]:
    path_value = raw.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise manifest_lib.ResultManifestError(f"{label}.path must be a non-empty string")
    path = Path(path_value)
    try:
        current = manifest_lib.file_record(path)
    except manifest_lib.ResultManifestError as exc:
        raise manifest_lib.ResultManifestError(f"{label}: {exc}") from exc
    if current != dict(raw):
        raise manifest_lib.ResultManifestError(f"{label} hash/path/size drift detected")
    return current


def validate_and_bind(
    raw_manifest_paths: Any,
    *,
    anchor: Path,
    final_roles: Mapping[str, str],
    evaluation_roles: frozenset[str],
    role_artifacts: Mapping[str, Mapping[str, Any]],
    jobs_by_name: Mapping[str, Mapping[str, Any]],
    protocol_path: Path,
    protocol_file_sha256: str,
    protocol_content_sha256: str,
    checkpoint_record: Mapping[str, Any],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Validate job manifests and return job/role digest bindings."""
    paths = _mapping(raw_manifest_paths, "bindings manifest.job_result_manifests")
    expected_jobs = {final_roles[role] for role in evaluation_roles}
    if set(paths) != expected_jobs:
        raise manifest_lib.ResultManifestError(
            "bindings manifest.job_result_manifests must exactly cover evaluation jobs: "
            f"missing={sorted(expected_jobs - set(paths))}, "
            f"extra={sorted(set(paths) - expected_jobs)}"
        )

    job_records: Dict[str, Dict[str, Any]] = {}
    role_bindings: Dict[str, Dict[str, Any]] = {}
    for job_name in sorted(expected_jobs):
        raw_path = paths[job_name]
        manifest_path = manifest_lib._resolve_file(
            raw_path,
            f"bindings manifest.job_result_manifests.{job_name}",
            anchor,
        )
        payload = manifest_lib.read_canonical_manifest(manifest_path)
        manifest_record = manifest_lib.file_record(manifest_path)
        digest = manifest_record["sha256"]

        runai = _mapping(payload.get("runai"), f"evaluation result manifest {job_name}.runai")
        if runai.get("project") != jobs_by_name[job_name].get("project"):
            raise manifest_lib.ResultManifestError(
                f"evaluation result manifest {job_name} project identity mismatch"
            )
        if runai.get("job") != job_name:
            raise manifest_lib.ResultManifestError(
                f"evaluation result manifest {job_name} job identity mismatch"
            )
        for key in manifest_lib.OPTIONAL_RUNAI_IDENTITY_KEYS:
            captured = jobs_by_name[job_name].get(key)
            if captured is not None and runai.get(key) != captured:
                raise manifest_lib.ResultManifestError(
                    f"evaluation result manifest {job_name} {key} identity mismatch"
                )

        if _current_file_record(
            _mapping(payload.get("checkpoint"), f"evaluation result manifest {job_name}.checkpoint"),
            f"evaluation result manifest {job_name}.checkpoint",
        ) != dict(checkpoint_record):
            raise manifest_lib.ResultManifestError(
                f"evaluation result manifest {job_name} checkpoint identity mismatch"
            )
        protocol = _mapping(payload.get("protocol"), f"evaluation result manifest {job_name}.protocol")
        expected_protocol = {
            "path": str(protocol_path),
            "file_sha256": protocol_file_sha256,
            "content_sha256": protocol_content_sha256,
        }
        if dict(protocol) != expected_protocol:
            raise manifest_lib.ResultManifestError(
                f"evaluation result manifest {job_name} protocol identity mismatch"
            )

        expected_roles = {
            role for role in evaluation_roles if final_roles[role] == job_name
        }
        evaluations = _mapping(
            payload.get("evaluations"),
            f"evaluation result manifest {job_name}.evaluations",
        )
        if set(evaluations) != expected_roles:
            raise manifest_lib.ResultManifestError(
                f"evaluation result manifest {job_name} role set mismatch: "
                f"missing={sorted(expected_roles - set(evaluations))}, "
                f"extra={sorted(set(evaluations) - expected_roles)}"
            )

        logs_path = Path(str(jobs_by_name[job_name].get("logs_path", "")))
        try:
            logs_text = logs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise manifest_lib.ResultManifestError(
                f"cannot read captured immutable log for {job_name}: {exc}"
            ) from exc
        captured_digests = LOG_DIGEST_RE.findall(logs_text)
        if captured_digests != [digest]:
            raise manifest_lib.ResultManifestError(
                f"captured immutable log for {job_name} must contain exactly one "
                "matching canonical result manifest digest"
            )

        for role in sorted(expected_roles):
            evaluation = _mapping(
                evaluations[role],
                f"evaluation result manifest {job_name}.evaluations.{role}",
            )
            manifest_artifacts = _mapping(
                evaluation.get("artifacts"),
                f"evaluation result manifest {job_name}.evaluations.{role}.artifacts",
            )
            for artifact_name, raw_record in manifest_artifacts.items():
                _current_file_record(
                    _mapping(raw_record, f"evaluation result manifest {job_name}/{role}/{artifact_name}"),
                    f"evaluation result manifest {job_name}/{role}/{artifact_name}",
                )
            bound_artifacts = _mapping(
                role_artifacts[role].get("artifacts"),
                f"bindings manifest.role_artifacts.{role}.artifacts",
            )
            if dict(manifest_artifacts) != dict(bound_artifacts):
                raise manifest_lib.ResultManifestError(
                    f"evaluation result manifest {job_name} role artifact mismatch for {role}"
                )
            role_bindings[role] = {
                "result_manifest_sha256": digest,
                "evaluation_identity": dict(
                    _mapping(
                        evaluation.get("evaluation_identity"),
                        f"evaluation result manifest {job_name}.evaluations.{role}.evaluation_identity",
                    )
                ),
            }

        job_records[job_name] = manifest_record

    return job_records, role_bindings
