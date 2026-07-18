#!/usr/bin/env python3
"""Build a deterministic, fail-closed schema-v2 Run:AI success ledger.

The bindings manifest is a schema-v2 JSON object with this shape::

  {
    "schema_version": 2,
    "bindings_type": "runai_final_success_ledger_bindings",
    "final_roles": {"selected_e3_training": "run-name", ...},
    "role_artifacts": {
      "selected_e3_training": {
        "artifacts": {"training_metrics": "/absolute/path/to/metrics.jsonl"},
        "checkpoint": "/absolute/path/to/model.pt"
      },
      ...
    },
    "job_result_manifests": {
      "evaluation-job": "/absolute/path/to/canonical-result-manifest.json"
    },
    "failure_chains": [
      {"failed_job": "old-run", "replacement_job": "new-run",
       "diagnosis": "...", "fix": "..."}
    ]
  }

Every final role needs a non-empty artifact mapping.  Checkpoints are optional
per role.  Every job serving an evaluation role needs one canonical result
manifest, and its captured log must contain that manifest's digest marker.
Legacy schema-v1 bindings are intentionally rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:
    from scripts import build_evaluation_result_manifest as result_manifest
    from scripts import validate_runai_result_manifests as result_binding
    from scripts.protocol_v2 import (
        ProtocolV2Error,
        frozen_checkpoint_artifact,
        validate_metrics_against_protocol_v2,
        validate_protocol_v2,
    )
except ImportError:  # Direct execution via script entry point.
    import build_evaluation_result_manifest as result_manifest  # type: ignore[no-redef]
    import validate_runai_result_manifests as result_binding  # type: ignore[no-redef]
    from protocol_v2 import (  # type: ignore[no-redef]
        ProtocolV2Error,
        validate_metrics_against_protocol_v2,
        frozen_checkpoint_artifact,
        validate_protocol_v2,
    )


SCHEMA_VERSION = 2
LEDGER_TYPE = "runai_final_success_ledger"
BINDINGS_SCHEMA_VERSION = 2
BINDINGS_TYPE = "runai_final_success_ledger_bindings"
RAW_EVIDENCE_SCHEMA_VERSION = 1
RAW_STATUS_RE = re.compile(r"^Status:\s+(Succeeded|Failed)\s*$", re.MULTILINE)
RAW_NAME_RE = re.compile(r"^Name:\s+(\S+)\s*(?=\n|\Z)", re.MULTILINE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SEALED_MATRIX_CELLS = ("r5", "r10", "h10", "f250")
SEALED_CONTROLS = (
    "real",
    "shuffled",
    "zero",
    "norm-matched-random",
    "no-prefix",
)
REQUIRED_FINAL_RUNAI_ROLES = frozenset(
    {
        "selected_e3_training",
        "e3_text_eval",
        "e4_ablation",
        "e5_ablation",
        "e6_feasibility",
        "routing_specialization",
        "representation_funnel",
        "sealed_matrix_analysis",
    }
    | {
        f"sealed:{cell_id}:{control}"
        for cell_id in SEALED_MATRIX_CELLS
        for control in SEALED_CONTROLS
    }
)
EVALUATION_RUNAI_ROLES = REQUIRED_FINAL_RUNAI_ROLES - {"selected_e3_training"}


class LedgerInputError(ValueError):
    """Raised when the supplied evidence cannot form a final success ledger."""


def canonical_json(value: Any) -> str:
    """Return the one canonical JSON representation accepted by this ledger."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LedgerInputError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LedgerInputError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LedgerInputError(f"JSON object has duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LedgerInputError(f"{label} must be a regular non-symlink file: {path}")

    def reject_constant(token: str) -> None:
        raise LedgerInputError(f"{label} contains non-standard numeric token {token!r}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except LedgerInputError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerInputError(f"cannot read {label} from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LedgerInputError(f"{label} root must be a JSON object")
    canonical_json(payload)
    return payload


def _read_jsonl(path: Path, label: str) -> List[Dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise LedgerInputError(f"{label} must be a regular non-symlink file: {path}")
    rows: List[Dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    LedgerInputError(
                        f"{label}:{line_number} contains non-standard numeric token {token!r}"
                    )
                ),
            )
            if not isinstance(row, dict):
                raise LedgerInputError(f"{label}:{line_number} must be an object")
            canonical_json(row)
            rows.append(row)
    except LedgerInputError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerInputError(f"cannot read {label} from {path}: {exc}") from exc
    if not rows:
        raise LedgerInputError(f"{label} must contain at least one row")
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LedgerInputError(f"{label} must be an object")
    return value


def _list(value: Any, label: str, *, nonempty: bool = True) -> List[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "a non-empty" if nonempty else "a"
        raise LedgerInputError(f"{label} must be {qualifier} list")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LedgerInputError(f"{label} must be a non-empty string")
    return value.strip()


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LedgerInputError(f"{label} must be an integer >= {minimum}")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise LedgerInputError(
            f"{label} has unexpected key set: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _resolve_file(raw_path: Any, label: str, anchor: Path) -> Path:
    if isinstance(raw_path, Path):
        value = raw_path.expanduser()
    else:
        value = Path(_text(raw_path, label)).expanduser()
    candidate = value if value.is_absolute() else anchor / value
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LedgerInputError(f"{label} does not exist: {candidate}") from exc
    if candidate.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise LedgerInputError(f"{label} must be a regular non-symlink file: {candidate}")
    return resolved


def _resolve_directory(raw_path: Path, label: str) -> Path:
    if raw_path.is_symlink():
        raise LedgerInputError(f"{label} cannot be a symlink: {raw_path}")
    try:
        resolved = raw_path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise LedgerInputError(f"{label} does not exist: {raw_path}") from exc
    if not resolved.is_dir():
        raise LedgerInputError(f"{label} must be a directory: {raw_path}")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def directory_fingerprint(path: Path) -> Dict[str, Any]:
    root = _resolve_directory(path, "selected root")
    records: List[Dict[str, Any]] = []
    for child in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_symlink():
            raise LedgerInputError(f"selected root contains symlink: {child}")
        if child.is_file():
            records.append(
                {
                    "relative_path": child.relative_to(root).as_posix(),
                    "sha256": sha256_file(child),
                    "bytes": child.stat().st_size,
                }
            )
    if not records:
        raise LedgerInputError(f"selected root contains no regular files: {root}")
    return {
        "path": str(root),
        "type": "directory",
        "sha256": canonical_sha256(records),
        "file_count": len(records),
        "files": records,
    }


def file_record(path: Path) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LedgerInputError(f"artifact must be a regular non-symlink file: {path}")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _parse_timestamp(value: Any, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerInputError(f"{label} must be ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LedgerInputError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _validate_protocol(protocol_path: Path, selected_root: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    protocol = _read_json(protocol_path, "frozen protocol")
    try:
        validate_protocol_v2(protocol)
    except ProtocolV2Error as exc:
        raise LedgerInputError(str(exc)) from exc
    stored_content_sha = protocol.get("protocol_content_sha256")
    without_hash = dict(protocol)
    without_hash.pop("protocol_content_sha256", None)
    if not isinstance(stored_content_sha, str) or stored_content_sha != canonical_sha256(without_hash):
        raise LedgerInputError("frozen protocol content hash mismatch")
    inputs = _mapping(protocol.get("inputs"), "frozen protocol.inputs")
    stored_root = _mapping(inputs.get("selected_root"), "frozen protocol.inputs.selected_root")
    actual_root = directory_fingerprint(selected_root)
    if dict(stored_root) != actual_root:
        raise LedgerInputError("frozen protocol selected-root fingerprint/hash drift detected")
    checkpoint = _mapping(protocol.get("checkpoint"), "frozen protocol.checkpoint")
    if _text(checkpoint.get("selected_root"), "frozen protocol.checkpoint.selected_root") != str(selected_root):
        raise LedgerInputError("frozen protocol checkpoint root disagrees with selected root")
    project = _text(protocol.get("runai_project"), "frozen protocol.runai_project")
    return protocol, {
        "selected_root": actual_root,
        "project": project,
        "checkpoint_artifact": frozen_checkpoint_artifact(protocol),
    }


def _validate_raw_job(raw: Any, label: str, anchor: Path, project: str) -> Dict[str, Any]:
    job = dict(_mapping(raw, label))
    name = _text(job.get("job"), f"{label}.job")
    if job.get("project") != project:
        raise LedgerInputError(f"{label} project does not match frozen protocol")
    status = job.get("status")
    if status not in {"Succeeded", "Failed"}:
        raise LedgerInputError(f"{label} status must be exactly Succeeded or Failed")
    result: Dict[str, Any] = dict(job)
    result["job"] = name
    for kind in ("describe", "logs"):
        if _integer(job.get(f"{kind}_returncode"), f"{label}.{kind}_returncode") != 0:
            raise LedgerInputError(f"{label} {kind} command did not return zero")
        path = _resolve_file(job.get(f"{kind}_path"), f"{label}.{kind}_path", anchor)
        expected_sha = str(job.get(f"{kind}_sha256", "")).lower()
        if not SHA256_RE.fullmatch(expected_sha) or sha256_file(path) != expected_sha:
            raise LedgerInputError(f"{label} {kind} SHA-256 mismatch/hash drift")
        result[f"{kind}_path"] = str(path)
        result[f"{kind}_sha256"] = expected_sha
    describe_text = Path(result["describe_path"]).read_text(encoding="utf-8", errors="replace")
    statuses = RAW_STATUS_RE.findall(describe_text)
    if len(statuses) != 1 or statuses[0] != status:
        raise LedgerInputError(f"{label} describe status does not exactly match evidence status")
    names = RAW_NAME_RE.findall(describe_text)
    if names != [name]:
        raise LedgerInputError(f"{label} describe job name does not exactly match evidence job")
    return result


def _merge_raw_evidence(index_paths: Sequence[Path], project: str) -> tuple[List[Dict[str, Any]], datetime]:
    if not index_paths:
        raise LedgerInputError("at least one raw evidence index is required")
    jobs_by_name: Dict[str, Dict[str, Any]] = {}
    captured_at: List[datetime] = []
    for raw_path in index_paths:
        index_path = _resolve_file(raw_path, "raw evidence index", Path.cwd())
        payload = _read_json(index_path, "raw evidence index")
        if payload.get("schema_version") != RAW_EVIDENCE_SCHEMA_VERSION:
            raise LedgerInputError("raw evidence index must use schema_version=1")
        if payload.get("project") != project:
            raise LedgerInputError("raw evidence index project does not match frozen protocol")
        captured_at.append(_parse_timestamp(payload.get("captured_at"), "raw evidence index.captured_at"))
        for index, raw_job in enumerate(_list(payload.get("jobs"), "raw evidence index.jobs")):
            job = _validate_raw_job(raw_job, f"raw evidence job[{index}]", index_path.parent, project)
            existing = jobs_by_name.get(job["job"])
            if existing is not None and canonical_json(existing) != canonical_json(job):
                raise LedgerInputError(f"conflicting duplicate raw evidence for job {job['job']}")
            jobs_by_name[job["job"]] = job
    return [jobs_by_name[name] for name in sorted(jobs_by_name)], max(captured_at)


def _validate_role_set(value: Any, label: str) -> Mapping[str, Any]:
    roles = _mapping(value, label)
    actual = set(roles)
    if actual != REQUIRED_FINAL_RUNAI_ROLES:
        raise LedgerInputError(
            f"{label} must contain the complete final role set: "
            f"missing={sorted(REQUIRED_FINAL_RUNAI_ROLES - actual)}, "
            f"extra={sorted(actual - REQUIRED_FINAL_RUNAI_ROLES)}"
        )
    return roles


def _validate_bindings(
    bindings_path: Path,
    jobs_by_name: Dict[str, Dict[str, Any]],
    protocol: Mapping[str, Any],
    protocol_path: Path,
    protocol_file_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> tuple[Dict[str, str], Dict[str, Dict[str, Any]], List[Dict[str, str]]]:
    bindings = _read_json(bindings_path, "bindings manifest")
    _require_exact_keys(
        bindings,
        {"schema_version", "bindings_type", "final_roles", "role_artifacts", "job_result_manifests", "failure_chains"},
        "bindings manifest",
    )
    if bindings.get("schema_version") != BINDINGS_SCHEMA_VERSION or bindings.get("bindings_type") != BINDINGS_TYPE:
        raise LedgerInputError("unsupported bindings manifest schema")
    raw_roles = _validate_role_set(bindings.get("final_roles"), "bindings manifest.final_roles")
    final_roles: Dict[str, str] = {}
    for role in sorted(REQUIRED_FINAL_RUNAI_ROLES):
        name = _text(raw_roles[role], f"bindings manifest.final_roles.{role}")
        job = jobs_by_name.get(name)
        if job is None:
            raise LedgerInputError(f"final role {role} references absent job {name}")
        if job.get("status") != "Succeeded":
            raise LedgerInputError(f"final role {role} must reference a Succeeded job")
        final_roles[role] = name

    raw_role_artifacts = _validate_role_set(bindings.get("role_artifacts"), "bindings manifest.role_artifacts")
    role_artifacts: Dict[str, Dict[str, Any]] = {}
    job_checkpoints: Dict[str, Dict[str, Any]] = {}
    for role in sorted(REQUIRED_FINAL_RUNAI_ROLES):
        raw_entry = _mapping(raw_role_artifacts[role], f"bindings manifest.role_artifacts.{role}")
        allowed = {"artifacts", "checkpoint"}
        if not set(raw_entry).issubset(allowed) or "artifacts" not in raw_entry:
            raise LedgerInputError(f"bindings manifest.role_artifacts.{role} has invalid keys")
        raw_artifacts = _mapping(raw_entry.get("artifacts"), f"bindings manifest.role_artifacts.{role}.artifacts")
        if not raw_artifacts:
            raise LedgerInputError(f"bindings manifest.role_artifacts.{role}.artifacts must be non-empty")
        artifacts: Dict[str, Dict[str, Any]] = {}
        for name in sorted(raw_artifacts):
            if not isinstance(name, str) or not name.strip():
                raise LedgerInputError(f"bindings manifest.role_artifacts.{role}.artifacts has empty name")
            path = _resolve_file(raw_artifacts[name], f"bindings manifest.role_artifacts.{role}.artifacts.{name}", bindings_path.parent)
            artifacts[name] = file_record(path)
        entry: Dict[str, Any] = {"job": final_roles[role], "artifacts": artifacts}
        if "checkpoint" in raw_entry:
            role_checkpoint_path = _resolve_file(raw_entry["checkpoint"], f"bindings manifest.role_artifacts.{role}.checkpoint", bindings_path.parent)
            checkpoint = file_record(role_checkpoint_path)
            entry["checkpoint"] = checkpoint
            existing = job_checkpoints.get(final_roles[role])
            if existing is not None and existing != checkpoint:
                raise LedgerInputError(f"job {final_roles[role]} has conflicting per-role checkpoint bindings")
            job_checkpoints[final_roles[role]] = checkpoint
        if role.startswith("sealed:"):
            _prefix, cell_id, control = role.split(":", 2)
            if set(artifacts) != {"metrics", "per_query"}:
                raise LedgerInputError(
                    f"{role} must bind exactly metrics and per_query artifacts"
                )
            metrics_path = Path(artifacts["metrics"]["path"])
            per_query_path = Path(artifacts["per_query"]["path"])
            try:
                validate_metrics_against_protocol_v2(
                    protocol,
                    _read_json(metrics_path, f"{role} metrics"),
                    _read_jsonl(per_query_path, f"{role} per_query"),
                    cell_id=cell_id,
                    control=control,
                    protocol_file_sha256=protocol_file_sha256,
                    per_query_file_sha256=artifacts["per_query"]["sha256"],
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha256,
                )
            except ProtocolV2Error as exc:
                raise LedgerInputError(f"{role}: {exc}") from exc
        role_artifacts[role] = entry

    try:
        job_result_manifests, role_manifest_bindings = result_binding.validate_and_bind(
            bindings.get("job_result_manifests"),
            anchor=bindings_path.parent,
            final_roles=final_roles,
            evaluation_roles=EVALUATION_RUNAI_ROLES,
            role_artifacts=role_artifacts,
            jobs_by_name=jobs_by_name,
            protocol_path=protocol_path,
            protocol_file_sha256=protocol_file_sha256,
            protocol_content_sha256=str(protocol["protocol_content_sha256"]),
            checkpoint_record=file_record(checkpoint_path),
        )
    except result_manifest.ResultManifestError as exc:
        raise LedgerInputError(str(exc)) from exc
    for job_name, manifest_record in job_result_manifests.items():
        jobs_by_name[job_name]["result_manifest"] = manifest_record
    for role, manifest_binding in role_manifest_bindings.items():
        role_artifacts[role].update(manifest_binding)

    raw_chains = _list(bindings.get("failure_chains"), "bindings manifest.failure_chains", nonempty=False)
    final_job_names = set(final_roles.values())
    failed_jobs = {name for name, job in jobs_by_name.items() if job.get("status") == "Failed"}
    seen_failed: set[str] = set()
    chains: List[Dict[str, str]] = []
    for index, raw_chain in enumerate(raw_chains):
        chain = _mapping(raw_chain, f"bindings manifest.failure_chains[{index}]")
        _require_exact_keys(chain, {"failed_job", "replacement_job", "diagnosis", "fix"}, f"bindings manifest.failure_chains[{index}]")
        failed_job = _text(chain.get("failed_job"), f"bindings manifest.failure_chains[{index}].failed_job")
        replacement_job = _text(chain.get("replacement_job"), f"bindings manifest.failure_chains[{index}].replacement_job")
        diagnosis = _text(chain.get("diagnosis"), f"bindings manifest.failure_chains[{index}].diagnosis")
        fix = _text(chain.get("fix"), f"bindings manifest.failure_chains[{index}].fix")
        if failed_job in seen_failed:
            raise LedgerInputError(f"failed job {failed_job} has multiple failure chains")
        if failed_job not in failed_jobs:
            raise LedgerInputError(f"failure chain references non-failed job {failed_job}")
        if replacement_job not in final_job_names:
            raise LedgerInputError(f"failure chain replacement {replacement_job} is not a final-role job")
        if jobs_by_name[replacement_job].get("status") != "Succeeded":
            raise LedgerInputError(f"failure chain replacement {replacement_job} did not Succeed")
        seen_failed.add(failed_job)
        chains.append({"failed_job": failed_job, "replacement_job": replacement_job, "diagnosis": diagnosis, "fix": fix})
    if seen_failed != failed_jobs:
        raise LedgerInputError(
            "every included failed job needs exactly one chain to a Succeeded final-role replacement: "
            f"missing={sorted(failed_jobs - seen_failed)}, extra={sorted(seen_failed - failed_jobs)}"
        )
    return final_roles, role_artifacts, sorted(chains, key=lambda item: item["failed_job"])


def build_ledger(
    raw_evidence_indexes: Sequence[Path],
    protocol_path: Path,
    selected_root: Path,
    selected_checkpoint: Path,
    bindings_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Validate all inputs and create a new canonical schema-v2 ledger."""
    if output_path.exists() or output_path.is_symlink():
        raise LedgerInputError(f"refusing to overwrite existing ledger: {output_path}")
    root = _resolve_directory(selected_root, "selected root")
    checkpoint = _resolve_file(selected_checkpoint, "selected checkpoint", Path.cwd())
    if not _is_within(checkpoint, root):
        raise LedgerInputError("selected checkpoint must be inside the selected root")
    selected_checkpoint_record = file_record(checkpoint)
    protocol_file = _resolve_file(protocol_path, "frozen protocol", Path.cwd())
    protocol, protocol_identity = _validate_protocol(protocol_file, root)
    jobs, captured_at = _merge_raw_evidence(raw_evidence_indexes, protocol_identity["project"])
    jobs_by_name = {str(job["job"]): job for job in jobs}
    manifest_file = _resolve_file(bindings_path, "bindings manifest", Path.cwd())
    protocol_checkpoint = protocol_identity["checkpoint_artifact"]
    if (
        selected_checkpoint_record["path"] != protocol_checkpoint["path"]
        or selected_checkpoint_record["sha256"] != protocol_checkpoint["sha256"]
    ):
        raise LedgerInputError(
            "selected checkpoint must exactly match "
            "protocol.checkpoint.artifact path/hash"
        )
    final_roles, role_artifacts, failure_chains = _validate_bindings(
        manifest_file,
        jobs_by_name,
        protocol,
        protocol_file,
        sha256_file(protocol_file),
        checkpoint,
        selected_checkpoint_record["sha256"],
    )
    role_checkpoint = role_artifacts["selected_e3_training"].get("checkpoint")
    if role_checkpoint is not None and role_checkpoint != selected_checkpoint_record:
        raise LedgerInputError("selected E3 checkpoint binding disagrees with selected checkpoint")

    for job_name, job in jobs_by_name.items():
        checkpoints = {
            canonical_json(entry["checkpoint"])
            for entry in role_artifacts.values()
            if entry["job"] == job_name and "checkpoint" in entry
        }
        if len(checkpoints) > 1:
            raise LedgerInputError(f"job {job_name} has conflicting checkpoint bindings")
        if checkpoints:
            job["checkpoint"] = json.loads(next(iter(checkpoints)))

    ledger: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ledger_type": LEDGER_TYPE,
        "captured_at": captured_at.isoformat(),
        "project": protocol_identity["project"],
        "identity": {
            "selected_root_sha256": protocol_identity["selected_root"]["sha256"],
            "selected_checkpoint_sha256": selected_checkpoint_record["sha256"],
            "protocol_content_sha256": protocol["protocol_content_sha256"],
        },
        "jobs": [jobs_by_name[name] for name in sorted(jobs_by_name)],
        "final_roles": {role: final_roles[role] for role in sorted(final_roles)},
        "role_artifacts": {role: role_artifacts[role] for role in sorted(role_artifacts)},
        "failure_chains": failure_chains,
    }
    rendered = canonical_json(ledger) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    except FileExistsError as exc:
        raise LedgerInputError(f"refusing to overwrite existing ledger: {output_path}") from exc
    return ledger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-evidence-index", "--evidence-index", action="append", dest="raw_evidence_indexes", type=Path, required=True, help="schema-v1 index from collect_runai_evidence.py; repeat for multiple indexes")
    parser.add_argument("--protocol", "--protocol-manifest", dest="protocol_path", type=Path, required=True, help="frozen sealed-evaluation protocol JSON")
    parser.add_argument("--selected-root", type=Path, required=True, help="frozen selected experiment root")
    parser.add_argument("--selected-checkpoint", type=Path, required=True, help="selected checkpoint inside --selected-root")
    parser.add_argument("--bindings", "--bindings-manifest", dest="bindings_path", type=Path, required=True, help="schema-v2 declarative role/artifact/result-manifest/failure-chain bindings JSON")
    parser.add_argument("--output", type=Path, required=True, help="new canonical schema-v2 ledger path; must not already exist")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ledger = build_ledger(
        args.raw_evidence_indexes,
        args.protocol_path,
        args.selected_root,
        args.selected_checkpoint,
        args.bindings_path,
        args.output,
    )
    print(canonical_json({"ledger": str(args.output.resolve()), "jobs": len(ledger["jobs"])}))


if __name__ == "__main__":
    main()
