from __future__ import annotations

import json
from pathlib import Path
from typing import Any, MutableMapping, Sequence

from scripts import build_evaluation_result_manifest as result_manifest
from scripts import build_runai_success_ledger as runai_ledger


def attach_canonical_result_manifests(
    root: Path,
    protocol_path: Path,
    checkpoint_path: Path,
    jobs: Sequence[MutableMapping[str, Any]],
    final_roles: MutableMapping[str, str],
    role_artifacts: MutableMapping[str, MutableMapping[str, Any]],
    *,
    commit: str = "4aabfbe249d01b57a4cd66e9232f988cd46adbc6",
) -> dict[str, Path]:
    """Add the immutable manifest bindings emitted by the production ledger."""
    fixture_root = root / "result-manifests"
    fixture_root.mkdir(parents=True, exist_ok=True)
    jobs_by_name = {str(job["job"]): job for job in jobs}

    for role in sorted(runai_ledger.EVALUATION_RUNAI_ROLES):
        artifacts = role_artifacts[role]["artifacts"]
        if len(artifacts) >= 2:
            continue
        support_path = fixture_root / f"{role.replace(':', '-')}-support.json"
        support_path.write_text(
            json.dumps({"role": role, "fixture": "independent-result-evidence"}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        artifacts["result_evidence"] = result_manifest.file_record(support_path)

    roles_by_job: dict[str, list[str]] = {}
    for role in sorted(runai_ledger.EVALUATION_RUNAI_ROLES):
        roles_by_job.setdefault(final_roles[role], []).append(role)

    manifest_paths: dict[str, Path] = {}
    for job_name, roles in sorted(roles_by_job.items()):
        evaluations: dict[str, Any] = {}
        for role in roles:
            artifacts = role_artifacts[role]["artifacts"]
            names = sorted(artifacts)
            metrics_name = "metrics" if "metrics" in artifacts else names[0]
            per_query_name = (
                "per_query"
                if "per_query" in artifacts
                else next(name for name in names if name != metrics_name)
            )
            evaluations[role] = {
                "evaluation_id": f"{job_name}:{role}",
                "artifacts": {
                    name: record["path"] for name, record in artifacts.items()
                },
                "metrics_artifact": metrics_name,
                "per_query_artifact": per_query_name,
            }

        spec_path = fixture_root / f"{job_name}.spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "spec_type": "runai_evaluation_result_manifest_spec",
                    "commit": commit,
                    "runai": {"project": jobs_by_name[job_name]["project"], "job": job_name},
                    "checkpoint": str(checkpoint_path.resolve()),
                    "protocol": str(protocol_path.resolve()),
                    "evaluations": evaluations,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_path = fixture_root / f"{job_name}.manifest.json"
        payload = result_manifest.build_manifest(spec_path, manifest_path)
        manifest_record = result_manifest.file_record(manifest_path)
        jobs_by_name[job_name]["result_manifest"] = manifest_record
        logs_path = Path(str(jobs_by_name[job_name]["logs_path"]))
        logs_path.write_text(
            logs_path.read_text(encoding="utf-8")
            + result_manifest.result_manifest_log_marker(manifest_record["sha256"])
            + "\n",
            encoding="utf-8",
        )
        jobs_by_name[job_name]["logs_sha256"] = result_manifest.sha256_file(
            logs_path
        )
        for role in roles:
            role_artifacts[role]["result_manifest_sha256"] = manifest_record[
                "sha256"
            ]
            role_artifacts[role]["evaluation_identity"] = payload["evaluations"][
                role
            ]["evaluation_identity"]
        manifest_paths[job_name] = manifest_path

    return manifest_paths
