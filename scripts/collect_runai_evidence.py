#!/usr/bin/env python3
"""Capture Run:AI job descriptions and logs into a hashed evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


STATUS_RE = re.compile(r"^Status:\s+(\S+)", re.MULTILINE)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def safe_job_name(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", value):
        raise ValueError(f"unsafe Run:AI job name: {value!r}")
    return value


def collect(project: str, jobs: Sequence[str], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for raw_name in jobs:
        name = safe_job_name(raw_name)
        describe_argv = ["runai", "describe", "job", name, "-p", project]
        logs_argv = ["runai", "logs", name, "--project", project]
        describe = run_command(describe_argv)
        logs = run_command(logs_argv)
        describe_path = output_dir / f"{name}.describe.txt"
        logs_path = output_dir / f"{name}.log"
        describe_path.write_bytes(describe.stdout)
        logs_path.write_bytes(logs.stdout)
        match = STATUS_RE.search(describe.stdout.decode("utf-8", errors="replace"))
        records.append({
            "job": name,
            "project": project,
            "status": match.group(1) if match else "unknown",
            "describe_command": describe_argv,
            "describe_returncode": describe.returncode,
            "describe_path": str(describe_path.resolve()),
            "describe_sha256": sha256_bytes(describe.stdout),
            "logs_command": logs_argv,
            "logs_returncode": logs.returncode,
            "logs_path": str(logs_path.resolve()),
            "logs_sha256": sha256_bytes(logs.stdout),
        })
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "jobs": records,
    }
    index_path = output_dir / "runai_evidence_index.json"
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["index_path"] = str(index_path.resolve())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.project, args.job, args.output_dir)
    print(json.dumps({"index": result["index_path"], "jobs": len(result["jobs"])}, sort_keys=True))


if __name__ == "__main__":
    main()
