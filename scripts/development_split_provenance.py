"""Strict provenance checks shared by development split runtime and audit paths."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import subprocess
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
BUILDER_COMMAND = "python scripts/materialize_eval_splits.py"
SPEECH_PARTITIONS = ("train", "dev", "eval")
SPEECH_COMMITMENT_POLICY = "explicit_source_partition_canonical_membership_v1"
SPEECH_ROW_IDENTITY = "source_dataset_plus_utterance_id_else_typed_source_row_id"
SPEECH_ROW_CONTENT = "canonical_json_sha256"
SPEECH_AUDIO_SHA256_FIELD = "audio_sha256"
DATA_CONTRACT_POLICY = "stage_a_canonical_data_root_single_snapshot_v1"
DATA_CONTRACT_SNAPSHOT_SEMANTICS = (
    "path_sha256_bytes_and_jsonl_rows_from_one_bytes_read_per_artifact"
)
DATA_CONTRACT_ARTIFACTS = {
    "dataset_manifest": ("manifest.json", False),
    "text_tasks": ("text_tasks.jsonl", True),
    "text_blocks_train": ("text_blocks_train.jsonl", True),
    "text_blocks_eval": ("text_blocks_eval.jsonl", True),
    "image_captions": ("image_captions.jsonl", True),
    "speech_transcripts": ("speech_transcripts.jsonl", True),
}
DATA_CONTRACT_SOURCE_ALIASES = {
    "image": "image_captions",
    "speech": "speech_transcripts",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()

def parse_jsonl_snapshot(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{label}:{line_number} is blank")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_number} is not an object")
        rows.append(value)
    return rows


def verify_data_contract(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Verify every canonical Stage-A data-root artifact from one byte snapshot."""

    raw_data_dir = manifest.get("data_dir")
    if not isinstance(raw_data_dir, str) or not raw_data_dir:
        raise ValueError("development split manifest data_dir must be explicit")
    data_dir = Path(raw_data_dir)
    if (
        not data_dir.is_absolute()
        or data_dir.is_symlink()
        or not data_dir.is_dir()
        or data_dir.resolve() != data_dir
    ):
        raise ValueError("development split manifest data_dir must be canonical")

    contract = manifest.get("data_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("development split data_contract is missing")
    if contract.get("policy") != DATA_CONTRACT_POLICY:
        raise ValueError("development split data_contract policy is invalid")
    if contract.get("data_dir") != str(data_dir):
        raise ValueError("development split data_contract data_dir mismatch")
    if contract.get("snapshot_semantics") != DATA_CONTRACT_SNAPSHOT_SEMANTICS:
        raise ValueError("development split data_contract snapshot semantics are invalid")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        DATA_CONTRACT_ARTIFACTS
    ):
        raise ValueError(
            "development split data_contract must contain exactly the canonical artifacts"
        )

    verified_artifacts: dict[str, dict[str, Any]] = {}
    parsed_rows: dict[str, list[dict[str, Any]]] = {}
    for name, (filename, is_jsonl) in DATA_CONTRACT_ARTIFACTS.items():
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"development split data_contract.{name} is missing")
        expected_path = data_dir / filename
        raw_path = record.get("path")
        if raw_path != str(expected_path):
            raise ValueError(
                f"development split data_contract.{name} path must equal canonical "
                f"data_dir/{filename}"
            )
        path = Path(str(raw_path))
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise ValueError(
                f"development split data_contract.{name} must be an exact regular "
                f"non-symlink data_dir/{filename}"
            )
        declared_sha = record.get("sha256")
        declared_bytes = record.get("bytes")
        if not isinstance(declared_sha, str) or SHA256_RE.fullmatch(declared_sha) is None:
            raise ValueError(
                f"development split data_contract.{name} sha256 must be exact lowercase SHA256"
            )
        if (
            isinstance(declared_bytes, bool)
            or not isinstance(declared_bytes, int)
            or declared_bytes < 0
        ):
            raise ValueError(
                f"development split data_contract.{name} bytes must be an exact nonnegative integer"
            )

        payload = path.read_bytes()
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != declared_sha:
            raise ValueError(
                f"development split data_contract.{name} SHA256 mismatch"
            )
        if len(payload) != declared_bytes:
            raise ValueError(
                f"development split data_contract.{name} byte count mismatch"
            )
        verified = {
            "path": str(path),
            "sha256": actual_sha,
            "bytes": len(payload),
            "read_status": "single_snapshot_for_hash_bytes_and_rows",
            "content_opened": True,
            "sha256_verified_this_run": True,
            "bytes_verified_this_run": True,
            "hash_bytes_and_rows_same_bytes": True,
        }
        if is_jsonl:
            rows = parse_jsonl_snapshot(
                payload, f"development split data_contract.{name}"
            )
            declared_rows = record.get("rows")
            if (
                isinstance(declared_rows, bool)
                or not isinstance(declared_rows, int)
                or declared_rows < 0
                or declared_rows != len(rows)
            ):
                raise ValueError(
                    f"development split data_contract.{name} row count mismatch"
                )
            verified["rows"] = len(rows)
            verified["rows_verified_this_run"] = True
            parsed_rows[name] = rows
        else:
            if "rows" in record:
                raise ValueError(
                    "development split data_contract.dataset_manifest must not declare rows"
                )
            try:
                dataset_manifest = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "development split data_contract.dataset_manifest is not valid UTF-8 JSON"
                ) from exc
            if not isinstance(dataset_manifest, Mapping):
                raise ValueError(
                    "development split data_contract.dataset_manifest root must be an object"
                )
            verified["json_object_verified_this_run"] = True
        verified_artifacts[name] = verified

    source_files = manifest.get("source_files")
    if not isinstance(source_files, Mapping):
        raise ValueError("development split source_files is missing")
    for alias, artifact_name in DATA_CONTRACT_SOURCE_ALIASES.items():
        record = source_files.get(alias)
        verified = verified_artifacts[artifact_name]
        filename = DATA_CONTRACT_ARTIFACTS[artifact_name][0]
        if not isinstance(record, Mapping):
            raise ValueError(f"development split source_files.{alias} is missing")
        if record.get("path") != verified["path"]:
            raise ValueError(
                f"development split source_files.{alias} must equal canonical "
                f"data_dir/{filename}"
            )
        if record.get("sha256") != verified["sha256"]:
            raise ValueError(
                f"development split source_files.{alias} SHA256 mismatch with data_contract"
            )
        if record.get("bytes") != verified["bytes"]:
            raise ValueError(
                f"development split source_files.{alias} byte count mismatch with data_contract"
            )
        if record.get("rows") != verified["rows"]:
            raise ValueError(
                f"development split source_files.{alias} row count mismatch with data_contract"
            )

    return (
        {
            "policy": DATA_CONTRACT_POLICY,
            "canonical_data_dir": str(data_dir),
            "snapshot_semantics": DATA_CONTRACT_SNAPSHOT_SEMANTICS,
            "artifacts": verified_artifacts,
            "all_artifacts_verified_this_run": True,
        },
        parsed_rows,
    )


def _typed_identity(value: Any) -> list[Any] | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)) or value == "":
        return None
    return [type(value).__name__, value]


def speech_source_row_identity(row: Mapping[str, Any], row_index: int) -> str:
    source_dataset = _typed_identity(row.get("source_dataset"))
    utterance_id = _typed_identity(row.get("utterance_id"))
    if source_dataset is not None and utterance_id is not None:
        identity: list[Any] = [
            "source_utterance",
            *source_dataset,
            *utterance_id,
        ]
    else:
        row_id = _typed_identity(row.get("id"))
        if row_id is None:
            raise ValueError(
                f"speech source row {row_index} has no canonical row identity"
            )
        identity = ["source_row_id", *row_id]
    return canonical_json_sha256(identity)


def speech_source_group_identity(row: Mapping[str, Any], row_index: int) -> str:
    source_dataset = _typed_identity(row.get("source_dataset"))
    speaker_id = _typed_identity(row.get("speaker_id"))
    if source_dataset is None or speaker_id is None:
        raise ValueError(
            f"speech source row {row_index} has no canonical speaker group"
        )
    return canonical_json_sha256([*source_dataset, *speaker_id])


def _source_partition_row(
    source: Mapping[str, Any], split: str, row_index: int, *, annotated: bool
) -> dict[str, Any]:
    row = dict(source)
    if annotated:
        if (
            row.get("eval_split_name") != f"speech_{split}"
            or row.get("eval_split_index") != row_index
        ):
            raise ValueError(
                f"speech_{split} row {row_index} has invalid split annotation"
            )
        row.pop("eval_split_name")
        row.pop("eval_split_index")
    if str(row.get("partition", "")) != split:
        raise ValueError(f"speech_{split} row {row_index} has wrong partition")
    audio_sha256 = row.get(SPEECH_AUDIO_SHA256_FIELD)
    if not isinstance(audio_sha256, str) or SHA256_RE.fullmatch(audio_sha256) is None:
        raise ValueError(
            f"speech_{split} row {row_index} has no exact audio_sha256"
        )
    return row


def speech_partition_record(
    rows: Sequence[Mapping[str, Any]], split: str, *, annotated: bool = False
) -> dict[str, Any]:
    if split not in SPEECH_PARTITIONS:
        raise ValueError(f"invalid speech partition: {split}")
    leaves: list[dict[str, str]] = []
    seen_identities: set[str] = set()
    for row_index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise ValueError(f"speech_{split} row {row_index} is not an object")
        row = _source_partition_row(
            source, split, row_index, annotated=annotated
        )
        identity = speech_source_row_identity(row, row_index)
        if identity in seen_identities:
            raise ValueError("speech partition has duplicate canonical row identity")
        seen_identities.add(identity)
        leaves.append(
            {
                "identity_sha256": identity,
                "content_sha256": canonical_json_sha256(row),
                "group_sha256": speech_source_group_identity(row, row_index),
                "partition": split,
            }
        )
    leaves.sort(key=lambda item: item["identity_sha256"])
    groups = sorted({leaf["group_sha256"] for leaf in leaves})
    return {
        "rows": len(leaves),
        "groups": len(groups),
        "row_identity_root_sha256": canonical_json_sha256(
            [leaf["identity_sha256"] for leaf in leaves]
        ),
        "row_content_root_sha256": canonical_json_sha256(
            [
                [leaf["identity_sha256"], leaf["content_sha256"]]
                for leaf in leaves
            ]
        ),
        "membership_root_sha256": canonical_json_sha256(leaves),
        "group_assignment_sha256": canonical_json_sha256(
            [[group, split] for group in groups]
        ),
    }


def partition_speech_source_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    partitions: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPEECH_PARTITIONS
    }
    for row_index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise ValueError(f"speech source row {row_index} is not an object")
        split = str(source.get("partition", ""))
        if split not in partitions:
            raise ValueError(
                f"speech source row {row_index} has invalid explicit partition"
            )
        partitions[split].append(dict(source))
    if any(not partitions[split] for split in SPEECH_PARTITIONS):
        raise ValueError("speech source does not contain all explicit partitions")
    return partitions


def annotate_speech_partition_rows(
    rows: Sequence[Mapping[str, Any]], split: str
) -> list[dict[str, Any]]:
    if split not in SPEECH_PARTITIONS:
        raise ValueError(f"invalid speech partition: {split}")
    output: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        row["eval_split_name"] = f"speech_{split}"
        row["eval_split_index"] = index
        output.append(row)
    return output


def speech_partition_commitments(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    partitions = partition_speech_source_rows(rows)
    partition_records: dict[str, Any] = {}
    seen_identities: set[str] = set()
    group_assignments: dict[str, str] = {}
    source_index = 0
    for split in SPEECH_PARTITIONS:
        leaves: list[dict[str, str]] = []
        for row in partitions[split]:
            _source_partition_row(row, split, source_index, annotated=False)
            identity = speech_source_row_identity(row, source_index)
            if identity in seen_identities:
                raise ValueError("speech source has duplicate canonical row identity")
            seen_identities.add(identity)
            group_identity = speech_source_group_identity(row, source_index)
            assigned_split = group_assignments.setdefault(group_identity, split)
            if assigned_split != split:
                raise ValueError("speech source speaker group crosses partitions")
            leaves.append(
                {
                    "identity_sha256": identity,
                    "content_sha256": canonical_json_sha256(row),
                    "group_sha256": group_identity,
                    "partition": split,
                }
            )
            source_index += 1
        leaves.sort(key=lambda item: item["identity_sha256"])
        groups = sorted({leaf["group_sha256"] for leaf in leaves})
        partition_records[split] = {
            "rows": len(leaves),
            "groups": len(groups),
            "row_identity_root_sha256": canonical_json_sha256(
                [leaf["identity_sha256"] for leaf in leaves]
            ),
            "row_content_root_sha256": canonical_json_sha256(
                [
                    [leaf["identity_sha256"], leaf["content_sha256"]]
                    for leaf in leaves
                ]
            ),
            "membership_root_sha256": canonical_json_sha256(leaves),
            "group_assignment_sha256": canonical_json_sha256(
                [[group, split] for group in groups]
            ),
        }
    return {
        "policy": SPEECH_COMMITMENT_POLICY,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "source_rows": len(rows),
        "row_identity_semantics": SPEECH_ROW_IDENTITY,
        "row_content_semantics": SPEECH_ROW_CONTENT,
        "group_key": ["source_dataset", "speaker_id"],
        "partitions": partition_records,
        "global_group_assignment_sha256": canonical_json_sha256(
            sorted(
                [[group, split] for group, split in group_assignments.items()]
            )
        ),
    }


def _git(
    repo_root: Path,
    *args: str,
    text: bool = True,
) -> str | bytes:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "-C",
            str(repo_root),
            *args,
        ],
        check=True,
        capture_output=True,
        text=text,
    ).stdout


def _git_root(path: Path) -> Path:
    try:
        root = subprocess.run(
            [
                "git",
                "-c",
                "safe.directory=*",
                "-C",
                str(path),
                "rev-parse",
                "--show-toplevel",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError("development split builder is not inside a Git worktree") from exc
    return Path(root).resolve()


def verify_builder_provenance(
    record: Any,
    *,
    expected_source_commit_sha: str | None = None,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("development split builder provenance is missing")
    raw_path = record.get("path")
    declared_sha = record.get("sha256")
    source_commit = record.get("source_commit_sha")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("development split builder path must be explicit")
    builder_path = Path(raw_path)
    if (
        not builder_path.is_absolute()
        or builder_path.is_symlink()
        or not builder_path.is_file()
        or builder_path.resolve() != builder_path
    ):
        raise ValueError(
            "development split builder must be an exact absolute regular file"
        )
    if not isinstance(declared_sha, str) or SHA256_RE.fullmatch(declared_sha) is None:
        raise ValueError("development split builder sha256 must be exact lowercase SHA256")
    if not isinstance(source_commit, str) or COMMIT_RE.fullmatch(source_commit) is None:
        raise ValueError(
            "development split builder source_commit_sha must be exact lowercase Git SHA"
        )
    if record.get("source_matches_commit") is not True:
        raise ValueError("development split builder must prove source_matches_commit=true")
    if record.get("command") != BUILDER_COMMAND:
        raise ValueError("development split builder command is missing or invalid")
    if (
        expected_source_commit_sha is not None
        and source_commit != expected_source_commit_sha
    ):
        raise ValueError(
            "development split builder source commit does not match launcher HEAD"
        )

    current_sha = sha256_file(builder_path)
    if current_sha != declared_sha:
        raise ValueError("development split builder current SHA256 mismatch")
    repo_root = _git_root(builder_path.parent)
    try:
        builder_relative_path = builder_path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ValueError("development split builder is outside its Git root") from exc
    try:
        _git(repo_root, "cat-file", "-e", f"{source_commit}^{{commit}}")
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "development split builder source commit does not exist"
        ) from exc
    try:
        committed_bytes = _git(
            repo_root,
            "show",
            f"{source_commit}:{builder_relative_path}",
            text=False,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "development split builder is absent at declared source commit"
        ) from exc
    committed_sha = hashlib.sha256(committed_bytes).hexdigest()
    if committed_sha != declared_sha or committed_sha != current_sha:
        raise ValueError(
            "development split builder bytes at source commit do not match current bytes"
        )
    return {
        "path": str(builder_path),
        "sha256": declared_sha,
        "source_commit_sha": source_commit,
        "source_commit_exists": True,
        "source_matches_commit": True,
        "current_bytes_match_commit": True,
        "git_root": str(repo_root),
        "git_relative_path": builder_relative_path,
        "command": BUILDER_COMMAND,
    }


def verify_speech_source_file(
    record: Any,
    *,
    expected_rows: int,
    expected_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(record, Mapping):
        raise ValueError("development split source_files.speech is missing")
    raw_path = record.get("path")
    declared_sha = record.get("sha256")
    declared_rows = record.get("rows")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("development split source_files.speech path must be explicit")
    path = Path(raw_path)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or (expected_path is not None and path != expected_path)
    ):
        raise ValueError(
            "development split source_files.speech must equal canonical "
            "data_dir/speech_transcripts.jsonl"
        )
    if not isinstance(declared_sha, str) or SHA256_RE.fullmatch(declared_sha) is None:
        raise ValueError(
            "development split source_files.speech sha256 must be exact lowercase SHA256"
        )
    if declared_rows != int(expected_rows):
        raise ValueError(
            "development split source_files.speech declared row count mismatch"
        )
    payload = path.read_bytes()
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != declared_sha:
        raise ValueError("development split source_files.speech SHA256 mismatch")
    rows = parse_jsonl_snapshot(
        payload, "development split source_files.speech"
    )
    if len(rows) != int(expected_rows):
        raise ValueError(
            "development split source_files.speech observed row count mismatch"
        )
    return ({
        "path": str(path),
        "sha256": actual_sha,
        "rows": len(rows),
        "size_bytes": len(payload),
        "read_status": "single_snapshot_for_integrity_and_partition_verification",
        "content_opened": True,
        "sha256_verified_this_run": True,
        "rows_verified_this_run": True,
        "content_used_for_partition_verification": True,
        "content_used_for_training": False,
        "hash_and_rows_same_bytes": True,
    }, rows)


def _speech_audio_relative_path(
    raw_path: str, *, data_dir: Path, row_index: int
) -> tuple[Path, Path]:
    canonical_data_dir = data_dir.resolve(strict=True)
    if (
        not data_dir.is_absolute()
        or data_dir.is_symlink()
        or not data_dir.is_dir()
        or canonical_data_dir != data_dir
    ):
        raise ValueError("speech audio data_dir must be canonical")
    candidate = Path(raw_path)
    if any(component == ".." for component in candidate.parts):
        raise ValueError(
            f"speech source row {row_index} audio path has parent traversal"
        )
    lexical_path = candidate if candidate.is_absolute() else data_dir / candidate
    try:
        relative_path = lexical_path.relative_to(data_dir)
    except ValueError as exc:
        raise ValueError(
            f"speech source row {row_index} audio path escapes canonical data_dir"
        ) from exc
    if not relative_path.parts:
        raise ValueError(
            f"speech source row {row_index} audio path is not a regular file"
        )
    return lexical_path, relative_path


def secure_speech_audio_snapshot(
    raw_path: str, *, data_dir: Path, row_index: int
) -> tuple[Path, bytes]:
    lexical_path, relative_path = _speech_audio_relative_path(
        raw_path, data_dir=data_dir, row_index=row_index
    )
    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, flag) for flag in required_flags):
        raise RuntimeError("strict speech snapshots require POSIX no-follow flags")
    read_flags = os.O_RDONLY | os.O_CLOEXEC
    directory_flags = read_flags | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = read_flags | os.O_NOFOLLOW
    directory_fds: list[int] = []
    try:
        current_fd = os.open(data_dir, directory_flags)
        directory_fds.append(current_fd)
        for component in relative_path.parts[:-1]:
            component_stat = os.stat(
                component, dir_fd=current_fd, follow_symlinks=False
            )
            if stat.S_ISLNK(component_stat.st_mode):
                raise ValueError(
                    f"speech source row {row_index} audio path has symlink component"
                )
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            directory_fds.append(next_fd)
            current_fd = next_fd

        final_component = relative_path.parts[-1]
        component_stat = os.stat(
            final_component, dir_fd=current_fd, follow_symlinks=False
        )
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValueError(
                f"speech source row {row_index} audio path has symlink component"
            )
        file_fd = os.open(final_component, file_flags, dir_fd=current_fd)
        try:
            opened_stat = os.fstat(file_fd)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise ValueError(
                    f"speech source row {row_index} audio path is not a regular file"
                )
            with os.fdopen(file_fd, "rb", closefd=True) as handle:
                file_fd = -1
                payload = handle.read()
        finally:
            if file_fd >= 0:
                os.close(file_fd)
    except FileNotFoundError as exc:
        raise ValueError(
            f"speech source row {row_index} audio file is missing"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"speech source row {row_index} audio path has symlink or invalid component"
        ) from exc
    finally:
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
    return lexical_path, payload


def resolve_speech_audio_path(
    raw_path: str, *, data_dir: Path, row_index: int
) -> Path:
    path, _ = secure_speech_audio_snapshot(
        raw_path, data_dir=data_dir, row_index=row_index
    )
    return path


def verify_speech_audio_rows(
    rows: Sequence[Mapping[str, Any]], *, data_dir: Path
) -> dict[str, Any]:
    canonical_data_dir = data_dir.resolve(strict=True)
    if (
        not data_dir.is_absolute()
        or data_dir.is_symlink()
        or not data_dir.is_dir()
        or canonical_data_dir != data_dir
    ):
        raise ValueError("speech audio data_dir must be canonical")
    path_cache: dict[Path, tuple[str, dict[str, Any]]] = {}
    verified: list[list[str]] = []
    for row_index, row in enumerate(rows):
        declared_sha = row.get(SPEECH_AUDIO_SHA256_FIELD)
        if not isinstance(declared_sha, str) or SHA256_RE.fullmatch(declared_sha) is None:
            raise ValueError(
                f"speech source row {row_index} has no exact audio_sha256"
            )
        raw_path = row.get("audio_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"speech source row {row_index} has no audio_path")
        path, payload = secure_speech_audio_snapshot(
            raw_path, data_dir=canonical_data_dir, row_index=row_index
        )
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != declared_sha:
            raise ValueError(
                f"speech source row {row_index} audio SHA256 mismatch"
            )
        cached = path_cache.get(path)
        if cached is None:
            try:
                with wave.open(io.BytesIO(payload), "rb") as wav_file:
                    channels = int(wav_file.getnchannels())
                    sample_width = int(wav_file.getsampwidth())
                    sample_rate = int(wav_file.getframerate())
                    frame_count = int(wav_file.getnframes())
                    compression_type = str(wav_file.getcomptype())
                    if channels <= 0:
                        raise wave.Error("non-positive channel count")
                    if sample_width <= 0:
                        raise wave.Error("non-positive sample width")
                    if sample_rate <= 0:
                        raise wave.Error("non-positive sample rate")
                    if frame_count <= 0:
                        raise wave.Error("non-positive frame count")
                    if compression_type != "NONE":
                        raise wave.Error(
                            f"unsupported compression type {compression_type!r}"
                        )
                    decoded_frames = wav_file.readframes(frame_count)
                    expected_bytes = frame_count * channels * sample_width
                    if len(decoded_frames) != expected_bytes:
                        raise wave.Error("decoded PCM frame payload is truncated")
            except (EOFError, wave.Error) as exc:
                raise ValueError(
                    f"speech source row {row_index} audio is not a valid "
                    f"decodable WAV container: {path} ({exc})"
                ) from exc
            audio_format = {
                "container": "WAV",
                "encoding": "PCM",
                "channels": channels,
                "sample_width_bytes": sample_width,
                "sample_rate_hz": sample_rate,
                "frame_count": frame_count,
            }
            path_cache[path] = (actual_sha, audio_format)
        else:
            _, audio_format = cached
        if actual_sha != declared_sha:
            raise ValueError(
                f"speech source row {row_index} audio SHA256 mismatch"
            )
        verified.append(
            [
                speech_source_row_identity(row, row_index),
                str(path),
                declared_sha,
                canonical_json_sha256(audio_format),
            ]
        )
    format_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for _, audio_format in path_cache.values():
        key = (
            audio_format["container"],
            audio_format["encoding"],
            audio_format["channels"],
            audio_format["sample_width_bytes"],
            audio_format["sample_rate_hz"],
        )
        group = format_groups.setdefault(
            key,
            {
                **{
                    name: audio_format[name]
                    for name in (
                        "container",
                        "encoding",
                        "channels",
                        "sample_width_bytes",
                        "sample_rate_hz",
                    )
                },
                "unique_files": 0,
                "min_frame_count": audio_format["frame_count"],
                "max_frame_count": audio_format["frame_count"],
                "total_frame_count": 0,
            },
        )
        group["unique_files"] += 1
        group["min_frame_count"] = min(
            group["min_frame_count"], audio_format["frame_count"]
        )
        group["max_frame_count"] = max(
            group["max_frame_count"], audio_format["frame_count"]
        )
        group["total_frame_count"] += audio_format["frame_count"]
    return {
        "audio_sha256_field": SPEECH_AUDIO_SHA256_FIELD,
        "audio_path_binding_semantics": (
            "row_identity_plus_data_dir_fd_component_nofollow_regular_file_fd_plus_sha256"
        ),
        "audio_rows_verified": len(rows),
        "unique_audio_files_verified": len(path_cache),
        "audio_row_binding_root_sha256": canonical_json_sha256(
            sorted([row[0], row[1], row[2]] for row in verified)
        ),
        "audio_bytes_verified_this_run": True,
        "audio_format_binding_semantics": (
            "row_identity_plus_audio_sha256_plus_decoded_pcm_wav_format_sha256"
        ),
        "audio_format_binding_root_sha256": canonical_json_sha256(
            sorted([row[0], row[2], row[3]] for row in verified)
        ),
        "audio_format_summary": {
            "parser": "python_stdlib_wave",
            "formats": [format_groups[key] for key in sorted(format_groups)],
        },
        "audio_wav_decoded_this_run": True,
    }



def verify_speech_partition_derivation(
    manifest: Mapping[str, Any],
    *,
    expected_partition_rows: Mapping[str, int],
    observed_partition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    verified_data_contract: tuple[
        dict[str, Any], dict[str, list[dict[str, Any]]]
    ]
    | None = None,
) -> dict[str, Any]:
    raw_data_dir = manifest.get("data_dir")
    if not isinstance(raw_data_dir, str) or not raw_data_dir:
        raise ValueError("development split manifest data_dir must be explicit")
    data_dir = Path(raw_data_dir)
    if (
        not data_dir.is_absolute()
        or data_dir.is_symlink()
        or not data_dir.is_dir()
        or data_dir.resolve() != data_dir
    ):
        raise ValueError("development split manifest data_dir must be canonical")
    expected_source_path = data_dir / "speech_transcripts.jsonl"
    if expected_source_path.parent.resolve() != data_dir:
        raise ValueError("development speech source escapes canonical data_dir")

    expected_total = sum(
        int(expected_partition_rows[split]) for split in SPEECH_PARTITIONS
    )
    data_contract, contract_rows = (
        verified_data_contract
        if verified_data_contract is not None
        else verify_data_contract(manifest)
    )
    source_provenance = dict(
        data_contract["artifacts"]["speech_transcripts"]
    )
    source_provenance.update(
        {
            "size_bytes": source_provenance["bytes"],
            "read_status": (
                "single_snapshot_for_integrity_and_partition_verification"
            ),
            "content_used_for_partition_verification": True,
            "content_used_for_training": False,
        }
    )
    source_rows = contract_rows["speech_transcripts"]
    if source_provenance["path"] != str(expected_source_path):
        raise ValueError(
            "development split source_files.speech must equal canonical "
            "data_dir/speech_transcripts.jsonl"
        )
    if len(source_rows) != expected_total:
        raise ValueError(
            "development split source_files.speech observed row count mismatch"
        )
    audio_provenance = verify_speech_audio_rows(source_rows, data_dir=data_dir)
    computed = speech_partition_commitments(
        source_rows,
        source_path=str(expected_source_path),
        source_sha256=source_provenance["sha256"],
    )
    declared = manifest.get("speech_partition_commitments")
    if declared != computed:
        raise ValueError(
            "development speech partition commitments do not reproduce raw source"
        )

    raw_partitions = partition_speech_source_rows(source_rows)
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("development split manifest is missing files")
    for split in SPEECH_PARTITIONS:
        expected_count = int(expected_partition_rows[split])
        if len(raw_partitions[split]) != expected_count:
            raise ValueError(
                f"development speech source {split} row count mismatch"
            )
        file_record = files.get(f"speech_{split}")
        if not isinstance(file_record, Mapping):
            raise ValueError(f"development files.speech_{split} is missing")
        expected_membership = computed["partitions"][split][
            "membership_root_sha256"
        ]
        if (
            file_record.get("source_partition_membership_sha256")
            != expected_membership
        ):
            raise ValueError(
                f"development files.speech_{split} membership commitment mismatch"
            )

    for split, observed in observed_partition_rows.items():
        if split not in SPEECH_PARTITIONS:
            raise ValueError(f"unexpected observed speech partition: {split}")
        expected_output = annotate_speech_partition_rows(
            raw_partitions[split], split
        )
        if canonical_json_bytes(list(observed)) != canonical_json_bytes(
            expected_output
        ):
            raise ValueError(
                f"development speech_{split} is not an exact reproduction "
                "of its raw source partition"
            )

    return {
        **source_provenance,
        **audio_provenance,
        "canonical_data_dir": str(data_dir),
        "derivation_verified": True,
        "exact_reproduction_splits": sorted(observed_partition_rows),
        "reserved_eval_split_file_opened": False,
        "raw_source_eval_rows_read_for_partition_verification": True,
        "partition_commitments": computed,
        "data_contract": data_contract,
    }
