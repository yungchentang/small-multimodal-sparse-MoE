#!/usr/bin/env python3
"""Build a compact, sanitized, reproducible public evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024
AGGREGATE_INPUT = "aggregate"
PER_QUERY_INPUT = "per_query"
CHECKPOINT_INPUT = "checkpoint"
COPIED_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".svg"}
PER_QUERY_SUFFIXES = {".csv", ".json", ".jsonl"}
ALLOWED_INPUT_SUFFIXES = COPIED_SUFFIXES | {".pdf"}
ROLE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

CACHE_COMPONENTS = {
    ".cache",
    "cache",
    "caches",
    "feature_cache",
    "feature-cache",
    "__pycache__",
}
SOURCE_DATA_COMPONENTS = {"dataset", "datasets", "raw_data", "raw-data", "source_data", "source-data"}
FORBIDDEN_INPUT_ROLES = {
    "audio",
    "cache",
    "checkpoint",
    "dataset",
    "features",
    "image",
    "media",
    "model_weights",
    "raw_data",
    "raw_media",
    "source_data",
    "source_dataset",
    "test_data",
    "train_data",
    "training_data",
    "validation_data",
    "video",
}

CREDENTIAL_KEY_RE = re.compile(
    r"(?:^|_)(?:api_?key|apikey|authorization|client_secret|credential(?:s)?|password|passwd|private_key|secret)(?:_|$)",
    re.IGNORECASE,
)
CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|client[_-]?secret|"
    r"credential(?:s)?|password|passwd|private[_-]?key|secret)\b\s*[:=]\s*[\"']?[^\s,\"'}]+"
)
CREDENTIAL_VALUE_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{8,}"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{8,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"https?://[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),
)

EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
EPFL_REGISTRY_IMAGE_RE = re.compile(
    r"(?i)\b(?:[a-z0-9-]+\.)*(?:registry|reg)[a-z0-9.-]*\.epfl\.ch(?::\d+)?/"
    r"[A-Za-z0-9_./-]+(?::[A-Za-z0-9_.-]+|@sha256:[a-f0-9]{64})?"
)
EPFL_HOST_RE = re.compile(r"(?i)\b(?:[a-z0-9-]+\.)+epfl\.ch(?::\d+)?\b")
CLUSTER_PROJECT_RE = re.compile(
    r"(?i)\b(?:runai-[a-z0-9][a-z0-9._-]*|(?:lidiap|rcp)(?:-[a-z0-9][a-z0-9._]*)+)\b"
)
WINDOWS_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])(?:[A-Z]:\\|\\\\)[^\r\n\"'<>|]+")
TILDE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_$])~/(?:[^\s\"'`<>|]+)")
POSIX_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_$:/<])/(?!/)(?:[^/\s\"'`<>|,;:{}()\[\]]+/)*"
    r"[^/\s\"'`<>|,;:{}()\[\]]+"
)
USER_FROM_PATH_RE = re.compile(r"(?i)/(?:home|users?|scratch)/([A-Za-z0-9._-]+)")

CONTENT_FIELD_PARTS = {
    "answer",
    "caption",
    "completion",
    "content",
    "conversation",
    "generated_text",
    "generation",
    "input_text",
    "instruction",
    "messages",
    "output_text",
    "prompt",
    "question",
    "query",
    "raw_text",
    "reasoning",
    "reference_text",
    "response",
    "target_text",
    "text",
    "transcript",
}
MEDIA_FIELD_PARTS = {
    "audio",
    "base64",
    "bytes",
    "image",
    "media",
    "pixels",
    "video",
    "waveform",
}
VECTOR_FIELD_PARTS = {"activation", "embedding", "feature", "hidden_state", "logits", "vector"}
PROTOCOL_FIELDS = {
    "aggregation",
    "benchmark",
    "condition",
    "dataset",
    "decoding",
    "direction",
    "evaluator",
    "fold",
    "k",
    "max_candidates",
    "metadata",
    "method",
    "mode",
    "model",
    "name",
    "phase",
    "protocol",
    "protocol_metadata",
    "protocol_version",
    "schema_version",
    "seed",
    "setting",
    "split",
    "subset",
    "task",
    "temperature",
    "threshold",
    "top_k",
    "type",
    "unit",
    "variant",
    "version",
}

MODEL_PAYLOAD_FIELD_PARTS = {"checkpoint", "checkpoints", "ckpt", "weight", "weights"}
PER_QUERY_SCALAR_FIELDS = (PROTOCOL_FIELDS - {"metadata", "protocol_metadata"}) | {
    "accuracy",
    "auc",
    "candidate_count",
    "class",
    "confidence",
    "correct",
    "correctness",
    "delta",
    "exact_match",
    "f1",
    "is_correct",
    "label",
    "loss",
    "margin",
    "metric",
    "modality",
    "n_candidates",
    "num_candidates",
    "passed",
    "precision",
    "prediction",
    "probability",
    "rank",
    "recall",
    "score",
    "success",
}


class BundleError(ValueError):
    """A validation failure that must abort bundle creation."""


@dataclass(frozen=True)
class SourceSpec:
    role: str
    path: Path
    input_type: str


@dataclass(frozen=True)
class SourceBlob:
    spec: SourceSpec
    resolved_path: Path
    data: bytes
    sha256: str


@dataclass
class FieldAudit:
    applied: bool = False
    kept: Set[str] = field(default_factory=set)
    dropped: Set[str] = field(default_factory=set)


@dataclass
class SanitizeStats:
    private_path_replacements: int = 0
    user_replacements: int = 0
    epfl_value_replacements: int = 0
    registry_image_replacements: int = 0


@dataclass(frozen=True)
class PreparedArtifact:
    role: str
    suffix: str
    output_path: str
    data: bytes
    manifest_entry: Mapping[str, Any]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_field_name(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_role_path(value: str, option: str) -> SourceSpec:
    input_types = {
        "--input": AGGREGATE_INPUT,
        "--per-query-input": PER_QUERY_INPUT,
        "--checkpoint": CHECKPOINT_INPUT,
    }
    if option not in input_types:
        raise BundleError(f"unsupported input option: {option}")
    if "=" not in value:
        raise BundleError(f"{option} must use ROLE=PATH: {value!r}")
    role, raw_path = value.split("=", 1)
    if not ROLE_RE.fullmatch(role):
        raise BundleError(
            f"invalid role {role!r}; use ASCII letters, digits, '.', '_' or '-' and start with a letter or digit"
        )
    if not raw_path:
        raise BundleError(f"{option} path is empty for role {role!r}")
    input_type = input_types[option]
    normalized_role = normalize_field_name(role)
    if normalized_role in FORBIDDEN_INPUT_ROLES and input_type != CHECKPOINT_INPUT:
        raise BundleError(f"role {role!r} denotes forbidden source data, media, cache, features, or checkpoint bytes")
    if CREDENTIAL_KEY_RE.search(normalized_role) or normalized_role.endswith("_token"):
        raise BundleError(f"role {role!r} is credential-like")
    if CLUSTER_PROJECT_RE.search(role) or EPFL_HOST_RE.search(role):
        raise BundleError(f"role {role!r} contains environment-specific information")
    return SourceSpec(role=role, path=Path(raw_path).expanduser(), input_type=input_type)


def reject_duplicate_roles(inputs: Sequence[SourceSpec], checkpoints: Sequence[SourceSpec]) -> None:
    seen: Dict[str, str] = {}
    for spec in list(inputs) + list(checkpoints):
        folded = spec.role.casefold()
        if folded in seen:
            raise BundleError(f"duplicate role {spec.role!r}; already defined as {seen[folded]!r}")
        seen[folded] = spec.role


def regular_file_stat(path: Path) -> Tuple[Path, os.stat_result]:
    try:
        initial = path.lstat()
    except FileNotFoundError as exc:
        raise BundleError(f"input does not exist: {path}") from exc
    except OSError as exc:
        raise BundleError(f"cannot inspect input {path}: {exc}") from exc
    if not stat.S_ISREG(initial.st_mode):
        raise BundleError(f"input is not a regular file (symlinks are not accepted): {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"cannot resolve input {path}: {exc}") from exc
    return resolved, initial


def ensure_unchanged(before: os.stat_result, after: os.stat_result, path: Path) -> None:
    before_key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_key != after_key:
        raise BundleError(f"input changed while it was being read: {path}")


def read_artifact(spec: SourceSpec, max_file_bytes: int) -> SourceBlob:
    resolved, initial = regular_file_stat(spec.path)
    suffix = resolved.suffix.lower()
    if suffix not in COPIED_SUFFIXES:
        allowed = ", ".join(sorted(COPIED_SUFFIXES))
        raise BundleError(f"unsupported artifact extension {suffix or '<none>'!r} for role {spec.role!r}; allowed: {allowed}")
    if spec.input_type == PER_QUERY_INPUT and suffix not in PER_QUERY_SUFFIXES:
        allowed = ", ".join(sorted(PER_QUERY_SUFFIXES))
        raise BundleError(
            f"per-query input for role {spec.role!r} must use a structured extension; allowed: {allowed}"
        )

    lower_parts = {part.lower() for part in resolved.parts}
    if lower_parts & CACHE_COMPONENTS:
        raise BundleError(f"cache artifacts are forbidden for role {spec.role!r}")
    if lower_parts & SOURCE_DATA_COMPONENTS:
        raise BundleError(f"source dataset artifacts are forbidden for role {spec.role!r}")
    if initial.st_size > max_file_bytes:
        raise BundleError(
            f"artifact for role {spec.role!r} is {initial.st_size} bytes, exceeding --max-file-bytes={max_file_bytes}"
        )

    try:
        with spec.path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise BundleError(f"input is not a regular file: {spec.path}")
            ensure_unchanged(initial, opened, spec.path)
            data = handle.read(max_file_bytes + 1)
            finished = os.fstat(handle.fileno())
    except OSError as exc:
        raise BundleError(f"cannot read input {spec.path}: {exc}") from exc
    ensure_unchanged(opened, finished, spec.path)
    if len(data) > max_file_bytes:
        raise BundleError(f"artifact for role {spec.role!r} exceeds --max-file-bytes={max_file_bytes}")
    if len(data) != finished.st_size:
        raise BundleError(f"could not read the complete input for role {spec.role!r}")
    return SourceBlob(spec=spec, resolved_path=resolved, data=data, sha256=sha256_bytes(data))


def fingerprint_file(spec: SourceSpec, source_kind: str, require_pdf: bool = False) -> Mapping[str, Any]:
    resolved, initial = regular_file_stat(spec.path)
    if require_pdf:
        if resolved.suffix.lower() != ".pdf":
            raise BundleError(f"PDF fingerprint input for role {spec.role!r} must use a .pdf extension")
        lower_parts = {part.lower() for part in resolved.parts}
        if lower_parts & CACHE_COMPONENTS:
            raise BundleError(f"cache artifacts are forbidden for role {spec.role!r}")
        if lower_parts & SOURCE_DATA_COMPONENTS:
            raise BundleError(f"source dataset artifacts are forbidden for role {spec.role!r}")

    digest = hashlib.sha256()
    total = 0
    head = b""
    tail = b""
    try:
        with spec.path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise BundleError(f"{source_kind} is not a regular file: {spec.path}")
            ensure_unchanged(initial, opened, spec.path)
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not head:
                    head = chunk[:5]
                tail = (tail + chunk)[-4096:]
                digest.update(chunk)
                total += len(chunk)
            finished = os.fstat(handle.fileno())
    except OSError as exc:
        raise BundleError(f"cannot fingerprint {source_kind} {spec.path}: {exc}") from exc
    ensure_unchanged(opened, finished, spec.path)
    if total != finished.st_size:
        raise BundleError(f"could not read the complete {source_kind} for role {spec.role!r}")
    if require_pdf and (not head.startswith(b"%PDF-") or b"%%EOF" not in tail):
        raise BundleError(f"artifact for role {spec.role!r} is not a complete PDF")
    return {
        "role": spec.role,
        "source_sha256": digest.hexdigest(),
        "source_size_bytes": total,
    }


def fingerprint_checkpoint(spec: SourceSpec) -> Mapping[str, Any]:
    return fingerprint_file(spec, "checkpoint")


def fingerprint_pdf(spec: SourceSpec) -> Mapping[str, Any]:
    return {"format": "pdf", **fingerprint_file(spec, "PDF", require_pdf=True)}


def decode_text(data: bytes, role: str) -> str:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BundleError(f"artifact for role {role!r} is not valid UTF-8") from exc
    if "\x00" in text:
        raise BundleError(f"artifact for role {role!r} contains NUL bytes")
    return text


def reject_credentials_in_text(text: str, location: str) -> None:
    if CREDENTIAL_ASSIGNMENT_RE.search(text):
        raise BundleError(f"credential-like assignment found in {location}")
    for pattern in CREDENTIAL_VALUE_PATTERNS:
        if pattern.search(text):
            raise BundleError(f"credential-like value found in {location}")


def is_credential_key(key: str, value: Any) -> bool:
    normalized = normalize_field_name(key)
    if CREDENTIAL_KEY_RE.search(normalized):
        return True
    if normalized == "token" or normalized.endswith("_token"):
        return isinstance(value, str)
    return False


def scan_structured_credentials(value: Any, location: str = "JSON") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if is_credential_key(str(key), child):
                raise BundleError(f"credential-like key {key!r} found in {location}")
            scan_structured_credentials(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan_structured_credentials(child, f"{location}[{index}]")
    elif isinstance(value, str):
        reject_credentials_in_text(value, location)


def discover_usernames(texts: Iterable[str], source_paths: Iterable[Path]) -> Set[str]:
    candidates: Set[str] = set()
    combined = list(texts) + [str(path) for path in source_paths]
    for text in combined:
        for match in USER_FROM_PATH_RE.finditer(text):
            candidates.add(match.group(1))
    ignored = {"home", "root", "scratch", "tmp", "user", "users"}
    return {
        value
        for value in candidates
        if len(value) >= 3 and value.lower() not in ignored and re.fullmatch(r"[A-Za-z0-9._-]+", value)
    }


class TextSanitizer:
    def __init__(self, repo_root: Path, usernames: Iterable[str]) -> None:
        self.repo_root = str(repo_root.resolve())
        self.usernames = tuple(sorted(set(usernames), key=lambda item: (-len(item), item)))

    @staticmethod
    def _sub(pattern: re.Pattern[str], replacement: str, text: str) -> Tuple[str, int]:
        return pattern.subn(replacement, text)

    def _path_placeholder(self, raw_path: str) -> str:
        normalized = raw_path.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        lower_parts = [part.lower() for part in parts]
        for markers, placeholder in (
            ({"data", "dataset", "datasets"}, "$DATA_ROOT"),
            ({"artifact", "artifacts", "output", "outputs", "run", "runs"}, "$RUN_ROOT"),
        ):
            for index, part in enumerate(lower_parts):
                if part in markers:
                    relative = "/".join(parts[index + 1 :])
                    return placeholder + (f"/{relative}" if relative else "")
        if normalized == self.repo_root or normalized.startswith(self.repo_root + "/"):
            relative = normalized[len(self.repo_root) :].lstrip("/")
            return "$REPO_ROOT" + (f"/{relative}" if relative else "")
        return "<private-path>"

    def sanitize(self, text: str, stats: SanitizeStats, key: Optional[str] = None) -> str:
        if key is not None and normalize_field_name(key) in {
            "container_image",
            "image_uri",
            "registry_image",
            "runner_image",
        } and text.strip():
            stats.registry_image_replacements += 1
            return "<registry-image>"

        text, count = self._sub(EPFL_REGISTRY_IMAGE_RE, "<registry-image>", text)
        stats.registry_image_replacements += count
        text, count = self._sub(EMAIL_RE, "<email>", text)
        stats.user_replacements += count

        def replace_posix(match: re.Match[str]) -> str:
            return self._path_placeholder(match.group(0))

        text, count = POSIX_PATH_RE.subn(replace_posix, text)
        stats.private_path_replacements += count
        text, count = WINDOWS_PATH_RE.subn("<private-path>", text)
        stats.private_path_replacements += count
        text, count = TILDE_PATH_RE.subn("<private-path>", text)
        stats.private_path_replacements += count

        text, count = self._sub(EPFL_HOST_RE, "<epfl-host>", text)
        stats.epfl_value_replacements += count
        text, count = self._sub(CLUSTER_PROJECT_RE, "<cluster-project>", text)
        stats.epfl_value_replacements += count

        for username in self.usernames:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_.-]){re.escape(username)}(?![A-Za-z0-9_.-])",
                re.IGNORECASE,
            )
            text, count = pattern.subn("<user>", text)
            stats.user_replacements += count
        self.assert_public(text)
        return text

    def assert_public(self, text: str) -> None:
        reject_credentials_in_text(text, "sanitized output")
        if EPFL_REGISTRY_IMAGE_RE.search(text) or EPFL_HOST_RE.search(text) or CLUSTER_PROJECT_RE.search(text):
            raise BundleError("environment-specific value remained after sanitization")
        if WINDOWS_PATH_RE.search(text) or TILDE_PATH_RE.search(text) or POSIX_PATH_RE.search(text):
            raise BundleError("absolute private path remained after sanitization")
        for username in self.usernames:
            if re.search(
                rf"(?<![A-Za-z0-9_.-]){re.escape(username)}(?![A-Za-z0-9_.-])",
                text,
                re.IGNORECASE,
            ):
                raise BundleError("private username remained after sanitization")


def json_object_without_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    raise BundleError(f"non-finite JSON number {value!r} is forbidden")


def parse_json(text: str, location: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=json_object_without_duplicates,
            parse_constant=reject_json_constant,
        )
    except BundleError:
        raise
    except json.JSONDecodeError as exc:
        raise BundleError(f"invalid JSON in {location}: {exc}") from exc


def field_is_id_or_hash(normalized: str) -> bool:
    return (
        normalized in {"id", "ids", "uid", "uuid", "hash", "sha", "sha256", "digest"}
        or normalized.endswith(("_id", "_ids", "_hash", "_sha", "_sha256", "_digest"))
    )


def aggregate_field_category(normalized: str) -> Optional[str]:
    if normalized in {"container_image", "registry_image", "runner_image"}:
        return None

    tokens = {token for token in normalized.split("_") if token}
    compact = normalized.replace("_", "")

    content_tokens = {
        "answer",
        "answers",
        "caption",
        "captions",
        "completion",
        "completions",
        "content",
        "conversation",
        "generation",
        "generations",
        "instruction",
        "messages",
        "prompt",
        "prompts",
        "question",
        "questions",
        "query",
        "reasoning",
        "response",
        "responses",
        "text",
        "transcript",
        "transcripts",
    }
    if normalized in CONTENT_FIELD_PARTS or normalized.endswith("_text") or tokens & content_tokens:
        return "content"

    if tokens & MODEL_PAYLOAD_FIELD_PARTS or "statedict" in compact:
        return "checkpoint/weights/state_dict"

    media_tokens = {
        "audio",
        "audios",
        "frame",
        "frames",
        "image",
        "images",
        "media",
        "pixel",
        "pixels",
        "thumbnail",
        "thumbnails",
        "video",
        "videos",
        "waveform",
    }
    base64_like = (
        "base64" in compact
        or re.search(r"(?:^|_)(?:b64|base_64)(?:_|$)", normalized) is not None
    )
    if (
        normalized in MEDIA_FIELD_PARTS
        or tokens & media_tokens
        or normalized.endswith(("_bytes", "_data_uri"))
        or base64_like
    ):
        return "media/base64"

    vector_tokens = {
        "activation",
        "activations",
        "embedding",
        "embeddings",
        "feature",
        "features",
        "hiddenstate",
        "logit",
        "logits",
        "tensor",
        "tensors",
        "vector",
        "vectors",
    }
    if tokens & vector_tokens or "hiddenstate" in compact:
        return "vector"

    return None


def reject_aggregate_fields(value: Any, location: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            normalized = normalize_field_name(key_text)
            field_location = f"{location}.{key_text}"
            category = aggregate_field_category(normalized)
            if category is not None:
                raise BundleError(
                    f"forbidden aggregate {category} field {key_text!r} found at {field_location}"
                )
            reject_aggregate_fields(child, field_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_aggregate_fields(child, f"{location}[{index}]")


def field_is_per_query_scalar_allowed(normalized: str) -> bool:
    return field_is_id_or_hash(normalized) or normalized in PER_QUERY_SCALAR_FIELDS


def string_is_content_free(value: str, normalized_key: str) -> bool:
    if any(character in value for character in "\r\n\x00"):
        return False
    if value.startswith(("/", "~/", "\\\\")) or re.match(r"(?i)^[A-Z]:\\", value):
        return False
    if re.match(r"(?i)^(?:https?|file|s3|gs)://", value):
        return False
    protocol_field = normalized_key in PROTOCOL_FIELDS
    limit = 512 if field_is_id_or_hash(normalized_key) else 256 if protocol_field else 128
    word_limit = 32 if protocol_field else 12
    return len(value) <= limit and len(value.split()) <= word_limit


def per_query_scalar_is_allowed(value: Any, normalized_key: str) -> bool:
    if value is None or isinstance(value, (bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, str):
        return string_is_content_free(value, normalized_key)
    return False


def filter_per_query_record(record: Mapping[str, Any], audit: FieldAudit) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    audit.applied = True
    for key, value in record.items():
        key_text = str(key)
        normalized = normalize_field_name(key_text)
        if not field_is_per_query_scalar_allowed(normalized) or not per_query_scalar_is_allowed(
            value, normalized
        ):
            audit.dropped.add(key_text)
            continue
        result[key_text] = value
        audit.kept.add(key_text)
    return result


def filter_per_query_json(value: Any, audit: FieldAudit) -> Any:
    if isinstance(value, list):
        if not value or not all(isinstance(item, Mapping) for item in value):
            raise BundleError("per-query JSON must contain one or more object records")
        filtered = [filter_per_query_record(item, audit) for item in value]
        if any(not row for row in filtered):
            raise BundleError("per-query JSON contains a row with no public evidence fields")
        return filtered
    if isinstance(value, Mapping):
        filtered_record = filter_per_query_record(value, audit)
        if not filtered_record:
            raise BundleError("per-query JSON contains no public evidence fields")
        return filtered_record
    raise BundleError("per-query JSON must be an object or an array of objects")


def sanitize_structure(value: Any, sanitizer: TextSanitizer, stats: SanitizeStats) -> Any:
    if isinstance(value, Mapping):
        result: MutableMapping[str, Any] = {}
        for key, child in value.items():
            sanitized_key = sanitizer.sanitize(str(key), stats)
            if sanitized_key in result:
                raise BundleError(f"sanitization caused duplicate key {sanitized_key!r}")
            result[sanitized_key] = sanitize_structure_with_key(child, sanitizer, stats, sanitized_key)
        return dict(result)
    if isinstance(value, list):
        return [sanitize_structure(child, sanitizer, stats) for child in value]
    if isinstance(value, str):
        return sanitizer.sanitize(value, stats)
    return value


def sanitize_structure_with_key(value: Any, sanitizer: TextSanitizer, stats: SanitizeStats, key: str) -> Any:
    if isinstance(value, str):
        return sanitizer.sanitize(value, stats, key=key)
    return sanitize_structure(value, sanitizer, stats)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
    except (TypeError, ValueError) as exc:
        raise BundleError(f"could not serialize sanitized JSON: {exc}") from exc


def prepare_json(blob: SourceBlob, sanitizer: TextSanitizer, stats: SanitizeStats, audit: FieldAudit) -> bytes:
    text = decode_text(blob.data, blob.spec.role)
    value = parse_json(text, blob.spec.role)
    scan_structured_credentials(value, blob.spec.role)
    if blob.spec.input_type == PER_QUERY_INPUT:
        value = filter_per_query_json(value, audit)
    else:
        reject_aggregate_fields(value, blob.spec.role)
    sanitized = sanitize_structure(value, sanitizer, stats)
    output = canonical_json_bytes(sanitized)
    sanitizer.assert_public(output.decode("ascii"))
    return output


def prepare_jsonl(blob: SourceBlob, sanitizer: TextSanitizer, stats: SanitizeStats, audit: FieldAudit) -> bytes:
    text = decode_text(blob.data, blob.spec.role)
    rows: List[bytes] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        location = f"{blob.spec.role} line {line_number}"
        value = parse_json(line, location)
        if not isinstance(value, Mapping):
            raise BundleError(f"JSONL row {line_number} for role {blob.spec.role!r} is not an object")
        scan_structured_credentials(value, location)
        if blob.spec.input_type == PER_QUERY_INPUT:
            value = filter_per_query_record(value, audit)
            if not value:
                raise BundleError(
                    f"JSONL row {line_number} for role {blob.spec.role!r} has no public evidence fields"
                )
        else:
            reject_aggregate_fields(value, location)
        sanitized = sanitize_structure(value, sanitizer, stats)
        rows.append(
            json.dumps(
                sanitized,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        )
    if not rows:
        raise BundleError(f"JSONL artifact for role {blob.spec.role!r} has no records")
    output = b"\n".join(rows) + b"\n"
    sanitizer.assert_public(output.decode("ascii"))
    return output



def parse_csv(text: str, role: str) -> Tuple[List[str], List[List[str]]]:
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise BundleError(f"invalid CSV for role {role!r}: {exc}") from exc
    if not rows:
        raise BundleError(f"CSV artifact for role {role!r} is empty")
    header = rows[0]
    if not header or any(not cell for cell in header):
        raise BundleError(f"CSV artifact for role {role!r} has an empty header")
    if len(set(header)) != len(header):
        raise BundleError(f"CSV artifact for role {role!r} has duplicate headers")
    width = len(header)
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != width:
            raise BundleError(f"CSV row {row_number} for role {role!r} has {len(row)} cells; expected {width}")
    return header, rows[1:]


def prepare_csv(blob: SourceBlob, sanitizer: TextSanitizer, stats: SanitizeStats, audit: FieldAudit) -> bytes:
    text = decode_text(blob.data, blob.spec.role)
    reject_credentials_in_text(text, blob.spec.role)
    header, rows = parse_csv(text, blob.spec.role)
    for column, key in enumerate(header):
        normalized = normalize_field_name(key)
        values = [row[column] for row in rows]
        sample_value = next((value for value in values if value), "")
        if is_credential_key(key, sample_value):
            raise BundleError(f"credential-like CSV header {key!r} found in role {blob.spec.role!r}")
        if blob.spec.input_type == AGGREGATE_INPUT:
            category = aggregate_field_category(normalized)
            if category is not None:
                raise BundleError(
                    f"forbidden aggregate {category} field {key!r} found in {blob.spec.role}"
                )
        for value in values:
            reject_credentials_in_text(value, f"{blob.spec.role}.{key}")

    kept_indexes = list(range(len(header)))
    if blob.spec.input_type == PER_QUERY_INPUT:
        audit.applied = True
        kept_indexes = []
        for index, key in enumerate(header):
            normalized = normalize_field_name(key)
            values = [row[index] for row in rows]
            keep = field_is_per_query_scalar_allowed(normalized) and all(
                string_is_content_free(value, normalized) for value in values
            )
            if keep:
                kept_indexes.append(index)
                audit.kept.add(key)
            else:
                audit.dropped.add(key)
        if not kept_indexes:
            raise BundleError(f"per-query CSV for role {blob.spec.role!r} has no public evidence columns")

    sanitized_header = [sanitizer.sanitize(header[index], stats) for index in kept_indexes]
    if len(set(sanitized_header)) != len(sanitized_header):
        raise BundleError(f"sanitization caused duplicate CSV headers for role {blob.spec.role!r}")
    sanitized_rows = [
        [sanitizer.sanitize(row[index], stats, key=header[index]) for index in kept_indexes]
        for row in rows
    ]
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(sanitized_header)
    writer.writerows(sanitized_rows)
    encoded = output.getvalue().encode("utf-8")
    sanitizer.assert_public(encoded.decode("utf-8"))
    return encoded



def prepare_markdown(blob: SourceBlob, sanitizer: TextSanitizer, stats: SanitizeStats) -> bytes:
    text = decode_text(blob.data, blob.spec.role)
    reject_credentials_in_text(text, blob.spec.role)
    sanitized = sanitizer.sanitize(text, stats)
    if not sanitized.endswith("\n"):
        sanitized += "\n"
    return sanitized.encode("utf-8")


def validate_svg(text: str, role: str) -> None:
    if "<!DOCTYPE" in text.upper():
        raise BundleError(f"SVG for role {role!r} contains a forbidden DOCTYPE")
    if re.search(r"<\?xml-stylesheet\b", text, re.IGNORECASE):
        raise BundleError(f"SVG for role {role!r} contains a forbidden stylesheet instruction")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise BundleError(f"invalid SVG for role {role!r}: {exc}") from exc
    if root.tag.rsplit("}", 1)[-1].lower() != "svg":
        raise BundleError(f"artifact for role {role!r} is not an SVG document")

    forbidden_tags = {
        "animate",
        "animatemotion",
        "animatetransform",
        "audio",
        "discard",
        "embed",
        "feimage",
        "foreignobject",
        "iframe",
        "image",
        "object",
        "script",
        "set",
        "style",
        "video",
    }
    for element in root.iter():
        local_tag = element.tag.rsplit("}", 1)[-1].lower()
        if local_tag in forbidden_tags:
            raise BundleError(f"SVG for role {role!r} contains forbidden element <{local_tag}>")
        for attribute, value in element.attrib.items():
            local_attribute = attribute.rsplit("}", 1)[-1].lower()
            lowered = value.strip().lower()
            if local_attribute.startswith("on"):
                raise BundleError(
                    f"SVG for role {role!r} contains event-handler attribute {local_attribute!r}"
                )
            if local_attribute == "style":
                raise BundleError(f"SVG for role {role!r} contains a forbidden style attribute")
            if local_attribute in {"href", "src"} and lowered and not re.fullmatch(
                r"#[^\s]+", lowered
            ):
                raise BundleError(f"SVG for role {role!r} contains a non-fragment reference")
            if re.search(r"(?:[a-z][a-z0-9+.-]*:|//)", lowered):
                raise BundleError(f"SVG for role {role!r} contains an external or active URL")

            url_references = re.findall(r"url\(\s*([^)]*?)\s*\)", lowered)
            if "url(" in lowered and not url_references:
                raise BundleError(f"SVG for role {role!r} contains a malformed URL reference")
            for reference in url_references:
                unquoted = reference.strip().strip("'\"")
                if not re.fullmatch(r"#[^\s]+", unquoted):
                    raise BundleError(f"SVG for role {role!r} contains an external URL")


def prepare_svg(blob: SourceBlob, sanitizer: TextSanitizer, stats: SanitizeStats) -> bytes:
    text = decode_text(blob.data, blob.spec.role)
    reject_credentials_in_text(text, blob.spec.role)
    validate_svg(text, blob.spec.role)
    sanitized = sanitizer.sanitize(text, stats)
    validate_svg(sanitized, blob.spec.role)
    if not sanitized.endswith("\n"):
        sanitized += "\n"
    return sanitized.encode("utf-8")


def prepare_artifact(blob: SourceBlob, sanitizer: TextSanitizer) -> PreparedArtifact:
    stats = SanitizeStats()
    audit = FieldAudit()
    suffix = blob.resolved_path.suffix.lower()
    if suffix == ".json":
        output = prepare_json(blob, sanitizer, stats, audit)
    elif suffix == ".jsonl":
        output = prepare_jsonl(blob, sanitizer, stats, audit)
    elif suffix == ".csv":
        output = prepare_csv(blob, sanitizer, stats, audit)
    elif suffix == ".md":
        output = prepare_markdown(blob, sanitizer, stats)
    elif suffix == ".svg":
        output = prepare_svg(blob, sanitizer, stats)
    else:  # The extension was already checked before reading.
        raise BundleError(f"unsupported artifact extension {suffix!r}")

    output_path = f"artifacts/{blob.spec.role}{suffix}"
    output_sha = sha256_bytes(output)
    entry = {
        "field_policy": {
            "applied": audit.applied,
            "dropped_fields": sorted(audit.dropped),
            "kept_fields": sorted(audit.kept),
        },
        "format": suffix.lstrip("."),
        "input_type": blob.spec.input_type,
        "output_path": output_path,
        "role": blob.spec.role,
        "sanitization": {
            "epfl_value_replacements": stats.epfl_value_replacements,
            "private_path_replacements": stats.private_path_replacements,
            "registry_image_replacements": stats.registry_image_replacements,
            "user_replacements": stats.user_replacements,
        },
        "sanitized_output_sha256": output_sha,
        "sanitized_output_size_bytes": len(output),
        "source_sha256": blob.sha256,
        "source_size_bytes": len(blob.data),
    }
    return PreparedArtifact(
        role=blob.spec.role,
        suffix=suffix,
        output_path=output_path,
        data=output,
        manifest_entry=entry,
    )


def render_bundle_readme(
    artifacts: Sequence[PreparedArtifact],
    pdf_fingerprints: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
) -> bytes:
    lines = [
        "# Public Evidence Bundle",
        "",
        "This directory contains only explicitly selected, compact public evidence artifacts.",
        "Aggregate structured inputs passed fail-closed field validation; explicit per-query inputs were reduced to scalar allowlisted fields.",
        "PDF and checkpoint bytes are not included. Their entries are full-source SHA-256 fingerprints and byte sizes only.",
        "",
        "## Artifacts",
        "",
        "| Role | File | Source SHA-256 | Sanitized SHA-256 |",
        "| --- | --- | --- | --- |",
    ]
    if artifacts:
        for artifact in artifacts:
            entry = artifact.manifest_entry
            lines.append(
                f"| {artifact.role} | `{artifact.output_path}` | `{entry['source_sha256']}` | "
                f"`{entry['sanitized_output_sha256']}` |"
            )
    else:
        lines.append("| _none_ | | | |")

    lines.extend(
        [
            "",
            "## PDFs (fingerprints only)",
            "",
            "| Role | Source SHA-256 | Size (bytes) |",
            "| --- | --- | ---: |",
        ]
    )
    if pdf_fingerprints:
        for pdf in pdf_fingerprints:
            lines.append(f"| {pdf['role']} | `{pdf['source_sha256']}` | {pdf['source_size_bytes']} |")
    else:
        lines.append("| _none_ | | |")

    lines.extend(
        [
            "",
            "## Checkpoints (fingerprints only)",
            "",
            "| Role | Source SHA-256 | Size (bytes) |",
            "| --- | --- | ---: |",
        ]
    )
    if checkpoints:
        for checkpoint in checkpoints:
            lines.append(
                f"| {checkpoint['role']} | `{checkpoint['source_sha256']}` | {checkpoint['source_size_bytes']} |"
            )
    else:
        lines.append("| _none_ | | |")

    lines.extend(
        [
            "",
            "## Verify",
            "",
            "Run from this directory:",
            "",
            "```bash",
            "sha256sum -c SHA256SUMS",
            "```",
            "",
            "`manifest.json` records source and sanitized-output hashes, fingerprint-only PDFs and checkpoints, sanitization counts, and kept/dropped per-query fields.",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")



def write_exclusive(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise BundleError(f"refusing to overwrite unexpected output file: {path}") from exc


def build_bundle(
    inputs: Sequence[SourceSpec],
    checkpoints: Sequence[SourceSpec],
    output_dir: Path,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> Mapping[str, int]:
    if max_file_bytes <= 0:
        raise BundleError("--max-file-bytes must be positive")
    if not inputs and not checkpoints:
        raise BundleError("at least one --input, --per-query-input, or --checkpoint is required")
    if any(spec.input_type not in {AGGREGATE_INPUT, PER_QUERY_INPUT} for spec in inputs):
        raise BundleError("copied inputs must be aggregate or per-query inputs")
    if any(spec.input_type != CHECKPOINT_INPUT for spec in checkpoints):
        raise BundleError("checkpoint inputs must use the checkpoint input type")
    reject_duplicate_roles(inputs, checkpoints)

    copied_specs: List[SourceSpec] = []
    pdf_specs: List[SourceSpec] = []
    for spec in sorted(inputs, key=lambda item: item.role):
        suffix = spec.path.suffix.lower()
        if suffix not in ALLOWED_INPUT_SUFFIXES:
            allowed = ", ".join(sorted(ALLOWED_INPUT_SUFFIXES))
            raise BundleError(
                f"unsupported artifact extension {suffix or '<none>'!r} for role {spec.role!r}; allowed: {allowed}"
            )
        if suffix == ".pdf":
            if spec.input_type != AGGREGATE_INPUT:
                raise BundleError(
                    f"PDF for role {spec.role!r} must use --input and is fingerprint-only"
                )
            pdf_specs.append(spec)
        else:
            copied_specs.append(spec)

    blobs = [read_artifact(spec, max_file_bytes) for spec in copied_specs]
    pdf_fingerprints = [fingerprint_pdf(spec) for spec in pdf_specs]
    checkpoint_entries = [
        fingerprint_checkpoint(spec) for spec in sorted(checkpoints, key=lambda item: item.role)
    ]

    discovery_texts = [decode_text(blob.data, blob.spec.role) for blob in blobs]
    all_source_paths = (
        [blob.resolved_path for blob in blobs]
        + [spec.path for spec in pdf_specs]
        + [spec.path for spec in checkpoints]
    )
    usernames = discover_usernames(discovery_texts, all_source_paths)
    repo_root = Path(__file__).resolve().parents[1]
    sanitizer = TextSanitizer(repo_root, usernames)
    prepared = [prepare_artifact(blob, sanitizer) for blob in blobs]

    manifest = {
        "artifacts": [artifact.manifest_entry for artifact in prepared],
        "checkpoints": checkpoint_entries,
        "max_file_bytes": max_file_bytes,
        "pdf_fingerprints": pdf_fingerprints,
        "schema_version": 2,
        "tool": "build_public_evidence_bundle",
    }
    manifest_bytes = canonical_json_bytes(manifest)
    readme_bytes = render_bundle_readme(prepared, pdf_fingerprints, checkpoint_entries)

    output_dir = output_dir.expanduser()
    try:
        output_dir.mkdir(mode=0o755, exist_ok=False)
    except FileExistsError as exc:
        raise BundleError(f"output directory already exists; refusing to overwrite: {output_dir}") from exc
    except OSError as exc:
        raise BundleError(f"cannot create output directory {output_dir}: {exc}") from exc

    try:
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(mode=0o755, exist_ok=False)
        emitted: Dict[str, bytes] = {
            "README.md": readme_bytes,
            "manifest.json": manifest_bytes,
        }
        for artifact in prepared:
            emitted[artifact.output_path] = artifact.data
        for relative_path in sorted(emitted):
            write_exclusive(output_dir / relative_path, emitted[relative_path])
        sums = "".join(
            f"{sha256_bytes(emitted[relative_path])}  {relative_path}\n"
            for relative_path in sorted(emitted)
        ).encode("ascii")
        write_exclusive(output_dir / "SHA256SUMS", sums)
    except Exception:
        shutil.rmtree(output_dir)
        raise

    return {
        "artifacts": len(prepared),
        "checkpoints": len(checkpoint_entries),
        "files": len(emitted) + 1,
        "pdf_fingerprints": len(pdf_fingerprints),
    }



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="aggregate JSON/JSONL/CSV/MD/SVG to sanitize and copy, or PDF to fingerprint only; repeatable",
    )
    parser.add_argument(
        "--per-query-input",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="per-query JSON/JSONL/CSV to reduce to strict scalar allowlisted fields; repeatable",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="checkpoint to fingerprint without copying; repeatable",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory; must not already exist")
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help=f"maximum copied artifact size in bytes (default: {DEFAULT_MAX_FILE_BYTES}, 25 MiB)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        inputs = [parse_role_path(value, "--input") for value in args.input]
        inputs.extend(
            parse_role_path(value, "--per-query-input") for value in args.per_query_input
        )
        checkpoints = [parse_role_path(value, "--checkpoint") for value in args.checkpoint]
        summary = build_bundle(inputs, checkpoints, args.output_dir, args.max_file_bytes)
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
