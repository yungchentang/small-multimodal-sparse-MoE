"""Materialize provenance-rich multimodal train/dev/eval manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from PIL import Image

try:
    from scripts.development_split_provenance import (
        DATA_CONTRACT_ARTIFACTS,
        DATA_CONTRACT_POLICY,
        DATA_CONTRACT_SNAPSHOT_SEMANTICS,
        speech_partition_commitments,
    )
except ImportError:  # Direct script execution.
    from development_split_provenance import (  # type: ignore[no-redef]
        DATA_CONTRACT_ARTIFACTS,
        DATA_CONTRACT_POLICY,
        DATA_CONTRACT_SNAPSHOT_SEMANTICS,
        speech_partition_commitments,
    )


class SplitError(RuntimeError):
    """Raised when a source manifest cannot satisfy the split contract."""


SOURCE_SNAPSHOT_POLICY = "single_read_bytes_rows_sha256_commitments_v1"


def parse_jsonl_snapshot(path: Path, payload: bytes) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SplitError(f"{path} is not valid UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            raise SplitError(f"{path}:{line_number} is blank")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SplitError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def builder_provenance() -> Dict[str, Any]:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    commit = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "-C",
            str(repo_root),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise SplitError("builder source commit is not an exact lowercase Git SHA")
    relative_path = script_path.relative_to(repo_root).as_posix()
    tracked_bytes = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "-C",
            str(repo_root),
            "show",
            f"{commit}:{relative_path}",
        ],
        check=True,
        capture_output=True,
    ).stdout
    builder_sha = sha256_file(script_path)
    tracked_sha = hashlib.sha256(tracked_bytes).hexdigest()
    if tracked_sha != builder_sha:
        raise SplitError(
            "builder script bytes do not match the declared source commit"
        )
    return {
        "path": str(script_path),
        "sha256": builder_sha,
        "source_commit_sha": commit,
        "source_matches_commit": True,
        "command": "python scripts/materialize_eval_splits.py",
    }


def pixel_sha256(path: Path) -> str:
    """Hash decoded RGB pixels so copied files remain one source-image group."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        header = f"RGB:{rgb.width}x{rgb.height}\n".encode("ascii")
        return hashlib.sha256(header + rgb.tobytes()).hexdigest()


def resolve_media_path(value: Any, data_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise SplitError(f"image_path must be a non-empty string, got {value!r}")
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [data_dir / raw, Path.cwd() / raw]
    resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        raise SplitError(f"missing image file: {value}")
    return resolved


def annotate_image_content(
    rows: Sequence[Dict[str, Any]], data_dir: Path
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Attach auditable file/pixel hashes used for group-disjoint splitting."""
    output: List[Dict[str, Any]] = []
    path_cache: Dict[Path, tuple[str, str]] = {}
    group_sizes: Dict[str, int] = {}
    for index, source in enumerate(rows):
        path = resolve_media_path(source.get("image_path"), data_dir)
        hashes = path_cache.get(path)
        if hashes is None:
            hashes = (sha256_file(path), pixel_sha256(path))
            path_cache[path] = hashes
        media_hash, content_hash = hashes
        row = dict(source)
        for field, observed in (
            ("media_sha256", media_hash),
            ("content_sha256", content_hash),
            ("resized_content_sha256", content_hash),
        ):
            declared = row.get(field)
            if declared not in (None, "") and str(declared) != observed:
                raise SplitError(
                    f"image row {index} has mismatched {field}: {declared!r} != {observed}"
                )
            row[field] = observed
        row["image_path"] = str(path)
        output.append(row)
        group_sizes[content_hash] = group_sizes.get(content_hash, 0) + 1
    return output, group_sizes


def exact_group_partition(
    group_sizes: Dict[str, int], dev_count: int, eval_count: int, seed: int
) -> tuple[set[str], set[str]]:
    """Find deterministic, exact dev/eval row totals without splitting image groups."""
    group_ids = sorted(
        group_sizes,
        key=lambda group_id: hashlib.sha256(
            f"{int(seed)}\0{group_id}".encode("utf-8")
        ).digest(),
    )
    nodes: List[tuple[int, str, str] | None] = [None]
    states: Dict[tuple[int, int], int] = {(0, 0): 0}
    target = (int(dev_count), int(eval_count))
    for group_id in group_ids:
        size = int(group_sizes[group_id])
        additions: Dict[tuple[int, int], int] = {}
        for (dev_rows, eval_rows), parent_node in list(states.items()):
            for assignment, state in (
                ("dev", (dev_rows + size, eval_rows)),
                ("eval", (dev_rows, eval_rows + size)),
            ):
                if state[0] > target[0] or state[1] > target[1]:
                    continue
                if state in states or state in additions:
                    continue
                nodes.append((parent_node, group_id, assignment))
                additions[state] = len(nodes) - 1
        states.update(additions)
        if target in states:
            break
    node_id = states.get(target)
    if node_id is None:
        sizes = sorted(group_sizes.values())
        raise SplitError(
            "cannot form exact image dev/eval row targets from complete content groups: "
            f"dev={dev_count}, eval={eval_count}, group_sizes={sizes}"
        )
    selected: Dict[str, set[str]] = {"dev": set(), "eval": set()}
    while node_id:
        node = nodes[node_id]
        assert node is not None
        parent_node, group_id, assignment = node
        selected[assignment].add(group_id)
        node_id = parent_node
    if selected["dev"] & selected["eval"]:
        raise AssertionError("internal image content-group overlap")
    return selected["dev"], selected["eval"]


def split_image_rows(
    rows: Sequence[Dict[str, Any]],
    data_dir: Path,
    dev_count: int,
    eval_count: int,
    seed: int,
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    annotated, group_sizes = annotate_image_content(rows, data_dir)
    dev_groups, eval_groups = exact_group_partition(
        group_sizes, dev_count, eval_count, seed
    )
    raw: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "eval": []}
    for row in annotated:
        group_id = str(row["content_sha256"])
        split = "dev" if group_id in dev_groups else "eval" if group_id in eval_groups else "train"
        raw[split].append(row)
    observed_counts = {split: len(values) for split, values in raw.items()}
    expected_counts = {
        "train": len(rows) - int(dev_count) - int(eval_count),
        "dev": int(dev_count),
        "eval": int(eval_count),
    }
    if observed_counts != expected_counts:
        raise AssertionError(
            f"internal image split count mismatch: {observed_counts} != {expected_counts}"
        )
    group_sets = {
        split: {str(row["content_sha256"]) for row in values}
        for split, values in raw.items()
    }
    overlaps = {
        f"{left}_{right}": sorted(group_sets[left] & group_sets[right])
        for left, right in (("train", "dev"), ("train", "eval"), ("dev", "eval"))
    }
    if any(overlaps.values()):
        raise AssertionError(f"internal image content-group overlap: {overlaps}")
    summary = {
        "policy": "seeded_exact_image_content_disjoint_v1",
        "seed": int(seed),
        "group_key": "content_sha256_of_decoded_resized_rgb_pixels",
        "row_counts": observed_counts,
        "group_counts": {split: len(values) for split, values in group_sets.items()},
        "pairwise_group_overlaps": overlaps,
        "pairwise_group_overlap_count": sum(len(values) for values in overlaps.values()),
        "source_group_count": len(group_sizes),
    }
    return (
        {split: annotate(values, "image", split) for split, values in raw.items()},
        summary,
    )


def annotate(rows: Sequence[Dict[str, Any]], modality: str, split: str) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        row["eval_split_name"] = f"{modality}_{split}"
        row["eval_split_index"] = index
        output.append(row)
    return output


def split_rows(
    rows: Sequence[Dict[str, Any]],
    modality: str,
    dev_count: int,
    eval_count: int,
) -> tuple[Dict[str, List[Dict[str, Any]]], str]:
    partitions = {str(row.get("partition", "")) for row in rows}
    has_named_partitions = partitions <= {"train", "dev", "eval"} and {"dev", "eval"} <= partitions
    if has_named_partitions:
        raw = {
            split: [row for row in rows if row.get("partition") == split]
            for split in ("train", "dev", "eval")
        }
        if len(raw["dev"]) != dev_count or len(raw["eval"]) != eval_count:
            raise SplitError(
                f"{modality} named partition counts are "
                f"dev={len(raw['dev'])}, eval={len(raw['eval'])}; "
                f"expected dev={dev_count}, eval={eval_count}"
            )
        policy = "explicit_source_partition"
    else:
        needed = dev_count + eval_count
        if needed <= 0 or len(rows) <= needed:
            raise SplitError(f"{modality} needs more than {needed} rows, found {len(rows)}")
        raw = {
            "train": list(rows[:-needed]),
            "dev": list(rows[-needed:-eval_count] if eval_count else rows[-needed:]),
            "eval": list(rows[-eval_count:] if eval_count else []),
        }
        policy = "deterministic_heldout_tail"
    return (
        {split: annotate(values, modality, split) for split, values in raw.items()},
        policy,
    )


def group_overlap(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
    groups = {
        split: {
            f"{row.get('source_dataset', row.get('source', ''))}|{row.get('speaker_id', '')}"
            for row in rows
            if row.get("speaker_id") not in (None, "")
        }
        for split, rows in splits.items()
    }
    return {
        "train_dev": sorted(groups["train"] & groups["dev"]),
        "train_eval": sorted(groups["train"] & groups["eval"]),
        "dev_eval": sorted(groups["dev"] & groups["eval"]),
    }


def materialize(
    data_dir: Path,
    output_dir: Path,
    dev_count: int,
    eval_count: int,
    image_split_seed: int = 42,
) -> Dict[str, Any]:
    if output_dir.exists():
        raise SplitError(f"refusing existing output directory: {output_dir}")
    raw_data_dir = Path(data_dir)
    if raw_data_dir.is_symlink():
        raise SplitError("data directory must not be a symlink")
    data_dir = raw_data_dir.resolve(strict=True)
    if not data_dir.is_dir():
        raise SplitError("data directory must be canonical")
    artifact_paths = {
        name: data_dir / filename
        for name, (filename, _is_jsonl) in DATA_CONTRACT_ARTIFACTS.items()
    }
    artifact_snapshots: Dict[str, Dict[str, Any]] = {}
    artifact_rows: Dict[str, List[Dict[str, Any]]] = {}
    for name, path in artifact_paths.items():
        filename, is_jsonl = DATA_CONTRACT_ARTIFACTS[name]
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise SplitError(
                f"{name} must be canonical data_dir/{filename} regular non-symlink file"
            )
        payload = path.read_bytes()
        snapshot: Dict[str, Any] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        if is_jsonl:
            snapshot_rows = parse_jsonl_snapshot(path, payload)
            snapshot["rows"] = len(snapshot_rows)
            artifact_rows[name] = snapshot_rows
        else:
            try:
                dataset_manifest = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SplitError(f"{path} is not valid UTF-8 JSON") from exc
            if not isinstance(dataset_manifest, dict):
                raise SplitError(f"{path} root must be a JSON object")
        artifact_snapshots[name] = snapshot
    rows = {
        "image": artifact_rows["image_captions"],
        "speech": artifact_rows["speech_transcripts"],
    }
    source_paths = {
        "image": artifact_paths["image_captions"],
        "speech": artifact_paths["speech_transcripts"],
    }
    source_snapshots = {
        "image": artifact_snapshots["image_captions"],
        "speech": artifact_snapshots["speech_transcripts"],
    }
    speech_source_path = source_paths["speech"]
    image, image_partition = split_image_rows(
        rows["image"], data_dir, dev_count, eval_count, image_split_seed
    )
    speech, speech_policy = split_rows(rows["speech"], "speech", dev_count, eval_count)
    overlaps = group_overlap(speech)
    if any(overlaps.values()):
        raise SplitError(f"speech speaker overlap across partitions: {overlaps}")
    try:
        speech_commitments = speech_partition_commitments(
            rows["speech"],
            source_path=str(speech_source_path),
            source_sha256=source_snapshots["speech"]["sha256"],
        )
    except ValueError as exc:
        raise SplitError(
            f"invalid speech source partition derivation: {exc}"
        ) from exc

    outputs: Dict[str, Path] = {}
    for modality, splits in (("image", image), ("speech", speech)):
        for split, split_rows_value in splits.items():
            path = output_dir / f"{modality}_{split}.jsonl"
            write_jsonl(path, split_rows_value)
            outputs[f"{modality}_{split}"] = path

    file_records: Dict[str, Dict[str, Any]] = {}
    for name, path in outputs.items():
        modality, split = name.split("_", 1)
        record: Dict[str, Any] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": len(image[split] if modality == "image" else speech[split]),
        }
        if modality == "speech":
            record["source_partition_membership_sha256"] = (
                speech_commitments["partitions"][split]["membership_root_sha256"]
            )
        file_records[name] = record

    manifest: Dict[str, Any] = {
        "schema_version": 3,
        "builder": builder_provenance(),
        "data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "real_subset": True,
        "synthetic_evidence_used": False,
        "sealed_data_used": False,
        "dev_count": dev_count,
        "eval_count": eval_count,
        "split_policy": {
            "image": image_partition["policy"],
            "speech": speech_policy,
        },
        "counts": {
            f"{modality}_{split}": len(values)
            for modality, splits in (("image", image), ("speech", speech))
            for split, values in splits.items()
        },
        "speech_group_key": ["source_dataset", "speaker_id"],
        "speech_group_overlap": overlaps,
        "speech_partition_commitments": speech_commitments,
        "image_content_partition": image_partition,
        "source_snapshot_policy": SOURCE_SNAPSHOT_POLICY,
        "data_contract": {
            "policy": DATA_CONTRACT_POLICY,
            "data_dir": str(data_dir),
            "snapshot_semantics": DATA_CONTRACT_SNAPSHOT_SEMANTICS,
            "artifacts": {
                name: {
                    "path": str(artifact_paths[name]),
                    **artifact_snapshots[name],
                }
                for name in DATA_CONTRACT_ARTIFACTS
            },
        },
        "source_files": {
            name: {
                "path": str(path.resolve()),
                "sha256": source_snapshots[name]["sha256"],
                "bytes": source_snapshots[name]["bytes"],
                "rows": source_snapshots[name]["rows"],
                "snapshot_semantics": (
                    "rows_and_sha256_derived_from_the_same_single_bytes_read"
                ),
            }
            for name, path in source_paths.items()
        },
        "files": file_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/real_subset_clean_260708b")
    parser.add_argument("--output-dir", default="outputs/development_eval_v2")
    parser.add_argument("--dev-count", type=int, default=125)
    parser.add_argument("--eval-count", type=int, default=125)
    parser.add_argument("--image-split-seed", type=int, default=42)
    args = parser.parse_args()
    materialize(
        Path(args.data_dir),
        Path(args.output_dir),
        args.dev_count,
        args.eval_count,
        args.image_split_seed,
    )


if __name__ == "__main__":
    main()
