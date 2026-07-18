"""Materialize deterministic sealed image and speech evaluation sets.

Evaluation content is written only to the output data files. Console output and
the tracked-friendly index contain hashes, counts, and source identifiers only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hf_sources import load_dataset_ref, resolve_dataset


IMAGE_SOURCES = (
    {
        "dataset": "Multimodal-Fatima/COCO_captions_validation",
        "config": None,
        "split": "validation",
        "revision": "bfa149029bb1e2975cb0b9bea8ad948db9e9ddb2",
        "partition": "COCO 2014 validation-derived data",
    },
)
SPEECH_SOURCE = {
    "dataset": "openslr/librispeech_asr",
    "config": "clean",
    "split": "test",
    "revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    "partition": "LibriSpeech test-clean",
}
MANIFEST_NAME = "sealed_eval_manifest.json"
DEFAULT_INDEX_NAME = "sealed_eval_index.jsonl"


def structured_hf_source(source: Mapping[str, Any]) -> Dict[str, Any]:
    ref = resolve_dataset(source["dataset"], source.get("revision"))
    return {
        **ref.as_dict(),
        "config": source.get("config"),
        "split": source["split"],
    }


def hf_source_for_provenance(
    provenance: Mapping[str, Any], profiles: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    key = (provenance.get("dataset"), provenance.get("config"), provenance.get("split"))
    for source in profiles:
        if key == (source["dataset"], source.get("config"), source["split"]):
            return structured_hf_source(source)
    raise ValueError("loaded source is outside the closed Hugging Face source profile")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any, limit: int = 4096) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ").strip()
    return " ".join(text.split())[:limit]


def canonical_strings(values: Iterable[Any]) -> List[str]:
    unique = {normalize_text(value) for value in values}
    unique.discard("")
    return sorted(unique, key=lambda value: (value.casefold(), value))


def extract_captions(row: Mapping[str, Any]) -> List[str]:
    values: List[Any] = []
    for key in ("sentences_raw", "captions", "caption", "sentence"):
        value = row.get(key)
        if value is None:
            continue
        if not isinstance(value, (list, tuple)):
            value = [value]
        for item in value:
            if isinstance(item, Mapping):
                item = item.get("raw") or item.get("caption") or item.get("text")
            values.append(item)
    return canonical_strings(values)


def make_provenance(
    dataset: str,
    config: str | None,
    split: str,
    source_ids: Iterable[Any],
    partition: str | None = None,
) -> Dict[str, Any]:
    if not dataset or not split:
        raise ValueError("dataset and split are required for provenance")
    provenance: Dict[str, Any] = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "source_ids": canonical_strings(source_ids),
    }
    if partition:
        provenance["partition"] = partition
    return provenance


def validate_source_policy(provenance: Mapping[str, Any], modality: str) -> None:
    dataset = provenance.get("dataset")
    config = provenance.get("config")
    split = str(provenance.get("split", ""))
    if "train" in split.casefold():
        raise ValueError(f"sealed {modality} source cannot use a training split")
    if modality == "image":
        allowed = {(item["dataset"], item["config"], item["split"]) for item in IMAGE_SOURCES}
        if (dataset, config, split) not in allowed:
            raise ValueError("image source is not an approved COCO validation-style split")
    elif modality == "speech":
        expected = (SPEECH_SOURCE["dataset"], SPEECH_SOURCE["config"], SPEECH_SOURCE["split"])
        if (dataset, config, split) != expected:
            raise ValueError("speech source must be openslr/librispeech_asr clean test")
    else:
        raise ValueError(f"unknown modality: {modality}")


def _source_ids(record: Mapping[str, Any]) -> List[str]:
    values: List[Any] = []
    direct = record.get("source_ids", [])
    values.extend(direct if isinstance(direct, (list, tuple, set)) else [direct])
    source = record.get("source")
    if isinstance(source, Mapping):
        nested = source.get("source_ids", [])
        values.extend(nested if isinstance(nested, (list, tuple, set)) else [nested])
    for key in ("source_id", "utterance_id"):
        if record.get(key) is not None:
            values.append(record[key])
    return canonical_strings(values)


def group_image_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Group caption records by canonical image-content SHA-256."""
    groups: MutableMapping[str, Dict[str, Any]] = {}
    for position, record in enumerate(records):
        content_hash = normalize_text(record.get("content_sha256"))
        if len(content_hash) != 64:
            raise ValueError("each image record requires a SHA-256 content hash")
        captions = extract_captions(record)
        if not captions and record.get("captions"):
            captions = canonical_strings(record["captions"])
        if not captions:
            raise ValueError("each image record requires at least one caption")
        group = groups.setdefault(
            content_hash,
            {
                "content_sha256": content_hash,
                "captions": set(),
                "source_ids": set(),
                "source_row_count": 0,
                "first_source_index": int(record.get("source_index", position)),
            },
        )
        group["captions"].update(captions)
        group["source_ids"].update(_source_ids(record))
        group["source_row_count"] += 1
        group["first_source_index"] = min(
            group["first_source_index"], int(record.get("source_index", position))
        )

    output: List[Dict[str, Any]] = []
    for group in groups.values():
        captions = canonical_strings(group["captions"])
        output.append(
            {
                "content_sha256": group["content_sha256"],
                "canonical_caption": captions[0],
                "captions": captions,
                "caption_count": len(captions),
                "source_ids": canonical_strings(group["source_ids"]),
                "source_row_count": group["source_row_count"],
                "first_source_index": group["first_source_index"],
            }
        )
    return sorted(output, key=lambda item: (item["first_source_index"], item["content_sha256"]))


def _speaker_sort_key(value: Any) -> Tuple[int, Any]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def balanced_speaker_round_robin(
    rows: Iterable[Mapping[str, Any]],
    limit: int,
    excluded_source_ids: Iterable[str] = (),
    candidate_is_excluded: Callable[[Mapping[str, Any]], bool] | None = None,
    selection_summary: MutableMapping[str, int] | None = None,
) -> List[Dict[str, Any]]:
    """Select rows in deterministic rounds, at most one utterance per speaker per round."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    speakers: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    seen_ids = set()
    for row in rows:
        if row.get("speaker_id") is None or row.get("source_id") is None:
            raise ValueError("speech rows require speaker_id and source_id")
        source_id = str(row["source_id"])
        if source_id in seen_ids:
            raise ValueError("speech source IDs must be unique")
        seen_ids.add(source_id)
        speakers[row["speaker_id"]].append(dict(row))
    for speaker_rows in speakers.values():
        speaker_rows.sort(key=lambda row: (str(row["source_id"]), int(row.get("source_index", 0))))

    excluded_ids = set(excluded_source_ids)
    selected: List[Dict[str, Any]] = []
    ordered_speakers = sorted(speakers, key=_speaker_sort_key)
    speaker_positions = {speaker_id: 0 for speaker_id in ordered_speakers}
    excluded_count = 0
    while len(selected) < limit:
        added = False
        for speaker_id in ordered_speakers:
            speaker_rows = speakers[speaker_id]
            position = speaker_positions[speaker_id]
            while position < len(speaker_rows):
                candidate = speaker_rows[position]
                position += 1
                source_id = str(candidate["source_id"])
                candidate_ids = {source_id, f"utterance:{source_id}"}
                excluded = bool(candidate_ids & excluded_ids)
                if not excluded and candidate_is_excluded is not None:
                    excluded = candidate_is_excluded(candidate)
                if excluded:
                    excluded_count += 1
                    continue
                selected.append(candidate)
                added = True
                break
            speaker_positions[speaker_id] = position
            if len(selected) == limit:
                break
        if not added:
            break
    if selection_summary is not None:
        selection_summary["excluded_candidates"] = excluded_count
    return selected


def overlap_report(
    candidates: Iterable[Mapping[str, Any]], references: Iterable[Mapping[str, Any]]
) -> Dict[str, Any]:
    candidate_rows = list(candidates)
    reference_rows = list(references)

    def hashes(rows: Sequence[Mapping[str, Any]]) -> set[str]:
        output = set()
        for row in rows:
            for key in ("media_sha256", "content_sha256", "resized_content_sha256"):
                value = normalize_text(row.get(key))
                if len(value) == 64:
                    output.add(value)
        return output

    candidate_hashes = hashes(candidate_rows)
    reference_hashes = hashes(reference_rows)
    candidate_ids = {value for row in candidate_rows for value in _source_ids(row)}
    reference_ids = {value for row in reference_rows for value in _source_ids(row)}
    hash_overlap = candidate_hashes & reference_hashes
    source_overlap = candidate_ids & reference_ids
    return {
        "candidate_count": len(candidate_rows),
        "reference_count": len(reference_rows),
        "hash_overlap_count": len(hash_overlap),
        "source_id_overlap_count": len(source_overlap),
        "reference_check_performed": bool(reference_rows),
        "passed": not hash_overlap and not source_overlap,
    }


def assert_no_overlap(report: Mapping[str, Any], label: str) -> None:
    if not report.get("passed"):
        raise ValueError(
            f"{label} overlap assertion failed: "
            f"hashes={report.get('hash_overlap_count', 0)} "
            f"source_ids={report.get('source_id_overlap_count', 0)}"
        )


def _load_real_subset_helpers() -> Any:
    helper_path = Path(__file__).with_name("build_real_subset.py")
    spec = importlib.util.spec_from_file_location("_sealed_eval_real_subset_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load build_real_subset.py helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_image_hash(image: Any) -> str:
    rgb = image.convert("RGB")
    header = f"RGB:{rgb.width}x{rgb.height}\n".encode("ascii")
    return sha256_bytes(header + rgb.tobytes())


def _image_output_hashes(image: Any) -> Tuple[str, str]:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image).BICUBIC
    resized = image.convert("RGB").resize((224, 224), resampling)
    with io.BytesIO() as buffer:
        resized.save(buffer, format="PNG", optimize=False, compress_level=9)
        media_hash = sha256_bytes(buffer.getvalue())
    return _canonical_image_hash(resized), media_hash


def _image_source_ids(
    row: Mapping[str, Any], source_index: int, dataset: str, split: str
) -> List[str]:
    values = [f"hf_row:{dataset}:{split}:{row.get('id', source_index)}"]
    for field, prefix in (
        ("cocoid", "coco_image"),
        ("imgid", "image"),
        ("filename", "filename"),
        ("filepath", "filepath"),
    ):
        if row.get(field) is not None:
            values.append(f"{prefix}:{row[field]}")
    return canonical_strings(values)


def _load_image_groups(
    sample_count: int,
    excluded_hashes: Iterable[str] = (),
    excluded_source_ids: Iterable[str] = (),
    selection_summary: MutableMapping[str, int] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    helpers = _load_real_subset_helpers()
    excluded_hash_set = set(excluded_hashes)
    excluded_source_id_set = set(excluded_source_ids)
    has_exclusions = bool(excluded_hash_set or excluded_source_id_set)
    failures = []
    for source in IMAGE_SOURCES:
        try:
            dataset = load_dataset_ref(
                source["dataset"],
                source["config"],
                revision=source["revision"],
                split=source["split"],
            )
            records: List[Dict[str, Any]] = []
            output_hashes: Dict[str, Tuple[str, str]] = {}
            images: Dict[str, Any] = {}
            selected_hashes = set()
            for source_index in range(len(dataset)):
                raw = dict(dataset[source_index])
                captions = extract_captions(raw)
                if not captions:
                    continue
                image = helpers.extract_image(raw)
                if image is None:
                    continue
                content_hash = _canonical_image_hash(image)
                if not has_exclusions and content_hash not in selected_hashes:
                    if len(selected_hashes) >= sample_count:
                        continue
                    selected_hashes.add(content_hash)
                    images[content_hash] = image.convert("RGB").copy()
                elif (
                    excluded_hash_set and content_hash not in output_hashes
                ):
                    output_hashes[content_hash] = _image_output_hashes(image)
                records.append(
                    {
                        "content_sha256": content_hash,
                        "captions": captions,
                        "source_ids": _image_source_ids(
                            raw, source_index, source["dataset"], source["split"]
                        ),
                        "source_index": source_index,
                    }
                )
            groups = group_image_records(records)
            excluded_count = 0
            if has_exclusions:
                selected_groups = []
                for group in groups:
                    candidate_hashes = {group["content_sha256"]}
                    candidate_hashes.update(output_hashes.get(group["content_sha256"], ()))
                    if (
                        candidate_hashes & excluded_hash_set
                        or set(group["source_ids"]) & excluded_source_id_set
                    ):
                        excluded_count += 1
                        continue
                    selected_groups.append(group)
                    if len(selected_groups) == sample_count:
                        break
            else:
                selected_groups = groups[:sample_count]
            if selection_summary is not None:
                selection_summary["excluded_candidates"] = excluded_count
            if len(selected_groups) < sample_count:
                if has_exclusions:
                    failures.append(
                        f"{source['dataset']}:{source['split']} "
                        f"disjoint_count={len(selected_groups)} excluded={excluded_count}"
                    )
                else:
                    failures.append(
                        f"{source['dataset']}:{source['split']} count={len(groups)}"
                    )
                continue
            if has_exclusions:
                for group in selected_groups:
                    raw = dict(dataset[group["first_source_index"]])
                    image = helpers.extract_image(raw)
                    if image is None or _canonical_image_hash(image) != group["content_sha256"]:
                        raise RuntimeError("selected image source row could not be reconstructed")
                    images[group["content_sha256"]] = image.convert("RGB").copy()
            provenance = make_provenance(
                source["dataset"], source["config"], source["split"], [], source["partition"]
            )
            validate_source_policy(provenance, "image")
            return selected_groups, images, provenance
        except Exception as exc:
            failures.append(f"{source['dataset']}:{source['split']} error={type(exc).__name__}")
    raise RuntimeError("no approved image source produced enough unique images; " + "; ".join(failures))


def _write_images(
    output_dir: Path, groups: Sequence[Mapping[str, Any]], images: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from PIL import Image

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    preprocess = {
        "mode": "RGB",
        "resize": [224, 224],
        "interpolation": "bicubic",
        "format": "PNG",
    }
    rows = []
    index_rows = []
    for position, group in enumerate(groups):
        row_id = f"image-{position:04d}"
        relative_path = Path("images") / f"{row_id}.png"
        image = images[group["content_sha256"]].resize((224, 224), resampling)
        resized_content_hash = _canonical_image_hash(image)
        path = output_dir / relative_path
        image.save(path, format="PNG", optimize=False, compress_level=9)
        media_hash = sha256_file(path)
        source = dict(group["source"])
        source["source_ids"] = list(group["source_ids"])
        row = {
            "id": row_id,
            "task": "image_captioning",
            "image_path": str(relative_path),
            "caption": group["canonical_caption"],
            "captions": list(group["captions"]),
            "content_sha256": group["content_sha256"],
            "resized_content_sha256": resized_content_hash,
            "media_sha256": media_hash,
            "source": source,
            "preprocessing": preprocess,
            "group": {
                "source_row_count": group["source_row_count"],
                "caption_count": group["caption_count"],
            },
        }
        rows.append(row)
        index_rows.append({key: value for key, value in row.items() if key not in {"caption", "captions", "image_path"}})
    return rows, index_rows


def _speech_metadata(dataset: Any) -> List[Dict[str, Any]]:
    columns = [
        name
        for name in ("audio", "file", "text", "transcript")
        if name in dataset.column_names
    ]
    metadata = dataset.remove_columns(columns) if columns else dataset
    rows = []
    for source_index in range(len(metadata)):
        row = dict(metadata[source_index])
        source_id = row.get("id")
        if source_id is None or row.get("speaker_id") is None or row.get("chapter_id") is None:
            continue
        rows.append(
            {
                "source_id": str(source_id),
                "speaker_id": row["speaker_id"],
                "chapter_id": row["chapter_id"],
                "source_index": source_index,
            }
        )
    return rows


def _resample_and_fix_length(audio: Any, source_rate: int, duration_seconds: float) -> Any:
    import numpy as np

    if source_rate <= 0 or duration_seconds <= 0:
        raise ValueError("source rate and duration must be positive")
    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if signal.size == 0:
        signal = np.zeros(1, dtype=np.float32)
    target_rate = 16000
    if source_rate != target_rate:
        output_length = max(1, int(round(signal.size * target_rate / source_rate)))
        old_positions = np.arange(signal.size, dtype=np.float64)
        new_positions = np.arange(output_length, dtype=np.float64) * source_rate / target_rate
        signal = np.interp(new_positions, old_positions, signal).astype(np.float32)
    target_length = int(round(target_rate * duration_seconds))
    if signal.size < target_length:
        signal = np.pad(signal, (0, target_length - signal.size))
    else:
        signal = signal[:target_length]
    return np.clip(signal, -1.0, 1.0).astype(np.float32)


def _write_speech(
    output_dir: Path,
    sample_count: int,
    duration_seconds: float,
    excluded_media_hashes: Iterable[str] = (),
    excluded_source_ids: Iterable[str] = (),
    selection_summary: MutableMapping[str, int] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    import soundfile as sf
    from datasets import Audio

    helpers = _load_real_subset_helpers()
    dataset = load_dataset_ref(
        SPEECH_SOURCE["dataset"],
        SPEECH_SOURCE["config"],
        revision=SPEECH_SOURCE["revision"],
        split=SPEECH_SOURCE["split"],
    )
    try:
        dataset = dataset.cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    excluded_media_hash_set = set(excluded_media_hashes)
    excluded_source_id_set = set(excluded_source_ids)
    materialized: Dict[int, Dict[str, Any]] = {}

    def materialize(selected_row: Mapping[str, Any]) -> Dict[str, Any]:
        source_index = int(selected_row["source_index"])
        if source_index in materialized:
            return materialized[source_index]
        raw = dict(dataset[source_index])
        audio_pair = helpers.extract_audio(raw)
        transcript = normalize_text(raw.get("text") or raw.get("transcript"), limit=4096)
        if audio_pair is None or not transcript:
            raise RuntimeError(f"speech source row {selected_row['source_id']} is incomplete")
        audio, source_rate = audio_pair
        processed = _resample_and_fix_length(audio, source_rate, duration_seconds)
        candidate = {
            "audio": audio,
            "processed": processed,
            "source_rate": source_rate,
            "transcript": transcript,
        }
        if excluded_media_hash_set:
            with io.BytesIO() as buffer:
                sf.write(buffer, processed, 16000, format="WAV", subtype="PCM_16")
                wav_bytes = buffer.getvalue()
            candidate["wav_bytes"] = wav_bytes
            candidate["media_sha256"] = sha256_bytes(wav_bytes)
        materialized[source_index] = candidate
        return candidate

    def media_is_excluded(candidate: Mapping[str, Any]) -> bool:
        return materialize(candidate)["media_sha256"] in excluded_media_hash_set

    selected = balanced_speaker_round_robin(
        _speech_metadata(dataset),
        sample_count,
        excluded_source_id_set,
        media_is_excluded if excluded_media_hash_set else None,
        selection_summary,
    )
    if len(selected) < sample_count:
        qualifier = " disjoint" if excluded_media_hash_set or excluded_source_id_set else ""
        raise RuntimeError(
            f"speech source produced {len(selected)}{qualifier} rows; expected {sample_count}"
        )

    provenance = make_provenance(
        SPEECH_SOURCE["dataset"],
        SPEECH_SOURCE["config"],
        SPEECH_SOURCE["split"],
        [],
        SPEECH_SOURCE["partition"],
    )
    validate_source_policy(provenance, "speech")
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    preprocess = {
        "channels": 1,
        "sample_rate_hz": 16000,
        "duration_seconds": duration_seconds,
        "num_samples": int(round(16000 * duration_seconds)),
        "length_policy": "truncate_or_zero_pad",
        "resampling": "deterministic_linear_interpolation",
        "format": "WAV_PCM_16",
    }
    rows = []
    index_rows = []
    for position, selected_row in enumerate(selected):
        candidate = materialize(selected_row)
        audio = candidate["audio"]
        source_rate = candidate["source_rate"]
        row_id = f"speech-{position:04d}"
        relative_path = Path("audio") / f"{row_id}.wav"
        path = output_dir / relative_path
        if excluded_media_hash_set:
            path.write_bytes(candidate["wav_bytes"])
        else:
            sf.write(str(path), candidate["processed"], 16000, format="WAV", subtype="PCM_16")
        media_hash = sha256_file(path)
        source_ids = [f"utterance:{selected_row['source_id']}"]
        source = dict(provenance)
        source["source_ids"] = source_ids
        row = {
            "id": row_id,
            "task": "speech_transcription",
            "audio_path": str(relative_path),
            "transcript": candidate["transcript"],
            "speaker_id": selected_row["speaker_id"],
            "chapter_id": selected_row["chapter_id"],
            "utterance_id": selected_row["source_id"],
            "source_id": selected_row["source_id"],
            "media_sha256": media_hash,
            "source": source,
            "preprocessing": {
                **preprocess,
                "source_sample_rate_hz": int(source_rate),
                "source_num_samples": int(len(audio)),
            },
        }
        rows.append(row)
        index_rows.append({key: value for key, value in row.items() if key not in {"transcript", "audio_path"}})
    return rows, index_rows, provenance


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_reference_records(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    records = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        try:
            value = json.loads(text)
            if isinstance(value, list):
                records.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                nested_found = False
                for key in ("rows", "items", "index"):
                    items = value.get(key)
                    if isinstance(items, list):
                        records.extend(item for item in items if isinstance(item, dict))
                        nested_found = True
                if not nested_found and any(
                    key in value for key in ("media_sha256", "content_sha256", "source_ids")
                ):
                    records.append(value)
        except json.JSONDecodeError:
            for line in text.splitlines():
                if line.strip():
                    item = json.loads(line)
                    if isinstance(item, dict):
                        records.append(item)
    return records


def _read_exclusion_indexes(paths: Sequence[Path]) -> Dict[str, Any]:
    references: List[Dict[str, Any]] = []
    index_metadata = []
    for path in paths:
        rows = _read_reference_records([path])
        references.extend(rows)
        index_metadata.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
        )

    image_reference = [
        row
        for row in references
        if (row.get("modality") or row.get("task"))
        in (None, "image", "image_captioning")
    ]
    speech_reference = [
        row
        for row in references
        if (row.get("modality") or row.get("task"))
        in (None, "speech", "speech_transcription")
    ]

    def hashes(rows: Iterable[Mapping[str, Any]], keys: Sequence[str]) -> set[str]:
        return {
            value
            for row in rows
            for key in keys
            if len(value := normalize_text(row.get(key))) == 64
        }

    return {
        "reference_indexes": index_metadata,
        "image_reference": image_reference,
        "speech_reference": speech_reference,
        "image_hashes": hashes(
            image_reference,
            ("content_sha256", "resized_content_sha256", "media_sha256"),
        ),
        "image_source_ids": {value for row in image_reference for value in _source_ids(row)},
        "speech_media_hashes": hashes(speech_reference, ("media_sha256",)),
        "speech_source_ids": {value for row in speech_reference for value in _source_ids(row)},
    }


def _attach_image_source(groups: List[Dict[str, Any]], provenance: Mapping[str, Any]) -> None:
    for group in groups:
        source = dict(provenance)
        source["source_ids"] = list(group["source_ids"])
        group["source"] = source


def _managed_paths(output_dir: Path, index_name: str) -> List[Path]:
    return [
        output_dir / "images",
        output_dir / "audio",
        output_dir / "image_test.jsonl",
        output_dir / "speech_test.jsonl",
        output_dir / MANIFEST_NAME,
        output_dir / index_name,
    ]


def _validate_index_name(index_name: str) -> None:
    path = Path(index_name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError("--index-file must be a filename within --output-dir")
    reserved = {"images", "audio", "image_test.jsonl", "speech_test.jsonl", MANIFEST_NAME}
    if index_name in reserved:
        raise ValueError("--index-file conflicts with a fixed sealed evaluation output")


def build_sealed_eval(args: argparse.Namespace) -> Dict[str, Any]:
    if args.image_samples <= 0 or args.speech_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.speech_duration_seconds <= 0:
        raise ValueError("speech duration must be positive")
    _validate_index_name(args.index_file)
    output_dir = Path(args.output_dir)
    existing = [path for path in _managed_paths(output_dir, args.index_file) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            f"refusing to overwrite {len(existing)} sealed evaluation outputs; pass --force"
        )

    exclude_paths = [Path(path) for path in args.exclude_index]
    exclusions = _read_exclusion_indexes(exclude_paths)
    image_selection = {"excluded_candidates": 0}
    speech_selection = {"excluded_candidates": 0}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".sealed-eval-", dir=output_dir.parent))
    try:
        image_groups, images, image_source = _load_image_groups(
            args.image_samples,
            exclusions["image_hashes"],
            exclusions["image_source_ids"],
            image_selection,
        )
        _attach_image_source(image_groups, image_source)
        image_rows, image_index = _write_images(stage, image_groups, images)
        speech_rows, speech_index, speech_source = _write_speech(
            stage,
            args.speech_samples,
            args.speech_duration_seconds,
            exclusions["speech_media_hashes"],
            exclusions["speech_source_ids"],
            speech_selection,
        )

        image_reference = exclusions["image_reference"]
        speech_reference = exclusions["speech_reference"]
        image_overlap = overlap_report(image_index, image_reference)
        speech_overlap = overlap_report(speech_index, speech_reference)
        assert_no_overlap(image_overlap, "image")
        assert_no_overlap(speech_overlap, "speech")

        for row in image_index:
            row["modality"] = "image"
        for row in speech_index:
            row["modality"] = "speech"
        all_index = image_index + speech_index
        _write_jsonl(stage / "image_test.jsonl", image_rows)
        _write_jsonl(stage / "speech_test.jsonl", speech_rows)
        _write_jsonl(stage / args.index_file, all_index)

        manifest = {
            "schema_version": 1,
            "sealed": True,
            "sources": {"image": image_source, "speech": speech_source},
            "hf_sources": {
                "image": hf_source_for_provenance(image_source, IMAGE_SOURCES),
                "speech": hf_source_for_provenance(speech_source, (SPEECH_SOURCE,)),
            },
            "counts": {
                "image_rows": len(image_rows),
                "speech_rows": len(speech_rows),
                "total_rows": len(all_index),
            },
            "exclusions": {
                "active": bool(exclude_paths),
                "reference_indexes": exclusions["reference_indexes"],
                "excluded_candidate_counts": {
                    "image": image_selection["excluded_candidates"],
                    "speech": speech_selection["excluded_candidates"],
                },
                "selection_policy": {
                    "name": "deterministic_exclude_then_select",
                    "image": (
                        "group_unique_content_in_source_order_skip_excluded_select_first_n"
                    ),
                    "speech": "balanced_speaker_round_robin_skip_excluded_select_first_n",
                },
            },
            "group_counts": {
                "image_source_rows": sum(group["source_row_count"] for group in image_groups),
                "unique_image_content_hashes": len(image_groups),
                "image_captions": sum(group["caption_count"] for group in image_groups),
                "duplicate_image_rows_grouped": sum(
                    max(0, group["source_row_count"] - 1) for group in image_groups
                ),
                "speech_speakers": len({row["speaker_id"] for row in speech_rows}),
            },
            "preprocessing": {
                "image": image_index[0]["preprocessing"],
                "speech": {
                    key: value
                    for key, value in speech_index[0]["preprocessing"].items()
                    if not key.startswith("source_")
                },
            },
            "overlap_assertions": {
                "source_partition_policy": {
                    "image_non_training_validation_style": True,
                    "speech_is_librispeech_test_clean": True,
                    "passed": True,
                },
                "image": image_overlap,
                "speech": speech_overlap,
                "reference_indexes": [str(path) for path in args.exclude_index],
            },
            "files": {
                "image_test.jsonl": {"sha256": sha256_file(stage / "image_test.jsonl"), "rows": len(image_rows)},
                "speech_test.jsonl": {"sha256": sha256_file(stage / "speech_test.jsonl"), "rows": len(speech_rows)},
                args.index_file: {"sha256": sha256_file(stage / args.index_file), "rows": len(all_index)},
            },
        }
        _write_json(stage / MANIFEST_NAME, manifest)

        output_dir.mkdir(parents=True, exist_ok=True)
        for staged_path in _managed_paths(stage, args.index_file):
            target = output_dir / staged_path.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            staged_path.replace(target)
        return manifest
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/sealed_eval")
    parser.add_argument("--image-samples", type=int, default=250)
    parser.add_argument("--speech-samples", type=int, default=250)
    parser.add_argument("--speech-duration-seconds", type=float, default=6.0)
    parser.add_argument("--index-file", default=DEFAULT_INDEX_NAME)
    parser.add_argument(
        "--exclude-index",
        action="append",
        default=[],
        help="Tracked-friendly JSON/JSONL index whose hashes and source IDs must not overlap.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_sealed_eval(args)
    manifest_path = Path(args.output_dir) / MANIFEST_NAME
    print(
        json.dumps(
            {
                "counts": manifest["counts"],
                "manifest_sha256": sha256_file(manifest_path),
                "overlap_passed": all(
                    manifest["overlap_assertions"][name]["passed"]
                    for name in ("image", "speech")
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
