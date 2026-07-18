"""Build real dataset manifests for ACDL Project 18.

The generated data comes from public benchmark datasets:
C4, CodeParrot, LogiQA, GSM8K, AGIEval SAT, COCO captions, and LibriSpeech.
Images are resized to 224x224 RGB PNGs. Audio is loaded, resampled to 16 kHz,
and padded/truncated to a fixed duration before being saved as WAV.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hf_sources import load_dataset_ref, load_pretrained, resolve_dataset, resolve_model


@dataclass
class Candidate:
    repo: str
    config: Optional[str]
    split: str
    streaming: bool = False
    revision: Optional[str] = None

    def __post_init__(self) -> None:
        self.revision = resolve_dataset(self.repo, self.revision).revision

    @property
    def label(self) -> str:
        cfg = f"/{self.config}" if self.config else ""
        return f"{self.repo}{cfg}:{self.split}"

    def hf_source(self) -> Dict[str, Any]:
        return {
            "repo_id": self.repo,
            "revision": self.revision,
            "config": self.config,
            "split": self.split,
            "streaming": self.streaming,
        }


REAL_SOURCE_PROFILES = {
    "text": (
        Candidate("NeelNanda/c4-10k", None, "train", False),
        Candidate("allenai/c4", "en", "train", True),
    ),
    "code": (Candidate("codeparrot/codeparrot-clean-valid", None, "train", False),),
    "reasoning": (Candidate("hails/agieval-logiqa-en", None, "test", False),),
    "math": (Candidate("openai/gsm8k", "main", "train", False),),
    "education": (
        Candidate("hails/agieval-sat-en", None, "test", False),
        Candidate("hails/agieval-sat-math", None, "test", False),
        Candidate("hails/agieval-lsat-ar", None, "test", False),
        Candidate("hails/agieval-lsat-lr", None, "test", False),
    ),
}

IMAGE_SOURCE_PROFILE = (Candidate("jxie/coco_captions", None, "train", True),)

SPEECH_SOURCE_PROFILE = (
    Candidate("openslr/librispeech_asr", "clean", "train.360", True),
    Candidate("openslr/librispeech_asr", "clean", "train.100", True),
    Candidate("openslr/librispeech_asr", "clean", "validation", True),
    Candidate("openslr/librispeech_asr", "clean", "test", True),
)

SPEECH_PARTITION_POLICY = "seeded_exact_heldout_speaker_disjoint_v1"
SPEECH_PARTITIONS = ("train", "dev", "eval")


class SpeechProvenanceError(RuntimeError):
    """Raised when a speech row cannot be tied to an auditable source ID."""


def structured_hf_sources(
    candidates: Iterable[Candidate], source_labels: str
) -> List[Dict[str, Any]]:
    by_label = {candidate.label: candidate for candidate in candidates}
    labels = [label for label in source_labels.split(";") if label]
    if not labels:
        raise ValueError("dataset source labels cannot be empty")
    output = []
    seen = set()
    for label in labels:
        if label not in by_label:
            raise ValueError(f"dataset source {label!r} is outside the closed source profile")
        if label not in seen:
            output.append(by_label[label].hf_source())
            seen.add(label)
    return output


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def take_rows(dataset: Any, max_rows: int) -> Iterable[Dict[str, Any]]:
    if hasattr(dataset, "select"):
        return dataset.select(range(min(max_rows, len(dataset))))
    if hasattr(dataset, "take"):
        return dataset.take(max_rows)
    return itertools.islice(dataset, max_rows)


def iter_dataset_rows(candidate: Candidate, max_rows: int) -> Iterable[Dict[str, Any]]:
    kwargs: Dict[str, Any] = {"split": candidate.split, "streaming": candidate.streaming}
    dataset = load_dataset_ref(
        candidate.repo,
        candidate.config,
        revision=candidate.revision,
        **kwargs,
    )
    if "librispeech" in candidate.repo.lower():
        try:
            from datasets import Audio

            dataset = dataset.cast_column("audio", Audio(decode=False))
        except Exception:
            pass
    return take_rows(dataset, max_rows)


def try_load_dataset(candidate: Candidate, max_rows: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    return [dict(row) for row in iter_dataset_rows(candidate, max_rows)], None


def first_present(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def clean_text(value: Any, limit: int = 4096) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value if v is not None)
    text = str(value).replace("\x00", " ").strip()
    return " ".join(text.split())[:limit]


def normalize_choices(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [clean_text(value[k]) for k in sorted(value) if clean_text(value[k])]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_text(v) for v in value if clean_text(v)]
    return [clean_text(value)]


def answer_index_from_row(row: Dict[str, Any], choices: List[str]) -> Optional[int]:
    raw = first_present(row, ["answer", "label", "gold", "target", "correct", "correct_option", "answer_idx"])
    if raw is None:
        return None
    if isinstance(raw, (int, np.integer)):
        idx = int(raw)
        return idx if 0 <= idx < len(choices) else None
    text = clean_text(raw)
    if not text:
        return None
    if text.isdigit():
        idx = int(text)
        return idx if 0 <= idx < len(choices) else None
    letter = text.strip().upper()[:1]
    if letter in "ABCDE":
        idx = ord(letter) - ord("A")
        return idx if 0 <= idx < len(choices) else None
    for idx, choice in enumerate(choices):
        if text.lower() == choice.lower() or text.lower() in choice.lower():
            return idx
    return None


def final_number(text: str) -> Optional[float]:
    import re

    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def numeric_distractors(answer: str) -> List[str]:
    number = final_number(answer)
    if number is None:
        return []
    vals = [number + 1, number - 1, number * 2 if number != 0 else 2]
    out = []
    for value in vals:
        if abs(value - round(value)) < 1e-6:
            out.append(str(int(round(value))))
        else:
            out.append(f"{value:.2f}")
    return out


def build_text_like_rows(task: str, candidates: List[Candidate], max_samples: int, mapper: Callable[[Dict[str, Any], int, str], Optional[Dict[str, Any]]], errors: List[Dict[str, str]], allow_short: bool = False) -> Tuple[List[Dict[str, Any]], str]:
    best_rows: List[Dict[str, Any]] = []
    best_label = ""
    collected: List[Dict[str, Any]] = []
    used_labels: List[str] = []
    seen_texts = set()
    for candidate in candidates:
        try:
            raw_rows = iter_dataset_rows(candidate, max_samples * 4)
            rows = []
            for raw in raw_rows:
                raw = dict(raw)
                mapped = mapper(raw, len(rows), candidate.label)
                if mapped is not None:
                    mapped["task"] = task
                    rows.append(mapped)
                    text_key = str(mapped.get("text", ""))[:1000]
                    if text_key and text_key not in seen_texts and len(collected) < max_samples:
                        copy = dict(mapped)
                        copy["id"] = len(collected)
                        collected.append(copy)
                        seen_texts.add(text_key)
                if len(rows) >= max_samples and len(collected) >= max_samples:
                    break
            if len(rows) > len(best_rows):
                best_rows = rows
                best_label = candidate.label
            if rows:
                used_labels.append(candidate.label)
            if len(collected) >= max_samples:
                return collected, ";".join(used_labels)
        except Exception as exc:  # Continue within the closed, pinned source profile.
            errors.append({"task": task, "candidate": candidate.label, "error": repr(exc), "traceback": traceback.format_exc(limit=4)})
    if allow_short and (collected or best_rows):
        return (collected or best_rows), (";".join(used_labels) or best_label)
    raise RuntimeError(f"Could not load enough real dataset rows for task={task}; got={len(collected)} expected={max_samples}; best_single={len(best_rows)} from {best_label}; errors={errors[-3:]}")


def map_plain_text(row: Dict[str, Any], idx: int, source: str) -> Optional[Dict[str, Any]]:
    text = clean_text(first_present(row, ["text", "content", "document", "article", "code"]), limit=4096)
    if len(text) < 20:
        return None
    return {"id": idx, "source": source, "text": text, "prompt": text[:400], "target": text[400:800] or text[:400]}


def map_code(row: Dict[str, Any], idx: int, source: str) -> Optional[Dict[str, Any]]:
    code = clean_text(first_present(row, ["content", "code", "text"]), limit=8192)
    if len(code) < 20:
        return None
    return {"id": idx, "source": source, "text": code, "prompt": code[:500], "target": code[500:1000] or code[:500]}


def map_logiqa(row: Dict[str, Any], idx: int, source: str) -> Optional[Dict[str, Any]]:
    context = clean_text(first_present(row, ["context", "passage", "article", "paragraph", "input"]), limit=4096)
    question = clean_text(first_present(row, ["question", "query", "instruction" , "prompt"]), limit=1024)
    choices = normalize_choices(first_present(row, ["options", "choices", "endings"] ))
    if not choices:
        choices = [clean_text(row.get(k)) for k in ["A", "B", "C", "D", "option_a", "option_b", "option_c", "option_d"] if clean_text(row.get(k))]
    answer_idx = answer_index_from_row(row, choices)
    if not question or not choices or answer_idx is None:
        return None
    prompt = f"{context}\nQuestion: {question}\n" if context else f"Question: {question}\n"
    prompt += "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)) + "\nAnswer:"
    return {"id": idx, "source": source, "text": f"{prompt} {choices[answer_idx]}", "prompt": prompt, "target": choices[answer_idx], "choices": choices, "answer_index": answer_idx}


def map_gsm8k(row: Dict[str, Any], idx: int, source: str) -> Optional[Dict[str, Any]]:
    question = clean_text(first_present(row, ["question", "problem", "prompt"]), limit=2048)
    answer = clean_text(first_present(row, ["answer", "solution", "target"]), limit=4096)
    if not question or not answer:
        return None
    target = answer.split("####")[-1].strip() if "####" in answer else answer
    choices = [target] + numeric_distractors(target)
    seen = []
    for choice in choices:
        if choice and choice not in seen:
            seen.append(choice)
    prompt = f"Solve the math problem.\nQuestion: {question}\nAnswer:"
    return {"id": idx, "source": source, "text": f"{prompt} {answer}", "prompt": prompt, "target": target, "choices": seen, "answer_index": 0, "full_answer": answer}


def map_sat(row: Dict[str, Any], idx: int, source: str) -> Optional[Dict[str, Any]]:
    question = clean_text(first_present(row, ["question", "query", "prompt", "input"] ), limit=2048)
    passage = clean_text(first_present(row, ["passage", "context"] ), limit=4096)
    choices = normalize_choices(first_present(row, ["choices", "options"] ))
    answer_idx = answer_index_from_row(row, choices)
    if not question or not choices or answer_idx is None:
        return None
    prompt = f"{passage}\nQuestion: {question}\n" if passage else f"Question: {question}\n"
    prompt += "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)) + "\nAnswer:"
    return {"id": idx, "source": source, "text": f"{prompt} {choices[answer_idx]}", "prompt": prompt, "target": choices[answer_idx], "choices": choices, "answer_index": answer_idx}


def load_tokenizer(model_name: str, revision: Optional[str] = None):
    from transformers import AutoTokenizer

    model_ref = resolve_model(model_name, revision)
    tokenizer = load_pretrained(AutoTokenizer, model_ref)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def pack_token_blocks(
    rows: List[Dict[str, Any]],
    tokenizer,
    task: str,
    block_size: int,
    train_blocks: int,
    eval_blocks: int,
    allow_short: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    token_stream: List[int] = []
    source = ";".join(sorted({str(row.get("source", "")) for row in rows if row.get("source")})) if rows else ""
    for row in rows:
        text = str(row.get("text") or (str(row.get("prompt", "")) + " " + str(row.get("target", "")))).strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if ids:
            token_stream.extend(int(x) for x in ids)
            if eos_id is not None:
                token_stream.append(int(eos_id))
    total_needed = int(train_blocks + eval_blocks)
    blocks: List[Dict[str, Any]] = []
    for start in range(0, len(token_stream) - block_size + 1, block_size):
        ids = token_stream[start:start + block_size]
        blocks.append({"id": len(blocks), "task": task, "source": source, "input_ids": ids, "length": len(ids)})
        if len(blocks) >= total_needed:
            break
    if not allow_short and len(blocks) < total_needed:
        raise RuntimeError(f"Task {task} produced {len(blocks)} packed blocks, expected {total_needed}")
    return blocks[:train_blocks], blocks[train_blocks:train_blocks + eval_blocks]


def write_text_blocks(
    out_dir: Path,
    rows_by_task: Dict[str, List[Dict[str, Any]]],
    tokenizer_model: str,
    block_size: int,
    train_block_targets: Dict[str, int],
    eval_blocks_per_task: int,
    allow_short: bool,
    tokenizer_revision: Optional[str] = None,
) -> Dict[str, Any]:
    tokenizer_ref = resolve_model(tokenizer_model, tokenizer_revision)
    tokenizer = load_tokenizer(tokenizer_ref.repo_id, tokenizer_ref.revision)
    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    per_task: Dict[str, Dict[str, int]] = {}
    for task, rows in rows_by_task.items():
        train, eval_ = pack_token_blocks(
            rows,
            tokenizer,
            task,
            block_size,
            int(train_block_targets.get(task, 0)),
            int(eval_blocks_per_task),
            allow_short,
        )
        for row in train:
            row["split"] = "train"
            row["id"] = len(train_rows)
            train_rows.append(row)
        for row in eval_:
            row["split"] = "eval"
            row["id"] = len(eval_rows)
            eval_rows.append(row)
        per_task[task] = {"train_blocks": len(train), "eval_blocks": len(eval_)}
    write_jsonl(out_dir / "text_blocks_train.jsonl", train_rows)
    write_jsonl(out_dir / "text_blocks_eval.jsonl", eval_rows)
    return {
        "tokenizer": tokenizer_ref.repo_id,
        "tokenizer_revision": tokenizer_ref.revision,
        "block_size": block_size,
        "train_blocks": len(train_rows),
        "eval_blocks": len(eval_rows),
        "per_task": per_task,
    }


def extract_caption(row: Dict[str, Any]) -> str:
    value = first_present(row, ["caption", "captions", "sentences_raw", "sentences", "text", "description"])
    if isinstance(value, dict):
        value = first_present(value, ["raw", "text", "caption"])
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            if isinstance(item, dict):
                item = first_present(item, ["raw", "text", "caption"])
            item = clean_text(item)
            if item:
                flattened.append(item)
        value = flattened[0] if flattened else ""
    return clean_text(value, limit=300)


def caption_ascii_letter_stats(caption: str) -> Tuple[int, int, int, float]:
    letters = [ch for ch in caption if ch.isalpha()]
    ascii_letters = [ch for ch in letters if ch.isascii()]
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", caption)
    ratio = len(ascii_letters) / len(letters) if letters else 0.0
    return len(ascii_letters), len(letters), len(words), ratio


def is_english_caption(caption: str, min_ascii_ratio: float, min_letters: int, min_words: int = 2) -> bool:
    ascii_letters, _letters, words, ratio = caption_ascii_letter_stats(caption)
    return ascii_letters >= min_letters and words >= min_words and ratio >= min_ascii_ratio


def extract_image(row: Dict[str, Any]) -> Optional[Image.Image]:
    value = first_present(row, ["image", "jpg", "png", "file", "img"])
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if "path" in value and value["path"]:
            return Image.open(value["path"])
        if "bytes" in value and value["bytes"]:
            import io
            return Image.open(io.BytesIO(value["bytes"]))
    try:
        return Image.open(value)
    except Exception:
        return None


def build_image_rows(
    out_dir: Path,
    max_samples: int,
    errors: List[Dict[str, str]],
    caption_min_ascii_ratio: float,
    caption_min_letters: int,
    allow_short: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    candidates = list(IMAGE_SOURCE_PROFILE)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    used_sources: List[str] = []
    seen_captions = set()
    for candidate in candidates:
        try:
            raw_rows = iter_dataset_rows(candidate, max_samples * 4)
            before = len(rows)
            for raw in raw_rows:
                raw = dict(raw)
                caption = extract_caption(raw)
                if caption in seen_captions:
                    continue
                if not caption or not is_english_caption(caption, caption_min_ascii_ratio, caption_min_letters):
                    continue
                image = extract_image(raw)
                if image is None:
                    continue
                image = image.convert("RGB").resize((224, 224), Image.BICUBIC)
                image_path = image_dir / f"coco_{len(rows):05d}.png"
                image.save(image_path)
                rows.append({
                    "id": len(rows),
                    "task": "image",
                    "source": candidate.label,
                    "image_path": str(image_path),
                    "caption": caption,
                    "preprocess": {
                        "resize": [224, 224],
                        "mode": "RGB",
                        "caption_min_ascii_ratio": caption_min_ascii_ratio,
                        "caption_min_letters": caption_min_letters,
                    },
                })
                seen_captions.add(caption)
                if len(rows) >= max_samples:
                    break
            if len(rows) > before:
                used_sources.append(candidate.label)
            if len(rows) >= max_samples:
                return rows, ";".join(used_sources)
        except Exception as exc:
            errors.append({"task": "image", "candidate": candidate.label, "error": repr(exc), "traceback": traceback.format_exc(limit=4)})
    if allow_short and rows:
        return rows, ";".join(used_sources)
    raise RuntimeError(f"Could not load {max_samples} real image-caption rows; got={len(rows)} from {used_sources}; errors={errors[-3:]}")


def extract_audio(row: Dict[str, Any]) -> Optional[Tuple[np.ndarray, int]]:
    value = first_present(row, ["audio", "file", "path"])
    if value is None:
        return None
    if isinstance(value, dict):
        if "array" in value and value["array"] is not None:
            sr = int(value.get("sampling_rate") or value.get("sample_rate") or 16000)
            return np.asarray(value["array"], dtype=np.float32), sr
        if "bytes" in value and value["bytes"]:
            return load_audio_bytes(value["bytes"])
        if "path" in value and value["path"]:
            return load_audio_file(value["path"])
    if isinstance(value, (str, Path)):
        return load_audio_file(value)
    return None


def load_audio_file(path: Any) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


def load_audio_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    import io
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr)


def resample_and_pad(audio: np.ndarray, sr: int, target_sr: int, max_seconds: float) -> np.ndarray:
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        except Exception:
            # Minimal linear interpolation fallback for environments without librosa internals.
            duration = len(audio) / max(sr, 1)
            x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
            x_new = np.linspace(0, duration, num=max(1, int(duration * target_sr)), endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    max_len = int(target_sr * max_seconds)
    if len(audio) > max_len:
        audio = audio[:max_len]
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32)


def extract_transcript(row: Dict[str, Any]) -> str:
    return clean_text(first_present(row, ["text", "transcript", "sentence", "normalized_text"]), limit=300)


def count_words(text: str) -> int:
    return len(text.split())


LIBRISPEECH_ID_RE = re.compile(
    r"^(?P<speaker>\d+)-(?P<chapter>\d+)-(?P<utterance>\d+)(?:\.[A-Za-z0-9]+)?$"
)


def _canonical_numeric_id(value: Any, field: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text.isdigit():
        raise SpeechProvenanceError(f"invalid LibriSpeech {field}: {value!r}")
    return str(int(text))


def _parse_librispeech_id(value: Any) -> Optional[Tuple[str, str, str]]:
    if value is None:
        return None
    match = LIBRISPEECH_ID_RE.fullmatch(Path(str(value)).name)
    if match is None:
        return None
    speaker = str(int(match.group("speaker")))
    chapter = str(int(match.group("chapter")))
    utterance = match.group("utterance")
    return speaker, chapter, f"{speaker}-{chapter}-{utterance}"


def _source_audio_value(row: Dict[str, Any]) -> Any:
    return first_present(row, ["audio", "file", "path"])


def extract_librispeech_provenance(
    row: Dict[str, Any], candidate: Candidate
) -> Dict[str, str]:
    """Resolve source IDs from explicit metadata and/or a canonical audio basename."""
    if "librispeech" not in candidate.repo.lower():
        raise SpeechProvenanceError(
            f"speech provenance parser only supports LibriSpeech, got {candidate.repo!r}"
        )

    speaker_claims: List[Tuple[str, str]] = []
    chapter_claims: List[Tuple[str, str]] = []
    utterance_claims: List[Tuple[str, str]] = []
    explicit_speaker = first_present(row, ["speaker_id", "speaker"])
    explicit_chapter = first_present(row, ["chapter_id", "chapter"])
    if explicit_speaker is not None:
        speaker_claims.append(
            ("metadata.speaker_id", _canonical_numeric_id(explicit_speaker, "speaker_id"))
        )
    if explicit_chapter is not None:
        chapter_claims.append(
            ("metadata.chapter_id", _canonical_numeric_id(explicit_chapter, "chapter_id"))
        )

    for key in ("utterance_id", "id"):
        value = row.get(key)
        parsed = _parse_librispeech_id(value)
        if parsed is not None:
            speaker, chapter, utterance = parsed
            speaker_claims.append((f"metadata.{key}", speaker))
            chapter_claims.append((f"metadata.{key}", chapter))
            utterance_claims.append((f"metadata.{key}", utterance))
        elif key == "utterance_id" and value is not None:
            segment = str(value).strip()
            if segment.isdigit() and explicit_speaker is not None and explicit_chapter is not None:
                speaker = _canonical_numeric_id(explicit_speaker, "speaker_id")
                chapter = _canonical_numeric_id(explicit_chapter, "chapter_id")
                utterance_claims.append(
                    ("metadata.utterance_id", f"{speaker}-{chapter}-{segment}")
                )
            else:
                raise SpeechProvenanceError(
                    f"invalid LibriSpeech utterance_id metadata: {value!r}"
                )

    audio_value = _source_audio_value(row)
    path_values: List[Any] = [row.get("file"), row.get("path")]
    if isinstance(audio_value, dict):
        path_values.append(audio_value.get("path"))
    elif isinstance(audio_value, (str, Path)):
        path_values.append(audio_value)
    for value in path_values:
        parsed = _parse_librispeech_id(value)
        if parsed is None:
            continue
        speaker, chapter, utterance = parsed
        speaker_claims.append(("audio_path", speaker))
        chapter_claims.append(("audio_path", chapter))
        utterance_claims.append(("audio_path", utterance))

    claims = {
        "speaker_id": speaker_claims,
        "chapter_id": chapter_claims,
        "utterance_id": utterance_claims,
    }
    resolved: Dict[str, str] = {}
    for field, field_claims in claims.items():
        if not field_claims:
            raise SpeechProvenanceError(
                f"missing LibriSpeech {field}; require explicit metadata or canonical audio path"
            )
        values = {value for _origin, value in field_claims}
        if len(values) != 1:
            raise SpeechProvenanceError(
                f"conflicting LibriSpeech {field}: {field_claims!r}"
            )
        resolved[field] = values.pop()

    parsed_utterance = _parse_librispeech_id(resolved["utterance_id"])
    assert parsed_utterance is not None
    if parsed_utterance[:2] != (resolved["speaker_id"], resolved["chapter_id"]):
        raise SpeechProvenanceError(
            "LibriSpeech utterance_id does not match speaker_id/chapter_id"
        )
    return {
        "source_dataset": candidate.repo,
        "source_config": candidate.config or "",
        "source_split": candidate.split,
        **resolved,
    }


def _sha256_json(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_audio_fingerprint(row: Dict[str, Any]) -> Dict[str, Any]:
    value = _source_audio_value(row)
    path_value: Any = None
    if isinstance(value, dict):
        path_value = value.get("path")
    elif isinstance(value, (str, Path)):
        path_value = value
    if path_value is None:
        path_value = first_present(row, ["file", "path"])
    if path_value:
        path = Path(str(path_value))
        if path.is_file():
            stat = path.stat()
            record = {
                "path": str(path_value),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            return {
                "algorithm": "sha256",
                "semantics": "path_utf8_size_bytes_mtime_ns_v1_not_audio_content",
                "value": _sha256_json(record),
                **record,
            }
    if isinstance(value, dict) and value.get("bytes") is not None:
        payload = bytes(value["bytes"])
        return {
            "algorithm": "sha256",
            "semantics": "encoded_audio_bytes_v1",
            "value": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    if isinstance(value, dict) and value.get("array") is not None:
        array = np.ascontiguousarray(np.asarray(value["array"]))
        header = json.dumps(
            {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sampling_rate": value.get("sampling_rate") or value.get("sample_rate"),
            },
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(header)
        digest.update(array.view(np.uint8))
        return {
            "algorithm": "sha256",
            "semantics": "decoded_audio_array_dtype_shape_rate_bytes_v1",
            "value": digest.hexdigest(),
            "num_elements": int(array.size),
        }
    raise SpeechProvenanceError(
        "cannot fingerprint source audio; require a readable path, encoded bytes, or decoded array"
    )


def path_stat_fingerprint(path: Path, path_label: Optional[str] = None) -> Dict[str, Any]:
    stat = path.stat()
    record = {
        "path": path_label if path_label is not None else str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return {
        "algorithm": "sha256",
        "semantics": "path_utf8_size_bytes_mtime_ns_v1_not_audio_content",
        "value": _sha256_json(record),
        **record,
    }


def _speech_group_id(row: Dict[str, Any]) -> str:
    return f"{row['source_dataset']}:{row['speaker_id']}"


def partition_speech_rows(
    rows: List[Dict[str, Any]], heldout_rows: int, seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if heldout_rows < 2:
        raise SpeechProvenanceError(
            "speech held-out size must be at least 2 rows to create non-empty dev/eval partitions"
        )
    if heldout_rows >= len(rows):
        raise SpeechProvenanceError(
            f"speech held-out size {heldout_rows} leaves no training rows out of {len(rows)}"
        )

    groups: Dict[str, List[Dict[str, Any]]] = {}
    source_ids = set()
    for index, row in enumerate(rows):
        for field in (
            "source_dataset",
            "source_config",
            "source_split",
            "speaker_id",
            "chapter_id",
            "utterance_id",
            "source_audio_fingerprint",
            "audio_fingerprint",
        ):
            if row.get(field) in (None, ""):
                raise SpeechProvenanceError(f"speech row {index} missing required {field}")
        source_id = (str(row["source_dataset"]), str(row["utterance_id"]))
        if source_id in source_ids:
            raise SpeechProvenanceError(f"duplicate speech source utterance ID: {source_id!r}")
        source_ids.add(source_id)
        groups.setdefault(_speech_group_id(row), []).append(row)
    if len(groups) < 3:
        raise SpeechProvenanceError(
            f"speech split requires at least 3 speaker groups; got {len(groups)}"
        )

    group_ids = sorted(groups)
    group_ids.sort(
        key=lambda group_id: hashlib.sha256(
            f"{seed}\0{group_id}".encode("utf-8")
        ).digest()
    )
    states: Dict[Tuple[int, int], Tuple[str, ...]] = {(0, 0): ()}
    for group_id in group_ids:
        size = len(groups[group_id])
        additions: Dict[Tuple[int, int], Tuple[str, ...]] = {}
        for (total, count_class), selected in list(states.items()):
            new_total = total + size
            if new_total > heldout_rows:
                continue
            new_count_class = min(2, count_class + 1)
            key = (new_total, new_count_class)
            if key not in states and key not in additions:
                additions[key] = selected + (group_id,)
        states.update(additions)
    heldout_group_tuple = states.get((heldout_rows, 2))
    if heldout_group_tuple is None:
        sizes = sorted(len(group) for group in groups.values())
        raise SpeechProvenanceError(
            "cannot form the exact speech held-out row target from at least two complete "
            f"speaker groups: target={heldout_rows}, group_sizes={sizes}"
        )

    heldout_group_ids = list(heldout_group_tuple)
    dev_groups: List[str] = []
    eval_groups: List[str] = []
    dev_rows = 0
    eval_rows = 0
    for index, group_id in enumerate(heldout_group_ids):
        size = len(groups[group_id])
        if index == 0 or (index > 1 and dev_rows <= eval_rows):
            dev_groups.append(group_id)
            dev_rows += size
        else:
            eval_groups.append(group_id)
            eval_rows += size
    if not dev_groups or not eval_groups:
        raise SpeechProvenanceError("speech held-out groups could not create non-empty dev/eval")

    partition_by_group = {group_id: "train" for group_id in group_ids}
    partition_by_group.update({group_id: "dev" for group_id in dev_groups})
    partition_by_group.update({group_id: "eval" for group_id in eval_groups})
    partitioned: Dict[str, List[Dict[str, Any]]] = {name: [] for name in SPEECH_PARTITIONS}
    for row in rows:
        copy = dict(row)
        copy["partition"] = partition_by_group[_speech_group_id(row)]
        partitioned[copy["partition"]].append(copy)
    def sort_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
        return (
            str(row["source_dataset"]),
            str(row["speaker_id"]),
            str(row["chapter_id"]),
            str(row["utterance_id"]),
        )
    ordered: List[Dict[str, Any]] = []
    for partition in SPEECH_PARTITIONS:
        partitioned[partition].sort(key=sort_key)
        ordered.extend(partitioned[partition])
    for index, row in enumerate(ordered):
        row["id"] = index

    group_sets = {
        partition: {_speech_group_id(row) for row in partition_rows}
        for partition, partition_rows in partitioned.items()
    }
    overlap = {
        f"{left}_{right}": sorted(group_sets[left] & group_sets[right])
        for left, right in (("train", "dev"), ("train", "eval"), ("dev", "eval"))
    }
    overlap_count = sum(len(values) for values in overlap.values())
    if overlap_count:
        raise SpeechProvenanceError(f"internal speaker overlap after partitioning: {overlap!r}")
    summary = {
        "schema_version": 1,
        "policy": SPEECH_PARTITION_POLICY,
        "seed": int(seed),
        "group_key": ["source_dataset", "speaker_id"],
        "partition_order": list(SPEECH_PARTITIONS),
        "legacy_tail_semantics": "dev and eval together are the held-out tail",
        "heldout_row_target": int(heldout_rows),
        "row_counts": {name: len(partitioned[name]) for name in SPEECH_PARTITIONS},
        "group_counts": {name: len(group_sets[name]) for name in SPEECH_PARTITIONS},
        "overlap_audit": {
            "pairwise_group_overlap_count": overlap_count,
            "pairwise_group_overlaps": overlap,
        },
        "source_id_fields": [
            "source_dataset", "source_config", "source_split",
            "speaker_id", "chapter_id", "utterance_id",
        ],
        "audio_hash_semantics": {
            "source_audio_fingerprint": "content hash when bytes/array are supplied; otherwise path/stat hash",
            "audio_fingerprint": "legacy path/stat fingerprint of materialized WAV",
            "audio_sha256": "exact SHA256 of materialized WAV bytes",
        },
    }
    return ordered, summary


def speech_partition_ledger(
    rows: List[Dict[str, Any]], partition_summary: Dict[str, Any]
) -> Dict[str, Any]:
    fields = (
        "id", "partition", "source_dataset", "source_config", "source_split",
        "speaker_id", "chapter_id", "utterance_id", "audio_path",
        "source_audio_fingerprint", "audio_fingerprint",
    )
    return {
        **partition_summary,
        "records": [{field: row[field] for field in fields} for row in rows],
    }


def build_audio_rows(
    out_dir: Path,
    max_samples: int,
    target_sr: int,
    max_seconds: float,
    errors: List[Dict[str, str]],
    max_source_audio_seconds: float = 0.0,
    max_transcript_words: int = 0,
    allow_short: bool = False,
    heldout_rows: int = 250,
    split_seed: int = 0,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    candidates = list(SPEECH_SOURCE_PROFILE)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    rows: List[Dict[str, Any]] = []
    used_sources: List[str] = []
    seen_transcripts = set()
    probe_multiplier = 12 if max_source_audio_seconds > 0.0 or max_transcript_words > 0 else 4
    for candidate in candidates:
        try:
            raw_rows = iter_dataset_rows(candidate, max_samples * probe_multiplier)
            before = len(rows)
            for raw in raw_rows:
                raw = dict(raw)
                transcript = extract_transcript(raw)
                if (
                    not transcript
                    or transcript in seen_transcripts
                    or (max_transcript_words > 0 and count_words(transcript) > max_transcript_words)
                ):
                    continue
                provenance = extract_librispeech_provenance(raw, candidate)
                source_fingerprint = source_audio_fingerprint(raw)
                audio_pair = extract_audio(raw)
                if audio_pair is None:
                    continue
                audio, sr = audio_pair
                source_duration_seconds = float(len(audio) / max(sr, 1))
                if max_source_audio_seconds > 0.0 and source_duration_seconds > max_source_audio_seconds:
                    continue
                processed = resample_and_pad(audio, sr, target_sr, max_seconds)
                audio_relative_path = Path("audio") / (
                    f"librispeech_{candidate.split.replace('.', '_')}_"
                    f"{len(rows):05d}.wav"
                )
                audio_path = out_dir / audio_relative_path
                sf.write(str(audio_path), processed, target_sr)
                audio_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
                rows.append({
                    "id": len(rows),
                    "task": "speech",
                    "source": candidate.label,
                    **provenance,
                    "audio_path": audio_relative_path.as_posix(),
                    "source_audio_fingerprint": source_fingerprint,
                    "audio_fingerprint": path_stat_fingerprint(
                        audio_path, audio_relative_path.as_posix()
                    ),
                    "audio_sha256": audio_sha256,
                    "transcript": transcript,
                    "sample_rate": target_sr,
                    "preprocess": {
                        "resampled_to": target_sr,
                        "max_seconds": max_seconds,
                        "num_samples": int(len(processed)),
                        "source_duration_seconds": source_duration_seconds,
                        "max_source_audio_seconds": max_source_audio_seconds,
                        "max_transcript_words": max_transcript_words,
                        "truncated_source_audio": bool(source_duration_seconds > max_seconds),
                    },
                })
                seen_transcripts.add(transcript)
                if len(rows) >= max_samples:
                    break
            if len(rows) > before:
                used_sources.append(candidate.label)
            if len(rows) >= max_samples:
                partitioned, summary = partition_speech_rows(rows, heldout_rows, split_seed)
                return partitioned, ";".join(used_sources), summary
        except SpeechProvenanceError:
            raise
        except Exception as exc:
            errors.append({"task": "speech", "candidate": candidate.label, "error": repr(exc), "traceback": traceback.format_exc(limit=4)})
    if allow_short and rows:
        partitioned, summary = partition_speech_rows(rows, heldout_rows, split_seed)
        return partitioned, ";".join(used_sources), summary
    raise RuntimeError(f"Could not load {max_samples} real speech rows; got={len(rows)} from {used_sources}; errors={errors[-3:]}")


def build_all(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors: List[Dict[str, str]] = []
    sources: Dict[str, str] = {}
    hf_sources: Dict[str, Any] = {}

    specs = [
        ("text", list(REAL_SOURCE_PROFILES["text"]), args.text_samples, map_plain_text),
        ("code", list(REAL_SOURCE_PROFILES["code"]), args.code_samples, map_code),
        (
            "reasoning",
            list(REAL_SOURCE_PROFILES["reasoning"]),
            args.reasoning_samples,
            map_logiqa,
        ),
        ("math", list(REAL_SOURCE_PROFILES["math"]), args.math_samples, map_gsm8k),
        (
            "education",
            list(REAL_SOURCE_PROFILES["education"]),
            args.education_samples,
            map_sat,
        ),
    ]

    text_rows: List[Dict[str, Any]] = []
    rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for task, candidates, count, mapper in specs:
        rows, source = build_text_like_rows(task, candidates, count, mapper, errors, allow_short=args.allow_short)
        sources[task] = source
        hf_sources[task] = structured_hf_sources(candidates, source)
        rows_by_task[task] = rows
        text_rows.extend(rows)
    write_jsonl(out_dir / "text_tasks.jsonl", text_rows)

    block_targets = {
        "text": args.text_train_blocks,
        "code": args.code_train_blocks,
        "reasoning": args.reasoning_train_blocks,
        "math": args.math_train_blocks,
        "education": args.education_train_blocks,
    }
    tokenizer_ref = resolve_model(
        args.tokenizer_model, getattr(args, "tokenizer_revision", None)
    )
    block_manifest = write_text_blocks(
        out_dir,
        rows_by_task,
        tokenizer_ref.repo_id,
        args.block_size,
        block_targets,
        args.eval_blocks_per_task,
        args.allow_short,
        tokenizer_revision=tokenizer_ref.revision,
    )
    hf_sources["tokenizer"] = tokenizer_ref.as_dict()

    image_rows, image_source = build_image_rows(
        out_dir,
        args.image_samples,
        errors,
        args.caption_min_ascii_ratio,
        args.caption_min_letters,
        allow_short=args.allow_short,
    )
    sources["image"] = image_source
    hf_sources["image"] = structured_hf_sources(IMAGE_SOURCE_PROFILE, image_source)
    write_jsonl(out_dir / "image_captions.jsonl", image_rows)

    audio_rows, audio_source, speech_partition = build_audio_rows(
        out_dir,
        args.speech_samples,
        args.sample_rate,
        args.max_audio_seconds,
        errors,
        max_source_audio_seconds=args.max_source_audio_seconds,
        max_transcript_words=args.max_transcript_words,
        allow_short=args.allow_short,
        heldout_rows=args.speech_eval_samples,
        split_seed=getattr(args, "speech_split_seed", 0),
    )
    sources["speech"] = audio_source
    hf_sources["speech"] = structured_hf_sources(SPEECH_SOURCE_PROFILE, audio_source)
    write_jsonl(out_dir / "speech_transcripts.jsonl", audio_rows)
    speech_ledger = speech_partition_ledger(audio_rows, speech_partition)
    speech_ledger_path = out_dir / "speech_partition_ledger.json"
    save_json(speech_ledger_path, speech_ledger)
    speech_ledger_bytes = json.dumps(
        speech_ledger, indent=2, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    speech_ledger_sha256 = hashlib.sha256(speech_ledger_bytes).hexdigest()

    speech_audio_commitment_rows = [
        {
            "id": row["id"],
            "audio_path": row["audio_path"],
            "audio_sha256": row["audio_sha256"],
        }
        for row in audio_rows
    ]
    speech_audio_bytes_commitment = {
        "policy": "exact_materialized_wav_sha256_per_row_v1",
        "field": "audio_sha256",
        "rows": len(speech_audio_commitment_rows),
        "row_commitment_sha256": hashlib.sha256(
            json.dumps(
                speech_audio_commitment_rows,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }

    text_counts = {task: sum(1 for row in text_rows if row["task"] == task) for task in ["text", "code", "reasoning", "math", "education"]}
    manifest = {
        "output_dir": str(out_dir),
        "sources": sources,
        "hf_sources": hf_sources,
        "counts": {
            "text_tasks": len(text_rows),
            "image_captions": len(image_rows),
            "speech_transcripts": len(audio_rows),
            **text_counts,
            "text_blocks_train": block_manifest["train_blocks"],
            "text_blocks_eval": block_manifest["eval_blocks"],
            "image_train_pairs": max(0, len(image_rows) - args.image_eval_samples),
            "image_eval_pairs": min(args.image_eval_samples, len(image_rows)),
            "speech_train_utterances": speech_partition["row_counts"]["train"],
            "speech_eval_utterances": (
                speech_partition["row_counts"]["dev"]
                + speech_partition["row_counts"]["eval"]
            ),
            "speech_dev_utterances": speech_partition["row_counts"]["dev"],
            "speech_final_eval_utterances": speech_partition["row_counts"]["eval"],
        },
        "block_counts": block_manifest,
        "speech_partition": speech_partition,
        "speech_audio_bytes_commitment": speech_audio_bytes_commitment,
        "speech_partition_ledger": {
            "path": str(speech_ledger_path),
            "sha256": speech_ledger_sha256,
            "rows": len(speech_ledger["records"]),
        },
        "preprocessing": {
            "text": f"OLMoE tokenizer {tokenizer_ref.repo_id}@{tokenizer_ref.revision}; packed into {args.block_size}-token blocks; train/eval JSONL files store input_ids for reproducibility.",
            "image": f"captions filtered with caption_min_ascii_ratio={args.caption_min_ascii_ratio} and caption_min_letters={args.caption_min_letters}; RGB conversion and 224x224 resize performed here; CLIP normalization is performed by AutoImageProcessor in the runner.",
            "audio": f"speech rows filtered with max_source_audio_seconds={args.max_source_audio_seconds} and max_transcript_words={args.max_transcript_words}; audio loaded, mono-converted, resampled to {args.sample_rate} Hz, padded/truncated to {args.max_audio_seconds} seconds, saved as WAV; Whisper padding/truncation is performed in the runner.",
            "batching": "runner builds text, image-caption, and speech-transcript micro-batches from these manifests.",
            "alignment": "image/audio prefixes are prepended as continuous tokens, prefix labels are masked with -100, and text/caption/transcript targets remain supervised.",
        },
        "allow_short": bool(args.allow_short),
        "errors": errors,
    }
    save_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/real_subset")
    parser.add_argument("--tokenizer-model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument(
        "--tokenizer-revision",
        default=None,
        help="Exact 40-hex commit required when --tokenizer-model is not registered.",
    )
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--text-samples", type=int, default=30000)
    parser.add_argument("--code-samples", type=int, default=1000)
    parser.add_argument("--reasoning-samples", type=int, default=651)
    parser.add_argument("--math-samples", type=int, default=2500)
    parser.add_argument("--education-samples", type=int, default=900)
    parser.add_argument("--text-train-blocks", type=int, default=14086)
    parser.add_argument("--code-train-blocks", type=int, default=1865)
    parser.add_argument("--reasoning-train-blocks", type=int, default=290)
    parser.add_argument("--math-train-blocks", type=int, default=611)
    parser.add_argument("--education-train-blocks", type=int, default=424)
    parser.add_argument("--eval-blocks-per-task", type=int, default=64)
    parser.add_argument("--image-samples", type=int, default=5250)
    parser.add_argument("--speech-samples", type=int, default=5250)
    parser.add_argument("--image-eval-samples", type=int, default=250)
    parser.add_argument("--speech-eval-samples", type=int, default=250)
    parser.add_argument(
        "--speech-split-seed",
        type=int,
        default=0,
        help="Seed for deterministic speaker-group selection within the speech held-out tail.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-audio-seconds", type=float, default=6.0)
    parser.add_argument("--caption-min-ascii-ratio", type=float, default=0.85)
    parser.add_argument("--caption-min-letters", type=int, default=8)
    parser.add_argument("--max-source-audio-seconds", type=float, default=0.0)
    parser.add_argument("--max-transcript-words", type=int, default=0)
    parser.add_argument("--allow-short", action="store_true", help="Allow debug/probe manifests that do not meet final-scale count targets.")
    args = parser.parse_args()
    manifest = build_all(args)
    print(json.dumps({"manifest": str(Path(args.output_dir) / "manifest.json"), "counts": manifest["counts"], "sources": manifest["sources"], "allow_short": manifest["allow_short"]}, sort_keys=True), flush=True)
    # Some HF audio/image backends can abort during interpreter finalization after all files are written.
    # Exit immediately after a successful manifest write so schedulers do not retry completed preprocessing.
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
