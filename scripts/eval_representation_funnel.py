"""Evaluate representation retention through a trained multimodal OLMoE path.

The protocol is deliberately split into two phases.  Ridge selection and fitting
consume development data only; sealed metrics are computed only after every
probe has been frozen.  Feature caches are keyed by checkpoint, manifest, code,
and split role so a development cache cannot be mistaken for a sealed cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from scripts.eval_conditional_retrieval import load_trained_wrapper
from training.olmoe_real_subset_runs import load_stage_b_initialization_checkpoint


RIDGE_VALUES: Tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
VALIDATION_FRACTION = 0.2
SELECTION_SEED = 260709
OVERFIT_PAIRS = 32
OVERFIT_R1_THRESHOLD = 0.95
STAGES: Tuple[str, ...] = (
    "encoder_pooled",
    "query_resampler_prefix_mean",
    "post_shared_olmoe_prefix_hidden_mean",
    "trained_retrieval_head",
)
DEVELOPMENT_SPLITS = frozenset({"dev", "development"})
UNPOOLED_PREFIX_KEY = "query_resampler_prefix_unpooled"
DIAGNOSTIC_MAX_VECTORS = 4096


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_records(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"manifest must contain a non-empty list: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"manifest contains a non-object row: {path}")
    return [dict(row) for row in rows]


def _resolve_media_paths(rows: Sequence[Dict[str, Any]], manifest_path: Path, key: str) -> None:
    for row in rows:
        raw = row.get(key)
        if not raw:
            raise ValueError(f"manifest row is missing {key}: {manifest_path}")
        path = Path(str(raw))
        if not path.is_absolute():
            candidates = [manifest_path.parent / path, Path.cwd() / path]
            existing = {
                candidate.resolve() for candidate in candidates if candidate.is_file()
            }
            if len(existing) > 1:
                raise ValueError(
                    f"ambiguous relative {key} resolves to multiple files: {raw}"
                )
            path = next(iter(existing)) if existing else candidates[0]
        if not path.is_file():
            raise FileNotFoundError(path)
        row[key] = str(path.resolve())


def _canonical_texts(row: Mapping[str, Any], modality: str) -> List[str]:
    raw: Any
    if modality == "image":
        raw = row.get("captions") or row.get("caption") or row.get("canonical_caption")
    else:
        raw = row.get("transcripts") or row.get("transcript") or row.get("text")
    values = raw if isinstance(raw, list) else [raw]
    output: List[str] = []
    seen = set()
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    if not output:
        raise ValueError(f"{modality} row has no non-empty target text")
    return output


def _row_id(row: Mapping[str, Any], modality: str, index: int) -> str:
    for key in ("uid", "id", "utterance_id", "source_id", "content_sha256", "media_sha256"):
        if row.get(key) not in (None, ""):
            return f"{modality}:{row[key]}"
    return f"{modality}:row:{index}"


def build_target_index(
    rows: Sequence[Mapping[str, Any]], modality: str
) -> Tuple[List[str], List[str], List[str], List[List[int]]]:
    media_ids: List[str] = []
    text_ids: List[str] = []
    texts: List[str] = []
    positives: List[List[int]] = []
    for row_index, row in enumerate(rows):
        media_id = _row_id(row, modality, row_index)
        media_ids.append(media_id)
        row_positives: List[int] = []
        for local_index, text in enumerate(_canonical_texts(row, modality)):
            row_positives.append(len(texts))
            text_ids.append(f"{media_id}:text:{local_index}")
            texts.append(text)
        positives.append(row_positives)
    if len(set(media_ids)) != len(media_ids):
        raise ValueError(f"{modality} manifest has duplicate media IDs")
    return media_ids, text_ids, texts, positives


def _nested_source_ids(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "source_ids" and isinstance(child, list):
                yield from (str(item) for item in child)
            else:
                yield from _nested_source_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_source_ids(child)


def assert_disjoint_manifests(
    development: Sequence[Mapping[str, Any]], sealed: Sequence[Mapping[str, Any]], modality: str
) -> None:
    hash_keys = ("media_sha256", "content_sha256", "resized_content_sha256")
    dev_hashes = {str(row[key]) for row in development for key in hash_keys if row.get(key)}
    test_hashes = {str(row[key]) for row in sealed for key in hash_keys if row.get(key)}
    dev_sources = set(_nested_source_ids(development))
    test_sources = set(_nested_source_ids(sealed))
    media_key = "image_path" if modality == "image" else "audio_path"
    dev_paths = {str(row.get(media_key)) for row in development if row.get(media_key)}
    test_paths = {str(row.get(media_key)) for row in sealed if row.get(media_key)}
    overlaps = {
        "content_hashes": sorted(dev_hashes & test_hashes),
        "source_ids": sorted(dev_sources & test_sources),
        "media_paths": sorted(dev_paths & test_paths),
    }
    if any(overlaps.values()):
        counts = {key: len(value) for key, value in overlaps.items()}
        raise ValueError(f"development/sealed {modality} overlap detected: {counts}")


@dataclass(frozen=True)
class RidgeProbe:
    weight: torch.Tensor
    intercept: torch.Tensor
    alpha: float
    fit_split: str

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return features.float() @ self.weight + self.intercept


def _require_development(split_name: str) -> None:
    if str(split_name).lower() not in DEVELOPMENT_SPLITS:
        raise ValueError(
            f"ridge fitting is development-only; received split_name={split_name!r}"
        )


def solve_ridge(
    features: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    *,
    split_name: str,
) -> RidgeProbe:
    """Fit an affine ridge map with an unregularized intercept on CPU."""
    _require_development(split_name)
    x = torch.as_tensor(features, dtype=torch.float64, device="cpu")
    y = torch.as_tensor(targets, dtype=torch.float64, device="cpu")
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge features/targets must be aligned rank-2 tensors")
    if x.shape[0] < 1 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("ridge inputs must be non-empty and finite")
    if not math.isfinite(float(alpha)) or float(alpha) < 0.0:
        raise ValueError("ridge alpha must be finite and non-negative")
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    xc = x - x_mean
    yc = y - y_mean
    if float(alpha) == 0.0:
        weight = torch.linalg.lstsq(xc, yc).solution
    elif x.shape[0] <= x.shape[1]:
        gram = xc @ xc.T
        gram.diagonal().add_(float(alpha))
        weight = xc.T @ torch.linalg.solve(gram, yc)
    else:
        gram = xc.T @ xc
        gram.diagonal().add_(float(alpha))
        weight = torch.linalg.solve(gram, xc.T @ yc)
    intercept = y_mean.squeeze(0) - x_mean.squeeze(0) @ weight
    return RidgeProbe(
        weight=weight.float(),
        intercept=intercept.float(),
        alpha=float(alpha),
        fit_split=str(split_name),
    )


def fit_ridge_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    *,
    split_name: str,
) -> RidgeProbe:
    """Guarded public fitting entry point used by protocol and tests."""
    return solve_ridge(features, targets, alpha, split_name=split_name)


def multi_positive_ranks(
    similarity: torch.Tensor, positive_indices: Sequence[Sequence[int]]
) -> List[int]:
    """Return tie-aware best-positive zero-based ranks.

    Counting strictly better candidates makes ranks invariant to candidate order,
    including when candidates tie.
    """
    scores = torch.as_tensor(similarity).float().cpu()
    if scores.ndim != 2 or scores.shape[0] != len(positive_indices):
        raise ValueError("similarity/positive_indices shape mismatch")
    if not torch.isfinite(scores).all():
        raise ValueError("similarity contains non-finite values")
    ranks: List[int] = []
    for row_index, positives in enumerate(positive_indices):
        unique = sorted({int(index) for index in positives})
        if not unique or unique[0] < 0 or unique[-1] >= scores.shape[1]:
            raise ValueError("each query needs valid positive candidate indices")
        best_positive = scores[row_index, unique].max()
        ranks.append(int((scores[row_index] > best_positive).sum().item()))
    return ranks


def _transpose_positives(
    media_to_text: Sequence[Sequence[int]], text_count: int
) -> List[List[int]]:
    output: List[List[int]] = [[] for _ in range(int(text_count))]
    for media_index, text_indices in enumerate(media_to_text):
        for text_index in text_indices:
            output[int(text_index)].append(int(media_index))
    if any(not values for values in output):
        raise ValueError("every text target must have a positive media query")
    return output


def rank_metrics(ranks: Sequence[int]) -> Dict[str, Any]:
    if not ranks:
        raise ValueError("cannot compute retrieval metrics without ranks")
    tensor = torch.tensor([int(rank) + 1 for rank in ranks], dtype=torch.float64)
    return {
        "query_count": len(ranks),
        "r_at_1": float((tensor <= 1).double().mean().item()),
        "r_at_5": float((tensor <= 5).double().mean().item()),
        "r_at_10": float((tensor <= 10).double().mean().item()),
        "mrr": float((1.0 / tensor).mean().item()),
        "median_rank": float(torch.quantile(tensor, 0.5).item()),
    }


def evaluate_similarity(
    similarity: torch.Tensor, media_to_text: Sequence[Sequence[int]]
) -> Dict[str, Any]:
    if similarity.ndim != 2 or similarity.shape[0] != len(media_to_text):
        raise ValueError("similarity shape does not match media positives")
    media_ranks = multi_positive_ranks(similarity, media_to_text)
    text_to_media = _transpose_positives(media_to_text, similarity.shape[1])
    text_ranks = multi_positive_ranks(similarity.T, text_to_media)
    return {
        "media_to_text": rank_metrics(media_ranks),
        "text_to_media": rank_metrics(text_ranks),
        "media_to_text_ranks": media_ranks,
        "text_to_media_ranks": text_ranks,
    }


def tensor_diagnostics(tensor: torch.Tensor) -> Dict[str, Any]:
    value = torch.as_tensor(tensor).detach().float().cpu()
    if value.ndim < 2 or value.shape[-1] < 1:
        raise ValueError("representation diagnostics require shape [..., dimension]")
    finite = torch.isfinite(value)
    matrix = value.reshape(-1, value.shape[-1])
    valid = matrix[torch.isfinite(matrix).all(dim=1)]
    norms = torch.linalg.vector_norm(valid, dim=-1)
    norm_summary = {
        "min": float(norms.min().item()) if norms.numel() else None,
        "max": float(norms.max().item()) if norms.numel() else None,
        "mean": float(norms.mean().item()) if norms.numel() else None,
        "std": float(norms.std(unbiased=False).item()) if norms.numel() else None,
    }
    variance = (
        valid.var(dim=0, unbiased=False)
        if valid.numel()
        else torch.full((value.shape[-1],), float("nan"))
    )
    finite_variance = variance[torch.isfinite(variance)]
    if valid.shape[0] > DIAGNOSTIC_MAX_VECTORS:
        indices = torch.linspace(
            0, valid.shape[0] - 1, DIAGNOSTIC_MAX_VECTORS, dtype=torch.long
        )
        sampled = valid[indices]
    else:
        sampled = valid
    centered = sampled - sampled.mean(dim=0, keepdim=True) if sampled.numel() else sampled
    if centered.numel():
        singular_values = torch.linalg.svdvals(centered.double())
        finite_positive = singular_values[
            torch.isfinite(singular_values) & (singular_values > 0)
        ]
        spectrum = finite_positive.square()
    else:
        spectrum = torch.empty(0, dtype=torch.float64)
    spectrum_total = spectrum.sum()
    if spectrum.numel() and torch.isfinite(spectrum_total) and spectrum_total > 0:
        probabilities = spectrum / spectrum_total
        entropy_terms = torch.where(
            probabilities > 0,
            probabilities * probabilities.log(),
            torch.zeros_like(probabilities),
        )
        effective_rank = float(torch.exp(-entropy_terms.sum()).item())
        if not math.isfinite(effective_rank):
            raise RuntimeError("effective-rank computation produced a non-finite value")
    else:
        effective_rank = 0.0
    normalized = F.normalize(sampled, dim=-1)
    pair_count = int(normalized.shape[0] * (normalized.shape[0] - 1) // 2)
    if pair_count:
        vector_sum = normalized.sum(dim=0)
        self_cosine = (normalized * normalized).sum()
        pairwise_mean = float(
            ((vector_sum.square().sum() - self_cosine) / (2 * pair_count)).item()
        )
    else:
        pairwise_mean = None
    return {
        "shape": list(value.shape),
        "dimension": int(value.shape[-1]),
        "finite": bool(finite.all().item()),
        "nonfinite_values": int((~finite).sum().item()),
        "l2_norm": norm_summary,
        "prefix_norm": dict(norm_summary),
        "per_dimension_variance": {
            "values": [
                float(item) if math.isfinite(float(item)) else None
                for item in variance.tolist()
            ],
            "min": float(finite_variance.min().item()) if finite_variance.numel() else None,
            "max": float(finite_variance.max().item()) if finite_variance.numel() else None,
            "mean": float(finite_variance.mean().item()) if finite_variance.numel() else None,
            "zero_dimensions": int((variance == 0).sum().item()),
        },
        "effective_rank": effective_rank,
        "off_diagonal_pairwise_cosine": {
            "pair_count": pair_count,
            "mean": pairwise_mean,
        },
        "diagnostic_vectors": {
            "total": int(matrix.shape[0]),
            "finite": int(valid.shape[0]),
            "used_for_rank_and_cosine": int(sampled.shape[0]),
            "deterministic_cap": DIAGNOSTIC_MAX_VECTORS,
        },
    }


def _pair_tensors(
    media_features: torch.Tensor,
    targets: torch.Tensor,
    media_to_text: Sequence[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    media_indices: List[int] = []
    text_indices: List[int] = []
    for media_index, positives in enumerate(media_to_text):
        for text_index in positives:
            media_indices.append(int(media_index))
            text_indices.append(int(text_index))
    return (
        media_features[media_indices],
        targets[text_indices],
        media_indices,
        text_indices,
    )


def _subset_retrieval(
    media_features: torch.Tensor,
    targets: torch.Tensor,
    media_to_text: Sequence[Sequence[int]],
    selected_media: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
    target_indices: List[int] = []
    for media_index in selected_media:
        target_indices.extend(int(value) for value in media_to_text[int(media_index)])
    target_indices = list(dict.fromkeys(target_indices))
    remap = {old: new for new, old in enumerate(target_indices)}
    positives = [
        [remap[int(value)] for value in media_to_text[int(media_index)]]
        for media_index in selected_media
    ]
    return media_features[list(selected_media)], targets[target_indices], positives


def select_ridge_fixed_split(
    media_features: torch.Tensor,
    targets: torch.Tensor,
    media_to_text: Sequence[Sequence[int]],
    *,
    split_name: str,
    ridge_values: Sequence[float] = RIDGE_VALUES,
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = SELECTION_SEED,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, List[int]]]:
    """Select alpha on a fixed development media-level train/validation split."""
    _require_development(split_name)
    media_count = int(media_features.shape[0])
    if media_count < 2:
        raise ValueError("ridge selection needs at least two development media rows")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(media_count, generator=generator).tolist()
    validation_count = min(media_count - 1, max(1, int(round(media_count * validation_fraction))))
    validation_media = sorted(int(value) for value in order[:validation_count])
    train_media = sorted(int(value) for value in order[validation_count:])
    train_x, train_y, _, _ = _pair_tensors(
        media_features[train_media],
        targets,
        [media_to_text[index] for index in train_media],
    )
    val_x, val_y, val_positives = _subset_retrieval(
        media_features, targets, media_to_text, validation_media
    )
    trace: List[Dict[str, Any]] = []
    best_alpha: Optional[float] = None
    best_score = -float("inf")
    for alpha in ridge_values:
        probe = fit_ridge_probe(train_x, train_y, float(alpha), split_name=split_name)
        prediction = F.normalize(probe.predict(val_x), dim=-1)
        target = F.normalize(val_y.float(), dim=-1)
        result = evaluate_similarity(prediction @ target.T, val_positives)
        score = 0.5 * (
            result["media_to_text"]["r_at_1"] + result["text_to_media"]["r_at_1"]
        )
        trace.append(
            {
                "alpha": float(alpha),
                "selection_score_mean_bidirectional_r_at_1": float(score),
                "media_to_text": result["media_to_text"],
                "text_to_media": result["text_to_media"],
            }
        )
        if score > best_score + 1e-12:
            best_score = float(score)
            best_alpha = float(alpha)
    assert best_alpha is not None
    return best_alpha, trace, {"train_media_indices": train_media, "validation_media_indices": validation_media}


def overfit_diagnostic(
    media_features: torch.Tensor,
    targets: torch.Tensor,
    media_to_text: Sequence[Sequence[int]],
    alpha: float,
    *,
    split_name: str,
    pair_limit: int = OVERFIT_PAIRS,
) -> Dict[str, Any]:
    pair_x, pair_y, pair_media, _ = _pair_tensors(media_features, targets, media_to_text)
    count = min(int(pair_limit), int(pair_x.shape[0]))
    pair_x = pair_x[:count]
    pair_y = pair_y[:count]
    pair_media = pair_media[:count]
    probe = fit_ridge_probe(pair_x, pair_y, alpha, split_name=split_name)
    similarity = F.normalize(probe.predict(pair_x), dim=-1) @ F.normalize(pair_y.float(), dim=-1).T
    positives = [
        [candidate for candidate, media_index in enumerate(pair_media) if media_index == query_media]
        for query_media in pair_media
    ]
    result = evaluate_similarity(similarity, positives)
    train_r1 = 0.5 * (
        result["media_to_text"]["r_at_1"] + result["text_to_media"]["r_at_1"]
    )
    return {
        "pair_count": count,
        "train_mean_bidirectional_r_at_1": float(train_r1),
        "threshold": OVERFIT_R1_THRESHOLD,
        "reaches_threshold": bool(train_r1 >= OVERFIT_R1_THRESHOLD),
        "media_to_text": result["media_to_text"],
        "text_to_media": result["text_to_media"],
    }


@dataclass
class RetrievalFeatures:
    modality: str
    split_name: str
    media_ids: List[str]
    text_ids: List[str]
    texts: List[str]
    media_to_text: List[List[int]]
    stages: Dict[str, torch.Tensor]
    targets: torch.Tensor
    cache_path: Path
    cache_sha256: str
    pooling: str
    unpooled_prefix: Optional[torch.Tensor] = None


def _cache_payload_to_features(payload: Mapping[str, Any], cache_path: Path) -> RetrievalFeatures:
    return RetrievalFeatures(
        modality=str(payload["modality"]),
        split_name=str(payload["split_name"]),
        media_ids=list(payload["media_ids"]),
        text_ids=list(payload["text_ids"]),
        texts=list(payload["texts"]),
        media_to_text=[list(map(int, values)) for values in payload["media_to_text"]],
        stages={name: tensor.float().cpu() for name, tensor in payload["stages"].items()},
        targets=payload["targets"].float().cpu(),
        unpooled_prefix=(
            payload["unpooled_prefix"].float().cpu()
            if payload.get("unpooled_prefix") is not None
            else None
        ),
        cache_path=cache_path,
        cache_sha256=sha256_file(cache_path),
        pooling=str(payload.get("pooling", "unknown")),
    )


def _image_encoder_pool(tokens: torch.Tensor, vision_model: Any) -> Tuple[torch.Tensor, str]:
    projection = getattr(vision_model, "visual_projection", None)
    vision_core = getattr(vision_model, "vision_model", None)
    post_layernorm = getattr(vision_core, "post_layernorm", None)
    if projection is not None:
        pooled = tokens[:, 0]
        if post_layernorm is not None:
            pooled = post_layernorm(pooled)
        return projection(pooled).float(), "clip_cls_post_layernorm_visual_projection"
    return tokens.float().mean(dim=1), "mean_last_hidden_state"


@torch.no_grad()
def _extract_batch_stages(
    wrapper: Any,
    encoder_tokens: torch.Tensor,
    modality: str,
    vision_model: Any,
) -> Tuple[Dict[str, torch.Tensor], str]:
    if modality == "image":
        encoder_pooled, pooling = _image_encoder_pool(encoder_tokens, vision_model)
        prefix = wrapper.image_prefix(encoder_tokens)
        head = wrapper.image_retrieval_head
    else:
        encoder_pooled = encoder_tokens.float().mean(dim=1)
        pooling = "mean_last_hidden_state"
        prefix = wrapper.audio_prefix(encoder_tokens)
        head = wrapper.audio_retrieval_head
    target_dtype = wrapper.lm.get_input_embeddings().weight.dtype
    lm_prefix = prefix.to(dtype=target_dtype)
    attention_mask = torch.ones(lm_prefix.shape[:2], dtype=torch.long, device=lm_prefix.device)
    outputs = wrapper.lm(
        inputs_embeds=lm_prefix,
        attention_mask=attention_mask,
        output_hidden_states=True,
        output_router_logits=False,
        return_dict=True,
    )
    prefix_mean = prefix.float().mean(dim=1)
    post_olmoe = outputs.hidden_states[-1].float().mean(dim=1)
    head_input = post_olmoe + prefix_mean if wrapper.use_prefix_residual_alignment else post_olmoe
    final_vector = F.normalize(head(head_input).float(), dim=-1)
    return {
        "encoder_pooled": encoder_pooled.detach().float().cpu(),
        "query_resampler_prefix_mean": prefix_mean.detach().float().cpu(),
        "post_shared_olmoe_prefix_hidden_mean": post_olmoe.detach().float().cpu(),
        "trained_retrieval_head": final_vector.detach().float().cpu(),
        UNPOOLED_PREFIX_KEY: prefix.detach().float().cpu(),
    }, pooling


def extract_or_load_features(
    *,
    wrapper: Any,
    tokenizer: Any,
    image_processor: Any,
    vision_model: Any,
    speech_processor: Any,
    speech_model: Any,
    rows: Sequence[Dict[str, Any]],
    manifest_path: Path,
    modality: str,
    split_name: str,
    device: torch.device,
    batch_size: int,
    cache_root: Path,
    checkpoint_sha256: str,
    gamma_sha256: Optional[str],
    code_sha256: str,
    config: Any,
) -> RetrievalFeatures:
    from training.olmoe_real_subset_runs import (
        FeatureCache,
        clip_text_embeddings,
        lm_text_embeddings,
        load_speech_text_tokenizer,
        load_vision_text_tokenizer,
        speech_text_embeddings,
    )

    manifest_sha256 = sha256_file(manifest_path)
    cache_key = hashlib.sha256(
        f"{modality}|{split_name}|{manifest_sha256}|{checkpoint_sha256}|{gamma_sha256}|{code_sha256}".encode("utf-8")
    ).hexdigest()
    cache_path = cache_root / "representation_stages" / f"{modality}_{split_name}_{cache_key[:20]}.pt"
    expected = {
        "modality": modality,
        "split_name": split_name,
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "gamma_sha256": gamma_sha256,
        "code_sha256": code_sha256,
    }
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        actual = payload.get("cache_identity", {})
        if actual != expected:
            raise ValueError(f"cache identity mismatch: {cache_path}")
        return _cache_payload_to_features(payload, cache_path)

    media_ids, text_ids, texts, media_to_text = build_target_index(rows, modality)
    encoder_cache = FeatureCache(cache_root / "encoder_tokens" / modality / split_name)
    stage_parts: Dict[str, List[torch.Tensor]] = {name: [] for name in STAGES}
    unpooled_prefix_parts: List[torch.Tensor] = []
    pooling: Optional[str] = None
    for start in range(0, len(rows), int(batch_size)):
        batch_rows = list(rows[start : start + int(batch_size)])
        if modality == "image":
            tokens = encoder_cache.image_batch(
                image_processor, vision_model, batch_rows, device, config.encoder_feature_tokens
            )
        else:
            tokens = encoder_cache.audio_batch(
                speech_processor,
                speech_model,
                batch_rows,
                device,
                config.sample_rate,
                config.encoder_feature_tokens,
            )
        batch_stages, batch_pooling = _extract_batch_stages(
            wrapper, tokens, modality, vision_model
        )
        pooling = pooling or batch_pooling
        if pooling != batch_pooling:
            raise RuntimeError("encoder pooling changed between batches")
        for name in STAGES:
            stage_parts[name].append(batch_stages[name])
        unpooled_prefix_parts.append(batch_stages[UNPOOLED_PREFIX_KEY])
    stages = {name: torch.cat(parts, dim=0) for name, parts in stage_parts.items()}
    unpooled_prefix = torch.cat(unpooled_prefix_parts, dim=0)

    if modality == "image":
        text_tokenizer = load_vision_text_tokenizer(config.vision_model)
        targets = clip_text_embeddings(
            vision_model, text_tokenizer, texts, device, int(batch_size)
        )
    elif config.speech_target_space == "whisper_decoder_text":
        text_tokenizer = load_speech_text_tokenizer(config.speech_model)
        targets = speech_text_embeddings(
            text_tokenizer, speech_model, texts, device, int(batch_size)
        )
    elif config.speech_target_space == "olmoe_text_hidden":
        targets = lm_text_embeddings(
            wrapper.lm, tokenizer, texts, device, config.max_length, int(batch_size)
        )
    else:
        raise ValueError(f"unsupported checkpoint speech_target_space={config.speech_target_space!r}")
    payload = {
        "cache_identity": expected,
        "modality": modality,
        "split_name": split_name,
        "media_ids": media_ids,
        "text_ids": text_ids,
        "texts": texts,
        "media_to_text": media_to_text,
        "stages": stages,
        "unpooled_prefix": unpooled_prefix,
        "targets": targets.float().cpu(),
        "pooling": pooling,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return _cache_payload_to_features(payload, cache_path)


def _checkpoint_config(state: Mapping[str, Any]) -> SimpleNamespace:
    raw = dict(state.get("args") or {})
    defaults = {
        "base_model": "allenai/OLMoE-1B-7B-0924",
        "vision_model": "openai/clip-vit-base-patch32",
        "speech_model": "openai/whisper-base.en",
        "speech_target_space": "olmoe_text_hidden",
        "alignment_prefix_residual": False,
        "image_prefix_tokens": 50,
        "audio_prefix_tokens": 64,
        "encoder_feature_tokens": 100,
        "sample_rate": 16000,
        "max_length": 512,
        "capacity_factor": 4.0,
        "aux_coef": 0.01,
    }
    for key, value in defaults.items():
        raw.setdefault(key, value)
    return SimpleNamespace(**raw)


def _bind_runtime_base_identity(
    model_meta: Dict[str, Any],
    stage_b_checkpoint: str,
    stage_b_checkpoint_sha256: str,
) -> Dict[str, Any]:
    restoration = model_meta.get("checkpoint_restoration")
    if not isinstance(restoration, Mapping):
        raise ValueError("loaded model metadata is missing checkpoint_restoration")
    stage_b_state, stage_b_provenance = load_stage_b_initialization_checkpoint(
        stage_b_checkpoint, stage_b_checkpoint_sha256
    )
    if stage_b_state is None:
        raise ValueError("Stage-B checkpoint is required for final evaluation")
    restored_stage_b = restoration.get("stage_b_checkpoint")
    if not isinstance(restored_stage_b, Mapping) or any(
        restored_stage_b.get(key) != value
        for key, value in stage_b_provenance.items()
    ):
        raise ValueError("loaded Stage-B provenance differs from the exact supplied checkpoint")
    resume_contract = stage_b_state.get("resume_contract")
    runtime_identity = (
        resume_contract.get("base_model_identity")
        if isinstance(resume_contract, Mapping)
        else None
    )
    if not isinstance(runtime_identity, Mapping) or not runtime_identity:
        raise ValueError("Stage-B checkpoint is missing runtime base-model identity")
    identity = dict(runtime_identity)
    identity_sha256 = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    bound = dict(restoration)
    bound["runtime_base_model_identity"] = identity
    bound["runtime_base_model_identity_sha256"] = identity_sha256
    bound["runtime_base_model_identity_verified_against_stage_b"] = True
    model_meta["checkpoint_restoration"] = bound
    return bound


def load_selected_wrapper(
    checkpoint: Path,
    gamma_path: Optional[Path],
    stage_b_checkpoint: str,
    stage_b_checkpoint_sha256: str,
) -> Tuple[Any, ...]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("E3 checkpoint payload must be a mapping")
    config = _checkpoint_config(state)
    checkpoint_args = state.get("args")
    if not isinstance(checkpoint_args, Mapping) or not checkpoint_args.get("output_dir"):
        raise ValueError("E3 checkpoint is missing args.output_dir")
    evaluation_args = SimpleNamespace(**vars(config))
    evaluation_args.checkpoint = checkpoint
    evaluation_args.run_output_dir = Path(str(checkpoint_args["output_dir"]))
    evaluation_args.evaluation_scope = "final"
    evaluation_args.stage_b_checkpoint = stage_b_checkpoint
    evaluation_args.stage_b_checkpoint_sha256 = stage_b_checkpoint_sha256
    evaluation_args.top_k = 2
    (
        wrapper,
        tokenizer,
        model_meta,
        image_processor,
        vision_model,
        speech_processor,
        speech_model,
        device,
    ) = load_trained_wrapper(evaluation_args)
    restoration = _bind_runtime_base_identity(
        model_meta, stage_b_checkpoint, stage_b_checkpoint_sha256
    )
    if restoration.get("restoration_order") != [
        "base_model",
        "stage_b_student_checkpoint",
        "e3_training_checkpoint",
    ]:
        raise ValueError("unexpected E3 checkpoint restoration order")
    checkpoint_gamma = model_meta.get("gamma_provenance")
    if gamma_path is not None and (
        not isinstance(checkpoint_gamma, Mapping)
        or Path(str(checkpoint_gamma.get("path"))).resolve(strict=True)
        != gamma_path.resolve(strict=True)
    ):
        raise ValueError("--gamma-json differs from checkpoint-bound gamma provenance")
    return (
        wrapper,
        tokenizer,
        image_processor,
        vision_model,
        speech_processor,
        speech_model,
        device,
        evaluation_args,
        model_meta,
    )


@dataclass
class PreparedStage:
    probe: RidgeProbe
    report: Dict[str, Any]


def prepare_stage(
    development: RetrievalFeatures, stage_name: str
) -> PreparedStage:
    features = development.stages[stage_name]
    targets = development.targets
    selected_alpha, trace, assignment = select_ridge_fixed_split(
        features,
        targets,
        development.media_to_text,
        split_name=development.split_name,
    )
    pair_x, pair_y, _, _ = _pair_tensors(features, targets, development.media_to_text)
    probe = fit_ridge_probe(
        pair_x, pair_y, selected_alpha, split_name=development.split_name
    )
    report = {
        "development_diagnostics": tensor_diagnostics(features),
        "target_diagnostics": tensor_diagnostics(targets),
        "ridge_selection": {
            "method": "fixed_development_train_validation_split",
            "ridge_values": list(RIDGE_VALUES),
            "validation_fraction": VALIDATION_FRACTION,
            "seed": SELECTION_SEED,
            "selected_alpha": selected_alpha,
            "assignment": assignment,
            "trace": trace,
        },
        "development_overfit_diagnostic": overfit_diagnostic(
            features,
            targets,
            development.media_to_text,
            selected_alpha,
            split_name=development.split_name,
        ),
        "final_probe_fit": {
            "fit_split": development.split_name,
            "pair_count": int(pair_x.shape[0]),
            "input_dimension": int(pair_x.shape[1]),
            "output_dimension": int(pair_y.shape[1]),
            "alpha": selected_alpha,
        },
    }
    if stage_name == "query_resampler_prefix_mean":
        if development.unpooled_prefix is None:
            raise ValueError("development cache is missing unpooled prefix representations")
        report["development_unpooled_prefix_diagnostics"] = tensor_diagnostics(
            development.unpooled_prefix
        )
        report["development_unpooled_prefix_diagnostics"]["selection_scope"] = {
            "development_only": True,
            "used_for_model_selection": False,
            "sealed_selection": False,
        }
    return PreparedStage(probe=probe, report=report)


def _append_rank_rows(
    rows: List[Dict[str, Any]],
    result: Mapping[str, Any],
    features: RetrievalFeatures,
    stage_name: str,
    method: str,
) -> None:
    for direction, query_ids, candidate_ids in (
        ("media_to_text", features.media_ids, features.text_ids),
        ("text_to_media", features.text_ids, features.media_ids),
    ):
        for query_index, (query_id, rank) in enumerate(
            zip(query_ids, result[f"{direction}_ranks"])
        ):
            rows.append(
                {
                    "modality": features.modality,
                    "stage": stage_name,
                    "method": method,
                    "direction": direction,
                    "query_index": query_index,
                    "query_id": query_id,
                    "candidate_count": len(candidate_ids),
                    "rank": int(rank) + 1,
                    "rank_base": 1,
                }
            )


def evaluate_frozen_stage(
    sealed: RetrievalFeatures,
    stage_name: str,
    prepared: PreparedStage,
    rank_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    features = sealed.stages[stage_name]
    targets = sealed.targets
    output = dict(prepared.report)
    output["sealed_diagnostics"] = tensor_diagnostics(features)
    output["sealed_target_diagnostics"] = tensor_diagnostics(targets)
    output["direct_retrieval"] = {
        "applicable": bool(features.shape[1] == targets.shape[1]),
        "reason": None
        if features.shape[1] == targets.shape[1]
        else f"dimension_mismatch:{features.shape[1]}!={targets.shape[1]}",
    }
    if features.shape[1] == targets.shape[1]:
        direct = evaluate_similarity(
            F.normalize(features.float(), dim=-1) @ F.normalize(targets.float(), dim=-1).T,
            sealed.media_to_text,
        )
        output["direct_retrieval"].update(
            {
                "media_to_text": direct["media_to_text"],
                "text_to_media": direct["text_to_media"],
            }
        )
        _append_rank_rows(rank_rows, direct, sealed, stage_name, "direct")
    predicted = F.normalize(prepared.probe.predict(features), dim=-1)
    probed = evaluate_similarity(
        predicted @ F.normalize(targets.float(), dim=-1).T, sealed.media_to_text
    )
    output["ridge_probe_retrieval"] = {
        "media_to_text": probed["media_to_text"],
        "text_to_media": probed["text_to_media"],
        "sealed_evaluation_count": 1,
    }
    _append_rank_rows(rank_rows, probed, sealed, stage_name, "ridge_probe")
    return output


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage-b-checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint-sha256", required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--gamma-json", type=Path)
    parser.add_argument("--image-dev-manifest", type=Path, required=True)
    parser.add_argument("--speech-dev-manifest", type=Path, required=True)
    parser.add_argument("--image-test-manifest", type=Path, required=True)
    parser.add_argument("--speech-test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("representation extraction requires one CUDA GPU")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    protocol_manifest = args.protocol_manifest.resolve(strict=True)
    frozen_protocol = json.loads(protocol_manifest.read_text(encoding="utf-8"))
    if frozen_protocol.get("protocol") != "sealed_evaluation_protocol":
        raise ValueError("unsupported frozen protocol manifest")
    selected_root = frozen_protocol.get("checkpoint", {}).get("selected_root")
    if not selected_root or Path(str(selected_root)).resolve() != checkpoint.parent.parent:
        raise ValueError("checkpoint root disagrees with frozen protocol")
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "representation_funnel.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = (args.cache_dir or output_dir / "cache").resolve()
    code_path = Path(__file__).resolve()
    checkpoint_sha256 = sha256_file(checkpoint)
    code_sha256 = sha256_file(code_path)
    gamma_candidate = (
        args.gamma_json.resolve()
        if args.gamma_json is not None
        else checkpoint.parent.parent / "calibration" / "gamma.json"
    )
    gamma_sha256 = sha256_file(gamma_candidate) if gamma_candidate.is_file() else None

    manifests = {
        "image_development": args.image_dev_manifest.resolve(),
        "speech_development": args.speech_dev_manifest.resolve(),
        "image_sealed_test": args.image_test_manifest.resolve(),
        "speech_sealed_test": args.speech_test_manifest.resolve(),
    }
    frozen_inputs = frozen_protocol.get("inputs", {})
    for role, manifest_key in (
        ("image_test", "image_sealed_test"),
        ("speech_test", "speech_sealed_test"),
    ):
        stored_path = frozen_inputs.get(role, {}).get("path")
        if not stored_path or Path(str(stored_path)).resolve() != manifests[manifest_key]:
            raise ValueError(f"{manifest_key} disagrees with frozen protocol")
    records = {name: _read_records(path) for name, path in manifests.items()}
    _resolve_media_paths(records["image_development"], manifests["image_development"], "image_path")
    _resolve_media_paths(records["image_sealed_test"], manifests["image_sealed_test"], "image_path")
    _resolve_media_paths(records["speech_development"], manifests["speech_development"], "audio_path")
    _resolve_media_paths(records["speech_sealed_test"], manifests["speech_sealed_test"], "audio_path")
    assert_disjoint_manifests(
        records["image_development"], records["image_sealed_test"], "image"
    )
    assert_disjoint_manifests(
        records["speech_development"], records["speech_sealed_test"], "speech"
    )

    (
        wrapper,
        tokenizer,
        image_processor,
        vision_model,
        speech_processor,
        speech_model,
        device,
        config,
        model_meta,
    ) = load_selected_wrapper(
        checkpoint,
        args.gamma_json,
        args.stage_b_checkpoint,
        args.stage_b_checkpoint_sha256,
    )
    checkpoint_restoration = model_meta.get("checkpoint_restoration")
    if not isinstance(checkpoint_restoration, Mapping):
        raise ValueError("representation evaluator is missing checkpoint_restoration")
    feature_sets: Dict[Tuple[str, str], RetrievalFeatures] = {}
    for modality in ("image", "speech"):
        for role, split_name in (("development", "development"), ("sealed_test", "sealed_test")):
            key = f"{modality}_{role}"
            feature_sets[(modality, role)] = extract_or_load_features(
                wrapper=wrapper,
                tokenizer=tokenizer,
                image_processor=image_processor,
                vision_model=vision_model,
                speech_processor=speech_processor,
                speech_model=speech_model,
                rows=records[key],
                manifest_path=manifests[key],
                modality=modality,
                split_name=split_name,
                device=device,
                batch_size=args.batch_size,
                cache_root=cache_root,
                checkpoint_sha256=checkpoint_sha256,
                gamma_sha256=gamma_sha256,
                code_sha256=code_sha256,
                config=config,
            )

    # Phase 1: all model selection and fitting is completed using development only.
    prepared: Dict[Tuple[str, str], PreparedStage] = {}
    for modality in ("image", "speech"):
        development = feature_sets[(modality, "development")]
        for stage_name in STAGES:
            prepared[(modality, stage_name)] = prepare_stage(development, stage_name)

    # Phase 2: frozen probes are evaluated exactly once on the sealed test split.
    rank_rows: List[Dict[str, Any]] = []
    modality_reports: Dict[str, Any] = {}
    direction_names = {
        "image": {"media_to_text": "image_to_text", "text_to_media": "text_to_image"},
        "speech": {"media_to_text": "speech_to_text", "text_to_media": "text_to_speech"},
    }
    for modality in ("image", "speech"):
        sealed = feature_sets[(modality, "sealed_test")]
        stage_reports: Dict[str, Any] = {}
        for stage_name in STAGES:
            stage_report = evaluate_frozen_stage(
                sealed, stage_name, prepared[(modality, stage_name)], rank_rows
            )
            for method in ("direct_retrieval", "ridge_probe_retrieval"):
                if method in stage_report:
                    for generic, specific in direction_names[modality].items():
                        if generic in stage_report[method]:
                            stage_report[method][specific] = stage_report[method].pop(generic)
            stage_reports[stage_name] = stage_report
        modality_reports[modality] = {
            "target_space": "clip_text"
            if modality == "image"
            else str(config.speech_target_space),
            "encoder_pooling": sealed.pooling,
            "development_media_count": len(feature_sets[(modality, "development")].media_ids),
            "development_target_count": len(feature_sets[(modality, "development")].text_ids),
            "sealed_media_count": len(sealed.media_ids),
            "sealed_target_count": len(sealed.text_ids),
            "stages": stage_reports,
        }

    ranks_path = output_dir / "per_query_ranks.jsonl"
    ranks_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rank_rows),
        encoding="utf-8",
    )
    cache_provenance = {
        f"{modality}_{role}": {
            "path": str(features.cache_path),
            "sha256": features.cache_sha256,
        }
        for (modality, role), features in feature_sets.items()
    }
    report = {
        "protocol": {
            "name": "single_gpu_representation_retention_funnel",
            "selection_policy": "development_only_fixed_split",
            "sealed_metrics_used_for_model_selection": False,
            "representation_diagnostics_status": "development_only",
            "representation_diagnostics_used_for_model_selection": False,
            "sealed_evaluations_per_frozen_method": 1,
            "ridge_values": list(RIDGE_VALUES),
            "validation_fraction": VALIDATION_FRACTION,
            "selection_seed": SELECTION_SEED,
            "overfit_pairs": OVERFIT_PAIRS,
            "overfit_r1_threshold": OVERFIT_R1_THRESHOLD,
        },
        "checkpoint_config": {
            "base_model": config.base_model,
            "vision_model": config.vision_model,
            "speech_model": config.speech_model,
            "speech_target_space": config.speech_target_space,
            "image_prefix_tokens": int(config.image_prefix_tokens),
            "audio_prefix_tokens": int(config.audio_prefix_tokens),
            "encoder_feature_tokens": int(config.encoder_feature_tokens),
            "alignment_prefix_residual": bool(config.alignment_prefix_residual),
            "model_load_meta": model_meta,
        },
        "modalities": modality_reports,
        "provenance": {
            "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
            "stage_b_checkpoint_sha256": checkpoint_restoration.get(
                "source_checkpoint_hashes", {}
            ).get("stage_b"),
            "checkpoint_restoration": checkpoint_restoration,
            "frozen_protocol": {
                "path": str(protocol_manifest),
                "sha256": sha256_file(protocol_manifest),
            },
            "gamma": {
                "path": str(gamma_candidate) if gamma_candidate.is_file() else None,
                "sha256": gamma_sha256,
            },
            "manifests": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in manifests.items()
            },
            "evaluator_code": {"path": str(code_path), "sha256": code_sha256},
            "cached_features": cache_provenance,
            "per_query_ranks": {
                "path": str(ranks_path),
                "sha256": sha256_file(ranks_path),
                "rows": len(rank_rows),
            },
        },
    }
    _json_dump(report_path, report)
    print(json.dumps({"report": str(report_path), "sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
