"""Encoder-only image/speech retrieval baselines for held-out controls.

These metrics are controls, not the main Project 18 shared-prefix result. They
measure how much signal is already present in CLIP image-text space and in a
Whisper audio-encoder / decoder-token-embedding space without passing through
OLMoE, learned prefixes, or sparse routing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from training.olmoe_required_runs import load_encoders, save_json
from training.olmoe_real_subset_runs import (
    FeatureCache,
    absolutize_media_paths,
    clip_text_embeddings,
    load_speech_text_tokenizer,
    load_vision_text_tokenizer,
    read_jsonl,
    speech_text_embeddings,
    split_tail,
)
from scripts.eval_conditional_retrieval import bootstrap_r_at_1_ci, hard_negative_indices, recall_from_ranks


def pooled_encoder_tensor(outputs: Any) -> torch.Tensor:
    if torch.is_tensor(outputs):
        return outputs
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state[:, 0]
    raise TypeError(f"Unsupported encoder output type: {type(outputs)!r}")


def clip_image_embeddings(image_processor, vision_model, rows: Sequence[Dict[str, Any]], device: torch.device, batch_size: int) -> torch.Tensor:
    vectors: List[torch.Tensor] = []
    for start in range(0, len(rows), batch_size):
        sub = list(rows[start:start + batch_size])
        images = [Image.open(str(row["image_path"])).convert("RGB") for row in sub]
        batch = image_processor(images=images, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            if hasattr(vision_model, "get_image_features"):
                feats = pooled_encoder_tensor(vision_model.get_image_features(pixel_values=batch["pixel_values"]))
            else:
                feats = pooled_encoder_tensor(vision_model(**batch))
                projection = getattr(vision_model, "visual_projection", None)
                if projection is not None:
                    feats = projection(feats)
        vectors.append(feats.detach().float().cpu())
    return F.normalize(torch.cat(vectors, dim=0), dim=-1)


def whisper_audio_embeddings(cache: FeatureCache, speech_processor, speech_model, rows: Sequence[Dict[str, Any]], device: torch.device, sample_rate: int, max_tokens: int, batch_size: int) -> torch.Tensor:
    vectors: List[torch.Tensor] = []
    for start in range(0, len(rows), batch_size):
        sub = list(rows[start:start + batch_size])
        features = cache.audio_batch(speech_processor, speech_model, sub, device, sample_rate, max_tokens)
        vectors.append(features.detach().float().mean(dim=1).cpu())
    return F.normalize(torch.cat(vectors, dim=0), dim=-1)


def candidate_indices(total: int, query_idx: int, negatives: int, mode: str, hard_indices: Sequence[Sequence[int]]) -> List[int]:
    if negatives < 0:
        return list(range(total))
    out = [int(query_idx)]
    used = {int(query_idx)}
    if mode == "hard_text" and hard_indices:
        for neg in hard_indices[query_idx]:
            neg_i = int(neg)
            if neg_i in used:
                continue
            used.add(neg_i)
            out.append(neg_i)
            if len(out) >= negatives + 1:
                return out
    if mode == "random":
        generator = torch.Generator()
        generator.manual_seed(104729 + int(query_idx))
        for neg in torch.randperm(max(1, total), generator=generator).tolist():
            neg_i = int(neg)
            if neg_i in used:
                continue
            used.add(neg_i)
            out.append(neg_i)
            if len(out) >= negatives + 1:
                return out
    stride = 37
    offset = 17
    j = 0
    while len(out) < negatives + 1 and len(used) < total:
        neg_i = (int(query_idx) + offset + stride * j) % max(1, total)
        j += 1
        if neg_i in used:
            continue
        used.add(neg_i)
        out.append(neg_i)
    return out


def rank_gold(score_vector: torch.Tensor, gold_local_idx: int) -> int:
    order = torch.argsort(score_vector.float(), descending=True).tolist()
    return int(order.index(int(gold_local_idx)))


def evaluate_pair(left: torch.Tensor, right: torch.Tensor, query_indices: Sequence[int], negatives: int, mode: str, hard_indices: Sequence[Sequence[int]]) -> tuple[List[int], List[Dict[str, Any]]]:
    ranks: List[int] = []
    rows: List[Dict[str, Any]] = []
    total = int(right.shape[0])
    for query_idx in query_indices:
        cand = candidate_indices(total, int(query_idx), int(negatives), mode, hard_indices)
        scores = left[int(query_idx)].float() @ right[cand].float().T
        gold_local = cand.index(int(query_idx))
        rank = rank_gold(scores, gold_local)
        ranks.append(rank)
        rows.append({"query_index": int(query_idx), "rank": int(rank), "candidate_count": len(cand), "gold_candidate_index": int(gold_local)})
    return ranks, rows


def query_window(total: int, query_offset: int, candidate_offset: int, queries: int, candidates: int, local_mode: bool) -> tuple[List[int], int, int]:
    if local_mode:
        start = max(0, min(int(query_offset), total))
        end = min(total, start + int(queries))
        return list(range(start, end)), 0, total
    cand_start_value = int(query_offset) if int(candidate_offset) < 0 else int(candidate_offset)
    start = max(0, min(cand_start_value, total))
    end = min(total, start + max(1, int(candidates)))
    query_start = max(0, int(query_offset))
    query_end = min(total, query_start + int(queries))
    queries_global = [idx for idx in range(query_start, query_end) if start <= idx < end]
    return queries_global, start, end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/real_subset_clean_260708b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vision-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--speech-model", default="openai/whisper-base.en")
    parser.add_argument("--feature-cache-dir", default="")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--encoder-feature-tokens", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-eval-samples", type=int, default=250)
    parser.add_argument("--speech-eval-samples", type=int, default=250)
    parser.add_argument("--conditional-queries", type=int, default=250)
    parser.add_argument("--conditional-candidates", type=int, default=250)
    parser.add_argument("--conditional-negatives", type=int, default=-1)
    parser.add_argument("--negative-mode", choices=["stride", "random", "hard_text"], default="stride")
    parser.add_argument("--eval-split-name", default="eval_tail")
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--candidate-offset", type=int, default=-1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--per-query-output", default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_rows = read_jsonl(data_dir / "image_captions.jsonl")
    speech_rows = read_jsonl(data_dir / "speech_transcripts.jsonl")
    _, image_eval = split_tail(image_rows, args.image_eval_samples)
    _, speech_eval = split_tail(speech_rows, args.speech_eval_samples)
    image_eval = list(image_eval)
    speech_eval = list(speech_eval)
    absolutize_media_paths(image_eval, data_dir)
    absolutize_media_paths(speech_eval, data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor, vision_model, speech_processor, speech_model = load_encoders(args.vision_model, args.speech_model, device)
    cache_root = Path(args.feature_cache_dir) if args.feature_cache_dir else Path(args.output).parent / "feature_cache"
    cache = FeatureCache(cache_root)

    vision_text_tokenizer = load_vision_text_tokenizer(args.vision_model)
    speech_text_tokenizer = load_speech_text_tokenizer(args.speech_model)
    image_text = [str(row["caption"]) for row in image_eval]
    speech_text = [str(row["transcript"]) for row in speech_eval]

    image_left = clip_image_embeddings(image_processor, vision_model, image_eval, device, args.batch_size)
    image_right = clip_text_embeddings(vision_model, vision_text_tokenizer, image_text, device, args.batch_size)
    speech_left = whisper_audio_embeddings(cache, speech_processor, speech_model, speech_eval, device, args.sample_rate, args.encoder_feature_tokens, args.batch_size)
    speech_right = speech_text_embeddings(speech_text_tokenizer, speech_model, speech_text, device, args.batch_size)
    if image_left.shape[-1] != image_right.shape[-1]:
        raise RuntimeError(f"CLIP image/text dim mismatch: {tuple(image_left.shape)} vs {tuple(image_right.shape)}")
    if speech_left.shape[-1] != speech_right.shape[-1]:
        raise RuntimeError(f"Speech audio/text dim mismatch: {tuple(speech_left.shape)} vs {tuple(speech_right.shape)}")

    local_mode = int(args.conditional_negatives) >= 0
    image_queries, image_candidate_start, image_candidate_end = query_window(len(image_eval), args.query_offset, args.candidate_offset, args.conditional_queries, args.conditional_candidates, local_mode)
    speech_queries, speech_candidate_start, speech_candidate_end = query_window(len(speech_eval), args.query_offset, args.candidate_offset, args.conditional_queries, args.conditional_candidates, local_mode)
    if not image_queries or not speech_queries:
        raise RuntimeError(
            "empty encoder-baseline query window: "
            f"image_queries={len(image_queries)} speech_queries={len(speech_queries)} "
            f"query_offset={args.query_offset} candidate_offset={args.candidate_offset} "
            f"conditional_candidates={args.conditional_candidates}"
        )
    if not local_mode:
        image_left_eval = image_left[image_candidate_start:image_candidate_end]
        image_right_eval = image_right[image_candidate_start:image_candidate_end]
        speech_left_eval = speech_left[speech_candidate_start:speech_candidate_end]
        speech_right_eval = speech_right[speech_candidate_start:speech_candidate_end]
        image_queries_eval = [idx - image_candidate_start for idx in image_queries]
        speech_queries_eval = [idx - speech_candidate_start for idx in speech_queries]
    else:
        image_left_eval, image_right_eval = image_left, image_right
        speech_left_eval, speech_right_eval = speech_left, speech_right
        image_queries_eval, speech_queries_eval = image_queries, speech_queries

    image_hard = hard_negative_indices(image_right_eval, int(args.conditional_negatives)) if args.negative_mode == "hard_text" and local_mode else [[] for _ in range(image_right_eval.shape[0])]
    speech_hard = hard_negative_indices(speech_right_eval, int(args.conditional_negatives)) if args.negative_mode == "hard_text" and local_mode else [[] for _ in range(speech_right_eval.shape[0])]
    image_ranks, image_rows_out = evaluate_pair(image_left_eval, image_right_eval, image_queries_eval, args.conditional_negatives, args.negative_mode, image_hard)
    speech_ranks, speech_rows_out = evaluate_pair(speech_left_eval, speech_right_eval, speech_queries_eval, args.conditional_negatives, args.negative_mode, speech_hard)

    image_candidate_count = int(args.conditional_negatives) + 1 if local_mode else int(image_right_eval.shape[0])
    speech_candidate_count = int(args.conditional_negatives) + 1 if local_mode else int(speech_right_eval.shape[0])
    metrics: Dict[str, Any] = {
        "mode": "encoder_baseline_local_negatives" if local_mode else "encoder_baseline_full_matrix",
        "eval_path": "encoder_baseline",
        "image_encoder_baseline_path": "clip_image_text_projection",
        "speech_encoder_baseline_path": "whisper_audio_encoder_to_decoder_token_embedding",
        "conditional_uses_lm_logits": False,
        "conditional_uses_direct_encoder_pooling": True,
        "conditional_uses_multimodal_prefix": False,
        "retrieval_uses_direct_encoder_pooling": True,
        "uses_olmoe_lm": False,
        "prefix_control": "encoder_only",
        "negative_mode": str(args.negative_mode),
        "eval_split_name": str(args.eval_split_name),
        "query_offset": int(args.query_offset),
        "candidate_offset": int(args.candidate_offset),
        "image_eval_tail_count": len(image_eval),
        "speech_eval_tail_count": len(speech_eval),
        "image_candidate_start": int(image_candidate_start),
        "image_candidate_end_exclusive": int(image_candidate_end),
        "speech_candidate_start": int(speech_candidate_start),
        "speech_candidate_end_exclusive": int(speech_candidate_end),
        "image_eval_count": len(image_queries),
        "speech_eval_count": len(speech_queries),
        "image_query_start": int(image_queries[0]) if image_queries else None,
        "image_query_end_exclusive": int(image_queries[-1] + 1) if image_queries else None,
        "speech_query_start": int(speech_queries[0]) if speech_queries else None,
        "speech_query_end_exclusive": int(speech_queries[-1] + 1) if speech_queries else None,
        "candidate_count": image_candidate_count,
        "speech_candidate_count": speech_candidate_count,
        "image_chance_r_at_1": 1.0 / max(1, image_candidate_count),
        "speech_chance_r_at_1": 1.0 / max(1, speech_candidate_count),
        **recall_from_ranks(image_ranks, "image_to_text"),
        **recall_from_ranks(speech_ranks, "speech_to_text"),
        **{f"image_to_text_{k}": v for k, v in bootstrap_r_at_1_ci(image_ranks, args.bootstrap_samples, args.bootstrap_seed).items()},
        **{f"speech_to_text_{k}": v for k, v in bootstrap_r_at_1_ci(speech_ranks, args.bootstrap_samples, args.bootstrap_seed + 1).items()},
    }
    metrics.update({
        "conditional_candidates_per_query": image_candidate_count,
        "conditional_speech_candidates_per_query": speech_candidate_count,
        "conditional_image_eval_count": len(image_queries),
        "conditional_speech_eval_count": len(speech_queries),
        "conditional_image_chance_r_at_1": metrics["image_chance_r_at_1"],
        "conditional_speech_chance_r_at_1": metrics["speech_chance_r_at_1"],
        "conditional_image_to_text_r_at_1": metrics.get("image_to_text_r_at_1"),
        "conditional_image_to_text_r_at_5": metrics.get("image_to_text_r_at_5"),
        "conditional_image_to_text_r_at_10": metrics.get("image_to_text_r_at_10"),
        "conditional_speech_to_text_r_at_1": metrics.get("speech_to_text_r_at_1"),
        "conditional_speech_to_text_r_at_5": metrics.get("speech_to_text_r_at_5"),
        "conditional_speech_to_text_r_at_10": metrics.get("speech_to_text_r_at_10"),
        "conditional_image_to_text_r_at_1_bootstrap_ci_low": metrics.get("image_to_text_r_at_1_bootstrap_ci_low"),
        "conditional_image_to_text_r_at_1_bootstrap_ci_high": metrics.get("image_to_text_r_at_1_bootstrap_ci_high"),
        "conditional_speech_to_text_r_at_1_bootstrap_ci_low": metrics.get("speech_to_text_r_at_1_bootstrap_ci_low"),
        "conditional_speech_to_text_r_at_1_bootstrap_ci_high": metrics.get("speech_to_text_r_at_1_bootstrap_ci_high"),
    })
    save_json(Path(args.output), metrics)
    if args.per_query_output:
        per_query_path = Path(args.per_query_output)
        per_query_path.parent.mkdir(parents=True, exist_ok=True)
        with per_query_path.open("w", encoding="utf-8") as handle:
            for row in image_rows_out:
                row = {**row, "modality": "image", "query_index": row["query_index"] + (0 if local_mode else image_candidate_start)}
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            for row in speech_rows_out:
                row = {**row, "modality": "speech", "query_index": row["query_index"] + (0 if local_mode else speech_candidate_start)}
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
