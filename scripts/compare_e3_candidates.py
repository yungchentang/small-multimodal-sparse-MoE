"""Compare clean E3 candidate roots and write table/figure artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def as_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def fmt(value: Any) -> str:
    num = as_float(value)
    if num is None:
        return ""
    if abs(num) >= 100:
        return f"{num:.1f}"
    if abs(num) >= 10:
        return f"{num:.2f}"
    return f"{num:.3f}"


def pick_metric(primary: Dict[str, Any], fallback: Dict[str, Any], key: str) -> Any:
    value = primary.get(key)
    if value is not None and value != "":
        return value
    return fallback.get(key)


def load_root(root: Path) -> Dict[str, Any]:
    summary = read_json(root / "summary.json")
    e3 = summary.get("E3") if isinstance(summary.get("E3"), dict) else read_json(root / "E3_final_multimodal_top2" / "metrics.json")
    text_eval = e3.get("text_eval", {}) if isinstance(e3.get("text_eval"), dict) else {}
    retrieval = e3.get("retrieval_eval", {}) if isinstance(e3.get("retrieval_eval"), dict) else {}
    meta = e3.get("meta", {}) if isinstance(e3.get("meta"), dict) else {}
    manifest = summary.get("manifest", {}) if isinstance(summary.get("manifest"), dict) else read_json(root / "manifest.json")
    args = manifest.get("args", {}) if isinstance(manifest.get("args"), dict) else {}
    cond5 = read_json(root / "conditional_eval_5way" / "metrics.json")
    cond10 = read_json(root / "conditional_eval_10way" / "metrics.json")
    e3_cond_candidates = retrieval.get("conditional_candidates_per_query")
    row = {
        "candidate": root.name,
        "output_root": str(root),
        "job": manifest.get("runai_job_name", ""),
        "steps": len(e3.get("steps") or []),
        "capacity_factor": meta.get("capacity_factor"),
        "aux_coef": meta.get("aux_coef"),
        "learning_rate": args.get("learning_rate"),
        "lm_head_lr": args.get("lm_head_learning_rate"),
        "retrieval_lr": args.get("retrieval_head_learning_rate"),
        "contrastive_coef": args.get("contrastive_coef"),
        "image_contrastive_coef": args.get("image_contrastive_coef"),
        "speech_contrastive_coef": args.get("speech_contrastive_coef"),
        "modality_cycle": args.get("modality_cycle"),
        "image_prefix_tokens": args.get("image_prefix_tokens"),
        "audio_prefix_tokens": args.get("audio_prefix_tokens"),
        "encoder_feature_tokens": args.get("encoder_feature_tokens"),
        "train_router_gates": meta.get("train_router_gates"),
        "train_lm_head": meta.get("train_lm_head"),
        "router_z_loss_coef": args.get("router_z_loss_coef"),
        "expert_dropout_prob": args.get("expert_dropout_prob"),
        "dynamic_expert_bias_lr": args.get("dynamic_expert_bias_lr"),
        "dynamic_expert_bias_abs_max": (e3.get("final_dynamic_expert_bias") or {}).get("dynamic_expert_bias_abs_max"),
        "conditional_ranking_negative_mode": args.get("conditional_ranking_negative_mode"),
        "conditional_ranking_negatives": args.get("conditional_ranking_negatives"),
        "conditional_ranking_hard_pool_size": args.get("conditional_ranking_hard_pool_size"),
        "text_ppl": text_eval.get("perplexity"),
        "text_acc": text_eval.get("next_token_accuracy"),
        "image_emb_r1": retrieval.get("image_to_text_r_at_1"),
        "speech_emb_r1": retrieval.get("speech_to_text_r_at_1"),
        "cond_candidates_in_e3": e3_cond_candidates,
        "cond_image_r1_in_e3": retrieval.get("conditional_image_to_text_r_at_1"),
        "cond_speech_r1_in_e3": retrieval.get("conditional_speech_to_text_r_at_1"),
        "cond5_image_r1": pick_metric(cond5, retrieval if e3_cond_candidates == 5 else {}, "conditional_image_to_text_r_at_1"),
        "cond5_speech_r1": pick_metric(cond5, retrieval if e3_cond_candidates == 5 else {}, "conditional_speech_to_text_r_at_1"),
        "cond5_image_chance": pick_metric(cond5, retrieval if e3_cond_candidates == 5 else {}, "conditional_image_chance_r_at_1"),
        "cond5_speech_chance": pick_metric(cond5, retrieval if e3_cond_candidates == 5 else {}, "conditional_speech_chance_r_at_1"),
        "cond10_image_r1": pick_metric(cond10, retrieval if e3_cond_candidates == 10 else {}, "conditional_image_to_text_r_at_1"),
        "cond10_speech_r1": pick_metric(cond10, retrieval if e3_cond_candidates == 10 else {}, "conditional_speech_to_text_r_at_1"),
        "cond10_image_chance": pick_metric(cond10, retrieval if e3_cond_candidates == 10 else {}, "conditional_image_chance_r_at_1"),
        "cond10_speech_chance": pick_metric(cond10, retrieval if e3_cond_candidates == 10 else {}, "conditional_speech_chance_r_at_1"),
        "inactive": e3.get("final_inactive_expert_ratio_mean"),
        "overflow": e3.get("final_capacity_overflow_ratio_mean"),
        "checkpoint": e3.get("checkpoint_path", ""),
    }
    ppl = as_float(row.get("text_ppl"))
    c5i = as_float(row.get("cond5_image_r1"))
    c5s = as_float(row.get("cond5_speech_r1"))
    c5ic = as_float(row.get("cond5_image_chance")) or 0.2
    c5sc = as_float(row.get("cond5_speech_chance")) or 0.2
    c10i = as_float(row.get("cond10_image_r1"))
    c10s = as_float(row.get("cond10_speech_r1"))
    c10ic = as_float(row.get("cond10_image_chance")) or 0.1
    c10sc = as_float(row.get("cond10_speech_chance")) or 0.1
    row["target_text_ppl"] = bool(ppl is not None and ppl <= 20.0)
    row["target_5way_image"] = bool(c5i is not None and c5i >= 0.30 and c5i >= c5ic + 0.10)
    row["target_5way_speech"] = bool(c5s is not None and c5s >= 0.25 and c5s >= c5sc + 0.05)
    row["target_10way"] = bool(c10i is not None and c10s is not None and c10i > c10ic and c10s >= c10sc)
    row["core_target_pass"] = bool(row["target_text_ppl"] and row["target_5way_image"] and row["target_5way_speech"])
    return row


def score(row: Dict[str, Any]) -> float:
    ppl = as_float(row.get("text_ppl"))
    cond_img = as_float(row.get("cond10_image_r1")) or as_float(row.get("cond_image_r1_in_e3")) or 0.0
    cond_sp = as_float(row.get("cond10_speech_r1")) or as_float(row.get("cond_speech_r1_in_e3")) or 0.0
    emb_img = as_float(row.get("image_emb_r1")) or 0.0
    emb_sp = as_float(row.get("speech_emb_r1")) or 0.0
    overflow = as_float(row.get("overflow")) or 0.0
    ppl_term = max(0.0, min(1.0, (30.0 - (ppl or 30.0)) / 20.0))
    base = 0.30 * ppl_term + 0.25 * cond_img + 0.30 * cond_sp + 0.05 * emb_img + 0.05 * emb_sp - 0.05 * overflow
    if not row.get("target_5way_image"):
        base -= 0.12
    if not row.get("target_5way_speech"):
        base -= 0.12
    if not row.get("target_text_ppl"):
        base -= 0.08
    return base


def write_svg(rows: List[Dict[str, Any]], path: Path) -> None:
    metrics = ["text_ppl", "cond10_image_r1", "cond10_speech_r1", "image_emb_r1", "speech_emb_r1", "overflow"]
    width = 1180
    row_h = 32
    left = 230
    top = 70
    group_gap = 16
    height = top + len(rows) * row_h * len(metrics) + (len(metrics) + 1) * group_gap + 60
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append('<text x="24" y="32" font-family="Arial" font-size="22" font-weight="700">Clean E3 Candidate Comparison</text>')
    colors = {"text_ppl": "#4c78a8", "cond10_image_r1": "#59a14f", "cond10_speech_r1": "#f28e2b", "image_emb_r1": "#b07aa1", "speech_emb_r1": "#e15759", "overflow": "#9c755f"}
    y = top
    for metric in metrics:
        values = [as_float(r.get(metric)) for r in rows]
        finite = [v for v in values if v is not None]
        max_v = max(finite) if finite else 1.0
        if metric == "text_ppl":
            max_v = max(max_v, 20.0)
        max_v = max(max_v, 1e-9)
        parts.append(f'<text x="24" y="{y - 10}" font-family="Arial" font-size="15" font-weight="700">{html.escape(metric)}</text>')
        for idx, row in enumerate(rows):
            yy = y + idx * row_h
            name = html.escape(str(row.get("candidate", ""))[:32])
            value = as_float(row.get(metric))
            bar_w = 0 if value is None else max(1, int((value / max_v) * 760))
            parts.append(f'<text x="24" y="{yy + 20}" font-family="Arial" font-size="12">{name}</text>')
            parts.append(f'<rect x="{left}" y="{yy + 7}" width="{bar_w}" height="18" fill="{colors[metric]}" opacity="0.82"/>')
            parts.append(f'<text x="{left + bar_w + 8}" y="{yy + 21}" font-family="Arial" font-size="12">{fmt(value)}</text>')
        y += len(rows) * row_h + group_gap
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="*", help="Candidate output roots")
    parser.add_argument(
        "--root-csv",
        action="append",
        default=[],
        help="Existing candidate_comparison.csv file(s) whose output_root column should be included",
    )
    parser.add_argument("--output-dir", default="autoresearch/iterate-260708-2257")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = list(args.roots)
    for csv_path in args.root_csv:
        path = Path(csv_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                root = row.get("output_root") or row.get("candidate")
                if root:
                    roots.append(root)
    deduped_roots = []
    seen = set()
    for root in roots:
        key = str(Path(root))
        if key in seen:
            continue
        seen.add(key)
        deduped_roots.append(root)
    if not deduped_roots:
        raise SystemExit("No candidate roots were provided")
    rows = [load_root(Path(root)) for root in deduped_roots]
    rows.sort(key=score, reverse=True)
    for row in rows:
        row["selection_score"] = score(row)
    fields = [
        "candidate", "job", "steps", "capacity_factor", "aux_coef", "learning_rate", "lm_head_lr", "retrieval_lr",
        "contrastive_coef", "image_contrastive_coef", "speech_contrastive_coef", "modality_cycle", "image_prefix_tokens",
        "audio_prefix_tokens", "encoder_feature_tokens", "train_router_gates", "train_lm_head", "router_z_loss_coef",
        "expert_dropout_prob", "dynamic_expert_bias_lr", "dynamic_expert_bias_abs_max",
        "conditional_ranking_negative_mode", "conditional_ranking_negatives", "conditional_ranking_hard_pool_size", "text_ppl", "text_acc",
        "cond_candidates_in_e3", "cond_image_r1_in_e3", "cond_speech_r1_in_e3", "cond5_image_r1", "cond5_speech_r1",
        "cond10_image_r1", "cond10_speech_r1", "image_emb_r1", "speech_emb_r1", "inactive", "overflow",
        "target_text_ppl", "target_5way_image", "target_5way_speech", "target_10way", "core_target_pass", "selection_score",
        "checkpoint", "output_root",
    ]
    csv_path = out_dir / "candidate_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    md_path = out_dir / "candidate_comparison.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Clean E3 Candidate Comparison\n\n")
        handle.write("| candidate | pass | cap | aux | dyn bias lr | rank mode | rank negs | steps | PPL | 5w img | 5w speech | 10w img | 10w speech | emb img | emb speech | inactive | overflow | score |\n")
        handle.write("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            target = "yes" if row.get("core_target_pass") else "no"
            handle.write(
                f"| {row.get('candidate')} | {target} | {fmt(row.get('capacity_factor'))} | {fmt(row.get('aux_coef'))} | "
                f"{fmt(row.get('dynamic_expert_bias_lr'))} | {row.get('conditional_ranking_negative_mode') or ''} | "
                f"{row.get('conditional_ranking_negatives') or ''} | {row.get('steps')} | "
                f"{fmt(row.get('text_ppl'))} | {fmt(row.get('cond5_image_r1'))} | {fmt(row.get('cond5_speech_r1'))} | "
                f"{fmt(row.get('cond10_image_r1'))} | {fmt(row.get('cond10_speech_r1'))} | "
                f"{fmt(row.get('image_emb_r1'))} | {fmt(row.get('speech_emb_r1'))} | {fmt(row.get('inactive'))} | "
                f"{fmt(row.get('overflow'))} | {fmt(row.get('selection_score'))} |\n"
            )
    write_svg(rows, out_dir / "candidate_comparison.svg")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "svg": str(out_dir / "candidate_comparison.svg"), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
