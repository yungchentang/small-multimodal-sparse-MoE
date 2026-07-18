"""Summarize real ACDL Project 18 OLMoE required runs.

Dependency-free by design: reads ``summary.json`` and emits CSV/Markdown/SVG
artifacts that can be regenerated after ``bash run.sh real-required-runs``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


EXPERIMENT_LABELS = {
    "E0": "Top-8 teacher baseline",
    "E1": "Hard Top-2",
    "E2": "Calibrated Top-2",
    "E3": "Final multimodal Top-2",
    "E4": "No-aux load-balancing ablation",
    "E5": "Capacity-factor ablation",
    "E6": "Trainable-experts ablation",
}


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize(values: Iterable[float]) -> List[float]:
    values = [float(v) for v in values]
    total = sum(values)
    if total <= 0.0:
        return [0.0 for _ in values]
    return [v / total for v in values]


def write_svg_line(path: Path, series: Dict[str, List[float]], title: str) -> None:
    width, height = 900, 400
    values = [v for seq in series.values() for v in seq]
    if not values:
        values = [0.0]
    min_v, max_v = min(values), max(values)
    span = max(max_v - min_v, 1e-9)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    body = [f'<text x="30" y="30" font-size="18" font-family="Arial">{title}</text>']
    for idx, (name, seq) in enumerate(series.items()):
        if not seq:
            continue
        points = []
        for i, value in enumerate(seq):
            x = 70 + i * (690 / max(1, len(seq) - 1))
            y = 325 - 255 * ((value - min_v) / span)
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[idx % len(colors)]
        body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text x="665" y="{58 + 22 * idx}" font-size="13" font-family="Arial" fill="{color}">{name}</text>')
    body.append('<line x1="60" y1="325" x2="820" y2="325" stroke="#333"/>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' + "".join(body) + "</svg>", encoding="utf-8")


def write_svg_bar(path: Path, values: Dict[str, float], title: str) -> None:
    width, height = 900, 400
    items = list(values.items())
    max_v = max([abs(v) for _, v in items] or [1.0])
    bar_w = 680 / max(1, len(items))
    body = [f'<text x="30" y="30" font-size="18" font-family="Arial">{title}</text>']
    for i, (name, value) in enumerate(items):
        h = 250 * (abs(value) / max_v if max_v else 0.0)
        x = 75 + i * bar_w
        y = 325 - h
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(7, bar_w - 8):.1f}" height="{h:.1f}" fill="#2563eb"/>')
        body.append(f'<text x="{x:.1f}" y="350" font-size="10" font-family="Arial">{name}</text>')
    body.append('<line x1="60" y1="325" x2="820" y2="325" stroke="#333"/>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' + "".join(body) + "</svg>", encoding="utf-8")


def write_svg_architecture(path: Path) -> None:
    width, height = 1000, 520
    boxes = [
        (40, 90, 190, 70, "CLIP/ViT", "image encoder"),
        (40, 220, 190, 70, "Whisper/Wav2Vec2", "speech encoder"),
        (40, 350, 190, 70, "OLMoE tokenizer", "text blocks"),
        (300, 90, 190, 70, "image query", "resampler"),
        (300, 220, 190, 70, "audio query", "resampler"),
        (560, 155, 180, 110, "projection +", "concat fusion"),
        (800, 155, 170, 110, "OLMoE Top-2", "Sparse MoE"),
        (800, 330, 170, 70, "LM / retrieval", "outputs"),
    ]
    body = ['<rect width="100%" height="100%" fill="white"/>', '<text x="35" y="38" font-size="22" font-family="Arial">Architecture: shared multimodal Top-2 OLMoE prefix path</text>']
    for x, y, w, h, line1, line2 in boxes:
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#eef2ff" stroke="#1f2937" stroke-width="2"/>')
        body.append(f'<text x="{x + 14}" y="{y + 31}" font-size="17" font-family="Arial" fill="#111827">{line1}</text>')
        body.append(f'<text x="{x + 14}" y="{y + 55}" font-size="14" font-family="Arial" fill="#374151">{line2}</text>')
    arrows = [
        (230, 125, 300, 125), (230, 255, 300, 255), (230, 385, 560, 240),
        (490, 125, 560, 190), (490, 255, 560, 230), (740, 210, 800, 210), (885, 265, 885, 330),
    ]
    for x1, y1, x2, y2 in arrows:
        body.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#2563eb" stroke-width="3" marker-end="url(#arrow)"/>')
    body.insert(1, '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#2563eb"/></marker></defs>')
    body.append('<text x="35" y="480" font-size="14" font-family="Arial" fill="#374151">Image/audio prefixes and text tokens share the same OLMoE router; logs report prefix-token expert utilization.</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' + "".join(body) + "</svg>", encoding="utf-8")


def experiment_keys(summary: Dict[str, Any]) -> List[str]:
    return [key for key in ["E0", "E1", "E2", "E3", "E4", "E5", "E6"] if isinstance(summary.get(key), dict)]


def compact_row(key: str, data: Dict[str, Any]) -> Dict[str, Any]:
    steps = data.get("steps") if isinstance(data.get("steps"), list) else []
    final_step = steps[-1] if steps else {}
    meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}
    text_eval = data.get("text_eval", {}) if isinstance(data.get("text_eval"), dict) else data
    retrieval = data.get("retrieval_eval", {}) if isinstance(data.get("retrieval_eval"), dict) else {}
    source = final_step if final_step else {**meta, **data}
    return {
        "experiment": key,
        "label": EXPERIMENT_LABELS.get(key, key),
        "top_k": source.get("top_k", meta.get("top_k", data.get("top_k", ""))),
        "capacity_factor": source.get("capacity_factor", meta.get("capacity_factor", data.get("capacity_factor", ""))),
        "aux_coef": source.get("aux_coef", meta.get("aux_coef", data.get("aux_coef", ""))),
        "train_router_gates": source.get("train_router_gates", meta.get("train_router_gates", "")),
        "train_experts": source.get("train_experts", meta.get("train_experts", "")),
        "trainable_params": source.get("trainable_params", meta.get("trainable_params", "")),
        "steps": len(steps),
        "loss_start_or_eval": data.get("loss", data.get("first_loss", "")),
        "loss_last": data.get("last_loss", ""),
        "loss_min": data.get("min_loss", ""),
        "ce_loss_start": steps[0].get("ce_loss", "") if steps else data.get("loss", ""),
        "ce_loss_last": steps[-1].get("ce_loss", "") if steps else "",
        "ce_loss_min": min((as_float(row.get("ce_loss")) for row in steps), default="") if steps else "",
        "text_perplexity": text_eval.get("perplexity", data.get("perplexity", "")),
        "text_next_token_accuracy": text_eval.get("next_token_accuracy", data.get("next_token_accuracy", "")),
        "image_to_text_r_at_1": retrieval.get("image_to_text_r_at_1", ""),
        "image_to_text_r_at_5": retrieval.get("image_to_text_r_at_5", ""),
        "image_to_text_r_at_10": retrieval.get("image_to_text_r_at_10", ""),
        "text_to_image_r_at_1": retrieval.get("text_to_image_r_at_1", ""),
        "speech_to_text_r_at_1": retrieval.get("speech_to_text_r_at_1", ""),
        "speech_to_text_r_at_5": retrieval.get("speech_to_text_r_at_5", ""),
        "speech_to_text_r_at_10": retrieval.get("speech_to_text_r_at_10", ""),
        "text_to_speech_r_at_1": retrieval.get("text_to_speech_r_at_1", ""),
        "conditional_image_to_text_r_at_1": retrieval.get("conditional_image_to_text_r_at_1", ""),
        "conditional_speech_to_text_r_at_1": retrieval.get("conditional_speech_to_text_r_at_1", ""),
        "conditional_candidates_per_query": retrieval.get("conditional_candidates_per_query", ""),
        "gate_entropy_final": source.get("gate_entropy_mean", data.get("gate_entropy_mean", data.get("final_gate_entropy_mean", ""))),
        "inactive_expert_ratio_final": source.get("inactive_expert_ratio_mean", data.get("inactive_expert_ratio_mean", data.get("final_inactive_expert_ratio_mean", ""))),
        "capacity_overflow_ratio_final": source.get("capacity_overflow_ratio_mean", data.get("capacity_overflow_ratio_mean", data.get("final_capacity_overflow_ratio_mean", ""))),
        "cuda_memory_reserved_gb": source.get("cuda_memory_reserved_gb", data.get("cuda_memory_reserved_gb", "")),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/real_required_runs")
    parser.add_argument("--report-dir", default="report")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing required summary: {summary_path}")
    summary = read_json(summary_path)
    manifest = summary.get("manifest", {}) if isinstance(summary.get("manifest"), dict) else {}
    data_manifest = manifest.get("data_manifest", {}) if isinstance(manifest.get("data_manifest"), dict) else {}
    counts = data_manifest.get("counts", {}) if isinstance(data_manifest.get("counts"), dict) else {}
    block_counts = data_manifest.get("block_counts", {}) if isinstance(data_manifest.get("block_counts"), dict) else {}

    rows = [compact_row(key, summary[key]) for key in experiment_keys(summary)]
    csv_path = out_dir / "real_required_runs_table.csv"
    write_csv(csv_path, rows)

    count_rows = [{"name": key, "value": value} for key, value in sorted(counts.items())]
    count_rows.extend({"name": f"block_{key}", "value": value} for key, value in sorted(block_counts.items()) if not isinstance(value, dict))
    dataset_csv = out_dir / "dataset_counts.csv"
    write_csv(dataset_csv, count_rows)

    write_svg_architecture(figures_dir / "fig_architecture.svg")

    train_series = {}
    entropy_series = {}
    overflow_series = {}
    for key in experiment_keys(summary):
        steps = summary[key].get("steps", []) if isinstance(summary[key].get("steps"), list) else []
        if steps:
            label = EXPERIMENT_LABELS.get(key, key)
            train_series[label] = [as_float(row.get("ce_loss", row.get("loss"))) for row in steps]
            entropy_series[label] = [as_float(row.get("gate_entropy_mean")) for row in steps]
            overflow_series[label] = [as_float(row.get("capacity_overflow_ratio_mean")) for row in steps]
    write_svg_line(figures_dir / "fig_real_training_loss.svg", train_series, "Real-data CE loss by training step")
    write_svg_line(figures_dir / "fig_real_gate_entropy.svg", entropy_series, "Real-data gate entropy")
    write_svg_line(figures_dir / "fig_real_capacity_overflow.svg", overflow_series, "Capacity overflow ratio")
    write_svg_bar(figures_dir / "fig_top2_text_perplexity.svg", {row["experiment"]: as_float(row.get("text_perplexity")) for row in rows if row.get("text_perplexity") != ""}, "Text perplexity")
    retrieval_values = {}
    for row in rows:
        if row.get("image_to_text_r_at_1") != "":
            retrieval_values[f"{row['experiment']} embImgR1"] = as_float(row.get("image_to_text_r_at_1"))
            retrieval_values[f"{row['experiment']} embSpR1"] = as_float(row.get("speech_to_text_r_at_1"))
            if row.get("conditional_image_to_text_r_at_1") != "":
                retrieval_values[f"{row['experiment']} condImgR1"] = as_float(row.get("conditional_image_to_text_r_at_1"))
                retrieval_values[f"{row['experiment']} condSpR1"] = as_float(row.get("conditional_speech_to_text_r_at_1"))
    write_svg_bar(figures_dir / "fig_retrieval_r1.svg", retrieval_values, "Retrieval Recall@1")
    e3 = summary.get("E3", {}) if isinstance(summary.get("E3"), dict) else {}
    e3_steps = e3.get("steps", []) if isinstance(e3.get("steps"), list) else []
    final_counts = e3_steps[-1].get("expert_counts_total", []) if e3_steps else []
    utilization = normalize(final_counts)
    top_values = {str(i): value for i, value in sorted(enumerate(utilization), key=lambda item: item[1], reverse=True)[:16]}
    write_svg_bar(figures_dir / "fig_final_expert_utilization_top16.svg", top_values, "Final E3 expert utilization top-16")

    md_lines = [
        "# Real OLMoE Required Run Summary",
        "",
        f"Output directory: `{out_dir}`",
        f"Run:AI job: `{manifest.get('runai_job_name', '')}` / project `{manifest.get('runai_project', '')}`",
        f"Base model: `{manifest.get('base_model', '')}`",
        f"Vision model: `{manifest.get('vision_model', '')}`",
        f"Speech model: `{manifest.get('speech_model', '')}`",
        "",
        "## Dataset Counts",
        "",
        "| Name | Value |",
        "|---|---:|",
    ]
    for row in count_rows:
        md_lines.append(f"| {row['name']} | {row['value']} |")
    md_lines.extend([
        "",
        "## Experiment Metrics",
        "",
        "| Exp | Label | Top-k | C | Aux | Steps | CE start | CE last | PPL | Acc | Emb Img R@1 | Emb Sp R@1 | Cond Img R@1 | Cond Sp R@1 | Entropy | Inactive | Overflow |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        md_lines.append(
            "| {experiment} | {label} | {top_k} | {capacity_factor} | {aux_coef} | {steps} | {ce_loss_start} | {ce_loss_last} | {text_perplexity} | {text_next_token_accuracy} | {image_to_text_r_at_1} | {speech_to_text_r_at_1} | {conditional_image_to_text_r_at_1} | {conditional_speech_to_text_r_at_1} | {gate_entropy_final} | {inactive_expert_ratio_final} | {capacity_overflow_ratio_final} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    md_lines.extend([
        "",
        "## Generated Artifacts",
        "",
        f"- `{csv_path}`",
        f"- `{dataset_csv}`",
        f"- `{figures_dir / 'fig_architecture.svg'}`",
        f"- `{figures_dir / 'fig_real_training_loss.svg'}`",
        f"- `{figures_dir / 'fig_real_gate_entropy.svg'}`",
        f"- `{figures_dir / 'fig_real_capacity_overflow.svg'}`",
        f"- `{figures_dir / 'fig_top2_text_perplexity.svg'}`",
        f"- `{figures_dir / 'fig_retrieval_r1.svg'}`",
        f"- `{figures_dir / 'fig_final_expert_utilization_top16.svg'}`",
    ])
    report_path = report_dir / "real_required_runs_report.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "dataset_counts": str(dataset_csv), "report": str(report_path), "figures": sorted(p.name for p in figures_dir.glob("*.svg"))}, sort_keys=True))


if __name__ == "__main__":
    main()
