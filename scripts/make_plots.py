"""Generate compact SVG/PNG figures from metrics JSON/JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_svg_bar(path: Path, values: List[float], title: str) -> None:
    width, height = 720, 360
    max_v = max(values) if values else 1.0
    bars = []
    n = max(1, len(values))
    bar_w = (width - 100) / n
    for i, value in enumerate(values):
        h = 260 * (value / max_v if max_v else 0)
        x = 60 + i * bar_w
        y = 310 - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1, bar_w-2):.1f}" height="{h:.1f}" fill="#4C78A8"/>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="30" y="30" font-size="18">{title}</text>{"".join(bars)}<line x1="50" y1="310" x2="700" y2="310" stroke="#333"/></svg>'
    path.write_text(svg, encoding="utf-8")


def write_svg_line(path: Path, values: List[float], title: str) -> None:
    width, height = 720, 360
    if not values:
        values = [0.0]
    max_v, min_v = max(values), min(values)
    span = max(max_v - min_v, 1e-6)
    points = []
    for i, value in enumerate(values):
        x = 60 + i * (620 / max(1, len(values) - 1))
        y = 310 - 250 * ((value - min_v) / span)
        points.append(f"{x:.1f},{y:.1f}")
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="30" y="30" font-size="18">{title}</text><polyline points="{" ".join(points)}" fill="none" stroke="#F58518" stroke-width="3"/><line x1="50" y1="310" x2="700" y2="310" stroke="#333"/></svg>'
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/smoke")
    args = parser.parse_args()
    out = Path(args.output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    train = read_jsonl(out / "train_metrics.jsonl")
    router = read_jsonl(out / "eval_router_metrics.jsonl") or read_jsonl(out / "router_metrics.jsonl")
    moe = read_json(out / "metrics_moe.json", {})
    calibration = read_json(out / "metrics_top2_calibration.json", {})
    write_svg_line(fig_dir / "fig_training_loss.svg", [float(r.get("loss", 0.0)) for r in train], "Training loss")
    write_svg_line(fig_dir / "fig_gate_entropy.svg", [float(r.get("entropy", r.get("gate_entropy", 0.0))) for r in router or train], "Gate entropy")
    write_svg_bar(fig_dir / "fig_expert_utilization.svg", moe.get("expert_utilization", []), "Expert utilization")
    by_mod = moe.get("by_modality", {})
    if by_mod:
        first = next(iter(by_mod.values()))
    else:
        first = []
    write_svg_bar(fig_dir / "fig_modality_experts.svg", first, "Modality expert usage")
    write_svg_bar(fig_dir / "fig_routing_heatmap.svg", moe.get("assigned_expert_counts", []), "Routing frequency")
    write_svg_line(fig_dir / "fig_top2_calibration.svg", calibration.get("gamma", []), "Top-2 gamma calibration")
    print(json.dumps({"figures": sorted(p.name for p in fig_dir.glob("*.svg"))}, sort_keys=True))


if __name__ == "__main__":
    main()
