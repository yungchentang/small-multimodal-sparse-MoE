"""Metric aggregation for sparse MoE experiments."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def perplexity(loss: float) -> float:
    return float(math.exp(min(20.0, loss)))


def normalize_counts(counts: List[int]) -> List[float]:
    total = float(sum(counts))
    if total <= 0:
        return [0.0 for _ in counts]
    return [float(x) / total for x in counts]


def aggregate_router_metrics(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}
    num_experts = len(rows[0].get("accepted_expert_counts", rows[0].get("expert_counts", [])))
    accepted = [0 for _ in range(num_experts)]
    assigned = [0 for _ in range(num_experts)]
    by_modality: Dict[str, List[int]] = {}
    for row in rows:
        for i, value in enumerate(row.get("accepted_expert_counts", [])):
            accepted[i] += int(value)
        for i, value in enumerate(row.get("expert_counts", [])):
            assigned[i] += int(value)
        for modality, counts in row.get("by_modality", {}).items():
            by_modality.setdefault(modality, [0 for _ in range(num_experts)])
            for i, value in enumerate(counts):
                by_modality[modality][i] += int(value)
    return {
        "gate_entropy": safe_mean(float(r.get("entropy", 0.0)) for r in rows),
        "inactive_expert_ratio": safe_mean(float(r.get("inactive_ratio", 0.0)) for r in rows),
        "overflow_ratio": safe_mean(float(r.get("overflow_ratio", 0.0)) for r in rows),
        "assigned_expert_counts": assigned,
        "accepted_expert_counts": accepted,
        "expert_utilization": normalize_counts(accepted),
        "by_modality": {k: normalize_counts(v) for k, v in by_modality.items()},
    }
