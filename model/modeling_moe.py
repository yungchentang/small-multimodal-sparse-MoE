"""Sparse MoE blocks with Top-k routing, capacity limits, and routing metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MODALITY_NAMES = {
    0: "text",
    1: "image_prefix",
    2: "audio_prefix",
    3: "target_text",
}


@dataclass
class SparseMoEOutput:
    hidden_states: Tensor
    aux_loss: Tensor
    metrics: Dict[str, object]


class SwiGLUExpert(nn.Module):
    """SwiGLU FFN expert used by the sparse MoE layer."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Top2CalibratedSparseMoE(nn.Module):
    """Capacity-aware Sparse MoE with normalized Top-k routing and layer gamma."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        router_aux_loss_coef: float = 0.01,
        normalize_topk_prob: bool = True,
        gamma_init: float = 1.0,
        gamma_trainable: bool = True,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router_aux_loss_coef = router_aux_loss_coef
        self.normalize_topk_prob = normalize_topk_prob
        self.layer_id = layer_id

        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLUExpert(hidden_size, intermediate_size) for _ in range(num_experts)
        )
        gamma = torch.tensor(float(gamma_init))
        if gamma_trainable:
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer("gamma", gamma)

    def set_top_k(self, top_k: int) -> None:
        if top_k < 1 or top_k > self.num_experts:
            raise ValueError(f"invalid top_k={top_k} for {self.num_experts} experts")
        self.top_k = top_k

    def set_gamma(self, value: float) -> None:
        with torch.no_grad():
            self.gamma.fill_(float(value))

    def _load_balancing_loss(self, probs: Tensor, top_ids: Tensor) -> Tensor:
        selected = F.one_hot(top_ids, num_classes=self.num_experts).sum(dim=1).float()
        density = selected.mean(dim=0) / float(self.top_k)
        density_proxy = probs.mean(dim=0)
        return self.num_experts * torch.sum(density * density_proxy)

    def forward(self, hidden_states: Tensor, modality_ids: Optional[Tensor] = None) -> SparseMoEOutput:
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(batch_size * seq_len, hidden_size)
        num_tokens = flat_hidden.shape[0]

        router_logits = self.router(flat_hidden)
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        top_probs, top_ids = torch.topk(router_probs, k=self.top_k, dim=-1)
        if self.normalize_topk_prob:
            top_weights = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            top_weights = top_probs
        top_weights = top_weights.to(flat_hidden.dtype)

        capacity = max(
            1,
            int(math.ceil(self.capacity_factor * num_tokens * self.top_k / self.num_experts)),
        )
        combined = torch.zeros_like(flat_hidden)
        flat_expert_ids = top_ids.reshape(-1)
        flat_weights = top_weights.reshape(-1)
        flat_token_ids = torch.arange(num_tokens, device=flat_hidden.device).repeat_interleave(self.top_k)
        overflow_flat = torch.ones_like(flat_expert_ids, dtype=torch.bool)
        accepted_counts: List[int] = []

        for expert_id, expert in enumerate(self.experts):
            positions = torch.nonzero(flat_expert_ids == expert_id, as_tuple=False).flatten()
            if positions.numel() == 0:
                accepted_counts.append(0)
                continue
            expert_weights = flat_weights[positions]
            order = torch.argsort(expert_weights.float(), descending=True)
            accepted_positions = positions[order[:capacity]]
            token_positions = flat_token_ids[accepted_positions]
            expert_input = flat_hidden.index_select(0, token_positions)
            expert_output = expert(expert_input)
            expert_output = expert_output * flat_weights[accepted_positions].unsqueeze(-1)
            combined.index_add_(0, token_positions, expert_output)
            overflow_flat[accepted_positions] = False
            accepted_counts.append(int(accepted_positions.numel()))

        aux_loss = self._load_balancing_loss(router_probs, top_ids) * self.router_aux_loss_coef
        combined = combined * self.gamma.to(combined.dtype)
        output = combined.reshape(batch_size, seq_len, hidden_size)

        assigned_counts = torch.bincount(flat_expert_ids, minlength=self.num_experts)
        accepted_counts_tensor = torch.tensor(
            accepted_counts, dtype=torch.long, device=flat_hidden.device
        )
        entropy = -(router_probs * (router_probs + 1e-9).log()).sum(dim=-1)
        overflow_ratio = overflow_flat.float().mean() if overflow_flat.numel() else torch.tensor(0.0)
        inactive_ratio = (accepted_counts_tensor == 0).float().mean()
        metrics: Dict[str, object] = {
            "layer": self.layer_id,
            "top_k": self.top_k,
            "capacity": capacity,
            "gamma": float(self.gamma.detach().float().cpu()),
            "entropy": float(entropy.mean().detach().cpu()),
            "inactive_ratio": float(inactive_ratio.detach().cpu()),
            "overflow_ratio": float(overflow_ratio.detach().cpu()),
            "expert_counts": assigned_counts.detach().cpu().tolist(),
            "accepted_expert_counts": accepted_counts_tensor.detach().cpu().tolist(),
            "moe_output_norm_mean": float(output.detach().float().norm(dim=-1).mean().cpu()),
        }

        if modality_ids is not None:
            flat_modalities = modality_ids.reshape(-1)
            by_modality: Dict[str, List[int]] = {}
            assignment_modalities = flat_modalities.index_select(0, flat_token_ids)
            for modality_id, modality_name in MODALITY_NAMES.items():
                mask = assignment_modalities == modality_id
                if mask.any():
                    counts = torch.bincount(
                        flat_expert_ids[mask], minlength=self.num_experts
                    ).detach().cpu().tolist()
                else:
                    counts = [0 for _ in range(self.num_experts)]
                by_modality[modality_name] = counts
            metrics["by_modality"] = by_modality

        return SparseMoEOutput(hidden_states=output, aux_loss=aux_loss, metrics=metrics)


def copy_moe_weights(source: nn.Module, target: nn.Module) -> None:
    """Copy matching parameters between MoE models, ignoring top-k policy differences."""

    target_state = target.state_dict()
    source_state = source.state_dict()
    compatible = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    target_state.update(compatible)
    target.load_state_dict(target_state)
