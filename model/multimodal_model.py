"""Small multimodal sparse MoE language model used for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .fusion import QueryResampler
from .modeling_moe import Top2CalibratedSparseMoE


@dataclass
class SmallMoEConfig:
    vocab_size: int = 256
    max_seq_len: int = 256
    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25
    router_aux_loss_coef: float = 0.01
    normalize_topk_prob: bool = True
    gamma_trainable: bool = True
    image_input_dim: int = 32
    audio_input_dim: int = 32
    image_prefix_tokens: int = 8
    audio_prefix_tokens: int = 16
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "SmallMoEConfig":
        model = dict(raw.get("model", {}))
        moe = dict(raw.get("moe", {}))
        vision = dict(raw.get("vision", {}))
        speech = dict(raw.get("speech", {}))
        return cls(
            vocab_size=int(model.get("vocab_size", cls.vocab_size)),
            max_seq_len=int(model.get("max_seq_len", cls.max_seq_len)),
            hidden_size=int(model.get("hidden_size", cls.hidden_size)),
            intermediate_size=int(model.get("intermediate_size", cls.intermediate_size)),
            num_layers=int(model.get("num_layers", cls.num_layers)),
            num_heads=int(model.get("num_heads", cls.num_heads)),
            num_experts=int(moe.get("num_experts", cls.num_experts)),
            top_k=int(moe.get("final_top_k", moe.get("top_k", cls.top_k))),
            capacity_factor=float(moe.get("capacity_factor_train", moe.get("capacity_factor", cls.capacity_factor))),
            router_aux_loss_coef=float(moe.get("router_aux_loss_coef", cls.router_aux_loss_coef)),
            normalize_topk_prob=bool(moe.get("normalize_top2_weights", cls.normalize_topk_prob)),
            gamma_trainable=bool(moe.get("gamma_trainable", cls.gamma_trainable)),
            image_input_dim=int(vision.get("input_dim", cls.image_input_dim)),
            audio_input_dim=int(speech.get("input_dim", cls.audio_input_dim)),
            image_prefix_tokens=int(vision.get("num_prefix_tokens", cls.image_prefix_tokens)),
            audio_prefix_tokens=int(speech.get("num_prefix_tokens", cls.audio_prefix_tokens)),
            dropout=float(model.get("dropout", cls.dropout)),
        )


class TransformerMoEBlock(nn.Module):
    def __init__(self, config: SmallMoEConfig, layer_id: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.moe_norm = nn.LayerNorm(config.hidden_size)
        self.moe = Top2CalibratedSparseMoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            capacity_factor=config.capacity_factor,
            router_aux_loss_coef=config.router_aux_loss_coef,
            normalize_topk_prob=config.normalize_topk_prob,
            gamma_trainable=config.gamma_trainable,
            layer_id=layer_id,
        )

    def forward(self, hidden_states: Tensor, causal_mask: Tensor, modality_ids: Tensor) -> Dict[str, object]:
        normed = self.attn_norm(hidden_states)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
        hidden_states = hidden_states + attn_out
        moe_out = self.moe(self.moe_norm(hidden_states), modality_ids=modality_ids)
        hidden_states = hidden_states + moe_out.hidden_states
        return {
            "hidden_states": hidden_states,
            "aux_loss": moe_out.aux_loss,
            "metrics": moe_out.metrics,
        }


class SmallMultimodalMoEModel(nn.Module):
    """A compact, fully trainable multimodal MoE LM for smoke and ablation runs."""

    def __init__(self, config: SmallMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.modality_embed = nn.Embedding(4, config.hidden_size)
        self.image_resampler = QueryResampler(
            config.image_input_dim,
            config.hidden_size,
            config.image_prefix_tokens,
            num_heads=config.num_heads,
        )
        self.audio_resampler = QueryResampler(
            config.audio_input_dim,
            config.hidden_size,
            config.audio_prefix_tokens,
            num_heads=config.num_heads,
        )
        self.blocks = nn.ModuleList(
            TransformerMoEBlock(config, layer_id=i) for i in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _compose_inputs(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor],
        image_features: Optional[Tensor],
        audio_features: Optional[Tensor],
        modality_ids: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        token_embeds = self.token_embed(input_ids)
        batch_size = token_embeds.shape[0]
        embed_parts: List[Tensor] = []
        modality_parts: List[Tensor] = []
        label_parts: List[Tensor] = []

        if image_features is not None:
            image_prefix = self.image_resampler(image_features)
            embed_parts.append(image_prefix)
            modality_parts.append(torch.full((batch_size, image_prefix.shape[1]), 1, device=input_ids.device))
            if labels is not None:
                label_parts.append(torch.full((batch_size, image_prefix.shape[1]), -100, device=input_ids.device))

        if audio_features is not None:
            audio_prefix = self.audio_resampler(audio_features)
            embed_parts.append(audio_prefix)
            modality_parts.append(torch.full((batch_size, audio_prefix.shape[1]), 2, device=input_ids.device))
            if labels is not None:
                label_parts.append(torch.full((batch_size, audio_prefix.shape[1]), -100, device=input_ids.device))

        embed_parts.append(token_embeds)
        if modality_ids is None:
            text_modality = 3 if image_features is not None or audio_features is not None else 0
            modality_ids = torch.full(input_ids.shape, text_modality, device=input_ids.device)
        modality_parts.append(modality_ids)
        if labels is not None:
            label_parts.append(labels)

        inputs_embeds = torch.cat(embed_parts, dim=1)
        full_modality_ids = torch.cat(modality_parts, dim=1).long()
        full_labels = torch.cat(label_parts, dim=1).long() if labels is not None else None
        if inputs_embeds.shape[1] > self.config.max_seq_len:
            inputs_embeds = inputs_embeds[:, : self.config.max_seq_len]
            full_modality_ids = full_modality_ids[:, : self.config.max_seq_len]
            if full_labels is not None:
                full_labels = full_labels[:, : self.config.max_seq_len]
        return {
            "inputs_embeds": inputs_embeds,
            "labels": full_labels,
            "modality_ids": full_modality_ids,
        }

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        image_features: Optional[Tensor] = None,
        audio_features: Optional[Tensor] = None,
        modality_ids: Optional[Tensor] = None,
    ) -> Dict[str, object]:
        composed = self._compose_inputs(input_ids, labels, image_features, audio_features, modality_ids)
        hidden_states = composed["inputs_embeds"]
        labels = composed["labels"]
        modality_ids = composed["modality_ids"]
        seq_len = hidden_states.shape[1]
        positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        hidden_states = hidden_states + self.pos_embed(positions) + self.modality_embed(modality_ids)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
            diagonal=1,
        )

        aux_losses: List[Tensor] = []
        router_metrics: List[Dict[str, object]] = []
        for block in self.blocks:
            block_out = block(hidden_states, causal_mask, modality_ids)
            hidden_states = block_out["hidden_states"]
            aux_losses.append(block_out["aux_loss"])
            router_metrics.append(block_out["metrics"])

        logits = self.lm_head(self.final_norm(hidden_states))
        aux_loss = torch.stack(aux_losses).sum() if aux_losses else logits.new_tensor(0.0)
        loss = None
        lm_loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = lm_loss + aux_loss
        return {
            "loss": loss,
            "lm_loss": lm_loss,
            "aux_loss": aux_loss,
            "logits": logits,
            "router_metrics": router_metrics,
            "labels": labels,
        }
