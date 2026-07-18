"""Optional OLMoE loading helpers for full-size ACDL experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hf_sources import load_pretrained

from .fusion import make_prefix_bridge


def _make_retrieval_head(hidden_size: int, output_size: Optional[int] = None) -> nn.Module:
    output_size = int(output_size or hidden_size)
    return nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, output_size),
    )


def _mean_std_pool(features: Tensor) -> Tensor:
    features = features.float()
    mean = features.mean(dim=1)
    std = features.std(dim=1, unbiased=False)
    return torch.cat([mean, std], dim=-1)


@dataclass
class OLMoELoadConfig:
    base_model: str = "allenai/OLMoE-1B-7B-0924"
    top_k: int = 2
    router_aux_loss_coef: float = 0.01
    dtype: str = "bfloat16"
    device_map: Optional[str] = "auto"
    output_router_logits: bool = True


def _set_runtime_top_k(model: nn.Module, top_k: int, aux_coef: float, output_router_logits: bool) -> None:
    """Update OLMoE routing after native checkpoint loading."""
    if hasattr(model.config, "num_experts_per_tok"):
        model.config.num_experts_per_tok = int(top_k)
    if hasattr(model.config, "router_aux_loss_coef"):
        model.config.router_aux_loss_coef = float(aux_coef)
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = bool(output_router_logits)
    for obj in (model, getattr(model, "model", None)):
        if obj is None:
            continue
        if hasattr(obj, "num_experts_per_tok"):
            obj.num_experts_per_tok = int(top_k)
        if hasattr(obj, "router_aux_loss_coef"):
            obj.router_aux_loss_coef = float(aux_coef)
    for layer in getattr(model.model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        for obj in (mlp, getattr(mlp, "gate", None) if mlp is not None else None):
            if obj is None:
                continue
            for attr in ("top_k", "num_experts_per_tok", "k"):
                if hasattr(obj, attr) and isinstance(getattr(obj, attr), int):
                    setattr(obj, attr, int(top_k))


def load_olmoe_causal_lm(config: OLMoELoadConfig):
    """Load OLMoE natively, then set the runtime Top-k router policy."""

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Install transformers to load OLMoE") from exc

    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    model = load_pretrained(
        AutoModelForCausalLM,
        config.base_model,
        torch_dtype=dtype,
        device_map=config.device_map,
    )
    _set_runtime_top_k(model, config.top_k, config.router_aux_loss_coef, config.output_router_logits)
    return model


class OLMoEMultimodalPrefixWrapper(nn.Module):
    """Prepend image/audio prefix embeddings and run an OLMoE causal LM."""

    def __init__(
        self,
        lm: nn.Module,
        hidden_size: int,
        image_input_dim: int,
        audio_input_dim: int,
        image_prefix_tokens: int = 8,
        audio_prefix_tokens: int = 16,
        image_retrieval_dim: Optional[int] = None,
        audio_retrieval_dim: Optional[int] = None,
        use_prefix_residual_alignment: bool = False,
        image_bridge_type: str = "query_resampler",
        audio_bridge_type: str = "query_resampler",
        bridge_num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.lm = lm
        self.image_resampler = make_prefix_bridge(
            image_bridge_type,
            image_input_dim,
            hidden_size,
            image_prefix_tokens,
            num_heads=bridge_num_heads,
        )
        self.audio_resampler = make_prefix_bridge(
            audio_bridge_type,
            audio_input_dim,
            hidden_size,
            audio_prefix_tokens,
            num_heads=bridge_num_heads,
        )
        self.image_retrieval_head = _make_retrieval_head(hidden_size, image_retrieval_dim)
        self.audio_retrieval_head = _make_retrieval_head(hidden_size, audio_retrieval_dim)
        self.use_prefix_residual_alignment = bool(use_prefix_residual_alignment)
        self.image_direct_retrieval_head = _make_retrieval_head(image_input_dim * 2, image_retrieval_dim)
        self.audio_direct_retrieval_head = _make_retrieval_head(audio_input_dim * 2, audio_retrieval_dim)

    def image_prefix(self, image_features: Tensor) -> Tensor:
        return self.image_resampler(image_features)

    def audio_prefix(self, audio_features: Tensor) -> Tensor:
        return self.audio_resampler(audio_features)

    def _shared_prefix_hidden(self, prefix: Tensor) -> Tensor:
        target_dtype = self.lm.get_input_embeddings().weight.dtype
        prefix = prefix.to(dtype=target_dtype)
        attention_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)
        outputs = self.lm(
            inputs_embeds=prefix,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_router_logits=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1].float().mean(dim=1)
        if self.use_prefix_residual_alignment:
            hidden = hidden + prefix.float().mean(dim=1)
        return hidden

    def image_alignment_from_prefix(self, image_prefix: Tensor) -> Tensor:
        hidden = self._shared_prefix_hidden(image_prefix)
        return F.normalize(self.image_retrieval_head(hidden).float(), dim=-1)

    def audio_alignment_from_prefix(self, audio_prefix: Tensor) -> Tensor:
        hidden = self._shared_prefix_hidden(audio_prefix)
        return F.normalize(self.audio_retrieval_head(hidden).float(), dim=-1)

    def image_alignment_vector(self, image_features: Tensor) -> Tensor:
        # Retrieval evaluation/training must use the same OLMoE prefix path as multimodal generation.
        return self.image_alignment_from_prefix(self.image_prefix(image_features))

    def audio_alignment_vector(self, audio_features: Tensor) -> Tensor:
        # Retrieval evaluation/training must use the same OLMoE prefix path as multimodal generation.
        return self.audio_alignment_from_prefix(self.audio_prefix(audio_features))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        image_features: Optional[Tensor] = None,
        audio_features: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ):
        token_embeds = self.lm.get_input_embeddings()(input_ids)
        target_dtype = token_embeds.dtype
        embed_parts = []
        label_parts = []
        mask_parts = []
        if image_features is not None:
            image_prefix = self.image_prefix(image_features).to(dtype=target_dtype)
            embed_parts.append(image_prefix)
            mask_parts.append(torch.ones(image_prefix.shape[:2], dtype=torch.long, device=input_ids.device))
            if labels is not None:
                label_parts.append(torch.full(image_prefix.shape[:2], -100, dtype=torch.long, device=input_ids.device))
        if audio_features is not None:
            audio_prefix = self.audio_prefix(audio_features).to(dtype=target_dtype)
            embed_parts.append(audio_prefix)
            mask_parts.append(torch.ones(audio_prefix.shape[:2], dtype=torch.long, device=input_ids.device))
            if labels is not None:
                label_parts.append(torch.full(audio_prefix.shape[:2], -100, dtype=torch.long, device=input_ids.device))
        embed_parts.append(token_embeds)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask_parts.append(attention_mask)
        if labels is not None:
            label_parts.append(labels)
        inputs_embeds = torch.cat(embed_parts, dim=1)
        full_attention_mask = torch.cat(mask_parts, dim=1)
        full_labels = torch.cat(label_parts, dim=1) if labels is not None else None
        lm_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": full_attention_mask,
            "labels": full_labels,
            "output_router_logits": True,
        }
        if output_hidden_states:
            lm_kwargs.update(output_hidden_states=True, return_dict=True)
        return self.lm(**lm_kwargs)
