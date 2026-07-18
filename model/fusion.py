"""Projection and prefix-token fusion utilities."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _validate_encoder_states(encoder_states: Tensor, input_dim: int) -> None:
    if encoder_states.ndim != 3:
        raise ValueError(
            "prefix bridge expects encoder_states with shape "
            f"[batch, tokens, {input_dim}], got {tuple(encoder_states.shape)}"
        )
    if encoder_states.shape[1] < 1:
        raise ValueError("prefix bridge requires at least one encoder token")
    if encoder_states.shape[-1] != input_dim:
        raise ValueError(
            f"prefix bridge input dimension mismatch: expected {input_dim}, "
            f"got {encoder_states.shape[-1]}"
        )


class QueryResampler(nn.Module):
    """Compress a variable-length encoder sequence into a fixed prefix length."""

    def __init__(self, input_dim: int, hidden_size: int, num_prefix_tokens: int, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.input_dim = int(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.queries = nn.Parameter(torch.randn(num_prefix_tokens, hidden_size) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, encoder_states: Tensor) -> Tensor:
        _validate_encoder_states(encoder_states, self.input_dim)
        batch_size = encoder_states.shape[0]
        keys = self.input_proj(encoder_states)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        prefix, _ = self.attn(queries, keys, keys, need_weights=False)
        return self.norm(prefix)


class PrefixProjector(nn.Module):
    """Simple projection when encoder output already has the desired token count."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_prefix_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_prefix_tokens = (
            int(num_prefix_tokens) if num_prefix_tokens is not None else None
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, encoder_states: Tensor) -> Tensor:
        _validate_encoder_states(encoder_states, self.input_dim)
        if (
            self.num_prefix_tokens is not None
            and encoder_states.shape[1] != self.num_prefix_tokens
        ):
            raise ValueError(
                "linear projector does not resample tokens: expected "
                f"{self.num_prefix_tokens}, got {encoder_states.shape[1]}"
            )
        return self.net(encoder_states)


class NormalizedPrefixProjector(nn.Module):
    """Project each token independently, then normalize its hidden features."""

    def __init__(self, input_dim: int, hidden_size: int, num_prefix_tokens: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.projection = nn.Linear(input_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def audit_metadata(self) -> dict[str, object]:
        return {
            "bridge_type": "linear_projector_norm",
            "projection_mode": "tokenwise_linear",
            "normalization": "layer_norm",
            "input_tokens": self.num_prefix_tokens,
            "output_tokens": self.num_prefix_tokens,
            "input_dim": self.input_dim,
            "hidden_size": self.hidden_size,
        }

    def forward(self, encoder_states: Tensor) -> Tensor:
        _validate_encoder_states(encoder_states, self.input_dim)
        if encoder_states.shape[1] != self.num_prefix_tokens:
            raise ValueError(
                "linear projector norm does not resample tokens: expected "
                f"{self.num_prefix_tokens}, got {encoder_states.shape[1]}"
            )
        return self.norm(self.projection(encoder_states))


class IdentityPrefixBridge(nn.Module):
    """No-op control for already compatible encoder prefix tensors."""

    def __init__(self, input_dim: int, hidden_size: int, num_prefix_tokens: int) -> None:
        super().__init__()
        if input_dim != hidden_size:
            raise ValueError(
                "identity prefix bridge requires input_dim == hidden_size, "
                f"got {input_dim} != {hidden_size}"
            )
        self.input_dim = int(input_dim)
        self.num_prefix_tokens = int(num_prefix_tokens)

    def forward(self, encoder_states: Tensor) -> Tensor:
        _validate_encoder_states(encoder_states, self.input_dim)
        if encoder_states.shape[1] != self.num_prefix_tokens:
            raise ValueError(
                "identity prefix bridge cannot resample tokens: expected "
                f"{self.num_prefix_tokens}, got {encoder_states.shape[1]}"
            )
        return encoder_states


class AttentionPoolPrefixBridge(nn.Module):
    """Learn one attention distribution per output prefix token."""

    def __init__(self, input_dim: int, hidden_size: int, num_prefix_tokens: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.attention_logits = nn.Linear(hidden_size, num_prefix_tokens, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, encoder_states: Tensor) -> Tensor:
        _validate_encoder_states(encoder_states, self.input_dim)
        values = self.input_proj(encoder_states)
        weights = self.attention_logits(values).transpose(1, 2).softmax(dim=-1)
        return self.norm(weights @ values)


class TemporalResamplePrefixBridge(nn.Module):
    """Linearly interpolate the token axis to a deterministic fixed length."""

    def __init__(self, input_dim: int, hidden_size: int, num_prefix_tokens: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, encoder_states: Tensor) -> Tensor:
        _validate_encoder_states(encoder_states, self.input_dim)
        projected = self.input_proj(encoder_states)
        resampled = F.interpolate(
            projected.transpose(1, 2),
            size=self.num_prefix_tokens,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return self.norm(resampled)


class LocalPoolLinearPrefixBridge(nn.Module):
    """Preserve image-grid locality with deterministic 2D pooling and projection."""

    def __init__(self, input_dim: int, hidden_size: int, num_prefix_tokens: int) -> None:
        super().__init__()
        if num_prefix_tokens < 1:
            raise ValueError("local_pool_linear requires at least one prefix token")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.projection = nn.Linear(input_dim, hidden_size)
        self._last_pooling_metadata: dict[str, object] | None = None

    @staticmethod
    def _square_side(tokens: int) -> int | None:
        if tokens < 1:
            return None
        side = math.isqrt(tokens)
        return side if side * side == tokens else None

    def pooling_metadata(self, input_tokens: int) -> dict[str, object]:
        input_tokens = int(input_tokens)
        cls_input_side = self._square_side(input_tokens - 1)
        plain_input_side = self._square_side(input_tokens)
        if cls_input_side is not None:
            has_cls = True
            input_side = cls_input_side
            output_patch_tokens = self.num_prefix_tokens - 1
        elif plain_input_side is not None:
            has_cls = False
            input_side = plain_input_side
            output_patch_tokens = self.num_prefix_tokens
        else:
            raise ValueError(
                "local_pool_linear requires square image patch geometry, optionally "
                f"preceded by one CLS token; got {input_tokens} input tokens"
            )

        output_side = self._square_side(output_patch_tokens)
        if output_side is None:
            cls_note = "after preserving CLS" if has_cls else "without CLS"
            raise ValueError(
                "local_pool_linear output patch count must form a square grid "
                f"{cls_note}; got {self.num_prefix_tokens} prefix tokens"
            )
        if output_side > input_side:
            raise ValueError(
                "local_pool_linear does not upsample image grids: "
                f"input={input_side}x{input_side}, output={output_side}x{output_side}"
            )
        return {
            "bridge_type": "local_pool_linear",
            "pooling_mode": "adaptive_avg_pool2d",
            "cls_handling": "preserve_first_token" if has_cls else "no_cls_token",
            "input_tokens": input_tokens,
            "input_grid": [input_side, input_side],
            "output_tokens": self.num_prefix_tokens,
            "output_grid": [output_side, output_side],
            "input_dim": self.input_dim,
            "hidden_size": self.hidden_size,
        }

    def audit_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "bridge_type": "local_pool_linear",
            "pooling_mode": "adaptive_avg_pool2d",
            "geometry_policy": "square_grid_with_optional_leading_cls_fail_closed",
            "output_tokens": self.num_prefix_tokens,
            "input_dim": self.input_dim,
            "hidden_size": self.hidden_size,
        }
        if self._last_pooling_metadata is not None:
            metadata["observed_geometry"] = dict(self._last_pooling_metadata)
        return metadata

    def forward(self, encoder_states: Tensor) -> Tensor:
        _validate_encoder_states(encoder_states, self.input_dim)
        metadata = self.pooling_metadata(encoder_states.shape[1])
        input_side = int(metadata["input_grid"][0])
        output_side = int(metadata["output_grid"][0])
        has_cls = metadata["cls_handling"] == "preserve_first_token"

        patch_states = encoder_states[:, 1:] if has_cls else encoder_states
        patch_grid = patch_states.transpose(1, 2).reshape(
            encoder_states.shape[0], self.input_dim, input_side, input_side
        )
        pooled = F.adaptive_avg_pool2d(patch_grid, (output_side, output_side))
        pooled_tokens = pooled.flatten(2).transpose(1, 2)
        if has_cls:
            pooled_tokens = torch.cat((encoder_states[:, :1], pooled_tokens), dim=1)
        if pooled_tokens.shape[1] != self.num_prefix_tokens:
            raise RuntimeError("local_pool_linear produced an unexpected token count")
        self._last_pooling_metadata = metadata
        return self.projection(pooled_tokens)


def make_prefix_bridge(
    bridge_type: str,
    input_dim: int,
    hidden_size: int,
    num_prefix_tokens: int,
    *,
    num_heads: int = 4,
) -> nn.Module:
    """Build a fixed-shape encoder-to-prefix bridge."""
    normalized = str(bridge_type).strip().lower().replace("-", "_")
    aliases = {
        "resampler": "query_resampler",
        "projector": "linear_projector",
        "linear": "linear_projector",
        "no_resampling": "identity",
        "attention": "attention_pool",
        "temporal": "temporal_resample",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "query_resampler":
        return QueryResampler(input_dim, hidden_size, num_prefix_tokens, num_heads)
    if normalized == "linear_projector":
        return PrefixProjector(input_dim, hidden_size, num_prefix_tokens)
    if normalized == "linear_projector_norm":
        return NormalizedPrefixProjector(input_dim, hidden_size, num_prefix_tokens)
    if normalized == "identity":
        return IdentityPrefixBridge(input_dim, hidden_size, num_prefix_tokens)
    if normalized == "attention_pool":
        return AttentionPoolPrefixBridge(input_dim, hidden_size, num_prefix_tokens)
    if normalized == "temporal_resample":
        return TemporalResamplePrefixBridge(input_dim, hidden_size, num_prefix_tokens)
    if normalized == "local_pool_linear":
        return LocalPoolLinearPrefixBridge(input_dim, hidden_size, num_prefix_tokens)
    supported = (
        "query_resampler, linear_projector, linear_projector_norm, identity, "
        "attention_pool, temporal_resample, local_pool_linear"
    )
    raise ValueError(f"unsupported prefix bridge {bridge_type!r}; choose one of: {supported}")
