"""Optional image and speech encoders for the multimodal pipeline."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import Tensor, nn

from hf_sources import load_pretrained


class TinyImageEncoder(nn.Module):
    """Small image encoder used for smoke tests without external downloads."""

    def __init__(self, output_dim: int, num_tokens: int = 8) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, output_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((num_tokens, 1))

    def forward(self, images: Tensor) -> Tensor:
        features = self.conv(images)
        pooled = self.pool(features).squeeze(-1).transpose(1, 2)
        return pooled


class TinySpeechEncoder(nn.Module):
    """Small speech encoder used for smoke tests without external downloads."""

    def __init__(self, output_dim: int, num_tokens: int = 16) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(32, output_dim, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)

    def forward(self, audio: Tensor) -> Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return self.pool(self.conv(audio)).transpose(1, 2)


class HFVisionEncoder(nn.Module):
    """Frozen HuggingFace vision encoder wrapper for CLIP, ViT, or SigLIP."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError("Install transformers to use HFVisionEncoder") from exc
        self.processor = load_pretrained(AutoImageProcessor, model_name)
        self.encoder = load_pretrained(AutoModel, model_name)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode_images(self, images: Iterable[object], device: Optional[torch.device] = None) -> Tensor:
        batch = self.processor(images=list(images), return_tensors="pt")
        if device is not None:
            batch = {key: value.to(device) for key, value in batch.items()}
            self.encoder.to(device)
        outputs = self.encoder(**batch)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs[0]


class HFSpeechEncoder(nn.Module):
    """Frozen HuggingFace speech encoder wrapper for Whisper, Wav2Vec2, or HuBERT."""

    def __init__(self, model_name: str = "openai/whisper-base.en") -> None:
        super().__init__()
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise ImportError("Install transformers to use HFSpeechEncoder") from exc
        self.processor = load_pretrained(AutoFeatureExtractor, model_name)
        self.encoder = load_pretrained(AutoModel, model_name)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode_audio(
        self,
        waveforms: Iterable[object],
        sampling_rate: int = 16000,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        batch = self.processor(
            list(waveforms),
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        if device is not None:
            batch = {key: value.to(device) for key, value in batch.items()}
            self.encoder.to(device)
        outputs = self.encoder(**batch)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs[0]
