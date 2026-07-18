"""Pinned Hugging Face model and dataset references.

All production source resolution in this repository goes through this module.
Known repositories use audited commits; alternate repositories are accepted
only when the caller supplies an exact commit SHA.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping


_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


MODEL_REGISTRY: Mapping[str, str] = MappingProxyType(
    {
        "allenai/OLMoE-1B-7B-0924": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
        "openai/clip-vit-base-patch32": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "openai/whisper-base.en": "911407f4214e0e1d82085af863093ec0b66f9cd6",
        "openai/whisper-tiny.en": "87c7102498dcde7456f24cfd30239ca606ed9063",
    }
)

DATASET_REGISTRY: Mapping[str, str] = MappingProxyType(
    {
        "NeelNanda/c4-10k": "bdb17e3672308890562fe8f5ebe5d07bc88d764a",
        "allenai/c4": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "codeparrot/codeparrot-clean-valid": "4db92d2ec0c1b4c41eeb439cfae16854511d9dcd",
        "hails/agieval-logiqa-en": "e01aa0f040456cd8d58ee9986a58b64e26c1b782",
        "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
        "hails/agieval-sat-en": "848ee12cf003124f5a1e33446fa3cc6d2ec028e4",
        "hails/agieval-sat-math": "51ab661f5a2e48370671c87c29d037b5b2b4853e",
        "hails/agieval-lsat-ar": "052cc636b612f5563329dd182fb6c2cad56681c8",
        "hails/agieval-lsat-lr": "d876c675a8d47aa4d8a6d682ca8400b7d2ffe1c4",
        "jxie/coco_captions": "a2ed90d49b61dd13dd71f399c70f5feb897f8bec",
        "openslr/librispeech_asr": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
        "Multimodal-Fatima/COCO_captions_validation": "bfa149029bb1e2975cb0b9bea8ad948db9e9ddb2",
    }
)

# Readable aliases for callers and provenance checks.
KNOWN_MODELS = MODEL_REGISTRY
KNOWN_DATASETS = DATASET_REGISTRY


def validate_revision(revision: str) -> str:
    """Return an exact lowercase 40-hex commit or fail closed."""
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("Hugging Face revision must be an exact lowercase 40-hex commit")
    return revision


@dataclass(frozen=True)
class HFRef:
    """A Hugging Face repository resolved to an immutable commit."""

    repo_id: str
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.repo_id, str) or not self.repo_id.strip():
            raise ValueError("Hugging Face repo_id must be a non-empty string")
        object.__setattr__(self, "revision", validate_revision(self.revision))

    def as_dict(self) -> dict[str, str]:
        return {"repo_id": self.repo_id, "revision": self.revision}


def _resolve(
    repo_id: str | HFRef,
    revision: str | None,
    registry: Mapping[str, str],
    source_kind: str,
) -> HFRef:
    if isinstance(repo_id, HFRef):
        if revision is not None:
            raise ValueError("revision cannot be supplied with an already resolved HFRef")
        return repo_id
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("Hugging Face repo_id must be a non-empty string")
    if revision is None:
        revision = registry.get(repo_id)
        if revision is None:
            raise ValueError(
                f"unknown Hugging Face {source_kind} {repo_id!r}; "
                "supply an exact 40-hex revision"
            )
    return HFRef(repo_id=repo_id, revision=revision)


def resolve_model(repo_id: str | HFRef, revision: str | None = None) -> HFRef:
    return _resolve(repo_id, revision, MODEL_REGISTRY, "model")


def resolve_dataset(repo_id: str | HFRef, revision: str | None = None) -> HFRef:
    return _resolve(repo_id, revision, DATASET_REGISTRY, "dataset")


def resolve(
    repo_id: str | HFRef,
    revision: str | None = None,
    *,
    kind: str | None = None,
) -> HFRef:
    """Resolve a known ref, or an alternate ref with an explicit commit."""
    if kind == "model":
        return resolve_model(repo_id, revision)
    if kind == "dataset":
        return resolve_dataset(repo_id, revision)
    if kind is not None:
        raise ValueError("kind must be 'model', 'dataset', or None")
    if isinstance(repo_id, HFRef) or revision is not None:
        return _resolve(repo_id, revision, {}, "repository")
    if repo_id in MODEL_REGISTRY:
        return resolve_model(repo_id)
    if repo_id in DATASET_REGISTRY:
        return resolve_dataset(repo_id)
    raise ValueError(
        f"unknown Hugging Face repository {repo_id!r}; supply an exact 40-hex revision"
    )


def load_pretrained(
    factory: Any,
    repo_id: str | HFRef,
    *args: Any,
    revision: str | None = None,
    **kwargs: Any,
) -> Any:
    """Call ``from_pretrained`` with a resolved model commit."""
    ref = resolve_model(repo_id, revision)
    loader = getattr(factory, "from_pretrained", None)
    if loader is None:
        if not callable(factory):
            raise TypeError("factory must be callable or expose from_pretrained")
        loader = factory
    return loader(ref.repo_id, *args, revision=ref.revision, **kwargs)


def load_dataset_ref(
    repo_id: str | HFRef,
    config: str | None = None,
    *,
    revision: str | None = None,
    loader: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Call ``datasets.load_dataset`` with a resolved dataset commit."""
    ref = resolve_dataset(repo_id, revision)
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset
    if config is None:
        return loader(ref.repo_id, revision=ref.revision, **kwargs)
    return loader(ref.repo_id, config, revision=ref.revision, **kwargs)
