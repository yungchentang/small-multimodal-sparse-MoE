"""Tests for fail-closed Hugging Face source resolution."""

from __future__ import annotations

import re
import unittest
from unittest.mock import Mock, sentinel

import hf_sources


class RegistryTests(unittest.TestCase):
    def test_registries_contain_only_exact_commit_revisions(self) -> None:
        self.assertEqual(
            hf_sources.MODEL_REGISTRY,
            {
                "allenai/OLMoE-1B-7B-0924": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
                "openai/clip-vit-base-patch32": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                "openai/whisper-base.en": "911407f4214e0e1d82085af863093ec0b66f9cd6",
                "openai/whisper-tiny.en": "87c7102498dcde7456f24cfd30239ca606ed9063",
            },
        )
        self.assertEqual(len(hf_sources.DATASET_REGISTRY), 12)
        for revision in (*hf_sources.MODEL_REGISTRY.values(), *hf_sources.DATASET_REGISTRY.values()):
            self.assertRegex(revision, re.compile(r"^[0-9a-f]{40}$"))

    def test_known_unknown_and_main_resolution(self) -> None:
        known = hf_sources.resolve_model("allenai/OLMoE-1B-7B-0924")
        self.assertEqual(known.revision, "6d84c48581ece794365f2b8e9cfb043c68ade9c5")

        explicit = hf_sources.resolve_dataset("example/alternate", "a" * 40)
        self.assertEqual(explicit, hf_sources.HFRef("example/alternate", "a" * 40))

        with self.assertRaisesRegex(ValueError, "unknown Hugging Face dataset"):
            hf_sources.resolve_dataset("example/alternate")
        with self.assertRaisesRegex(ValueError, "exact lowercase 40-hex"):
            hf_sources.resolve_dataset("example/alternate", "main")
        with self.assertRaisesRegex(ValueError, "exact lowercase 40-hex"):
            hf_sources.resolve_model("allenai/OLMoE-1B-7B-0924", "main")


class LoaderTests(unittest.TestCase):
    def test_load_pretrained_propagates_registered_revision(self) -> None:
        factory = Mock()
        factory.from_pretrained.return_value = sentinel.model

        result = hf_sources.load_pretrained(
            factory,
            "openai/clip-vit-base-patch32",
            local_files_only=True,
        )

        self.assertIs(result, sentinel.model)
        factory.from_pretrained.assert_called_once_with(
            "openai/clip-vit-base-patch32",
            revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
            local_files_only=True,
        )

    def test_load_dataset_ref_propagates_registered_revision_and_config(self) -> None:
        loader = Mock(return_value=sentinel.dataset)

        result = hf_sources.load_dataset_ref(
            "openai/gsm8k",
            "main",
            split="train",
            streaming=False,
            loader=loader,
        )

        self.assertIs(result, sentinel.dataset)
        loader.assert_called_once_with(
            "openai/gsm8k",
            "main",
            revision="740312add88f781978c0658806c59bc2815b9866",
            split="train",
            streaming=False,
        )


if __name__ == "__main__":
    unittest.main()
