from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import torch

from scripts.eval_development_representation_diagnostics import (
    _assert_development_artifact,
    _metrics_only,
    direct_development_retrieval,
)


class DevelopmentRepresentationDiagnosticsTest(unittest.TestCase):
    def test_rejects_sealed_and_synthetic_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _assert_development_artifact(root / "development" / "image.jsonl")
            with self.assertRaisesRegex(ValueError, "sealed/synthetic"):
                _assert_development_artifact(root / "sealed" / "image.jsonl")
            with self.assertRaisesRegex(ValueError, "sealed/synthetic"):
                _assert_development_artifact(root / "synthetic_debug" / "image.jsonl")

    def test_dimension_mismatch_is_reported_without_invalid_similarity(self) -> None:
        features = types.SimpleNamespace(
            stages={"encoder_pooled": torch.ones(2, 512)},
            targets=torch.ones(2, 2048),
            media_to_text=[[0], [1]],
        )

        result = direct_development_retrieval(features, "encoder_pooled")

        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "not_comparable_dimension")
        self.assertEqual(result["media_dimension"], 512)
        self.assertEqual(result["target_dimension"], 2048)

    def test_rank_arrays_are_not_copied_into_summary(self) -> None:
        result = _metrics_only(
            {
                "media_to_text": {"r_at_1": 0.5},
                "media_to_text_ranks": [0, 2],
                "text_to_media_ranks": [1, 0],
            }
        )
        self.assertEqual(result, {"media_to_text": {"r_at_1": 0.5}})


if __name__ == "__main__":
    unittest.main()
