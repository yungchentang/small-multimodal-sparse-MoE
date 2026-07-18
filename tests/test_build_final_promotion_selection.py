from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import build_final_promotion_selection as selector


def write_metric(path: Path, **overrides: object) -> None:
    payload = {
        "evaluation_scope": "development",
        "sealed_protocol": False,
        "conditional_uses_multimodal_prefix": True,
        "conditional_uses_direct_encoder_pooling": False,
        "conditional_uses_lm_logits": True,
        "eval_path": "shared_prefix",
        "e3_checkpoint_sha256": "a" * 64,
        "conditional_image_eval_count": 137,
        "conditional_speech_eval_count": 137,
        "conditional_candidates_per_query": 10,
        "conditional_speech_candidates_per_query": 10,
        "prefix_control": "real",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
class PromotionSelectionTests(unittest.TestCase):
    def test_require_dev_metric_accepts_shared_prefix_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.json"
            write_metric(path)
            result = selector.require_dev_metric(path, "a" * 64, 10)
        self.assertEqual(result["evaluation_scope"], "development")

    def test_require_dev_metric_rejects_invalid_contract(self) -> None:
        invalid_values = (
            ("evaluation_scope", "sealed_test"),
            ("sealed_protocol", True),
            ("conditional_uses_multimodal_prefix", False),
            ("conditional_uses_direct_encoder_pooling", True),
            ("conditional_uses_lm_logits", False),
            ("eval_path", "encoder_only"),
            ("e3_checkpoint_sha256", "b" * 64),
            ("conditional_image_eval_count", 136),
            ("conditional_speech_eval_count", 136),
            ("conditional_candidates_per_query", 5),
            ("conditional_speech_candidates_per_query", 5),
            ("prefix_control", "zero"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.json"
            for field, value in invalid_values:
                with self.subTest(field=field, value=value):
                    write_metric(path, **{field: value})
                    with self.assertRaises(selector.SelectionError):
                        selector.require_dev_metric(path, "a" * 64, 10)

    def test_canonical_hash_is_order_independent(self) -> None:
        self.assertEqual(
            selector.canonical_sha256({"a": 1, "b": 2}),
            selector.canonical_sha256({"b": 2, "a": 1}),
        )
