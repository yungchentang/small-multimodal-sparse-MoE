from __future__ import annotations

import unittest

from scripts.run_e6_feasibility import validate_source_manifest, validate_steps


class E6FeasibilityTests(unittest.TestCase):
    def test_source_manifest_accepts_completed_positive_step_root(self):
        manifest = {
            "args": {"final_steps": 1000},
            "completion": {"status": "completed", "e3_steps": 1000},
        }
        self.assertEqual(validate_source_manifest(manifest), 1000)
        manifest["completion"]["e3_steps"] = 999
        with self.assertRaisesRegex(ValueError, "completed positive-step"):
            validate_source_manifest(manifest)

    def test_accepts_exact_three_modality_steps(self) -> None:
        digest = "a" * 64
        rows = [
            {
                "step": index,
                "modality": modality,
                "optimizer_step": True,
                "initial_checkpoint_state_restored": True,
                "source_selected_checkpoint_sha256": digest,
                "loss": float(index),
            }
            for index, modality in enumerate(("text", "image", "speech"), start=1)
        ]
        self.assertEqual(validate_steps({"steps": rows}, digest), rows)

    def test_rejects_unbound_checkpoint_rows(self) -> None:
        rows = [
            {
                "step": index,
                "modality": modality,
                "optimizer_step": True,
                "initial_checkpoint_state_restored": True,
                "source_selected_checkpoint_sha256": "b" * 64,
            }
            for index, modality in enumerate(("text", "image", "speech"), start=1)
        ]
        with self.assertRaisesRegex(ValueError, "source checkpoint hash mismatch"):
            validate_steps({"steps": rows}, "a" * 64)

    def test_rejects_wrong_modality_coverage(self) -> None:
        digest = "a" * 64
        rows = [
            {
                "step": index,
                "modality": "text",
                "optimizer_step": True,
                "initial_checkpoint_state_restored": True,
                "source_selected_checkpoint_sha256": digest,
            }
            for index in (1, 2, 3)
        ]
        with self.assertRaisesRegex(ValueError, "text, image, and speech"):
            validate_steps({"steps": rows}, digest)


if __name__ == "__main__":
    unittest.main()
