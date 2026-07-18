import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.audit_requirements import canonical_sha256, validate_e3_checkpoint_sidecar
from scripts.extract_checkpoint_provenance import (
    compare_args,
    sha256_file,
    validate_e3_metrics_checkpoint_identity,
)


class CheckpointProvenanceTest(unittest.TestCase):
    def audit_fixture(self, root: Path) -> dict[str, object]:
        experiment = root / "E3_final_multimodal_top2"
        experiment.mkdir()
        rows = [
            {"step": 1, "modality": "image", "loss": 2.0},
            {"step": 2, "modality": "text", "loss": 1.0},
        ]
        state = {"args": {"top_k": 2}, "last_row": rows[-1]}
        checkpoint = experiment / "checkpoint_final.pt"
        torch.save(state, checkpoint)
        metrics_path = experiment / "metrics.json"
        metrics_path.write_text(json.dumps({"steps": rows}), encoding="utf-8")
        raw_path = experiment / "train_metrics.jsonl"
        raw_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        sidecar = {
            "schema_version": 1,
            "passed": True,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_state_keys": sorted(state),
            "checkpoint_last_row_sha256": canonical_sha256(rows[-1]),
            "checkpoint_last_row_matches_metrics_last_row": True,
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": sha256_file(metrics_path),
            "metrics_step_rows": len(rows),
            "metrics_steps_sha256": canonical_sha256(rows),
            "raw_training_path": str(raw_path.resolve()),
            "raw_training_sha256": sha256_file(raw_path),
            "raw_training_rows": len(rows),
            "training_rows_sha256": canonical_sha256(rows),
        }
        return {
            "checkpoint": checkpoint,
            "metrics_path": metrics_path,
            "raw_path": raw_path,
            "rows": rows,
            "sidecar": sidecar,
        }


    def test_equal_non_path_args_pass(self):
        self.assertEqual(
            compare_args(
                {"capacity_factor": 7.0, "output_dir": "a"},
                {"capacity_factor": 7.0, "output_dir": "b"},
            ),
            {},
        )

    def test_non_path_mismatch_is_reported(self):
        mismatch = compare_args(
            {"aux_coef": 0.02, "top_k": 2},
            {"aux_coef": 0.01, "top_k": 2},
        )
        self.assertEqual(mismatch["aux_coef"]["checkpoint"], 0.02)
        self.assertEqual(mismatch["aux_coef"]["manifest"], 0.01)


    def test_rejects_cross_checkpoint_metrics_and_embedded_text_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "selected" / "checkpoint_final.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"selected-checkpoint")
            copied_checkpoint = root / "copied" / "checkpoint_final.pt"
            copied_checkpoint.parent.mkdir()
            copied_checkpoint.write_bytes(b"copied-checkpoint")
            metrics_path = checkpoint.parent / "metrics.json"
            metrics = {
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "text_eval_provenance": {
                    "source_checkpoint": str(checkpoint.resolve()),
                    "source_checkpoint_sha256": sha256_file(checkpoint),
                },
            }
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            validate_e3_metrics_checkpoint_identity(
                metrics, checkpoint, metrics_path
            )

            for label, mutate in (
                (
                    "E3 metrics checkpoint path mismatch",
                    lambda value: value.__setitem__(
                        "checkpoint_path", str(copied_checkpoint.resolve())
                    ),
                ),
                (
                    "E3 metrics checkpoint SHA-256 mismatch",
                    lambda value: value.__setitem__(
                        "checkpoint_sha256", sha256_file(copied_checkpoint)
                    ),
                ),
                (
                    "embedded text provenance checkpoint path mismatch",
                    lambda value: value["text_eval_provenance"].__setitem__(
                        "source_checkpoint", str(copied_checkpoint.resolve())
                    ),
                ),
                (
                    "embedded text provenance checkpoint SHA-256 mismatch",
                    lambda value: value["text_eval_provenance"].__setitem__(
                        "source_checkpoint_sha256", sha256_file(copied_checkpoint)
                    ),
                ),
            ):
                with self.subTest(label=label):
                    drifted = json.loads(json.dumps(metrics))
                    mutate(drifted)
                    with self.assertRaisesRegex(ValueError, label):
                        validate_e3_metrics_checkpoint_identity(
                            drifted, checkpoint, metrics_path
                        )


    def test_audit_recomputes_checkpoint_state_instead_of_trusting_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.audit_fixture(Path(directory))
            valid = validate_e3_checkpoint_sidecar(
                fixture["sidecar"],
                fixture["checkpoint"],
                fixture["metrics_path"],
                metrics_rows=2,
                raw_training_path=fixture["raw_path"],
            )
            self.assertTrue(valid["passed"], valid["errors"])

            forged = dict(fixture["sidecar"])
            forged["checkpoint_state_keys"] = ["args", "last_row", "forged"]
            forged["checkpoint_last_row_sha256"] = "0" * 64
            forged["checkpoint_last_row_matches_metrics_last_row"] = True
            result = validate_e3_checkpoint_sidecar(
                forged,
                fixture["checkpoint"],
                fixture["metrics_path"],
                metrics_rows=2,
                raw_training_path=fixture["raw_path"],
            )
            self.assertFalse(result["passed"])
            self.assertIn("checkpoint sidecar state keys mismatch", result["errors"])
            self.assertIn(
                "checkpoint sidecar last-row SHA-256 mismatch", result["errors"]
            )

    def test_audit_rejects_replaced_prefix_even_with_updated_raw_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.audit_fixture(Path(directory))
            replaced = [
                {"step": 1, "modality": "image", "loss": -99.0},
                fixture["rows"][-1],
            ]
            fixture["raw_path"].write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in replaced),
                encoding="utf-8",
            )
            forged = dict(fixture["sidecar"])
            forged["raw_training_sha256"] = sha256_file(fixture["raw_path"])
            forged["training_rows_sha256"] = canonical_sha256(replaced)

            result = validate_e3_checkpoint_sidecar(
                forged,
                fixture["checkpoint"],
                fixture["metrics_path"],
                metrics_rows=2,
                raw_training_path=fixture["raw_path"],
            )
            self.assertFalse(result["passed"])
            self.assertIn(
                "E3 metrics steps do not equal raw training full row sequence",
                result["errors"],
            )

    def test_audit_rejects_minimal_fake_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.audit_fixture(Path(directory))
            result = validate_e3_checkpoint_sidecar(
                {"schema_version": 1, "passed": True},
                fixture["checkpoint"],
                fixture["metrics_path"],
                metrics_rows=2,
                raw_training_path=fixture["raw_path"],
            )
            self.assertFalse(result["passed"])
            self.assertIn("checkpoint sidecar state keys mismatch", result["errors"])
            self.assertIn(
                "checkpoint sidecar full training-row digest mismatch",
                result["errors"],
            )



if __name__ == "__main__":
    unittest.main()
