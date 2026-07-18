import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts import build_final_conference_report as report
from scripts import summarize_single_seed_continuations as summary


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class SingleSeedContinuationTest(unittest.TestCase):
    def fixture(self, root: Path):
        checkpoints = {}
        for role in ("e3", "e4", "e5"):
            checkpoint = root / f"{role}.pt"
            checkpoint.write_bytes(f"{role}-checkpoint\n".encode("ascii"))
            checkpoints[role] = checkpoint
        e3_sha = summary.sha256_file(checkpoints["e3"])
        e3_metrics = write_json(root / "e3_metrics.json", {"checkpoint_sha256": e3_sha})

        text_paths = {}
        for role, ppl, accuracy, overflow, inactive in (
            ("e3", 11.0, 0.54, 0.01, 0.08),
            ("e4", 12.0, 0.52, 0.03, 0.12),
            ("e5", 13.0, 0.50, 0.08, 0.20),
        ):
            text_paths[role] = write_json(
                root / f"{role}_text.json",
                {
                    "perplexity": ppl,
                    "next_token_accuracy": accuracy,
                    "capacity_overflow_ratio_mean": overflow,
                    "inactive_expert_ratio_mean": inactive,
                },
            )

        metrics_paths = {}
        for role in ("E4", "E5"):
            key = role.lower()
            checkpoint = checkpoints[key]
            metrics_paths[key] = write_json(
                root / f"{key}_metrics.json",
                {
                    "schema_version": 1,
                    "artifact_type": "matched_ablation_final",
                    "experiment_role": role,
                    "source_selected_checkpoint_sha256": e3_sha,
                    "checkpoint": {
                        "path": str(checkpoint.resolve()),
                        "sha256": summary.sha256_file(checkpoint),
                        "bytes": checkpoint.stat().st_size,
                    },
                    "training_iterations": 300,
                    "optimizer_step_count": 300,
                    "frozen_text_row_count": 0,
                    "steps": [
                        {
                            "step": step,
                            "optimizer_step": True,
                            "modality": "image",
                            "loss": 2.0,
                        }
                        for step in range(1, 301)
                    ],
                },
            )
        args = argparse.Namespace(
            e3_metrics=e3_metrics,
            e3_text_eval=text_paths["e3"],
            e4_metrics=metrics_paths["e4"],
            e4_checkpoint=checkpoints["e4"],
            e4_text_eval=text_paths["e4"],
            e5_metrics=metrics_paths["e5"],
            e5_checkpoint=checkpoints["e5"],
            e5_text_eval=text_paths["e5"],
            seed=42,
        )
        return args

    def test_producer_and_report_validator_accept_single_seed_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = summary.build_summary(self.fixture(root))
            output = write_json(root / "single_seed.json", payload)
            validated = report.validate_matched_ablations(output)
            self.assertTrue(validated["single_seed_only"])
            self.assertFalse(validated["cross_seed_stability_claim"])
            self.assertFalse(validated["training_budget_matched"])
            self.assertEqual(validated["seeds"], [42])
            self.assertEqual(len(validated["paired_delta_summary"]), 8)

    def test_validator_rejects_unsafe_scope_and_artifact_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            payload = summary.build_summary(args)
            unsafe = copy.deepcopy(payload)
            unsafe["cross_seed_stability_claim"] = True
            unsafe_path = write_json(root / "unsafe.json", unsafe)
            with self.assertRaisesRegex(report.ReportInputError, "scope flags"):
                report.validate_matched_ablations(unsafe_path)

            drift = copy.deepcopy(payload)
            drift["artifacts"]["e4_metrics"]["sha256"] = "0" * 64
            drift_path = write_json(root / "drift.json", drift)
            with self.assertRaisesRegex(report.ReportInputError, "SHA-256"):
                report.validate_matched_ablations(drift_path)


    def test_accepts_only_explicit_frozen_text_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            metrics = json.loads(args.e4_metrics.read_text(encoding="utf-8"))
            for index, row in enumerate(metrics["steps"]):
                if index % 4 == 0:
                    row.update(
                        {
                            "optimizer_step": False,
                            "modality": "text",
                            "train_router_gates": False,
                            "train_experts": False,
                            "train_lm_head": False,
                        }
                    )
            metrics["optimizer_step_count"] = 225
            metrics["frozen_text_row_count"] = 75
            write_json(args.e4_metrics, metrics)
            payload = summary.build_summary(args)
            output = write_json(root / "single_seed_noop.json", payload)
            self.assertEqual(
                report.validate_matched_ablations(output)["seeds"], [42]
            )

            metrics["steps"][0]["modality"] = "image"
            write_json(args.e4_metrics, metrics)
            with self.assertRaisesRegex(summary.SummaryError, "unexplained"):
                summary.build_summary(args)


if __name__ == "__main__":
    unittest.main()
