import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import select_corrected_candidate as base
from scripts import select_multimodal_quality_followup as followup


def make_candidate(name, values, overflow=0.18, inactive=0.59):
    candidate = base.Candidate(name, Path("/tmp/run"), Path("/tmp/dev"))
    candidate.metrics = {
        key: value for key, value in zip(followup.R1_KEYS, values)
    }
    candidate.metrics.update(
        {
            "text_ppl": 12.5,
            "routing_overflow": overflow,
            "routing_inactive": inactive,
            "routing_load_cv": 0.6,
        }
    )
    followup.add_quality_components(candidate)
    return candidate


class FollowupDominanceTests(unittest.TestCase):
    def setUp(self):
        self.baseline = make_candidate(
            "baseline_cap8",
            [0.40, 0.30, 0.23, 0.12, 0.15, 0.11],
        )

    def test_balanced_speech_improvement_passes(self):
        candidate = make_candidate(
            "speech3rank19",
            [0.39, 0.31, 0.22, 0.15, 0.14, 0.14],
        )
        self.assertTrue(followup.passes_dominance(candidate, self.baseline))

    def test_speech_gain_is_required(self):
        candidate = make_candidate(
            "hard19",
            [0.42, 0.31, 0.24, 0.13, 0.18, 0.12],
        )
        self.assertFalse(followup.passes_dominance(candidate, self.baseline))

    def test_routing_regression_blocks_switch(self):
        candidate = make_candidate(
            "speech3rank49",
            [0.39, 0.31, 0.22, 0.15, 0.14, 0.14],
            overflow=self.baseline.metrics["routing_overflow"]
            + followup.MAX_ROUTING_REGRESSION
            + 0.001,
        )
        self.assertFalse(followup.passes_dominance(candidate, self.baseline))

    def test_policy_is_explicitly_presealed(self):
        self.assertEqual(
            set(followup.SELECTION_POLICY),
            {"eligibility", "authoritative_order", "display_score", "sealed_policy"},
        )
        self.assertIn("before", followup.SELECTION_POLICY["sealed_policy"])

    def test_candidate_registry_matches_predeclared_recipes(self):
        self.assertEqual(followup.EXPECTED_NAMES, set(followup.EXPECTED_VARIATIONS))
        self.assertEqual(
            followup.EXPECTED_VARIATIONS["speechlight49"],
            {
                "modality_cycle": "text,image,image,speech,speech",
                "conditional_ranking_negatives": 49,
                "conditional_ranking_negative_mode": "random",
                "image_conditional_ranking_coef": 1.0,
                "speech_conditional_ranking_coef": 3.0,
                "speech_contrastive_coef": 0.7,
            },
        )

    def test_h10_must_share_the_r5_r10_evaluator(self):
        candidate = make_candidate(
            "hard19", [0.4, 0.3, 0.23, 0.13, 0.15, 0.14]
        )
        candidate.artifacts["dev_5way_evaluator"] = {"sha256": "a" * 64}
        candidate.artifacts["dev_10way_evaluator"] = {"sha256": "a" * 64}

        def fake_validate(*args, **kwargs):
            candidate.artifacts["dev_h10_evaluator"] = {"sha256": "b" * 64}

        with patch.object(base, "_validate_eval_cell", side_effect=fake_validate):
            with self.assertRaises(base.ArtifactError) as raised:
                followup.validate_h10(candidate, base.ArtifactCache())
        self.assertEqual(raised.exception.code, "development_protocol")

    def test_candidate_payload_serializes_mixed_protocol_keys(self):
        candidate = make_candidate(
            "baseline_cap8", [0.4, 0.3, 0.23, 0.12, 0.15, 0.11]
        )
        candidate.protocol_digests = {5: "r5", 10: "r10", "h10": "hard10"}

        payload = base._candidate_payload(candidate, True)

        self.assertEqual(
            payload["development_protocol_sha256"],
            {"5": "r5", "10": "r10", "h10": "hard10"},
        )

    def test_public_outputs_include_h10_and_followup_policy(self):
        candidate = make_candidate(
            "baseline_cap8", [0.4, 0.3, 0.23, 0.12, 0.15, 0.11]
        )
        candidate.rank = 1
        report = {
            "schema_version": 1,
            "status": "selected",
            "selected_candidate": candidate.name,
            "candidate_count": 1,
            "valid_candidate_count": 1,
            "selection_policy": followup.SELECTION_POLICY,
            "candidates": [base._candidate_payload(candidate, True)],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selection"
            followup._write_followup_outputs(report, output, [candidate])
            with (output / "candidate_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                row = next(csv.DictReader(handle))
            markdown = (output / "candidate_selection.md").read_text(
                encoding="utf-8"
            )
        self.assertEqual(row["dev_h10_speech_r1"], "0.11")
        self.assertIn("worst_normalized_lift", row)
        self.assertIn("h10 speech", markdown)
        self.assertIn("retain baseline", markdown)

    def test_launcher_uses_import_safe_module_invocation(self):
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "scripts/select_final_quality_candidate.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "python -m scripts.select_multimodal_quality_followup", launcher
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.select_multimodal_quality_followup",
                "--help",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
