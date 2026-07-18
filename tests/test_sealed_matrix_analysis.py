from __future__ import annotations

import unittest

from scripts.analyze_sealed_matrix import build_claim_status, holm_adjust


def effect(raw_p: float, r1_low: float, mrr_low: float) -> dict:
    return {
        "r_at_1_difference": {"ci_95": {"low": r1_low, "high": 0.5}},
        "mcnemar_exact": {"p_value_two_sided": raw_p},
        "mrr_difference": {
            "bootstrap": {"ci_95": {"low": mrr_low, "high": 0.5}},
            "permutation": {"p_value_two_sided": raw_p},
        },
    }


class SealedMatrixAnalysisTest(unittest.TestCase):
    def test_holm_adjustment_is_monotone_and_blocks_raw_significance(self) -> None:
        adjusted = holm_adjust(
            [("a", 0.01), ("b", 0.02), ("c", 0.03), ("d", 0.20)],
            0.05,
        )
        values = [row["adjusted_p"] for row in adjusted]
        self.assertEqual(values, sorted(values))
        self.assertEqual(adjusted[0]["adjusted_p"], 0.04)
        self.assertFalse(adjusted[1]["rejected"])

    def test_claim_requires_adjusted_p_and_directional_intervals(self) -> None:
        controls = ["shuffled", "zero", "norm-matched-random", "no-prefix"]
        protocol = {
            "evaluation_matrix": [
                {"id": "h10", "candidate_count": 10, "negative_mode": "hard_text", "role": "primary"}
            ],
            "holm_families": [{"controls": controls}],
        }
        comparisons = {
            {"norm-matched-random": "random", "no-prefix": "no_prefix"}.get(name, name): {
                "by_modality": {modality: effect(0.01, 0.05, 0.02) for modality in ("image", "speech")}
            }
            for name in controls
        }
        report = {
            "conditions": {
                "real": {
                    "by_modality": {
                        modality: {"r_at_1": {"rate": 0.4, "wilson_95": {"low": 0.3}}}
                        for modality in ("image", "speech")
                    }
                }
            },
            "chance": {"by_modality": {modality: {"r_at_1": 0.1} for modality in ("image", "speech")}},
            "comparisons_vs_real": comparisons,
        }
        holm = []
        for family_id in ("primary_r1_mcnemar", "primary_mrr_permutation"):
            rows = []
            for modality in ("image", "speech"):
                for control in controls:
                    rows.append({
                        "label": f"h10:{modality}:{control}",
                        "adjusted_p": 0.08,
                        "rejected": False,
                    })
            holm.append({"id": family_id, "comparisons": rows})
        claim = build_claim_status("a" * 64, protocol, {"h10": report}, holm)
        self.assertEqual(claim["modalities"]["image"]["status"], "not_established")
        self.assertIn("not established", claim["modalities"]["image"]["statement"])


if __name__ == "__main__":
    unittest.main()
