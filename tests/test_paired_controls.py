from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import analyze_paired_controls as paired


def make_row(
    query_uid: str,
    control: str,
    scores: list[float],
    gold_position: int,
    modality: str = "image",
    candidate_ids: list[str] | None = None,
) -> dict[str, object]:
    candidate_ids = candidate_ids or ["a", "b", "c", "d"]
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    return {
        "modality": modality,
        "query_uid": query_uid,
        "candidate_set_hash": hashlib.sha256(
            json.dumps(
                candidate_ids, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "rank": order.index(gold_position),
        "rank_base": 0,
        "scores": scores,
        "score_direction": "higher_is_better",
        "gold_position": gold_position,
        "predicted_position": order[0],
        "candidate_ids": candidate_ids,
        "gold_candidate_id": candidate_ids[gold_position],
        "control": control,
        "protocol": {
            "name": "conditional_matching_v1",
            "rank_base": 0,
            "score_direction": "higher_is_better",
            "tie_epsilon": 0.0,
        },
        "seed": 11,
    }


class PairedControlsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_rows(self, name: str, rows: list[dict[str, object]]) -> Path:
        path = self.root / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_alignment_rejects_missing_and_duplicate_keys(self) -> None:
        real_rows = [
            make_row("q1", "real", [4, 3, 2, 1], 0),
            make_row("q2", "real", [1, 4, 3, 2], 1),
        ]
        real = self.write_rows("real", real_rows)
        missing = self.write_rows("zero_missing", [make_row("q1", "zero", [4, 3, 2, 1], 0)])
        with self.assertRaisesRegex(paired.AnalysisError, "paired key mismatch"):
            paired.analyze({"real": real, "zero": missing}, bootstrap_samples=20, permutation_samples=20)

        duplicate = self.write_rows(
            "zero_duplicate",
            [
                make_row("q1", "zero", [4, 3, 2, 1], 0),
                make_row("q1", "zero", [4, 3, 2, 1], 0),
            ],
        )
        with self.assertRaisesRegex(paired.AnalysisError, "duplicate paired key"):
            paired.analyze({"real": self.write_rows("one_real", real_rows[:1]), "zero": duplicate})

    def test_alignment_rejects_candidate_identity_mismatch(self) -> None:
        real = make_row("q1", "real", [4, 3, 2, 1], 0)
        control = make_row("q1", "zero", [4, 3, 2, 1], 0)
        control["candidate_ids"] = ["a", "b", "c", "different"]
        control["gold_candidate_id"] = "a"
        control["candidate_set_hash"] = hashlib.sha256(
            json.dumps(
                control["candidate_ids"],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(paired.AnalysisError, "paired key mismatch"):
            paired.analyze(
                {
                    "real": self.write_rows("identity_real", [real]),
                    "zero": self.write_rows("identity_zero", [control]),
                }
            )

    def test_exact_metrics_and_wilson_counts(self) -> None:
        metrics = paired.retrieval_metrics([1, 2, 6, 11])
        self.assertEqual(metrics["n"], 4)
        self.assertEqual(metrics["r_at_1"]["count"], 1)
        self.assertEqual(metrics["r_at_5"]["count"], 2)
        self.assertEqual(metrics["r_at_10"]["count"], 3)
        self.assertEqual(metrics["r_at_1"]["total"], 4)
        self.assertAlmostEqual(metrics["r_at_1"]["rate"], 0.25)
        self.assertAlmostEqual(metrics["mrr"], (1 + 1 / 2 + 1 / 6 + 1 / 11) / 4)
        self.assertEqual(metrics["median_rank"], 4.0)
        self.assertAlmostEqual(metrics["r_at_1"]["wilson_95"]["low"], 0.0455872608)
        self.assertAlmostEqual(metrics["r_at_1"]["wilson_95"]["high"], 0.6993581574)

    def test_rejects_scoreless_and_noncanonical_identity_rows(self) -> None:
        base = make_row("q1", "real", [4.0, 3.0, 2.0, 1.0], 0)
        cases = (
            ("scores must be a finite list", lambda row: row.__setitem__("scores", None)),
            (
                "query_uid must be a non-empty string",
                lambda row: row.__setitem__("query_uid", 12345),
            ),
            ("rank must be an integer", lambda row: row.__setitem__("rank", 0.9)),
            (
                "gold_position must be an integer",
                lambda row: row.__setitem__("gold_position", 0.9),
            ),
        )
        for expected, mutate in cases:
            row = json.loads(json.dumps(base))
            mutate(row)
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(paired.AnalysisError, expected):
                    paired.validate_per_query_identity_row(
                        "real", "memory", 1, row
                    )

    def test_accepts_pessimistic_ties_and_multi_positive_rank(self) -> None:
        tied = make_row("tie", "real", [1.0, 1.0, 1.0, 1.0], 0)
        tied["rank"] = 3
        tied["predicted_position"] = 0
        tied["positive_indices"] = [0]
        tied["positive_candidate_ids"] = [tied["candidate_ids"][0]]
        loaded = paired.validate_per_query_identity_row(
            "real", "memory", 1, tied
        )
        self.assertEqual(loaded.rank_one_based, 4)

        multi = make_row("multi", "real", [1.0, 1.0, 1.0, 1.0], 0)
        multi["rank"] = 2
        multi["predicted_position"] = 0
        multi["positive_indices"] = [0, 1]
        multi["positive_candidate_ids"] = [
            multi["candidate_ids"][0],
            multi["candidate_ids"][1],
        ]
        loaded = paired.validate_per_query_identity_row(
            "real", "memory", 1, multi
        )
        self.assertEqual(loaded.rank_one_based, 3)

    def test_rejects_row_positive_indices_against_immutable_override(self) -> None:
        row = make_row("immutable", "real", [4.0, 3.0, 2.0, 1.0], 0)
        row["positive_indices"] = [0, 1]
        row["positive_candidate_ids"] = [
            row["candidate_ids"][0],
            row["candidate_ids"][1],
        ]
        with self.assertRaisesRegex(
            paired.AnalysisError, "immutable positive identities"
        ):
            paired.validate_per_query_identity_row(
                "real",
                "memory",
                1,
                row,
                expected_positive_indices=[0],
            )

    def test_reconstructs_and_rejects_every_production_derived_field(self) -> None:
        row = make_row(
            "multi-epsilon",
            "real",
            [1.0, 1.0 - 0.5e-8, 1.0 - 0.5e-8, 0.5],
            0,
        )
        row["protocol"]["tie_epsilon"] = 1e-8
        row["candidate_indices"] = [10, 11, 12, 13]
        row["positive_indices"] = [0, 1]
        row["positive_candidate_ids"] = [
            row["candidate_ids"][0],
            row["candidate_ids"][1],
        ]
        loaded = paired.reconstruct_and_validate_per_query_row(
            "real",
            "memory",
            1,
            row,
            validate_production_fields=False,
        )
        self.assertEqual(loaded.production_fields["strict_rank"], 1)
        self.assertEqual(loaded.production_fields["strict_r_at_1"], 0.0)
        self.assertEqual(loaded.production_fields["reciprocal_rank"], 0.5)
        self.assertEqual(loaded.production_fields["positive_count"], 2)
        self.assertEqual(
            loaded.production_fields["best_candidate_indices"], [0, 1, 2]
        )
        self.assertEqual(
            loaded.production_fields["best_positive_indices"], [0, 1]
        )
        self.assertEqual(loaded.production_fields["gold_tie_count"], 1)
        row.update(loaded.production_fields)
        paired.validate_per_query_identity_row(
            "real",
            "memory",
            1,
            row,
            require_production_fields=True,
        )

        for field, expected in loaded.production_fields.items():
            corrupted = json.loads(json.dumps(row))
            if isinstance(expected, list):
                corrupted[field] = list(expected) + ["corrupt"]
            elif isinstance(expected, str):
                corrupted[field] = expected + "-corrupt"
            else:
                corrupted[field] = expected + 1
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    paired.AnalysisError,
                    field,
                ):
                    paired.validate_per_query_identity_row(
                        "real",
                        "memory",
                        1,
                        corrupted,
                        require_production_fields=True,
                    )

        wrong_type = json.loads(json.dumps(row))
        wrong_type["strict_r_at_1"] = False
        with self.assertRaisesRegex(
            paired.AnalysisError,
            "strict_r_at_1 disagrees with canonical score reconstruction",
        ):
            paired.validate_per_query_identity_row(
                "real",
                "memory",
                1,
                wrong_type,
                require_production_fields=True,
            )

    def test_requires_complete_production_derived_fields(self) -> None:
        row = make_row("q1", "real", [4.0, 3.0, 2.0, 1.0], 0)
        with self.assertRaisesRegex(
            paired.AnalysisError, "missing production-derived fields"
        ):
            paired.validate_per_query_identity_row(
                "real",
                "memory",
                1,
                row,
                require_production_fields=True,
            )

    def test_positive_position_balance(self) -> None:
        rows = []
        for index, position in enumerate([0, 1, 2, 3, 0, 1, 2, 3]):
            row = make_row(f"q{index}", "real", [4, 3, 2, 1], position)
            loaded = paired._validate_row("real", "memory", index + 1, row)
            rows.append(loaded)
        balance = paired.positive_position_balance(rows)
        group = balance["by_candidate_count"]["4"]
        self.assertEqual(group["counts"], {"0": 2, "1": 2, "2": 2, "3": 2})
        self.assertTrue(group["balanced_within_one"])
        self.assertTrue(balance["balanced_within_one"])

    def test_null_subtraction_aligns_candidates_by_identity(self) -> None:
        real = make_row("q1", "real", [0.8, 0.75, 0.2], 1, candidate_ids=["a", "b", "c"])
        no_prefix = make_row(
            "q1",
            "no_prefix",
            [0.1, 0.5, 0.6],
            2,
            candidate_ids=["c", "a", "b"],
        )
        # Keep the paired gold identity as b despite the candidate reordering.
        no_prefix["gold_position"] = 2
        no_prefix["gold_candidate_id"] = "b"
        no_prefix["rank"] = 0
        no_prefix["predicted_position"] = 2
        real_loaded = paired._validate_row("real", "memory", 1, real)
        null_loaded = paired._validate_row("no_prefix", "memory", 1, no_prefix)
        result = paired.null_subtracted_result(real_loaded, null_loaded)
        self.assertEqual(result["candidate_ids"], ["a", "b", "c"])
        for actual, expected in zip(result["scores"], [0.3, 0.15, 0.1]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["predicted_position"], 0)
        self.assertEqual(result["rank_one_based"], 2)

    def test_resampling_is_deterministic(self) -> None:
        differences = [1.0, 0.0, -1.0, 1.0, 1.0]
        first = paired.paired_bootstrap_mean_ci(differences, samples=250, seed=123)
        second = paired.paired_bootstrap_mean_ci(differences, samples=250, seed=123)
        self.assertEqual(first, second)

        permutation_first = paired.paired_permutation_mean(differences, samples=17, seed=456)
        permutation_second = paired.paired_permutation_mean(differences, samples=17, seed=456)
        self.assertEqual(permutation_first, permutation_second)
        self.assertEqual(permutation_first["method"], "monte_carlo_sign_flip")

    def test_full_analysis_preserves_rows_and_is_deterministic_except_timestamp(self) -> None:
        real_rows = [
            make_row("q1", "real", [4, 3, 2, 1], 0),
            make_row("q2", "real", [2, 4, 3, 1], 1, modality="audio"),
        ]
        zero_rows = [
            make_row("q1", "zero", [3, 4, 2, 1], 0),
            make_row("q2", "zero", [4, 3, 2, 1], 1, modality="audio"),
        ]
        inputs = {
            "real": self.write_rows("full_real", real_rows),
            "zero": self.write_rows("full_zero", zero_rows),
        }
        first, paired_rows = paired.analyze(inputs, bootstrap_samples=100, permutation_samples=100, seed=99)
        second, _ = paired.analyze(inputs, bootstrap_samples=100, permutation_samples=100, seed=99)
        self.assertEqual(first["comparisons_vs_real"], second["comparisons_vs_real"])
        self.assertEqual(first["seeds"], second["seeds"])
        self.assertEqual(len(paired_rows), 2)
        preserved = {
            row["pair_key"]["query_uid"]: row["conditions"]["real"] for row in paired_rows
        }
        self.assertEqual(preserved, {row["query_uid"]: row for row in real_rows})
        self.assertEqual(first["conditions"]["real"]["overall"]["r_at_1"]["count"], 2)
        self.assertEqual(
            first["comparisons_vs_real"]["zero"]["overall"]["mcnemar_exact"]["p_value_two_sided"],
            0.5,
        )
        mrr = first["comparisons_vs_real"]["zero"]["overall"]["mrr_difference"]
        self.assertIn("bootstrap", mrr)
        self.assertIn("permutation", mrr)
        self.assertEqual(
            mrr,
            second["comparisons_vs_real"]["zero"]["overall"]["mrr_difference"],
        )


if __name__ == "__main__":
    unittest.main()
