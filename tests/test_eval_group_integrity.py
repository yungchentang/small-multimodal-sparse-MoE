"""Focused tests for image-group integrity in conditional retrieval evaluation."""

import unittest
from pathlib import Path

import torch

from scripts.eval_conditional_retrieval import (
    group_aware_chance_r_at_1,
    image_group_identity,
    lexical_hard_negative_indices,
    local_candidate_indices,
    multi_positive_nll_evidence,
    multi_positive_ranks,
    positive_indices_for_group,
)


class EvalGroupIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"caption": "red bird on fence"},
            {"caption": "red bird beside fence"},
            {"caption": "red bird on rail"},
            {"caption": "blue ocean and sky"},
            {"caption": "green field at noon"},
            {"caption": "city street at night"},
        ]
        self.group_ids = [
            "image-a",
            "image-a",
            "image-b",
            "image-c",
            "image-d",
            "image-e",
        ]

    def test_image_group_identity_prefers_source_image_fields_then_path(self) -> None:
        first = {
            "source": "coco",
            "image_id": 42,
            "image_path": "/tmp/copied-a.png",
        }
        duplicate = {
            "source": "coco",
            "image_id": 42,
            "image_path": "/tmp/copied-b.png",
        }
        other = {
            "source": "coco",
            "image_id": 43,
            "image_path": "/tmp/copied-a.png",
        }
        self.assertEqual(image_group_identity(first), image_group_identity(duplicate))
        self.assertNotEqual(image_group_identity(first), image_group_identity(other))

        path = Path("/tmp/shared-image.png")
        self.assertEqual(
            image_group_identity({"image_path": str(path)}),
            image_group_identity({"image_path": str(path.resolve())}),
        )

    def test_local_candidates_exclude_every_query_group_caption(self) -> None:
        for mode in ("stride", "random"):
            with self.subTest(mode=mode):
                indices, gold_position = local_candidate_indices(
                    self.rows,
                    index=0,
                    negatives=4,
                    mode=mode,
                    candidate_seed=17,
                    group_ids=self.group_ids,
                )
                self.assertEqual(indices[gold_position], 0)
                self.assertEqual(len(indices), 5)
                self.assertNotIn(1, indices)
                self.assertEqual(
                    [self.group_ids[index] for index in indices].count("image-a"),
                    1,
                )

    def test_hard_text_selector_only_sees_cross_group_candidates(self) -> None:
        texts = [row["caption"] for row in self.rows]
        hard_indices = lexical_hard_negative_indices(
            texts,
            negatives=3,
            group_ids=self.group_ids,
        )
        self.assertNotIn(1, hard_indices[0])
        self.assertTrue(
            all(self.group_ids[index] != self.group_ids[0] for index in hard_indices[0])
        )

        candidates, _ = local_candidate_indices(
            self.rows,
            index=0,
            negatives=3,
            mode="hard_text",
            hard_indices=hard_indices[0],
            group_ids=self.group_ids,
        )
        self.assertNotIn(1, candidates)

    def test_local_candidates_fail_closed_when_cross_group_pool_is_short(self) -> None:
        rows = self.rows[:3]
        with self.assertRaisesRegex(RuntimeError, "cross-group"):
            local_candidate_indices(
                rows,
                index=0,
                negatives=2,
                group_ids=["image-a", "image-a", "image-b"],
            )

    def test_full_bank_uses_all_group_captions_as_positives(self) -> None:
        candidate_groups = ["image-a", "image-b", "image-a", "image-c", "image-c"]
        positives = positive_indices_for_group(candidate_groups, "image-a")
        evidence = multi_positive_nll_evidence(
            [0.4, 0.1, 0.2, 0.2, 0.8],
            positives,
            tie_epsilon=1e-8,
        )

        self.assertEqual(positives, [0, 2])
        self.assertEqual(evidence["strict_rank"], 2)
        self.assertAlmostEqual(evidence["reciprocal_rank"], 1.0 / 3.0)
        self.assertAlmostEqual(evidence["best_positive_nll"], 0.2)
        self.assertAlmostEqual(evidence["best_positive_nll_margin"], -0.1)
        self.assertEqual(evidence["gold_tie_count"], 1)

    def test_multi_positive_evidence_is_permutation_invariant(self) -> None:
        scores = [0.4, 0.1, 0.2, 0.2, 0.8]
        positives = [0, 2]
        expected = multi_positive_nll_evidence(scores, positives, tie_epsilon=1e-8)
        permutation = [3, 0, 4, 2, 1]
        permuted_scores = [scores[index] for index in permutation]
        permuted_positives = [
            position
            for position, original_index in enumerate(permutation)
            if original_index in positives
        ]
        observed = multi_positive_nll_evidence(
            permuted_scores,
            permuted_positives,
            tie_epsilon=1e-8,
        )
        for key in (
            "strict_rank",
            "strict_r_at_1",
            "reciprocal_rank",
            "best_positive_nll",
            "best_positive_nll_margin",
            "gold_tie_count",
        ):
            self.assertEqual(observed[key], expected[key])

    def test_positive_ties_are_not_counted_as_negatives(self) -> None:
        evidence = multi_positive_nll_evidence(
            [0.2, 0.4, 0.2],
            positive_indices=[0, 2],
            tie_epsilon=1e-8,
        )
        self.assertEqual(evidence["strict_rank"], 0)
        self.assertEqual(evidence["reciprocal_rank"], 1.0)
        self.assertEqual(evidence["gold_tie_count"], 0)
        self.assertEqual(evidence["best_positive_indices"], [0, 2])

    def test_strict_multi_positive_ties_and_group_aware_chance(self) -> None:
        ranks = multi_positive_ranks(
            torch.tensor([[0.9, 0.9, 0.2, 0.1]]),
            positive_indices=[[1]],
        )
        self.assertEqual(ranks, [1])
        self.assertAlmostEqual(
            group_aware_chance_r_at_1(
                ["image-a", "image-a", "image-b", "image-c", "image-c"]
            ),
            1.0 / 3.0,
        )


if __name__ == "__main__":
    unittest.main()
