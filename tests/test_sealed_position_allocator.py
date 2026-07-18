from __future__ import annotations

import copy
import unittest

from scripts.sealed_position_allocator import (
    ALLOCATOR_NAME,
    ALLOCATOR_VERSION,
    AssignmentPlanError,
    build_allocator_manifest,
    canonical_sha256,
    enforce_gold_position_assignment,
    positions_for_query_indices,
    validate_allocator_manifest,
    validate_executed_positions,
)


# Count histograms and position-ledger hashes observed in
# outputs/final_sealed_analysis_s42_260713a/cells/*/aligned_queries.jsonl.
UNBALANCED_S42_260713A = {
    ("r5", "image"): ({37: 1, 40: 1, 55: 1, 56: 1, 62: 1}, "8cb5e51a482729542c1b0bb97a01fb64dc2fdb22d42d68bc641e57168be74b12"),
    ("r5", "speech"): ({39: 1, 48: 2, 55: 1, 60: 1}, "96da6592b62c4796639b80cd3fd543718ca82ce39a48983e35d8554085b759d9"),
    ("r10", "image"): ({19: 1, 22: 1, 24: 2, 25: 2, 27: 2, 28: 1, 29: 1}, "f62fc3f1682b6b13539d3a91985b81a0444d7d0de79b1ae1b56281bd0c1827d2"),
    ("r10", "speech"): ({21: 1, 22: 1, 23: 2, 25: 1, 26: 1, 27: 3, 29: 1}, "6e023015bc4cc48439b8d2f6afbc3c76b20a945cfd0cb00dbd61c5c72b5e4f7e"),
    ("h10", "image"): ({19: 1, 22: 1, 24: 2, 25: 2, 27: 2, 28: 1, 29: 1}, "f62fc3f1682b6b13539d3a91985b81a0444d7d0de79b1ae1b56281bd0c1827d2"),
    ("h10", "speech"): ({21: 1, 22: 1, 23: 2, 25: 1, 26: 1, 27: 3, 29: 1}, "6e023015bc4cc48439b8d2f6afbc3c76b20a945cfd0cb00dbd61c5c72b5e4f7e"),
    ("f250", "image"): ({0: 98, 1: 82, 2: 49, 3: 14, 4: 7}, "1572b74b91d0f9baf5b4a04933c8092038a1833882c9fe24f6a4921f5ee30d96"),
    ("f250", "speech"): ({0: 92, 1: 92, 2: 44, 3: 18, 4: 4}, "4c9af5d3b64b66458d8eb9179ec090d006b01b966cdcaa4d92faae3232aa3b62"),
}


CELLS = [
    {"id": "r5", "candidate_count": 5},
    {"id": "r10", "candidate_count": 10},
    {"id": "h10", "candidate_count": 10},
    {"id": "f250", "candidate_count": 250},
]


def counts_from_histogram(histogram: dict[int, int]) -> list[int]:
    return [
        count
        for count, number_of_positions in sorted(histogram.items())
        for _ in range(number_of_positions)
    ]


class SealedPositionAllocatorTest(unittest.TestCase):
    def test_enforcement_preserves_permutation_provenance(self) -> None:
        base = [10, 11, 12, 13, 14]
        permutation = [2, 4, 0, 3, 1]
        permuted = [base[index] for index in permutation]
        assigned, applied_permutation, gold_position = (
            enforce_gold_position_assignment(
                permuted,
                permutation,
                gold_candidate_index=10,
                assigned_position=4,
            )
        )
        self.assertEqual(gold_position, 4)
        self.assertEqual(assigned[gold_position], 10)
        self.assertEqual(
            assigned,
            [base[index] for index in applied_permutation],
        )

    def test_enforcement_rejects_non_bijective_permutation(self) -> None:
        with self.assertRaisesRegex(
            AssignmentPlanError, "must be a bijection"
        ):
            enforce_gold_position_assignment(
                [10, 11, 12],
                [0, 0, 2],
                gold_candidate_index=10,
                assigned_position=1,
            )

    def test_existing_unbalanced_fixture_is_rejected(self) -> None:
        manifest = build_allocator_manifest(
            CELLS,
            candidate_seed=314159,
            query_counts={"image": 250, "speech": 250},
        )
        plans = {
            (plan["cell_id"], plan["modality"]): plan
            for plan in manifest["plans"]
        }
        for key, (histogram, ledger_sha256) in UNBALANCED_S42_260713A.items():
            with self.subTest(cell=key[0], modality=key[1], sha256=ledger_sha256):
                counts = counts_from_histogram(histogram)
                self.assertEqual(len(counts), plans[key]["candidate_count"])
                self.assertEqual(sum(counts), 250)
                self.assertGreater(max(counts) - min(counts), 1)
                positions = [
                    position
                    for position, count in enumerate(counts)
                    for _ in range(count)
                ]
                with self.assertRaisesRegex(
                    AssignmentPlanError,
                    "executed gold positions disagree",
                ):
                    validate_executed_positions(positions, plans[key])

    def test_generated_assignments_are_deterministic_seed_bound_and_balanced(self) -> None:
        first = build_allocator_manifest(
            CELLS,
            candidate_seed=314159,
            query_counts={"image": 250, "speech": 250},
        )
        repeated = build_allocator_manifest(
            CELLS,
            candidate_seed=314159,
            query_counts={"image": 250, "speech": 250},
        )
        changed_seed = build_allocator_manifest(
            CELLS,
            candidate_seed=314160,
            query_counts={"image": 250, "speech": 250},
        )
        self.assertEqual(first, repeated)
        self.assertNotEqual(first["plans_sha256"], changed_seed["plans_sha256"])
        self.assertEqual(first["name"], ALLOCATOR_NAME)
        self.assertEqual(first["version"], ALLOCATOR_VERSION)

        plans = validate_allocator_manifest(
            first,
            CELLS,
            candidate_seed=314159,
            query_counts={"image": 250, "speech": 250},
        )
        self.assertEqual(len(plans), 8)
        for plan in plans.values():
            positions = positions_for_query_indices(plan, range(250))
            counts = [
                positions.count(position)
                for position in range(plan["candidate_count"])
            ]
            self.assertLessEqual(max(counts) - min(counts), 1)
            validate_executed_positions(positions, plan)

    def test_rejects_rehashed_mutually_consistent_unbalanced_manifest(self) -> None:
        manifest = build_allocator_manifest(
            CELLS,
            candidate_seed=314159,
            query_counts={"image": 250, "speech": 250},
        )
        tampered = copy.deepcopy(manifest)
        plan = tampered["plans"][0]
        plan["positions"] = [0] * plan["query_count"]
        plan["position_counts"] = [plan["query_count"]] + [
            0
        ] * (plan["candidate_count"] - 1)
        plan["positions_sha256"] = canonical_sha256(plan["positions"])
        tampered["plans_sha256"] = canonical_sha256(tampered["plans"])
        with self.assertRaisesRegex(
            AssignmentPlanError, "frozen seed-bound plan"
        ):
            validate_allocator_manifest(
                tampered,
                CELLS,
                candidate_seed=314159,
                query_counts={"image": 250, "speech": 250},
            )


if __name__ == "__main__":
    unittest.main()
