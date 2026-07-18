from __future__ import annotations

import copy
import unittest

from scripts.run_matched_ablations import compare_arm_protocols, verify_matched_arm_contracts


def make_arms():
    common = {
        "max_steps": 300,
        "seed": 17,
        "initialization_policy": "reset_global_seed_before_each_matched_arm",
        "expected_data_order_sha256": "data-order",
        "matched_args": {
            "learning_rate": 5e-4,
            "train_batch_size": 4,
            "modality_cycle": "text,image,image,speech,speech",
        },
    }
    e3 = {**common, "experiment_id": "E3_aux_cap7_300", "aux_coef": 0.02, "capacity_factor": 7.0}
    e4 = {**common, "experiment_id": "E4_noaux_cap7_300", "aux_coef": 0.0, "capacity_factor": 7.0}
    e5 = {**common, "experiment_id": "E5_aux_cap1p25_300", "aux_coef": 0.02, "capacity_factor": 1.25}
    return [e3, e4, e5]


class MatchedAblationProtocolTest(unittest.TestCase):
    def test_permits_only_intended_aux_and_capacity_differences(self):
        self.assertEqual(verify_matched_arm_contracts(make_arms()), [])

    def test_compare_catches_forbidden_difference(self):
        reference, candidate, _ = make_arms()
        candidate = copy.deepcopy(candidate)
        candidate["matched_args"]["learning_rate"] = 1e-3
        forbidden = compare_arm_protocols(
            reference,
            candidate,
            {"experiment_id", "aux_coef"},
        )
        self.assertEqual(forbidden, ["matched_args.learning_rate"])

    def test_capacity_change_is_forbidden_between_e3_and_e4(self):
        arms = make_arms()
        arms[1]["capacity_factor"] = 1.25
        errors = verify_matched_arm_contracts(arms)
        self.assertEqual(len(errors), 1)
        self.assertIn("capacity_factor", errors[0])


if __name__ == "__main__":
    unittest.main()
