import unittest
from types import SimpleNamespace

import torch

from model.olmoe_adapter import _set_runtime_top_k
from training.olmoe_required_runs import (
    _hard_load_stats,
    capacity_mask_with_accounting,
    modality_router_metrics,
    router_metrics,
    set_olmoe_runtime_routing,
)


class RoutingAccountingTest(unittest.TestCase):
    def test_runtime_setter_updates_model_level_aux_attributes(self):
        fake = SimpleNamespace(
            config=SimpleNamespace(
                num_experts_per_tok=8,
                output_router_logits=False,
                router_aux_loss_coef=0.01,
                norm_topk_prob=False,
            ),
            router_aux_loss_coef=0.01,
            num_experts_per_tok=8,
            model=SimpleNamespace(layers=[], num_experts_per_tok=8),
        )
        metadata = set_olmoe_runtime_routing(fake, top_k=2, aux_coef=0.02)
        self.assertEqual(fake.config.num_experts_per_tok, 2)
        self.assertEqual(fake.num_experts_per_tok, 2)
        self.assertEqual(fake.model.num_experts_per_tok, 2)
        self.assertAlmostEqual(fake.router_aux_loss_coef, 0.02)
        self.assertGreater(metadata["runtime_changed_attr_count"], 0)

    def test_optional_adapter_updates_model_level_aux_attributes(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                num_experts_per_tok=8,
                router_aux_loss_coef=0.01,
                output_router_logits=False,
            ),
            num_experts_per_tok=8,
            router_aux_loss_coef=0.01,
            model=SimpleNamespace(layers=[]),
        )
        _set_runtime_top_k(model, top_k=2, aux_coef=0.03, output_router_logits=True)
        self.assertEqual(model.num_experts_per_tok, 2)
        self.assertAlmostEqual(model.router_aux_loss_coef, 0.03)
        self.assertEqual(model.config.num_experts_per_tok, 2)
        self.assertAlmostEqual(model.config.router_aux_loss_coef, 0.03)

    def test_capacity_conservation_and_compliance(self):
        top_ids = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
        weights = torch.ones(4, 2)
        torch.manual_seed(7)
        masked, accounting = capacity_mask_with_accounting(
            top_ids,
            weights,
            num_experts=2,
            capacity_factor=0.5,
        )
        self.assertEqual(accounting["attempted_assignments"], 8)
        self.assertEqual(accounting["accepted_assignments"], 4)
        self.assertEqual(accounting["dropped_assignments"], 4)
        self.assertTrue(accounting["conservation_ok"])
        self.assertTrue(accounting["capacity_compliant"])
        self.assertTrue(torch.equal(
            accounting["attempted_expert_counts"],
            accounting["accepted_expert_counts"] + accounting["dropped_expert_counts"],
        ))
        self.assertEqual(int((masked != 0).sum().item()), 4)

    def test_seeded_tie_break_is_reproducible_and_not_fixed_order(self):
        top_ids = torch.zeros(12, 1, dtype=torch.long)
        weights = torch.ones(12, 1)
        patterns = set()
        for seed in range(8):
            torch.manual_seed(seed)
            _, accounting = capacity_mask_with_accounting(
                top_ids,
                weights,
                num_experts=2,
                capacity_factor=0.5,
            )
            patterns.add(tuple(accounting["accepted_mask"].flatten().tolist()))
        self.assertGreater(len(patterns), 1)
        torch.manual_seed(19)
        first, _ = capacity_mask_with_accounting(top_ids, weights, 2, 0.5)
        torch.manual_seed(19)
        second, _ = capacity_mask_with_accounting(top_ids, weights, 2, 0.5)
        self.assertTrue(torch.equal(first, second))

    def test_uniform_and_concentrated_load_statistics(self):
        uniform = _hard_load_stats(torch.tensor([10, 10, 10, 10]))
        concentrated = _hard_load_stats(torch.tensor([40, 0, 0, 0]))
        self.assertAlmostEqual(uniform["effective_experts"], 4.0, places=5)
        self.assertAlmostEqual(uniform["load_gini"], 0.0, places=6)
        self.assertEqual(concentrated["active_experts"], 1.0)
        self.assertGreater(concentrated["load_gini"], uniform["load_gini"])
        self.assertGreater(concentrated["load_cv"], uniform["load_cv"])

    def test_router_logits_fallback_conserves_tokens_layers_and_k(self):
        outputs = SimpleNamespace(router_logits=(torch.zeros(3, 4), torch.ones(3, 4)))
        metrics = router_metrics(
            outputs,
            top_k=2,
            num_experts=4,
            capacity_factor=1.0,
        )
        self.assertEqual(metrics["routing_accounting_source"], "router_logits_attempted_only")
        self.assertEqual(metrics["routing_token_count_across_layers"], 6)
        self.assertEqual(metrics["routing_expected_assignments_tokens_x_layers_x_k"], 12)
        self.assertEqual(metrics["routing_attempted_assignments_total"], 12)
        self.assertEqual(metrics["routing_accepted_assignments_total"], 12)
        self.assertEqual(metrics["routing_dropped_assignments_total"], 0)
        self.assertTrue(metrics["routing_token_k_conservation_ok"])
        self.assertTrue(metrics["routing_conservation_ok"])

    def test_router_and_modality_denominators(self):
        top_ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
        attempted = torch.ones_like(top_ids, dtype=torch.bool)
        accepted = attempted.clone()
        accepted[0, 1] = False
        dropped = attempted & ~accepted

        def counts(mask):
            return torch.bincount(top_ids[mask], minlength=4)

        snapshot = [{
            "layer": 0,
            "top_k_index": top_ids,
            "router_weights": torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]]),
            "attempted_mask": attempted,
            "pre_capacity_mask": attempted,
            "accepted_mask": accepted,
            "capacity_dropped_mask": dropped,
            "pre_capacity_dropped_mask": torch.zeros_like(dropped),
            "attempted_expert_counts": counts(attempted),
            "pre_capacity_expert_counts": counts(attempted),
            "accepted_expert_counts": counts(accepted),
            "capacity_dropped_expert_counts": counts(dropped),
            "pre_capacity_dropped_expert_counts": torch.zeros(4, dtype=torch.long),
            "num_tokens": 4,
            "top_k": 2,
            "capacity_per_expert": 2,
            "tie_break": "seeded_random_epsilon",
        }]
        outputs = SimpleNamespace(router_logits=(torch.zeros(4, 4),))
        aggregate = router_metrics(
            outputs,
            top_k=2,
            num_experts=4,
            capacity_factor=1.0,
            dispatch_snapshot=snapshot,
        )
        self.assertEqual(aggregate["routing_attempted_assignments_total"], 8)
        self.assertEqual(aggregate["routing_accepted_assignments_total"], 7)
        self.assertEqual(aggregate["routing_dropped_assignments_total"], 1)
        self.assertTrue(aggregate["routing_conservation_ok"])
        self.assertTrue(aggregate["routing_capacity_compliant"])
        self.assertEqual(aggregate["routing_expected_assignments_tokens_x_layers_x_k"], 8)
        self.assertTrue(aggregate["routing_token_k_conservation_ok"])

        modality = modality_router_metrics(
            outputs,
            top_k=2,
            num_experts=4,
            batch_size=1,
            image_prefix_tokens=1,
            audio_prefix_tokens=1,
            text_tokens=2,
            dispatch_snapshot=snapshot,
        )
        self.assertEqual(modality["modality_token_counts_across_layers"], {
            "image_prefix": 1,
            "audio_prefix": 1,
            "text": 2,
        })
        self.assertTrue(all(modality["modality_assignment_conservation"].values()))
        self.assertEqual(modality["modality_expected_assignments_tokens_x_layers_x_k"], 8)
        self.assertEqual(modality["modality_observed_assignments"], 8)
        self.assertTrue(modality["modality_token_k_conservation_ok"])
        self.assertEqual(modality["prefix_expected_assignments_tokens_x_layers_x_k"], 4)
        self.assertEqual(modality["prefix_observed_assignments"], 4)
        self.assertTrue(modality["prefix_routing_included"])
        self.assertEqual(
            sum(modality["modality_attempted_expert_counts"]["image_prefix"]),
            2,
        )
        self.assertEqual(
            sum(modality["modality_expert_counts"]["image_prefix"]),
            1,
        )
        image_layer = modality["modality_layer_accounting"][0]
        self.assertEqual(sum(image_layer["attempted_expert_counts"]), 2)
        self.assertEqual(len(image_layer["gate_score_sums"]), 4)
        self.assertAlmostEqual(sum(image_layer["gate_score_sums"]), 1.0)


if __name__ == "__main__":
    unittest.main()
