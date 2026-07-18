import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from training.olmoe_required_runs import (
    DevelopmentEvidenceError,
    build_esft_selection,
    configure_selected_full_expert_training,
    moe_reconstruction_diagnostics,
    selected_expert_anchor_loss,
    selected_expert_update_capability,
    validate_development_evidence,
)


class FakeExperts(nn.Module):
    def __init__(self, num_experts: int = 8, hidden_size: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        generator = torch.Generator().manual_seed(13)
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size * 2, hidden_size, generator=generator) * 0.2
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, hidden_size, generator=generator) * 0.2
        )
        self.act_fn = torch.tanh


class FakeMlp(nn.Module):
    def __init__(self, num_experts: int = 4) -> None:
        super().__init__()
        self.gate = nn.Linear(2, num_experts, bias=False)
        self.experts = FakeExperts(num_experts=num_experts)


class FakeLayer(nn.Module):
    def __init__(self, num_experts: int = 4) -> None:
        super().__init__()
        self.mlp = FakeMlp(num_experts=num_experts)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer(), FakeLayer()])
        self.embedding = nn.Linear(2, 2, bias=False)


class MoeReconstructionTest(unittest.TestCase):
    def test_topk_reconstruction_and_oracle_are_diagnostic(self) -> None:
        experts = FakeExperts(num_experts=8)
        hidden = torch.tensor([[0.2, -0.3], [0.7, 0.1], [-0.4, 0.5]])
        router_logits = torch.tensor([
            [4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1, -0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [1.1, -0.2, 2.2, 0.4, 0.3, 0.2, 0.1, -0.3],
        ])
        report = moe_reconstruction_diagnostics(
            experts,
            hidden,
            router_logits,
            top_ks=(2, 4, 8),
            oracle_candidate_k=8,
        )
        self.assertEqual(set(report["reconstructions"]), {"top_2", "top_4", "top_8"})
        self.assertAlmostEqual(report["reconstructions"]["top_8"]["mse"], 0.0, places=8)
        self.assertAlmostEqual(report["native_top8_equivalence_mse"], 0.0, places=8)
        self.assertLessEqual(
            report["reconstructions"]["top_2"]["router_mass_coverage_mean"],
            report["reconstructions"]["top_4"]["router_mass_coverage_mean"],
        )
        oracle = report["oracle_top2"]
        self.assertTrue(oracle["diagnostic_only"])
        self.assertFalse(oracle["inference_path"])
        self.assertLessEqual(oracle["mse"], oracle["router_selected_top2_mse"] + 1e-9)

    def test_esft_gate_and_token_selection_are_deterministic(self) -> None:
        rows = [
            {
                "split": "train",
                "modality": "image_prefix",
                "layer": 0,
                "top_k": 2,
                "token_count": 2,
                "attempted_expert_counts": [2, 2, 0, 0],
                "gate_score_sums": [0.2, 1.0, 0.0, 0.0],
            },
            {
                "split": "dev",
                "modality": "audio_prefix",
                "layer": 0,
                "top_k": 2,
                "token_count": 1,
                "attempted_expert_counts": [0, 1, 1, 0],
                "gate_score_sums": [0.0, 0.1, 0.9, 0.0],
            },
            {
                "split": "train",
                "modality": "image_prefix",
                "layer": 1,
                "top_k": 2,
                "token_count": 2,
                "attempted_expert_counts": [1, 1, 1, 1],
                "gate_score_sums": [0.5, 0.5, 0.5, 0.5],
            },
        ]
        first = build_esft_selection(rows, selected_experts_per_layer=2)
        second = build_esft_selection(rows, selected_experts_per_layer=2)
        self.assertEqual(first, second)
        self.assertEqual(
            first["methods"]["ESFT-Gate"]["0"]["selected_expert_ids"],
            [1, 2],
        )
        self.assertEqual(
            first["methods"]["ESFT-Token"]["0"]["selected_expert_ids"],
            [1, 0],
        )
        self.assertEqual(
            first["methods"]["ESFT-Gate"]["1"]["selected_expert_ids"],
            [0, 1],
        )
        accounting = first["routing_accounting"]
        self.assertEqual(accounting["prefix_tokens_across_layers"], 5)
        self.assertEqual(accounting["expected_assignments_tokens_x_layers_x_k"], 10)
        self.assertEqual(accounting["observed_assignments"], 10)
        self.assertTrue(accounting["conservation_ok"])

    def test_esft_rejects_non_prefix_and_non_conserving_rows(self) -> None:
        base = {
            "split": "train",
            "modality": "text",
            "layer": 0,
            "top_k": 2,
            "token_count": 1,
            "attempted_expert_counts": [1, 1],
            "gate_score_sums": [0.5, 0.5],
        }
        with self.assertRaises(DevelopmentEvidenceError):
            build_esft_selection([base], selected_experts_per_layer=1)
        broken = {**base, "modality": "image_prefix", "attempted_expert_counts": [1, 0]}
        with self.assertRaisesRegex(ValueError, "tokens x K conservation"):
            build_esft_selection([broken], selected_experts_per_layer=1)

    def test_development_provenance_rejects_sealed_and_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "dev-routing.jsonl"
            metadata = validate_development_evidence(
                [valid],
                [{"split": "dev", "real_subset": True, "modality": "image_prefix"}],
            )
            self.assertFalse(metadata["sealed_evidence_used"])
            with self.assertRaises(DevelopmentEvidenceError):
                validate_development_evidence(
                    [root / "sealed-routing.jsonl"],
                    [{"split": "dev", "real_subset": True}],
                )
            with self.assertRaises(DevelopmentEvidenceError):
                validate_development_evidence(
                    [valid],
                    [{"split": "train", "source": "synthetic_debug"}],
                )

    def test_selected_full_expert_training_masks_non_selected_rows(self) -> None:
        model = FakeModel()
        selected = {0: [1], 1: [2]}
        before = {
            layer_idx: layer.mlp.experts.gate_up_proj.detach().clone()
            for layer_idx, layer in enumerate(model.model.layers)
        }
        optimizer, anchors, handles, metadata = configure_selected_full_expert_training(
            model,
            selected,
            expert_learning_rate=1e-4,
            anchor_coefficient=0.1,
        )
        self.assertFalse(model.embedding.weight.requires_grad)
        self.assertFalse(model.model.layers[0].mlp.gate.weight.requires_grad)
        self.assertTrue(metadata["non_selected_experts_frozen"])
        self.assertEqual(metadata["weight_decay"], 0.0)
        self.assertEqual(metadata["lora_status"], "unavailable_fail_closed")

        objective = sum(
            parameter.sum()
            for parameter in model.parameters()
            if parameter.requires_grad
        ) + selected_expert_anchor_loss(model, anchors, coefficient=0.1)
        objective.backward()
        for layer_idx, layer in enumerate(model.model.layers):
            selected_id = selected[layer_idx][0]
            gradient = layer.mlp.experts.gate_up_proj.grad
            self.assertGreater(float(gradient[selected_id].abs().sum()), 0.0)
            non_selected = [idx for idx in range(4) if idx != selected_id]
            self.assertEqual(float(gradient[non_selected].abs().sum()), 0.0)
        optimizer.step()
        for layer_idx, layer in enumerate(model.model.layers):
            selected_id = selected[layer_idx][0]
            non_selected = [idx for idx in range(4) if idx != selected_id]
            self.assertTrue(torch.equal(
                layer.mlp.experts.gate_up_proj.detach()[non_selected],
                before[layer_idx][non_selected],
            ))
            self.assertFalse(torch.equal(
                layer.mlp.experts.gate_up_proj.detach()[selected_id],
                before[layer_idx][selected_id],
            ))
        for handle in handles:
            handle.remove()

    def test_lora_fallback_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no wired expert-LoRA"):
            selected_expert_update_capability("lora")


if __name__ == "__main__":
    unittest.main()
