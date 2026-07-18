from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


@unittest.skipIf(torch is None, "torch is not installed in the host test environment")
class SelectedExpertRealRunnerTest(unittest.TestCase):
    def make_selection(self, path: Path, selected_count: int = 2) -> dict:
        method_rows = {
            str(layer): {
                "splits": ["train"],
                "modalities": ["audio_prefix", "image_prefix"],
                "prefix_tokens": 10,
                "assignments": 20,
                "expert_scores": [
                    {
                        "expert_id": expert_id,
                        "gate_score_sum": float(4 - expert_id),
                        "gate_score_per_prefix_token": float(4 - expert_id) / 10.0,
                        "token_count": 4 - expert_id,
                        "token_frequency": float(4 - expert_id) / 10.0,
                    }
                    for expert_id in range(4)
                ],
                "selected_expert_ids": list(range(selected_count)),
            }
            for layer in range(2)
        }
        payload = {
            "artifact_type": "development_moe_reconstruction_and_esft_selection",
            "development_only": True,
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
            "model": {"base_model": "unit-test-model"},
            "esft_selection": {
                "selection_scope": "development_train_image_audio_prefix_only",
                "selected_experts_per_layer": selected_count,
                "routing_accounting": {
                    "expected_assignments_tokens_x_layers_x_k": 32,
                    "observed_assignments": 32,
                    "conservation_ok": True,
                },
                "methods": {
                    "ESFT-Gate": method_rows,
                    "ESFT-Token": method_rows,
                },
            },
            "provenance": {
                "routing": {
                    "policy": "development_only_real_train",
                    "splits": ["train"],
                    "source_paths": ["development/train-routing.jsonl"],
                    "source_files": [
                        {"path": "development/train-routing.jsonl", "sha256": "1" * 64},
                    ],
                    "sealed_evidence_used": False,
                    "synthetic_evidence_used": False,
                }
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def test_selection_loader_requires_prefix_train_only_provenance(self) -> None:
        from training import olmoe_real_subset_runs as real

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            self.make_selection(path)
            selected, provenance = real.load_prefix_expert_selection(
                path,
                "ESFT-Gate",
                num_layers=2,
                num_experts=4,
                expected_base_model="unit-test-model",
            )
            self.assertEqual(selected, {0: [0, 1], 1: [0, 1]})
            self.assertEqual(len(provenance["selection_json_sha256"]), 64)
            self.assertFalse(provenance["sealed_evidence_used"])
            self.assertFalse(provenance["synthetic_evidence_used"])

    def test_selection_loader_rejects_all_experts_and_synthetic_evidence(self) -> None:
        from training import olmoe_real_subset_runs as real

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            payload = self.make_selection(path, selected_count=4)
            with self.assertRaisesRegex(ValueError, "cannot select all experts"):
                real.load_prefix_expert_selection(
                    path, "ESFT-Gate", 2, 4, "unit-test-model"
                )
            payload["synthetic_evidence_used"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "synthetic_evidence_used=false"):
                real.load_prefix_expert_selection(
                    path, "ESFT-Gate", 2, 8, "unit-test-model"
                )

    def test_request_rejects_lora_and_full_expert_conflict(self) -> None:
        from training import olmoe_real_subset_runs as real

        base = dict(
            expert_selection_json="development/selection.json",
            expert_selection_method="ESFT-Gate",
            expert_learning_rate=1e-6,
            expert_anchor_coefficient=0.01,
            train_router_gates=False,
            train_lm_head=False,
            train_experts=False,
        )
        with self.assertRaisesRegex(RuntimeError, "no wired expert-LoRA"):
            real.validate_expert_selection_request(
                SimpleNamespace(**base, expert_update_mode="lora")
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            real.validate_expert_selection_request(
                SimpleNamespace(**{**base, "expert_update_mode": "full", "train_experts": True})
            )
        with self.assertRaisesRegex(ValueError, "keeps router gates frozen"):
            real.validate_expert_selection_request(
                SimpleNamespace(
                    **{**base, "expert_update_mode": "full", "train_router_gates": True}
                )
            )
        request = real.validate_expert_selection_request(
            SimpleNamespace(
                **{
                    **base,
                    "expert_update_mode": "full",
                    "train_router_gates": True,
                    "allow_selected_expert_router_tuning": True,
                }
            )
        )
        self.assertTrue(request["router_tuning_explicitly_enabled"])

    def test_combined_optimizer_keeps_nonselected_rows_exactly_unchanged(self) -> None:
        from training import olmoe_real_subset_runs as real
        from training.olmoe_required_runs import configure_selected_full_expert_training

        class Experts(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.gate_up_proj = nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(4, 4, 2))
                self.down_proj = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 2, 2))

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.experts = Experts()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([Layer()])

        model = Model()
        bridge = nn.Parameter(torch.tensor([1.0]))
        bridge_optimizer = torch.optim.AdamW([bridge], lr=1e-3)
        expert_optimizer, _anchors, handles, metadata = configure_selected_full_expert_training(
            model,
            {0: [1]},
            expert_learning_rate=1e-4,
            anchor_coefficient=0.01,
        )
        optimizer = real.CombinedOptimizer(bridge_optimizer, expert_optimizer)
        before = model.model.layers[0].mlp.experts.gate_up_proj.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss = bridge.sum() + sum(
            parameter.sum() for parameter in model.parameters() if parameter.requires_grad
        )
        loss.backward()
        optimizer.step()
        after = model.model.layers[0].mlp.experts.gate_up_proj.detach()
        self.assertTrue(torch.equal(after[[0, 2, 3]], before[[0, 2, 3]]))
        self.assertFalse(torch.equal(after[1], before[1]))
        self.assertEqual(metadata["weight_decay"], 0.0)
        self.assertTrue(metadata["non_selected_experts_frozen"])
        for handle in handles:
            handle.remove()

    def test_selected_checkpoint_restore_preserves_nonselected_rows(self) -> None:
        from training import olmoe_real_subset_runs as real

        class Experts(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.gate_up_proj = nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(4, 4, 2))
                self.down_proj = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 2, 2))

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.experts = Experts()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([Layer(), Layer()])

        model = Model()
        selected = {0: [1], 1: [2]}
        state = real.selected_expert_rows_state_dict(model, selected)
        with torch.no_grad():
            for layer in model.model.layers:
                layer.mlp.experts.gate_up_proj.add_(1000)
                layer.mlp.experts.down_proj.add_(1000)
        nonselected_before_restore = {
            layer_idx: layer.mlp.experts.gate_up_proj.detach()[
                [idx for idx in range(4) if idx not in selected[layer_idx]]
            ].clone()
            for layer_idx, layer in enumerate(model.model.layers)
        }
        real.restore_selected_expert_rows(model, state, selected)
        for layer_idx, layer in enumerate(model.model.layers):
            selected_id = selected[layer_idx][0]
            self.assertTrue(torch.equal(
                layer.mlp.experts.gate_up_proj.detach()[selected_id],
                state[f"layer_{layer_idx}"]["gate_up_proj"][0],
            ))
            nonselected = [idx for idx in range(4) if idx != selected_id]
            self.assertTrue(torch.equal(
                layer.mlp.experts.gate_up_proj.detach()[nonselected],
                nonselected_before_restore[layer_idx],
            ))


if __name__ == "__main__":
    unittest.main()
