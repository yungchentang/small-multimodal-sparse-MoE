import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from scripts.diagnose_moe_reconstruction import (
    EXPECTED_EXPERIMENT_ID,
    load_verified_checkpoint,
    restore_checkpoint_model_state,
    sha256_file,
)


class TinyExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(4, 2, 3))
        self.down_proj = nn.Parameter(torch.zeros(4, 3, 2))


class TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(2, 4, bias=False)
        self.experts = TinyExperts()
        self.register_buffer("expert_bias", torch.zeros(4))
        self.dynamic_expert_bias_enabled = False


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinyMlp()


class TinyModel(nn.Module):
    def __init__(self, tied_embeddings: bool = False) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.input_embeddings = nn.Embedding(5, 2)
        self.output_embeddings = (
            self.input_embeddings if tied_embeddings else nn.Linear(2, 5, bias=False)
        )

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings


def checkpoint_metadata(**trainable_meta):
    return {
        "args": {"base_model": "tiny/model"},
        "last_row": {
            "experiment_id": EXPECTED_EXPERIMENT_ID,
            "top_k": 2,
            "gamma_applied": False,
        },
        "trainable_meta": trainable_meta,
    }


def filled_state_dict(module: nn.Module, value: float):
    return {
        key: torch.full_like(tensor, value)
        for key, tensor in module.state_dict().items()
    }


class DiagnoseMoeReconstructionCheckpointTest(unittest.TestCase):
    def test_checkpoint_requires_existing_path_and_exact_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.pt"
            with self.assertRaises(FileNotFoundError):
                load_verified_checkpoint(missing, "0" * 64)

            checkpoint = root / "checkpoint.pt"
            torch.save(checkpoint_metadata(), checkpoint)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_verified_checkpoint(checkpoint, "0" * 64)

    def test_checkpoint_rejects_forbidden_path_and_non_e3_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forbidden = root / "synthetic-checkpoint.pt"
            torch.save(checkpoint_metadata(), forbidden)
            with self.assertRaisesRegex(ValueError, "forbidden term"):
                load_verified_checkpoint(forbidden, sha256_file(forbidden))

            checkpoint = root / "checkpoint.pt"
            state = checkpoint_metadata()
            state["last_row"]["experiment_id"] = "E2_calibrated_top2"
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "not E3_final_multimodal_top2"):
                load_verified_checkpoint(checkpoint, sha256_file(checkpoint))

            state = checkpoint_metadata()
            state["args"]["top_k"] = 8
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "args must declare Top-2"):
                load_verified_checkpoint(checkpoint, sha256_file(checkpoint))

    def test_trainable_metadata_fails_closed_when_state_is_missing(self) -> None:
        cases = (
            (
                {"train_router_gates": True},
                "omitted router_gates",
            ),
            (
                {"train_experts": True},
                "omitted experts",
            ),
            (
                {"selected_expert_training": True},
                "omitted selected_experts",
            ),
            (
                {"train_lm_head": True},
                "omitted output embeddings",
            ),
        )
        for trainable_meta, message in cases:
            with self.subTest(trainable_meta=trainable_meta):
                with self.assertRaisesRegex(ValueError, message):
                    restore_checkpoint_model_state(
                        TinyModel(), checkpoint_metadata(**trainable_meta)
                    )

        last_row_only = checkpoint_metadata()
        last_row_only["last_row"]["train_router_gates"] = True
        with self.assertRaisesRegex(ValueError, "omitted router_gates"):
            restore_checkpoint_model_state(TinyModel(), last_row_only)

    def test_restores_full_checkpoint_components(self) -> None:
        model = TinyModel()
        state = checkpoint_metadata(
            train_router_gates=True,
            train_experts=True,
            train_lm_head=True,
        )
        state["router_gates"] = {
            f"layer_{index}": filled_state_dict(layer.mlp.gate, 1.0 + index)
            for index, layer in enumerate(model.model.layers)
        }
        state["experts"] = {
            f"layer_{index}": filled_state_dict(layer.mlp.experts, 3.0 + index)
            for index, layer in enumerate(model.model.layers)
        }
        state["lm_input_embeddings"] = filled_state_dict(
            model.input_embeddings, 5.0
        )
        state["lm_output_embeddings"] = filled_state_dict(
            model.output_embeddings, 6.0
        )
        state["dynamic_expert_bias"] = {
            "layer_0": torch.full((4,), 7.0),
            "layer_1": torch.full((4,), 8.0),
        }

        provenance = restore_checkpoint_model_state(model, state)

        self.assertEqual(
            provenance["restored_components"],
            [
                "router_gates",
                "experts",
                "lm_output_embeddings",
                "lm_input_embeddings",
                "dynamic_expert_bias",
            ],
        )
        self.assertTrue(
            torch.equal(model.model.layers[0].mlp.gate.weight, torch.ones(4, 2))
        )
        self.assertTrue(
            torch.equal(
                model.model.layers[1].mlp.experts.gate_up_proj,
                torch.full((4, 2, 3), 4.0),
            )
        )
        self.assertTrue(
            torch.equal(model.input_embeddings.weight, torch.full((5, 2), 5.0))
        )
        self.assertTrue(
            torch.equal(model.output_embeddings.weight, torch.full((5, 2), 6.0))
        )
        self.assertTrue(
            torch.equal(
                model.model.layers[1].mlp.expert_bias, torch.full((4,), 8.0)
            )
        )
        self.assertTrue(
            model.model.layers[1].mlp.dynamic_expert_bias_enabled
        )

    def test_restores_selected_expert_rows_only(self) -> None:
        model = TinyModel(tied_embeddings=True)
        state = checkpoint_metadata(selected_expert_training=True)
        state["selected_experts"] = {
            "layer_0": {
                "expert_ids": [1],
                "gate_up_proj": torch.full((1, 2, 3), 2.0),
                "down_proj": torch.full((1, 3, 2), 3.0),
            },
            "layer_1": {
                "expert_ids": [2],
                "gate_up_proj": torch.full((1, 2, 3), 4.0),
                "down_proj": torch.full((1, 3, 2), 5.0),
            },
        }

        provenance = restore_checkpoint_model_state(model, state)

        self.assertEqual(
            provenance["restored_component_details"][
                "selected_expert_ids_by_layer"
            ],
            {"0": [1], "1": [2]},
        )
        self.assertTrue(
            torch.equal(
                model.model.layers[0].mlp.experts.gate_up_proj[1],
                torch.full((2, 3), 2.0),
            )
        )
        self.assertEqual(
            int(torch.count_nonzero(model.model.layers[0].mlp.experts.gate_up_proj[0])),
            0,
        )


if __name__ == "__main__":
    unittest.main()
