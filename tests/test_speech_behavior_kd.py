"""Focused tests for speech behavior distillation."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from training import olmoe_real_subset_runs as stage


class FakeTextTeacher(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.forward_calls = 0
        self.grad_enabled: list[bool] = []

    def forward(self, **kwargs):
        self.forward_calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        self.last_kwargs = kwargs
        return types.SimpleNamespace(logits=self.logits)


class SpeechBehaviorKDLossTests(unittest.TestCase):
    def make_batch(self) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[5, 6, 7, 8, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 7, 8, -100]], dtype=torch.long),
        }

    def test_zero_coefficient_skips_teacher_forward_and_returns_zero(self) -> None:
        teacher = FakeTextTeacher(torch.randn(1, 5, 4))
        student_logits = torch.randn(1, 7, 4, requires_grad=True)

        loss, token_count = stage.speech_behavior_kl_loss(
            teacher,
            student_logits,
            self.make_batch(),
            prefix_len=2,
            coefficient=0.0,
            temperature=2.0,
        )

        self.assertEqual(teacher.forward_calls, 0)
        self.assertEqual(token_count, 0)
        self.assertEqual(float(loss), 0.0)

    def test_positive_coefficient_masks_supervision_and_matches_t2_kl(self) -> None:
        teacher_logits = torch.tensor(
            [[[0.0, 0.1, 0.2], [1.0, -0.5, 0.0], [-0.2, 0.7, 0.1],
              [0.3, 0.2, -0.4], [9.0, -9.0, 1.0]]]
        )
        student_logits = torch.tensor(
            [[[4.0, -4.0, 0.0], [3.0, -3.0, 0.0],
              [8.0, -8.0, 0.0], [0.2, 0.4, -0.1],
              [0.6, -0.3, 0.2], [-7.0, 7.0, 0.0],
              [5.0, 5.0, -5.0]]],
            requires_grad=True,
        )
        teacher = FakeTextTeacher(teacher_logits)
        batch = self.make_batch()
        temperature = 2.5

        loss, token_count = stage.speech_behavior_kl_loss(
            teacher,
            student_logits,
            batch,
            prefix_len=2,
            coefficient=1.0,
            temperature=temperature,
        )

        mask = batch["labels"][:, 1:] != -100
        teacher_selected = teacher_logits[:, :-1][mask]
        student_selected = student_logits[:, 2:6][mask]
        expected = F.kl_div(
            F.log_softmax(student_selected / temperature, dim=-1),
            F.log_softmax(teacher_selected / temperature, dim=-1),
            reduction="batchmean",
            log_target=True,
        ) * temperature**2
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(token_count, 2)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(teacher.forward_calls, 1)
        self.assertEqual(teacher.grad_enabled, [False])
        self.assertFalse(teacher.last_kwargs["output_router_logits"])
        self.assertNotIn("inputs_embeds", teacher.last_kwargs)

        loss.backward()
        nonzero_positions = (
            student_logits.grad.detach().abs().sum(dim=-1).ne(0).nonzero().tolist()
        )
        self.assertEqual(nonzero_positions, [[0, 3], [0, 4]])

    def test_positive_kd_requires_frozen_olmoe_routing_and_lm(self) -> None:
        base = {
            "speech_behavior_kl_coef": 1.0,
            "speech_behavior_kl_temperature": 2.0,
            "train_router_gates": False,
            "train_experts": False,
            "train_lm_head": False,
            "expert_selection_json": "",
            "dynamic_expert_bias_lr": 0.0,
        }
        stage.validate_speech_behavior_kl_request(types.SimpleNamespace(**base))
        nonfinite = dict(base)
        nonfinite["speech_behavior_kl_coef"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            stage.validate_speech_behavior_kl_request(types.SimpleNamespace(**nonfinite))
        for field in ("train_router_gates", "train_experts", "train_lm_head"):
            invalid = dict(base)
            invalid[field] = True
            with self.assertRaisesRegex(ValueError, "frozen router/expert/LM"):
                stage.validate_speech_behavior_kl_request(
                    types.SimpleNamespace(**invalid)
                )

    def test_cli_defaults_preserve_behavior(self) -> None:
        with patch.object(__import__("sys"), "argv", ["olmoe_real_subset_runs.py"]):
            args = stage.parse_args()
        self.assertEqual(args.speech_behavior_kl_coef, 0.0)
        self.assertEqual(args.speech_behavior_kl_temperature, 1.0)


if __name__ == "__main__":
    unittest.main()
