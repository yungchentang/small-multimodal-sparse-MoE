"""CPU-only tests for the real-data Stage B distillation path."""

from __future__ import annotations

import hashlib
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
import numpy as np

from scripts import stage_b_checkpoint_provenance as stage_b_provenance
from training import distill_olmoe_top2_real as stage_b


class TinyMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(3, 4, bias=False)
        self.gate.top_k = 2
        self.gamma_scale = torch.nn.Parameter(torch.tensor(1.0))


class TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinyMlp()


class TinyCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])
        self.register_buffer("backbone_scale", torch.tensor([1.0, 2.0]))


class TinyStudent(torch.nn.Module):
    def __init__(self, top_k: int = 2) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(num_experts_per_tok=top_k)
        with torch.random.fork_rng():
            torch.manual_seed(0)
            self.model = TinyCore()
            self.input_embeddings = torch.nn.Embedding(7, 3)
            self.output_embeddings = torch.nn.Linear(3, 7, bias=False)

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings


class DistillationEquationTests(unittest.TestCase):
    def test_logits_kl_matches_temperature_scaled_equation(self) -> None:
        student = torch.tensor([[[1.0, 0.0], [0.2, 0.8]]])
        teacher = torch.tensor([[[0.0, 1.0], [0.9, 0.1]]])
        labels = torch.tensor([[4, -100]])
        temperature = 2.0

        actual = stage_b.logits_kl(student, teacher, labels, temperature)
        expected = F.kl_div(
            F.log_softmax(student[:, :1].reshape(1, 2) / temperature, dim=-1),
            F.softmax(teacher[:, :1].reshape(1, 2) / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature**2

        self.assertTrue(torch.allclose(actual, expected))

    def test_selected_hidden_state_mse_uses_only_unmasked_tokens(self) -> None:
        labels = torch.tensor([[1, -100]])
        student = [
            torch.zeros(1, 3, 2),
            torch.tensor([[[1.0, 3.0], [9.0, 9.0], [0.0, 0.0]]]),
        ]
        teacher = [torch.zeros(1, 3, 2), torch.zeros(1, 3, 2)]

        loss = stage_b.hidden_match(student, teacher, labels, [1], "mse")

        self.assertEqual(float(loss), 5.0)

    def test_moe_reconstruction_reports_masked_mse_rmse_and_cosine(self) -> None:
        labels = torch.tensor([[1, -100]])
        student = {
            0: torch.tensor([[[1.0, 3.0], [20.0, 20.0], [0.0, 0.0]]])
        }
        teacher = {0: torch.zeros(1, 3, 2)}

        loss, metrics = stage_b.moe_output_reconstruction(student, teacher, labels, [0])

        self.assertEqual(float(loss), 5.0)
        self.assertAlmostEqual(metrics["moe_reconstruction_rmse"], 5.0**0.5)
        self.assertEqual(metrics["moe_reconstruction_cosine"], 0.0)


class RouterDistributionTests(unittest.TestCase):
    def test_router_kl_uses_full_expert_dimension_and_token_mask(self) -> None:
        student = torch.tensor(
            [[2.0, 1.0, 0.0, -1.0], [8.0, -8.0, 4.0, -4.0]]
        )
        teacher = torch.tensor(
            [[1.0, 2.0, -1.0, 0.0], [-8.0, 8.0, -4.0, 4.0]]
        )
        mask = torch.tensor([True, False])

        actual = stage_b.router_kl(
            [student],
            [teacher],
            1.0,
            token_mask=mask,
            expected_num_experts=4,
        )
        expected = F.kl_div(
            F.log_softmax(student[:1], dim=-1),
            F.softmax(teacher[:1], dim=-1),
            reduction="batchmean",
        )

        self.assertTrue(torch.allclose(actual, expected))
        with self.assertRaisesRegex(ValueError, "expert dimension"):
            stage_b.router_kl([student], [teacher], 1.0, expected_num_experts=3)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            stage_b.router_kl([student], [teacher[:, :3]], 1.0)


class CurriculumTests(unittest.TestCase):
    def test_optional_curriculum_has_8_4_2_stages_and_ends_at_two(self) -> None:
        schedule = stage_b.parse_k_curriculum("8,4,2", 8, 2)
        effective = [
            stage_b.student_k_for_step(step, 6, schedule)
            for step in range(1, 7)
        ]

        self.assertEqual(effective, [8, 8, 4, 4, 2, 2])
        self.assertEqual(effective[-1], stage_b.FINAL_STUDENT_TOP_K)

    def test_curriculum_rejects_missing_final_top_two_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 8,4,2"):
            stage_b.parse_k_curriculum("8,4", 8, 2)
        with self.assertRaisesRegex(ValueError, "cover every"):
            stage_b.student_k_for_step(1, 2, [8, 4, 2])


class TextReplayTests(unittest.TestCase):
    def test_text_replay_is_standard_masked_next_token_ce(self) -> None:
        logits = torch.tensor(
            [[[3.0, 0.0, -1.0], [100.0, -100.0, 0.0]]],
            requires_grad=True,
        )
        labels = torch.tensor([[0, -100]])

        actual = stage_b.text_replay_loss(logits, labels)
        expected = F.cross_entropy(logits[:, :1].reshape(1, 3), torch.tensor([0]))
        actual.backward()

        self.assertTrue(torch.allclose(actual, expected))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.equal(logits.grad[:, 1], torch.zeros_like(logits.grad[:, 1])))


class CheckpointTests(unittest.TestCase):
    def full_args(self) -> dict:
        with patch.object(sys, "argv", ["distill_olmoe_top2_real.py"]):
            args = vars(stage_b.parse_args())
        args.update({
            "data_dir": "/tmp/development-real-data",
            "base_model": "development/model",
            "distill_steps": 9,
            "train_lm_head": True,
            "train_router_gates": True,
            "train_gamma_scale": True,
        })
        return args

    def trainable_meta(self, student: TinyStudent) -> dict:
        return stage_b.set_trainable(student, True, True, True)

    def optimizer(self, student: TinyStudent, args: dict) -> torch.optim.Optimizer:
        trainable = self.trainable_meta(student)
        return torch.optim.AdamW([
            {
                "params": group["params"],
                "param_names": group["parameter_names"],
                "lr": args["learning_rate"],
                "weight_decay": args["weight_decay"],
            }
            for group in trainable["groups"]
        ])

    def data_contract(self, digest: str = "a" * 64) -> dict:
        return {
            "schema_version": 1,
            "canonical_data_root": "/tmp/development-real-data",
            "jsonl_inputs": [
                {
                    "role": "train",
                    "canonical_path": "/tmp/development-real-data/train.jsonl",
                    "sha256": digest,
                    "size_bytes": 10,
                    "row_count": 1,
                }
            ],
        }

    def model_identity(self, student: TinyStudent, revision: str = "rev-a") -> dict:
        identity = stage_b.base_model_identity(student, "development/model")
        identity["commit_hash"] = revision
        return identity

    def verified_run_provenance(self) -> dict:
        dataset = {"policy": "development_only_real_manifests"}
        return {
            "run_uuid": "123e4567e89b42d3a456426614174000",
            "source_commit_sha": "a" * 40,
            "runai_job_name": "stage-b-parent",
            "runai_project": "development",
            "producer_code": {
                "path": "training/distill_olmoe_top2_real.py",
                "sha256": "b" * 64,
            },
            "dataset_split_provenance": dataset,
            "dataset_split_provenance_sha256": (
                stage_b_provenance.canonical_json_sha256(dataset)
            ),
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
        }

    def test_checkpoint_preserves_old_keys_and_adds_args_provenance(self) -> None:
        student = TinyStudent(top_k=2)
        args = self.full_args()
        optimizer = self.optimizer(student, args)
        dataset = {
            "policy": "development_only_real_manifests",
            "data_dir": "/tmp/development-data",
            "files": {},
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
        }
        provenance = {
            "run_uuid": "123e4567e89b42d3a456426614174000",
            "source_commit_sha": "a" * 40,
            "runai_job_name": "stage-b-test",
            "runai_project": "project",
            "producer_code": {
                "path": "training/distill_olmoe_top2_real.py",
                "sha256": "b" * 64,
            },
            "dataset_split_provenance": dataset,
            "dataset_split_provenance_sha256": (
                stage_b_provenance.canonical_json_sha256(dataset)
            ),
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
            "stage": "B",
            "checkpoint_version": stage_b.CHECKPOINT_VERSION,
            "development_data_only": True,
            "final_inference_top_k": 2,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            stage_b.save_student_checkpoint(
                student,
                path,
                self.trainable_meta(student),
                args,
                {"loss": 1.25},
                provenance=provenance,
                optimizer=optimizer,
                completed_steps=9,
                data_contract=self.data_contract(),
                model_identity=self.model_identity(student),
            )
            state = torch.load(path, map_location="cpu", weights_only=False)

        self.assertTrue(
            {
                "trainable_meta",
                "args",
                "metrics",
                "lm_output_embeddings",
                "lm_input_embeddings",
                "lm_embeddings_tied",
                "router_gates",
                "gamma_scale",
            }
            <= set(state)
        )
        self.assertFalse(state["lm_embeddings_tied"])
        self.assertEqual(state["args"], args)
        self.assertEqual(state["provenance"], provenance)
        self.assertEqual(
            state["run_identity"],
            stage_b_provenance.build_checkpoint_run_identity(provenance),
        )
        self.assertEqual(state["completed_steps"], 9)
        self.assertEqual(state["final_inference_top_k"], 2)

    def test_legacy_checkpoint_state_restores_without_new_keys(self) -> None:
        source = TinyStudent(top_k=2)
        target = TinyStudent(top_k=2)
        with torch.no_grad():
            source.output_embeddings.weight.fill_(0.25)
            source.model.layers[0].mlp.gate.weight.fill_(0.5)
            source.model.layers[0].mlp.gamma_scale.fill_(1.75)
            source.model.layers[1].mlp.gamma_scale.fill_(1.5)
        legacy = {
            "args": {"base_model": "development/model", "student_top_k": 2},
            "lm_output_embeddings": source.output_embeddings.state_dict(),
            "router_gates": {
                "layer_0": source.model.layers[0].mlp.gate.state_dict(),
            },
            "gamma_scale": [1.75, 1.5],
        }

        meta = stage_b.restore_student_checkpoint(
            target,
            legacy,
            current_args={"base_model": "development/model", "student_top_k": 2},
        )

        self.assertEqual(meta["checkpoint_version"], 1)
        self.assertFalse(meta["optimizer_state_restored"])
        self.assertEqual(
            meta["legacy_v1_behavior"],
            "partial_model_state_without_optimizer_or_rng_contract",
        )
        self.assertTrue(
            torch.equal(target.output_embeddings.weight, source.output_embeddings.weight)
        )
        self.assertTrue(
            torch.equal(
                target.model.layers[0].mlp.gate.weight,
                source.model.layers[0].mlp.gate.weight,
            )
        )
        self.assertEqual(stage_b.gamma_scale_values(target), [1.75, 1.5])

    def test_v2_resume_rejects_incomplete_or_mismatched_trainable_state(self) -> None:
        source = TinyStudent(top_k=2)
        args = self.full_args()
        source_trainable = self.trainable_meta(source)
        source_optimizer = self.optimizer(source, args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            stage_b.save_student_checkpoint(
                source,
                path,
                source_trainable,
                args,
                {"loss": 1.0},
                provenance=self.verified_run_provenance(),
                completed_steps=1,
                optimizer=source_optimizer,
                data_contract=self.data_contract(),
                model_identity=self.model_identity(source),
            )
            state = torch.load(
                path, map_location="cpu", weights_only=True
            )

        for missing_key, expected_error in (
            ("lm_input_embeddings", "missing input embeddings"),
            ("gamma_scale", "missing trainable gamma state"),
        ):
            with self.subTest(missing_key=missing_key):
                incomplete = dict(state)
                del incomplete[missing_key]
                target = TinyStudent(top_k=2)
                with self.assertRaisesRegex(
                    ValueError, expected_error
                ):
                    stage_b.restore_student_checkpoint(
                        target,
                        incomplete,
                        optimizer=self.optimizer(target, args),
                        current_args=args,
                        current_trainable=self.trainable_meta(target),
                        current_data_contract=self.data_contract(),
                        current_model_identity=self.model_identity(target),
                    )

        mismatch = dict(state)
        mismatch["lm_embeddings_tied"] = True
        target = TinyStudent(top_k=2)
        with self.assertRaisesRegex(
            ValueError, "topology disagrees"
        ):
            stage_b.restore_student_checkpoint(
                target,
                mismatch,
                optimizer=self.optimizer(target, args),
                current_args=args,
                current_trainable=self.trainable_meta(target),
                current_data_contract=self.data_contract(),
                current_model_identity=self.model_identity(target),
            )

    def test_v2_resume_rejects_missing_or_tampered_provenance_and_identity(
        self,
    ) -> None:
        source = TinyStudent(top_k=2)
        args = self.full_args()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            stage_b.save_student_checkpoint(
                source,
                path,
                self.trainable_meta(source),
                args,
                {},
                provenance=self.verified_run_provenance(),
                completed_steps=1,
                optimizer=self.optimizer(source, args),
                data_contract=self.data_contract(),
                model_identity=self.model_identity(source),
            )
            state = torch.load(path, map_location="cpu", weights_only=True)

        def restore(candidate: dict) -> None:
            target = TinyStudent(top_k=2)
            stage_b.restore_student_checkpoint(
                target,
                candidate,
                optimizer=self.optimizer(target, args),
                current_args=args,
                current_trainable=self.trainable_meta(target),
                current_data_contract=self.data_contract(),
                current_model_identity=self.model_identity(target),
            )

        missing_provenance = dict(state)
        del missing_provenance["provenance"]
        with self.assertRaisesRegex(ValueError, "provenance is missing"):
            restore(missing_provenance)

        incomplete_provenance = dict(state)
        incomplete_provenance["provenance"] = dict(state["provenance"])
        del incomplete_provenance["provenance"]["source_commit_sha"]
        with self.assertRaisesRegex(ValueError, "provenance is incomplete"):
            restore(incomplete_provenance)

        tampered_provenance = dict(state)
        tampered_provenance["provenance"] = dict(state["provenance"])
        tampered_provenance["provenance"]["source_commit_sha"] = "c" * 40
        with self.assertRaisesRegex(ValueError, "disagrees with provenance"):
            restore(tampered_provenance)

        missing_identity = dict(state)
        del missing_identity["run_identity"]
        with self.assertRaisesRegex(ValueError, "run_identity is missing"):
            restore(missing_identity)

        tampered_identity = dict(state)
        tampered_identity["run_identity"] = dict(state["run_identity"])
        tampered_identity["run_identity"]["runai_job_name"] = "tampered"
        with self.assertRaisesRegex(ValueError, "disagrees with provenance"):
            restore(tampered_identity)

    def test_resumed_child_provenance_includes_verified_parent_identity(
        self,
    ) -> None:
        parent = TinyStudent(top_k=2)
        parent_args = self.full_args()
        parent_provenance = self.verified_run_provenance()
        with tempfile.TemporaryDirectory() as directory:
            parent_path = Path(directory) / "parent.pt"
            stage_b.save_student_checkpoint(
                parent,
                parent_path,
                self.trainable_meta(parent),
                parent_args,
                {},
                provenance=parent_provenance,
                completed_steps=1,
                optimizer=self.optimizer(parent, parent_args),
                data_contract=self.data_contract(),
                model_identity=self.model_identity(parent),
            )
            parent_state, parent_file_identity = (
                stage_b.load_runner_resume_checkpoint(parent_path, "cpu")
            )
            child = TinyStudent(top_k=2)
            child_optimizer = self.optimizer(child, parent_args)
            resume_meta = stage_b.restore_student_checkpoint(
                child,
                parent_state,
                optimizer=child_optimizer,
                current_args=parent_args,
                current_trainable=self.trainable_meta(child),
                current_data_contract=self.data_contract(),
                current_model_identity=self.model_identity(child),
            )
            parent_file_identity.update(
                {
                    "run_identity": resume_meta["run_identity"],
                    "source_commit_sha": resume_meta["source_commit_sha"],
                }
            )
            child_run_provenance = self.verified_run_provenance()
            child_run_provenance["run_uuid"] = "123e4567e89b42d3a456426614174001"
            child_run_provenance["source_commit_sha"] = "d" * 40
            child_provenance = stage_b.make_checkpoint_provenance(
                types.SimpleNamespace(**parent_args),
                parent_file_identity,
                child_run_provenance,
            )
            child_path = Path(directory) / "child.pt"
            stage_b.save_student_checkpoint(
                child,
                child_path,
                self.trainable_meta(child),
                parent_args,
                {},
                provenance=child_provenance,
                completed_steps=2,
                optimizer=child_optimizer,
                data_contract=self.data_contract(),
                model_identity=self.model_identity(child),
                curriculum_plan=resume_meta["curriculum_plan"],
            )
            child_state = torch.load(
                child_path, map_location="cpu", weights_only=True
            )

        recorded = child_state["provenance"]
        self.assertEqual(
            recorded["resume_checkpoint_run_identity"],
            parent_state["run_identity"],
        )
        self.assertEqual(
            recorded["resume_checkpoint_source_commit_sha"],
            parent_provenance["source_commit_sha"],
        )
        self.assertEqual(
            recorded["resume_checkpoint"],
            str(parent_path.resolve()),
        )
        self.assertEqual(
            recorded["resume_checkpoint_sha256"],
            parent_file_identity["sha256"],
        )
        self.assertEqual(
            recorded["resume_checkpoint_size_bytes"],
            parent_file_identity["size_bytes"],
        )

    def test_checkpoint_rejects_missing_or_invalid_run_provenance(self) -> None:
        student = TinyStudent(top_k=2)
        args = self.full_args()
        save_kwargs = {
            "optimizer": self.optimizer(student, args),
            "completed_steps": 0,
            "data_contract": self.data_contract(),
            "model_identity": self.model_identity(student),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            with self.assertRaisesRegex(
                ValueError, "requires verified run provenance"
            ):
                stage_b.save_student_checkpoint(
                    student,
                    path,
                    self.trainable_meta(student),
                    args,
                    {},
                    **save_kwargs,
                )

            invalid = self.verified_run_provenance()
            invalid["dataset_split_provenance_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                ValueError, "dataset provenance SHA mismatch"
            ):
                stage_b.save_student_checkpoint(
                    student,
                    path,
                    self.trainable_meta(student),
                    args,
                    {},
                    provenance=invalid,
                    **save_kwargs,
                )

    def test_checkpoint_rejects_non_top_two_runtime_or_args(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            with self.assertRaisesRegex(ValueError, "final Top-2"):
                stage_b.save_student_checkpoint(
                    TinyStudent(top_k=2),
                    path,
                    {},
                    {"student_top_k": 4},
                    {},
                )
            with self.assertRaisesRegex(ValueError, "top_k is not 2"):
                stage_b.save_student_checkpoint(
                    TinyStudent(top_k=4),
                    path,
                    {},
                    {"student_top_k": 2},
                    {},
                )
            local_mismatch = TinyStudent(top_k=2)
            local_mismatch.model.layers[0].mlp.gate.top_k = 4
            with self.assertRaisesRegex(ValueError, "top_k is not 2"):
                stage_b.save_student_checkpoint(
                    local_mismatch,
                    path,
                    {},
                    {"student_top_k": 2},
                    {},
                )

    def test_v2_resume_rejects_any_continuation_arg_drift(self) -> None:
        source = TinyStudent(top_k=2)
        args = self.full_args()
        trainable = self.trainable_meta(source)
        optimizer = self.optimizer(source, args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            stage_b.save_student_checkpoint(
                source,
                path,
                trainable,
                args,
                {},
                provenance=self.verified_run_provenance(),
                optimizer=optimizer,
                completed_steps=1,
                data_contract=self.data_contract(),
                model_identity=self.model_identity(source),
            )
            state = torch.load(path, map_location="cpu", weights_only=True)

        drifted = dict(args)
        drifted["distill_logit_coef"] += 0.1
        target = TinyStudent(top_k=2)
        with self.assertRaisesRegex(ValueError, "mismatch for args"):
            stage_b.restore_student_checkpoint(
                target,
                state,
                optimizer=self.optimizer(target, args),
                current_args=drifted,
                current_trainable=self.trainable_meta(target),
                current_data_contract=self.data_contract(),
                current_model_identity=self.model_identity(target),
            )

    def test_v2_resume_allows_target_increase_but_requires_remaining_steps(self) -> None:
        source = TinyStudent(top_k=2)
        args = self.full_args()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            stage_b.save_student_checkpoint(
                source,
                path,
                self.trainable_meta(source),
                args,
                {},
                provenance=self.verified_run_provenance(),
                optimizer=self.optimizer(source, args),
                completed_steps=4,
                data_contract=self.data_contract(),
                model_identity=self.model_identity(source),
            )
            state = torch.load(path, map_location="cpu", weights_only=True)

        increased = dict(args)
        increased["distill_steps"] = 12
        target = TinyStudent(top_k=2)
        meta = stage_b.restore_student_checkpoint(
            target,
            state,
            optimizer=self.optimizer(target, increased),
            current_args=increased,
            current_trainable=self.trainable_meta(target),
            current_data_contract=self.data_contract(),
            current_model_identity=self.model_identity(target),
        )
        self.assertEqual(meta["completed_steps"], 4)

        no_remaining = dict(args)
        no_remaining["distill_steps"] = 4
        target = TinyStudent(top_k=2)
        with self.assertRaisesRegex(ValueError, "target distill_steps"):
            stage_b.restore_student_checkpoint(
                target,
                state,
                optimizer=self.optimizer(target, no_remaining),
                current_args=no_remaining,
                current_trainable=self.trainable_meta(target),
                current_data_contract=self.data_contract(),
                current_model_identity=self.model_identity(target),
            )

    def test_curriculum_plan_survives_6_to_12_and_multiple_resumes(self) -> None:
        source = TinyStudent(top_k=2)
        original = self.full_args()
        original["distill_steps"] = 6
        original["student_k_curriculum"] = "8,4,2"
        plan = stage_b.build_k_curriculum_plan([8, 4, 2], 6)
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "checkpoint_step_00000005.pt"
            stage_b.save_student_checkpoint(
                source,
                first_path,
                self.trainable_meta(source),
                original,
                {},
                provenance=self.verified_run_provenance(),
                optimizer=self.optimizer(source, original),
                completed_steps=5,
                data_contract=self.data_contract(),
                model_identity=self.model_identity(source),
                curriculum_plan=plan,
            )
            first_state = torch.load(
                first_path, map_location="cpu", weights_only=True
            )

            increased = dict(original)
            increased["distill_steps"] = 12
            resumed = TinyStudent(top_k=2)
            resumed_optimizer = self.optimizer(resumed, increased)
            first_meta = stage_b.restore_student_checkpoint(
                resumed,
                first_state,
                optimizer=resumed_optimizer,
                current_args=increased,
                current_trainable=self.trainable_meta(resumed),
                current_data_contract=self.data_contract(),
                current_model_identity=self.model_identity(resumed),
            )
            self.assertEqual(first_meta["curriculum_plan"], plan)
            self.assertEqual(
                [
                    stage_b.student_k_for_step(
                        step,
                        first_meta["curriculum_plan"]["reference_target_steps"],
                        first_meta["curriculum_plan"]["schedule"],
                    )
                    for step in range(1, 13)
                ],
                [8, 8, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2],
            )

            second_path = Path(directory) / "checkpoint_step_00000008.pt"
            stage_b.save_student_checkpoint(
                resumed,
                second_path,
                self.trainable_meta(resumed),
                increased,
                {},
                provenance=self.verified_run_provenance(),
                optimizer=resumed_optimizer,
                completed_steps=8,
                data_contract=self.data_contract(),
                model_identity=self.model_identity(resumed),
                curriculum_plan=first_meta["curriculum_plan"],
            )
            second_state = torch.load(
                second_path, map_location="cpu", weights_only=True
            )

        further_increased = dict(original)
        further_increased["distill_steps"] = 18
        twice_resumed = TinyStudent(top_k=2)
        second_meta = stage_b.restore_student_checkpoint(
            twice_resumed,
            second_state,
            optimizer=self.optimizer(twice_resumed, further_increased),
            current_args=further_increased,
            current_trainable=self.trainable_meta(twice_resumed),
            current_data_contract=self.data_contract(),
            current_model_identity=self.model_identity(twice_resumed),
        )
        self.assertEqual(second_meta["curriculum_plan"], plan)
        self.assertTrue(
            all(
                stage_b.student_k_for_step(
                    step,
                    plan["reference_target_steps"],
                    plan["schedule"],
                )
                == stage_b.FINAL_STUDENT_TOP_K
                for step in range(7, 19)
            )
        )

    def test_v2_resume_rejects_data_base_and_parameter_identity_drift(self) -> None:
        source = TinyStudent(top_k=2)
        args = self.full_args()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            stage_b.save_student_checkpoint(
                source,
                path,
                self.trainable_meta(source),
                args,
                {},
                provenance=self.verified_run_provenance(),
                optimizer=self.optimizer(source, args),
                completed_steps=1,
                data_contract=self.data_contract(),
                model_identity=self.model_identity(source),
            )
            state = torch.load(path, map_location="cpu", weights_only=True)

        cases = []
        data_drift_target = TinyStudent(top_k=2)
        cases.append((
            "data",
            data_drift_target,
            self.trainable_meta(data_drift_target),
            self.data_contract("b" * 64),
            self.model_identity(data_drift_target),
        ))
        model_drift_target = TinyStudent(top_k=2)
        cases.append((
            "base_model_identity",
            model_drift_target,
            self.trainable_meta(model_drift_target),
            self.data_contract(),
            self.model_identity(model_drift_target, "rev-b"),
        ))
        parameter_drift_target = TinyStudent(top_k=2)
        parameter_drift = self.trainable_meta(parameter_drift_target)
        parameter_drift["trainable_parameter_specs"] = list(
            reversed(parameter_drift["trainable_parameter_specs"])
        )
        cases.append((
            "trainable",
            parameter_drift_target,
            parameter_drift,
            self.data_contract(),
            self.model_identity(parameter_drift_target),
        ))
        for expected, target, trainable, data, identity in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(
                ValueError, f"mismatch for {expected}"
            ):
                stage_b.restore_student_checkpoint(
                    target,
                    state,
                    optimizer=self.optimizer(target, args),
                    current_args=args,
                    current_trainable=trainable,
                    current_data_contract=data,
                    current_model_identity=identity,
                )

    def test_periodic_checkpoint_is_atomic_and_practically_continuable(self) -> None:
        args = self.full_args()
        args["distill_steps"] = 3
        stage_b.set_seed(77)
        source = TinyStudent(top_k=2)
        base_state = {
            name: tensor.detach().clone()
            for name, tensor in source.state_dict().items()
        }
        base_identity = self.model_identity(source)
        trainable = self.trainable_meta(source)
        optimizer = self.optimizer(source, args)

        def step(model, current_optimizer):
            scale = (
                random.random()
                + float(np.random.random())
                + float(torch.rand(()))
            )
            current_optimizer.zero_grad(set_to_none=True)
            loss = sum(
                parameter.sum() * scale
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            loss.backward()
            current_optimizer.step()

        step(source, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_step_00000001.pt"
            stage_b.save_student_checkpoint(
                source,
                path,
                trainable,
                args,
                {},
                provenance=self.verified_run_provenance(),
                optimizer=optimizer,
                completed_steps=1,
                data_contract=self.data_contract(),
                model_identity=base_identity,
                allow_intermediate_routing=True,
            )
            self.assertTrue(path.is_file())
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            state = torch.load(path, map_location="cpu", weights_only=True)
            step(source, optimizer)
            expected = {
                name: parameter.detach().clone()
                for name, parameter in source.named_parameters()
            }

            resumed = TinyStudent(top_k=2)
            resumed.load_state_dict(base_state)
            resumed_optimizer = self.optimizer(resumed, args)
            stage_b.restore_student_checkpoint(
                resumed,
                state,
                optimizer=resumed_optimizer,
                current_args=args,
                current_trainable=self.trainable_meta(resumed),
                current_data_contract=self.data_contract(),
                current_model_identity=self.model_identity(resumed),
            )
            step(resumed, resumed_optimizer)
        for name, parameter in resumed.named_parameters():
            self.assertTrue(torch.equal(parameter, expected[name]), name)

    def test_v2_resume_restores_python_numpy_torch_rng(self) -> None:
        source = TinyStudent(top_k=2)
        args = self.full_args()
        stage_b.set_seed(1234)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            stage_b.save_student_checkpoint(
                source,
                path,
                self.trainable_meta(source),
                args,
                {},
                provenance=self.verified_run_provenance(),
                optimizer=self.optimizer(source, args),
                completed_steps=1,
                data_contract=self.data_contract(),
                model_identity=self.model_identity(source),
            )
            state = torch.load(path, map_location="cpu", weights_only=True)
            expected = (
                random.random(),
                float(np.random.random()),
                torch.rand(3),
            )
            stage_b.set_seed(9999)
            target = TinyStudent(top_k=2)
            stage_b.restore_student_checkpoint(
                target,
                state,
                optimizer=self.optimizer(target, args),
                current_args=args,
                current_trainable=self.trainable_meta(target),
                current_data_contract=self.data_contract(),
                current_model_identity=self.model_identity(target),
            )
            actual = (
                random.random(),
                float(np.random.random()),
                torch.rand(3),
            )
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_resume_checkpoint_hash_and_load_use_same_single_read(self) -> None:
        source = TinyStudent(top_k=2)
        args = self.full_args()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            stage_b.save_student_checkpoint(
                source,
                path,
                self.trainable_meta(source),
                args,
                {},
                provenance=self.verified_run_provenance(),
                optimizer=self.optimizer(source, args),
                completed_steps=0,
                data_contract=self.data_contract(),
                model_identity=self.model_identity(source),
            )
            expected_bytes = path.read_bytes()
            original_read_bytes = Path.read_bytes
            with patch.object(
                Path,
                "read_bytes",
                autospec=True,
                side_effect=lambda value: original_read_bytes(value),
            ) as read_bytes:
                state, identity = stage_b.load_resume_checkpoint_once(
                    path,
                    "cpu",
                )
        self.assertEqual(read_bytes.call_count, 1)
        self.assertEqual(identity["sha256"], hashlib.sha256(expected_bytes).hexdigest())
        self.assertEqual(state["checkpoint_version"], stage_b.CHECKPOINT_VERSION)
    def test_real_runner_rejects_legacy_v1_before_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"args": {"base_model": "development/model"}}, path)
            with self.assertRaisesRegex(
                ValueError, "real runner refuses legacy v1"
            ):
                stage_b.load_runner_resume_checkpoint(path, "cpu")




class ValidationTests(unittest.TestCase):
    def default_args(self):
        with patch.object(sys, "argv", ["distill_olmoe_top2_real.py"]):
            args = stage_b.parse_args()
        args.data_dir = "/tmp/development-real-data"
        args.distill_steps = 6
        return args

    def test_new_cli_defaults_preserve_prior_objectives(self) -> None:
        args = self.default_args()

        self.assertEqual(args.moe_reconstruction_coef, 0.0)
        self.assertEqual(args.text_replay_coef, 0.0)
        self.assertEqual(args.student_k_curriculum, "none")
        self.assertEqual(args.resume_checkpoint, "")
        self.assertEqual(stage_b.validate_stage_b_args(args), [2])

    def test_base_identity_binds_parameters_buffers_and_revision(self) -> None:
        teacher = TinyStudent(top_k=8)
        student = TinyStudent(top_k=2)
        student.load_state_dict(teacher.state_dict())
        self.assertEqual(
            stage_b.base_model_identity(teacher, "development/model"),
            stage_b.base_model_identity(student, "development/model"),
        )

        with torch.no_grad():
            student.input_embeddings.weight[0, 0].add_(1.0)
        self.assertNotEqual(
            stage_b.base_model_identity(teacher, "development/model"),
            stage_b.base_model_identity(student, "development/model"),
        )

        student.load_state_dict(teacher.state_dict())
        with torch.no_grad():
            student.model.backbone_scale[0].add_(1.0)
        self.assertNotEqual(
            stage_b.base_model_identity(teacher, "development/model"),
            stage_b.base_model_identity(student, "development/model"),
        )

        student.load_state_dict(teacher.state_dict())
        student.config._commit_hash = "different-revision"
        self.assertNotEqual(
            stage_b.base_model_identity(teacher, "development/model"),
            stage_b.base_model_identity(student, "development/model"),
        )

    def test_prepatch_identity_accepts_asymmetric_routing_mutation(self) -> None:
        teacher = TinyStudent(top_k=8)
        student = TinyStudent(top_k=2)
        student.load_state_dict(teacher.state_dict())
        teacher_prepatch = stage_b.base_model_identity(
            teacher, "development/model"
        )
        student_prepatch = stage_b.base_model_identity(
            student, "development/model"
        )
        self.assertEqual(teacher_prepatch, student_prepatch)

        student.model.register_buffer(
            "gamma_scale", torch.ones(1, dtype=torch.float32)
        )
        student.model.register_buffer(
            "expert_bias", torch.zeros(2, dtype=torch.float32)
        )
        self.assertNotEqual(
            stage_b.base_model_identity(teacher, "development/model"),
            stage_b.base_model_identity(student, "development/model"),
        )
        self.assertEqual(teacher_prepatch, student_prepatch)

    def test_invalid_stage_b_settings_are_rejected(self) -> None:
        args = self.default_args()
        args.teacher_top_k = 4
        with self.assertRaisesRegex(ValueError, "Top-8 teacher"):
            stage_b.validate_stage_b_args(args)

        args = self.default_args()
        args.router_distill_coef = 1.0
        with self.assertRaisesRegex(ValueError, "train-router-gates"):
            stage_b.validate_stage_b_args(args)

        args = self.default_args()
        args.text_replay_coef = 1.0
        with self.assertRaisesRegex(ValueError, "text-replay-manifest"):
            stage_b.validate_stage_b_args(args)

        args = self.default_args()
        args.data_dir = "/tmp/sealed/evaluation"
        with self.assertRaisesRegex(ValueError, "development-only"):
            stage_b.validate_stage_b_args(args)

    def test_manifest_rows_reject_sealed_or_synthetic_splits(self) -> None:
        for split in ("sealed_test", "synthetic_train"):
            with self.assertRaisesRegex(ValueError, "development-only"):
                stage_b.validate_development_rows([{"split": split}], "manifest")
        stage_b.validate_development_rows([{"source_split": "contest_train"}], "manifest")


class LauncherWiringTests(unittest.TestCase):
    def test_stage_b_objectives_are_forwarded_end_to_end(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        runner = (repo_root / "run.sh").read_text(encoding="utf-8")
        submit = (repo_root / "scripts" / "submit_runai.sh").read_text(
            encoding="utf-8"
        )
        launcher = (
            repo_root / "scripts" / "submit_top2_distill_runai.sh"
        ).read_text(encoding="utf-8")

        variables = (
            "MOE_RECONSTRUCTION_COEF",
            "MOE_RECONSTRUCTION_LAYERS",
            "TEXT_REPLAY_COEF",
            "TEXT_REPLAY_MANIFEST",
            "STUDENT_K_CURRICULUM",
            "RESUME_CHECKPOINT",
            "CHECKPOINT_EVERY_STEPS",
        )
        for variable in variables:
            self.assertIn(variable, runner)
            self.assertIn(variable, submit)
            self.assertIn(variable, launcher)

        self.assertIn(
            'TRAIN_ROUTER_GATES="${TRAIN_ROUTER_GATES:-1}"', launcher
        )
        self.assertIn(
            'STUDENT_K_CURRICULUM="${STUDENT_K_CURRICULUM:-8,4,2}"',
            launcher,
        )
        self.assertIn(
            '--student-k-curriculum "${STUDENT_K_CURRICULUM:-none}"', runner
        )


if __name__ == "__main__":
    unittest.main()
