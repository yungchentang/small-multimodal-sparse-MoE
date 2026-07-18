"""Focused tests for speech shared-path teacher-bank contrastive training."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from training import olmoe_real_subset_runs as stage


class DeterministicTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.name_or_path = "allenai/OLMoE-1B-7B-0924"
        self.model_max_length = 128
        self.padding_side = "right"
        self.truncation_side = "right"
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1

    def get_vocab(self):
        return {"<pad>": 0, "<bos>": 1, "<eos>": 2, "token": 3}

    def __call__(self, texts, **kwargs):
        texts = [str(text) for text in texts]
        self.calls.append(texts)
        max_length = int(kwargs["max_length"])
        encoded = [
            [(ord(character) % 17) + 1 for character in text][:max_length]
            for text in texts
        ]
        width = max(len(values) for values in encoded)
        input_ids = torch.tensor(
            [values + [0] * (width - len(values)) for values in encoded],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(0).long(),
        }


class DeterministicConfig:
    _name_or_path = "allenai/OLMoE-1B-7B-0924"

    def to_dict(self):
        return {"model_type": "deterministic", "hidden_size": 3}


class DeterministicHiddenLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grad_enabled: list[bool] = []
        self.config = DeterministicConfig()

    def forward(self, input_ids, **kwargs):
        self.grad_enabled.append(torch.is_grad_enabled())
        values = input_ids.float()
        hidden = torch.stack((values, values.square(), values + 1.0), dim=-1)
        return types.SimpleNamespace(hidden_states=(hidden,))


class SpeechSharedContrastiveTests(unittest.TestCase):
    def make_batch(self) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[2, 3, 4, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "labels": torch.tensor([[-100, 3, 4, -100]]),
        }

    def train_rows(self) -> list[dict[str, object]]:
        return [
            {
                "id": f"speech-{index}",
                "partition": "train",
                "source_dataset": "openslr/librispeech_asr",
                "speaker_id": f"speaker-{index}",
                "utterance_id": f"{index + 1}-10-0001",
                "audio_sha256": f"{index + 1}" * 64,
                "transcript": transcript,
                "eval_split_name": "speech_train",
                "eval_split_index": index,
            }
            for index, transcript in enumerate(
                ("first transcript", "second transcript", "third transcript")
            )
        ]

    def strict_split_provenance(self, rows: int = 3) -> dict[str, object]:
        train_commitment = stage.speech_partition_record(
            self.train_rows(), "train", annotated=True
        )
        return {
            "policy": "manifest_train_and_dev_only_reserved_eval_split_file_unread",
            "strict_manifest_verified": True,
            "reserved_files_opened": False,
            "manifest_path": "/tmp/development_splits/manifest.json",
            "manifest_sha256": "b" * 64,
            "manifest_hash_and_parse_same_bytes": True,
            "trusted_digest_verified": True,
            "expected_speech_source_sha256": "9" * 64,
            "source_commit_sha": "a" * 40,
            "builder": {
                "path": "/tmp/repo/scripts/materialize_eval_splits.py",
                "sha256": "f" * 64,
                "source_commit_sha": "a" * 40,
                "source_commit_exists": True,
                "source_matches_commit": True,
                "current_bytes_match_commit": True,
                "command": "python scripts/materialize_eval_splits.py",
            },
            "source_files": {
                "speech": {
                    "path": "/tmp/development_data/speech_transcripts.jsonl",
                    "sha256": "9" * 64,
                    "rows": 5250,
                    "read_status": "single_snapshot_for_integrity_and_partition_verification",
                    "content_opened": True,
                    "sha256_verified_this_run": True,
                    "rows_verified_this_run": True,
                    "content_used_for_partition_verification": True,
                    "content_used_for_training": False,
                    "hash_and_rows_same_bytes": True,
                    "derivation_verified": True,
                    "reserved_eval_split_file_opened": False,
                    "raw_source_eval_rows_read_for_partition_verification": True,
                    "audio_bytes_verified_this_run": True,
                    "audio_rows_verified": 5250,
                    "audio_row_binding_root_sha256": "4" * 64,
                    "partition_commitments": {
                        "policy": "explicit_source_partition_canonical_membership_v1",
                        "partitions": {
                            "train": train_commitment,
                            "dev": {"membership_root_sha256": "2" * 64},
                            "eval": {"membership_root_sha256": "3" * 64},
                        },
                    },
                }
            },
            "files": {
                "speech_train": {
                    "path": "/tmp/development_splits/speech_train.jsonl",
                    "sha256": "c" * 64,
                    "rows": rows,
                    "read_status": "single_snapshot_for_hash_and_rows",
                    "content_opened": True,
                    "sha256_verified_this_run": True,
                    "hash_and_rows_same_bytes": True,
                },
                "image_eval": {
                    "path": "/tmp/development_splits/image_eval.jsonl",
                    "sha256": "d" * 64,
                    "rows": 1,
                    "read_status": "reserved_unread",
                    "content_opened": False,
                    "sha256_verified_this_run": False,
                },
                "speech_eval": {
                    "path": "/tmp/development_splits/speech_eval.jsonl",
                    "sha256": "e" * 64,
                    "rows": 1,
                    "read_status": "reserved_unread",
                    "content_opened": False,
                    "sha256_verified_this_run": False,
                },
            },
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
        }

    def test_teacher_row_identity_preserves_id_type(self) -> None:
        integer_identity = stage.speech_shared_teacher_row_identity(
            {"id": 1}, 0
        )
        string_identity = stage.speech_shared_teacher_row_identity(
            {"id": "1"}, 1
        )
        self.assertNotEqual(integer_identity, string_identity)
        integer_utterance = stage.speech_shared_teacher_row_identity(
            {
                "source_dataset": "dataset",
                "utterance_id": 1,
                "id": "fallback",
            },
            0,
        )
        string_utterance = stage.speech_shared_teacher_row_identity(
            {
                "source_dataset": "dataset",
                "utterance_id": "1",
                "id": "fallback",
            },
            1,
        )
        self.assertNotEqual(integer_utterance, string_utterance)

    def strict_binding(self, rows: int = 3) -> dict[str, object]:
        return stage.validate_speech_shared_split_binding(
            self.strict_split_provenance(rows), rows
        )

    @staticmethod
    def teacher_identity_kwargs() -> dict[str, dict[str, object]]:
        return {
            "teacher_model_identity": {
                "identity_sha256": "6" * 64,
                "weights_identity_sha256": "7" * 64,
            },
            "tokenizer_identity": {
                "identity_sha256": "8" * 64,
                "revision": "a" * 40,
            },
        }

    def valid_args(self, **overrides):
        values = {
            "speech_shared_contrastive_coef": 0.5,
            "speech_shared_contrastive_temperature": 0.2,
            "train_batch_size": 1,
            "train_router_gates": False,
            "speech_teacher_bank_batch_size": 64,
            "train_experts": False,
            "eval_batch_size": 1,
            "train_lm_head": False,
            "expert_selection_json": "",
            "dynamic_expert_bias_lr": 0.0,
            "development_split_manifest": "/tmp/development_splits/manifest.json",
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_default_cli_is_disabled_and_batch_one_is_valid(self) -> None:
        with patch.object(sys, "argv", ["olmoe_real_subset_runs.py"]):
            args = stage.parse_args()
        self.assertEqual(args.speech_shared_contrastive_coef, 0.0)
        self.assertEqual(args.speech_shared_contrastive_temperature, 0.07)
        self.assertEqual(args.speech_teacher_bank_batch_size, 64)
        stage.validate_speech_shared_contrastive_request(self.valid_args())

    def test_batch_one_full_bank_loss_is_finite_with_student_gradients(self) -> None:
        torch.manual_seed(9)
        student_hidden = torch.randn(1, 6, 3, requires_grad=True)
        teacher_bank = F.normalize(torch.randn(3, 3), dim=-1)

        loss, query_count, excluded_count = (
            stage.speech_shared_hidden_bank_infonce_loss(
                student_hidden,
                self.make_batch(),
                prefix_len=2,
                teacher_bank=teacher_bank,
                positive_indices=[1],
                bank_row_identities=["row-0", "row-1", "row-2"],
                bank_duplicate_keys=["text-0", "text-1", "text-2"],
                temperature=0.2,
            )
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(query_count, 1)
        self.assertEqual(excluded_count, 0)
        loss.backward()
        self.assertGreater(float(student_hidden.grad[:, 2:].abs().sum()), 0.0)
        self.assertEqual(float(student_hidden.grad[:, :2].abs().sum()), 0.0)
        self.assertIsNone(teacher_bank.grad)

    def test_positive_uses_audio_train_row_index_and_masks_duplicates(self) -> None:
        student_hidden = torch.zeros(1, 3, 2, requires_grad=True)
        student_hidden.data[:, 1:] = torch.tensor([0.0, 1.0])
        teacher_bank = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0]]
        )
        temperature = 0.5

        loss, query_count, excluded_count = (
            stage.speech_shared_hidden_bank_infonce_loss(
                student_hidden,
                {
                    "attention_mask": torch.ones(1, 2, dtype=torch.long),
                    "input_ids": torch.ones(1, 2, dtype=torch.long),
                },
                prefix_len=1,
                teacher_bank=teacher_bank,
                positive_indices=[2],
                bank_row_identities=["row-0", "row-1", "row-2", "row-3"],
                bank_duplicate_keys=["a", "duplicate", "duplicate", "d"],
                temperature=temperature,
            )
        )

        student_vector = F.normalize(torch.tensor([[0.0, 1.0]]), dim=-1)
        bank = F.normalize(teacher_bank, dim=-1)
        expected_logits = (student_vector @ bank.T) / temperature
        expected_logits[:, 1] = torch.finfo(expected_logits.dtype).min
        expected = F.cross_entropy(expected_logits, torch.tensor([2]))
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(query_count, 1)
        self.assertEqual(excluded_count, 1)

    def test_teacher_bank_is_deterministic_row_aligned_and_reusable(self) -> None:
        rows = self.train_rows()
        tokenizer = DeterministicTokenizer()
        lm = DeterministicHiddenLM()
        lm.train()

        raw_audio_bank = stage.lm_text_embeddings(
            lm,
            tokenizer,
            [str(row["transcript"]) for row in rows],
            torch.device("cpu"),
            max_length=32,
            batch_size=2,
        )
        first = stage.build_speech_shared_teacher_bank(
            lm,
            tokenizer,
            rows,
            torch.device("cpu"),
            max_length=32,
            strict_split_binding=self.strict_binding(),
            build_batch_size=64,
            **self.teacher_identity_kwargs(),
        )
        second = stage.build_speech_shared_teacher_bank(
            lm,
            tokenizer,
            rows,
            torch.device("cpu"),
            max_length=32,
            strict_split_binding=self.strict_binding(),
            build_batch_size=64,
            **self.teacher_identity_kwargs(),
        )

        self.assertTrue(torch.equal(raw_audio_bank, first[0]))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[3]["row_identity_sha256"], second[3]["row_identity_sha256"])
        self.assertTrue(any(len(batch) > 1 for batch in tokenizer.calls))
        self.assertEqual(lm.grad_enabled, [False] * len(tokenizer.calls))
        self.assertTrue(lm.training)
        self.assertEqual(first[3]["teacher_embedding_batch_size"], 64)
        self.assertEqual(first[3]["bank_order"], "audio_train_row_order")
        self.assertEqual(
            first[3]["bank_tensor_identity"]["sha256"],
            second[3]["bank_tensor_identity"]["sha256"],
        )
        self.assertEqual(
            first[3]["speech_train_file"]["sha256"], "c" * 64
        )

        calls_before_reuse = len(tokenizer.calls)
        raw_audio_identity = stage.tensor_content_identity(raw_audio_bank)
        reusable = stage.build_speech_shared_teacher_bank(
            lm,
            tokenizer,
            rows,
            torch.device("cpu"),
            max_length=32,
            strict_split_binding=self.strict_binding(),
            build_batch_size=64,
            reusable_bank=raw_audio_bank,
            reusable_bank_exact=True,
            reusable_bank_expected_sha256=raw_audio_identity["sha256"],
            **self.teacher_identity_kwargs(),
        )
        self.assertTrue(torch.equal(reusable[0], raw_audio_bank))
        self.assertEqual(reusable[1], first[1])
        self.assertEqual(
            reusable[3]["bank_source"], "reused_row_aligned_audio_raw_bank"
        )
        self.assertEqual(len(tokenizer.calls), calls_before_reuse)
        self.assertEqual(
            reusable[3]["reusable_bank_input_identity"]["sha256"],
            raw_audio_identity["sha256"],
        )
        eval_batch_banks = []
        for eval_batch_size in (1, 8, 32):
            args = self.valid_args(eval_batch_size=eval_batch_size)
            fixed_bank_batch_size = stage.speech_teacher_bank_batch_size(args)
            self.assertEqual(fixed_bank_batch_size, 64)
            eval_batch_bank = stage.build_speech_shared_teacher_bank(
                lm,
                tokenizer,
                rows,
                torch.device("cpu"),
                max_length=32,
                strict_split_binding=self.strict_binding(),
                build_batch_size=fixed_bank_batch_size,
                **self.teacher_identity_kwargs(),
            )
            self.assertEqual(
                eval_batch_bank[3]["teacher_embedding_batch_size"], 64
            )
            self.assertTrue(eval_batch_bank[3]["eval_batch_size_independent"])
            eval_batch_banks.append(eval_batch_bank[0])
        for bank in eval_batch_banks:
            self.assertTrue(torch.equal(bank, first[0]))

    def test_same_shape_reusable_bank_has_content_bound_provenance(self) -> None:
        rows = self.train_rows()
        tokenizer = DeterministicTokenizer()
        lm = DeterministicHiddenLM()
        first_bank = F.normalize(
            torch.tensor(
                [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
            ),
            dim=-1,
        )
        second_bank = first_bank.clone()
        second_bank[0] = F.normalize(torch.tensor([3.0, 1.0, 2.0]), dim=0)
        first_identity = stage.tensor_content_identity(first_bank)
        second_identity = stage.tensor_content_identity(second_bank)

        first = stage.build_speech_shared_teacher_bank(
            lm,
            tokenizer,
            rows,
            torch.device("cpu"),
            max_length=32,
            strict_split_binding=self.strict_binding(),
            reusable_bank=first_bank,
            reusable_bank_exact=True,
            reusable_bank_expected_sha256=first_identity["sha256"],
            **self.teacher_identity_kwargs(),
        )
        second = stage.build_speech_shared_teacher_bank(
            lm,
            tokenizer,
            rows,
            torch.device("cpu"),
            max_length=32,
            strict_split_binding=self.strict_binding(),
            reusable_bank=second_bank,
            reusable_bank_exact=True,
            reusable_bank_expected_sha256=second_identity["sha256"],
            **self.teacher_identity_kwargs(),
        )
        self.assertNotEqual(
            first[3]["bank_tensor_identity"]["sha256"],
            second[3]["bank_tensor_identity"]["sha256"],
        )
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            stage.build_speech_shared_teacher_bank(
                lm,
                tokenizer,
                rows,
                torch.device("cpu"),
                max_length=32,
                strict_split_binding=self.strict_binding(),
                reusable_bank=second_bank,
                reusable_bank_exact=True,
                reusable_bank_expected_sha256=first_identity["sha256"],
                **self.teacher_identity_kwargs(),
            )

    def test_runtime_identities_bind_model_weights_and_tokenizer_config(self) -> None:
        model_identity, tokenizer_identity = (
            stage.speech_shared_teacher_runtime_identities(
                DeterministicHiddenLM(),
                DeterministicTokenizer(),
                "allenai/OLMoE-1B-7B-0924",
                multimodal_initialization=None,
                stage_b_initialization=None,
            )
        )
        self.assertEqual(
            model_identity["weight_sources"]["immutable_hf_checkpoint"]["revision"],
            "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
        )
        self.assertEqual(
            model_identity["weights_identity_semantics"],
            "immutable_hf_revision_plus_applied_checkpoint_sha256",
        )
        self.assertEqual(
            tokenizer_identity["name_or_path"],
            "allenai/OLMoE-1B-7B-0924",
        )
        self.assertEqual(tokenizer_identity["revision"], "6d84c48581ece794365f2b8e9cfb043c68ade9c5")
        self.assertRegex(tokenizer_identity["config_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(tokenizer_identity["vocab_sha256"], r"^[0-9a-f]{64}$")

    def test_teacher_bank_rejects_nontrain_and_sealed_rows(self) -> None:
        tokenizer = DeterministicTokenizer()
        lm = DeterministicHiddenLM()
        for partition in ("dev", "eval", "sealed"):
            rows = self.train_rows()
            rows[1]["partition"] = partition
            with self.subTest(partition=partition):
                with self.assertRaisesRegex(ValueError, "train partition rows only"):
                    stage.build_speech_shared_teacher_bank(
                        lm,
                        tokenizer,
                        rows,
                        torch.device("cpu"),
                        max_length=32,
                        strict_split_binding=self.strict_binding(),
                        **self.teacher_identity_kwargs(),
                    )

        rows = self.train_rows()
        rows[0]["audio_path"] = "/tmp/sealed/audio.wav"
        with self.assertRaisesRegex(ValueError, "forbidden 'sealed'"):
            stage.build_speech_shared_teacher_bank(
                lm,
                tokenizer,
                rows,
                torch.device("cpu"),
                max_length=32,
                strict_split_binding=self.strict_binding(),
                **self.teacher_identity_kwargs(),
            )

        bank = stage.build_speech_shared_teacher_bank(
            lm,
            tokenizer,
            self.train_rows(),
            torch.device("cpu"),
            max_length=32,
            strict_split_binding=self.strict_binding(),
            **self.teacher_identity_kwargs(),
        )
        provenance = bank[3]
        self.assertEqual(provenance["source_partitions"], ["train"])
        self.assertFalse(provenance["sealed_evidence_used"])
        self.assertFalse(provenance["dev_partition_used"])
        self.assertFalse(provenance["eval_partition_used"])

    def test_split_binding_binds_manifest_file_rows_and_source_commit(self) -> None:
        binding = stage.validate_speech_shared_split_binding(
            self.strict_split_provenance(), 3
        )
        self.assertTrue(binding["binding_verified"])
        self.assertEqual(
            binding["development_split_manifest"],
            {
                "path": "/tmp/development_splits/manifest.json",
                "sha256": "b" * 64,
                "source_commit_sha": "a" * 40,
                "builder": {
                    "path": "/tmp/repo/scripts/materialize_eval_splits.py",
                    "sha256": "f" * 64,
                    "source_commit_sha": "a" * 40,
                    "source_commit_exists": True,
                    "source_matches_commit": True,
                    "current_bytes_match_commit": True,
                    "command": "python scripts/materialize_eval_splits.py",
                },
                "expected_speech_source_sha256": "9" * 64,
                "trusted_digest_verified": True,
            },
        )
        self.assertEqual(
            binding["speech_train_file"],
            {
                "path": "/tmp/development_splits/speech_train.jsonl",
                "sha256": "c" * 64,
                "rows": 3,
            },
        )
        self.assertEqual(
            binding["speech_source_file"],
            {
                "path": "/tmp/development_data/speech_transcripts.jsonl",
                "sha256": "9" * 64,
                "rows": 5250,
                "audio_bytes_verified_this_run": True,
                "audio_rows_verified": 5250,
                "audio_row_binding_root_sha256": "4" * 64,
                "partition_commitments": {
                    "policy": "explicit_source_partition_canonical_membership_v1",
                    "partitions": {
                        "train": stage.speech_partition_record(
                            self.train_rows(), "train", annotated=True
                        ),
                        "dev": {"membership_root_sha256": "2" * 64},
                        "eval": {"membership_root_sha256": "3" * 64},
                    },
                },
                "expected_speech_source_sha256": "9" * 64,
                "trusted_digest_verified": True,
            },
        )
        self.assertTrue(binding["reserved_eval_split_file_unread"])
        self.assertTrue(
            binding["raw_source_eval_rows_read_for_partition_verification"]
        )

        legacy = {
            "policy": "legacy_tail_split",
            "selection_splits": ["train", "tail_eval"],
        }
        with self.assertRaisesRegex(ValueError, "legacy split provenance"):
            stage.validate_speech_shared_split_binding(legacy, 3)
        with self.assertRaisesRegex(ValueError, "verified speech_train"):
            stage.validate_speech_shared_split_binding(
                self.strict_split_provenance(), 4
            )
        opened_reserved = self.strict_split_provenance()
        opened_reserved["files"]["speech_eval"]["content_opened"] = True
        with self.assertRaisesRegex(ValueError, "reserved eval split file unread"):
            stage.validate_speech_shared_split_binding(opened_reserved, 3)
        mismatched_builder = self.strict_split_provenance()
        mismatched_builder["builder"]["source_commit_sha"] = "0" * 40
        with self.assertRaisesRegex(
            ValueError, "builder provenance is incomplete"
        ):
            stage.validate_speech_shared_split_binding(mismatched_builder, 3)

    def test_binding_rejects_missing_external_trust_digest(self) -> None:
        provenance = self.strict_split_provenance()
        provenance.pop("expected_speech_source_sha256")
        provenance["trusted_digest_verified"] = False
        with self.assertRaisesRegex(ValueError, "externally trusted source digest"):
            stage.validate_speech_shared_split_binding(provenance, 3)

    def test_teacher_bank_rejects_forged_same_count_train_rows(self) -> None:
        rows = self.train_rows()
        rows[1]["transcript"] = "forged transcript with the same row count"
        with self.assertRaisesRegex(ValueError, "row commitments mismatch"):
            stage.build_speech_shared_teacher_bank(
                DeterministicHiddenLM(),
                DeterministicTokenizer(),
                rows,
                torch.device("cpu"),
                max_length=32,
                strict_split_binding=self.strict_binding(),
                **self.teacher_identity_kwargs(),
            )

    def test_provenance_names_one_direction_full_train_bank(self) -> None:
        provenance = stage.speech_shared_contrastive_provenance(self.valid_args())
        self.assertTrue(provenance["enabled"])
        self.assertIn("shared_olmoe", provenance["student_path"])
        self.assertIn("full_train_bank", provenance["teacher_path"])
        self.assertEqual(
            provenance["student_pooling"],
            stage.SPEECH_SHARED_CONTRASTIVE_STUDENT_POOLING,
        )
        self.assertEqual(provenance["normalization"], "l2")
        self.assertEqual(
            provenance["objective"],
            "student_to_full_transcript_teacher_train_bank_infonce",
        )
        self.assertNotIn("symmetric", provenance["objective"])
        self.assertEqual(
            provenance["teacher_bank_partition_policy"], "speech_train_rows_only"
        )

    def test_validation_rejects_invalid_or_non_frozen_configuration(self) -> None:
        invalid_cases = (
            ({"speech_shared_contrastive_coef": float("nan")}, "finite and non-negative"),
            ({"speech_shared_contrastive_temperature": 0.0}, "finite and positive"),
            ({"speech_teacher_bank_batch_size": 0}, "batch size must be positive"),
            ({"development_split_manifest": ""}, "strict development split manifest"),
            ({"train_router_gates": True}, "frozen router/expert/LM"),
            ({"expert_selection_json": "selection.json"}, "selected experts"),
            ({"dynamic_expert_bias_lr": 0.1}, "dynamic router-bias"),
        )
        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    stage.validate_speech_shared_contrastive_request(
                        self.valid_args(**overrides)
                    )

        disabled = self.valid_args(
            speech_shared_contrastive_coef=0.0,
            train_router_gates=True,
        )
        stage.validate_speech_shared_contrastive_request(disabled)

    def test_env_is_forwarded_through_runai_and_runner_scripts(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        submit = (repo_root / "scripts" / "submit_runai.sh").read_text()
        runner = (repo_root / "run.sh").read_text()
        for name in (
            "SPEECH_SHARED_CONTRASTIVE_COEF",
            "SPEECH_SHARED_CONTRASTIVE_TEMPERATURE",
            "SPEECH_TEACHER_BANK_BATCH_SIZE",
        ):
            self.assertIn(f"--environment {name}=", submit)
            self.assertIn(name, runner)


if __name__ == "__main__":
    unittest.main()
