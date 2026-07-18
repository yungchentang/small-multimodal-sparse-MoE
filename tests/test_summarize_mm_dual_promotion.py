"""Tests for the fail-closed MM dual 3k promotion summarizer."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_mm_dual_promotion as summary


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PromotionFixture:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.root = base / "campaign"
        self.output = base / "report"
        self.root.mkdir()
        self.data = base / "real_subset_fixture"
        self.data.mkdir()
        self.data_manifest = {"dataset_policy": "real_development", "version": 1}
        write_json(self.data / "manifest.json", self.data_manifest)
        self.stage_b = base / "stage_b.pt"
        self.stage_b.write_bytes(b"stage-b-checkpoint")
        self.image = self.make_initializer("image_norm", "image", b"image-norm")
        self.speech = self.make_initializer(
            "speech_last1_ln", "speech", b"speech-last1-ln"
        )
        for index, (name, spec) in enumerate(summary.ARM_SPECS.items()):
            self.make_arm(name, spec, index)

    def make_initializer(self, name: str, scope: str, content: bytes) -> dict:
        checkpoint = self.base / "initializers" / name / "checkpoint_final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(content)
        manifest_path = checkpoint.parent / "manifest.json"
        source_commit = hashlib.sha1(name.encode()).hexdigest()
        provenance = {
            "source_commit_sha": source_commit,
            "runai_job_name": f"job-{name}",
            "runai_project": "test-project",
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
        }
        manifest = {
            "args": {
                "final_steps": summary.INITIALIZER_STEPS,
                "alignment_pretrain_steps": summary.ALIGNMENT_STEPS,
                "alignment_pretrain_modalities": scope,
            },
            "run_provenance": provenance,
            "completion": {
                "status": "completed",
                "e3_checkpoint_path": str(checkpoint.resolve()),
                "e3_checkpoint_sha256": digest(checkpoint),
                "e3_steps": summary.INITIALIZER_STEPS,
            },
        }
        write_json(manifest_path, manifest)
        return {
            "path": str(checkpoint.resolve()),
            "sha256": digest(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": digest(manifest_path),
            "source_commit_sha": source_commit,
            "runai_job_name": provenance["runai_job_name"],
            "runai_project": provenance["runai_project"],
            "completion_status": "completed",
            "completion_step": summary.INITIALIZER_STEPS,
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
            "scope": scope,
        }

    def make_arm(self, name: str, spec: dict, index: int) -> None:
        run = self.root / name
        stage = run / summary.E3_DIR
        text_dir = run / summary.E3_TEXT_DIR
        stage.mkdir(parents=True)
        text_dir.mkdir()
        checkpoint = stage / "checkpoint_final.pt"
        checkpoint.write_bytes(f"final-{name}".encode())
        stage_b = {
            "path": str(self.stage_b.resolve()),
            "sha256": digest(self.stage_b),
            "size_bytes": self.stage_b.stat().st_size,
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
        }
        args = {
            "final_steps": summary.MAIN_STEPS,
            "alignment_pretrain_steps": summary.ALIGNMENT_STEPS,
            "alignment_pretrain_modalities": "speech",
            "modality_cycle": "text,speech,speech",
            "seed": 42,
            "data_dir": str(self.data.resolve()),
            "output_dir": str(run.resolve()),
            "image_bridge_type": "linear_projector_norm",
            "multimodal_initialization_scope": "image",
            "multimodal_initial_checkpoint": self.image["path"],
            "multimodal_initial_checkpoint_sha256": self.image["sha256"],
            "multimodal_initial_manifest": self.image["manifest_path"],
            "speech_initial_checkpoint": self.speech["path"],
            "speech_initial_checkpoint_sha256": self.speech["sha256"],
            "speech_initial_manifest": self.speech["manifest_path"],
            "stage_b_checkpoint": stage_b["path"],
            "stage_b_checkpoint_sha256": stage_b["sha256"],
            "speech_behavior_kl_coef": spec["kd_coefficient"],
            "speech_behavior_kl_temperature": 1.0,
        }
        provenance = {
            "source_commit_sha": summary.EXPECTED_SOURCE_COMMIT,
            "runai_job_name": spec["runai_job_name"],
            "runai_project": "test-project",
            "resolved_data_root": str(self.data.resolve()),
            "resolved_output_root": str(run.resolve()),
            "final_main_steps": summary.MAIN_STEPS,
            "alignment_pretrain_steps": summary.ALIGNMENT_STEPS,
            "checkpoint_completed_step": summary.MAIN_STEPS,
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
        }
        manifest = {
            "source_commit_sha": provenance["source_commit_sha"],
            "runai_job_name": provenance["runai_job_name"],
            "runai_project": provenance["runai_project"],
            "data_dir": str(self.data.resolve()),
            "data_manifest": self.data_manifest,
            "args": args,
            "run_provenance": provenance,
            "stage_b_initialization": stage_b,
            "multimodal_initialization": dict(self.image),
            "speech_initialization": dict(self.speech),
            "completion": {
                "status": "completed",
                "e3_checkpoint_path": str(checkpoint.resolve()),
                "e3_checkpoint_sha256": digest(checkpoint),
                "e3_checkpoint_size_bytes": checkpoint.stat().st_size,
                "e3_steps": summary.MAIN_STEPS,
            },
        }
        write_json(run / "manifest.json", manifest)

        rows = []
        for step in range(1, summary.MAIN_STEPS + 1):
            modality = ("text", "speech", "speech")[(step - 1) % 3]
            is_speech = modality == "speech"
            ce_loss = 3.0 - step / 10000.0
            rows.append(
                {
                    "step": step,
                    "optimizer_step": is_speech,
                    "modality": modality,
                    "native_top_k": 8,
                    "runtime_top_k": 2,
                    "top_k": 2,
                    "loss": ce_loss + 0.1,
                    "lm_ce_loss": ce_loss,
                    "gate_entropy_mean": 2.0 + index * 0.01,
                    "inactive_expert_ratio_mean": 0.1,
                    "capacity_overflow_ratio_mean": 0.02,
                    "initial_checkpoint_state_restored": True,
                    "source_selected_checkpoint_sha256": self.image["sha256"],
                    "speech_initial_checkpoint_state_restored": True,
                    "source_speech_initial_checkpoint_sha256": self.speech["sha256"],
                    "stage_b_checkpoint_state_restored": True,
                    "source_stage_b_checkpoint_sha256": stage_b["sha256"],
                    "speech_behavior_kl_coef": spec["kd_coefficient"],
                    "prefix_routing_included": is_speech,
                    "modality_token_k_conservation_ok": is_speech,
                    "modality_token_counts_across_layers": {
                        "audio_prefix": 128 if is_speech else 0,
                        "text": 256,
                    },
                    "modality_assignment_conservation": {
                        "audio_prefix": is_speech,
                        "text": True,
                    },
                    "modality_routing_denominator": (
                        "token_expert_assignments_across_layers" if is_speech else ""
                    ),
                    "prefix_expected_assignments_tokens_x_layers_x_k": (
                        256 if is_speech else 0
                    ),
                    "prefix_observed_assignments": 256 if is_speech else 0,
                }
            )
        write_jsonl(stage / "train_metrics.jsonl", rows)
        write_jsonl(
            stage / "alignment_pretrain_metrics.jsonl",
            [
                {
                    "step": step,
                    "modality": "speech",
                    "loss": 1.5 - step / 1000.0,
                }
                for step in range(1, summary.ALIGNMENT_STEPS + 1)
            ],
        )

        text_provenance = {
            "source_experiment_id": summary.E3_DIR,
            "source_checkpoint": str(checkpoint.resolve()),
            "source_checkpoint_size_bytes": checkpoint.stat().st_size,
            "source_checkpoint_sha256": digest(checkpoint),
            "source_training_steps": summary.MAIN_STEPS,
            "source_checkpoint_saved_before_eval": True,
            "copied_from_e2": False,
        }
        text_metrics = {
            "real_subset": True,
            "perplexity": 12.5 - index * 0.2,
            "next_token_accuracy": 0.50 + index * 0.01,
            "provenance": text_provenance,
        }
        write_json(text_dir / "metrics.json", text_metrics)
        retrieval = {
            "retrieval_path": "shared_olmoe_prefix_hidden",
            "retrieval_uses_lm_hidden_states": True,
            "retrieval_uses_direct_encoder_pooling": False,
            "conditional_eval_path": "shared_olmoe_prefix_lm_nll",
            "conditional_uses_lm_logits": True,
            "conditional_uses_direct_encoder_pooling": False,
            "image_eval_count": 250,
            "speech_eval_count": 250,
            "conditional_image_eval_count": 250,
            "conditional_speech_eval_count": 250,
            "conditional_image_to_text_r_at_1": 0.30 + index * 0.01,
            "conditional_speech_to_text_r_at_1": 0.20 + index * 0.01,
            "image_to_text_r_at_1": 0.04 + index * 0.001,
            "speech_to_text_r_at_1": 0.03 + index * 0.001,
        }
        write_json(
            stage / "metrics.json",
            {
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": digest(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "text_eval_provenance": text_provenance,
                "real_subset": True,
                "steps": rows,
                "first_loss": rows[0]["loss"],
                "last_loss": rows[-1]["loss"],
                "min_loss": min(row["loss"] for row in rows),
                "text_eval": text_metrics,
                "retrieval_eval": retrieval,
                "final_gate_entropy_mean": rows[-1]["gate_entropy_mean"],
                "final_inactive_expert_ratio_mean": rows[-1][
                    "inactive_expert_ratio_mean"
                ],
                "final_capacity_overflow_ratio_mean": rows[-1][
                    "capacity_overflow_ratio_mean"
                ],
            },
        )

    def arm_path(self, name: str) -> Path:
        return self.root / name

    def load(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def load_rows(self, name: str) -> list[dict]:
        path = self.arm_path(name) / summary.E3_DIR / "train_metrics.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def write_rows(self, name: str, rows: list[dict], sync_metrics: bool = True) -> None:
        run = self.arm_path(name)
        write_jsonl(run / summary.E3_DIR / "train_metrics.jsonl", rows)
        if sync_metrics:
            metrics_path = run / summary.E3_DIR / "metrics.json"
            metrics = self.load(metrics_path)
            metrics["steps"] = rows
            write_json(metrics_path, metrics)

    def rewrite_final_fingerprint(self, name: str) -> None:
        run = self.arm_path(name)
        checkpoint = run / summary.E3_DIR / "checkpoint_final.pt"
        manifest_path = run / "manifest.json"
        metrics_path = run / summary.E3_DIR / "metrics.json"
        text_path = run / summary.E3_TEXT_DIR / "metrics.json"
        manifest = self.load(manifest_path)
        metrics = self.load(metrics_path)
        text = self.load(text_path)
        sha = digest(checkpoint)
        size = checkpoint.stat().st_size
        manifest["completion"].update(
            e3_checkpoint_sha256=sha, e3_checkpoint_size_bytes=size
        )
        metrics.update(checkpoint_sha256=sha, checkpoint_size_bytes=size)
        for provenance in (
            text["provenance"],
            metrics["text_eval_provenance"],
            metrics["text_eval"]["provenance"],
        ):
            provenance.update(
                source_checkpoint_sha256=sha, source_checkpoint_size_bytes=size
            )
        write_json(manifest_path, manifest)
        write_json(metrics_path, metrics)
        write_json(text_path, text)


class SummarizeMMDualPromotionTests(unittest.TestCase):
    def test_valid_fixture_writes_json_csv_markdown_and_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PromotionFixture(Path(temporary))
            payload = summary.summarize(fixture.root, fixture.output)
            self.assertTrue(payload["validation_passed"])
            self.assertEqual(payload["required_arms"], list(summary.ARM_SPECS))
            candidate = payload["arms"][1]
            self.assertAlmostEqual(
                candidate["deltas_vs_c_dual"]["perplexity"], -0.2
            )
            self.assertEqual(
                payload["comparison"]["candidate_deltas_vs_c_dual"],
                candidate["deltas_vs_c_dual"],
            )
            with (fixture.output / f"{summary.OUTPUT_STEM}.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("delta_perplexity_vs_c_dual", rows[0])
            markdown = (fixture.output / f"{summary.OUTPUT_STEM}.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Overall validation: **PASS**", markdown)
            self.assertIn("All deltas are candidate minus C_DUAL", markdown)

    def test_rejects_copied_checkpoint_with_consistent_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PromotionFixture(Path(temporary))
            source = (
                fixture.arm_path("c_dual_seed42")
                / summary.E3_DIR
                / "checkpoint_final.pt"
            )
            target = (
                fixture.arm_path("c_dual_kd025_seed42")
                / summary.E3_DIR
                / "checkpoint_final.pt"
            )
            target.write_bytes(source.read_bytes())
            fixture.rewrite_final_fingerprint("c_dual_kd025_seed42")
            payload = summary.summarize(fixture.root, fixture.output)
            codes = {
                item["code"] for arm in payload["arms"] for item in arm["issues"]
            }
            self.assertFalse(payload["validation_passed"])
            self.assertIn("copied_checkpoint_hash", codes)

    def test_rejects_wrong_steps_and_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PromotionFixture(Path(temporary))
            short_rows = fixture.load_rows("c_dual_seed42")[:-1]
            fixture.write_rows("c_dual_seed42", short_rows)
            wrong_top_k = fixture.load_rows("c_dual_kd025_seed42")
            wrong_top_k[100]["native_top_k"] = 2
            fixture.write_rows("c_dual_kd025_seed42", wrong_top_k)
            payload = summary.summarize(fixture.root, fixture.output)
            codes = {
                item["code"] for arm in payload["arms"] for item in arm["issues"]
            }
            self.assertFalse(payload["validation_passed"])
            self.assertIn("wrong_main_steps", codes)
            self.assertIn("wrong_runtime_top_k", codes)

    def test_rejects_nonfinite_and_divergent_losses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PromotionFixture(Path(temporary))
            nonfinite = fixture.load_rows("c_dual_seed42")
            nonfinite[10]["loss"] = math.inf
            fixture.write_rows("c_dual_seed42", nonfinite, sync_metrics=False)
            divergent = fixture.load_rows("c_dual_kd025_seed42")
            for row in divergent:
                row["lm_ce_loss"] = 1.0 + row["step"] / 1000.0
            fixture.write_rows("c_dual_kd025_seed42", divergent)
            payload = summary.summarize(fixture.root, fixture.output)
            codes = {
                item["code"] for arm in payload["arms"] for item in arm["issues"]
            }
            self.assertFalse(payload["validation_passed"])
            self.assertIn("invalid_jsonl", codes)
            self.assertIn("divergent_loss", codes)

    def test_rejects_text_metrics_copied_from_other_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PromotionFixture(Path(temporary))
            source_run = fixture.arm_path("c_dual_seed42")
            target_run = fixture.arm_path("c_dual_kd025_seed42")
            source_text = fixture.load(source_run / summary.E3_TEXT_DIR / "metrics.json")
            source_metrics = fixture.load(source_run / summary.E3_DIR / "metrics.json")
            target_metrics_path = target_run / summary.E3_DIR / "metrics.json"
            target_metrics = fixture.load(target_metrics_path)
            target_metrics["text_eval"] = copy.deepcopy(source_metrics["text_eval"])
            target_metrics["text_eval_provenance"] = copy.deepcopy(
                source_metrics["text_eval_provenance"]
            )
            write_json(target_metrics_path, target_metrics)
            write_json(target_run / summary.E3_TEXT_DIR / "metrics.json", source_text)
            payload = summary.summarize(fixture.root, fixture.output)
            candidate = payload["arms"][1]
            self.assertFalse(candidate["validation_passed"])
            self.assertIn(
                "copied_text_metrics",
                {item["code"] for item in candidate["issues"]},
            )


if __name__ == "__main__":
    unittest.main()
