"""Tests for the fail-closed MM dual-development summarizer."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_mm_dual_development as summary


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


class CampaignFixture:
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
        self.initializers = {
            "baseline": self.make_initializer(
                "image_linear", "image", b"image-linear", "linear_projector"
            ),
            "norm": self.make_initializer(
                "image_norm", "image", b"image-norm", "linear_projector_norm"
            ),
            "speech": self.make_initializer(
                "speech_last1_ln", "speech", b"speech-last1-ln", None
            ),
        }
        for index, (name, spec) in enumerate(summary.ARM_SPECS.items()):
            self.make_arm(name, spec, index)

    def make_initializer(
        self, name: str, scope: str, content: bytes, bridge: str | None
    ) -> dict:
        checkpoint = self.base / "initializers" / name / "checkpoint_final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(content)
        manifest_path = checkpoint.parent / "manifest.json"
        source = hashlib.sha1(name.encode()).hexdigest()
        provenance = {
            "source_commit_sha": source,
            "runai_job_name": f"job-{name}",
            "runai_project": "test-project",
            "resolved_data_root": str(self.data.resolve()),
            "resolved_output_root": str(checkpoint.parent.resolve()),
            "final_main_steps": 500,
            "alignment_pretrain_steps": 400,
            "checkpoint_completed_step": 500,
            "policy": "development_only_stage_a_multimodal_initialization",
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
        }
        args = {
            "final_steps": 500,
            "alignment_pretrain_steps": 400,
            "alignment_pretrain_modalities": scope,
        }
        if bridge is not None:
            args.update(image_bridge_type=bridge, image_prefix_tokens=50)
        else:
            args.update(
                speech_unfreeze_last_blocks=1,
                speech_unfreeze_layer_norm=True,
                audio_bridge_type="attention_pool",
                audio_prefix_tokens=64,
            )
        manifest = {
            "args": args,
            "run_provenance": provenance,
            "completion": {
                "status": "completed",
                "e3_checkpoint_path": str(checkpoint.resolve()),
                "e3_checkpoint_sha256": digest(checkpoint),
                "e3_steps": 500,
            },
        }
        write_json(manifest_path, manifest)
        return {
            "path": str(checkpoint.resolve()),
            "sha256": digest(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": digest(manifest_path),
            "source_commit_sha": source,
            "runai_job_name": f"job-{name}",
            "runai_project": "test-project",
            "completion_status": "completed",
            "completion_step": 500,
            "policy": "development_only_stage_a_multimodal_initialization",
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
        image = dict(self.initializers[spec["image_group"]])
        speech = dict(self.initializers["speech"]) if spec["speech_initializer"] else {}
        stage_b = {
            "path": str(self.stage_b.resolve()),
            "sha256": digest(self.stage_b),
            "size_bytes": self.stage_b.stat().st_size,
            "policy": "development_only_stage_b_top8_to_top2_initialization",
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
        }
        args = {
            "final_steps": 500,
            "alignment_pretrain_steps": 400,
            "alignment_pretrain_modalities": "speech",
            "modality_cycle": "text,speech,speech",
            "seed": 42,
            "data_dir": str(self.data.resolve()),
            "output_dir": str(run.resolve()),
            "image_bridge_type": spec["image_bridge"],
            "multimodal_initialization_scope": "image",
            "multimodal_initial_checkpoint": image["path"],
            "multimodal_initial_checkpoint_sha256": image["sha256"],
            "multimodal_initial_manifest": image["manifest_path"],
            "speech_initial_checkpoint": speech.get("path", ""),
            "speech_initial_checkpoint_sha256": speech.get("sha256", ""),
            "speech_initial_manifest": speech.get("manifest_path", ""),
            "stage_b_checkpoint": stage_b["path"],
            "stage_b_checkpoint_sha256": stage_b["sha256"],
            "speech_behavior_kl_coef": spec["kd_coefficient"],
            "speech_behavior_kl_temperature": 1.0,
        }
        provenance = {
            "source_commit_sha": "a" * 40,
            "runai_job_name": f"job-{name}",
            "runai_project": "test-project",
            "resolved_data_root": str(self.data.resolve()),
            "resolved_output_root": str(run.resolve()),
            "final_main_steps": 500,
            "alignment_pretrain_steps": 400,
            "checkpoint_completed_step": 500,
            "policy": "development_only_stage_a_multimodal_initialization",
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
        }
        completion = {
            "status": "completed",
            "e3_checkpoint_path": str(checkpoint.resolve()),
            "e3_checkpoint_sha256": digest(checkpoint),
            "e3_checkpoint_size_bytes": checkpoint.stat().st_size,
            "e3_steps": 500,
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
            "multimodal_initialization": image,
            "speech_initialization": speech,
            "completion": completion,
        }
        write_json(run / "manifest.json", manifest)

        rows = []
        for step in range(1, 501):
            modality = ("text", "speech", "speech")[(step - 1) % 3]
            ce_loss = 3.0 - step / 1000.0
            is_speech = modality == "speech"
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
                    "source_selected_checkpoint_sha256": image["sha256"],
                    "stage_b_checkpoint_state_restored": True,
                    "source_stage_b_checkpoint_sha256": stage_b["sha256"],
                    "speech_initial_checkpoint_state_restored": bool(speech),
                    "source_speech_initial_checkpoint_sha256": speech.get("sha256"),
                    "speech_behavior_kl_coef": spec["kd_coefficient"],
                    "prefix_routing_included": is_speech,
                    "modality_token_k_conservation_ok": is_speech,
                    "modality_token_counts_across_layers": {
                        "audio_prefix": 128 if is_speech else 0,
                        "image_prefix": 0,
                        "text": 256,
                    },
                    "modality_assignment_conservation": {
                        "audio_prefix": is_speech,
                        "image_prefix": True,
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
                    "stage": "alignment_pretrain",
                    "modality": "speech",
                    "loss": 1.5 - step / 1000.0,
                }
                for step in range(1, 401)
            ],
        )

        ppl = 12.5 - index * 0.1
        text_provenance = {
            "source_experiment_id": summary.E3_DIR,
            "source_checkpoint": str(checkpoint.resolve()),
            "source_checkpoint_size_bytes": checkpoint.stat().st_size,
            "source_checkpoint_sha256": digest(checkpoint),
            "source_training_steps": 500,
            "source_checkpoint_saved_before_eval": True,
            "model_state_source": "in_memory_wrapper_after_training_saved_to_checkpoint",
            "copied_from_e2": False,
            "lm_trainable": False,
        }
        text_metrics = {
            "real_subset": True,
            "perplexity": ppl,
            "next_token_accuracy": 0.5 + index * 0.01,
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
        metrics = {
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
        }
        write_json(stage / "metrics.json", metrics)

    def arm_path(self, name: str) -> Path:
        return self.root / name

    def load(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

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


class SummarizeMMDualDevelopmentTests(unittest.TestCase):
    def test_valid_fixture_writes_json_csv_markdown_and_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            payload = summary.summarize(fixture.root, fixture.output)
            self.assertTrue(payload["validation_passed"])
            self.assertEqual(len(payload["arms"]), 5)
            dual = payload["arms"][3]
            self.assertAlmostEqual(dual["deltas_vs_c0"]["perplexity"], -0.3)
            self.assertTrue(dual["promotion_flags"]["promote"])
            with (fixture.output / f"{summary.OUTPUT_STEM}.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertIn("delta_perplexity_vs_c0", rows[0])
            markdown = (
                fixture.output / f"{summary.OUTPUT_STEM}.md"
            ).read_text(encoding="utf-8")
            self.assertIn("Overall validation: **PASS**", markdown)
            self.assertIn("Gate entropy", markdown)

    def test_rejects_copied_checkpoint_hash_even_when_metadata_is_updated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            source = (
                fixture.arm_path("c0_seed42")
                / summary.E3_DIR
                / "checkpoint_final.pt"
            )
            target = (
                fixture.arm_path("c_dual_seed42")
                / summary.E3_DIR
                / "checkpoint_final.pt"
            )
            target.write_bytes(source.read_bytes())
            fixture.rewrite_final_fingerprint("c_dual_seed42")
            payload = summary.summarize(fixture.root, fixture.output)
            codes = {
                item["code"]
                for arm in payload["arms"]
                for item in arm["issues"]
            }
            self.assertFalse(payload["validation_passed"])
            self.assertIn("copied_checkpoint_hash", codes)

    def test_rejects_wrong_speech_initializer_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            path = fixture.arm_path("c_dual_seed42") / "manifest.json"
            manifest = fixture.load(path)
            manifest["speech_initialization"]["scope"] = "image"
            write_json(path, manifest)
            payload = summary.summarize(fixture.root, fixture.output)
            arm = next(item for item in payload["arms"] if item["name"] == "c_dual_seed42")
            self.assertFalse(arm["validation_passed"])
            self.assertIn("wrong_initializer_scope", {item["code"] for item in arm["issues"]})

    def test_rejects_nonfinite_and_divergent_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            nonfinite_path = (
                fixture.arm_path("c0_seed42")
                / summary.E3_DIR
                / "train_metrics.jsonl"
            )
            rows = [json.loads(line) for line in nonfinite_path.read_text().splitlines()]
            rows[10]["loss"] = math.inf
            write_jsonl(nonfinite_path, rows)

            divergent_path = (
                fixture.arm_path("c_dual_seed42")
                / summary.E3_DIR
                / "train_metrics.jsonl"
            )
            divergent = [json.loads(line) for line in divergent_path.read_text().splitlines()]
            for row in divergent:
                row["lm_ce_loss"] = 1.0 + row["step"] / 1000.0
            write_jsonl(divergent_path, divergent)
            metrics_path = fixture.arm_path("c_dual_seed42") / summary.E3_DIR / "metrics.json"
            metrics = fixture.load(metrics_path)
            metrics["steps"] = divergent
            write_json(metrics_path, metrics)

            payload = summary.summarize(fixture.root, fixture.output)
            codes = {
                item["code"]
                for arm in payload["arms"]
                for item in arm["issues"]
            }
            self.assertFalse(payload["validation_passed"])
            self.assertIn("invalid_jsonl", codes)
            self.assertIn("divergent_loss", codes)

    def test_rejects_missing_conditional_and_prefix_path_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CampaignFixture(Path(temporary))
            run = fixture.arm_path("c_dual_kd025_seed42")
            train_path = run / summary.E3_DIR / "train_metrics.jsonl"
            rows = [json.loads(line) for line in train_path.read_text().splitlines()]
            for row in rows:
                if row["modality"] == "speech":
                    row["prefix_routing_included"] = False
            write_jsonl(train_path, rows)
            metrics_path = run / summary.E3_DIR / "metrics.json"
            metrics = fixture.load(metrics_path)
            metrics["steps"] = rows
            metrics["retrieval_eval"].pop("conditional_uses_lm_logits")
            write_json(metrics_path, metrics)

            payload = summary.summarize(fixture.root, fixture.output)
            arm = next(
                item
                for item in payload["arms"]
                if item["name"] == "c_dual_kd025_seed42"
            )
            self.assertFalse(arm["validation_passed"])
            self.assertIn(
                "missing_prefix_path_flags",
                {item["code"] for item in arm["issues"]},
            )


if __name__ == "__main__":
    unittest.main()
