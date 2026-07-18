from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_stageb_stagec_development as summary


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class StageBStageCDevelopmentSummaryTest(unittest.TestCase):
    def test_step_validation_accepts_resume_offset_but_rejects_gaps(self):
        issues = []
        observed = summary.validate_steps(
            [{"step": 3001, "loss": 1.0}, {"step": 3002, "loss": 0.9}],
            Path("development/resume.jsonl"),
            issues,
            ("loss",),
        )
        self.assertEqual(observed, 3002)
        self.assertEqual(issues, [])
        summary.validate_steps(
            [{"step": 3001}, {"step": 3003}],
            Path("development/gap.jsonl"),
            issues,
        )
        self.assertIn("non_contiguous_steps", {item["code"] for item in issues})

    def test_cli_accepts_repeatable_stageb_roots(self):
        args = summary.parse_args(
            [
                "--stageb-root",
                "development/b1",
                "--stageb-root",
                "development/b2",
                "--stagec-root",
                "development/c",
                "--output-dir",
                "development/report",
            ]
        )
        self.assertEqual(
            args.stageb_root, [Path("development/b1"), Path("development/b2")]
        )

    def make_stage_b(self, root: Path, run_id: str = "b-fixture_seed42", steps: int = 2) -> Path:
        run = root / run_id
        write_json(
            run / "manifest.json",
            {
                "args": {"data_dir": "/data/real_subset_fixture", "distill_steps": steps, "output_dir": str(run)},
                "data_policy": "development_only_real_manifests",
            },
        )
        for offset, variant in enumerate(summary.STAGE_B_VARIANTS):
            exp = run / variant
            checkpoint = exp / "checkpoint_final.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{run_id}-{variant}".encode("ascii"))
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            metrics = {
                "experiment_id": variant,
                "stage": "CE_control" if variant == "E2_CE_only" else "B",
                "perplexity": 15.0 - offset,
                "next_token_accuracy": 0.5 + offset * 0.01,
                "teacher_student_kl": 1.7 - offset * 0.1,
                "router_kl": 0.04 - offset * 0.001,
                "moe_reconstruction_cosine": 0.60 + offset * 0.01,
                "moe_reconstruction_mse": 0.002 - offset * 0.0001,
                "moe_reconstruction_rmse": 0.044 - offset * 0.001,
                "capacity_overflow_ratio_mean": 0.001 + offset * 0.0001,
                "inactive_expert_ratio_mean": 0.004 + offset * 0.001,
                "training_completed_steps": steps,
                "checkpoint_provenance": {
                    "development_data_only": True,
                    "saved_checkpoint": str(checkpoint),
                    "saved_checkpoint_sha256": digest,
                    "saved_checkpoint_size_bytes": checkpoint.stat().st_size,
                    "stage": "B",
                },
            }
            write_json(exp / "metrics.json", metrics)
            write_jsonl(exp / "train_metrics.jsonl", [{"step": step, "loss": 2.0} for step in range(1, steps + 1)])
        return run

    def make_multimodal_source(self, root: Path) -> dict[str, object]:
        source = root.parent / "stage-a-source"
        checkpoint = source / summary.STAGE_C_DIR / "checkpoint_final.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"verified-stage-a-checkpoint")
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        source_commit = "c" * 40
        job_name = "stage-a-fixture-job"
        project = "fixture-project"
        companion = {
            "completion": {
                "e3_checkpoint_path": str(checkpoint),
                "e3_checkpoint_sha256": checkpoint_sha,
                "e3_steps": 500,
                "status": "completed",
            },
            "run_provenance": {
                "runai_job_name": job_name,
                "runai_project": project,
                "sealed_evidence_used": False,
                "source_commit_sha": source_commit,
                "synthetic_evidence_used": False,
            },
            "runai_job_name": job_name,
            "runai_project": project,
            "source_commit_sha": source_commit,
        }
        manifest_path = source / "manifest.json"
        write_json(manifest_path, companion)
        return {
            "completion_status": "completed",
            "completion_step": 500,
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "path": str(checkpoint),
            "policy": "development_only_stage_a_multimodal_initialization",
            "runai_job_name": job_name,
            "runai_project": project,
            "sealed_evidence_used": False,
            "sha256": checkpoint_sha,
            "source_commit_sha": source_commit,
            "synthetic_evidence_used": False,
        }

    def make_stage_c(self, root: Path, run_id: str = "c-fixture_seed42", steps: int = 2) -> Path:
        run = root / run_id
        stage = run / summary.STAGE_C_DIR
        checkpoint = stage / "checkpoint_final.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"stage-c-checkpoint")
        stage_b_sha = "a" * 64
        multimodal = self.make_multimodal_source(root)
        multimodal_sha = multimodal["sha256"]
        manifest = {
            "args": {
                "alignment_pretrain_steps": steps,
                "data_dir": "/data/real_subset_fixture",
                "final_steps": steps,
                "multimodal_initial_checkpoint": multimodal["path"],
                "multimodal_initial_checkpoint_sha256": multimodal_sha,
                "multimodal_initial_manifest": multimodal["manifest_path"],
                "output_dir": str(run),
                "stage_b_checkpoint": "/development/stage-b.pt",
                "stage_b_checkpoint_sha256": stage_b_sha,
            },
            "data_manifest": {"counts": {"image_eval_pairs": 10, "speech_eval_utterances": 10}},
            "multimodal_initialization": multimodal,
            "stage_b_initialization": {
                "path": "/development/stage-b.pt",
                "policy": "development_only_stage_b_top8_to_top2_initialization",
                "sealed_evidence_used": False,
                "sha256": stage_b_sha,
                "synthetic_evidence_used": False,
            },
        }
        write_json(run / "manifest.json", manifest)
        base_row = {
            "capacity_overflow_ratio_mean": 0.02,
            "inactive_expert_ratio_mean": 0.1,
            "initial_checkpoint_state_restored": True,
            "loss": 2.0,
            "modality": "speech",
            "modality_assignment_conservation": {"audio_prefix": True},
            "modality_token_counts_across_layers": {"audio_prefix": 8},
            "modality_token_k_conservation_ok": True,
            "prefix_expected_assignments_tokens_x_layers_x_k": 16,
            "prefix_observed_assignments": 16,
            "prefix_routing_included": True,
            "source_selected_checkpoint_sha256": multimodal_sha,
            "source_stage_b_checkpoint_sha256": stage_b_sha,
            "stage_b_checkpoint_state_restored": True,
        }
        write_jsonl(stage / "train_metrics.jsonl", [{**base_row, "step": step} for step in range(1, steps + 1)])
        write_jsonl(stage / "alignment_pretrain_metrics.jsonl", [{"loss": 1.0, "step": step} for step in range(1, steps + 1)])
        retrieval = {
            "conditional_image_to_text_r_at_1": 0.3,
            "conditional_image_chance_r_at_1": 0.1,
            "conditional_speech_to_text_r_at_1": 0.2,
            "conditional_speech_chance_r_at_1": 0.1,
            "image_to_text_r_at_1": 0.02,
            "image_chance_r_at_1": 0.01,
            "speech_to_text_r_at_1": 0.03,
            "speech_chance_r_at_1": 0.01,
            "conditional_uses_direct_encoder_pooling": False,
            "conditional_uses_lm_logits": True,
            "retrieval_uses_direct_encoder_pooling": False,
            "retrieval_uses_lm_hidden_states": True,
        }
        write_json(stage / "metrics.json", {"real_subset": True, "retrieval_eval": retrieval})
        write_json(
            run / summary.STAGE_C_TEXT_DIR / "metrics.json",
            {
                "capacity_overflow_ratio_mean": 0.001,
                "inactive_expert_ratio_mean": 0.004,
                "next_token_accuracy": 0.51,
                "perplexity": 14.5,
                "provenance": {
                    "copied_from_e2": False,
                    "source_checkpoint": str(checkpoint),
                    "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "source_checkpoint_size_bytes": checkpoint.stat().st_size,
                    "source_experiment_id": summary.STAGE_C_DIR,
                    "source_training_steps": steps,
                },
                "real_subset": True,
            },
        )
        return run

    def roots(self, temporary: str) -> tuple[Path, Path, Path, Path]:
        base = Path(temporary)
        return base / "stageb-screen", base / "stageb-followup", base / "stagec", base / "report"

    def test_complete_fixture_writes_deterministic_json_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            screen, followup, stagec, output = self.roots(temporary)
            self.make_stage_b(screen)
            followup.mkdir()
            self.make_stage_c(stagec)

            first = summary.summarize((screen, followup), stagec, output)
            json_path = output / f"{summary.OUTPUT_STEM}.json"
            first_bytes = json_path.read_bytes()
            second = summary.summarize((screen, followup), stagec, output)

            self.assertTrue(first["validation_passed"])
            self.assertEqual(first, second)
            self.assertEqual(first_bytes, json_path.read_bytes())
            comparison = first["stage_b_runs"][0]["comparison"]
            self.assertEqual(comparison["delta_e2d_minus_ce"]["perplexity"], -1.0)
            self.assertEqual(first["stage_c_runs"][0]["retrieval_metrics"]["conditional_image_r1"], 0.3)
            with (output / f"{summary.OUTPUT_STEM}.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[-1]["embedding_speech_chance_r1"], "0.01")
            markdown = (output / f"{summary.OUTPUT_STEM}.md").read_text()
            self.assertIn("E2D-CE PPL", markdown)
            self.assertIn("Cond image/chance", markdown)

    def test_rejects_path_only_multimodal_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            screen, followup, stagec, output = self.roots(temporary)
            screen.mkdir()
            followup.mkdir()
            run = self.make_stage_c(stagec)
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["args"].pop("multimodal_initial_manifest")
            for field in (
                "completion_status",
                "completion_step",
                "manifest_path",
                "manifest_sha256",
                "runai_job_name",
                "runai_project",
                "source_commit_sha",
            ):
                manifest["multimodal_initialization"].pop(field)
            write_json(manifest_path, manifest)

            payload = summary.summarize((screen, followup), stagec, output)
            result = payload["stage_c_runs"][0]
            self.assertFalse(result["validation_passed"])
            self.assertEqual(result["status"], "rejected")
            self.assertIn(
                "unverified_multimodal_initial_manifest",
                {item["code"] for item in result["issues"]},
            )

    def test_multimodal_companion_manifest_validation_fails_closed(self):
        cases = (
            "copied_checkpoint_wrong_path",
            "missing_manifest_hash",
            "missing_completion",
            "missing_source",
            "sealed_flag",
            "synthetic_flag",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                screen, followup, stagec, output = self.roots(temporary)
                screen.mkdir()
                followup.mkdir()
                run = self.make_stage_c(stagec)
                manifest_path = run / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                initialization = manifest["multimodal_initialization"]
                companion_path = Path(initialization["manifest_path"])
                companion = json.loads(companion_path.read_text())

                if case == "copied_checkpoint_wrong_path":
                    copied = companion_path.parent / "copied_checkpoint.pt"
                    copied.write_bytes(Path(initialization["path"]).read_bytes())
                    initialization["path"] = str(copied)
                    manifest["args"]["multimodal_initial_checkpoint"] = str(copied)
                elif case == "missing_manifest_hash":
                    initialization.pop("manifest_sha256")
                elif case == "missing_completion":
                    companion.pop("completion")
                elif case == "missing_source":
                    companion.pop("source_commit_sha")
                    companion["run_provenance"].pop("source_commit_sha")
                elif case == "sealed_flag":
                    initialization["sealed_evidence_used"] = True
                elif case == "synthetic_flag":
                    companion["run_provenance"]["synthetic_evidence_used"] = True

                if case in {"missing_completion", "missing_source", "synthetic_flag"}:
                    write_json(companion_path, companion)
                    initialization["manifest_sha256"] = hashlib.sha256(
                        companion_path.read_bytes()
                    ).hexdigest()
                write_json(manifest_path, manifest)

                payload = summary.summarize((screen, followup), stagec, output)
                result = payload["stage_c_runs"][0]
                self.assertFalse(result["validation_passed"])
                self.assertEqual(result["status"], "rejected")
                self.assertIn(
                    "unverified_multimodal_initial_manifest",
                    {item["code"] for item in result["issues"]},
                )

    def test_running_root_is_explicit_and_missing_values_are_not_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            screen, followup, stagec, output = self.roots(temporary)
            screen.mkdir()
            followup.mkdir()
            run = self.make_stage_c(stagec, run_id="c-running_seed42")
            manifest = json.loads((run / "manifest.json").read_text())
            multimodal_sha = manifest["multimodal_initialization"]["sha256"]
            write_jsonl(
                run / summary.STAGE_C_DIR / "train_metrics.jsonl",
                [{"initial_checkpoint_state_restored": True, "loss": 2.0, "source_selected_checkpoint_sha256": multimodal_sha, "source_stage_b_checkpoint_sha256": "a" * 64, "stage_b_checkpoint_state_restored": True, "step": 1}],
            )
            (run / summary.STAGE_C_DIR / "metrics.json").unlink()
            (run / summary.STAGE_C_DIR / "checkpoint_final.pt").unlink()
            (run / summary.STAGE_C_TEXT_DIR / "metrics.json").unlink()
            payload = summary.summarize((screen, followup), stagec, output)
            result = payload["stage_c_runs"][0]
            self.assertEqual(result["status"], "running")
            self.assertIsNone(result["checkpoint_sha256"])
            self.assertIn("missing_checkpoint", {item["code"] for item in result["issues"]})
            csv_text = (output / f"{summary.OUTPUT_STEM}.csv").read_text()
            self.assertNotIn(",0,0,0,0,", csv_text)
            self.assertIn("NA", (output / f"{summary.OUTPUT_STEM}.md").read_text())

    def test_rejects_copied_text_metrics_and_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            screen, followup, stagec, output = self.roots(temporary)
            screen.mkdir()
            followup.mkdir()
            run = self.make_stage_c(stagec)
            path = run / summary.STAGE_C_TEXT_DIR / "metrics.json"
            metrics = json.loads(path.read_text())
            metrics["provenance"]["copied_from_e2"] = True
            path.write_text(json.dumps(metrics), encoding="utf-8")
            payload = summary.summarize((screen, followup), stagec, output)
            result = payload["stage_c_runs"][0]
            self.assertEqual(result["status"], "rejected")
            self.assertIn("copied_text_metrics", {item["code"] for item in result["issues"]})

    def test_rejects_bypass_retrieval_and_missing_prefix_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            screen, followup, stagec, output = self.roots(temporary)
            screen.mkdir()
            followup.mkdir()
            run = self.make_stage_c(stagec)
            metrics_path = run / summary.STAGE_C_DIR / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["retrieval_eval"]["conditional_uses_lm_logits"] = False
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            train_path = run / summary.STAGE_C_DIR / "train_metrics.jsonl"
            rows = [json.loads(line) for line in train_path.read_text().splitlines()]
            rows[-1]["prefix_routing_included"] = False
            write_jsonl(train_path, rows)
            payload = summary.summarize((screen, followup), stagec, output)
            result = payload["stage_c_runs"][0]
            self.assertEqual(result["status"], "rejected")
            codes = {item["code"] for item in result["issues"]}
            self.assertIn("bypass_retrieval_path", codes)
            self.assertIn("missing_prefix_routing_accounting", codes)

    def test_rejects_nonfinite_string_metric_nonreal_data_and_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            screen, followup, stagec, output = self.roots(temporary)
            run = self.make_stage_b(screen)
            followup.mkdir()
            stagec.mkdir()
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["args"]["data_dir"] = "/data/generated_fixture"
            manifest["data_policy"] = "generated"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            ce_path = run / "E2_CE_only" / "metrics.json"
            ce = json.loads(ce_path.read_text())
            ce["perplexity"] = "15.0"
            ce_path.write_text(json.dumps(ce), encoding="utf-8")
            e2d_path = run / "E2D_logits_kl" / "metrics.json"
            e2d = json.loads(e2d_path.read_text())
            e2d["router_kl"] = float("inf")
            e2d_path.write_text(json.dumps(e2d), encoding="utf-8")
            (run / "E2_CE_only" / "checkpoint_final.pt").unlink()

            payload = summary.summarize((screen, followup), stagec, output)
            result = payload["stage_b_runs"][0]
            self.assertEqual(result["status"], "rejected")
            codes = {item["code"] for item in result["issues"]}
            self.assertTrue({"non_real_data", "invalid_metric_type", "invalid_metrics", "missing_checkpoint"}.issubset(codes))

    def test_refuses_forbidden_root_and_output_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            safe = base / "development"
            safe.mkdir()
            with self.assertRaisesRegex(summary.EvidenceError, "forbidden word"):
                summary.summarize((base / "sealed-root", safe), safe, base / "report")
            with self.assertRaisesRegex(summary.EvidenceError, "forbidden word"):
                summary.summarize((safe, safe), safe, base / "synthetic-report")


if __name__ == "__main__":
    unittest.main()
