import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts import select_corrected_candidate as selector


CANDIDATE_SPECS = {
    "cap6a02": {
        "capacity": 6.0,
        "aux": 0.02,
        "ppl": 13.0,
        "drops": 5,
        "inactive": 0.05,
        "load_cv": 0.50,
        "recall_counts": {5: (65, 55), 10: (30, 28)},
    },
    "cap7a01": {
        "capacity": 7.0,
        "aux": 0.01,
        "ppl": 12.8,
        "drops": 4,
        "inactive": 0.04,
        "load_cv": 0.45,
        "recall_counts": {5: (70, 60), 10: (32, 30)},
    },
    "cap7a02": {
        "capacity": 7.0,
        "aux": 0.02,
        "ppl": 13.5,
        "drops": 3,
        "inactive": 0.03,
        "load_cv": 0.40,
        "recall_counts": {5: (100, 90), 10: (55, 50)},
    },
    "cap7a04": {
        "capacity": 7.0,
        "aux": 0.04,
        "ppl": 12.5,
        "drops": 2,
        "inactive": 0.02,
        "load_cv": 0.35,
        "recall_counts": {5: (90, 80), 10: (45, 42)},
    },
    "cap8a02": {
        "capacity": 8.0,
        "aux": 0.02,
        "ppl": 12.0,
        "drops": 1,
        "inactive": 0.01,
        "load_cv": 0.30,
        "recall_counts": {5: (80, 70), 10: (40, 38)},
    },
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CorrectedCandidateFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_dir = str(root / "real_data")
        self.evaluator = root / "eval_conditional_retrieval.py"
        self.evaluator.write_text("# deterministic fixture evaluator\n", encoding="utf-8")
        self.development_root = root / "development_eval_v1"
        self.image_manifest = self.development_root / "image_val.jsonl"
        self.speech_manifest = self.development_root / "speech_val.jsonl"
        image_rows = [
            {"uid": f"image-{index}", "caption": f"caption {index}", "source": "fixture-image"}
            for index in range(selector.REQUIRED_QUERIES_PER_MODALITY)
        ]
        speech_rows = [
            {
                "utterance_id": f"speech-{index}",
                "transcript": f"transcript {index}",
                "source": "fixture-speech",
            }
            for index in range(selector.REQUIRED_QUERIES_PER_MODALITY)
        ]
        write_jsonl(self.image_manifest, image_rows)
        write_jsonl(self.speech_manifest, speech_rows)
        write_json(
            self.development_root / "manifest.json",
            {
                "compatible_conditional_eval": {
                    "validation": {
                        "CONDITIONAL_QUERIES": selector.REQUIRED_QUERIES_PER_MODALITY,
                        "QUERY_OFFSET": 0,
                    }
                },
                "data_dir": self.data_dir,
                "files": {
                    "image_val": str(self.image_manifest.resolve()),
                    "speech_val": str(self.speech_manifest.resolve()),
                },
                "val_count": selector.REQUIRED_QUERIES_PER_MODALITY,
            },
        )
        self.run_roots: dict[str, Path] = {}
        self.dev_roots: dict[str, Path] = {}

    def build_all(self) -> None:
        for name, spec in CANDIDATE_SPECS.items():
            run_root = self.root / "runs" / name
            dev_root = self.root / "development" / name
            self.write_candidate(name, spec, run_root, dev_root, selector.REQUIRED_STEPS)
            self.run_roots[name] = run_root.resolve()
            self.dev_roots[name] = dev_root.resolve()

    def training_row(self, step: int, spec: dict) -> dict:
        cycle = ("text", "image_caption", "image_caption", "speech_transcript", "speech_transcript")
        modality = cycle[(step - 1) % len(cycle)]
        ce = 3.2 - step / 12000.0
        raw_aux = 2.0 + (step % 7) * 0.01
        weighted_aux = spec["aux"] * raw_aux
        reconciliation_gap = selector.BASE_LOSS_RECONCILIATION_ABS_TOL * 0.8
        drops = int(spec["drops"])
        attempted_counts = [50 + drops, 50 - drops]
        accepted_counts = [50, 50 - drops]
        dropped_counts = [drops, 0]
        row = {
            "accepted_load_cv": spec["load_cv"],
            "aux_coef": spec["aux"],
            "capacity_enforced": True,
            "capacity_factor": spec["capacity"],
            "capacity_overflow_ratio_mean": drops / 100.0,
            "ce_loss": ce,
            "conditional_ranking_coef": 0.0,
            "conditional_ranking_loss": 0.0,
            "contrastive_coef": 0.0,
            "contrastive_loss": 0.0,
            "experiment_id": selector.E3_DIR,
            "hf_reported_loss": ce + weighted_aux + reconciliation_gap,
            "hf_reported_loss_minus_explicit_base": reconciliation_gap,
            "inactive_expert_ratio_mean": spec["inactive"],
            "lm_ce_loss": ce,
            "loss": ce + weighted_aux,
            "loss_equation": "lm_ce_loss + aux_coef * router_aux_loss_raw + modality_losses + router_z_loss",
            "modality": modality,
            "optimizer_step": True,
            "router_aux_loss_raw": raw_aux,
            "router_aux_loss_weighted": weighted_aux,
            "router_layers": 1,
            "router_z_loss": 0.0,
            "router_z_loss_coef": 0.0,
            "routing_accepted_assignments_total": 100 - drops,
            "routing_accounting_source": "patched_dispatch_masks_after_capacity",
            "routing_attempted_assignments_total": 100,
            "routing_capacity_compliant": True,
            "routing_conservation_ok": True,
            "routing_denominator": "token_expert_assignments_across_layers",
            "routing_dropped_assignments_total": drops,
            "routing_layer_accounting": [
                {
                    "accepted_assignments": 100 - drops,
                    "accepted_expert_counts": accepted_counts,
                    "attempted_assignments": 100,
                    "attempted_expert_counts": attempted_counts,
                    "capacity_compliant": True,
                    "capacity_per_expert": 50,
                    "conservation_ok": True,
                    "dropped_assignments": drops,
                    "dropped_expert_counts": dropped_counts,
                    "layer": 0,
                    "token_count": 50,
                    "top_k": 2,
                }
            ],
            "routing_token_count_across_layers": 50,
            "runtime_top_k": 2,
            "step": step,
            "top_k": 2,
        }
        if modality != "text":
            prefix = "image_prefix" if modality == "image_caption" else "audio_prefix"
            row["modality_assignment_conservation"] = {prefix: True, "text": True}
        return row

    def write_candidate(
        self,
        name: str,
        spec: dict,
        run_root: Path,
        dev_root: Path,
        step_count: int,
    ) -> None:
        run_root.mkdir(parents=True, exist_ok=True)
        args = {
            "aux_coef": spec["aux"],
            "base_model": "fixture/olmoe",
            "capacity_factor": spec["capacity"],
            "data_dir": self.data_dir,
            "feature_cache_dir": str((run_root / "feature_cache").resolve()),
            "final_steps": selector.REQUIRED_STEPS,
            "learning_rate": 0.0005,
            "output_dir": str(run_root.resolve()),
            "seed": 42,
            "train_lm_head": True,
        }
        manifest = {
            "args": args,
            "base_model": "fixture/olmoe",
            "command_mode": "real-required-runs",
            "data_dir": self.data_dir,
            "data_manifest": {
                "counts": {"image": 5250, "speech": 5250, "text": 30000},
                "output_dir": self.data_dir,
                "sources": {
                    "image": "fixture-image-source",
                    "speech": "fixture-speech-source",
                    "text": "fixture-text-source",
                },
            },
            "output_dir": str(run_root.resolve()),
            "speech_model": "fixture/whisper",
            "splits": {"image_eval_pairs": 250, "speech_eval_utterances": 250, "text_eval_blocks": 160},
            "vision_model": "fixture/clip",
        }
        write_json(run_root / "manifest.json", manifest)

        checkpoint = run_root / selector.E3_DIR / "checkpoint_final.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes((f"checkpoint:{name}:" * 4).encode("ascii"))
        rows = [self.training_row(step, spec) for step in range(1, step_count + 1)]
        write_jsonl(run_root / selector.E3_DIR / "train_metrics.jsonl", rows)
        provenance = {
            "copied_from_e2": False,
            "lm_trainable": True,
            "model_state_source": "in_memory_wrapper_after_training_saved_to_checkpoint",
            "source_checkpoint": str(checkpoint.resolve()),
            "source_checkpoint_saved_before_eval": True,
            "source_checkpoint_size_bytes": checkpoint.stat().st_size,
            "source_experiment_id": selector.E3_DIR,
            "source_training_steps": step_count,
            "text_eval_note": "Metrics come from the final E3 checkpoint.",
        }
        text_metrics = {
            "aux_coef": spec["aux"],
            "capacity_enforced": True,
            "capacity_factor": spec["capacity"],
            "loss": math.log(spec["ppl"]),
            "perplexity": spec["ppl"],
            "provenance": provenance,
            "runtime_top_k": 2,
            "top_k": 2,
        }
        independent_text = {**text_metrics, "expert_counts_total": [50, 50 - int(spec["drops"])]}
        write_json(run_root / selector.E3_TEXT_DIR / "metrics.json", independent_text)
        retrieval = {
            "conditional_candidates_per_query": 10,
            "conditional_eval_path": "shared_olmoe_prefix_lm_nll",
            "conditional_image_chance_r_at_1": 0.1,
            "conditional_image_eval_count": 250,
            "conditional_image_to_text_r_at_1": 0.2,
            "conditional_speech_chance_r_at_1": 0.1,
            "conditional_speech_eval_count": 250,
            "conditional_speech_to_text_r_at_1": 0.15,
            "conditional_uses_direct_encoder_pooling": False,
            "conditional_uses_lm_logits": True,
            "image_to_text_r_at_1": 0.02,
            "retrieval_path": "shared_olmoe_prefix_hidden",
            "retrieval_uses_direct_encoder_pooling": False,
            "retrieval_uses_lm_hidden_states": True,
            "speech_to_text_r_at_1": 0.012,
        }
        e3_metrics = {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "final_capacity_overflow_ratio_mean": rows[-1]["capacity_overflow_ratio_mean"],
            "final_inactive_expert_ratio_mean": rows[-1]["inactive_expert_ratio_mean"],
            "first_loss": rows[0]["loss"],
            "last_loss": rows[-1]["loss"],
            "meta": {
                "aux_coef": spec["aux"],
                "capacity_enforced": True,
                "capacity_factor": spec["capacity"],
                "runtime_top_k": 2,
                "top_k": 2,
            },
            "min_loss": min(row["loss"] for row in rows),
            "retrieval_eval": retrieval,
            "steps": rows,
            "text_eval": text_metrics,
            "text_eval_provenance": provenance,
        }
        write_json(run_root / selector.E3_DIR / "metrics.json", e3_metrics)
        self.write_development_evals(name, spec, run_root, dev_root, checkpoint)

    def query_rows(self, candidate_count: int, image_successes: int, speech_successes: int) -> list[dict]:
        rows: list[dict] = []
        for modality, successes in (("image", image_successes), ("speech", speech_successes)):
            uids = [f"{modality}:{modality}-{index}" for index in range(250)]
            for index, query_uid in enumerate(uids):
                gold = index % candidate_count
                candidate_ids = [uids[(index + offset) % len(uids)] for offset in range(1, candidate_count)]
                candidate_ids.insert(gold, query_uid)
                if index < successes:
                    scores = [0.0] * candidate_count
                    scores[gold] = 2.0
                else:
                    scores = [0.0] * candidate_count
                    scores[gold] = 1.0
                    negative = 0 if gold != 0 else 1
                    scores[negative] = 2.0
                predicted = max(range(candidate_count), key=lambda position: scores[position])
                rank = sorted(
                    range(candidate_count), key=lambda position: scores[position], reverse=True
                ).index(gold)
                candidate_hash = hashlib.sha256(
                    json.dumps(candidate_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                rows.append(
                    {
                        "candidate_count": candidate_count,
                        "candidate_ids": candidate_ids,
                        "candidate_index": gold,
                        "candidate_set_hash": candidate_hash,
                        "condition": "real",
                        "eval_path": "shared_prefix",
                        "eval_split_name": "development_selection",
                        "gold_candidate_id": query_uid,
                        "gold_position": gold,
                        "modality": modality,
                        "negative_mode": "random",
                        "predicted_candidate_id": candidate_ids[predicted],
                        "predicted_position": predicted,
                        "prefix_control": "real",
                        "protocol": {
                            "candidate_count": candidate_count,
                            "candidate_seed": 271828,
                            "eval_split_name": "development_selection",
                            "manifest_sha256": None,
                            "name": "development_conditional_v2",
                            "negative_mode": "random",
                            "hard_negative_selector": None,
                            "randomized_positive_position": True,
                            "rank_base": 0,
                            "score_direction": "higher_is_better",
                        },
                        "query_index": index,
                        "query_uid": query_uid,
                        "rank": rank,
                        "rank_base": 0,
                        "score_direction": "higher_is_better",
                        "scores": scores,
                    }
                )
        return rows

    def write_development_evals(
        self, name: str, spec: dict, run_root: Path, dev_root: Path, checkpoint: Path
    ) -> None:
        manifest_path = run_root / "manifest.json"
        for candidate_count in (5, 10):
            cell = dev_root / f"r{candidate_count}"
            image_successes, speech_successes = spec["recall_counts"][candidate_count]
            rows = self.query_rows(candidate_count, image_successes, speech_successes)
            per_query = cell / "per_query.jsonl"
            write_jsonl(per_query, rows)
            image_positions = [0] * candidate_count
            speech_positions = [0] * candidate_count
            for row in rows:
                target = image_positions if row["modality"] == "image" else speech_positions
                target[row["gold_position"]] += 1
            image_r1 = image_successes / 250.0
            speech_r1 = speech_successes / 250.0
            chance = 1.0 / candidate_count
            metrics = {
                "candidate_count": candidate_count,
                "candidate_offset": -1,
                "candidate_seed": 271828,
                "checkpoint": str(checkpoint.resolve()),
                "condition": "real",
                "conditional_candidates_per_query": candidate_count,
                "conditional_eval_path": "shared_olmoe_prefix_lm_nll",
                "conditional_image_chance_r_at_1": chance,
                "conditional_image_eval_count": 250,
                "conditional_image_to_text_r_at_1": image_r1,
                "conditional_speech_candidates_per_query": candidate_count,
                "conditional_speech_chance_r_at_1": chance,
                "conditional_speech_eval_count": 250,
                "conditional_speech_to_text_r_at_1": speech_r1,
                "conditional_uses_direct_encoder_pooling": False,
                "conditional_uses_lm_logits": True,
                "conditional_uses_multimodal_prefix": True,
                "control_seed": 42,
                "eval_path": "shared_prefix",
                "eval_split_name": "development_selection",
                "image_chance_r_at_1": chance,
                "image_eval_count": 250,
                "image_gold_position_counts": image_positions,
                "image_split_source": "explicit_manifest",
                "image_to_text_r_at_1": image_r1,
                "meta": {
                    "aux_coef": spec["aux"],
                    "capacity_enforced": True,
                    "capacity_factor": spec["capacity"],
                    "runtime_top_k": 2,
                    "top_k": 2,
                },
                "mode": "conditional_nll_local_negatives",
                "negative_mode": "random",
                "hard_negative_selector": None,
                "per_query_output": str(per_query.resolve()),
                "per_query_rows": len(rows),
                "per_query_sha256": sha256(per_query),
                "prefix_control": "real",
                "protocol_manifest_path": None,
                "protocol_manifest_sha256": None,
                "protocol_name": "development_conditional_v2",
                "provenance": {
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256(checkpoint),
                    "evaluator_path": str(self.evaluator.resolve()),
                    "evaluator_sha256": sha256(self.evaluator),
                    "feature_cache_dir": str((cell / "feature_cache").resolve()),
                    "image_manifest_path": str(self.image_manifest.resolve()),
                    "image_manifest_sha256": sha256(self.image_manifest),
                    "protocol_manifest_path": None,
                    "protocol_manifest_sha256": None,
                    "source_run_manifest_path": str(manifest_path.resolve()),
                    "source_run_manifest_sha256": sha256(manifest_path),
                    "speech_manifest_path": str(self.speech_manifest.resolve()),
                    "speech_manifest_sha256": sha256(self.speech_manifest),
                },
                "query_offset": 0,
                "randomized_positive_position": True,
                "run_output_dir": str(run_root.resolve()),
                "sealed_protocol": False,
                "speech_candidate_count": candidate_count,
                "speech_chance_r_at_1": chance,
                "speech_eval_count": 250,
                "speech_gold_position_counts": speech_positions,
                "speech_split_source": "explicit_manifest",
                "speech_to_text_r_at_1": speech_r1,
            }
            write_json(cell / "metrics.json", metrics)


class SelectCorrectedCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.fixture = CorrectedCandidateFixture(cls.base)
        cls.fixture.build_all()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_deterministic_selection_and_refuse_overwrite(self) -> None:
        output_a = self.base / "selection-a"
        output_b = self.base / "selection-b"
        report_a = selector.select_candidates(
            dict(reversed(list(self.fixture.run_roots.items()))),
            dict(reversed(list(self.fixture.dev_roots.items()))),
            output_a,
        )
        report_b = selector.select_candidates(
            self.fixture.run_roots,
            self.fixture.dev_roots,
            output_b,
        )
        self.assertEqual(report_a["selected_candidate"], "cap7a02")
        self.assertEqual(report_b["selected_candidate"], "cap7a02")
        self.assertEqual(
            (output_a / "candidate_selection.json").read_bytes(),
            (output_b / "candidate_selection.json").read_bytes(),
        )
        self.assertTrue((output_a / "candidate_selection.md").is_file())
        self.assertTrue((output_a / "candidate_metrics.csv").is_file())
        with self.assertRaisesRegex(selector.SelectionError, "overwrite"):
            selector.select_candidates(self.fixture.run_roots, self.fixture.dev_roots, output_a)

    def test_copied_and_mismatched_development_metrics_are_rejected(self) -> None:
        copied_root = self.base / "copied-development"
        shutil.copytree(self.fixture.dev_roots["cap7a02"], copied_root)
        copied_dirs = dict(self.fixture.dev_roots)
        copied_dirs["cap7a01"] = copied_root
        copied_report = selector.select_candidates(
            self.fixture.run_roots,
            copied_dirs,
            self.base / "selection-copied",
        )
        copied_candidate = next(
            candidate for candidate in copied_report["candidates"] if candidate["name"] == "cap7a01"
        )
        self.assertFalse(copied_candidate["valid"])
        self.assertIn("checkpoint_hash", {reason["code"] for reason in copied_candidate["reasons"]})

        mismatched_root = self.base / "mismatched-development"
        shutil.copytree(self.fixture.dev_roots["cap6a02"], mismatched_root)
        metrics_path = mismatched_root / "r5" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["per_query_output"] = str((mismatched_root / "r5" / "per_query.jsonl").resolve())
        metrics["image_to_text_r_at_1"] = 0.999
        metrics["conditional_image_to_text_r_at_1"] = 0.999
        write_json(metrics_path, metrics)
        mismatched_dirs = dict(self.fixture.dev_roots)
        mismatched_dirs["cap6a02"] = mismatched_root
        mismatch_report = selector.select_candidates(
            self.fixture.run_roots,
            mismatched_dirs,
            self.base / "selection-mismatched",
        )
        mismatched_candidate = next(
            candidate for candidate in mismatch_report["candidates"] if candidate["name"] == "cap6a02"
        )
        self.assertFalse(mismatched_candidate["valid"])
        self.assertIn("mismatched_metrics", {reason["code"] for reason in mismatched_candidate["reasons"]})

    def test_partial_training_is_ineligible(self) -> None:
        name = "cap6a02"
        partial_run = self.base / "partial-run" / name
        partial_dev = self.base / "partial-development" / name
        self.fixture.write_candidate(
            name,
            CANDIDATE_SPECS[name],
            partial_run,
            partial_dev,
            selector.REQUIRED_STEPS - 1,
        )
        run_roots = dict(self.fixture.run_roots)
        dev_roots = dict(self.fixture.dev_roots)
        run_roots[name] = partial_run
        dev_roots[name] = partial_dev
        report = selector.select_candidates(
            run_roots,
            dev_roots,
            self.base / "selection-partial",
        )
        partial = next(candidate for candidate in report["candidates"] if candidate["name"] == name)
        self.assertFalse(partial["valid"])
        self.assertIn("partial_training", {reason["code"] for reason in partial["reasons"]})
        self.assertIsNotNone(report["selected_candidate"])

    def test_reconciliation_tolerance_rejects_material_gap(self) -> None:
        spec = CANDIDATE_SPECS["cap7a02"]
        row = self.fixture.training_row(1, spec)
        excessive_gap = selector.BASE_LOSS_RECONCILIATION_ABS_TOL * 2.0
        row["hf_reported_loss"] = (
            row["lm_ce_loss"] + row["router_aux_loss_weighted"] + excessive_gap
        )
        row["hf_reported_loss_minus_explicit_base"] = excessive_gap
        with self.assertRaises(selector.ArtifactError) as context:
            selector._validate_objective_row(
                row,
                "material-gap",
                spec["capacity"],
                spec["aux"],
            )
        self.assertEqual(context.exception.code, "objective_mismatch")

    def test_sealed_path_is_refused_before_metrics_are_opened(self) -> None:
        sealed_root = self.base / "sealed_eval_do_not_open"
        (sealed_root / "r5").mkdir(parents=True)
        (sealed_root / "r5" / "metrics.json").write_text("not json", encoding="utf-8")
        dev_roots = dict(self.fixture.dev_roots)
        dev_roots["cap6a02"] = sealed_root
        output = self.base / "selection-sealed"
        with self.assertRaisesRegex(selector.SelectionError, "sealed"):
            selector.select_candidates(self.fixture.run_roots, dev_roots, output)
        self.assertFalse(output.exists())

    def test_no_valid_candidate_fails_after_writing_diagnostics(self) -> None:
        names = sorted(self.fixture.dev_roots)
        rotated = {
            name: self.fixture.dev_roots[names[(index + 1) % len(names)]]
            for index, name in enumerate(names)
        }
        output = self.base / "selection-none"
        with self.assertRaises(selector.NoValidCandidateError):
            selector.select_candidates(self.fixture.run_roots, rotated, output)
        report = json.loads((output / "candidate_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "no_valid_candidate")
        self.assertIsNone(report["selected_candidate"])
        self.assertEqual(report["valid_candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
