from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import run_missing_ablations as ablations
from scripts.run_missing_ablations import (
    make_train_args,
    materialize_gamma,
    sha256_file,
    write_matched_final_artifact,
)


class SelectedFinalAblationTests(unittest.TestCase):
    def test_materializes_byte_identical_gamma_for_checkpoint_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "output"
            (source / "calibration").mkdir(parents=True)
            payload = b'{"gamma":[1.0,0.5]}\n'
            source_gamma = source / "calibration" / "gamma.json"
            source_gamma.write_bytes(payload)

            self.assertEqual(materialize_gamma(source, output), [1.0, 0.5])
            target = output / "calibration" / "gamma.json"
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(sha256_file(target), sha256_file(source_gamma))

            target.write_bytes(b'{"gamma":[2.0]}\n')
            with self.assertRaisesRegex(ValueError, "disagrees"):
                materialize_gamma(source, output)

    def test_source_checkpoint_configuration_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "args": {
                            "audio_bridge_type": "attention_pool",
                            "development_split_manifest": "/tmp/dev.json",
                            "expert_selection_json": "/tmp/esft.json",
                            "final_steps": 1000,
                            "speech_unfreeze_last_blocks": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )
            cli = type(
                "Args",
                (),
                {
                    "data_dir": None,
                    "base_model": None,
                    "vision_model": None,
                    "speech_model": None,
                    "feature_cache_dir": None,
                    "ablation_steps": 300,
                    "capacity_ablation_steps": 300,
                    "text_eval_blocks": None,
                    "retrieval_eval_samples": None,
                    "conditional_eval_samples": None,
                    "conditional_batch_size": None,
                },
            )()
            args = make_train_args(root, root / "out", cli)
            self.assertEqual(args.audio_bridge_type, "attention_pool")
            self.assertEqual(args.development_split_manifest, "/tmp/dev.json")
            self.assertEqual(args.expert_selection_json, "/tmp/esft.json")
            self.assertEqual(args.final_steps, 1000)
            self.assertEqual(args.speech_unfreeze_last_blocks, 1)

    def test_strict_split_loader_keeps_reserved_eval_unread(self) -> None:
        real = Mock()
        real.load_manifest.return_value = {"real_subset": True}
        real.read_jsonl.side_effect = [[{"text": "train"}], [{"text": "eval"}]]
        real.load_development_multimodal_partitions.return_value = (
            [{"image": "train"}],
            [{"image": "dev"}],
            [{"audio": "train"}],
            [{"audio": "dev"}],
            {"reserved_files_opened": False},
        )
        args = SimpleNamespace(
            data_dir="/canonical/data",
            development_split_manifest="/canonical/dev.json",
            development_speech_source_sha256="a" * 64,
            image_eval_samples=137,
            speech_eval_samples=137,
        )
        with patch.object(ablations, "_REAL_MODULE", real):
            result = ablations.load_splits(args)
        self.assertEqual(result[3], [{"image": "train"}])
        self.assertEqual(result[4], [{"image": "dev"}])
        self.assertEqual(result[5], [{"audio": "train"}])
        self.assertEqual(result[6], [{"audio": "dev"}])
        real.load_development_multimodal_partitions.assert_called_once_with(
            "/canonical/dev.json",
            expected_data_dir=Path("/canonical/data"),
            expected_speech_source_sha256="a" * 64,
        )
        self.assertEqual(real.read_jsonl.call_count, 2)
        self.assertIs(args.development_split_provenance["reserved_files_opened"], False)
        real.absolutize_media_paths.assert_not_called()
        real.split_tail.assert_not_called()

    def test_writes_checkpoint_bound_final_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "selected.pt"
            source.write_bytes(b"selected")
            source_sha = sha256_file(source)
            experiment_id = "E4_no_aux_load_balance_ablation"
            experiment_dir = root / experiment_id
            experiment_dir.mkdir()
            checkpoint = experiment_dir / "checkpoint_final.pt"
            checkpoint.write_bytes(b"e4")
            rows = [
                {
                    "step": step,
                    "optimizer_step": True,
                    "modality": "image",
                    "loss": float(step),
                    "aux_coef": 0.0,
                    "router_aux_loss_weighted": 0.0,
                    "initial_checkpoint_state_restored": True,
                    "source_selected_checkpoint_sha256": source_sha,
                }
                for step in range(1, 301)
            ]
            raw = {
                "meta": {"capacity_factor": 8.0},
                "checkpoint_path": str(checkpoint),
                "steps": rows,
            }
            (experiment_dir / "metrics.json").write_text(json.dumps(raw), encoding="utf-8")
            output = write_matched_final_artifact(
                "E4", experiment_id, raw, root, source, source_sha
            )
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["artifact_type"], "matched_ablation_final")
            self.assertEqual(artifact["source_selected_checkpoint_sha256"], source_sha)
            self.assertEqual(artifact["claim_scope"], "continuation_sensitivity_only")
            self.assertEqual(artifact["training_iterations"], 300)
            self.assertEqual(artifact["optimizer_step_count"], 300)
            self.assertEqual(artifact["frozen_text_row_count"], 0)
            self.assertEqual(artifact["checkpoint"]["sha256"], sha256_file(checkpoint))

    def test_records_frozen_text_rows_without_calling_them_optimizer_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "selected.pt"
            source.write_bytes(b"selected")
            source_sha = sha256_file(source)
            experiment_id = "E4_no_aux_load_balance_ablation"
            experiment_dir = root / experiment_id
            experiment_dir.mkdir()
            checkpoint = experiment_dir / "checkpoint_final.pt"
            checkpoint.write_bytes(b"e4")
            rows = []
            for step in range(1, 301):
                frozen_text = (step - 1) % 4 == 0
                rows.append(
                    {
                        "step": step,
                        "optimizer_step": not frozen_text,
                        "modality": "text" if frozen_text else "speech",
                        "train_router_gates": False,
                        "train_experts": False,
                        "train_lm_head": False,
                        "loss": float(step),
                        "initial_checkpoint_state_restored": True,
                        "source_selected_checkpoint_sha256": source_sha,
                    }
                )
            raw = {
                "meta": {"capacity_factor": 8.0},
                "checkpoint_path": str(checkpoint),
                "steps": rows,
            }
            (experiment_dir / "metrics.json").write_text(
                json.dumps(raw), encoding="utf-8"
            )
            output = write_matched_final_artifact(
                "E4", experiment_id, raw, root, source, source_sha
            )
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["training_iterations"], 300)
            self.assertEqual(artifact["optimizer_step_count"], 225)
            self.assertEqual(artifact["frozen_text_row_count"], 75)

            rows[0]["modality"] = "image"
            with self.assertRaisesRegex(ValueError, "unexplained"):
                write_matched_final_artifact(
                    "E4", experiment_id, raw, root, source, source_sha
                )

    def test_rejects_rows_not_restored_from_selected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "selected.pt"
            source.write_bytes(b"selected")
            source_sha = sha256_file(source)
            experiment_id = "E5_capacity_1p25_ablation"
            experiment_dir = root / experiment_id
            experiment_dir.mkdir()
            checkpoint = experiment_dir / "checkpoint_final.pt"
            checkpoint.write_bytes(b"e5")
            rows = [
                {
                    "step": step,
                    "optimizer_step": True,
                    "modality": "speech",
                    "loss": 1.0,
                    "initial_checkpoint_state_restored": step != 1,
                    "source_selected_checkpoint_sha256": source_sha,
                }
                for step in range(1, 301)
            ]
            raw = {
                "meta": {"capacity_factor": 1.25},
                "checkpoint_path": str(checkpoint),
                "steps": rows,
            }
            (experiment_dir / "metrics.json").write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not restore"):
                write_matched_final_artifact(
                    "E5", experiment_id, raw, root, source, source_sha
                )

    def test_launcher_forwards_partial_experiment_selection(self) -> None:
        launcher = (Path(__file__).resolve().parents[1] / "scripts" / "submit_selected_final_ablations.sh").read_text(encoding="utf-8")
        submit = (Path(__file__).resolve().parents[1] / "scripts" / "submit_runai.sh").read_text(encoding="utf-8")
        self.assertIn('ABLATION_EXPERIMENTS="${ABLATION_EXPERIMENTS:-E4,E5}"', launcher)
        self.assertIn(
            'OUT="${OUT:?OUT is required and must be external to RUN_ROOT}"',
            launcher,
        )
        self.assertIn("final ablation OUT must be external to RUN_ROOT", launcher)
        self.assertIn('OUT="$OUT" SOURCE_OUTPUT_DIR="$RUN_ROOT"', launcher)
        self.assertIn('done["status"]=="completed"', launcher)
        self.assertNotIn('final_steps"])==6000', launcher)
        self.assertIn('--environment ABLATION_EXPERIMENTS="${ABLATION_EXPERIMENTS:-}"', submit)
        self.assertIn('--environment RECOVER_EXISTING="${RECOVER_EXISTING:-}"', submit)
        run_script = (Path(__file__).resolve().parents[1] / "run.sh").read_text(encoding="utf-8")
        self.assertIn('RECOVER_EXISTING_ARG=(--skip-existing)', run_script)


if __name__ == "__main__":
    unittest.main()
