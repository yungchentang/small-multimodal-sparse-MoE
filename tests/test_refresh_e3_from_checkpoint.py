from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scripts import refresh_e3_from_checkpoint as refresh
from scripts.audit_requirements import validate_e3_checkpoint_sidecar


class RefreshE3FromCheckpointTests(unittest.TestCase):
    def checkpoint_args(self, output_dir: Path) -> dict[str, object]:
        return {
            "base_model": "development/base",
            "vision_model": "development/vision",
            "speech_model": "development/speech",
            "top_k": 2,
            "capacity_factor": 4.0,
            "aux_coef": 0.01,
            "output_dir": str(output_dir),
        }

    def refresh_args(self, root: Path) -> SimpleNamespace:
        return SimpleNamespace(
            data_dir=root / "data",
            output_dir=root / "output",
            source_output_dir=root / "source",
            feature_cache_dir=None,
            refreshed_checkpoint=root / "output/E3_final_multimodal_top2/checkpoint_final.pt",
            stage_b_checkpoint=root / "stage_b.pt",
            stage_b_checkpoint_sha256="b" * 64,
            base_model="",
            vision_model="",
            speech_model="",
        )

    def strict_e3_fixture(
        self, root: Path, rows: list[dict[str, object]]
    ) -> dict[str, object]:
        experiment = root / "E3_final_multimodal_top2"
        experiment.mkdir(parents=True)
        checkpoint = experiment / "checkpoint_final.pt"
        state = {
            "args": self.checkpoint_args(root),
            "last_row": rows[-1],
            "trainable_meta": {},
        }
        torch.save(state, checkpoint)

        raw_path = experiment / "train_metrics.jsonl"
        raw_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        metrics_path = experiment / "metrics.json"
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        metrics = {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "text_eval_provenance": {
                "source_checkpoint": str(checkpoint.resolve()),
                "source_checkpoint_sha256": checkpoint_sha256,
            },
            "steps": rows,
        }
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

        checkpoint_args_path = experiment / "checkpoint_args.json"
        checkpoint_args_path.write_text(
            json.dumps(state["args"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run_manifest_path = root / "manifest.json"
        run_manifest_path.write_text(
            json.dumps({"args": state["args"]}), encoding="utf-8"
        )
        last_row = rows[-1]
        summary_keys = (
            "step",
            "modality",
            "loss",
            "lm_ce_loss",
            "router_aux_loss_raw",
            "router_aux_loss_weighted",
            "hf_reported_loss_minus_explicit_base",
        )
        sidecar_path = experiment / "checkpoint_provenance.json"
        sidecar = {
            "schema_version": 1,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_state_keys": sorted(state),
            "checkpoint_last_row_sha256": refresh.canonical_sha256(last_row),
            "checkpoint_last_row_summary": {
                key: last_row.get(key) for key in summary_keys
            },
            "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            "metrics_step_rows": len(rows),
            "checkpoint_last_row_matches_metrics_last_row": True,
            "checkpoint_args_path": str(checkpoint_args_path.resolve()),
            "checkpoint_args_sha256": hashlib.sha256(
                checkpoint_args_path.read_bytes()
            ).hexdigest(),
            "run_manifest_path": str(run_manifest_path.resolve()),
            "run_manifest_sha256": hashlib.sha256(
                run_manifest_path.read_bytes()
            ).hexdigest(),
            "ignored_path_only_arg_keys": ["feature_cache_dir", "output_dir"],
            "non_path_arg_mismatches": {},
            "passed": True,
        }
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        return {
            "checkpoint": checkpoint,
            "state": state,
            "raw_path": raw_path,
            "metrics_path": metrics_path,
            "checkpoint_args_path": checkpoint_args_path,
            "run_manifest_path": run_manifest_path,
            "sidecar_path": sidecar_path,
        }

    def test_verified_e3_checkpoint_requires_exact_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            torch.save({"args": {"top_k": 2}}, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            resolved, payload, state = refresh.load_verified_e3_checkpoint(
                checkpoint, digest
            )

            self.assertEqual(resolved, checkpoint.resolve())
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
            self.assertEqual(state["args"]["top_k"], 2)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                refresh.load_verified_e3_checkpoint(checkpoint, "0" * 64)
            with self.assertRaisesRegex(ValueError, "exact 64-character"):
                refresh.load_verified_e3_checkpoint(checkpoint, "short")

    def test_mirrored_refresh_binds_standard_output_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = Path("E3_final_multimodal_top2/checkpoint_final.pt")
            source_checkpoint = root / "source" / relative
            output_checkpoint = root / "output" / relative
            source_checkpoint.parent.mkdir(parents=True)
            output_checkpoint.parent.mkdir(parents=True)
            source_checkpoint.write_bytes(b"e3-checkpoint")
            output_checkpoint.write_bytes(source_checkpoint.read_bytes())
            digest = hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()

            resolved = refresh.resolve_refreshed_checkpoint(
                root / "source",
                root / "output",
                source_checkpoint.resolve(),
                digest,
                mirrored=True,
            )
            self.assertEqual(resolved, output_checkpoint.resolve())

            output_checkpoint.write_bytes(b"drifted")
            with self.assertRaisesRegex(ValueError, "refreshed E3 checkpoint SHA256"):
                refresh.resolve_refreshed_checkpoint(
                    root / "source",
                    root / "output",
                    source_checkpoint.resolve(),
                    digest,
                    mirrored=True,
                )

    def test_checkpoint_namespace_fails_closed_on_missing_base_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.refresh_args(root)
            state = {"args": self.checkpoint_args(root / "source")}
            run_args = refresh.namespace_from_checkpoint(state, args)
            self.assertEqual(run_args.evaluation_scope, "final")
            self.assertEqual(run_args.run_output_dir, str(root / "source"))
            self.assertEqual(run_args.checkpoint, str(args.refreshed_checkpoint))
            self.assertEqual(
                run_args.stage_b_checkpoint_sha256,
                args.stage_b_checkpoint_sha256,
            )

            del state["args"]["base_model"]
            with self.assertRaisesRegex(ValueError, "required refresh args"):
                refresh.namespace_from_checkpoint(state, args)

    def test_checkpoint_namespace_rejects_model_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.refresh_args(root)
            args.base_model = "development/other-base"
            state = {"args": self.checkpoint_args(root / "source")}
            with self.assertRaisesRegex(ValueError, "base_model disagrees"):
                refresh.namespace_from_checkpoint(state, args)

    def test_shared_runtime_loader_propagates_base_identity_drift(self) -> None:
        args = SimpleNamespace()
        checkpoint_bytes = b"checkpoint"
        with mock.patch.object(
            refresh,
            "load_trained_wrapper",
            side_effect=ValueError(
                "E3 runtime pre-routing base-model identity differs from Stage-B checkpoint"
            ),
        ) as loader:
            with self.assertRaisesRegex(
                ValueError, "pre-routing base-model identity differs"
            ):
                refresh.load_refresh_runtime(args, checkpoint_bytes)
        loader.assert_called_once_with(args, checkpoint_bytes=checkpoint_bytes)

    def test_metrics_and_text_provenance_bind_actual_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint_final.pt"
            checkpoint.write_bytes(b"refreshed-e3")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            metrics_fields = refresh.checkpoint_artifact_provenance(
                checkpoint, digest
            )
            text_fields = refresh.build_text_eval_provenance(
                checkpoint, digest, training_steps=6000, lm_trainable=True
            )

            self.assertEqual(
                metrics_fields,
                {
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": digest,
                    "checkpoint_size_bytes": checkpoint.stat().st_size,
                },
            )
            self.assertEqual(text_fields["source_checkpoint"], str(checkpoint))
            self.assertEqual(text_fields["source_checkpoint_sha256"], digest)
            self.assertEqual(
                text_fields["source_checkpoint_size_bytes"],
                checkpoint.stat().st_size,
            )
            self.assertEqual(text_fields["source_training_steps"], 6000)
            self.assertIs(text_fields["source_checkpoint_saved_before_eval"], True)
            self.assertIs(text_fields["copied_from_e2"], False)

    def test_refreshes_checkpoint_sidecar_from_checkpoint_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {"step": 1, "modality": "image", "loss": 2.0},
                {"step": 2, "modality": "text", "loss": 1.25},
            ]
            fixture = self.strict_e3_fixture(Path(directory), rows)
            verified = refresh.verify_original_e3_artifacts(
                fixture["checkpoint"],
                fixture["state"],
                fixture["metrics_path"],
                fixture["raw_path"],
            )
            stale = json.loads(fixture["sidecar_path"].read_text(encoding="utf-8"))
            stale["forged_stale_field"] = True
            fixture["sidecar_path"].write_text(json.dumps(stale), encoding="utf-8")

            sidecar = refresh.refresh_checkpoint_sidecar(
                fixture["checkpoint"],
                fixture["state"],
                fixture["metrics_path"],
                fixture["raw_path"],
                verified["checkpoint_args_path"],
                verified["run_manifest_path"],
            )

            self.assertNotIn("forged_stale_field", sidecar)
            self.assertEqual(
                sidecar["training_rows_sha256"], refresh.canonical_sha256(rows)
            )
            self.assertEqual(
                sidecar["metrics_steps_sha256"], refresh.canonical_sha256(rows)
            )
            quality = validate_e3_checkpoint_sidecar(
                sidecar,
                fixture["checkpoint"],
                fixture["metrics_path"],
                metrics_rows=len(rows),
                raw_training_path=fixture["raw_path"],
            )
            self.assertTrue(quality["passed"], quality["errors"])

    def test_refresh_rejects_replaced_prefix_training_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {"step": 1, "modality": "image", "loss": 2.0},
                {"step": 2, "modality": "text", "loss": 1.25},
            ]
            fixture = self.strict_e3_fixture(Path(directory), rows)
            replaced = [
                {"step": 1, "modality": "image", "loss": -99.0},
                rows[-1],
            ]
            fixture["raw_path"].write_text(
                "".join(json.dumps(row) + "\n" for row in replaced),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "full row sequence"):
                refresh.verify_original_e3_artifacts(
                    fixture["checkpoint"],
                    fixture["state"],
                    fixture["metrics_path"],
                    fixture["raw_path"],
                )

    def test_refresh_rejects_minimal_fake_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = [{"step": 1, "modality": "text", "loss": 1.25}]
            fixture = self.strict_e3_fixture(Path(directory), rows)
            fixture["sidecar_path"].write_text(
                json.dumps({"schema_version": 1, "passed": True}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing strict fields"):
                refresh.verify_original_e3_artifacts(
                    fixture["checkpoint"],
                    fixture["state"],
                    fixture["metrics_path"],
                    fixture["raw_path"],
                )

    def test_run_sh_requires_and_forwards_exact_artifacts(self) -> None:
        run_sh = (
            Path(__file__).resolve().parents[1] / "run.sh"
        ).read_text(encoding="utf-8")
        for requirement in (
            'CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required for e3-refresh}"',
            'CHECKPOINT_SHA256="${CHECKPOINT_SHA256:?CHECKPOINT_SHA256 is required for e3-refresh}"',
            'STAGE_B_CHECKPOINT="${STAGE_B_CHECKPOINT:?STAGE_B_CHECKPOINT is required for e3-refresh}"',
            'STAGE_B_CHECKPOINT_SHA256="${STAGE_B_CHECKPOINT_SHA256:?STAGE_B_CHECKPOINT_SHA256 is required for e3-refresh}"',
        ):
            self.assertIn(requirement, run_sh)
        for flag in (
            '--checkpoint "$CHECKPOINT"',
            '--checkpoint-sha256 "$CHECKPOINT_SHA256"',
            '--stage-b-checkpoint "$STAGE_B_CHECKPOINT"',
            '--stage-b-checkpoint-sha256 "$STAGE_B_CHECKPOINT_SHA256"',
        ):
            self.assertIn(flag, run_sh)
        self.assertNotIn("CHECKPOINT_ARG", run_sh)

    def test_direct_refresh_cli_requires_stage_b_artifacts(self) -> None:
        source = Path(refresh.__file__).read_text(encoding="utf-8")
        self.assertIn(
            'parser.add_argument("--stage-b-checkpoint", type=Path, required=True)',
            source,
        )
        self.assertIn(
            'parser.add_argument("--stage-b-checkpoint-sha256", required=True)',
            source,
        )


if __name__ == "__main__":
    unittest.main()
