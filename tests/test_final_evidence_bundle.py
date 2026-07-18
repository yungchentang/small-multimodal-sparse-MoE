import json
import tempfile
import unittest
from pathlib import Path

from scripts import build_final_evidence_bundle as bundle


class FinalEvidenceBundleTest(unittest.TestCase):
    def _fixture(self, base: Path):
        selected = base / "selected"
        selected.mkdir()
        checkpoint = selected / "checkpoint.pt"
        checkpoint.write_bytes(b"selected-checkpoint\n")
        artifacts = {}
        for role in bundle.REQUIRED_ARTIFACT_ROLES:
            root = selected if role in bundle.SELECTED_ROOT_ARTIFACT_ROLES else base / "external"
            root.mkdir(exist_ok=True)
            path = root / f"{role}.json"
            path.write_text(json.dumps({"role": role}) + "\n", encoding="utf-8")
            artifacts[role] = path
        return selected, checkpoint, artifacts

    def _write_bundle(self, base: Path, selected: Path, checkpoint: Path, artifacts):
        path = base / "final_evidence_bundle.json"
        payload = bundle.build_bundle(selected, checkpoint, artifacts)
        bundle.write_bundle(path, payload)
        return path

    def test_schema_v3_bundle_binds_all_roles_root_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            selected, checkpoint, artifacts = self._fixture(base)
            path = self._write_bundle(base, selected, checkpoint, artifacts)

            validated = bundle.validate_bundle(
                path,
                expected_selected_root=selected,
                expected_selected_checkpoint=checkpoint,
                expected_artifacts=artifacts,
            )
            self.assertEqual(validated["schema_version"], 3)
            self.assertEqual(
                set(validated["_artifact_paths"]),
                set(bundle.REQUIRED_ARTIFACT_ROLES),
            )
            external_roles = {
                "e4_metrics",
                "e4_training_jsonl",
                "e4_checkpoint",
                "e4_text_eval",
                "e5_metrics",
                "e5_training_jsonl",
                "e5_checkpoint",
                "e5_text_eval",
                "e6_feasibility",
                "e6_checkpoint",
                "failure_ledger",
            }
            for role in external_roles:
                self.assertEqual(validated["artifacts"][role]["root_scope"], "external")
                self.assertFalse(
                    validated["_artifact_paths"][role].is_relative_to(selected.resolve())
                )

    def test_missing_role_and_cross_root_selected_artifact_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            selected, checkpoint, artifacts = self._fixture(base)
            missing = dict(artifacts)
            missing.pop("e6_feasibility")
            with self.assertRaisesRegex(bundle.EvidenceBundleError, "role set is incomplete"):
                bundle.build_bundle(selected, checkpoint, missing)

            escaped = dict(artifacts)
            escaped_path = base / "escaped-e3.json"
            escaped_path.write_text("{}\n", encoding="utf-8")
            escaped["e3_metrics"] = escaped_path
            with self.assertRaisesRegex(bundle.EvidenceBundleError, "outside selected root"):
                bundle.build_bundle(selected, checkpoint, escaped)

            stale_ablation = dict(artifacts)
            stale_path = selected / "stale-e4.json"
            stale_path.write_text("{}\n", encoding="utf-8")
            stale_ablation["e4_metrics"] = stale_path
            with self.assertRaisesRegex(bundle.EvidenceBundleError, "must be external"):
                bundle.build_bundle(selected, checkpoint, stale_ablation)

    def test_hash_drift_and_cross_checkpoint_binding_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            selected, checkpoint, artifacts = self._fixture(base)
            path = self._write_bundle(base, selected, checkpoint, artifacts)
            artifacts["candidate_selection"].write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(bundle.EvidenceBundleError, "SHA-256 mismatch"):
                bundle.validate_bundle(path)

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            selected, checkpoint, artifacts = self._fixture(base)
            path = self._write_bundle(base, selected, checkpoint, artifacts)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["artifacts"]["frozen_protocol"]["selected_checkpoint_sha256"] = "0" * 64
            payload.pop("bundle_content_sha256")
            payload["bundle_content_sha256"] = bundle.canonical_sha256(payload)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(bundle.EvidenceBundleError, "cross-checkpoint"):
                bundle.validate_bundle(path)


if __name__ == "__main__":
    unittest.main()
