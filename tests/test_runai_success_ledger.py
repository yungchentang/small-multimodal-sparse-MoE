import json
import tempfile
import types
import unittest
from pathlib import Path

from scripts import build_evaluation_result_manifest as result_manifest
from scripts import build_final_conference_report as report
from scripts import build_runai_success_ledger as ledger
from scripts import freeze_evaluation_protocol as freezer
from scripts.analyze_paired_controls import (
    production_bootstrap_r_at_1_ci,
    production_retrieval_metrics,
    validate_per_query_identity_row,
)
from scripts.protocol_v2 import (
    per_query_jsonl_content,
    per_query_jsonl_sha256,
)
from tests.sealed_metrics_fixture import (
    complete_production_metrics,
    rebind_identity,
    sealed_image_manifest_content,
    sealed_image_manifest_rows,
    sealed_metrics,
    sealed_per_query_rows_for_run,
    sealed_speech_manifest_content,
    sealed_speech_manifest_rows,
)


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def add_production_aggregates(metrics, rows, run):
    aliases = (
        "r_at_1",
        "r_at_5",
        "r_at_10",
        "strict_r_at_1",
        "mrr",
        "mean_gold_nll_margin",
        "tie_count",
        "tie_rate",
        "r_at_1_bootstrap_ci_low",
        "r_at_1_bootstrap_ci_high",
    )
    for offset, (modality, prefix) in enumerate((
        ("image", "image_to_text"),
        ("speech", "speech_to_text"),
    )):
        validated = [
            validate_per_query_identity_row(
                row["condition"],
                "ledger fixture",
                index + 1,
                row,
                require_production_fields=True,
            )
            for index, row in enumerate(rows)
            if row["modality"] == modality
        ]
        aggregates = production_retrieval_metrics(validated, prefix)
        bootstrap = production_bootstrap_r_at_1_ci(
            validated,
            int(run["bootstrap_samples"]),
            int(run["bootstrap_seed"]) + offset,
        )
        aggregates.update({f"{prefix}_{key}": value for key, value in bootstrap.items()})
        metrics.update(aggregates)
        for suffix in aliases:
            metrics[f"conditional_{prefix}_{suffix}"] = aggregates[
                f"{prefix}_{suffix}"
            ]
    metrics["image_positive_counts"] = [1]
    metrics["conditional_image_positive_counts"] = [1]


class RunAISuccessLedgerFixture:
    def __init__(self, root: Path):
        self.root = root
        self.selected_root = root / "selected"
        self.selected_root.mkdir()
        self.checkpoint = self.selected_root / "model.pt"
        self.checkpoint.write_bytes(b"selected checkpoint\n")
        self.artifact = self.selected_root / "metrics.json"
        self.artifact.write_text('{"loss":1.0}\n', encoding="utf-8")
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text("# evaluator\n", encoding="utf-8")
        self.allocator_source = Path(freezer.__file__).with_name(
            "sealed_position_allocator.py"
        )
        image_count = max(
            cell["candidate_count"]
            for cell in report.SEALED_MATRIX_CELLS.values()
        )
        self.image_manifest_rows = sealed_image_manifest_rows(image_count)
        self.image_test = self.root / "image_test.jsonl"
        self.image_test.write_text(
            sealed_image_manifest_content(self.image_manifest_rows),
            encoding="utf-8",
        )
        self.speech_manifest_rows = sealed_speech_manifest_rows(image_count)
        self.speech_test = self.root / "speech_test.jsonl"
        self.speech_test.write_text(
            sealed_speech_manifest_content(self.speech_manifest_rows),
            encoding="utf-8",
        )
        self.protocol = self._write_protocol()
        self.bindings = self._write_bindings()
        self.result_manifest = self._write_result_manifest()
        self.index = self._write_raw_index()
        self.output = root / "runai_final_success_ledger.json"

    def _write_protocol(self) -> Path:
        cells = [dict(value) for value in report.SEALED_MATRIX_CELLS.values()]
        run_config = types.SimpleNamespace(
            image_query_count=250,
            speech_query_count=250,
            query_offset=0,
            candidate_offset=0,
            tie_epsilon=1e-8,
            candidate_permutation="query_identity_seeded",
            randomize_positive_position=True,
            candidate_seed=17,
            control_seed=23,
            bootstrap_samples=8,
            bootstrap_seed=12345,
            protocol_name="sealed_evaluation_v1",
            eval_split_name="sealed_test",
            max_length=512,
            conditional_batch_size=8,
        )
        payload = {
            "schema_version": 2,
            "protocol": "sealed_evaluation_protocol",
            "inputs": {
                "selected_root": ledger.directory_fingerprint(self.selected_root),
                "image_test": report.fingerprint_path(self.image_test),
                "speech_test": report.fingerprint_path(self.speech_test),
                "evaluator_scripts": [
                    {
                        "path": str(self.evaluator.resolve()),
                        "sha256": ledger.sha256_file(self.evaluator),
                    },
                    report.fingerprint_path(self.allocator_source),
                ],
            },
            "checkpoint": {
                "selected_root": str(self.selected_root.resolve()),
                "artifact": {
                    "path": str(self.checkpoint.resolve()),
                    "type": "file",
                    "sha256": ledger.sha256_file(self.checkpoint),
                },
            },
            "runai_project": "test-project",
            "seeds": {"candidate_seed": 17, "control_seed": 23},
            "query_counts": {"image": 250, "speech": 250},
            "controls": list(freezer.REQUIRED_CONTROLS),
            "gold_position_allocator": freezer.build_allocator_manifest(
                cells,
                candidate_seed=17,
                query_counts={"image": 250, "speech": 250},
            ),
            "evaluation_matrix": cells,
            "evaluation_runs": freezer.build_evaluation_runs(cells, run_config),
        }
        payload["protocol_content_sha256"] = ledger.canonical_sha256(payload)
        return write_json(self.root / "protocol.json", payload)

    def _raw_job(self, name: str, status: str):
        evidence_dir = self.root / "raw-evidence"
        evidence_dir.mkdir(exist_ok=True)
        describe = evidence_dir / f"{name}.describe.txt"
        logs = evidence_dir / f"{name}.log"
        describe.write_text(f"Name: {name}\nStatus: {status}\n", encoding="utf-8")
        log_lines = [f"{name} {status}"]
        if status == "Succeeded":
            log_lines.append(
                result_manifest.result_manifest_log_marker(
                    ledger.sha256_file(self.result_manifest)
                )
            )
        logs.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return {
            "job": name,
            "project": "test-project",
            "status": status,
            "describe_command": ["runai", "describe", "job", name, "-p", "test-project"],
            "describe_returncode": 0,
            "describe_path": str(describe.resolve()),
            "describe_sha256": ledger.sha256_file(describe),
            "logs_command": ["runai", "logs", name, "--project", "test-project"],
            "logs_returncode": 0,
            "logs_path": str(logs.resolve()),
            "logs_sha256": ledger.sha256_file(logs),
        }

    def _write_raw_index(self) -> Path:
        jobs = [self._raw_job("shared-success", "Succeeded"), self._raw_job("failed-run", "Failed")]
        return write_json(
            self.root / "raw-index.json",
            {
                "schema_version": 1,
                "captured_at": "2026-07-10T12:00:00+00:00",
                "project": "test-project",
                "jobs": jobs,
            },
        )

    def _write_bindings(self) -> Path:
        roles = {role: "shared-success" for role in ledger.REQUIRED_FINAL_RUNAI_ROLES}
        role_artifacts = {}
        for role in sorted(
            value
            for value in ledger.REQUIRED_FINAL_RUNAI_ROLES
            if not value.startswith("sealed:")
        ):
            artifact_dir = self.root / "role-results" / role
            metrics_path = write_json(
                artifact_dir / "metrics.json", {"role": role, "score": 1.0}
            )
            per_query_path = artifact_dir / "per_query.jsonl"
            per_query_path.write_text(
                json.dumps({"query_id": f"{role}-q0", "rank": 1}, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            role_artifacts[role] = {
                "artifacts": {
                    "metrics": str(metrics_path.resolve()),
                    "per_query": str(per_query_path.resolve()),
                }
            }
        protocol = json.loads(self.protocol.read_text(encoding="utf-8"))
        protocol_sha = ledger.sha256_file(self.protocol)
        for role in sorted(
            value
            for value in ledger.REQUIRED_FINAL_RUNAI_ROLES
            if value.startswith("sealed:")
        ):
            _prefix, cell_id, control = role.split(":", 2)
            run = next(
                value
                for value in protocol["evaluation_runs"]
                if value["id"] == f"{cell_id}:{control}"
            )
            rows = sealed_per_query_rows_for_run(
                protocol,
                run,
                self.image_manifest_rows,
                self.speech_manifest_rows,
            )
            artifact_dir = self.root / "sealed" / cell_id / control
            per_query = artifact_dir / "per_query.jsonl"
            per_query.parent.mkdir(parents=True, exist_ok=True)
            metrics = sealed_metrics(
                protocol,
                run,
                rows,
                protocol_file_sha256=protocol_sha,
                checkpoint_sha256=ledger.sha256_file(self.checkpoint),
                checkpoint_path=str(self.checkpoint.resolve()),
                evaluator_path=str(self.evaluator.resolve()),
                evaluator_sha256=ledger.sha256_file(self.evaluator),
            )
            complete_production_metrics(metrics, rows, run)
            per_query.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            metrics_path = write_json(artifact_dir / "metrics.json", metrics)
            role_artifacts[role] = {
                "artifacts": {
                    "metrics": str(metrics_path.resolve()),
                    "per_query": str(per_query.resolve()),
                }
            }
        role_artifacts["selected_e3_training"]["checkpoint"] = str(self.checkpoint.resolve())
        return write_json(
            self.root / "bindings.json",
            {
                "schema_version": 2,
                "bindings_type": "runai_final_success_ledger_bindings",
                "final_roles": roles,
                "role_artifacts": role_artifacts,
                "job_result_manifests": {},
                "failure_chains": [
                    {
                        "failed_job": "failed-run",
                        "replacement_job": "shared-success",
                        "diagnosis": "transient cluster failure",
                        "fix": "resubmitted with verified configuration",
                    }
                ],
            },
        )


    def _write_result_manifest(self) -> Path:
        bindings = json.loads(self.bindings.read_text(encoding="utf-8"))
        evaluations = {}
        for role in sorted(ledger.EVALUATION_RUNAI_ROLES):
            evaluations[role] = {
                "evaluation_id": f"{role}-evaluation-v1",
                "artifacts": bindings["role_artifacts"][role]["artifacts"],
                "metrics_artifact": "metrics",
                "per_query_artifact": "per_query",
            }
        spec = write_json(
            self.root / "result-manifest-spec.json",
            {
                "schema_version": 1,
                "spec_type": "runai_evaluation_result_manifest_spec",
                "commit": "a" * 40,
                "runai": {
                    "project": "test-project",
                    "job": "shared-success",
                    "job_uid": "job-uid-1",
                    "pod": "shared-success-pod-0",
                    "pod_uid": "pod-uid-1",
                },
                "checkpoint": str(self.checkpoint.resolve()),
                "protocol": str(self.protocol.resolve()),
                "evaluations": evaluations,
            },
        )
        manifest_path = self.root / "shared-success.result-manifest.json"
        result_manifest.build_manifest(spec, manifest_path)
        bindings["job_result_manifests"] = {
            "shared-success": str(manifest_path.resolve())
        }
        write_json(self.bindings, bindings)
        return manifest_path


class RunAISuccessLedgerTest(unittest.TestCase):
    def test_role_contract_matches_report_validator(self):
        self.assertEqual(
            ledger.REQUIRED_FINAL_RUNAI_ROLES,
            frozenset(report.REQUIRED_FINAL_RUNAI_ROLES),
        )

    def build(self, fixture: RunAISuccessLedgerFixture):
        return ledger.build_ledger(
            [fixture.index],
            fixture.protocol,
            fixture.selected_root,
            fixture.checkpoint,
            fixture.bindings,
            fixture.output,
        )

    def test_successful_round_trip_is_canonical_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            built = self.build(fixture)
            on_disk = json.loads(fixture.output.read_text(encoding="utf-8"))
            self.assertEqual(on_disk, built)
            self.assertEqual(fixture.output.read_text(encoding="utf-8"), ledger.canonical_json(built) + "\n")
            self.assertEqual(built["schema_version"], 2)
            self.assertEqual(built["ledger_type"], "runai_final_success_ledger")
            self.assertEqual(built["captured_at"], "2026-07-10T12:00:00+00:00")
            self.assertEqual(set(built["final_roles"]), ledger.REQUIRED_FINAL_RUNAI_ROLES)
            self.assertEqual(set(built["role_artifacts"]), ledger.REQUIRED_FINAL_RUNAI_ROLES)
            self.assertEqual(built["role_artifacts"]["selected_e3_training"]["checkpoint"]["sha256"], ledger.sha256_file(fixture.checkpoint))
            self.assertEqual(built["jobs"][0]["job"], "failed-run")
            self.assertEqual(built["jobs"][1]["job"], "shared-success")
            self.assertEqual(built["failure_chains"][0]["replacement_job"], "shared-success")
            success_job = built["jobs"][1]
            self.assertEqual(
                success_job["result_manifest"]["sha256"],
                ledger.sha256_file(fixture.result_manifest),
            )
            self.assertEqual(
                built["role_artifacts"]["e3_text_eval"]["result_manifest_sha256"],
                success_job["result_manifest"]["sha256"],
            )

    def test_rejects_missing_final_role(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            bindings = json.loads(fixture.bindings.read_text(encoding="utf-8"))
            bindings["final_roles"].pop("e3_text_eval")
            write_json(fixture.bindings, bindings)
            with self.assertRaisesRegex(ledger.LedgerInputError, "complete final role set"):
                self.build(fixture)
            self.assertFalse(fixture.output.exists())

    def test_rejects_schema_v2_protocol_without_evaluation_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            protocol = json.loads(fixture.protocol.read_text(encoding="utf-8"))
            del protocol["evaluation_runs"]
            unhashed = dict(protocol)
            unhashed.pop("protocol_content_sha256")
            protocol["protocol_content_sha256"] = ledger.canonical_sha256(unhashed)
            write_json(fixture.protocol, protocol)
            with self.assertRaisesRegex(
                ledger.LedgerInputError, "evaluation_runs"
            ):
                self.build(fixture)
            self.assertFalse(fixture.output.exists())

    def test_rejects_rehashed_sealed_metrics_contract_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            bindings = json.loads(
                fixture.bindings.read_text(encoding="utf-8")
            )
            path = Path(
                bindings["role_artifacts"]["sealed:r5:real"][
                    "artifacts"
                ]["metrics"]
            )
            metrics = json.loads(path.read_text(encoding="utf-8"))
            metrics["evaluation_provenance"]["metric_affecting_args"][
                "max_length"
            ] = 1024
            rebind_identity(metrics)
            write_json(path, metrics)
            with self.assertRaisesRegex(
                ledger.LedgerInputError, "max_length"
            ):
                self.build(fixture)
            self.assertFalse(fixture.output.exists())

    def test_rejects_per_query_exact_byte_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            bindings = json.loads(
                fixture.bindings.read_text(encoding="utf-8")
            )
            path = Path(
                bindings["role_artifacts"]["sealed:r5:real"][
                    "artifacts"
                ]["per_query"]
            )
            path.write_text(
                " " + path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ledger.LedgerInputError, "exact file SHA256"
            ):
                self.build(fixture)
            self.assertFalse(fixture.output.exists())

    def test_rejects_rehashed_minimal_per_query_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            bindings = json.loads(fixture.bindings.read_text(encoding="utf-8"))
            artifacts = bindings["role_artifacts"]["sealed:r5:real"]["artifacts"]
            per_query_path = Path(artifacts["per_query"])
            metrics_path = Path(artifacts["metrics"])
            rows = [
                {
                    "modality": row["modality"],
                    "candidate_ids": row["candidate_ids"],
                }
                for row in ledger._read_jsonl(per_query_path, "test per_query")
            ]
            per_query_path.write_text(
                per_query_jsonl_content(rows), encoding="utf-8"
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["per_query_sha256"] = per_query_jsonl_sha256(rows)
            write_json(metrics_path, metrics)
            with self.assertRaisesRegex(
                ledger.LedgerInputError, "missing required fields"
            ):
                self.build(fixture)
            self.assertFalse(fixture.output.exists())

    def test_rejects_final_role_with_non_succeeded_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            bindings = json.loads(fixture.bindings.read_text(encoding="utf-8"))
            bindings["final_roles"]["e3_text_eval"] = "failed-run"
            write_json(fixture.bindings, bindings)
            with self.assertRaisesRegex(ledger.LedgerInputError, "must reference a Succeeded job"):
                self.build(fixture)
            self.assertFalse(fixture.output.exists())

    def test_rejects_raw_evidence_hash_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            index = json.loads(fixture.index.read_text(encoding="utf-8"))
            Path(index["jobs"][0]["describe_path"]).write_text("Status: Succeeded\nchanged\n", encoding="utf-8")
            with self.assertRaisesRegex(ledger.LedgerInputError, "SHA-256 mismatch/hash drift"):
                self.build(fixture)
            self.assertFalse(fixture.output.exists())

    def test_rejects_incomplete_failure_chain(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            bindings = json.loads(fixture.bindings.read_text(encoding="utf-8"))
            bindings["failure_chains"] = []
            write_json(fixture.bindings, bindings)
            with self.assertRaisesRegex(ledger.LedgerInputError, "every included failed job"):
                self.build(fixture)
            self.assertFalse(fixture.output.exists())

    def test_rejects_sidecar_manifest_without_captured_log_digest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            index = json.loads(fixture.index.read_text(encoding="utf-8"))
            job = next(value for value in index["jobs"] if value["status"] == "Succeeded")
            logs_path = Path(job["logs_path"])
            logs_path.write_text("shared-success Succeeded\n", encoding="utf-8")
            job["logs_sha256"] = ledger.sha256_file(logs_path)
            write_json(fixture.index, index)
            with self.assertRaisesRegex(
                ledger.LedgerInputError, "captured immutable log"
            ):
                self.build(fixture)
            self.assertFalse(fixture.output.exists())

    def test_rejects_arbitrary_role_artifact_reassignment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            bindings = json.loads(fixture.bindings.read_text(encoding="utf-8"))
            left = bindings["role_artifacts"]["e3_text_eval"]["artifacts"]
            right = bindings["role_artifacts"]["routing_specialization"]["artifacts"]
            bindings["role_artifacts"]["e3_text_eval"]["artifacts"] = right
            bindings["role_artifacts"]["routing_specialization"]["artifacts"] = left
            write_json(fixture.bindings, bindings)
            with self.assertRaisesRegex(
                ledger.LedgerInputError, "role artifact mismatch"
            ):
                self.build(fixture)
            self.assertFalse(fixture.output.exists())

    def test_rejects_noncanonical_result_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            payload = json.loads(fixture.result_manifest.read_text(encoding="utf-8"))
            fixture.result_manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ledger.LedgerInputError, "exact canonical JSON bytes"
            ):
                self.build(fixture)
            self.assertFalse(fixture.output.exists())

    def test_explicitly_rejects_legacy_bindings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            bindings = json.loads(fixture.bindings.read_text(encoding="utf-8"))
            bindings["schema_version"] = 1
            write_json(fixture.bindings, bindings)
            with self.assertRaisesRegex(
                ledger.LedgerInputError, "unsupported bindings manifest schema"
            ):
                self.build(fixture)
            self.assertFalse(fixture.output.exists())

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = RunAISuccessLedgerFixture(Path(temp_dir))
            fixture.output.write_text("existing\n", encoding="utf-8")
            with self.assertRaisesRegex(ledger.LedgerInputError, "refusing to overwrite"):
                self.build(fixture)
            self.assertEqual(fixture.output.read_text(encoding="utf-8"), "existing\n")


if __name__ == "__main__":
    unittest.main()
