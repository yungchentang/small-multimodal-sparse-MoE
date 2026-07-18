import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_public_evidence_bundle.py"


class PublicEvidenceBundleTest(unittest.TestCase):
    def run_cli(
        self, *arguments: object, env_overrides: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(env_overrides or {})
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(argument) for argument in arguments)],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def file_bytes(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_bundles_are_deterministic_and_input_order_independent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metrics = root / "metrics.json"
            metrics.write_text(
                json.dumps({"accuracy": 0.875, "owner": "first-env-user", "protocol_version": 3, "run_path": "/private/run/metrics.json"}),
                encoding="utf-8",
            )
            notes = root / "notes.md"
            notes.write_text("Final evidence.\n", encoding="utf-8")
            first = root / "bundle-first"
            second = root / "bundle-second"

            first_run = self.run_cli(
                "--input",
                f"metrics={metrics}",
                "--input",
                f"notes={notes}",
                "--output-dir",
                first,
                env_overrides={"USER": "first-env-user", "LOGNAME": "first-env-user"},
            )
            second_run = self.run_cli(
                "--input",
                f"notes={notes}",
                "--input",
                f"metrics={metrics}",
                "--output-dir",
                second,
                env_overrides={"USER": "second-env-user", "LOGNAME": "second-env-user"},
            )

            self.assertEqual(first_run.returncode, 0, first_run.stderr)
            self.assertEqual(second_run.returncode, 0, second_run.stderr)
            self.assertEqual(self.file_bytes(first), self.file_bytes(second))

    def test_private_values_are_sanitized_and_credentials_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "summary.json"
            source.write_text(
                json.dumps(
                    {
                        "accuracy": 0.75,
                        "data_path": "/mnt/lidiap-alignai/scratch/alice/datasets/eval.jsonl",
                        "email": "alice@epfl.ch",
                        "home": "/home/alice/private/results.json",
                        "namespace": "runai-lidiap-alignai-alice",
                        "owner": "alice",
                        "registry_image": "registry.rcp.epfl.ch/team/private-image:latest",
                        "repo": str(REPO_ROOT),
                        "run_path": str(REPO_ROOT / "artifacts" / "final" / "metrics.json"),
                    }
                ),
                encoding="utf-8",
            )
            bundle = root / "public"
            result = self.run_cli("--input", f"summary={source}", "--output-dir", bundle)
            self.assertEqual(result.returncode, 0, result.stderr)

            combined = b"\n".join(self.file_bytes(bundle).values()).decode("utf-8")
            for private_value in (
                "/home/alice",
                "/mnt/lidiap-alignai",
                "alice",
                "epfl.ch",
                "runai-lidiap",
                "registry.rcp",
                str(REPO_ROOT),
            ):
                self.assertNotIn(private_value, combined)
            self.assertIn("$DATA_ROOT", combined)
            self.assertIn("$REPO_ROOT", combined)
            self.assertIn("$RUN_ROOT", combined)
            self.assertIn("<cluster-project>", combined)
            self.assertIn("<registry-image>", combined)
            sanitized = json.loads((bundle / "artifacts" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(sanitized["accuracy"], 0.75)

            credential_source = root / "credential.json"
            credential_source.write_text(json.dumps({"api_key": "not-public", "score": 1.0}), encoding="utf-8")
            rejected_bundle = root / "credential-bundle"
            rejected = self.run_cli(
                "--input",
                f"metrics={credential_source}",
                "--output-dir",
                rejected_bundle,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("credential-like", rejected.stderr)
            self.assertFalse(rejected_bundle.exists())

    def test_checkpoint_is_fingerprinted_but_never_copied(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "model.pt"
            checkpoint_bytes = b"checkpoint-byte-marker\x00\x01\x02" * 11
            checkpoint.write_bytes(checkpoint_bytes)
            bundle = root / "public"

            result = self.run_cli(
                "--checkpoint",
                f"final_model={checkpoint}",
                "--output-dir",
                bundle,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifacts"], [])
            self.assertEqual(
                manifest["checkpoints"],
                [
                    {
                        "role": "final_model",
                        "source_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                        "source_size_bytes": len(checkpoint_bytes),
                    }
                ],
            )
            self.assertFalse(any(path.is_file() for path in (bundle / "artifacts").iterdir()))
            self.assertNotIn(checkpoint_bytes, self.file_bytes(bundle).values())
            self.assertNotIn(b"checkpoint-byte-marker", b"\n".join(self.file_bytes(bundle).values()))

    def test_compressed_pdf_is_fingerprinted_but_never_copied(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            secret_payload = b"checkpoint-byte-marker\napi_key=not-public\n"
            compressed_payload = zlib.compress(secret_payload)
            pdf_bytes = (
                b"%PDF-1.7\n"
                b"1 0 obj\n"
                + f"<< /Length {len(compressed_payload)} /Filter /FlateDecode >>\n".encode("ascii")
                + b"stream\n"
                + compressed_payload
                + b"\nendstream\nendobj\n%%EOF\n"
            )
            pdf = root / "report.pdf"
            pdf.write_bytes(pdf_bytes)
            bundle = root / "public"

            result = self.run_cli("--input", f"report={pdf}", "--output-dir", bundle)

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifacts"], [])
            self.assertEqual(
                manifest["pdf_fingerprints"],
                [
                    {
                        "format": "pdf",
                        "role": "report",
                        "source_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
                        "source_size_bytes": len(pdf_bytes),
                    }
                ],
            )
            bundle_files = self.file_bytes(bundle)
            self.assertFalse(any(relative_path.endswith(".pdf") for relative_path in bundle_files))
            self.assertFalse(any(path.is_file() for path in (bundle / "artifacts").iterdir()))
            combined = b"\n".join(bundle_files.values())
            self.assertNotIn(secret_payload, combined)
            self.assertNotIn(compressed_payload, combined)
            self.assertNotIn(b"checkpoint-byte-marker", combined)

    def test_checkpoint_b64_is_rejected_in_aggregate_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "metrics.json"
            source.write_text(
                json.dumps({"checkpoint_b64": "Q" * 128, "score": 1.0}),
                encoding="utf-8",
            )
            bundle = root / "public"

            result = self.run_cli("--input", f"metrics={source}", "--output-dir", bundle)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checkpoint_b64", result.stderr)
            self.assertIn("forbidden aggregate checkpoint/weights/state_dict field", result.stderr)
            self.assertFalse(bundle.exists())

    def test_per_query_nested_metadata_is_dropped_without_recursion(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "per_query.json"
            source.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-001",
                        "score": 0.75,
                        "metadata": {
                            "content": "private nested prompt marker",
                            "arbitrary": {"payload": "must not be copied"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            bundle = root / "public"

            result = self.run_cli(
                "--per-query-input",
                f"per_query={source}",
                "--output-dir",
                bundle,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            sanitized = json.loads((bundle / "artifacts" / "per_query.json").read_text(encoding="ascii"))
            self.assertEqual(sanitized, {"sample_id": "sample-001", "score": 0.75})
            combined = b"\n".join(self.file_bytes(bundle).values())
            self.assertNotIn(b"private nested prompt marker", combined)
            self.assertNotIn(b"must not be copied", combined)
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("metadata", manifest["artifacts"][0]["field_policy"]["dropped_fields"])

    def test_aggregate_content_is_rejected_without_per_query_heuristics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            singleton = root / "per_query.json"
            singleton.write_text(
                json.dumps({"prompt": "private singleton prompt", "score": 1.0}),
                encoding="utf-8",
            )
            singleton_bundle = root / "singleton-bundle"
            singleton_result = self.run_cli(
                "--input",
                f"per_query={singleton}",
                "--output-dir",
                singleton_bundle,
            )
            self.assertNotEqual(singleton_result.returncode, 0)
            self.assertIn("forbidden aggregate content field", singleton_result.stderr)
            self.assertFalse(singleton_bundle.exists())

            no_id_csv = root / "rows.csv"
            no_id_csv.write_text("prompt,score\nprivate csv prompt,1.0\n", encoding="utf-8")
            csv_bundle = root / "csv-bundle"
            csv_result = self.run_cli(
                "--input",
                f"rows={no_id_csv}",
                "--output-dir",
                csv_bundle,
            )
            self.assertNotEqual(csv_result.returncode, 0)
            self.assertIn("forbidden aggregate content field", csv_result.stderr)
            self.assertFalse(csv_bundle.exists())

    def test_active_svg_payloads_are_rejected(self):
        payloads = {
            "animation": '<svg xmlns="http://www.w3.org/2000/svg"><animate attributeName="x"/></svg>',
            "event_handler": '<svg xmlns="http://www.w3.org/2000/svg"><rect onclick="alert(1)"/></svg>',
            "external_href": '<svg xmlns="http://www.w3.org/2000/svg"><use href="https://example.com/a.svg#x"/></svg>',
            "external_url": '<svg xmlns="http://www.w3.org/2000/svg"><rect fill="url(https://example.com/a.svg#x)"/></svg>',
            "foreign_object": '<svg xmlns="http://www.w3.org/2000/svg"><foreignObject/></svg>',
            "image": '<svg xmlns="http://www.w3.org/2000/svg"><image href="#embedded"/></svg>',
            "script": '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
            "style_attribute": '<svg xmlns="http://www.w3.org/2000/svg"><rect style="fill:red"/></svg>',
            "style_element": '<svg xmlns="http://www.w3.org/2000/svg"><style>rect { fill: red; }</style></svg>',
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for case, payload in payloads.items():
                with self.subTest(case=case):
                    source = root / f"{case}.svg"
                    source.write_text(payload, encoding="utf-8")
                    bundle = root / f"{case}-bundle"

                    result = self.run_cli(
                        "--input",
                        f"figure_{case}={source}",
                        "--output-dir",
                        bundle,
                    )

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("SVG", result.stderr)
                    self.assertFalse(bundle.exists())


    def test_jsonl_strips_content_and_records_field_policy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            per_query = root / "per_query.jsonl"
            row = {
                "sample_id": "sample-001",
                "sample_sha256": "a" * 64,
                "modality": "image",
                "rank": 1,
                "prediction": "cat",
                "correct": True,
                "candidate_count": 4,
                "score": 0.9,
                "delta": 0.2,
                "protocol": "sealed-v1",
                "caption": "private caption",
                "transcript": "private transcript",
                "generated_text": "generated answer",
                "reference_text": "reference answer",
                "image_path": "/private/media/image.jpg",
                "feature_vector": [0.1, 0.2, 0.3],
            }
            per_query.write_text(json.dumps(row) + "\n", encoding="utf-8")
            bundle = root / "public"

            result = self.run_cli("--per-query-input", f"per_query={per_query}", "--output-dir", bundle)
            self.assertEqual(result.returncode, 0, result.stderr)
            sanitized_row = json.loads((bundle / "artifacts" / "per_query.jsonl").read_text(encoding="utf-8"))

            for kept_field in (
                "sample_id",
                "sample_sha256",
                "modality",
                "rank",
                "prediction",
                "correct",
                "candidate_count",
                "score",
                "delta",
                "protocol",
            ):
                self.assertIn(kept_field, sanitized_row)
            for dropped_field in (
                "caption",
                "transcript",
                "generated_text",
                "reference_text",
                "image_path",
                "feature_vector",
            ):
                self.assertNotIn(dropped_field, sanitized_row)
            self.assertEqual(sanitized_row["protocol"], "sealed-v1")

            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            policy = manifest["artifacts"][0]["field_policy"]
            self.assertEqual(manifest["artifacts"][0]["input_type"], "per_query")
            self.assertTrue(policy["applied"])
            self.assertTrue({"sample_id", "score", "protocol"}.issubset(policy["kept_fields"]))
            self.assertTrue(
                {"caption", "transcript", "generated_text", "reference_text", "image_path", "feature_vector"}.issubset(
                    policy["dropped_fields"]
                )
            )

    def test_oversize_artifact_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "evidence.md"
            source.write_text("x" * 64, encoding="utf-8")
            bundle = root / "public"
            result = self.run_cli(
                "--input",
                f"notes={source}",
                "--max-file-bytes",
                "32",
                "--output-dir",
                bundle,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exceeding --max-file-bytes=32", result.stderr)
            self.assertFalse(bundle.exists())

    def test_duplicate_roles_are_rejected_across_input_types(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "metrics.json"
            source.write_text(json.dumps({"score": 1.0}), encoding="utf-8")
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            bundle = root / "public"

            result = self.run_cli(
                "--input",
                f"final={source}",
                "--checkpoint",
                f"FINAL={checkpoint}",
                "--output-dir",
                bundle,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate role", result.stderr)
            self.assertFalse(bundle.exists())

    def test_sha256sums_covers_and_verifies_every_other_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "metrics.json"
            source.write_text(json.dumps({"accuracy": 0.5, "score_delta": -0.1}), encoding="utf-8")
            bundle = root / "public"
            result = self.run_cli("--input", f"metrics={source}", "--output-dir", bundle)
            self.assertEqual(result.returncode, 0, result.stderr)

            sums_lines = (bundle / "SHA256SUMS").read_text(encoding="ascii").splitlines()
            self.assertEqual(sums_lines, sorted(sums_lines, key=lambda line: line.split("  ", 1)[1]))
            recorded = {}
            for line in sums_lines:
                digest, relative_path = line.split("  ", 1)
                recorded[relative_path] = digest

            actual_files = {
                path.relative_to(bundle).as_posix()
                for path in bundle.rglob("*")
                if path.is_file() and path.name != "SHA256SUMS"
            }
            self.assertEqual(set(recorded), actual_files)
            for relative_path, expected_digest in recorded.items():
                self.assertEqual(hashlib.sha256((bundle / relative_path).read_bytes()).hexdigest(), expected_digest)

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "metrics.json"
            source.write_text(json.dumps({"score": 1.0}), encoding="utf-8")
            bundle = root / "public"
            bundle.mkdir()
            marker = bundle / "owner-data.txt"
            marker.write_text("must remain", encoding="utf-8")

            result = self.run_cli("--input", f"metrics={source}", "--output-dir", bundle)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8"), "must remain")
            self.assertEqual([path.name for path in bundle.iterdir()], ["owner-data.txt"])


if __name__ == "__main__":
    unittest.main()
