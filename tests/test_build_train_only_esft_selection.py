from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_train_only_esft_selection.py"
SPEC = importlib.util.spec_from_file_location("build_train_only_esft_selection", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collector_fixture(
    root: Path, source_overrides: dict[str, str] | None = None
) -> tuple[Path, str, Path, Path]:
    source_overrides = source_overrides or {}
    source_repo = root / "source_repo"
    collector_source = (
        source_repo / "scripts" / "collect_development_prefix_routing.py"
    )
    collector_source.parent.mkdir(parents=True)
    collector_bytes = (
        ROOT / "scripts" / "collect_development_prefix_routing.py"
    ).read_bytes()
    collector_source.write_bytes(collector_bytes)
    subprocess.run(["git", "init", "-q"], cwd=source_repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=source_repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=source_repo,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=source_repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "Test collector source"],
        cwd=source_repo,
        check=True,
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=source_repo,
        text=True,
    ).strip()

    strict_files = {
        "image_train": {"path": str(root / "image_train.jsonl"), "sha256": "1" * 64},
        "speech_train": {"path": str(root / "speech_train.jsonl"), "sha256": "2" * 64},
        "image_dev": {"path": str(root / "image_dev.jsonl"), "sha256": "3" * 64},
        "speech_dev": {"path": str(root / "speech_dev.jsonl"), "sha256": "4" * 64},
        "image_eval": {"path": str(root / "image_eval.jsonl"), "sha256": "5" * 64},
        "speech_eval": {"path": str(root / "speech_eval.jsonl"), "sha256": "6" * 64},
    }
    strict_manifest = root / "strict_split_manifest.json"
    write_json(
        strict_manifest,
        {
            "schema_version": 3,
            "files": strict_files,
        },
    )
    strict_sha = digest(strict_manifest)
    rows = []
    for modality, scores in (
        ("image_prefix", [0.9, 0.1, 0.0, 0.0]),
        ("audio_prefix", [0.0, 0.2, 0.8, 0.0]),
    ):
        default_key = MODULE.TRAIN_SOURCE_KEYS[modality]
        source_key = source_overrides.get(modality, default_key)
        rows.append(
            {
                "split": "train",
                "modality": modality,
                "real_subset": True,
                "source_manifest_key": source_key,
                "source_manifest_sha256": strict_files[source_key]["sha256"],
                "strict_split_manifest_sha256": strict_sha,
                "modality_layer_accounting": [
                    {
                        "modality": modality,
                        "layer": 0,
                        "top_k": 2,
                        "token_count": 2,
                        "attempted_expert_counts": [2, 1, 1, 0],
                        "gate_score_sums": scores,
                    }
                ],
            }
        )
    routing = root / "train.jsonl"
    routing.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    e3 = root / "e3.pt"
    stage_b = root / "stage_b.pt"
    e3.write_bytes(b"e3")
    stage_b.write_bytes(b"stage-b")
    e3_companion = root / "e3_manifest.json"
    stage_b_companion = root / "stage_b_companion.json"
    write_json(e3_companion, {"kind": "e3"})
    write_json(stage_b_companion, {"kind": "stage-b"})
    run = {
        "source_commit_sha": source_commit,
        "runai_job_name": "routing-job",
        "runai_project": "project",
    }
    collector_manifest = root / "collector_manifest.json"
    write_json(
        collector_manifest,
        {
            "artifact_type": "development_real_prefix_routing_collection",
            "development_only": True,
            "real_subset": True,
            "sealed": False,
            "synthetic": False,
            "collection_split": "train",
            "collection_splits": ["train"],
            "dev_files_read": False,
            "eval_files_read": False,
            "strict_split_manifest": {
                "path": str(strict_manifest.resolve()),
                "sha256": strict_sha,
                "schema_version": 3,
                "collection_split": "train",
                "requested_files": ["image_train", "speech_train"],
                "unread_files": [
                    "image_dev",
                    "image_eval",
                    "speech_dev",
                    "speech_eval",
                ],
            },
            "source_manifests": {
                "image_train": {
                    "sha256": "1" * 64,
                    "strict_split_record": strict_files["image_train"],
                },
                "speech_train": {
                    "sha256": "2" * 64,
                    "strict_split_record": strict_files["speech_train"],
                },
            },
            "outputs": {
                "train": {"path": str(routing.resolve()), "sha256": digest(routing)}
            },
            "code": {
                "source_commit_sha": source_commit,
                "collector_path": str(collector_source.resolve()),
                "collector_sha256": hashlib.sha256(collector_bytes).hexdigest(),
            },
            "checkpoint": {
                "path": str(e3.resolve()),
                "sha256": digest(e3),
                "run_provenance": run,
                "companion_manifest": {
                    "path": str(e3_companion.resolve()),
                    "sha256": digest(e3_companion),
                    "source_commit_sha": source_commit,
                    "runai_job_name": "routing-job",
                    "runai_project": "project",
                    "development_split_manifest_sha256": strict_sha,
                },
            },
            "stage_b_checkpoint": {
                "path": str(stage_b.resolve()),
                "sha256": digest(stage_b),
                "companion_manifest": {
                    "path": str(stage_b_companion.resolve()),
                    "sha256": digest(stage_b_companion),
                    "source_commit_sha": source_commit,
                    "runai_job_name": "stage-b-job",
                    "runai_project": "project",
                },
            },
            "model_state_restoration": {
                "restoration_order": ["stage_b_student", "e3_adapter"]
            },
        },
    )
    return collector_manifest, digest(collector_manifest), routing, source_repo


class TrainOnlyEsftSelectionTests(unittest.TestCase):
    def test_cli_requires_exact_collector_manifest_not_bare_routing(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("--collector-manifest", source)
        self.assertIn("--expected-collector-manifest-sha256", source)
        self.assertNotIn('add_argument("--routing-source"', source)

    def test_git_provenance_records_head_and_cleanliness(self) -> None:
        provenance = MODULE.git_provenance(ROOT)
        self.assertEqual(len(provenance["head"]), 40)
        self.assertIsInstance(provenance["working_tree_dirty"], bool)

    def test_builds_deterministic_train_only_selection(self) -> None:
        rows = []
        for modality, scores in (
            ("image_prefix", [0.9, 0.1, 0.0, 0.0]),
            ("audio_prefix", [0.0, 0.2, 0.8, 0.0]),
        ):
            rows.append(
                {
                    "split": "train",
                    "modality": modality,
                    "layer": 0,
                    "top_k": 2,
                    "token_count": 2,
                    "attempted_expert_counts": [2, 1, 1, 0],
                    "gate_score_sums": scores,
                    "real_subset": True,
                }
            )
        selection = MODULE.build_selection(rows, 2)
        self.assertEqual(
            selection["selection_scope"],
            "development_train_image_audio_prefix_only",
        )
        layer = selection["methods"]["ESFT-Gate"]["0"]
        self.assertEqual(layer["splits"], ["train"])
        self.assertEqual(layer["selected_expert_ids"], [0, 2])
        self.assertTrue(selection["routing_accounting"]["conservation_ok"])

    def test_reader_rejects_bare_rows_and_nonconserving_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routing.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bare routing rows"):
                MODULE.read_train_routing(path)
        bad = [
            {
                "split": "train",
                "real_subset": True,
                "modality": "image_prefix",
                "layer": 0,
                "top_k": 2,
                "token_count": 2,
                "attempted_expert_counts": [1, 0, 0, 0],
                "gate_score_sums": [1.0, 0.0, 0.0, 0.0],
            }
        ]
        with self.assertRaisesRegex(ValueError, "tokens x K"):
            MODULE.build_selection(bad, 2)

    def test_verified_collector_manifest_produces_both_esft_methods(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, manifest_sha, _routing, source_repo = collector_fixture(
                Path(directory)
            )
            rows, provenance = MODULE.load_verified_collector_routing(
                manifest, manifest_sha, source_repo
            )
            selection = MODULE.build_selection(rows, 2)
        self.assertEqual(provenance["splits"], ["train"])
        self.assertFalse(provenance["dev_files_read"])
        self.assertFalse(provenance["eval_files_read"])
        self.assertEqual(len(provenance["source_files"]), 7)
        self.assertTrue(
            all(len(record["sha256"]) == 64 for record in provenance["source_files"])
        )
        self.assertEqual(
            set(selection["methods"]), {"ESFT-Gate", "ESFT-Token"}
        )
        self.assertEqual(
            selection["methods"]["ESFT-Gate"]["0"]["splits"], ["train"]
        )
        self.assertEqual(
            selection["methods"]["ESFT-Token"]["0"]["splits"], ["train"]
        )

    def test_relabels_from_dev_or_eval_are_rejected(self) -> None:
        for modality, source_key in (
            ("image_prefix", "image_dev"),
            ("audio_prefix", "speech_eval"),
        ):
            with self.subTest(modality=modality), tempfile.TemporaryDirectory() as directory:
                manifest, manifest_sha, _routing, source_repo = collector_fixture(
                    Path(directory), {modality: source_key}
                )
                with self.assertRaisesRegex(
                    ValueError, "source manifest is not trusted train data"
                ):
                    MODULE.load_verified_collector_routing(
                        manifest, manifest_sha, source_repo
                    )

    def test_routing_tamper_is_rejected_by_collector_manifest_sha_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, manifest_sha, routing, source_repo = collector_fixture(
                Path(directory)
            )
            routing.write_text(routing.read_text(encoding="utf-8") + "{}\n")
            with self.assertRaisesRegex(ValueError, "train routing SHA-256 mismatch"):
                MODULE.load_verified_collector_routing(
                    manifest, manifest_sha, source_repo
                )


if __name__ == "__main__":
    unittest.main()
