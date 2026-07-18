"""Pure CPU tests for the representation-retention evaluator."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts import eval_representation_funnel as funnel
from scripts.eval_representation_funnel import (
    _resolve_media_paths,
    evaluate_similarity,
    fit_ridge_probe,
    multi_positive_ranks,
    tensor_diagnostics,
    solve_ridge,
)


class RestorationContractTests(unittest.TestCase):
    def test_stage_b_arguments_are_required(self) -> None:
        with self.assertRaises(SystemExit):
            funnel.parse_args(
                [
                    "--checkpoint",
                    "e3.pt",
                    "--protocol-manifest",
                    "protocol.json",
                    "--image-dev-manifest",
                    "image-dev.jsonl",
                    "--speech-dev-manifest",
                    "speech-dev.jsonl",
                    "--image-test-manifest",
                    "image-test.jsonl",
                    "--speech-test-manifest",
                    "speech-test.jsonl",
                    "--output-dir",
                    "out",
                ]
            )

    def test_runtime_base_identity_is_bound_to_verified_stage_b(self) -> None:
        identity = {
            "requested_base_model": "example/base",
            "config_sha256": "1" * 64,
            "loaded_tensor_identity": {
                "algorithm": "sha256_ordered_named_tensor_records_v1",
                "sha256": "2" * 64,
            },
        }
        stage_b_provenance = {
            "path": "/tmp/stage-b.pt",
            "sha256": "3" * 64,
            "size_bytes": 123,
            "policy": "development_only_stage_b_top8_to_top2_initialization",
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
        }
        model_meta = {
            "checkpoint_restoration": {
                "stage_b_checkpoint": {
                    **stage_b_provenance,
                    "declared_by_trainable_meta": True,
                    "declared_by_last_row": True,
                },
            }
        }
        with mock.patch.object(
            funnel,
            "load_stage_b_initialization_checkpoint",
            return_value=(
                {"resume_contract": {"base_model_identity": identity}},
                stage_b_provenance,
            ),
        ):
            restoration = funnel._bind_runtime_base_identity(
                model_meta, "/tmp/stage-b.pt", "3" * 64
            )

        self.assertEqual(restoration["runtime_base_model_identity"], identity)
        self.assertTrue(
            restoration["runtime_base_model_identity_verified_against_stage_b"]
        )
        self.assertEqual(
            restoration["runtime_base_model_identity_sha256"],
            funnel.hashlib.sha256(
                funnel.json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
        )


class RidgeSolveTests(unittest.TestCase):
    def test_ridge_recovers_affine_map(self) -> None:
        torch.manual_seed(11)
        features = torch.randn(24, 4)
        weight = torch.tensor(
            [[1.0, -2.0], [0.5, 0.25], [-0.75, 1.5], [2.0, 0.1]]
        )
        intercept = torch.tensor([0.3, -1.2])
        targets = features @ weight + intercept
        probe = solve_ridge(features, targets, 1e-8, split_name="development")
        self.assertTrue(torch.allclose(probe.predict(features), targets, atol=2e-5, rtol=2e-5))

    def test_test_split_cannot_be_used_for_fit(self) -> None:
        with self.assertRaisesRegex(ValueError, "development-only"):
            fit_ridge_probe(
                torch.eye(3),
                torch.eye(3),
                1e-2,
                split_name="sealed_test",
            )


class ManifestPathTests(unittest.TestCase):
    def test_supports_manifest_and_repo_relative_media_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            manifest = manifest_dir / "rows.jsonl"

            local_media = manifest_dir / "local.wav"
            local_media.write_bytes(b"local")
            rows = [{"audio_path": "local.wav"}]
            _resolve_media_paths(rows, manifest, "audio_path")
            self.assertEqual(Path(rows[0]["audio_path"]), local_media.resolve())

            repo_media = root / "repo.wav"
            repo_media.write_bytes(b"repo")
            rows = [{"audio_path": str(repo_media.relative_to(Path.cwd()))}]
            _resolve_media_paths(rows, manifest, "audio_path")
            self.assertEqual(Path(rows[0]["audio_path"]), repo_media.resolve())


class RankingTests(unittest.TestCase):
    def test_multi_positive_uses_best_positive(self) -> None:
        similarity = torch.tensor(
            [
                [0.8, 0.9, 0.1, 0.2],
                [0.7, 0.6, 0.5, 0.4],
            ]
        )
        self.assertEqual(multi_positive_ranks(similarity, [[0, 1], [2, 3]]), [0, 2])

    def test_candidate_order_invariance_including_ties(self) -> None:
        similarity = torch.tensor(
            [
                [0.4, 0.9, 0.9, 0.2],
                [0.7, 0.1, 0.3, 0.7],
            ]
        )
        positives = [[2], [0, 3]]
        expected = multi_positive_ranks(similarity, positives)
        permutation = [3, 1, 0, 2]
        inverse = {old: new for new, old in enumerate(permutation)}
        permuted_positives = [[inverse[index] for index in row] for row in positives]
        actual = multi_positive_ranks(similarity[:, permutation], permuted_positives)
        self.assertEqual(actual, expected)

    def test_bidirectional_metrics_support_multiple_captions(self) -> None:
        similarity = torch.tensor(
            [
                [0.9, 0.8, 0.1],
                [0.2, 0.1, 0.95],
            ]
        )
        result = evaluate_similarity(similarity, [[0, 1], [2]])
        self.assertEqual(result["media_to_text_ranks"], [0, 0])
        self.assertEqual(result["text_to_media_ranks"], [0, 0, 0])
        self.assertEqual(result["media_to_text"]["r_at_1"], 1.0)
        self.assertEqual(result["text_to_media"]["r_at_1"], 1.0)




class RepresentationDiagnosticsTests(unittest.TestCase):
    def test_unpooled_prefix_statistics(self) -> None:
        prefix = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        result = tensor_diagnostics(prefix)

        self.assertEqual(result["shape"], [1, 2, 2])
        self.assertEqual(result["diagnostic_vectors"]["total"], 2)
        self.assertEqual(result["prefix_norm"]["mean"], 1.0)
        self.assertEqual(result["per_dimension_variance"]["values"], [0.25, 0.25])
        self.assertAlmostEqual(result["effective_rank"], 1.0, places=6)
        self.assertEqual(result["off_diagonal_pairwise_cosine"]["pair_count"], 1)
        self.assertEqual(result["off_diagonal_pairwise_cosine"]["mean"], 0.0)

    def test_high_dimensional_rank_deficient_effective_rank_is_finite(self) -> None:
        torch.manual_seed(17)
        basis = torch.randn(3, 2048)
        coefficients = torch.randn(250, 3)
        representations = coefficients @ basis

        result = tensor_diagnostics(representations)

        self.assertTrue(math.isfinite(result["effective_rank"]))
        self.assertGreaterEqual(result["effective_rank"], 1.0)
        self.assertLessEqual(result["effective_rank"], 3.000001)

    def test_diagnostics_reject_scalar_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            tensor_diagnostics(torch.tensor([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
