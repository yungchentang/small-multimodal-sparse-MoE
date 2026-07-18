import hashlib
import json
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch
from torch import nn

from training.olmoe_real_subset_runs import (
    FeatureCache,
    audio_feature_extraction_implementation_identity,
    lm_embeddings_are_tied,
)

from scripts.eval_conditional_retrieval import (
    apply_checkpoint_architecture_args,
    canonical_sha256,
    METRIC_AFFECTING_ARG_FIELDS,
    apply_prefix_control,
    assert_output_paths_available,
    atomic_write_text_exclusive,
    balanced_positive_position,
    bind_per_query_evaluation_provenance,
    candidate_set_hash,
    checkpoint_protocol_digests,
    deterministic_derangement_index,
    enforce_gold_position_assignment,
    extract_e3_source_checkpoint_hashes,
    lexical_hard_negative_indices,
    load_checkpoint_bound_gamma,
    load_verified_frozen_protocol,
    load_explicit_eval_rows,
    load_run_environment_provenance,
    load_verified_stage_b_for_e3,
    local_candidate_indices,
    main,
    multi_positive_ranks,
    path_is_within,
    prepare_feature_cache_root,
    permute_candidates_for_query,
    rank_gold,
    resolve_git_head,
    restore_evaluation_checkpoint_state,
    select_frozen_evaluation_run,
    sha256_file,
    tie_aware_metrics,
    tie_aware_nll_evidence,
    validate_complete_evaluation_identity,
    validate_evaluation_scope,
    validate_e3_stage_b_provenance,
    validate_e3_training_checkpoint_state,
    validate_eval_top_k,
    verify_file_sha256,
)
from scripts.sealed_position_allocator import (
    build_assignment_plan,
    validate_executed_positions,
)


def complete_evaluation_identity(**overrides):
    identity = {
        "source_commit_sha": "1" * 40,
        "runai_job_name": "fixture-job",
        "runai_project": "fixture-project",
        "evaluation_scope": "diagnostic",
        "eval_split_name": "eval_tail",
        "strict_control": False,
        "condition": "real",
        "prefix_control": "real",
        "eval_path": "shared_prefix",
        "control_seed": 42,
        "candidate_seed": 314159,
        "negative_mode": "stride",
        "requested_query_count": 2,
        "image_query_count": 2,
        "speech_query_count": 2,
        "requested_candidate_count": 5,
        "image_candidate_count": 5,
        "speech_candidate_count": 5,
        "conditional_negatives": 4,
        "conditional_candidates": 250,
        "query_offset": 0,
        "candidate_offset": -1,
        "tie_epsilon": 1e-8,
        "bootstrap_seed": 12345,
        "bootstrap_samples": 1000,
        "image_eval_samples": 250,
        "speech_eval_samples": 250,
        "checkpoint_architecture": {"base_model": "fixture/base"},
        "metric_affecting_args": {
            field: None for field in METRIC_AFFECTING_ARG_FIELDS
        },
        "protocol_name": "conditional_matching_v2",
        "protocol_manifest_path": None,
        "protocol_manifest_sha256": None,
        "protocol_content_sha256": None,
        "frozen_input_hashes": {},
        "image_manifest_path": None,
        "image_manifest_sha256": None,
        "speech_manifest_path": None,
        "speech_manifest_sha256": None,
        "gamma_path": "/tmp/gamma.json",
        "gamma_sha256": "2" * 64,
        "evaluator_path": "/tmp/eval_conditional_retrieval.py",
        "evaluator_sha256": "3" * 64,
        "image_cache_identity_sha256": "5" * 64,
        "audio_cache_identity_sha256": "6" * 64,
        "image_cache_payload_set_sha256": "7" * 64,
        "audio_cache_payload_set_sha256": "8" * 64,
        "image_produced_features_sha256": "9" * 64,
        "audio_produced_features_sha256": "a" * 64,
        "feature_cache_policy": {
            "mode": "verified_read_write",
            "preexisting_root_allowed": True,
            "cache_reads_allowed": True,
            "writes": "atomic_replace_or_create",
            "feature_source": "verified_media_paths",
        },
        "frozen_evaluation_run_id": None,
        "frozen_evaluation_cell_id": None,
        "frozen_evaluation_control": None,
        "e3_checkpoint_path": "/tmp/checkpoint.pt",
        "source_run_manifest_sha256": None,
        "e3_checkpoint_sha256": "4" * 64,
        "stage_b_checkpoint_sha256": None,
        "source_checkpoint_hashes": {"stage_b": None},
        "restoration_order": ["base_model", "e3_training_checkpoint"],
    }
    identity.update(overrides)
    return identity


def frozen_file_record(path: Path):
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "type": "file",
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def frozen_directory_record(path: Path):
    resolved = path.resolve(strict=True)
    files = [
        {
            "relative_path": child.relative_to(resolved).as_posix(),
            "sha256": sha256_file(child),
            "bytes": child.stat().st_size,
        }
        for child in sorted(
            (item for item in resolved.rglob("*") if item.is_file()),
            key=lambda item: item.as_posix(),
        )
    ]
    return {
        "path": str(resolved),
        "type": "directory",
        "sha256": canonical_sha256(files),
        "file_count": len(files),
        "files": files,
    }


class EvaluationMetricTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": idx, "caption": f"caption {idx}", "source": "fixture"}
            for idx in range(20)
        ]

    def test_eval_top_k_diagnostic_and_sealed_boundary(self):
        self.assertEqual(validate_eval_top_k(2, False), 2)
        self.assertEqual(validate_eval_top_k(4, False), 4)
        self.assertEqual(validate_eval_top_k(8, False), 8)
        with self.assertRaisesRegex(ValueError, "one of 2, 4, or 8"):
            validate_eval_top_k(3, False)
        with self.assertRaisesRegex(ValueError, "frozen to final Top-2"):
            validate_eval_top_k(8, True)

    def test_checkpoint_architecture_restores_new_bridge_fields(self):
        args = types.SimpleNamespace(capacity_factor=8.0, aux_coef=0.02)
        state = {
            "args": {
                "capacity_factor": 8.0,
                "aux_coef": 0.02,
                "image_bridge_type": "linear_projector",
                "audio_bridge_type": "attention_pool",
                "bridge_num_heads": 4,
                "image_prefix_tokens": 50,
                "audio_prefix_tokens": 64,
                "encoder_feature_tokens": 100,
            }
        }

        resolved = apply_checkpoint_architecture_args(args, state)

        self.assertEqual(args.image_bridge_type, "linear_projector")
        self.assertEqual(args.audio_bridge_type, "attention_pool")
        self.assertEqual(args.audio_prefix_tokens, 64)
        self.assertEqual(resolved["encoder_feature_tokens"], 100)
        args.capacity_factor = 6.0
        with self.assertRaisesRegex(ValueError, "capacity_factor"):
            apply_checkpoint_architecture_args(args, state)

    def test_toy_identity_retrieval_is_perfect(self):
        similarity = torch.eye(6)
        ranks = multi_positive_ranks(similarity, [[idx] for idx in range(6)])
        self.assertEqual(ranks, [0] * 6)

    def test_multi_positive_retrieval_uses_best_positive(self):
        similarity = torch.tensor([
            [0.1, 0.9, 0.8],
            [0.7, 0.2, 0.1],
        ])
        ranks = multi_positive_ranks(similarity, [[0, 1], [1, 2]])
        self.assertEqual(ranks, [0, 1])

    def test_candidate_order_invariance(self):
        scores = [0.1, 0.7, 0.3, 0.2]
        gold = 2
        expected = rank_gold(scores, gold)
        permutation = [2, 0, 3, 1]
        permuted_scores = [scores[idx] for idx in permutation]
        permuted_gold = permutation.index(gold)
        self.assertEqual(rank_gold(permuted_scores, permuted_gold), expected)

    def test_query_seeded_permutation_is_reproducible_and_moves_gold(self):
        positions = []
        for query_index in range(20):
            first = permute_candidates_for_query(
                [0, 1, 2, 3, 4], 0, 42, f"image:query-{query_index}"
            )
            repeated = permute_candidates_for_query(
                [0, 1, 2, 3, 4], 0, 42, f"image:query-{query_index}"
            )
            self.assertEqual(first, repeated)
            positions.append(first[2])
        self.assertGreater(len(set(positions)), 1)
        self.assertTrue(any(position != 0 for position in positions))

    def test_tie_aware_evidence_is_permutation_invariant(self):
        candidate_indices = [10, 11, 12, 13]
        nll_scores = [0.2, 0.4, 0.8, 0.1]
        expected = tie_aware_nll_evidence(nll_scores, gold_index=0, tie_epsilon=1e-8)
        permuted, permutation, gold_index, _ = permute_candidates_for_query(
            candidate_indices, 10, 42, "image:stable-query"
        )
        self.assertEqual(permuted, [candidate_indices[index] for index in permutation])
        observed = tie_aware_nll_evidence(
            [nll_scores[index] for index in permutation],
            gold_index=gold_index,
            tie_epsilon=1e-8,
        )
        for key in ("strict_rank", "strict_r_at_1", "reciprocal_rank", "gold_nll_margin", "gold_tie_count"):
            self.assertEqual(observed[key], expected[key])

    def test_all_tie_fails_strict_r_at_1(self):
        evidence = tie_aware_nll_evidence([1.0, 1.0, 1.0], 0, tie_epsilon=1e-8)
        metrics = tie_aware_metrics([evidence["strict_rank"]], [evidence], "fixture")
        self.assertEqual(evidence["strict_rank"], 2)
        self.assertEqual(rank_gold([1.0, 1.0, 1.0], 0, tie_epsilon=1e-8), 2)
        self.assertEqual(metrics["fixture_strict_r_at_1"], 0.0)
        self.assertAlmostEqual(metrics["fixture_mrr"], 1.0 / 3.0)
        self.assertEqual(metrics["fixture_tie_count"], 2)
        self.assertEqual(metrics["fixture_tie_rate"], 1.0)

    def test_positive_position_is_randomized_and_reproducible(self):
        positions = []
        for idx in range(20):
            positive_position = balanced_positive_position(idx, 5, seed=314159)
            first, gold = local_candidate_indices(
                self.rows,
                idx,
                negatives=4,
                mode="stride",
                candidate_seed=5000 + idx,
                randomize_positive_position=True,
                positive_position=positive_position,
            )
            second, gold_again = local_candidate_indices(
                self.rows,
                idx,
                negatives=4,
                mode="stride",
                candidate_seed=5000 + idx,
                randomize_positive_position=True,
                positive_position=positive_position,
            )
            self.assertEqual(first, second)
            self.assertEqual(gold, gold_again)
            positions.append(gold)
        counts = [positions.count(position) for position in range(5)]
        self.assertEqual(max(counts) - min(counts), 0)
        self.assertEqual(len(set(positions)), 5)

    def test_frozen_plan_overrides_unbalanced_query_hash_permutations(self):
        plan = build_assignment_plan(
            candidate_seed=314159,
            cell_id="r5",
            modality="image",
            candidate_count=5,
            query_count=20,
        )
        observed = []
        for query_index, assigned_position in enumerate(plan["positions"]):
            candidates = [
                query_index,
                (query_index + 1) % 20,
                (query_index + 2) % 20,
                (query_index + 3) % 20,
                (query_index + 4) % 20,
            ]
            permuted, permutation, _, _ = permute_candidates_for_query(
                candidates,
                query_index,
                42,
                f"image:{query_index}",
            )
            assigned, applied_permutation, gold_position = (
                enforce_gold_position_assignment(
                    permuted,
                    permutation,
                    query_index,
                    assigned_position,
                )
            )
            self.assertEqual(assigned[gold_position], query_index)
            self.assertEqual(
                assigned,
                [candidates[index] for index in applied_permutation],
            )
            observed.append(gold_position)
        validate_executed_positions(observed, plan)

    def test_random_candidates_follow_frozen_seed(self):
        first, _ = local_candidate_indices(
            self.rows, 0, negatives=6, mode="random", candidate_seed=17
        )
        repeated, _ = local_candidate_indices(
            self.rows, 0, negatives=6, mode="random", candidate_seed=17
        )
        different, _ = local_candidate_indices(
            self.rows, 0, negatives=6, mode="random", candidate_seed=29
        )
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)

    def test_lexical_hard_negatives_are_deterministic(self):
        texts = [
            "red bird fence",
            "red bird rail",
            "blue ocean",
            "bird fence red",
            "speech unrelated",
        ]
        first = lexical_hard_negative_indices(texts, negatives=2)
        repeated = lexical_hard_negative_indices(texts, negatives=2)

        self.assertEqual(first, repeated)
        self.assertEqual(first[0], [3, 1])
        self.assertTrue(
            all(index not in negatives for index, negatives in enumerate(first))
        )

    def test_sealed_cache_path_must_be_outside_selected_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "selected"
            root.mkdir()
            self.assertTrue(path_is_within(root / "feature_cache", root))
            self.assertFalse(path_is_within(Path(directory) / "sealed_cache", root))

    def test_final_cache_root_rejects_preseeded_payload_tree(self):
        with TemporaryDirectory() as directory:
            run_root = Path(directory) / "new-evaluation"
            run_root.mkdir()
            cache_root = run_root / "feature_cache"
            cache_root.mkdir()
            (cache_root / "preseed.pt").write_bytes(b"attacker-controlled")
            with self.assertRaisesRegex(FileExistsError, "must not preexist"):
                prepare_feature_cache_root(
                    cache_root,
                    "final",
                    [run_root / "metrics.json", run_root / "per-query.jsonl"],
                )

    def test_write_only_cache_never_reads_self_consistent_preseed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "feature_cache"
            cache_root.mkdir()
            image_path = root / "image.png"
            image_path.write_bytes(b"verified-image")
            config = types.SimpleNamespace(
                _name_or_path="fixture/vision",
                _commit_hash="revision-1",
                revision="main",
            )
            config.to_dict = lambda: {"hidden_size": 3}
            vision_model = types.SimpleNamespace(
                name_or_path="fixture/vision", config=config
            )
            image_processor = types.SimpleNamespace()
            image_processor.to_dict = lambda: {"size": 224}
            cache = FeatureCache(
                cache_root,
                image_cache_context={
                    "checkpoint_sha256": "a" * 64,
                    "evaluator_sha256": "b" * 64,
                },
                access_policy="exclusive_write_only",
            )
            record = {
                "id": "image-0",
                "source": "fixture",
                "image_path": str(image_path),
            }
            metadata = cache._image_cache_metadata(
                image_processor, vision_model, record, 2
            )
            payload_path = cache._image_cache_path(metadata)
            assert payload_path is not None
            payload_path.parent.mkdir(parents=True)
            seeded = torch.full((1, 2, 3), 99.0)
            from training.olmoe_real_subset_runs import tensor_content_digest
            torch.save(
                {
                    "metadata": metadata,
                    "features": seeded,
                    "tensor_digest": tensor_content_digest(seeded),
                },
                payload_path,
            )
            with mock.patch(
                "training.olmoe_real_subset_runs.image_features_from_paths",
                return_value=torch.ones(1, 2, 3),
            ) as recompute:
                with self.assertRaisesRegex(
                    FileExistsError, "refusing to reuse existing"
                ):
                    cache.image_batch(
                        image_processor,
                        vision_model,
                        [record],
                        torch.device("cpu"),
                        2,
                    )
            recompute.assert_called_once()

    def test_explicit_manifests_resolve_repo_relative_media_outside_repo_cwd(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            data_dir = repo / "data" / "real_subset_clean_260708b"
            image_path = data_dir / "images" / "foo.png"
            audio_path = data_dir / "audio" / "foo.wav"
            image_path.parent.mkdir(parents=True)
            audio_path.parent.mkdir(parents=True)
            image_path.touch()
            audio_path.touch()

            image_manifest = data_dir / "image_test.jsonl"
            speech_manifest = data_dir / "speech_test.jsonl"
            image_manifest.write_text(
                json.dumps({
                    "image_path": "data/real_subset_clean_260708b/images/foo.png",
                    "caption": "fixture",
                }) + "\n",
                encoding="utf-8",
            )
            speech_manifest.write_text(
                json.dumps({
                    "audio_path": "data/real_subset_clean_260708b/audio/foo.wav",
                    "transcript": "fixture",
                }) + "\n",
                encoding="utf-8",
            )

            away = root / "detached-cwd"
            away.mkdir()
            original_cwd = Path.cwd()
            try:
                os.chdir(away)
                image_rows = load_explicit_eval_rows(
                    image_manifest.resolve(), "image", data_dir.resolve(), limit=0
                )
                speech_rows = load_explicit_eval_rows(
                    speech_manifest.resolve(), "speech", data_dir.resolve(), limit=0
                )
            finally:
                os.chdir(original_cwd)

            self.assertEqual(Path(image_rows[0]["image_path"]), image_path.resolve())
            self.assertEqual(Path(speech_rows[0]["audio_path"]), audio_path.resolve())

    def test_candidate_hash_depends_on_order(self):
        self.assertNotEqual(
            candidate_set_hash(["a", "b", "c"]),
            candidate_set_hash(["b", "a", "c"]),
        )

    def test_derangement_has_no_fixed_points(self):
        for index in range(11):
            self.assertNotEqual(
                deterministic_derangement_index(index, 11, seed=17),
                index,
            )

    def test_random_prefix_is_norm_matched(self):
        feature = torch.randn(2, 3, 5)
        random = apply_prefix_control(feature, "random", seed=9)
        self.assertTrue(torch.allclose(
            feature.float().norm(dim=-1),
            random.float().norm(dim=-1),
            rtol=1e-5,
            atol=1e-5,
        ))

    def test_gamma_requires_checkpoint_digest_for_claim_scopes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            recorded = root / "recorded"
            gamma_path = recorded / "calibration" / "gamma.json"
            gamma_path.parent.mkdir(parents=True)
            gamma_path.write_text(
                json.dumps({"gamma": [1.0, 2.0]}), encoding="utf-8"
            )
            state = {
                "args": {"output_dir": str(recorded)},
                "gamma_provenance": {
                    "path": str(gamma_path.resolve()),
                    "relative_path": "calibration/gamma.json",
                    "output_dir": str(recorded.resolve()),
                    "sha256": sha256_file(gamma_path),
                    "size_bytes": gamma_path.stat().st_size,
                },
            }
            gamma, provenance = load_checkpoint_bound_gamma(
                recorded, state, "development"
            )
            self.assertEqual(gamma, [1.0, 2.0])
            self.assertTrue(provenance["checkpoint_bound"])

            gamma_path.write_text(
                json.dumps({"gamma": [9.0, 9.0]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "disagrees with E3 checkpoint"
            ):
                load_checkpoint_bound_gamma(
                    recorded, state, "development"
                )

            legacy = {"args": {"output_dir": str(recorded)}}
            load_checkpoint_bound_gamma(recorded, legacy, "diagnostic")
            with self.assertRaisesRegex(
                ValueError, "checkpoint-stored gamma provenance"
            ):
                load_checkpoint_bound_gamma(
                    recorded, legacy, "development"
                )

    def test_gamma_must_match_checkpoint_output_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            supplied = root / "supplied"
            recorded = root / "recorded"
            for output_dir in (supplied, recorded):
                gamma_dir = output_dir / "calibration"
                gamma_dir.mkdir(parents=True)
                (gamma_dir / "gamma.json").write_text(
                    json.dumps({"gamma": [1.0, 2.0]}), encoding="utf-8"
                )

            with self.assertRaisesRegex(ValueError, "does not match"):
                load_checkpoint_bound_gamma(
                    supplied,
                    {"args": {"output_dir": str(recorded)}},
                    "diagnostic",
                )

    def test_evaluation_scope_is_explicit_and_not_name_inferred(self):
        self.assertFalse(validate_evaluation_scope(
            "diagnostic", "", "", "", "", ""
        ))
        with self.assertRaisesRegex(
            ValueError, "explicit image and speech manifests"
        ):
            validate_evaluation_scope(
                "development", "", "", "", "", ""
            )
        self.assertTrue(validate_evaluation_scope(
            "development",
            "/tmp/image.jsonl",
            "/tmp/speech.jsonl",
            "",
            "/tmp/per_query.jsonl",
            "/tmp/cache",
        ))
        with self.assertRaisesRegex(
            ValueError, "final evaluation requires explicit artifacts"
        ):
            validate_evaluation_scope(
                "final",
                "/tmp/image.jsonl",
                "/tmp/speech.jsonl",
                "",
                "",
                "",
            )
        self.assertTrue(validate_evaluation_scope(
            "final",
            "/tmp/image.jsonl",
            "/tmp/speech.jsonl",
            "/tmp/protocol.json",
            "/tmp/per_query.jsonl",
            "/tmp/cache",
        ))

    def test_frozen_protocol_rejects_content_and_input_tamper(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_manifest = root / "image.jsonl"
            speech_manifest = root / "speech.jsonl"
            sealed_manifest = root / "sealed.json"
            paired_script = root / "paired.py"
            selected_root = root / "selected"
            selected_root.mkdir()
            selected_file = selected_root / "checkpoint.pt"
            checkpoint_state = {
                "args": {"top_k": 2},
                "run_provenance": {"source_commit_sha": "a" * 40},
                "gamma_provenance": {"sha256": "b" * 64},
                "trainable_meta": {"trainable_params": 1},
                "last_row": {"step": 1},
            }
            torch.save(checkpoint_state, selected_file)
            checkpoint_digests = checkpoint_protocol_digests(
                checkpoint_state
            )
            checkpoint_artifact = {
                **frozen_file_record(selected_file),
                **checkpoint_digests,
            }
            protocol_path = root / "protocol.json"
            image_manifest.write_text('{"id": "image"}\n', encoding="utf-8")
            speech_manifest.write_text('{"id": "speech"}\n', encoding="utf-8")
            sealed_manifest.write_text('{"sealed": true}\n', encoding="utf-8")
            paired_script.write_text("pass\n", encoding="utf-8")
            evaluator_path = Path(__file__).resolve()
            allocator_path = (
                Path(__file__).resolve().parents[1]
                / "scripts/sealed_position_allocator.py"
            )
            protocol = {
                "schema_version": 2,
                "protocol": "sealed_evaluation_protocol",
                "label": "封存",
                "inputs": {
                    "selected_root": frozen_directory_record(selected_root),
                    "sealed_manifest": frozen_file_record(sealed_manifest),
                    "image_test": frozen_file_record(image_manifest),
                    "speech_test": frozen_file_record(speech_manifest),
                    "evaluator_scripts": [
                        frozen_file_record(evaluator_path),
                        frozen_file_record(allocator_path),
                    ],
                    "checkpoint_args": {
                        "type": "inline-json",
                        "sha256": canonical_sha256({"top_k": 2}),
                    },
                    "paired_analysis_scripts": [
                        frozen_file_record(paired_script)
                    ],
                },
                "checkpoint": {
                    "selected_root": str(selected_root.resolve()),
                    "artifact": checkpoint_artifact,
                    "args": {"top_k": 2},
                    **checkpoint_digests,
                },
            }
            protocol["protocol_content_sha256"] = hashlib.sha256(
                json.dumps(
                    protocol,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest()
            protocol_path.write_text(
                json.dumps(protocol, sort_keys=True), encoding="utf-8"
            )
            verified, _, inputs = load_verified_frozen_protocol(
                protocol_path,
                image_manifest,
                speech_manifest,
                evaluator_path,
                selected_file,
            )
            self.assertEqual(
                verified["protocol_content_sha256"],
                protocol["protocol_content_sha256"],
            )
            self.assertEqual(
                set(inputs),
                {
                    "selected_root",
                    "checkpoint_args",
                    "sealed_manifest",
                    "image_test",
                    "speech_test",
                    "evaluator_scripts[0]",
                    "evaluator_scripts[1]",
                    "paired_analysis_scripts[0]",
                    "e3_checkpoint",
                },
            )

            for name in ("removed", "tampered"):
                with self.subTest(required_source=name):
                    mutated = json.loads(json.dumps(protocol))
                    allocator_record = mutated["inputs"]["evaluator_scripts"][-1]
                    if name == "removed":
                        mutated["inputs"]["evaluator_scripts"].pop()
                        message = "exactly one resolved"
                    else:
                        allocator_record["sha256"] = "0" * 64
                        message = "current SHA256"
                    unhashed = dict(mutated)
                    unhashed.pop("protocol_content_sha256")
                    mutated["protocol_content_sha256"] = canonical_sha256(
                        unhashed
                    )
                    protocol_path.write_text(
                        json.dumps(mutated, sort_keys=True),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        load_verified_frozen_protocol(
                            protocol_path,
                            image_manifest,
                            speech_manifest,
                            evaluator_path,
                            selected_file,
                        )
            protocol_path.write_text(
                json.dumps(protocol, sort_keys=True), encoding="utf-8"
            )

            alternate_checkpoint = root / "alternate.pt"
            torch.save(checkpoint_state, alternate_checkpoint)
            with self.assertRaisesRegex(
                ValueError, "--checkpoint path disagrees"
            ):
                load_verified_frozen_protocol(
                    protocol_path,
                    image_manifest,
                    speech_manifest,
                    evaluator_path,
                    alternate_checkpoint,
                )

            tampered_protocol = dict(protocol)
            tampered_protocol["controls"] = ["tampered"]
            protocol_path.write_text(
                json.dumps(tampered_protocol, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "protocol content SHA256 mismatch"
            ):
                load_verified_frozen_protocol(
                    protocol_path,
                    image_manifest,
                    speech_manifest,
                    evaluator_path,
                    selected_file,
                )

            protocol_path.write_text(
                json.dumps(protocol, sort_keys=True), encoding="utf-8"
            )
            image_manifest.write_text(
                '{"id": "replaced"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "image_test SHA256 mismatch"
            ):
                load_verified_frozen_protocol(
                    protocol_path,
                    image_manifest,
                    speech_manifest,
                    evaluator_path,
                    selected_file,
                )

            image_manifest.write_text(
                '{"id": "image"}\n', encoding="utf-8"
            )
            selected_file.write_bytes(b"replaced-checkpoint")
            with self.assertRaisesRegex(
                ValueError, "selected_root directory SHA256 mismatch"
            ):
                load_verified_frozen_protocol(
                    protocol_path,
                    image_manifest,
                    speech_manifest,
                    evaluator_path,
                    selected_file,
                )

    def test_explicit_manifest_never_resolves_media_from_cwd(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_root = root / "manifests"
            data_root = root / "data"
            cwd_root = root / "cwd"
            manifest_root.mkdir()
            data_root.mkdir()
            cwd_root.mkdir()
            (cwd_root / "cwd-only.png").write_bytes(b"cwd")
            manifest = manifest_root / "image.jsonl"
            manifest.write_text(
                json.dumps({
                    "image_path": "cwd-only.png",
                    "caption": "fixture",
                }) + "\n",
                encoding="utf-8",
            )
            original_cwd = Path.cwd()
            try:
                os.chdir(cwd_root)
                with self.assertRaisesRegex(
                    FileNotFoundError, "manifest/data roots"
                ):
                    load_explicit_eval_rows(
                        manifest, "image", data_root, limit=0
                    )
            finally:
                os.chdir(original_cwd)

    def test_image_cache_binds_media_encoder_preprocess_and_evaluation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            image_path.write_bytes(b"image-v1")
            config = types.SimpleNamespace(
                _name_or_path="fixture/vision",
                _commit_hash="revision-1",
                revision="main",
            )
            config.to_dict = lambda: {
                "hidden_size": 4,
                "commit": config._commit_hash,
            }
            vision_model = types.SimpleNamespace(
                name_or_path="fixture/vision",
                config=config,
            )
            image_processor = types.SimpleNamespace()
            image_processor.to_dict = lambda: {
                "size": 224,
                "crop": "center",
            }
            cache = FeatureCache(
                root / "cache",
                image_cache_context={
                    "checkpoint_sha256": "a" * 64,
                    "evaluator_sha256": "b" * 64,
                },
            )
            record = {
                "id": "fixture",
                "source": "fixture",
                "image_path": str(image_path),
            }
            with mock.patch(
                "training.olmoe_real_subset_runs.image_features_from_paths",
                return_value=torch.ones(1, 2, 3),
            ) as image_features:
                cache.image_batch(
                    image_processor,
                    vision_model,
                    [record],
                    torch.device("cpu"),
                    50,
                )
                cache_files = list((root / "cache").rglob("*.pt"))
                self.assertEqual(len(cache_files), 1)
                payload = torch.load(
                    cache_files[0],
                    map_location="cpu",
                    weights_only=True,
                )
                metadata = payload["metadata"]
                self.assertEqual(
                    metadata["media"]["canonical_path"],
                    str(image_path.resolve()),
                )
                self.assertEqual(
                    metadata["media"]["sha256"],
                    sha256_file(image_path),
                )
                self.assertEqual(
                    metadata["vision_encoder"]["revision"]["commit_hash"],
                    "revision-1",
                )
                self.assertEqual(
                    metadata["preprocess"]["config"]["size"], 224
                )
                self.assertEqual(metadata["encoder_feature_tokens"], 50)
                self.assertEqual(
                    metadata["evaluation_identity"]["checkpoint_sha256"],
                    "a" * 64,
                )
                self.assertIn("tensor_digest", payload)

                injected = dict(payload)
                injected["features"] = torch.full_like(
                    payload["features"], 99.0
                )
                torch.save(injected, cache_files[0])
                with self.assertRaisesRegex(
                    ValueError, "tensor digest mismatch"
                ):
                    cache.image_batch(
                        image_processor,
                        vision_model,
                        [record],
                        torch.device("cpu"),
                        50,
                    )
                torch.save(payload, cache_files[0])

                payload["metadata"]["evaluation_identity"][
                    "evaluator_sha256"
                ] = "c" * 64
                torch.save(payload, cache_files[0])
                with self.assertRaisesRegex(
                    ValueError, "cache metadata mismatch"
                ):
                    cache.image_batch(
                        image_processor,
                        vision_model,
                        [record],
                        torch.device("cpu"),
                        50,
                    )

                image_path.write_bytes(b"image-v2")
                cache.image_batch(
                    image_processor,
                    vision_model,
                    [record],
                    torch.device("cpu"),
                    50,
                )
                config._commit_hash = "revision-2"
                cache.image_batch(
                    image_processor,
                    vision_model,
                    [record],
                    torch.device("cpu"),
                    50,
                )
            self.assertEqual(image_features.call_count, 3)
            self.assertEqual(
                len(list((root / "cache").rglob("*.pt"))), 3
            )

    def test_frozen_run_rejects_cell_and_cardinality_drift(self):
        args = types.SimpleNamespace(
            negative_mode="random",
            conditional_negatives=4,
            conditional_candidates=5,
            conditional_queries=250,
            image_eval_samples=250,
            speech_eval_samples=250,
            max_length=512,
            conditional_batch_size=8,
            query_offset=0,
            candidate_offset=0,
            tie_epsilon=1e-8,
            candidate_permutation="query_identity_seeded",
            randomize_positive_position=True,
            eval_path="shared_prefix",
            prefix_control="real",
            candidate_seed=314159,
            control_seed=42,
            bootstrap_samples=2000,
            bootstrap_seed=12345,
            protocol_name="sealed_evaluation_v1",
            eval_split_name="sealed_test",
        )
        contract = {
            "id": "r5:real",
            "cell_id": "r5",
            "role": "secondary",
            "negative_mode": "random",
            "requested_candidate_count": 5,
            "conditional_negatives": 4,
            "conditional_candidates": 5,
            "conditional_queries": 250,
            "image_query_count": 250,
            "speech_query_count": 250,
            "image_eval_samples": 250,
            "speech_eval_samples": 250,
            "max_length": 512,
            "conditional_batch_size": 8,
            "query_offset": 0,
            "candidate_offset": 0,
            "tie_epsilon": 1e-8,
            "candidate_permutation": "query_identity_seeded",
            "randomize_positive_position": True,
            "control": "real",
            "prefix_control": "real",
            "eval_path": "shared_prefix",
            "candidate_seed": 314159,
            "control_seed": 42,
            "bootstrap_samples": 2000,
            "bootstrap_seed": 12345,
            "protocol_name": "sealed_evaluation_v1",
            "eval_split_name": "sealed_test",
        }
        protocol = {"evaluation_runs": [contract]}
        self.assertEqual(
            select_frozen_evaluation_run(protocol, args, 5)["id"],
            "r5:real",
        )
        args.negative_mode = "hard_text"
        with self.assertRaisesRegex(ValueError, "exactly one frozen"):
            select_frozen_evaluation_run(protocol, args, 5)
        args.negative_mode = "random"
        args.conditional_queries = 249
        with self.assertRaisesRegex(ValueError, "exactly one frozen"):
            select_frozen_evaluation_run(protocol, args, 5)
        args.conditional_queries = 250
        args.max_length = 511
        with self.assertRaisesRegex(ValueError, "exactly one frozen"):
            select_frozen_evaluation_run(protocol, args, 5)
        args.max_length = 512
        args.conditional_batch_size = 4
        with self.assertRaisesRegex(ValueError, "exactly one frozen"):
            select_frozen_evaluation_run(protocol, args, 5)

    def test_frozen_media_replacement_is_rejected_before_eval(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "image.png"
            media.write_bytes(b"frozen-image")
            digest = sha256_file(media)
            manifest = root / "image.jsonl"
            manifest.write_text(
                json.dumps({
                    "id": "image-0",
                    "image_path": "image.png",
                    "caption": "fixture",
                    "media_sha256": digest,
                }) + "\n",
                encoding="utf-8",
            )
            commitment = {
                "image-0": {
                    "media_sha256": digest,
                    "media_size_bytes": len(b"frozen-image"),
                }
            }
            media.write_bytes(b"replacement")
            with self.assertRaisesRegex(
                ValueError, "media_sha256 does not match actual bytes"
            ):
                load_explicit_eval_rows(
                    manifest,
                    "image",
                    root,
                    limit=0,
                    expected_media=commitment,
                    snapshot_root=root / "snapshots",
                )

    def test_absolute_manifest_media_must_stay_under_allowed_roots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_root = root / "manifests"
            data_root = root / "data"
            outside_root = root / "outside"
            manifest_root.mkdir()
            data_root.mkdir()
            outside_root.mkdir()
            outside_image = outside_root / "outside.png"
            outside_image.write_bytes(b"outside")
            manifest = manifest_root / "image.jsonl"
            manifest.write_text(
                json.dumps({
                    "image_path": str(outside_image.resolve()),
                    "caption": "fixture",
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                FileNotFoundError, "manifest/data roots"
            ):
                load_explicit_eval_rows(
                    manifest, "image", data_root, limit=0
                )

    def test_audio_cache_binds_bytes_processor_preprocess_and_evaluation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first_audio = root / "first.wav"
            second_audio = root / "second.wav"
            first_audio.write_bytes(b"audio-v1")
            second_audio.write_bytes(b"audio-v2")
            config = types.SimpleNamespace(
                _name_or_path="fixture/speech",
                _commit_hash="revision-1",
                revision="main",
            )
            config.to_dict = lambda: {
                "d_model": 4,
                "commit": config._commit_hash,
            }
            speech_model = types.SimpleNamespace(
                name_or_path="fixture/speech",
                config=config,
                encoder=nn.Linear(1, 1),
            )
            for parameter in speech_model.encoder.parameters():
                parameter.requires_grad_(False)
            speech_processor = types.SimpleNamespace()
            speech_processor.to_dict = lambda: {
                "sampling_rate": 16000,
                "padding": "max_length",
            }
            cache = FeatureCache(
                root / "cache",
                audio_cache_context={
                    "checkpoint_sha256": "a" * 64,
                    "evaluator_sha256": "b" * 64,
                },
            )
            record = {
                "id": "same-id",
                "source": "fixture",
                "audio_path": str(first_audio),
                "sample_rate": 16000,
                "preprocess": {
                    "resampled_to": 16000,
                    "max_seconds": 6.0,
                    "num_samples": 96000,
                },
            }
            audio_feature_extraction_implementation_identity()
            with mock.patch(
                "training.olmoe_real_subset_runs.audio_features_from_paths",
                side_effect=[
                    torch.ones(1, 2, 3),
                    torch.full((1, 2, 3), 2.0),
                ],
            ) as audio_features:
                first = cache.audio_batch(
                    speech_processor,
                    speech_model,
                    [record],
                    torch.device("cpu"),
                    16000,
                    50,
                    6.0,
                )
                repeated = cache.audio_batch(
                    speech_processor,
                    speech_model,
                    [record],
                    torch.device("cpu"),
                    16000,
                    50,
                    6.0,
                )
                self.assertTrue(torch.equal(first, repeated))
                record["audio_path"] = str(second_audio)
                second = cache.audio_batch(
                    speech_processor,
                    speech_model,
                    [record],
                    torch.device("cpu"),
                    16000,
                    50,
                    6.0,
                )
            self.assertEqual(audio_features.call_count, 2)
            self.assertEqual(float(second.mean()), 2.0)
            cache_files = list((root / "cache" / "audio_v2").rglob("*.pt"))
            self.assertEqual(len(cache_files), 2)
            payload = torch.load(
                cache_files[0], map_location="cpu", weights_only=True
            )
            self.assertIn("metadata", payload)
            self.assertIn("features", payload)
            self.assertEqual(
                payload["metadata"]["evaluation_identity"][
                    "checkpoint_sha256"
                ],
                "a" * 64,
            )
            self.assertEqual(
                payload["metadata"]["preprocess"]["max_seconds"], 6.0
            )

    def test_tampered_final_protocol_fails_before_data_or_model_read(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            run_root = root / "run"
            cache_dir = root / "cache"
            data_dir.mkdir()
            run_root.mkdir()
            image_manifest = root / "image.jsonl"
            speech_manifest = root / "speech.jsonl"
            image_manifest.write_text("{}\n", encoding="utf-8")
            speech_manifest.write_text("{}\n", encoding="utf-8")
            protocol = root / "protocol.json"
            protocol.write_text(
                json.dumps({
                    "protocol": "sealed_evaluation_protocol",
                    "protocol_content_sha256": "0" * 64,
                }),
                encoding="utf-8",
            )
            argv = [
                "eval_conditional_retrieval.py",
                "--evaluation-scope", "final",
                "--data-dir", str(data_dir),
                "--run-output-dir", str(run_root),
                "--checkpoint", str(root / "checkpoint.pt"),
                "--image-manifest", str(image_manifest),
                "--speech-manifest", str(speech_manifest),
                "--protocol-manifest", str(protocol),
                "--feature-cache-dir", str(cache_dir),
                "--per-query-output", str(root / "per-query.jsonl"),
                "--output", str(root / "metrics.json"),
            ]
            with mock.patch.object(
                sys, "argv", argv
            ), mock.patch(
                "scripts.eval_conditional_retrieval.load_explicit_eval_rows"
            ) as load_rows, mock.patch(
                "scripts.eval_conditional_retrieval.load_trained_wrapper"
            ) as load_model:
                with self.assertRaisesRegex(
                    ValueError, "protocol content SHA256 mismatch"
                ):
                    main()
            load_rows.assert_not_called()
            load_model.assert_not_called()

    def test_incomplete_evaluation_identity_fails_closed(self):
        identity = complete_evaluation_identity()
        del identity["tie_epsilon"]
        with self.assertRaisesRegex(ValueError, "identity is incomplete"):
            validate_complete_evaluation_identity(identity)

    def test_metric_affecting_arg_identity_cannot_be_partial(self):
        identity = complete_evaluation_identity()
        del identity["metric_affecting_args"]["bootstrap_seed"]
        with self.assertRaisesRegex(
            ValueError, "incomplete metric_affecting_args"
        ):
            validate_complete_evaluation_identity(identity)

    def test_existing_outputs_cannot_be_reused_or_overwritten(self):
        with TemporaryDirectory() as directory:
            existing = Path(directory) / "metrics.json"
            existing.write_text("original", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to reuse"):
                assert_output_paths_available([existing])
            with self.assertRaisesRegex(FileExistsError, "refusing to reuse"):
                atomic_write_text_exclusive(existing, "replacement")
            self.assertEqual(existing.read_text(encoding="utf-8"), "original")

            fresh = Path(directory) / "fresh.json"
            atomic_write_text_exclusive(fresh, "published")
            self.assertEqual(fresh.read_text(encoding="utf-8"), "published")

    def test_conditional_metrics_propagate_environment_and_artifact_provenance(self):
        repo_root = Path(__file__).resolve().parents[1]
        source_commit = resolve_git_head(repo_root)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            run_output_dir = root / "run"
            data_dir.mkdir()
            run_output_dir.mkdir()
            gamma_path = run_output_dir / "calibration" / "gamma.json"
            gamma_path.parent.mkdir()
            gamma_path.write_text(
                json.dumps({"gamma": [1.0]}), encoding="utf-8"
            )
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"fixture checkpoint")
            image = data_dir / "image.png"
            audio = data_dir / "audio.wav"
            image.touch()
            audio.touch()
            image_manifest = data_dir / "image_eval.jsonl"
            speech_manifest = data_dir / "speech_eval.jsonl"
            image_manifest.write_text(
                json.dumps({"image_path": str(image), "caption": "fixture image"}) + "\n",
                encoding="utf-8",
            )
            speech_manifest.write_text(
                json.dumps({"audio_path": str(audio), "transcript": "fixture speech"}) + "\n",
                encoding="utf-8",
            )
            output = root / "metrics.json"
            per_query = root / "per_query.jsonl"
            argv = [
                "eval_conditional_retrieval.py",
                "--evaluation-scope", "diagnostic",
                "--data-dir", str(data_dir),
                "--image-manifest", str(image_manifest),
                "--speech-manifest", str(speech_manifest),
                "--run-output-dir", str(run_output_dir),
                "--checkpoint", str(checkpoint),
                "--output", str(output),
                "--per-query-output", str(per_query),
                "--tie-epsilon", "0.000001",
                "--conditional-queries", "1",
                "--conditional-negatives", "0",
                "--eval-path", "no_prefix_lm",
                "--bootstrap-samples", "0",
            ]
            environment = {
                "SOURCE_COMMIT_SHA": source_commit,
                "RUNAI_JOB_NAME": "conditional-eval-test",
                "RUNAI_PROJECT": "test-project",
            }
            fake_config = types.SimpleNamespace(
                _name_or_path="fixture/vision",
                _commit_hash="fixture-revision",
                revision="main",
            )
            fake_config.to_dict = lambda: {
                "hidden_size": 4,
                "commit": fake_config._commit_hash,
            }
            fake_vision_model = types.SimpleNamespace(
                name_or_path="fixture/vision",
                config=fake_config,
            )
            fake_image_processor = types.SimpleNamespace()
            fake_image_processor.to_dict = lambda: {"size": 224}
            fake_meta = {
                "checkpoint_architecture": {
                    "base_model": "fixture/base",
                    "image_bridge_type": "query_resampler",
                },
                "gamma_provenance": {
                    "path": str(gamma_path.resolve()),
                    "sha256": sha256_file(gamma_path),
                    "size_bytes": gamma_path.stat().st_size,
                    "checkpoint_output_dir": str(run_output_dir.resolve()),
                    "checkpoint_bound": False,
                    "checkpoint_expected_sha256": None,
                },
                "checkpoint_restoration": {
                    "e3_checkpoint_sha256": sha256_file(checkpoint),
                    "source_checkpoint_hashes": {
                        "stage_b": None,
                        "multimodal_initial": None,
                        "speech_initial": None,
                    },
                    "restoration_order": [
                        "base_model",
                        "e3_training_checkpoint",
                    ],
                },
            }
            fake_loader_result = (
                object(), object(), fake_meta,
                fake_image_processor, fake_vision_model, None, None,
                torch.device("cpu"),
            )
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                sys, "argv", argv
            ), mock.patch(
                "scripts.eval_conditional_retrieval.load_trained_wrapper",
                return_value=fake_loader_result,
            ), mock.patch(
                "scripts.eval_conditional_retrieval.score_no_prefix_query",
                return_value=[-1.0],
            ):
                main()

            metrics = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(metrics["source_commit_sha"], source_commit)
            self.assertEqual(metrics["runai_job_name"], "conditional-eval-test")
            self.assertEqual(metrics["runai_project"], "test-project")
            self.assertEqual(metrics["candidate_permutation_policy"], "query_identity_seeded")
            self.assertEqual(metrics["tie_policy"], "strict_pessimistic_epsilon")
            self.assertEqual(metrics["tie_epsilon"], 0.000001)
            self.assertEqual(metrics["image_to_text_strict_r_at_1"], 1.0)
            self.assertEqual(metrics["image_to_text_mrr"], 1.0)
            per_query_rows = [
                json.loads(line)
                for line in per_query.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(per_query_rows), 2)
            for row in per_query_rows:
                self.assertEqual(row["candidate_permutation"], [0])
                self.assertEqual(row["candidate_indices"], [0])
                self.assertEqual(row["gold_index"], 0)
                self.assertEqual(row["raw_nll_scores"], [1.0])
                self.assertEqual(row["gold_nll_margin"], 0.0)
                self.assertEqual(row["tie_count"], 0)
                self.assertEqual(row["control_provenance"]["control_seed"], 42)
                self.assertEqual(row["control_provenance"]["eval_path"], "no_prefix_lm")
                self.assertEqual(row["source_provenance"]["query_uid"], row["query_uid"])
                self.assertTrue(row["candidate_ids"])
                evaluation = row["evaluation_provenance"]
                self.assertEqual(
                    evaluation["checkpoint_architecture"][
                        "image_bridge_type"
                    ],
                    "query_resampler",
                )
                self.assertEqual(evaluation["source_commit_sha"], source_commit)
                self.assertEqual(evaluation["runai_job_name"], "conditional-eval-test")
                self.assertEqual(evaluation["eval_split_name"], "eval_tail")
                self.assertEqual(evaluation["condition"], "no_prefix")
                self.assertEqual(
                    evaluation["evaluation_scope"], "diagnostic"
                )
                self.assertEqual(evaluation["bootstrap_seed"], 12345)
                self.assertEqual(evaluation["bootstrap_samples"], 0)
                self.assertEqual(evaluation["image_eval_samples"], 250)
                self.assertEqual(evaluation["speech_eval_samples"], 250)
                self.assertEqual(
                    set(evaluation["metric_affecting_args"]),
                    set(METRIC_AFFECTING_ARG_FIELDS),
                )
                self.assertTrue(
                    evaluation["image_cache_identity_sha256"]
                )
                self.assertEqual(evaluation["control_seed"], 42)
                self.assertEqual(evaluation["candidate_seed"], 314159)
                self.assertEqual(evaluation["negative_mode"], "stride")
                self.assertEqual(evaluation["requested_query_count"], 1)
                self.assertEqual(evaluation["image_query_count"], 1)
                self.assertEqual(evaluation["speech_query_count"], 1)
                self.assertEqual(evaluation["requested_candidate_count"], 1)
                self.assertEqual(evaluation["image_candidate_count"], 1)
                self.assertEqual(evaluation["speech_candidate_count"], 1)
                self.assertEqual(evaluation["query_offset"], 0)
                self.assertEqual(evaluation["candidate_offset"], -1)
                self.assertEqual(evaluation["tie_epsilon"], 0.000001)
                self.assertEqual(
                    evaluation["image_manifest_path"], str(image_manifest.resolve())
                )
                self.assertEqual(
                    evaluation["image_manifest_sha256"], sha256_file(image_manifest)
                )
                self.assertEqual(
                    evaluation["speech_manifest_path"], str(speech_manifest.resolve())
                )
                self.assertEqual(
                    evaluation["speech_manifest_sha256"], sha256_file(speech_manifest)
                )
                self.assertEqual(evaluation["gamma_path"], str(gamma_path.resolve()))
                self.assertEqual(evaluation["gamma_sha256"], sha256_file(gamma_path))
                self.assertEqual(
                    evaluation["evaluator_sha256"],
                    sha256_file(
                        repo_root / "scripts" / "eval_conditional_retrieval.py"
                    ),
                )
                self.assertEqual(
                    evaluation["e3_checkpoint_sha256"], sha256_file(checkpoint)
                )
                self.assertIsNone(evaluation["stage_b_checkpoint_sha256"])
                self.assertEqual(
                    evaluation["restoration_order"],
                    ["base_model", "e3_training_checkpoint"],
                )
                self.assertEqual(
                    evaluation["evaluation_identity_sha256"],
                    metrics["evaluation_identity_sha256"],
                )
            provenance = metrics["provenance"]
            self.assertEqual(provenance["source_commit_sha"], source_commit)
            self.assertEqual(provenance["runai_job_name"], "conditional-eval-test")
            self.assertEqual(provenance["runai_project"], "test-project")
            self.assertEqual(provenance["checkpoint_path"], str(checkpoint.resolve()))
            self.assertEqual(provenance["checkpoint_sha256"], sha256_file(checkpoint))
            self.assertEqual(provenance["checkpoint_size_bytes"], checkpoint.stat().st_size)
            self.assertEqual(provenance["resolved_data_dir"], str(data_dir.resolve()))
            self.assertEqual(provenance["image_manifest_path"], str(image_manifest.resolve()))
            self.assertEqual(provenance["image_manifest_sha256"], sha256_file(image_manifest))
            self.assertEqual(provenance["speech_manifest_path"], str(speech_manifest.resolve()))
            self.assertEqual(provenance["speech_manifest_sha256"], sha256_file(speech_manifest))

    def test_stage_b_restore_precedes_full_e3_restore(self):
        events = []
        wrapper = types.SimpleNamespace(
            lm=types.SimpleNamespace(
                model=types.SimpleNamespace(layers=[]),
            )
        )
        e3_state = {
            "args": {},
            "trainable_meta": {},
            "image_resampler": {},
            "audio_resampler": {},
            "image_retrieval_head": {},
            "audio_retrieval_head": {},
            "image_direct_retrieval_head": {},
            "audio_direct_retrieval_head": {},
        }

        def restore_stage_b(*_args, **_kwargs):
            events.append("stage_b")
            return {"router_state_restored": True}

        def restore_e3(*_args, **_kwargs):
            events.append("e3")

        with mock.patch(
            "scripts.eval_conditional_retrieval.restore_stage_b_student_initialization",
            side_effect=restore_stage_b,
        ), mock.patch(
            "scripts.eval_conditional_retrieval.restore_training_checkpoint",
            side_effect=restore_e3,
        ):
            restored = restore_evaluation_checkpoint_state(
                wrapper,
                e3_state,
                {"provenance": {}},
                "fixture/base",
                {"requested_base_model": "fixture/base"},
                speech_model=object(),
            )

        self.assertEqual(
            restored["restoration_order"],
            ["base_model", "stage_b_student_checkpoint", "e3_training_checkpoint"],
        )
        self.assertEqual(events, ["stage_b", "e3"])
        self.assertTrue(restored["adapter_state_restored"])
        self.assertFalse(restored["speech_encoder_state_restored"])
        self.assertFalse(restored["selected_expert_state_restored"])

    def test_embedding_tie_detection_uses_shared_weight_parameter(self):
        class EmbeddingLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_embeddings = nn.Embedding(4, 2)
                self.output_embeddings = nn.Linear(2, 4, bias=False)

            def get_input_embeddings(self):
                return self.input_embeddings

            def get_output_embeddings(self):
                return self.output_embeddings

        model = EmbeddingLM()
        self.assertFalse(lm_embeddings_are_tied(model))
        model.output_embeddings.weight = model.input_embeddings.weight
        self.assertIsNot(
            model.input_embeddings, model.output_embeddings
        )
        self.assertTrue(lm_embeddings_are_tied(model))

    def test_e3_embedding_tie_metadata_must_match_runtime_model(self):
        class UntiedLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_embeddings = nn.Embedding(4, 2)
                self.output_embeddings = nn.Linear(2, 4, bias=False)
                self.model = types.SimpleNamespace(layers=[])

            def get_input_embeddings(self):
                return self.input_embeddings

            def get_output_embeddings(self):
                return self.output_embeddings

        wrapper = types.SimpleNamespace(lm=UntiedLM())
        state = {
            "args": {},
            "trainable_meta": {"train_lm_head": True},
            "image_resampler": {},
            "audio_resampler": {},
            "image_retrieval_head": {},
            "audio_retrieval_head": {},
            "image_direct_retrieval_head": {},
            "audio_direct_retrieval_head": {},
            "lm_output_embeddings": (
                wrapper.lm.output_embeddings.state_dict()
            ),
            "lm_embeddings_tied": True,
        }
        with self.assertRaisesRegex(
            ValueError, "disagrees with runtime model"
        ):
            restore_evaluation_checkpoint_state(
                wrapper,
                state,
                None,
                "fixture/base",
                {},
            )

    def test_full_e3_restore_includes_speech_and_selected_experts(self):
        class FakeExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Parameter(torch.zeros(3, 2, 2))
                self.down_proj = nn.Parameter(torch.zeros(3, 2, 2))

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(2, 3, bias=False)
                self.experts = FakeExperts()
                self.mlp = types.SimpleNamespace(
                    gate=self.gate, experts=self.experts
                )

        class FakeLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([FakeLayer()])
                self.model = types.SimpleNamespace(layers=self.layers)

            def get_output_embeddings(self):
                return None

            def get_input_embeddings(self):
                return None

        class FakeWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm = FakeLM()
                self.image_resampler = nn.Linear(2, 2)
                self.audio_resampler = nn.Linear(2, 2)
                self.image_retrieval_head = nn.Linear(2, 2)
                self.audio_retrieval_head = nn.Linear(2, 2)
                self.image_direct_retrieval_head = nn.Linear(2, 2)
                self.audio_direct_retrieval_head = nn.Linear(2, 2)

        class FakeSpeechModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(2, 2, bias=False)

        wrapper = FakeWrapper()
        speech_model = FakeSpeechModel()
        selected_ids = {"0": [1]}
        selection_provenance = {
            "selection_json_sha256": "a" * 64,
            "selection_method": "ESFT-Gate",
            "selection_scope": "all_layers",
            "selected_expert_ids_by_layer": selected_ids,
        }

        def filled_state(module, value):
            return {
                name: torch.full_like(tensor, value)
                for name, tensor in module.state_dict().items()
            }

        speech_value = torch.full_like(speech_model.encoder.weight, 5.0)
        selected_gate_up = torch.full((1, 2, 2), 7.0)
        selected_down = torch.full((1, 2, 2), 9.0)
        state = {
            "args": {},
            "trainable_meta": {
                "speech_encoder_trainable_names": ["weight"],
                "selected_expert_training": True,
                "selected_expert_ids_by_layer": selected_ids,
                "expert_selection_provenance": selection_provenance,
            },
            "image_resampler": filled_state(wrapper.image_resampler, 1.0),
            "audio_resampler": filled_state(wrapper.audio_resampler, 2.0),
            "image_retrieval_head": filled_state(
                wrapper.image_retrieval_head, 3.0
            ),
            "audio_retrieval_head": filled_state(
                wrapper.audio_retrieval_head, 3.0
            ),
            "image_direct_retrieval_head": filled_state(
                wrapper.image_direct_retrieval_head, 4.0
            ),
            "audio_direct_retrieval_head": filled_state(
                wrapper.audio_direct_retrieval_head, 4.0
            ),
            "speech_encoder_trainable_state": {"weight": speech_value},
            "selected_experts": {
                "layer_0": {
                    "expert_ids": [1],
                    "gate_up_proj": selected_gate_up,
                    "down_proj": selected_down,
                }
            },
            "selected_expert_selection_provenance": selection_provenance,
        }

        restored = restore_evaluation_checkpoint_state(
            wrapper,
            state,
            None,
            "fixture/base",
            {},
            speech_model=speech_model,
        )

        self.assertTrue(
            torch.equal(speech_model.encoder.weight, speech_value)
        )
        experts = wrapper.lm.model.layers[0].mlp.experts
        self.assertTrue(torch.equal(experts.gate_up_proj[1], selected_gate_up[0]))
        self.assertTrue(torch.equal(experts.down_proj[1], selected_down[0]))
        self.assertTrue(torch.equal(
            experts.gate_up_proj[0], torch.zeros_like(experts.gate_up_proj[0])
        ))
        self.assertTrue(restored["speech_encoder_state_restored"])
        self.assertTrue(restored["selected_expert_state_restored"])
        self.assertTrue(restored["selected_expert_provenance_validated"])
        self.assertEqual(
            restored["restoration_order"],
            ["base_model", "e3_training_checkpoint"],
        )

    def test_missing_required_speech_and_selected_state_fail_closed(self):
        required = {
            "args": {},
            "image_resampler": {},
            "audio_resampler": {},
            "image_retrieval_head": {},
            "audio_retrieval_head": {},
            "image_direct_retrieval_head": {},
            "audio_direct_retrieval_head": {},
        }
        with self.subTest("untied input embeddings"):
            state = {
                **required,
                "trainable_meta": {"train_lm_head": True},
                "lm_output_embeddings": {"weight": torch.ones(2, 2)},
                "lm_embeddings_tied": False,
            }
            with self.assertRaisesRegex(
                ValueError, "missing input embeddings"
            ):
                validate_e3_training_checkpoint_state(
                    state, layer_count=1
                )
        with self.subTest("missing tied metadata"):
            state = {
                **required,
                "trainable_meta": {"train_lm_head": True},
                "lm_output_embeddings": {"weight": torch.ones(2, 2)},
            }
            with self.assertRaisesRegex(
                ValueError, "missing tied-embedding metadata"
            ):
                validate_e3_training_checkpoint_state(
                    state, layer_count=1
                )

        with self.subTest("speech"):
            state = {
                **required,
                "trainable_meta": {
                    "speech_encoder_trainable_names": ["weight"],
                },
            }
            with self.assertRaisesRegex(
                ValueError, "missing required speech encoder"
            ):
                validate_e3_training_checkpoint_state(state, layer_count=1)
        with self.subTest("selected_experts"):
            state = {
                **required,
                "trainable_meta": {
                    "selected_expert_training": True,
                    "selected_expert_ids_by_layer": {"0": [1]},
                    "expert_selection_provenance": {
                        "selection_json_sha256": "a" * 64,
                        "selection_method": "ESFT-Gate",
                        "selection_scope": "all_layers",
                        "selected_expert_ids_by_layer": {"0": [1]},
                    },
                },
            }
            with self.assertRaisesRegex(
                ValueError, "missing selected_experts"
            ):
                validate_e3_training_checkpoint_state(state, layer_count=1)

    def test_stage_b_sha_mismatch_fails_before_checkpoint_load(self):
        with TemporaryDirectory() as directory:
            stage_b = Path(directory) / "stage_b.pt"
            stage_b.write_bytes(b"actual-stage-b-state")
            declared_sha = "0" * 64
            e3_state = {
                "args": {
                    "stage_b_checkpoint": str(stage_b.resolve()),
                    "stage_b_checkpoint_sha256": declared_sha,
                },
                "trainable_meta": {
                    "stage_b_initialization": {
                        "path": str(stage_b.resolve()),
                        "sha256": declared_sha,
                        "state_restored": True,
                    }
                },
                "last_row": {
                    "stage_b_checkpoint_state_restored": True,
                    "source_stage_b_checkpoint_sha256": declared_sha,
                },
            }
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_verified_stage_b_for_e3(
                    e3_state, str(stage_b), declared_sha
                )

    def test_stage_b_declaration_without_verified_input_fails_closed(self):
        e3_state = {
            "trainable_meta": {
                "stage_b_initialization": {
                    "path": "/tmp/stage_b.pt",
                    "sha256": "a" * 64,
                    "state_restored": True,
                }
            },
            "last_row": {
                "stage_b_checkpoint_state_restored": True,
                "source_stage_b_checkpoint_sha256": "a" * 64,
            },
        }
        with self.assertRaisesRegex(ValueError, "both --stage-b-checkpoint"):
            validate_e3_stage_b_provenance(e3_state, "", "")

    def test_copied_per_query_metrics_provenance_is_rejected(self):
        original = complete_evaluation_identity(
            evaluation_identity_sha256="b" * 64,
        )
        copied = complete_evaluation_identity(
            e3_checkpoint_sha256="c" * 64,
            evaluation_identity_sha256="d" * 64,
        )
        rows = [{
            "condition": "real",
            "eval_split_name": "eval_tail",
            "evaluation_provenance": original,
        }]
        with self.assertRaisesRegex(ValueError, "copied evaluation provenance"):
            bind_per_query_evaluation_provenance(rows, copied)

    def test_source_checkpoint_hashes_preserve_all_initializers(self):
        source_hashes = extract_e3_source_checkpoint_hashes(
            {
                "trainable_meta": {
                    "stage_b_initialization": {"sha256": "a" * 64},
                    "multimodal_initialization": {"sha256": "b" * 64},
                    "speech_initialization": {"sha256": "c" * 64},
                }
            },
            "a" * 64,
        )
        self.assertEqual(source_hashes, {
            "stage_b": "a" * 64,
            "multimodal_initial": "b" * 64,
            "speech_initial": "c" * 64,
        })

    def test_direct_internal_metric_semantics_remain_unchanged(self):
        self.assertIsNone(validate_e3_stage_b_provenance(
            {"args": {}, "trainable_meta": {}, "last_row": {}}, "", ""
        ))
        evidence = tie_aware_nll_evidence([0.1, 0.3], 0, tie_epsilon=1e-8)
        metrics = tie_aware_metrics(
            [evidence["strict_rank"]], [evidence], "direct_internal"
        )
        self.assertEqual(metrics["direct_internal_r_at_1"], 1.0)
        self.assertEqual(metrics["direct_internal_strict_r_at_1"], 1.0)
        self.assertEqual(metrics["direct_internal_mrr"], 1.0)

    def test_source_commit_rejects_malformed_full_sha(self):
        with mock.patch.dict(os.environ, {"SOURCE_COMMIT_SHA": "677d2be"}, clear=True):
            with self.assertRaisesRegex(ValueError, "exact 40-hex"):
                load_run_environment_provenance(Path(__file__).resolve().parents[1])

    def test_source_commit_rejects_git_head_mismatch(self):
        repo_root = Path(__file__).resolve().parents[1]
        actual = resolve_git_head(repo_root)
        mismatched = ("0" if actual[0] != "0" else "1") + actual[1:]
        with mock.patch.dict(os.environ, {"SOURCE_COMMIT_SHA": mismatched}, clear=True):
            with self.assertRaisesRegex(ValueError, "does not match git HEAD"):
                load_run_environment_provenance(repo_root)


if __name__ == "__main__":
    unittest.main()
