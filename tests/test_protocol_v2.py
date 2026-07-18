from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import types
import unittest
from pathlib import Path

from scripts.analyze_paired_controls import (
    group_aware_chance_r_at_1,
    image_group_identity,
    positive_indices_for_group,
    production_bootstrap_r_at_1_ci,
    production_retrieval_metrics,
    reconstruct_and_validate_per_query_row,
    validate_per_query_identity_row,
)
from scripts.freeze_evaluation_protocol import build_evaluation_runs
from scripts.protocol_v2 import (
    ProtocolV2Error,
    per_query_jsonl_content,
    per_query_jsonl_sha256,
    validate_metrics_against_protocol_v2,
    validate_protocol_v2,
)
from scripts.sealed_position_allocator import (
    ALLOCATOR_NAME,
    ALLOCATOR_VERSION,
    assignment_provenance,
    build_allocator_manifest,
    canonical_sha256,
    enforce_gold_position_assignment,
    permute_candidates_for_query,
    select_local_candidate_indices,
)
from tests.sealed_metrics_fixture import (
    rebind_identity,
    sealed_image_manifest_content,
    sealed_image_manifest_rows,
    sealed_metrics,
    sealed_per_query_rows,
)


class ProtocolV2MetricsContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = {
            "id": "r5",
            "candidate_count": 5,
            "negative_mode": "random",
            "role": "secondary",
        }
        config = types.SimpleNamespace(
            image_query_count=6,
            speech_query_count=6,
            query_offset=0,
            candidate_offset=0,
            tie_epsilon=1e-8,
            candidate_permutation="query_identity_seeded",
            randomize_positive_position=True,
            candidate_seed=17,
            control_seed=23,
            bootstrap_samples=2000,
            bootstrap_seed=12345,
            protocol_name="sealed_evaluation_v1",
            eval_split_name="sealed_test",
            max_length=512,
            conditional_batch_size=16,
        )
        evaluator = {"path": "/tmp/evaluator.py", "sha256": "d" * 64}
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.image_manifest_path = Path(self.temp_dir.name) / "image.jsonl"
        self.image_manifest_rows = sealed_image_manifest_rows(6)
        image_manifest_content = sealed_image_manifest_content(
            self.image_manifest_rows
        )
        self.image_manifest_path.write_text(
            image_manifest_content, encoding="utf-8"
        )
        image_input = {
            "path": str(self.image_manifest_path.resolve()),
            "type": "file",
            "sha256": hashlib.sha256(
                image_manifest_content.encode("utf-8")
            ).hexdigest(),
            "bytes": len(image_manifest_content.encode("utf-8")),
        }
        self.speech_manifest_path = Path(self.temp_dir.name) / "speech.jsonl"
        self.speech_manifest_rows = [
            {
                "id": f"speech-{index}",
                "audio_path": f"/frozen/speech-{index}.wav",
                "transcript": f"reference transcript {index}",
                "speaker_id": f"speaker-{index // 2}",
                "source": {"source_ids": [f"utterance:{index}"]},
                "media_sha256": hashlib.sha256(
                    f"speech-media-{index}".encode("utf-8")
                ).hexdigest(),
            }
            for index in range(6)
        ]
        speech_manifest_content = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in self.speech_manifest_rows
        )
        self.speech_manifest_path.write_text(
            speech_manifest_content, encoding="utf-8"
        )
        speech_input = {
            "path": str(self.speech_manifest_path.resolve()),
            "type": "file",
            "sha256": hashlib.sha256(
                speech_manifest_content.encode("utf-8")
            ).hexdigest(),
            "bytes": len(speech_manifest_content.encode("utf-8")),
        }
        self.protocol = {
            "schema_version": 2,
            "protocol": "sealed_evaluation_protocol",
            "evaluation_matrix": [self.cell],
            "controls": [
                "real",
                "shuffled",
                "zero",
                "norm-matched-random",
                "no-prefix",
            ],
            "evaluation_runs": build_evaluation_runs([self.cell], config),
            "query_counts": {"image": 6, "speech": 6},
            "seeds": {"candidate_seed": 17, "control_seed": 23},
            "gold_position_allocator": build_allocator_manifest(
                [self.cell],
                candidate_seed=17,
                query_counts={"image": 6, "speech": 6},
            ),
            "inputs": {
                "image_test": image_input,
                "speech_test": speech_input,
                "evaluator_scripts": [evaluator],
            },
            "checkpoint": {
                "artifact": {
                    "path": "/tmp/checkpoint.pt",
                    "type": "file",
                    "sha256": "c" * 64,
                }
            },
            "protocol_content_sha256": "b" * 64,
        }
        self.run = next(
            run for run in self.protocol["evaluation_runs"]
            if run["id"] == "r5:real"
        )
        self.rows = self.make_rows()
        self.metrics = sealed_metrics(
            self.protocol,
            self.run,
            self.rows,
            protocol_file_sha256="a" * 64,
            checkpoint_path="/tmp/checkpoint.pt",
            checkpoint_sha256="c" * 64,
            evaluator_path=evaluator["path"],
            evaluator_sha256=evaluator["sha256"],
        )
        self.bind_allocator_metrics_contract(self.metrics, self.rows)
        self.rebuild_derived_contract(self.rows, self.metrics)

    @staticmethod
    def frozen_identity(row, modality, index) -> str:
        for key in (
            "uid",
            "source_uid",
            "image_uid",
            "utterance_id",
            "source_id",
            "id",
        ):
            value = row.get(key)
            if value not in (None, ""):
                return f"{modality}:{value}"
        stable_fields = {
            key: str(row[key])
            for key in (
                "source",
                "image_path",
                "audio_path",
                "caption",
                "transcript",
                "speaker_id",
            )
            if row.get(key) not in (None, "")
        }
        if stable_fields:
            payload = json.dumps(
                stable_fields,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return f"{modality}:sha256:{hashlib.sha256(payload).hexdigest()}"
        return f"{modality}:index:{index}"

    def make_rows(self):
        rows = sealed_per_query_rows(
            ("image",) * 6 + ("speech",) * 6,
            5,
        )
        plans = {
            (plan["cell_id"], plan["modality"]): plan
            for plan in self.protocol["gold_position_allocator"]["plans"]
        }
        for modality, manifest_rows in (
            ("image", self.image_manifest_rows),
            ("speech", self.speech_manifest_rows),
        ):
            plan = plans[("r5", modality)]
            group_ids = (
                [image_group_identity(row) for row in manifest_rows]
                if modality == "image"
                else None
            )
            modality_rows = [
                row for row in rows if row["modality"] == modality
            ]
            for query_index, row in enumerate(modality_rows):
                query_seed = (
                    17
                    + (1000003 if modality == "speech" else 0)
                    + 1009 * query_index
                )
                base_candidates = select_local_candidate_indices(
                    len(manifest_rows),
                    query_index,
                    4,
                    "random",
                    candidate_seed=query_seed,
                    group_ids=group_ids,
                )
                query_uid = self.frozen_identity(
                    manifest_rows[query_index], modality, query_index
                )
                (
                    candidate_indices,
                    permutation,
                    _,
                    permutation_seed,
                ) = permute_candidates_for_query(
                    base_candidates,
                    query_index,
                    23,
                    query_uid,
                )
                candidate_indices, permutation, gold_position = (
                    enforce_gold_position_assignment(
                        candidate_indices,
                        permutation,
                        query_index,
                        int(plan["positions"][query_index]),
                    )
                )
                candidate_ids = [
                    self.frozen_identity(
                        manifest_rows[index], modality, index
                    )
                    for index in candidate_indices
                ]
                text_key = "caption" if modality == "image" else "transcript"
                candidate_texts = [
                    str(manifest_rows[index][text_key])
                    for index in candidate_indices
                ]
                query_source = str(
                    manifest_rows[query_index].get("source", "")
                )
                source_provenance = {
                    "query_uid": query_uid,
                    "query_index": query_index,
                    "query_source": query_source,
                }
                row.update({
                    "query_uid": query_uid,
                    "query_index": query_index,
                    "query_source": query_source,
                    "candidate_ids": candidate_ids,
                    "candidate_indices": candidate_indices,
                    "candidate_permutation": permutation,
                    "candidate_permutation_seed": permutation_seed,
                    "gold_position_assignment": {
                        **assignment_provenance(
                            plan, self.protocol["gold_position_allocator"]
                        ),
                        "assignment_index": query_index,
                        "assigned_position": gold_position,
                    },
                    "candidate_texts": candidate_texts,
                    "candidate_count": len(candidate_ids),
                    "candidate_index": gold_position,
                    "gold_index": gold_position,
                    "gold_position": gold_position,
                    "gold_candidate_index": query_index,
                    "gold_candidate_id": candidate_ids[gold_position],
                    "candidate_set_hash": hashlib.sha256(
                        json.dumps(
                            candidate_ids,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                })
                if modality == "image":
                    query_group_id = str(group_ids[query_index])
                    candidate_group_ids = [
                        str(group_ids[index]) for index in candidate_indices
                    ]
                    positive_indices = positive_indices_for_group(
                        candidate_group_ids, query_group_id
                    )
                    row.update({
                        "query_image_group_id": query_group_id,
                        "candidate_group_ids": candidate_group_ids,
                        "positive_indices": positive_indices,
                        "positive_candidate_indices": [
                            candidate_indices[position]
                            for position in positive_indices
                        ],
                        "positive_candidate_ids": [
                            candidate_ids[position]
                            for position in positive_indices
                        ],
                        "positive_count": len(positive_indices),
                        "unique_candidate_group_count": len(
                            set(candidate_group_ids)
                        ),
                        "group_aware_chance_r_at_1": (
                            group_aware_chance_r_at_1(candidate_group_ids)
                        ),
                        "caption_row_chance_r_at_1": (
                            len(positive_indices) / len(candidate_group_ids)
                        ),
                    })
                    source_provenance["query_image_group_id"] = query_group_id
                else:
                    speaker_id = manifest_rows[query_index]["speaker_id"]
                    row["speaker_id"] = speaker_id
                    source_provenance["speaker_id"] = speaker_id
                row["source_provenance"] = source_provenance
        return rows

    def bind_allocator_metrics_contract(self, metrics, rows) -> None:
        allocator = self.protocol["gold_position_allocator"]
        plans = {
            plan["modality"]: plan
            for plan in allocator["plans"]
            if plan["cell_id"] == "r5"
        }
        common = {
            "gold_position_allocator_name": ALLOCATOR_NAME,
            "gold_position_allocator_version": ALLOCATOR_VERSION,
            "gold_position_assignment_plans_sha256": allocator[
                "plans_sha256"
            ],
        }
        identity = metrics["evaluation_provenance"]
        identity.update(common)
        metrics.update(common)
        for modality, plan in plans.items():
            binding = assignment_provenance(plan, allocator)
            identity[f"{modality}_gold_position_assignment"] = binding
            metrics[f"{modality}_gold_position_assignment"] = binding
            metrics[f"{modality}_gold_position_counts"] = list(
                plan["position_counts"]
            )
        rebind_identity(metrics)
        for row in rows:
            row["protocol"].update(common)
            row["evaluation_provenance"] = copy.deepcopy(identity)

    def refresh_image_candidate_fields(self, row) -> None:
        candidate_indices = row["candidate_indices"]
        candidate_ids = [
            self.frozen_identity(
                self.image_manifest_rows[index], "image", index
            )
            for index in candidate_indices
        ]
        candidate_group_ids = [
            image_group_identity(self.image_manifest_rows[index])
            for index in candidate_indices
        ]
        query_group_id = image_group_identity(
            self.image_manifest_rows[row["query_index"]]
        )
        positive_indices = positive_indices_for_group(
            candidate_group_ids, query_group_id
        )
        row.update({
            "candidate_ids": candidate_ids,
            "candidate_texts": [
                self.image_manifest_rows[index]["caption"]
                for index in candidate_indices
            ],
            "candidate_set_hash": hashlib.sha256(
                json.dumps(
                    candidate_ids,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "gold_candidate_id": candidate_ids[row["gold_position"]],
            "candidate_group_ids": candidate_group_ids,
            "positive_indices": positive_indices,
            "positive_candidate_indices": [
                candidate_indices[position] for position in positive_indices
            ],
            "positive_candidate_ids": [
                candidate_ids[position] for position in positive_indices
            ],
            "positive_count": len(positive_indices),
            "unique_candidate_group_count": len(set(candidate_group_ids)),
            "group_aware_chance_r_at_1": group_aware_chance_r_at_1(
                candidate_group_ids
            ),
            "caption_row_chance_r_at_1": (
                len(positive_indices) / len(candidate_group_ids)
            ),
        })

    @staticmethod
    def rebuild_derived_contract(rows, metrics) -> None:
        condition = str(metrics["condition"])
        loaded_by_modality = {"image": [], "speech": []}
        for index, row in enumerate(rows):
            if row["modality"] == "image":
                row.setdefault("positive_indices", [int(row["gold_position"])])
                row["positive_candidate_ids"] = [
                    row["candidate_ids"][position]
                    for position in row["positive_indices"]
                ]
            loaded = reconstruct_and_validate_per_query_row(
                condition,
                "fixture",
                index + 1,
                row,
                validate_production_fields=False,
            )
            row.update(loaded.production_fields)
            validated = validate_per_query_identity_row(
                condition,
                "fixture",
                index + 1,
                row,
                require_production_fields=True,
            )
            loaded_by_modality[str(row["modality"])].append(validated)

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
            expected = production_retrieval_metrics(
                loaded_by_modality[modality], prefix
            )
            expected.update({
                f"{prefix}_{field}": value
                for field, value in production_bootstrap_r_at_1_ci(
                    loaded_by_modality[modality],
                    int(metrics["evaluation_provenance"][
                        "bootstrap_samples"
                    ]),
                    int(metrics["evaluation_provenance"]["bootstrap_seed"])
                    + offset,
                ).items()
            })
            metrics.update(expected)
            for suffix in aliases:
                field = f"{prefix}_{suffix}"
                metrics[f"conditional_{field}"] = expected[field]

        image_positive_counts = [
            int(row.production_fields["positive_count"])
            for row in loaded_by_modality["image"]
        ]
        metrics["image_positive_counts"] = image_positive_counts
        metrics["conditional_image_positive_counts"] = image_positive_counts
        image_rows = [row for row in rows if row["modality"] == "image"]
        unique_group_counts = [
            int(row["unique_candidate_group_count"]) for row in image_rows
        ]
        group_chance = sum(
            float(row["group_aware_chance_r_at_1"]) for row in image_rows
        ) / len(image_rows)
        caption_chance = sum(
            float(row["caption_row_chance_r_at_1"]) for row in image_rows
        ) / len(image_rows)
        metrics.update({
            "image_unique_candidate_group_counts": unique_group_counts,
            "conditional_image_unique_candidate_group_counts": unique_group_counts,
            "image_group_aware_chance_r_at_1": group_chance,
            "conditional_image_group_aware_chance_r_at_1": group_chance,
            "image_caption_row_chance_r_at_1": caption_chance,
            "conditional_image_caption_row_chance_r_at_1": caption_chance,
            "image_chance_r_at_1": caption_chance,
            "conditional_image_chance_r_at_1": caption_chance,
            "image_legacy_gold_caption_position_chance_r_at_1": 0.2,
            "conditional_image_legacy_gold_caption_position_chance_r_at_1": 0.2,
            "speech_chance_r_at_1": 0.2,
            "conditional_speech_chance_r_at_1": 0.2,
        })
        metrics["per_query_sha256"] = per_query_jsonl_sha256(rows)

    def validate(
        self,
        metrics=None,
        rows=None,
        per_query_file_sha256=None,
    ) -> None:
        selected_rows = self.rows if rows is None else rows
        validate_metrics_against_protocol_v2(
            self.protocol,
            self.metrics if metrics is None else metrics,
            selected_rows,
            cell_id="r5",
            control="real",
            protocol_file_sha256="a" * 64,
            per_query_file_sha256=(
                per_query_jsonl_sha256(selected_rows)
                if per_query_file_sha256 is None
                else per_query_file_sha256
            ),
            checkpoint_path="/tmp/checkpoint.pt",
            checkpoint_sha256="c" * 64,
        )

    def test_accepts_exact_metrics_run_contract(self) -> None:
        self.validate()

    def test_rejects_rehashed_unbalanced_frozen_allocator(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        plan = protocol["gold_position_allocator"]["plans"][0]
        plan["positions"] = [0] * plan["query_count"]
        plan["position_counts"] = [plan["query_count"]] + [
            0
        ] * (plan["candidate_count"] - 1)
        plan["positions_sha256"] = canonical_sha256(plan["positions"])
        allocator = protocol["gold_position_allocator"]
        allocator["plans_sha256"] = canonical_sha256(allocator["plans"])
        for run in protocol["evaluation_runs"]:
            run["gold_position_assignment_plans_sha256"] = allocator[
                "plans_sha256"
            ]
            run["image_gold_positions_sha256"] = plan["positions_sha256"]
        with self.assertRaisesRegex(
            ProtocolV2Error, "invalid gold-position allocator"
        ):
            validate_protocol_v2(protocol)

    def test_rejects_evaluation_run_allocator_binding_drift(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["evaluation_runs"][0][
            "speech_gold_position_assignment_id"
        ] = "r5:image"
        with self.assertRaisesRegex(
            ProtocolV2Error,
            "drifted field speech_gold_position_assignment_id",
        ):
            validate_protocol_v2(protocol)

    def test_rejects_wrong_gold_assignment_fields_and_executed_counts(self) -> None:
        cases = {
            "candidate_seed": 999,
            "assignment_index": 1,
            "assigned_position": (
                int(self.rows[0]["gold_position"]) + 1
            ) % self.cell["candidate_count"],
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[0]["gold_position_assignment"][field] = value
                metrics = copy.deepcopy(self.metrics)
                metrics["per_query_sha256"] = per_query_jsonl_sha256(rows)
                with self.assertRaisesRegex(
                    ProtocolV2Error, "gold_position_assignment"
                ):
                    self.validate(metrics=metrics, rows=rows)

        metrics = copy.deepcopy(self.metrics)
        metrics["image_gold_position_counts"][0] += 1
        with self.assertRaisesRegex(
            ProtocolV2Error, "image_gold_position_counts"
        ):
            self.validate(metrics=metrics)

    def test_rejects_invalid_candidate_permutation_bijection(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["candidate_permutation"][1] = rows[0][
            "candidate_permutation"
        ][0]
        metrics = copy.deepcopy(self.metrics)
        metrics["per_query_sha256"] = per_query_jsonl_sha256(rows)
        with self.assertRaisesRegex(ProtocolV2Error, "must be a bijection"):
            self.validate(metrics=metrics, rows=rows)

    def test_rejects_wrong_seed_substituted_negative_and_wrong_order(self) -> None:
        cases = ("wrong_seed", "substituted_negative", "wrong_order")
        for case in cases:
            with self.subTest(case=case):
                rows = copy.deepcopy(self.rows)
                image = rows[0]
                if case == "wrong_seed":
                    image["candidate_permutation_seed"] += 1
                elif case == "substituted_negative":
                    omitted = next(
                        index
                        for index in range(len(self.image_manifest_rows))
                        if index not in image["candidate_indices"]
                    )
                    position = next(
                        index
                        for index in range(len(image["candidate_indices"]))
                        if index != image["gold_position"]
                    )
                    image["candidate_indices"][position] = omitted
                    self.refresh_image_candidate_fields(image)
                else:
                    positions = [
                        index
                        for index in range(len(image["candidate_indices"]))
                        if index != image["gold_position"]
                    ][:2]
                    left, right = positions
                    for field in (
                        "candidate_indices",
                        "candidate_permutation",
                        "scores",
                    ):
                        image[field][left], image[field][right] = (
                            image[field][right],
                            image[field][left],
                        )
                    self.refresh_image_candidate_fields(image)
                metrics = copy.deepcopy(self.metrics)
                self.rebuild_derived_contract(rows, metrics)
                with self.assertRaisesRegex(
                    ProtocolV2Error,
                    "candidate_permutation_seed|deterministic candidate",
                ):
                    self.validate(metrics=metrics, rows=rows)

    def test_accepts_frozen_speech_manifest_contract(self) -> None:
        speech_rows = [
            row for row in self.rows if row["modality"] == "speech"
        ]
        self.assertEqual(
            speech_rows[0]["candidate_texts"],
            [
                self.speech_manifest_rows[index]["transcript"]
                for index in speech_rows[0]["candidate_indices"]
            ],
        )
        self.assertEqual(
            speech_rows[1]["query_uid"],
            f"speech:{self.speech_manifest_rows[1]['id']}",
        )
        self.validate()

    def test_rejects_rehashed_frozen_speech_identity_tampering(self) -> None:
        cases = (
            "query_uid",
            "query_index",
            "candidate_indices",
            "candidate_ids",
            "gold_mapping",
            "candidate_texts",
        )
        for case in cases:
            with self.subTest(case=case):
                rows = copy.deepcopy(self.rows)
                speech = next(
                    row for row in rows if row["modality"] == "speech"
                )
                if case == "query_uid":
                    speech["query_uid"] = "speech:forged-query"
                    speech["source_provenance"]["query_uid"] = speech["query_uid"]
                elif case == "query_index":
                    speech["query_index"] = 4
                    speech["source_provenance"]["query_index"] = 4
                elif case == "candidate_indices":
                    speech["candidate_indices"][1], speech["candidate_indices"][2] = (
                        speech["candidate_indices"][2],
                        speech["candidate_indices"][1],
                    )
                elif case == "candidate_ids":
                    speech["candidate_ids"][1] = "speech:forged-candidate"
                elif case == "candidate_texts":
                    speech["candidate_texts"][1] = "forged reference transcript"
                speech["candidate_set_hash"] = hashlib.sha256(
                    json.dumps(
                        speech["candidate_ids"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                metrics = copy.deepcopy(self.metrics)
                self.rebuild_derived_contract(rows, metrics)
                if case == "gold_mapping":
                    speech["gold_candidate_index"] = 4
                    metrics["per_query_sha256"] = per_query_jsonl_sha256(rows)
                with self.assertRaises(ProtocolV2Error):
                    self.validate(metrics=metrics, rows=rows)

    def test_rejects_exact_file_byte_drift(self) -> None:
        canonical = per_query_jsonl_content(self.rows)
        byte_drift = canonical.replace("{", "{ ", 1).encode("utf-8")
        drift_sha256 = hashlib.sha256(byte_drift).hexdigest()
        self.assertNotEqual(drift_sha256, self.metrics["per_query_sha256"])
        with self.assertRaisesRegex(
            ProtocolV2Error, "exact file SHA256"
        ):
            self.validate(per_query_file_sha256=drift_sha256)

    def test_rejects_rehashed_contract_drift(self) -> None:
        cases = {
            "run_id": lambda value: value["evaluation_provenance"].__setitem__(
                "frozen_evaluation_run_id", "r5:shuffled"
            ),
            "max_length": lambda value: value["evaluation_provenance"][
                "metric_affecting_args"
            ].__setitem__("max_length", 1024),
            "batch": lambda value: value["evaluation_provenance"][
                "metric_affecting_args"
            ].__setitem__("conditional_batch_size", 8),
            "bootstrap": lambda value: value["evaluation_provenance"][
                "metric_affecting_args"
            ].__setitem__("bootstrap_samples", 999),
            "offset": lambda value: value["evaluation_provenance"][
                "metric_affecting_args"
            ].__setitem__("candidate_offset", 1),
            "control": lambda value: value.__setitem__("condition", "shuffled"),
            "cache_policy": self._drift_cache_policy,
            "missing_produced_digest": self._remove_produced_digest,
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                metrics = copy.deepcopy(self.metrics)
                mutate(metrics)
                rebind_identity(metrics)
                with self.assertRaises(ProtocolV2Error):
                    self.validate(metrics=metrics)

    @staticmethod
    def _drift_cache_policy(metrics: dict) -> None:
        policy = metrics["evaluation_provenance"]["feature_cache_policy"]
        policy["cache_reads_allowed"] = True
        for kind in ("image", "audio"):
            metrics["provenance"][f"{kind}_feature_cache"]["policy"][
                "cache_reads_allowed"
            ] = True

    @staticmethod
    def _remove_produced_digest(metrics: dict) -> None:
        del metrics["evaluation_provenance"]["image_produced_features_sha256"]
        del metrics["provenance"]["image_feature_cache"][
            "produced_features_sha256"
        ]

    def test_rejects_actual_candidate_cardinality_drift(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0]["candidate_ids"].pop()
        with self.assertRaisesRegex(ProtocolV2Error, "candidate cardinality"):
            self.validate(rows=rows)

    def test_rejects_same_cardinality_rows_from_another_run(self) -> None:
        shuffled = next(
            run for run in self.protocol["evaluation_runs"]
            if run["id"] == "r5:shuffled"
        )
        copied_rows = self.make_rows()
        copied_metrics = sealed_metrics(
            self.protocol,
            shuffled,
            copied_rows,
            protocol_file_sha256="a" * 64,
            checkpoint_path="/tmp/checkpoint.pt",
            checkpoint_sha256="c" * 64,
            evaluator_path="/tmp/evaluator.py",
            evaluator_sha256="d" * 64,
        )
        metrics = copy.deepcopy(self.metrics)
        metrics["per_query_sha256"] = copied_metrics["per_query_sha256"]
        for row in copied_rows:
            row["evaluation_provenance"] = copy.deepcopy(
                metrics["evaluation_provenance"]
            )
        with self.assertRaisesRegex(
            ProtocolV2Error, "per-query row 0 condition"
        ):
            self.validate(metrics=metrics, rows=copied_rows)


    def test_rejects_minimal_per_query_identity_rows(self) -> None:
        rows = [
            {
                "modality": row["modality"],
                "candidate_ids": list(row["candidate_ids"]),
            }
            for row in self.rows
        ]
        metrics = copy.deepcopy(self.metrics)
        metrics["per_query_sha256"] = per_query_jsonl_sha256(rows)
        with self.assertRaisesRegex(ProtocolV2Error, "missing required fields"):
            self.validate(metrics=metrics, rows=rows)

    def test_rejects_duplicate_query_uid_and_candidate_hash_drift(self) -> None:
        for expected, mutate in (
            (
                "duplicates query_uid",
                lambda rows: rows[1].__setitem__(
                    "query_uid", rows[0]["query_uid"]
                ),
            ),
            (
                "candidate_set_hash disagrees",
                lambda rows: rows[0].__setitem__(
                    "candidate_set_hash", "0" * 64
                ),
            ),
        ):
            with self.subTest(expected=expected):
                rows = copy.deepcopy(self.rows)
                mutate(rows)
                metrics = copy.deepcopy(self.metrics)
                metrics["per_query_sha256"] = per_query_jsonl_sha256(rows)
                with self.assertRaisesRegex(ProtocolV2Error, expected):
                    self.validate(metrics=metrics, rows=rows)


    def test_rejects_rehashed_scoreless_and_noncanonical_rows(self) -> None:
        cases = (
            (
                "scores must be a finite list",
                lambda rows: rows[0].__setitem__("scores", None),
            ),
            (
                "query_uid must be a non-empty string",
                lambda rows: rows[0].__setitem__("query_uid", 12345),
            ),
            (
                "rank must be an integer",
                lambda rows: rows[0].__setitem__("rank", 0.9),
            ),
        )
        for expected, mutate in cases:
            with self.subTest(expected=expected):
                rows = copy.deepcopy(self.rows)
                mutate(rows)
                metrics = copy.deepcopy(self.metrics)
                metrics["per_query_sha256"] = per_query_jsonl_sha256(rows)
                with self.assertRaisesRegex(ProtocolV2Error, expected):
                    self.validate(metrics=metrics, rows=rows)

    def test_accepts_production_pessimistic_tie_and_group_aware_rows(self) -> None:
        tied_rows = copy.deepcopy(self.rows)
        for tied in tied_rows[:2]:
            tied["scores"] = [1.0] * len(tied["candidate_ids"])
        tied_metrics = copy.deepcopy(self.metrics)
        self.rebuild_derived_contract(tied_rows, tied_metrics)
        self.validate(metrics=tied_metrics, rows=tied_rows)

        self.image_manifest_rows[1]["media_sha256"] = (
            self.image_manifest_rows[0]["media_sha256"]
        )
        image_manifest_content = sealed_image_manifest_content(
            self.image_manifest_rows
        )
        self.image_manifest_path.write_text(
            image_manifest_content, encoding="utf-8"
        )
        self.protocol["inputs"]["image_test"].update({
            "sha256": hashlib.sha256(
                image_manifest_content.encode("utf-8")
            ).hexdigest(),
            "bytes": len(image_manifest_content.encode("utf-8")),
        })
        grouped_rows = self.make_rows()
        grouped_metrics = sealed_metrics(
            self.protocol,
            self.run,
            grouped_rows,
            protocol_file_sha256="a" * 64,
            checkpoint_path="/tmp/checkpoint.pt",
            checkpoint_sha256="c" * 64,
            evaluator_path="/tmp/evaluator.py",
            evaluator_sha256="d" * 64,
        )
        self.bind_allocator_metrics_contract(grouped_metrics, grouped_rows)
        self.rebuild_derived_contract(grouped_rows, grouped_metrics)
        self.assertTrue(all(
            row["positive_count"] == 1
            for row in grouped_rows
            if row["modality"] == "image"
        ))
        self.validate(metrics=grouped_metrics, rows=grouped_rows)

    def test_rejects_negative_group_declared_as_positive(self) -> None:
        rows = copy.deepcopy(self.rows)
        image = rows[0]
        negative_position = next(
            position
            for position in range(len(image["candidate_ids"]))
            if position != image["gold_position"]
        )
        image["positive_indices"] = sorted([
            image["gold_position"],
            negative_position,
        ])
        image["positive_candidate_indices"] = [
            image["candidate_indices"][position]
            for position in image["positive_indices"]
        ]
        image["positive_candidate_ids"] = [
            image["candidate_ids"][position]
            for position in image["positive_indices"]
        ]
        image["positive_count"] = 2
        metrics = copy.deepcopy(self.metrics)
        self.rebuild_derived_contract(rows, metrics)
        with self.assertRaisesRegex(
            ProtocolV2Error, "immutable positive_indices"
        ):
            self.validate(metrics=metrics, rows=rows)

    def test_rejects_multi_positive_speech_declaration(self) -> None:
        rows = copy.deepcopy(self.rows)
        speech = next(
            row for row in rows if row["modality"] == "speech"
        )
        negative_position = next(
            position
            for position in range(len(speech["candidate_ids"]))
            if position != speech["gold_position"]
        )
        speech["positive_indices"] = sorted([
            speech["gold_position"],
            negative_position,
        ])
        speech["positive_candidate_ids"] = [
            speech["candidate_ids"][position]
            for position in speech["positive_indices"]
        ]
        metrics = copy.deepcopy(self.metrics)
        self.rebuild_derived_contract(rows, metrics)
        with self.assertRaisesRegex(
            ProtocolV2Error, "immutable positive identities"
        ):
            self.validate(metrics=metrics, rows=rows)

    def test_rejects_rehashed_derived_row_field_drift(self) -> None:
        fields = (
            "positive_count",
            "strict_rank",
            "strict_r_at_1",
            "reciprocal_rank",
            "gold_nll_margin",
            "gold_tie_count",
            "best_tie_count",
            "predicted_position",
            "predicted_candidate_id",
            "best_candidate_indices",
        )
        for field in fields:
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                value = rows[0][field]
                if isinstance(value, list):
                    rows[0][field] = list(value) + ["corrupt"]
                elif isinstance(value, str):
                    rows[0][field] = value + "-corrupt"
                else:
                    rows[0][field] = value + 1
                metrics = copy.deepcopy(self.metrics)
                metrics["per_query_sha256"] = per_query_jsonl_sha256(rows)
                with self.assertRaisesRegex(
                    ProtocolV2Error,
                    rf"{field} disagrees with canonical score reconstruction",
                ):
                    self.validate(metrics=metrics, rows=rows)

    def test_rejects_aggregate_and_bootstrap_ci_drift(self) -> None:
        fields = (
            "image_to_text_r_at_5",
            "image_to_text_mrr",
            "image_to_text_tie_count",
            "conditional_speech_to_text_strict_r_at_1",
            "image_to_text_r_at_1_bootstrap_ci_low",
            "conditional_speech_to_text_r_at_1_bootstrap_ci_high",
            "image_positive_counts",
        )
        for field in fields:
            with self.subTest(field=field):
                metrics = copy.deepcopy(self.metrics)
                value = metrics[field]
                if isinstance(value, list):
                    metrics[field] = list(value) + [999]
                else:
                    metrics[field] = value + 1
                with self.assertRaisesRegex(
                    ProtocolV2Error, "sealed metrics contract mismatch"
                ):
                    self.validate(metrics=metrics)


class ProtocolV2TieEpsilonTest(unittest.TestCase):
    def test_rejects_non_numeric_nonfinite_and_negative_tie_epsilon(self) -> None:
        fixture = ProtocolV2MetricsContractTest()
        fixture.setUp()
        for value in ({}, math.nan, math.inf, -1e-9):
            with self.subTest(value=value):
                protocol = copy.deepcopy(fixture.protocol)
                protocol["evaluation_runs"][0]["tie_epsilon"] = value
                with self.assertRaisesRegex(
                    ProtocolV2Error, "finite nonnegative"
                ):
                    validate_protocol_v2(protocol)


if __name__ == "__main__":
    unittest.main()
