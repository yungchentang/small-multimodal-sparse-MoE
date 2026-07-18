"""Revision and provenance tests for the real subset builder."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import hf_sources


MODULE_PATH = Path(__file__).parents[1] / "datasets" / "build_real_subset.py"
SPEC = importlib.util.spec_from_file_location("build_real_subset_revisions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
real_subset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = real_subset
SPEC.loader.exec_module(real_subset)


class SourceProfileTests(unittest.TestCase):
    def test_default_profiles_are_closed_over_final_registered_datasets(self) -> None:
        candidates = [
            candidate
            for profile in real_subset.REAL_SOURCE_PROFILES.values()
            for candidate in profile
        ]
        candidates.extend(real_subset.IMAGE_SOURCE_PROFILE)
        candidates.extend(real_subset.SPEECH_SOURCE_PROFILE)

        self.assertEqual(
            {candidate.repo for candidate in candidates},
            {
                "NeelNanda/c4-10k",
                "allenai/c4",
                "codeparrot/codeparrot-clean-valid",
                "hails/agieval-logiqa-en",
                "openai/gsm8k",
                "hails/agieval-sat-en",
                "hails/agieval-sat-math",
                "hails/agieval-lsat-ar",
                "hails/agieval-lsat-lr",
                "jxie/coco_captions",
                "openslr/librispeech_asr",
            },
        )
        for candidate in candidates:
            self.assertEqual(candidate.revision, hf_sources.DATASET_REGISTRY[candidate.repo])

    def test_alternate_candidate_requires_exact_commit(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown Hugging Face dataset"):
            real_subset.Candidate("example/alternate", None, "train")
        with self.assertRaisesRegex(ValueError, "exact lowercase 40-hex"):
            real_subset.Candidate("example/alternate", None, "train", revision="main")

        candidate = real_subset.Candidate(
            "example/alternate", None, "train", revision="a" * 40
        )
        self.assertEqual(candidate.revision, "a" * 40)


class RevisionPropagationTests(unittest.TestCase):
    def test_dataset_loader_receives_candidate_revision(self) -> None:
        dataset = Mock()
        dataset.__len__ = Mock(return_value=1)
        dataset.select.return_value = [{"text": "row"}]
        candidate = real_subset.REAL_SOURCE_PROFILES["text"][0]

        with patch.object(real_subset, "load_dataset_ref", return_value=dataset) as loader:
            rows = list(real_subset.iter_dataset_rows(candidate, 1))

        self.assertEqual(rows, [{"text": "row"}])
        loader.assert_called_once_with(
            candidate.repo,
            candidate.config,
            revision=candidate.revision,
            split=candidate.split,
            streaming=candidate.streaming,
        )

    def test_tokenizer_loader_receives_registered_model_revision(self) -> None:
        factory = object()
        tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token_id=1)
        transformers = types.SimpleNamespace(AutoTokenizer=factory)

        with patch.dict(sys.modules, {"transformers": transformers}):
            with patch.object(
                real_subset, "load_pretrained", return_value=tokenizer
            ) as loader:
                result = real_subset.load_tokenizer("allenai/OLMoE-1B-7B-0924")

        self.assertIs(result, tokenizer)
        loader.assert_called_once_with(
            factory,
            hf_sources.HFRef(
                "allenai/OLMoE-1B-7B-0924",
                "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
            ),
        )


class SpeechSplitSeedPlumbingTests(unittest.TestCase):
    def test_seed_is_wired_through_cluster_entrypoints(self) -> None:
        root = MODULE_PATH.parents[1]
        run_sh = (root / "run.sh").read_text(encoding="utf-8")
        submit = (root / "scripts" / "submit_runai.sh").read_text(encoding="utf-8")
        self.assertIn('--speech-split-seed "${SPEECH_SPLIT_SEED:-0}"', run_sh)
        self.assertIn(
            '--environment SPEECH_SPLIT_SEED="${SPEECH_SPLIT_SEED:-}"', submit
        )


class ManifestTests(unittest.TestCase):
    @staticmethod
    def _args(output_dir: str) -> argparse.Namespace:
        return argparse.Namespace(
            output_dir=output_dir,
            tokenizer_model="allenai/OLMoE-1B-7B-0924",
            tokenizer_revision=None,
            block_size=16,
            text_samples=1,
            code_samples=1,
            reasoning_samples=1,
            math_samples=1,
            education_samples=1,
            text_train_blocks=1,
            code_train_blocks=1,
            reasoning_train_blocks=1,
            math_train_blocks=1,
            education_train_blocks=1,
            eval_blocks_per_task=1,
            image_samples=1,
            speech_samples=1,
            image_eval_samples=1,
            speech_eval_samples=2,
            speech_split_seed=19,
            sample_rate=16000,
            max_audio_seconds=1.0,
            caption_min_ascii_ratio=0.85,
            caption_min_letters=8,
            max_source_audio_seconds=0.0,
            max_transcript_words=0,
            allow_short=True,
        )

    def test_manifest_preserves_sources_and_adds_revision_provenance(self) -> None:
        def fake_text(task, candidates, *_args, **_kwargs):
            source = candidates[0].label
            return [{"task": task, "source": source}], source

        image_source = real_subset.IMAGE_SOURCE_PROFILE[0].label
        speech_source = real_subset.SPEECH_SOURCE_PROFILE[0].label
        fingerprint = {
            "algorithm": "sha256",
            "semantics": "encoded_audio_bytes_v1",
            "value": "a" * 64,
        }
        speech_rows = [
            {
                "id": index,
                "partition": partition,
                "source_dataset": "openslr/librispeech_asr",
                "source_config": "clean",
                "source_split": "train.360",
                "speaker_id": str(index + 1),
                "chapter_id": str(index + 11),
                "utterance_id": f"{index + 1}-{index + 11}-0000",
                "audio_path": f"audio/{index}.wav",
                "source_audio_fingerprint": fingerprint,
                "audio_fingerprint": fingerprint,
                "audio_sha256": fingerprint["value"],
                "task": "speech",
            }
            for index, partition in enumerate(("train", "dev", "eval"))
        ]
        speech_partition = {
            "schema_version": 1,
            "policy": real_subset.SPEECH_PARTITION_POLICY,
            "seed": 19,
            "group_key": ["source_dataset", "speaker_id"],
            "partition_order": ["train", "dev", "eval"],
            "legacy_tail_semantics": "dev and eval together are the held-out tail",
            "heldout_row_target": 2,
            "row_counts": {"train": 1, "dev": 1, "eval": 1},
            "group_counts": {"train": 1, "dev": 1, "eval": 1},
            "overlap_audit": {
                "pairwise_group_overlap_count": 0,
                "pairwise_group_overlaps": {
                    "train_dev": [], "train_eval": [], "dev_eval": [],
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(real_subset, "build_text_like_rows", side_effect=fake_text),
                patch.object(real_subset, "write_jsonl"),
                patch.object(real_subset, "save_json"),
                patch.object(
                    real_subset,
                    "write_text_blocks",
                    return_value={"train_blocks": 5, "eval_blocks": 5},
                ),
                patch.object(
                    real_subset,
                    "build_image_rows",
                    return_value=([{"task": "image"}], image_source),
                ),
                patch.object(
                    real_subset,
                    "build_audio_rows",
                    return_value=(speech_rows, speech_source, speech_partition),
                ),
            ):
                manifest = real_subset.build_all(self._args(directory))

        self.assertEqual(manifest["sources"]["image"], image_source)
        self.assertEqual(manifest["sources"]["speech"], speech_source)
        self.assertEqual(
            manifest["hf_sources"]["tokenizer"],
            {
                "repo_id": "allenai/OLMoE-1B-7B-0924",
                "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
            },
        )
        self.assertEqual(
            manifest["hf_sources"]["image"][0]["revision"],
            "a2ed90d49b61dd13dd71f399c70f5feb897f8bec",
        )
        self.assertEqual(
            manifest["hf_sources"]["speech"][0]["revision"],
            "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
        )
        for task in ("text", "code", "reasoning", "math", "education"):
            self.assertRegex(manifest["hf_sources"][task][0]["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(manifest["speech_partition"]["seed"], 19)
        self.assertEqual(manifest["speech_partition"]["row_counts"], {"train": 1, "dev": 1, "eval": 1})
        self.assertEqual(manifest["counts"]["speech_eval_utterances"], 2)
        self.assertEqual(manifest["speech_partition_ledger"]["rows"], 3)
        self.assertEqual(
            manifest["speech_audio_bytes_commitment"]["policy"],
            "exact_materialized_wav_sha256_per_row_v1",
        )
        self.assertEqual(manifest["speech_audio_bytes_commitment"]["rows"], 3)
        self.assertRegex(
            manifest["speech_audio_bytes_commitment"]["row_commitment_sha256"],
            r"^[0-9a-f]{64}$",
        )


if __name__ == "__main__":
    unittest.main()
