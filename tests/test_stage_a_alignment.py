"""Lightweight Stage A development-alignment tests."""

from __future__ import annotations

import hashlib
import io
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from training import olmoe_real_subset_runs as stage


class FakeConfig:
    def __init__(self, hidden_size: int = 2, commit: str = "rev-a") -> None:
        self.hidden_size = hidden_size
        self.d_model = hidden_size
        self._name_or_path = "fake/whisper"
        self._commit_hash = commit
        self.revision = commit

    def to_dict(self):
        return {
            "hidden_size": self.hidden_size,
            "_name_or_path": self._name_or_path,
            "_commit_hash": self._commit_hash,
            "revision": self.revision,
        }


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, 2)
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])
        self.layer_norm = nn.LayerNorm(2)

    def forward(self, input_features):
        hidden = self.input_projection(input_features.float())
        for layer in self.layers:
            hidden = layer(hidden)
        return types.SimpleNamespace(last_hidden_state=self.layer_norm(hidden))


class FakeSpeechModel(nn.Module):
    def __init__(self, commit: str = "rev-a") -> None:
        super().__init__()
        self.config = FakeConfig(commit=commit)
        self.encoder = FakeEncoder()


class FakeProcessorComponent:
    def __init__(self, normalization: str) -> None:
        self.normalization = normalization

    def to_dict(self):
        return {"normalization": self.normalization}


class FakeSpeechProcessor(FakeProcessorComponent):
    def __init__(self, normalization: str, tokenizer_mode: str = "base") -> None:
        super().__init__(normalization)
        self.feature_extractor = FakeProcessorComponent(normalization)
        self.tokenizer = FakeProcessorComponent(tokenizer_mode)


class FakeLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, 2)
        self.model = types.SimpleNamespace(layers=[])

    def get_output_embeddings(self):
        return None

    def get_input_embeddings(self):
        return None


class FakeWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm = FakeLM()
        self.image_resampler = nn.Linear(2, 2)
        self.audio_resampler = nn.Linear(2, 2)
        self.image_retrieval_head = nn.Linear(2, 2)
        self.audio_retrieval_head = nn.Linear(2, 2)
        self.image_direct_retrieval_head = nn.Linear(2, 2)
        self.audio_direct_retrieval_head = nn.Linear(2, 2)


class ImageAlignmentTests(unittest.TestCase):
    def test_wrapper_retrieval_dimension_tracks_image_target(self) -> None:
        model = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=8))
        vision_config = types.SimpleNamespace(
            vision_config=types.SimpleNamespace(hidden_size=4), projection_dim=3
        )
        vision = types.SimpleNamespace(config=vision_config)
        speech = FakeSpeechModel()
        base_args = dict(
            image_prefix_tokens=2,
            audio_prefix_tokens=2,
            alignment_prefix_residual=False,
            speech_target_space="olmoe_text_hidden",
            image_bridge_type="linear_projector",
            audio_bridge_type="attention_pool",
            bridge_num_heads=2,
        )

        with patch.object(stage, "OLMoEMultimodalPrefixWrapper", side_effect=lambda **kwargs: kwargs):
            clip = stage.make_wrapper(
                model,
                vision,
                speech,
                types.SimpleNamespace(**base_args, image_alignment_target="clip_text"),
            )
            hidden = stage.make_wrapper(
                model,
                vision,
                speech,
                types.SimpleNamespace(**base_args, image_alignment_target="olmoe_caption_hidden"),
            )

        self.assertEqual(clip["image_retrieval_dim"], 3)
        self.assertEqual(hidden["image_retrieval_dim"], 8)
        self.assertEqual(hidden["image_bridge_type"], "linear_projector")
        self.assertEqual(hidden["audio_bridge_type"], "attention_pool")
        self.assertEqual(hidden["bridge_num_heads"], 2)

    def test_caption_hidden_target_uses_lm_embedding_path(self) -> None:
        expected = torch.tensor([[1.0, 0.0]])
        with (
            patch.object(stage, "lm_text_embeddings", return_value=expected) as lm_embed,
            patch.object(stage, "clip_text_embeddings") as clip_embed,
        ):
            actual = stage.image_alignment_text_embeddings(
                "olmoe_caption_hidden", object(), object(), object(), object(),
                ["caption"], torch.device("cpu"), 16, 1,
            )

        self.assertIs(actual, expected)
        lm_embed.assert_called_once()
        clip_embed.assert_not_called()


class AudioFeatureTests(unittest.TestCase):
    def test_runtime_truncation_happens_after_resampling(self) -> None:
        speech = FakeSpeechModel()
        speech.requires_grad_(False)
        captured = []

        def processor(waveforms, **_kwargs):
            captured.extend(len(waveform) for waveform in waveforms)
            return {"input_features": torch.zeros(len(waveforms), 3, 2)}

        fake_librosa = types.SimpleNamespace(
            resample=lambda audio, orig_sr, target_sr: np.zeros(
                int(len(audio) * target_sr / orig_sr), dtype=np.float32
            )
        )
        with (
            patch.object(stage, "load_audio_file", return_value=(np.zeros(16000, dtype=np.float32), 8000)),
            patch.dict(sys.modules, {"librosa": fake_librosa}),
        ):
            stage.audio_features_from_paths(
                processor, speech, ["fake.wav"], torch.device("cpu"), 16000, 0, 0.5
            )

        self.assertEqual(captured, [8000])

    def test_strict_batch_decodes_the_verified_wav_snapshot(self) -> None:
        speech = FakeSpeechModel()
        speech.requires_grad_(False)
        captured_waveforms = []

        def processor(waveforms, **_kwargs):
            captured_waveforms.extend(waveform.copy() for waveform in waveforms)
            return {"input_features": torch.zeros(len(waveforms), 3, 2)}

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.wav"
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(
                    b"".join(
                        sample.to_bytes(2, "little", signed=True)
                        for sample in (0, 1000, -1000, 500)
                    )
                )
            expected_sha = hashlib.sha256(audio_path.read_bytes()).hexdigest()
            record = {
                "id": "1",
                "source": "fake",
                "audio_path": audio_path.name,
                "audio_sha256": expected_sha,
            }
            original_snapshot = stage.secure_speech_audio_snapshot
            reads = 0

            def mutate_after_read(*args, **kwargs):
                nonlocal reads
                path, payload = original_snapshot(*args, **kwargs)
                if path == audio_path:
                    reads += 1
                    path.write_bytes(b"mutated after verified snapshot")
                return path, payload

            with patch.object(
                stage,
                "secure_speech_audio_snapshot",
                side_effect=mutate_after_read,
            ):
                features = stage.FeatureCache(
                    Path(directory) / "cache",
                    strict_audio_integrity=True,
                    speech_audio_data_dir=Path(directory).resolve(),
                ).audio_batch(
                    processor,
                    speech,
                    [record],
                    torch.device("cpu"),
                    16000,
                    50,
                )

        self.assertEqual(reads, 1)
        self.assertEqual(len(captured_waveforms), 1)
        self.assertEqual(captured_waveforms[0].shape, (4,))
        self.assertEqual(tuple(features.shape), (1, 3, 2))

    def test_strict_batch_rejects_symlink_replacement_after_first_read(self) -> None:
        speech = FakeSpeechModel()
        speech.requires_grad_(False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data_dir = root / "data"
            data_dir.mkdir()
            audio_path = data_dir / "sample.wav"
            external_path = root / "external.wav"
            audio_path.write_bytes(b"same immutable payload")
            external_path.write_bytes(audio_path.read_bytes())
            record = {
                "id": "1",
                "source": "fake",
                "audio_path": audio_path.name,
                "audio_sha256": hashlib.sha256(
                    audio_path.read_bytes()
                ).hexdigest(),
            }
            cache = stage.FeatureCache(
                root / "cache",
                strict_audio_integrity=True,
                speech_audio_data_dir=data_dir,
            )
            with patch.object(
                stage,
                "audio_features_from_paths",
                return_value=torch.zeros(1, 3, 2),
            ):
                cache.audio_batch(
                    None,
                    speech,
                    [record],
                    torch.device("cpu"),
                    16000,
                    50,
                )
                audio_path.unlink()
                audio_path.symlink_to(external_path)
                with self.assertRaisesRegex(ValueError, "symlink component"):
                    cache.audio_batch(
                        None,
                        speech,
                        [record],
                        torch.device("cpu"),
                        16000,
                        50,
                    )

    def test_cache_identity_covers_runtime_and_encoder_state(self) -> None:
        speech = FakeSpeechModel()
        speech.requires_grad_(False)
        base = stage.encoder_cache_identity(speech, 16000, 50, 0.0)
        identities = {
            stage.encoder_cache_identity(speech, 8000, 50, 0.0),
            stage.encoder_cache_identity(speech, 16000, 25, 0.0),
            stage.encoder_cache_identity(speech, 16000, 50, 1.0),
        }
        speech.config._commit_hash = "rev-b"
        speech.config.revision = "rev-b"
        identities.add(stage.encoder_cache_identity(speech, 16000, 50, 0.0))
        next(speech.encoder.parameters()).requires_grad_(True)
        identities.add(stage.encoder_cache_identity(speech, 16000, 50, 0.0))

        self.assertNotIn(base, identities)
        self.assertEqual(len(identities), 5)

    def test_cache_identity_covers_processor_feature_extractor_and_tokenizer(self) -> None:
        speech = FakeSpeechModel()
        speech.requires_grad_(False)
        base = stage.encoder_cache_identity(
            speech,
            16000,
            50,
            0.0,
            speech_processor=FakeSpeechProcessor("normalized", "base"),
        )
        identities = {
            stage.encoder_cache_identity(
                speech,
                16000,
                50,
                0.0,
                speech_processor=FakeSpeechProcessor("raw", "base"),
            ),
            stage.encoder_cache_identity(
                speech,
                16000,
                50,
                0.0,
                speech_processor=FakeSpeechProcessor("normalized", "alternate"),
            ),
        }

        self.assertNotIn(base, identities)
        self.assertEqual(len(identities), 2)

    def test_cache_is_disabled_for_trainable_or_changed_encoder(self) -> None:
        speech = FakeSpeechModel()
        speech.requires_grad_(False)
        features = torch.ones(1, 2, 2)

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "fake.wav"
            audio_path.write_bytes(b"deterministic fake WAV bytes")
            audio_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
            record = {
                "id": "1", "source": "fake", "audio_path": str(audio_path),
                "audio_sha256": audio_sha256,
            }
            cache = stage.FeatureCache(Path(directory) / "cache")
            with (
                patch.object(
                    stage,
                    "encoder_weights_cache_identity",
                    wraps=stage.encoder_weights_cache_identity,
                ) as weights_identity,
                patch.object(
                    stage, "speech_processor_state_token",
                    wraps=stage.speech_processor_state_token,
                ) as processor_token,
                patch.object(
                    stage, "audio_features_from_paths", return_value=features
                ) as extractor,
            ):
                cache.audio_batch(None, speech, [record], torch.device("cpu"), 16000, 50, 0.0)
                cache.audio_batch(None, speech, [record], torch.device("cpu"), 16000, 50, 0.0)
                self.assertEqual(extractor.call_count, 1)
                self.assertEqual(weights_identity.call_count, 1)
                self.assertEqual(processor_token.call_count, 2)
                self.assertEqual(len(list((Path(directory) / "cache").rglob("*.pt"))), 1)

                speech.config._commit_hash = "rev-b"
                speech.config.revision = "rev-b"
                cache.audio_batch(None, speech, [record], torch.device("cpu"), 16000, 50, 0.0)
                self.assertEqual(extractor.call_count, 2)
                self.assertEqual(
                    len(list((Path(directory) / "cache").rglob("*.pt"))), 1
                )

            trainable_cache = stage.FeatureCache(Path(directory) / "trainable")
            next(speech.encoder.parameters()).requires_grad_(True)
            with patch.object(stage, "audio_features_from_paths", return_value=features):
                trainable_cache.audio_batch(None, speech, [record], torch.device("cpu"), 16000, 50, 0.0)
            self.assertEqual(list((Path(directory) / "trainable").rglob("*.pt")), [])

    def test_changed_encoder_weights_cannot_hit_self_consistent_stale_cache(self) -> None:
        speech = FakeSpeechModel()
        speech.requires_grad_(False)
        processor = FakeSpeechProcessor("normalized")
        first_features = torch.ones(1, 2, 2)
        second_features = torch.full((1, 2, 2), 2.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "sample.wav"
            audio_path.write_bytes(b"stable WAV bytes")
            record = {
                "id": "1",
                "source": "fake",
                "audio_path": str(audio_path),
                "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            }
            with patch.object(
                stage,
                "audio_features_from_paths",
                side_effect=[first_features, second_features],
            ) as extractor:
                stage.FeatureCache(root / "cache").audio_batch(
                    processor,
                    speech,
                    [record],
                    torch.device("cpu"),
                    16000,
                    50,
                )
                stale_path = next((root / "cache").rglob("*.pt"))
                stale_payload = torch.load(
                    stale_path, map_location="cpu", weights_only=True
                )
                self.assertEqual(
                    stale_payload["tensor_identity"],
                    stage.tensor_content_identity(stale_payload["tensor"]),
                )
                with torch.no_grad():
                    next(speech.encoder.parameters()).add_(1.0)
                actual = stage.FeatureCache(root / "cache").audio_batch(
                    processor,
                    speech,
                    [record],
                    torch.device("cpu"),
                    16000,
                    50,
                )

            self.assertEqual(extractor.call_count, 2)
            self.assertTrue(torch.equal(actual, second_features))
            self.assertEqual(len(list((root / "cache").rglob("*.pt"))), 2)

    def test_in_place_processor_config_mutation_cannot_hit_self_consistent_stale_cache(self) -> None:
        speech = FakeSpeechModel()
        speech.requires_grad_(False)
        processor = FakeSpeechProcessor("normalized")
        first_features = torch.ones(1, 2, 2)
        second_features = torch.full((1, 2, 2), 2.0)
        third_features = torch.full((1, 2, 2), 3.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "sample.wav"
            audio_path.write_bytes(b"stable WAV bytes")
            record = {
                "id": "1",
                "source": "fake",
                "audio_path": str(audio_path),
                "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            }
            cache = stage.FeatureCache(root / "cache")
            with (
                patch.object(
                    stage,
                    "encoder_weights_cache_identity",
                    wraps=stage.encoder_weights_cache_identity,
                ) as weights_identity,
                patch.object(
                    stage,
                    "speech_processor_state_token",
                    wraps=stage.speech_processor_state_token,
                ) as processor_token,
                patch.object(
                    stage,
                    "audio_features_from_paths",
                    side_effect=[first_features, second_features, third_features],
                ) as extractor,
            ):
                cache.audio_batch(
                    processor,
                    speech,
                    [record],
                    torch.device("cpu"),
                    16000,
                    50,
                )
                stale_path = next((root / "cache").rglob("*.pt"))
                stale_payload = torch.load(
                    stale_path, map_location="cpu", weights_only=True
                )
                self.assertEqual(
                    stale_payload["tensor_identity"],
                    stage.tensor_content_identity(stale_payload["tensor"]),
                )
                processor.feature_extractor.normalization = "raw"
                second_actual = cache.audio_batch(
                    processor,
                    speech,
                    [record],
                    torch.device("cpu"),
                    16000,
                    50,
                )
                self.assertTrue(torch.equal(second_actual, second_features))
                processor.feature_extractor.normalization = "normalized"
                actual = cache.audio_batch(
                    processor,
                    speech,
                    [record],
                    torch.device("cpu"),
                    16000,
                    50,
                )

            self.assertEqual(extractor.call_count, 3)
            self.assertEqual(weights_identity.call_count, 1)
            self.assertEqual(processor_token.call_count, 3)
            self.assertTrue(torch.equal(actual, third_features))
            self.assertEqual(len(list((root / "cache").rglob("*.pt"))), 2)

    def test_modified_wav_fails_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.wav"
            audio_path.write_bytes(b"original WAV bytes")
            expected = hashlib.sha256(audio_path.read_bytes()).hexdigest()
            stage.audio_file_snapshot(
                str(audio_path), expected, require_expected=True
            )
            audio_path.write_bytes(b"modified WAV bytes")
            with self.assertRaisesRegex(ValueError, "audio SHA256 mismatch"):
                stage.load_audio_file(
                    str(audio_path), expected, require_expected=True
                )

    def test_strict_cache_ignores_preseed_and_never_reads_or_writes_payloads(self) -> None:
        speech = FakeSpeechModel()
        speech.requires_grad_(False)
        preseeded = torch.ones(1, 2, 2)
        strict_first = torch.full((1, 2, 2), 2.0)
        strict_second = torch.full((1, 2, 2), 3.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "sample.wav"
            audio_path.write_bytes(b"stable WAV bytes")
            record = {
                "id": "1",
                "source": "fake",
                "audio_path": str(audio_path),
                "audio_sha256": hashlib.sha256(
                    audio_path.read_bytes()
                ).hexdigest(),
            }
            cache_root = root / "cache"
            with patch.object(
                stage, "audio_features_from_paths", return_value=preseeded
            ):
                stage.FeatureCache(cache_root).audio_batch(
                    None, speech, [record], torch.device("cpu"), 16000, 50
                )
            preseed_paths = list(cache_root.rglob("*.pt"))
            self.assertEqual(len(preseed_paths), 1)

            cache = stage.FeatureCache(
                cache_root,
                strict_audio_integrity=True,
                speech_audio_data_dir=root,
            )
            self.assertEqual(
                cache.speech_feature_cache_policy,
                stage.STRICT_SPEECH_FEATURE_CACHE_POLICY,
            )
            with (
                patch.object(
                    stage,
                    "audio_features_from_paths",
                    side_effect=[strict_first, strict_second],
                ) as extractor,
                patch.object(
                    stage.FeatureCache,
                    "_load_audio_payload",
                    side_effect=AssertionError("strict cache read"),
                ) as cache_load,
                patch.object(
                    stage.FeatureCache,
                    "_save_audio_payload",
                    side_effect=AssertionError("strict cache write"),
                ) as cache_save,
            ):
                first = cache.audio_batch(
                    None, speech, [record], torch.device("cpu"), 16000, 50
                )
                second = cache.audio_batch(
                    None, speech, [record], torch.device("cpu"), 16000, 50
                )

            self.assertTrue(torch.equal(first, strict_first))
            self.assertTrue(torch.equal(second, strict_second))
            self.assertEqual(extractor.call_count, 2)
            for call in extractor.call_args_list:
                self.assertEqual(
                    call.kwargs["audio_payloads"], [audio_path.read_bytes()]
                )
                self.assertTrue(call.kwargs["require_expected_sha256"])
            cache_load.assert_not_called()
            cache_save.assert_not_called()
            self.assertEqual(list(cache_root.rglob("*.pt")), preseed_paths)


class WhisperUnfreezeTests(unittest.TestCase):
    def test_only_selected_encoder_parts_are_trainable_at_low_lr(self) -> None:
        wrapper = FakeWrapper()
        speech = FakeSpeechModel()
        optimizer, meta = stage.configure_trainable(
            wrapper, False, False, False, 2e-4, 5e-5, 5e-5, 0.0, 1e-5, 0.01,
            speech_model=speech,
            speech_unfreeze_last_blocks=1,
            speech_unfreeze_layer_norm=True,
            speech_encoder_lr=7e-6,
        )

        trainable_names = {name for name, param in speech.encoder.named_parameters() if param.requires_grad}
        self.assertTrue(trainable_names)
        self.assertTrue(all(name.startswith("layers.3.") or name.startswith("layer_norm.") for name in trainable_names))
        self.assertFalse(any(param.requires_grad for param in speech.encoder.input_projection.parameters()))
        group = next(group for group in optimizer.param_groups if group["name"] == "speech_encoder_partial")
        self.assertEqual(group["lr"], 7e-6)
        self.assertEqual(set(meta["speech_encoder_trainable_names"]), trainable_names)

    def test_selected_speech_state_round_trips(self) -> None:
        wrapper = FakeWrapper()
        speech = FakeSpeechModel()
        _, meta = stage.configure_trainable(
            wrapper, False, False, False, 2e-4, 5e-5, 5e-5, 0.0, 1e-5, 0.01,
            speech_model=speech,
            speech_unfreeze_last_blocks=1,
            speech_unfreeze_layer_norm=False,
        )
        expected = {
            name: param.detach().clone()
            for name, param in speech.encoder.named_parameters()
            if param.requires_grad
        }
        bank_provenance = {
            "bank_source": "reused_row_aligned_audio_raw_bank",
            "bank_rows": 5000,
            "row_identity_sha256": "b" * 64,
            "transcript_duplicate_keys_sha256": "c" * 64,
            "teacher_embedding_batch_size": 64,
            "teacher_embedding_batch_size_source": "speech_teacher_bank_batch_size",
            "eval_batch_size_independent": True,
            "bank_tensor_identity": {
                "sha256": "1" * 64,
                "dtype": "torch.float32",
                "shape": [5000, 2048],
                "bytes": 40960000,
            },
            "reusable_bank_input_identity": {
                "sha256": "2" * 64,
                "dtype": "torch.float32",
                "shape": [5000, 2048],
                "bytes": 40960000,
            },
            "teacher_model_identity": {
                "identity_sha256": "3" * 64,
                "weights_identity_sha256": "4" * 64,
            },
            "tokenizer_identity": {
                "identity_sha256": "5" * 64,
                "revision": "6" * 40,
                "config_sha256": "7" * 64,
                "vocab_sha256": "8" * 64,
            },
            "speech_train_file": {
                "path": "/tmp/development/speech_train.jsonl",
                "sha256": "d" * 64,
                "rows": 5000,
            },
            "speech_source_file": {
                "path": "/tmp/development/speech_source.jsonl",
                "sha256": "9" * 64,
                "rows": 5250,
            },
            "development_split_manifest": {
                "path": "/tmp/development/manifest.json",
                "sha256": "e" * 64,
                "source_commit_sha": "f" * 40,
                "builder": {
                    "path": "/tmp/repo/scripts/materialize_eval_splits.py",
                    "sha256": "a" * 64,
                    "source_commit_sha": "f" * 40,
                    "source_matches_commit": True,
                    "source_commit_exists": True,
                    "command": "python scripts/materialize_eval_splits.py",
                    "current_bytes_match_commit": True,
                },
            },
        }
        shared_provenance = {
            "enabled": True,
            "objective": "student_to_full_transcript_teacher_train_bank_infonce",
            "teacher_bank": bank_provenance,
        }
        meta["speech_shared_contrastive_provenance"] = shared_provenance
        meta["speech_shared_teacher_bank_provenance"] = bank_provenance
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = types.SimpleNamespace(
                save_expert_weights=False,
                data_dir=str(root / "data"),
                output_dir=str(root / "output"),
                final_steps=1,
                alignment_pretrain_steps=0,
                sample_rate=16000,
                speech_feature_cache_policy=(
                    stage.STRICT_SPEECH_FEATURE_CACHE_POLICY
                ),
            )
            gamma_path = (
                Path(args.output_dir) / "calibration" / "gamma.json"
            )
            gamma_path.parent.mkdir(parents=True)
            gamma_path.write_text(
                json.dumps({"gamma": [1.0]}), encoding="utf-8"
            )
            path = root / "checkpoint.pt"
            environment = {
                "SOURCE_COMMIT_SHA": "a" * 40,
                "RUNAI_JOB_NAME": "stage-a-test",
                "RUNAI_PROJECT": "test-project",
            }
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(stage, "dynamic_expert_bias_state_dict", return_value={}),
            ):
                stage.save_checkpoint(
                    wrapper, path, meta, args, {"step": 1}, speech_model=speech
                )
            for name, param in speech.encoder.named_parameters():
                if name in expected:
                    with torch.no_grad():
                        param.add_(10.0)
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(
                state["gamma_provenance"]["path"],
                str(gamma_path.resolve()),
            )
            self.assertEqual(
                state["gamma_provenance"]["sha256"],
                stage.sha256_file(gamma_path),
            )
            self.assertEqual(state["run_provenance"]["source_commit_sha"], "a" * 40)
            self.assertEqual(state["run_provenance"]["runai_job_name"], "stage-a-test")
            self.assertEqual(state["run_provenance"]["checkpoint_completed_step"], 1)
            self.assertFalse(state["run_provenance"]["sealed_evidence_used"])
            self.assertFalse(state["run_provenance"]["synthetic_evidence_used"])
            self.assertEqual(
                state["run_provenance"]["speech_feature_cache_policy"],
                stage.STRICT_SPEECH_FEATURE_CACHE_POLICY,
            )
            self.assertEqual(
                state["speech_shared_contrastive_provenance"],
                shared_provenance,
            )
            self.assertEqual(
                state["speech_shared_teacher_bank_provenance"],
                bank_provenance,
            )
            self.assertEqual(
                state["trainable_meta"]["speech_shared_teacher_bank_provenance"],
                bank_provenance,
            )
            with patch.object(stage, "load_dynamic_expert_bias_state"):
                stage.restore_training_checkpoint(wrapper, state, speech_model=speech)

        actual = dict(speech.encoder.named_parameters())
        for name, value in expected.items():
            self.assertTrue(torch.equal(actual[name], value))



class FakeStageBLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = types.SimpleNamespace(
            gate=nn.Linear(2, 4, bias=False),
            gamma_scale=nn.Parameter(torch.tensor(1.0)),
        )


class FakeStageBLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_embeddings = nn.Embedding(4, 2)
        self.output_embeddings = nn.Linear(2, 4, bias=False)
        self.model = types.SimpleNamespace(
            layers=nn.ModuleList([FakeStageBLayer(), FakeStageBLayer()])
        )

    def get_output_embeddings(self):
        return self.output_embeddings

    def get_input_embeddings(self):
        return self.input_embeddings


class StageBInitializationTests(unittest.TestCase):
    def make_state(self, source: FakeStageBLM) -> dict:
        args = {"student_top_k": 2}
        trainable = {
            "train_router_gates": True,
            "train_lm_head": True,
            "train_gamma_scale": True,
        }
        parameter_names = ["output_embeddings.weight"]
        optimizer_groups = [{
            "lr": 1e-5,
            "parameter_count": 1,
            "parameter_names": parameter_names,
        }]
        data_contract = {
            "schema_version": 1,
            "canonical_data_root": "/tmp/development",
            "jsonl_inputs": [{
                "role": "train",
                "canonical_path": "/tmp/development/train.jsonl",
                "sha256": "a" * 64,
                "size_bytes": 1,
                "row_count": 1,
            }],
        }
        model_identity = {
            "requested_base_model": "fake/base",
            "commit_hash": "fixture-revision",
        }
        curriculum_plan = {
            "schedule": [8, 4, 2],
            "reference_target_steps": 500,
        }
        resume_contract = {
            "schema_version": 2,
            "args": args,
            "trainable": trainable,
            "optimizer_groups": optimizer_groups,
            "data": data_contract,
            "base_model_identity": model_identity,
            "curriculum_plan": curriculum_plan,
        }
        for field in (
            "args",
            "trainable",
            "optimizer_groups",
            "data",
            "base_model_identity",
            "curriculum_plan",
        ):
            resume_contract[f"{field}_sha256"] = hashlib.sha256(
                json.dumps(
                    resume_contract[field],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest()
        return {
            "checkpoint_version": 2,
            "final_inference_top_k": 2,
            "completed_steps": 500,
            "args": args,
            "trainable_meta": trainable,
            "resume_contract": resume_contract,
            "curriculum_plan": curriculum_plan,
            "optimizer_state": {
                "state": {},
                "param_groups": [{
                    "lr": 1e-5,
                    "param_names": parameter_names,
                    "params": [0],
                }],
            },
            "rng_state": {"fixture": True},
            "provenance": {
                "stage": "B",
                "development_data_only": True,
                "teacher_top_k": 8,
                "final_inference_top_k": 2,
                "base_model": "fake/base",
            },
            "router_gates": {
                f"layer_{index}": layer.mlp.gate.state_dict()
                for index, layer in enumerate(source.model.layers)
            },
            "lm_output_embeddings": source.output_embeddings.state_dict(),
            "lm_input_embeddings": source.input_embeddings.state_dict(),
            "lm_embeddings_tied": False,
            "gamma_scale": [1.5, 1.75],
        }

    def test_stage_b_top2_state_initializes_lm_without_bridge_keys(self) -> None:
        source = FakeStageBLM()
        target = FakeStageBLM()
        with torch.no_grad():
            source.output_embeddings.weight.fill_(0.25)
            source.input_embeddings.weight.fill_(0.5)
            for layer in source.model.layers:
                layer.mlp.gate.weight.fill_(0.75)
        wrapper = types.SimpleNamespace(lm=target)

        state = self.make_state(source)
        metadata = stage.restore_stage_b_student_initialization(
            wrapper,
            state,
            "fake/base",
            state["resume_contract"]["base_model_identity"],
        )

        self.assertTrue(metadata["router_state_restored"])
        self.assertTrue(metadata["lm_head_state_restored"])
        self.assertTrue(metadata["gamma_state_restored"])
        self.assertEqual(metadata["final_inference_top_k"], 2)
        self.assertTrue(
            torch.equal(target.output_embeddings.weight, source.output_embeddings.weight)
        )
        self.assertTrue(
            torch.equal(
                target.model.layers[0].mlp.gate.weight,
                source.model.layers[0].mlp.gate.weight,
            )
        )
        self.assertEqual(
            [
                float(layer.mlp.gamma_scale.detach())
                for layer in target.model.layers
            ],
            [1.5, 1.75],
        )

    def test_stage_b_initialization_rejects_pre_routing_identity_drift(self) -> None:
        source = FakeStageBLM()
        target = FakeStageBLM()
        state = self.make_state(source)
        runtime_identity = dict(
            state["resume_contract"]["base_model_identity"]
        )
        runtime_identity["commit_hash"] = "drifted-revision"
        original_router = target.model.layers[0].mlp.gate.weight.detach().clone()

        with self.assertRaisesRegex(
            ValueError, "pre-routing base-model identity differs"
        ):
            stage.restore_stage_b_student_initialization(
                types.SimpleNamespace(lm=target),
                state,
                "fake/base",
                runtime_identity,
            )
        self.assertTrue(
            torch.equal(
                target.model.layers[0].mlp.gate.weight, original_router
            )
        )

    def test_stage_b_untied_checkpoint_requires_input_embeddings(self) -> None:
        source = FakeStageBLM()
        target = FakeStageBLM()
        state = self.make_state(source)
        del state["lm_input_embeddings"]
        wrapper = types.SimpleNamespace(lm=target)
        with self.assertRaisesRegex(
            ValueError, "missing input embeddings"
        ):
            stage.restore_stage_b_student_initialization(
                wrapper,
                state,
                "fake/base",
                state["resume_contract"]["base_model_identity"],
            )

    def test_stage_b_trainable_gamma_requires_gamma_state(self) -> None:
        source = FakeStageBLM()
        target = FakeStageBLM()
        state = self.make_state(source)
        del state["gamma_scale"]
        wrapper = types.SimpleNamespace(lm=target)
        with self.assertRaisesRegex(
            ValueError, "missing gamma state"
        ):
            stage.restore_stage_b_student_initialization(
                wrapper,
                state,
                "fake/base",
                state["resume_contract"]["base_model_identity"],
            )

    def test_stage_b_checkpoint_loader_requires_exact_sha_and_rejects_sealed_path(self) -> None:
        source = FakeStageBLM()
        state = self.make_state(source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage_b.pt"
            torch.save(state, path)
            digest = stage.sha256_file(path)
            original_torch_load = torch.load
            with patch.object(
                stage.torch, "load", wraps=original_torch_load
            ) as load_mock:
                loaded, provenance = stage.load_stage_b_initialization_checkpoint(
                    str(path), digest
                )
            self.assertTrue(load_mock.call_args.kwargs["weights_only"])
            self.assertIsInstance(load_mock.call_args.args[0], io.BytesIO)
            self.assertEqual(loaded["completed_steps"], 500)
            self.assertEqual(provenance["sha256"], digest)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                stage.load_stage_b_initialization_checkpoint(str(path), "0" * 64)

            sealed = Path(directory) / "sealed_stage_b.pt"
            torch.save(state, sealed)
            with self.assertRaisesRegex(ValueError, "forbidden"):
                stage.load_stage_b_initialization_checkpoint(
                    str(sealed), stage.sha256_file(sealed)
                )

    def test_stage_b_loader_hashes_and_loads_the_same_bytes(self) -> None:
        source = FakeStageBLM()
        state = self.make_state(source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage_b.pt"
            torch.save(state, path)
            digest = stage.sha256_file(path)
            original_size = path.stat().st_size
            original_torch_load = torch.load

            def mutate_path_then_load(source_bytes, *args, **kwargs):
                path.write_bytes(b"replaced-after-read")
                return original_torch_load(source_bytes, *args, **kwargs)
            with patch.object(
                stage.torch,
                "load",
                side_effect=mutate_path_then_load,
            ):
                loaded, provenance = (
                    stage.load_stage_b_initialization_checkpoint(
                        str(path), digest
                    )
                )
            self.assertEqual(loaded["completed_steps"], 500)
            self.assertEqual(provenance["sha256"], digest)
            self.assertEqual(provenance["size_bytes"], original_size)
            self.assertEqual(path.read_bytes(), b"replaced-after-read")


    def test_image_scoped_restore_preserves_unselected_audio_bridge(self) -> None:
        source = FakeWrapper()
        target = FakeWrapper()
        with torch.no_grad():
            source.image_resampler.weight.fill_(0.25)
            source.audio_resampler.weight.fill_(0.75)
            target.image_resampler.weight.zero_()
            target.audio_resampler.weight.fill_(0.5)
        audio_before = target.audio_resampler.weight.detach().clone()
        state = {
            "image_resampler": source.image_resampler.state_dict(),
            "audio_resampler": source.audio_resampler.state_dict(),
        }

        metadata = stage.restore_scoped_multimodal_checkpoint(
            target, state, "image"
        )

        self.assertTrue(metadata["image_restored"])
        self.assertFalse(metadata["speech_restored"])
        self.assertTrue(
            torch.equal(target.image_resampler.weight, source.image_resampler.weight)
        )
        self.assertTrue(torch.equal(target.audio_resampler.weight, audio_before))

    def test_dual_scoped_restore_keeps_image_lm_router_and_experts_isolated(self) -> None:
        image_source = FakeWrapper()
        speech_source = FakeWrapper()
        target = FakeWrapper()
        source_speech_model = FakeSpeechModel()
        target_speech_model = FakeSpeechModel()
        with torch.no_grad():
            image_source.image_resampler.weight.fill_(0.25)
            image_source.audio_resampler.weight.fill_(0.35)
            speech_source.image_resampler.weight.fill_(0.65)
            speech_source.audio_resampler.weight.fill_(0.75)
            speech_source.audio_retrieval_head.weight.fill_(0.95)
            for name, parameter in source_speech_model.encoder.named_parameters():
                if name.startswith("layers.3.") or name.startswith("layer_norm."):
                    parameter.fill_(0.85)
            target.image_resampler.weight.zero_()
            target.audio_resampler.weight.zero_()
        lm_before = target.lm.embedding.weight.detach().clone()
        audio_retrieval_before = target.audio_retrieval_head.weight.detach().clone()
        speech_state = {
            "image_resampler": speech_source.image_resampler.state_dict(),
            "audio_resampler": speech_source.audio_resampler.state_dict(),
            "audio_retrieval_head": speech_source.audio_retrieval_head.state_dict(),
            "speech_encoder_trainable_state": {
                name: parameter.detach().clone()
                for name, parameter in source_speech_model.encoder.named_parameters()
                if name.startswith("layers.3.") or name.startswith("layer_norm.")
            },
            "lm_input_embeddings": {"weight": torch.full_like(lm_before, 9.0)},
            "router_gates": {"layer_0": {"weight": torch.ones(1)}},
            "experts": {"layer_0": {"weight": torch.ones(1)}},
        }
        image_state = {
            "image_resampler": image_source.image_resampler.state_dict(),
            "audio_resampler": image_source.audio_resampler.state_dict(),
        }

        stage.validate_dual_initialization_request(image_state, "image", speech_state)
        image_metadata = stage.restore_scoped_multimodal_checkpoint(
            target, image_state, "image"
        )
        speech_metadata = stage.restore_speech_initialization_checkpoint(
            target,
            speech_state,
            target_speech_model,
        )

        self.assertTrue(image_metadata["image_restored"])
        self.assertTrue(speech_metadata["speech_restored"])
        self.assertTrue(speech_metadata["speech_encoder_state_restored"])
        self.assertTrue(speech_metadata["lm_router_expert_state_ignored"])
        self.assertTrue(speech_metadata["audio_retrieval_state_ignored"])
        self.assertTrue(
            torch.equal(
                target.image_resampler.weight, image_source.image_resampler.weight
            )
        )
        self.assertTrue(
            torch.equal(
                target.audio_resampler.weight, speech_source.audio_resampler.weight
            )
        )
        self.assertTrue(torch.equal(target.lm.embedding.weight, lm_before))
        self.assertTrue(
            torch.equal(target.audio_retrieval_head.weight, audio_retrieval_before)
        )
        target_speech = dict(target_speech_model.encoder.named_parameters())
        for name, expected in speech_state["speech_encoder_trainable_state"].items():
            self.assertTrue(torch.equal(target_speech[name], expected))

    def test_dual_initialization_rejects_non_image_primary_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "SCOPE=image"):
            stage.validate_dual_initialization_request({}, "speech", {})

    def make_multimodal_artifacts(self, root: Path) -> tuple[Path, Path, str, dict, dict]:
        wrapper = FakeWrapper()
        data_root = (root / "real_data").resolve()
        output_root = (root / "output").resolve()
        data_root.mkdir()
        checkpoint = output_root / "E3_final_multimodal_top2" / "checkpoint_final.pt"
        checkpoint.parent.mkdir(parents=True)
        args = {
            "data_dir": str(data_root),
            "output_dir": str(output_root),
            "final_steps": 500,
            "alignment_pretrain_steps": 400,
        }
        run_provenance = {
            "source_commit_sha": "a" * 40,
            "runai_job_name": "stage-a-job",
            "runai_project": "stage-a-project",
            "resolved_data_root": str(data_root),
            "resolved_output_root": str(output_root),
            "final_main_steps": 500,
            "alignment_pretrain_steps": 400,
            "checkpoint_completed_step": 500,
            "policy": stage.STAGE_A_PROVENANCE_POLICY,
            "sealed_evidence_used": False,
            "synthetic_evidence_used": False,
        }
        state = {
            "image_resampler": wrapper.image_resampler.state_dict(),
            "audio_resampler": wrapper.audio_resampler.state_dict(),
            "args": dict(args),
            "last_row": {"step": 500},
            "run_provenance": dict(run_provenance),
        }
        torch.save(state, checkpoint)
        digest = stage.sha256_file(checkpoint)
        manifest = {
            "runai_job_name": "stage-a-job",
            "runai_project": "stage-a-project",
            "source_commit_sha": "a" * 40,
            "data_dir": str(data_root),
            "output_dir": str(output_root),
            "args": dict(args),
            "run_provenance": dict(run_provenance),
            "completion": {
                "status": "completed",
                "e3_checkpoint_path": str(checkpoint.resolve()),
                "e3_checkpoint_sha256": digest,
                "e3_steps": 500,
            },
        }
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return checkpoint, manifest_path, digest, state, manifest

    def test_multimodal_loader_verifies_checkpoint_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, manifest_path, digest, _, _ = self.make_multimodal_artifacts(
                Path(directory)
            )
            loaded, provenance = stage.load_multimodal_initialization_checkpoint(
                str(checkpoint), digest, str(manifest_path)
            )

            self.assertIn("image_resampler", loaded)
            self.assertEqual(provenance["sha256"], digest)
            self.assertEqual(provenance["manifest_path"], str(manifest_path.resolve()))
            self.assertEqual(
                provenance["manifest_sha256"], stage.sha256_file(manifest_path)
            )
            self.assertEqual(provenance["source_commit_sha"], "a" * 40)
            self.assertEqual(provenance["runai_job_name"], "stage-a-job")
            self.assertEqual(provenance["completion_status"], "completed")
            self.assertEqual(provenance["completion_step"], 500)

    def test_speech_companion_loader_requires_speech_last1_ln_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, manifest_path, _, state, manifest = (
                self.make_multimodal_artifacts(Path(directory))
            )
            speech_model = FakeSpeechModel()
            speech_state = {
                name: parameter.detach().clone()
                for name, parameter in speech_model.encoder.named_parameters()
                if name.startswith("layers.3.") or name.startswith("layer_norm.")
            }
            speech_args = {
                "alignment_pretrain_modalities": "speech",
                "speech_unfreeze_last_blocks": 1,
                "speech_unfreeze_layer_norm": True,
            }
            state["speech_encoder_trainable_state"] = speech_state
            state["args"].update(speech_args)
            manifest["args"].update(speech_args)
            torch.save(state, checkpoint)
            digest = stage.sha256_file(checkpoint)
            manifest["completion"]["e3_checkpoint_sha256"] = digest
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            loaded, provenance = stage.load_speech_initialization_checkpoint(
                str(checkpoint), digest, str(manifest_path)
            )

            self.assertIn("speech_encoder_trainable_state", loaded)
            self.assertEqual(provenance["scope"], "speech")
            self.assertEqual(provenance["path"], str(checkpoint.resolve()))
            self.assertEqual(provenance["sha256"], digest)
            self.assertEqual(
                provenance["manifest_sha256"], stage.sha256_file(manifest_path)
            )
            self.assertEqual(provenance["source_commit_sha"], "a" * 40)
            self.assertEqual(provenance["runai_job_name"], "stage-a-job")

            state["args"]["alignment_pretrain_modalities"] = "image"
            manifest["args"]["alignment_pretrain_modalities"] = "image"
            torch.save(state, checkpoint)
            digest = stage.sha256_file(checkpoint)
            manifest["completion"]["e3_checkpoint_sha256"] = digest
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "scope mismatch for alignment_pretrain_modalities"
            ):
                stage.load_speech_initialization_checkpoint(
                    str(checkpoint), digest, str(manifest_path)
                )

    def test_multimodal_loader_requires_manifest_and_exact_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, manifest_path, digest, _, _ = self.make_multimodal_artifacts(
                Path(directory)
            )
            with self.assertRaisesRegex(ValueError, "requires.*manifest"):
                stage.load_multimodal_initialization_checkpoint(
                    str(checkpoint), digest, ""
                )

            copied = Path(directory) / "copied_checkpoint.pt"
            shutil.copy2(checkpoint, copied)
            self.assertEqual(stage.sha256_file(copied), digest)
            with self.assertRaisesRegex(ValueError, "exact checkpoint"):
                stage.load_multimodal_initialization_checkpoint(
                    str(copied), digest, str(manifest_path)
                )

    def test_multimodal_loader_rejects_manifest_sha_source_and_completion_drift(self) -> None:
        mutations = (
            (
                "mismatched completion SHA",
                lambda manifest: manifest["completion"].update(
                    {"e3_checkpoint_sha256": "0" * 64}
                ),
                "completion SHA256",
            ),
            (
                "absent source commit",
                lambda manifest: manifest.pop("source_commit_sha"),
                "source commit",
            ),
            (
                "absent completion",
                lambda manifest: manifest.pop("completion"),
                "completion.status",
            ),
        )
        for label, mutate, error in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                checkpoint, manifest_path, digest, _, manifest = (
                    self.make_multimodal_artifacts(Path(directory))
                )
                mutate(manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    stage.load_multimodal_initialization_checkpoint(
                        str(checkpoint), digest, str(manifest_path)
                    )

    def test_multimodal_loader_rejects_checkpoint_without_embedded_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, manifest_path, _, state, manifest = (
                self.make_multimodal_artifacts(Path(directory))
            )
            state.pop("run_provenance")
            torch.save(state, checkpoint)
            digest = stage.sha256_file(checkpoint)
            manifest["completion"]["e3_checkpoint_sha256"] = digest
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing run_provenance"):
                stage.load_multimodal_initialization_checkpoint(
                    str(checkpoint), digest, str(manifest_path)
                )

    def test_multimodal_loader_rejects_checkpoint_manifest_disagreement(self) -> None:
        mutations = (
            (
                lambda manifest: manifest["run_provenance"].update(
                    {"runai_job_name": "different-job"}
                ),
                "run_provenance disagree",
            ),
            (
                lambda manifest: manifest["args"].update({"final_steps": 499}),
                "args disagree on final_steps",
            ),
        )
        for mutate, error in mutations:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                checkpoint, manifest_path, digest, _, manifest = (
                    self.make_multimodal_artifacts(Path(directory))
                )
                mutate(manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    stage.load_multimodal_initialization_checkpoint(
                        str(checkpoint), digest, str(manifest_path)
                    )

    def test_multimodal_loader_requires_bridge_states_and_disjoint_lm_state(self) -> None:
        wrapper = FakeWrapper()
        multimodal = {
            "image_resampler": wrapper.image_resampler.state_dict(),
            "audio_resampler": wrapper.audio_resampler.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            _, manifest_path, _, state, _ = self.make_multimodal_artifacts(
                Path(directory)
            )
            state.pop("audio_resampler")
            broken = Path(directory) / "stage_a_broken.pt"
            torch.save(state, broken)
            with self.assertRaisesRegex(ValueError, "missing bridge states"):
                stage.load_multimodal_initialization_checkpoint(
                    str(broken), stage.sha256_file(broken), str(manifest_path)
                )

        stage.validate_initialization_state_disjoint(
            multimodal, self.make_state(FakeStageBLM())
        )
        multimodal["router_gates"] = {}
        with self.assertRaisesRegex(ValueError, "overlapping LM state"):
            stage.validate_initialization_state_disjoint(
                multimodal, self.make_state(FakeStageBLM())
            )
        stage.validate_initialization_state_disjoint(
            multimodal, self.make_state(FakeStageBLM()), "image"
        )

    def test_stage_a_run_provenance_fails_closed_for_malformed_runai(self) -> None:
        args = types.SimpleNamespace(
            data_dir="data/real",
            output_dir="outputs/development",
            final_steps=500,
            alignment_pretrain_steps=400,
        )
        environment = {
            "SOURCE_COMMIT_SHA": "short",
            "RUNAI_JOB_NAME": "stage-a-job",
            "RUNAI_PROJECT": "stage-a-project",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, "full 40-character"):
                stage.build_stage_a_run_provenance(args, 500)

    def test_stage_b_initialization_rejects_provenance_and_partial_router(self) -> None:
        source = FakeStageBLM()
        wrapper = types.SimpleNamespace(lm=FakeStageBLM())
        state = self.make_state(source)
        state["provenance"]["development_data_only"] = False
        with self.assertRaisesRegex(ValueError, "development_data_only"):
            stage.restore_stage_b_student_initialization(
                wrapper,
                state,
                "fake/base",
                state["resume_contract"]["base_model_identity"],
            )

        state = self.make_state(source)
        state["router_gates"].pop("layer_1")
        with self.assertRaisesRegex(ValueError, "every router layer"):
            stage.restore_stage_b_student_initialization(
                wrapper,
                state,
                "fake/base",
                state["resume_contract"]["base_model_identity"],
            )


class AlignmentPretrainTests(unittest.TestCase):
    def test_alignment_modalities_support_dedicated_stage_a_arms(self) -> None:
        self.assertEqual(stage.parse_alignment_pretrain_modalities("image"), ["image"])
        self.assertEqual(stage.parse_alignment_pretrain_modalities("speech"), ["speech"])
        self.assertEqual(
            stage.parse_alignment_pretrain_modalities("image,speech"),
            ["image", "speech"],
        )
        with self.assertRaisesRegex(ValueError, "only image and/or speech"):
            stage.parse_alignment_pretrain_modalities("text")

    def test_pretrain_forces_bank_and_requires_trainable_bridge_or_encoder(self) -> None:
        args = types.SimpleNamespace(
            alignment_pretrain_steps=1,
            contrastive_coef=0.0,
            image_contrastive_coef=-1.0,
            speech_contrastive_coef=-1.0,
        )
        self.assertTrue(stage.needs_alignment_target_bank(args))

        wrapper = FakeWrapper()
        speech = FakeSpeechModel()
        wrapper.requires_grad_(False)
        speech.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "trainable bridge or speech encoder"):
            stage.validate_alignment_pretrain_trainable(wrapper, speech)

    def test_bridge_grad_norms_report_both_modalities(self) -> None:
        wrapper = FakeWrapper()
        wrapper.requires_grad_(False)
        wrapper.image_resampler.weight.requires_grad_(True)
        wrapper.image_resampler.weight.grad = torch.ones_like(wrapper.image_resampler.weight)

        norms = stage.bridge_grad_norms(wrapper)

        self.assertEqual(norms["image_bridge_grad_norm"], 2.0)
        self.assertEqual(norms["audio_bridge_grad_norm"], 0.0)

    def test_cli_accepts_new_image_bridge_types(self) -> None:
        for bridge_type in ("local_pool_linear", "linear_projector_norm"):
            with self.subTest(bridge_type=bridge_type):
                argv = ["olmoe_real_subset_runs.py", "--image-bridge-type", bridge_type]
                with patch.object(sys, "argv", argv):
                    args = stage.parse_args()
                self.assertEqual(args.image_bridge_type, bridge_type)

    def test_new_cli_defaults_preserve_prior_behavior(self) -> None:
        with patch.object(sys, "argv", ["olmoe_real_subset_runs.py"]):
            args = stage.parse_args()

        self.assertEqual(args.image_alignment_target, "clip_text")
        self.assertEqual(args.image_bridge_type, "query_resampler")
        self.assertEqual(args.audio_bridge_type, "query_resampler")
        self.assertEqual(args.bridge_num_heads, 4)
        self.assertEqual(args.alignment_pretrain_modalities, "image,speech")
        self.assertEqual(args.audio_max_seconds, 0.0)
        self.assertEqual(args.speech_unfreeze_last_blocks, 0)
        self.assertFalse(args.speech_unfreeze_layer_norm)
        self.assertEqual(args.stage_b_checkpoint, "")
        self.assertEqual(args.stage_b_checkpoint_sha256, "")
        self.assertEqual(args.multimodal_initial_checkpoint, "")
        self.assertEqual(args.multimodal_initial_checkpoint_sha256, "")
        self.assertEqual(args.multimodal_initial_manifest, "")
        self.assertEqual(args.multimodal_initialization_scope, "both")
        self.assertEqual(args.speech_initial_checkpoint, "")
        self.assertEqual(args.speech_initial_checkpoint_sha256, "")
        self.assertEqual(args.speech_initial_manifest, "")


if __name__ == "__main__":
    unittest.main()
