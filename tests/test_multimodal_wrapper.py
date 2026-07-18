import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from model.fusion import (
    LocalPoolLinearPrefixBridge,
    NormalizedPrefixProjector,
    QueryResampler,
    make_prefix_bridge,
)

from model.olmoe_adapter import OLMoEMultimodalPrefixWrapper


class TinyFakeCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 17, hidden_size: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        self.last_inputs_embeds = None
        self.last_attention_mask = None
        self.last_labels = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask=None, labels=None, **_kwargs):
        self.last_inputs_embeds = inputs_embeds.detach().clone()
        self.last_attention_mask = attention_mask.detach().clone()
        self.last_labels = labels.detach().clone() if labels is not None else None
        self.last_kwargs = dict(_kwargs)
        hidden = torch.cumsum(inputs_embeds, dim=1)
        logits = self.output(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss)


class PrefixBridgeFactoryTest(unittest.TestCase):
    def test_fixed_shape_bridge_variants(self) -> None:
        torch.manual_seed(3)
        source = torch.randn(2, 7, 4)
        for bridge_type in ("query_resampler", "attention_pool", "temporal_resample"):
            with self.subTest(bridge_type=bridge_type):
                bridge = make_prefix_bridge(bridge_type, 4, 8, 3)
                first = bridge(source)
                second = bridge(source)
                self.assertEqual(tuple(first.shape), (2, 3, 8))
                self.assertTrue(torch.equal(first, second))

    def test_linear_and_identity_controls_require_matching_token_count(self) -> None:
        projected = make_prefix_bridge("linear_projector", 4, 8, 3)
        self.assertEqual(tuple(projected(torch.randn(2, 3, 4)).shape), (2, 3, 8))
        with self.assertRaisesRegex(ValueError, "does not resample tokens"):
            projected(torch.randn(2, 4, 4))

        identity = make_prefix_bridge("identity", 8, 8, 3)
        source = torch.randn(2, 3, 8)
        self.assertIs(identity(source), source)
        with self.assertRaisesRegex(ValueError, "cannot resample tokens"):
            identity(torch.randn(2, 4, 8))

    def test_identity_and_factory_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "input_dim == hidden_size"):
            make_prefix_bridge("identity", 4, 8, 3)
        with self.assertRaisesRegex(ValueError, "unsupported prefix bridge"):
            make_prefix_bridge("unknown", 4, 8, 3)

    def test_local_pool_linear_compresses_clip_grid_to_local_averages(self) -> None:
        bridge = make_prefix_bridge("local_pool_linear", 2, 2, 17).double()
        self.assertIsInstance(bridge, LocalPoolLinearPrefixBridge)
        with torch.no_grad():
            bridge.projection.weight.copy_(torch.eye(2))
            bridge.projection.bias.zero_()
        cls = torch.tensor([[[100.0, -100.0]]], dtype=torch.float64)
        patches = torch.arange(49, dtype=torch.float64).view(1, 49, 1).repeat(1, 1, 2)
        inputs = torch.cat((cls, patches), dim=1)

        first = bridge(inputs)
        second = bridge(inputs)

        expected_patch_values = torch.tensor(
            [
                4.0, 5.5, 7.5, 9.0,
                14.5, 16.0, 18.0, 19.5,
                28.5, 30.0, 32.0, 33.5,
                39.0, 40.5, 42.5, 44.0,
            ],
            dtype=torch.float64,
        ).view(1, 16, 1).repeat(1, 1, 2)
        expected = torch.cat((cls, expected_patch_values), dim=1)
        self.assertEqual(tuple(inputs.shape), (1, 50, 2))
        self.assertEqual(tuple(first.shape), (1, 17, 2))
        self.assertEqual(first.dtype, patches.dtype)
        self.assertEqual(first.device, patches.device)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, expected))
        metadata = bridge.audit_metadata()["observed_geometry"]
        self.assertEqual(metadata["input_tokens"], 50)
        self.assertEqual(metadata["output_tokens"], 17)
        self.assertEqual(metadata["input_grid"], [7, 7])
        self.assertEqual(metadata["output_grid"], [4, 4])
        self.assertEqual(metadata["cls_handling"], "preserve_first_token")

    def test_local_pool_linear_supports_grid_without_cls_and_fails_closed(self) -> None:
        bridge = make_prefix_bridge("local_pool_linear", 2, 3, 4)
        output = bridge(torch.randn(2, 16, 2))
        self.assertEqual(tuple(output.shape), (2, 4, 3))
        self.assertEqual(
            bridge.audit_metadata()["observed_geometry"]["cls_handling"],
            "no_cls_token",
        )
        with self.assertRaisesRegex(ValueError, "square image patch geometry"):
            bridge(torch.randn(2, 7, 2))
        with self.assertRaisesRegex(ValueError, "output patch count"):
            make_prefix_bridge("local_pool_linear", 2, 3, 4)(torch.randn(2, 17, 2))
        with self.assertRaisesRegex(ValueError, "does not upsample"):
            make_prefix_bridge("local_pool_linear", 2, 3, 26)(torch.randn(2, 17, 2))

    def test_local_pool_linear_checkpoint_round_trip(self) -> None:
        source = make_prefix_bridge("local_pool_linear", 3, 4, 17)
        inputs = torch.randn(2, 50, 3)
        expected = source(inputs)
        restored = make_prefix_bridge("local_pool_linear", 3, 4, 17)
        restored.load_state_dict(source.state_dict())
        self.assertTrue(torch.equal(restored(inputs), expected))

    def test_linear_projector_norm_shape_normalization_and_checkpoint(self) -> None:
        torch.manual_seed(11)
        source = make_prefix_bridge("linear_projector_norm", 4, 8, 50)
        self.assertIsInstance(source, NormalizedPrefixProjector)
        inputs = torch.randn(2, 50, 4)
        projected = source.projection(inputs)
        reference = F.layer_norm(
            projected,
            (8,),
            source.norm.weight,
            source.norm.bias,
            source.norm.eps,
        )
        expected = source(inputs)

        self.assertEqual(tuple(expected.shape), (2, 50, 8))
        self.assertTrue(torch.allclose(expected.mean(dim=-1), torch.zeros(2, 50), atol=1e-6))
        self.assertTrue(torch.equal(expected, reference))
        self.assertTrue(
            torch.allclose(
                expected.square().mean(dim=-1),
                torch.ones(2, 50),
                atol=3e-4,
            )
        )
        self.assertEqual(source.audit_metadata()["normalization"], "layer_norm")
        restored = make_prefix_bridge("linear_projector_norm", 4, 8, 50)
        restored.load_state_dict(source.state_dict())
        self.assertTrue(torch.equal(restored(inputs), expected))
        with self.assertRaisesRegex(ValueError, "does not resample tokens"):
            restored(torch.randn(2, 49, 4))


class MultimodalWrapperTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.lm = TinyFakeCausalLM()
        self.wrapper = OLMoEMultimodalPrefixWrapper(
            lm=self.lm,
            hidden_size=8,
            image_input_dim=4,
            audio_input_dim=6,
            image_prefix_tokens=2,
            audio_prefix_tokens=3,
            image_retrieval_dim=8,
            audio_retrieval_dim=8,
        )
        self.input_ids = torch.tensor([[2, 3, 4, 5]])
        self.attention_mask = torch.ones_like(self.input_ids)
        self.labels = torch.tensor([[-100, 3, 4, 5]])

    def test_default_bridge_preserves_checkpoint_schema(self) -> None:
        self.assertIsInstance(self.wrapper.image_resampler, QueryResampler)
        expected = QueryResampler(4, 8, 2).state_dict()
        self.assertEqual(
            tuple(self.wrapper.image_resampler.state_dict()),
            tuple(expected),
        )

    def test_wrapper_wires_identity_image_control(self) -> None:
        wrapper = OLMoEMultimodalPrefixWrapper(
            lm=TinyFakeCausalLM(),
            hidden_size=8,
            image_input_dim=8,
            audio_input_dim=6,
            image_prefix_tokens=2,
            audio_prefix_tokens=3,
            image_bridge_type="identity",
        )
        image_features = torch.randn(1, 2, 8)
        self.assertIs(wrapper.image_prefix(image_features), image_features)
        with self.assertRaisesRegex(ValueError, "cannot resample tokens"):
            wrapper.image_prefix(torch.randn(1, 3, 8))

    def assert_prefix_contract(self, prefix_tokens: int) -> None:
        expected_length = prefix_tokens + self.input_ids.shape[1]
        self.assertEqual(tuple(self.lm.last_inputs_embeds.shape), (1, expected_length, 8))
        self.assertEqual(tuple(self.lm.last_attention_mask.shape), (1, expected_length))
        self.assertEqual(tuple(self.lm.last_labels.shape), (1, expected_length))
        self.assertTrue(torch.equal(
            self.lm.last_labels[:, :prefix_tokens],
            torch.full((1, prefix_tokens), -100, dtype=torch.long),
        ))
        self.assertTrue(torch.equal(self.lm.last_labels[:, prefix_tokens:], self.labels))

    def test_image_prefix_labels_and_lengths(self) -> None:
        image_features = torch.randn(1, 5, 4)
        outputs = self.wrapper(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            image_features=image_features,
        )
        self.assert_prefix_contract(prefix_tokens=2)
        self.assertEqual(tuple(outputs.logits.shape), (1, 6, 17))

    def test_audio_prefix_labels_and_lengths(self) -> None:
        audio_features = torch.randn(1, 7, 6)
        outputs = self.wrapper(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            audio_features=audio_features,
        )
        self.assert_prefix_contract(prefix_tokens=3)
        self.assertEqual(tuple(outputs.logits.shape), (1, 7, 17))
        self.assertNotIn("output_hidden_states", self.lm.last_kwargs)

        self.wrapper(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            audio_features=audio_features,
            output_hidden_states=True,
        )
        self.assertTrue(self.lm.last_kwargs["output_hidden_states"])
        self.assertTrue(self.lm.last_kwargs["return_dict"])

    def test_changing_image_and_audio_prefixes_changes_fake_lm_outputs(self) -> None:
        cases = [
            ("image_features", torch.randn(1, 5, 4), torch.randn(1, 5, 4)),
            ("audio_features", torch.randn(1, 7, 6), torch.randn(1, 7, 6)),
        ]
        for feature_name, first, second in cases:
            with self.subTest(feature_name=feature_name):
                first_output = self.wrapper(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    **{feature_name: first},
                ).logits
                second_output = self.wrapper(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    **{feature_name: second},
                ).logits
                text_length = self.input_ids.shape[1]
                self.assertFalse(torch.equal(
                    first_output[:, -text_length:],
                    second_output[:, -text_length:],
                ))


if __name__ == "__main__":
    unittest.main()
