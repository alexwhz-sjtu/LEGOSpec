from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from specforge.core.flashmtp import OnlineFlashMTPModel
from specforge.modeling.draft.flashmtp import (
    FLASHMTP_ARCHITECTURE_VERSION,
    FlashMTPDraftModel,
    FlashMTPGroupedConv,
    Qwen3FlashMTPAttention,
)
from specforge.training.strategies.base import FlashMTPTrainStrategy


def _config(head_type: str = "markov", output_mode: str = "additive"):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"],
    )
    config.num_target_layers = 4
    config.block_size = 4
    config.flashmtp_config = {
        "architecture_version": FLASHMTP_ARCHITECTURE_VERSION,
        "chs_num_layers": 2,
        "target_layer_ids": [0, 3],
        "markov_head_type": head_type,
        "markov_output_mode": output_mode,
        "markov_rank": 4,
    }
    return config


class FlashMTPIntegrationTest(unittest.TestCase):
    def test_grouped_conv_identity_and_block_boundary(self):
        conv = FlashMTPGroupedConv(
            hidden_size=2,
            block_size=3,
            taps=2,
            group_size=1,
        )
        inputs = torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [7.0, 8.0],
                    [9.0, 10.0],
                    [11.0, 12.0],
                ]
            ]
        )

        prepared, output_kernel = conv.prepare(inputs)
        torch.testing.assert_close(prepared, inputs)
        torch.testing.assert_close(conv.finish(inputs, output_kernel), inputs)

        with torch.no_grad():
            conv.base_kernel.zero_()
            conv.base_kernel[:, 1].fill_(1.0)
        shifted, _ = conv.prepare(inputs)
        expected = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [0.0, 0.0],
                    [7.0, 8.0],
                    [9.0, 10.0],
                ]
            ]
        )
        torch.testing.assert_close(shifted, expected)

    def test_backbone_conv_switch_builds_identity_wrappers(self):
        config = _config()
        config.flashmtp_config.update(
            {
                "backbone_conv_enabled": True,
                "conv_kernel_size": 2,
                "conv_group_size": 4,
            }
        )

        model = FlashMTPDraftModel(config)
        layer = model.layers[0]
        self.assertTrue(model.backbone_conv_enabled)
        self.assertEqual(layer.attention_conv.block_size, model.draft_query_length)
        self.assertEqual(layer.mlp_conv.block_size, model.draft_query_length)
        torch.testing.assert_close(
            layer.attention_conv.kernel_projection.weight,
            torch.zeros_like(layer.attention_conv.kernel_projection.weight),
        )
        self.assertEqual(model.config.flashmtp_config["conv_kernel_size"], 2)
        self.assertEqual(model.config.flashmtp_config["conv_group_size"], 4)

        model.set_config_block_size(6)
        self.assertEqual(layer.attention_conv.block_size, model.draft_query_length)
        self.assertEqual(layer.mlp_conv.block_size, model.draft_query_length)

    def test_backbone_conv_defaults_off_for_legacy_configs(self):
        model = FlashMTPDraftModel(_config())

        self.assertFalse(model.backbone_conv_enabled)
        self.assertIsNone(model.layers[0].attention_conv)
        self.assertIsNone(model.layers[0].mlp_conv)

    def test_backbone_conv_forward_backward_reaches_dynamic_kernel(self):
        config = _config()
        config._attn_implementation = "eager"
        config.flashmtp_config.update(
            {
                "backbone_conv_enabled": True,
                "conv_kernel_size": 2,
                "conv_group_size": 4,
            }
        )
        model = FlashMTPDraftModel(config)
        query_len = model.draft_query_length
        context_len = model.condition_slot_count
        output = model(
            position_ids=torch.arange(query_len).repeat(2, 1),
            rotary_position_ids=torch.arange(context_len + query_len).repeat(2, 1),
            noise_embedding=torch.randn(2, query_len, config.hidden_size),
            target_hidden=torch.randn(2, 1, context_len, config.hidden_size),
            history_hidden=torch.empty(2, 1, 0, config.hidden_size),
            attention_mask=torch.zeros(2, 1, query_len, context_len + query_len),
        )
        output.square().mean().backward()

        grad = model.layers[0].attention_conv.kernel_projection.weight.grad
        self.assertEqual(output.shape, (2, query_len, config.hidden_size))
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_flex_attention_defaults_to_triton_backend(self):
        config = _config()
        config._attn_implementation = "flex_attention"
        attention = Qwen3FlashMTPAttention(config, layer_idx=0)
        hidden_states = torch.randn(1, 4, config.hidden_size)
        target_hidden = torch.randn(1, 6, config.hidden_size)
        cos = torch.ones(1, 10, config.head_dim)
        sin = torch.zeros_like(cos)
        received = {}

        def fake_flex_attention(module, query, key, value, attention_mask, **kwargs):
            received.update(kwargs)
            return query.transpose(1, 2).contiguous(), None

        with (
            mock.patch(
                "specforge.modeling.draft.flashmtp.ALL_ATTENTION_FUNCTIONS",
                {"flex_attention": fake_flex_attention},
            ),
            mock.patch(
                "specforge.modeling.draft.flashmtp.flex_attention_backend",
                return_value=None,
            ),
        ):
            output, weights = attention(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                position_embeddings=(cos, sin),
                attention_mask=object(),
                kernel_options={"PRESCALE_QK": True},
            )

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertIsNone(weights)
        self.assertEqual(
            received["kernel_options"],
            {"PRESCALE_QK": True, "BACKEND": "TRITON"},
        )

    def test_training_wrapper_has_no_chunk_controls(self):
        parameters = inspect.signature(OnlineFlashMTPModel.__init__).parameters
        self.assertNotIn("ce_chunk_size", parameters)
        self.assertNotIn("anchor_chunk_size", parameters)

    def test_ce_projects_all_prediction_positions_in_one_call(self):
        class DraftStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.markov_head = None
                self.markov_output_mode = "additive"

        class CountingHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, hidden_states):
                self.batch_sizes.append(hidden_states.shape[0])
                return hidden_states

        lm_head = CountingHead()
        model = OnlineFlashMTPModel(
            draft_model=DraftStub(),
            target_lm_head=lm_head,
            target_embed_tokens=nn.Identity(),
            mask_token_id=0,
            block_size=3,
            tv_loss_weight=0.0,
        )
        hidden = torch.tensor([[[[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]]]])
        labels = torch.tensor([[[0, 2]]])
        weights = torch.tensor([[[1.0, 0.5]]])

        loss, _, _, final_ce, base_ce, tv_loss = model._weighted_ce_and_metrics(
            prediction_hidden=hidden,
            prev_token_ids=torch.zeros_like(labels),
            labels=labels,
            weight_mask=weights,
            binary_eval_mask=weights > 0,
            block_keep_mask=torch.ones((1, 1), dtype=torch.bool),
        )

        per_position = F.cross_entropy(
            hidden.reshape(-1, 3), labels.reshape(-1), reduction="none"
        )
        expected = (per_position * weights.reshape(-1)).sum() / (
            weights.sum() + 1e-6
        )
        self.assertEqual(lm_head.batch_sizes, [2])
        torch.testing.assert_close(final_ce, expected)
        torch.testing.assert_close(loss, expected)
        self.assertEqual(base_ce.item(), 0.0)
        self.assertEqual(tv_loss.item(), 0.0)

    def test_defaults_use_token_history_and_window_one(self):
        model = FlashMTPDraftModel(_config())

        self.assertEqual(model.history_mode, "token")
        self.assertEqual(model.sliding_window_size, 1)
        self.assertEqual(model.history_slot_count, 0)
        self.assertEqual(model.markov_head_type, "markov")
        self.assertEqual(model.markov_head.head_type, "vanilla")
        self.assertFalse(hasattr(model, "history_fuse"))

    def test_retained_head_profiles(self):
        additive = FlashMTPDraftModel(_config("markov", "additive"))
        direct = FlashMTPDraftModel(_config("rnn_easy", "direct"))

        self.assertEqual(additive.markov_output_mode, "additive")
        self.assertEqual(direct.markov_output_mode, "direct")
        self.assertEqual(direct.markov_head.head_type, "rnn_easy")

        for head_type, output_mode in (
            ("markov", "direct"),
            ("rnn_easy", "additive"),
            ("gated", "additive"),
        ):
            with self.subTest(head_type=head_type, output_mode=output_mode):
                with self.assertRaisesRegex(ValueError, "supports only"):
                    FlashMTPDraftModel(_config(head_type, output_mode))

    def test_strategy_splits_flat_capture_by_layer_id(self):
        class TrainingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.capture_layer_ids = [0, 3]
                self.draft_model = SimpleNamespace(
                    config=SimpleNamespace(hidden_size=4)
                )

        strategy = FlashMTPTrainStrategy(TrainingModel())
        capture = torch.arange(2 * 3 * 8).view(2, 3, 8)
        layers = strategy._captured_hidden_states(capture)

        self.assertEqual(set(layers), {0, 3})
        torch.testing.assert_close(layers[0], capture[..., :4])
        torch.testing.assert_close(layers[3], capture[..., 4:])

        with self.assertRaisesRegex(ValueError, "captured layers"):
            strategy._captured_hidden_states(capture[..., :-1])


if __name__ == "__main__":
    unittest.main()
