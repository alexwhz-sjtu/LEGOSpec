"""DSpARK's DFlash2 convolution switch and checkpoint behavior."""
import copy
import tempfile
import unittest

import torch
from transformers import Qwen3Config

from specforge.modeling.draft.dflash import Qwen3DFlashDecoderLayer
from specforge.modeling.draft.dflash2 import DFlashGroupedConv, Qwen3DFlash2DecoderLayer
from specforge.modeling.draft.dspark import DSparkDraftModel


def tiny_config(enabled=False):
    cfg = Qwen3Config(
        architectures=["DSparkDraftModel"], hidden_size=16, intermediate_size=32,
        num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2,
        num_target_layers=4, head_dim=4, max_position_embeddings=64,
        vocab_size=32, block_size=4, layer_types=["full_attention"] * 2,
        dflash_config={
            "projector_type": "dspark", "attention_mode": "gqa",
            "target_layer_ids": [1], "mask_token_id": 31,
            "markov_rank": 8, "markov_head_type": "vanilla",
            "confidence_head_alpha": 1.0, "confidence_head_with_markov": True,
            "dspark_backbone_conv_enabled": enabled,
            "conv_kernel_size": 2, "conv_group_size": 4,
        },
    )
    cfg._attn_implementation = "eager"
    return cfg


def forward(model, noise, context):
    length = context.shape[1]
    pos = torch.arange(length + noise.shape[1]).unsqueeze(0)
    mask = torch.ones(1, 1, noise.shape[1], length + noise.shape[1], dtype=torch.bool)
    return model(position_ids=pos, attention_mask=mask, noise_embedding=noise, target_hidden=context)


class DSparkConvSwitchTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_disabled_keeps_original_layers_and_state_keys(self):
        model = DSparkDraftModel(tiny_config(False))
        self.assertTrue(all(type(layer) is Qwen3DFlashDecoderLayer for layer in model.layers))
        self.assertFalse(any("_conv." in key for key in model.state_dict()))

    def test_full_mode_reuses_dflash2_wrappers_and_starts_as_identity(self):
        baseline = DSparkDraftModel(tiny_config(False)).eval()
        model = DSparkDraftModel(tiny_config(True)).eval()
        missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(all("_conv." in key for key in missing))
        self.assertTrue(all(type(layer) is Qwen3DFlash2DecoderLayer for layer in model.layers))
        self.assertTrue(all(isinstance(layer.attention_conv, DFlashGroupedConv) and
                            isinstance(layer.mlp_conv, DFlashGroupedConv) for layer in model.layers))
        noise, context = torch.randn(1, 4, 16), torch.randn(1, 3, 16)
        with torch.no_grad():
            torch.testing.assert_close(forward(model, noise, context),
                                       forward(baseline, noise, context), rtol=0, atol=0)

    def test_same_seed_keeps_non_conv_initial_weights_identical(self):
        torch.manual_seed(42)
        baseline = DSparkDraftModel(tiny_config(False))
        torch.manual_seed(42)
        model = DSparkDraftModel(tiny_config(True))
        for name, original in baseline.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], original, rtol=0, atol=0)

    def test_conv_receives_gradients_and_roundtrips_trained_weights(self):
        model = DSparkDraftModel(tiny_config(True)).eval()
        noise, context = torch.randn(1, 4, 16), torch.randn(1, 3, 16)
        forward(model, noise, context).square().mean().backward()
        for layer in model.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                self.assertGreater(conv.kernel_projection.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            model.layers[0].attention_conv.kernel_projection.weight.fill_(0.01)
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path)
            loaded, info = DSparkDraftModel.from_pretrained(
                path, attn_implementation="eager", output_loading_info=True)
            self.assertFalse(info["missing_keys"])
            self.assertFalse(info["unexpected_keys"])
            for name, original in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[name], original, rtol=0, atol=0)

    def test_requires_valid_configuration_only_when_enabled(self):
        config = tiny_config(False)
        config.dflash_config.pop("conv_kernel_size")
        DSparkDraftModel(config)
        config.dflash_config["dspark_backbone_conv_enabled"] = True
        with self.assertRaisesRegex(ValueError, "conv_kernel_size"):
            DSparkDraftModel(config)
        config.dflash_config["conv_kernel_size"] = 5
        with self.assertRaisesRegex(ValueError, "must not exceed block_size"):
            DSparkDraftModel(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
