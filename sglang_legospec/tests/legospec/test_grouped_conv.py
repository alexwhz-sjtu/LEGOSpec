"""Frozen grouped-convolution math; the eager implementation is oracle only."""

import unittest

import torch
import torch.nn.functional as F
from sglang.srt.models.dflash import DFlashGroupedConv
from sglang.test.test_utils import CustomTestCase


def reference(x, delta, base):
    blocks = x.reshape(-1, 8, 160, 16)
    coeff = base.reshape(1, 1, 2, 160, 16) + delta.reshape(-1, 8, 2, 160, 1)
    shifted = F.pad(blocks[:, :-1], (0, 0, 0, 0, 1, 0))
    return (coeff[:, :, 0] * blocks + coeff[:, :, 1] * shifted).reshape_as(x)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGroupedConv(CustomTestCase):
    @torch.inference_mode()
    def test_pre_post_share_the_original_dynamic_projection(self):
        torch.manual_seed(631)
        conv = DFlashGroupedConv(2560, 8, 2, 16, hf_semantics=True).to(
            device="cuda", dtype=torch.bfloat16
        )
        conv.base_kernel.normal_(mean=0.1, std=0.2)
        conv.kernel_projection.weight.normal_(std=0.01)
        x = torch.randn(64, 2560, device="cuda", dtype=torch.bfloat16)
        projected = conv.kernel_projection(x).reshape(64, 2, 2, 160)
        calls = []
        hook = conv.kernel_projection.register_forward_hook(lambda *a: calls.append(1))
        before, post_coeff = conv.prepare(x)
        y = torch.randn_like(x)
        after = conv.finish(y, post_coeff)
        hook.remove()
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(
            before,
            reference(x, projected[:, 0], conv.base_kernel[0]),
            atol=0.02,
            rtol=0.02,
        )
        torch.testing.assert_close(
            after,
            reference(y, projected[:, 1], conv.base_kernel[1]),
            atol=0.02,
            rtol=0.02,
        )

    @torch.inference_mode()
    def test_invalid_cross_block_tap_does_not_leak_nonfinite_data(self):
        conv = DFlashGroupedConv(2560, 8, 2, 16, hf_semantics=True).to(
            device="cuda", dtype=torch.bfloat16
        )
        conv.base_kernel.fill_(1)
        x = torch.ones(16, 2560, device="cuda", dtype=torch.bfloat16)
        x[7, :1280] = float("nan")
        x[7, 1280:] = float("inf")
        delta = torch.zeros(16, 2, 160, device="cuda", dtype=torch.bfloat16)
        actual = conv._convolve(x, delta, 0)
        torch.testing.assert_close(
            actual[8], torch.ones_like(actual[8]), atol=0, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
