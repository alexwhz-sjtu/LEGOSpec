"""Regression tests for BF16 boundaries that differ from stock fused math."""

import unittest

import torch
from sglang.kernels.ops.activation.hf_silu_mul import hf_silu_and_mul
from sglang.kernels.ops.attention.table_qk_norm_rope import table_qk_norm_rope_
from sglang.kernels.ops.layernorm.hf_add_rmsnorm import hf_add_rmsnorm
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPrecisionContracts(CustomTestCase):
    def test_every_bf16_silu_input(self):
        bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.uint16)
        gate = bits.view(torch.bfloat16).view(1, -1)
        packed = torch.cat((gate, torch.ones_like(gate)), -1)
        torch.testing.assert_close(
            hf_silu_and_mul(packed),
            torch.nn.functional.silu(gate),
            atol=0,
            rtol=0,
            equal_nan=True,
        )

    def test_residual_rounds_before_variance(self):
        torch.manual_seed(18)
        x = torch.randn(32, 2560, device="cuda", dtype=torch.bfloat16) * 0.1
        residual = torch.randn_like(x)
        weight = torch.randn(2560, device="cuda", dtype=torch.bfloat16)
        summed = x + residual
        expected = (
            weight
            * (
                summed.float()
                * torch.rsqrt(summed.float().square().mean(-1, keepdim=True) + 1e-6)
            ).bfloat16()
        )
        actual, actual_sum = hf_add_rmsnorm(x, residual, weight, 1e-6)
        torch.testing.assert_close(actual_sum, summed, atol=0, rtol=0)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    def test_silu_rounds_before_gate_up_multiply(self):
        torch.manual_seed(19)
        x = torch.randn(7, 2 * 9728, device="cuda", dtype=torch.bfloat16) * 3
        expected = torch.nn.functional.silu(x[:, :9728]) * x[:, 9728:]
        torch.testing.assert_close(hf_silu_and_mul(x), expected, atol=0.02, rtol=0.02)

    def test_fused_norm_rope_keeps_q_k_and_v_contracts(self):
        torch.manual_seed(20)
        hq, hk, d = 32, 8, 128
        source = torch.randn(17, (hq + 2 * hk) * d, device="cuda", dtype=torch.bfloat16)
        qw = torch.randn(d, device="cuda", dtype=torch.bfloat16)
        kw = torch.randn_like(qw)
        angles = torch.randn(32769, d // 2, device="cuda", dtype=torch.float32)
        cache = torch.cat((angles.cos(), angles.sin()), -1)
        positions = torch.tensor(
            [
                0,
                1,
                7,
                8,
                63,
                64,
                511,
                512,
                513,
                4095,
                4096,
                8191,
                8192,
                16383,
                16384,
                32767,
                32768,
            ],
            device="cuda",
        )
        cos, sin = cache[positions].bfloat16().chunk(2, -1)
        cos, sin = cos[:, None], sin[:, None]

        def reference(raw, w):
            raw = raw.reshape(raw.shape[0], -1, d)
            norm = (
                raw.float()
                * torch.rsqrt(raw.float().square().mean(-1, keepdim=True) + 1e-6)
            ).bfloat16() * w
            left, right = norm.chunk(2, -1)
            return torch.cat(
                (left * cos - right * sin, right * cos + left * sin), -1
            ).flatten(1)

        expected_q = reference(source[:, : hq * d], qw)
        expected_k = reference(source[:, hq * d : (hq + hk) * d], kw)
        actual = source.clone()
        table_qk_norm_rope_(
            actual, positions, qw, kw, cache, hq, hk, d, 1e-6, hf_semantics=True
        )
        torch.testing.assert_close(
            actual[:, : hq * d], expected_q, atol=0.02, rtol=0.02
        )
        torch.testing.assert_close(
            actual[:, hq * d : (hq + hk) * d], expected_k, atol=0.02, rtol=0.02
        )
        torch.testing.assert_close(
            actual[:, (hq + hk) * d :], source[:, (hq + hk) * d :], atol=0, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
