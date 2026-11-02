"""Draft cuBLAS rounding survives target dispatch, compilation and graph replay."""

import unittest

import torch
from sglang.kernels.ops.gemm.legospec_linear import legospec_cublas_linear
from sglang.srt.batch_invariant_ops import (
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
)
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDraftLinear(CustomTestCase):
    @torch.inference_mode()
    def test_frozen_gemm_under_target_dispatch_and_graph(self):
        torch.manual_seed(87)
        cases = []
        for batch in (1, 4, 16, 32):
            for width, out, bias in ((2560, 512, False), (3072, 768, True)):
                x = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(out, width, device="cuda", dtype=torch.bfloat16)
                b = (
                    torch.randn(out, device="cuda", dtype=torch.bfloat16)
                    if bias
                    else None
                )
                cases.append((x, weight, b, torch.nn.functional.linear(x, weight, b)))
        enable_batch_invariant_mode()
        try:
            compiled = torch.compile(legospec_cublas_linear, dynamic=True)
            for x, weight, bias, expected in cases:
                actual = compiled(x, weight, bias)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = compiled(x, weight, bias)
                graph.replay()
                torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        finally:
            disable_batch_invariant_mode()


if __name__ == "__main__":
    unittest.main()
