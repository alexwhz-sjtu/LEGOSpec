"""FP32 GDN chunks and short confirmed prefixes against an independent scan."""

import unittest

import torch
from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.kernels.ops.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.test.test_utils import CustomTestCase


def scan(k, v, g, beta, initial):
    state = initial.clone()
    k = k.float()
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    k = k.repeat_interleave(2, -2)
    for t in range(k.shape[0]):
        state = state * g[t].exp()[:, None, None]
        previous = (state * k[t, :, None, :]).sum(-1)
        delta = beta[t].float()[:, None] * (v[t].float() - previous)
        state = state + delta[:, :, None] * k[t, :, None, :]
    return state


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGDNStateUpdate(CustomTestCase):
    @torch.inference_mode()
    def test_fp32_chunks_with_extreme_nonpositive_gates(self):
        torch.manual_seed(83)
        for length in (17, 63, 64, 65, 129):
            raw_k = torch.randn(length, 16, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(length, 32, 128, device="cuda", dtype=torch.bfloat16)
            # Large decays followed by near-zero decays expose nonmonotone
            # parallel FP32 cumulative sums. A positive ULP is still a valid edge.
            g = -torch.pow(10.0, torch.rand(length, 32, device="cuda") * 33 - 30)
            beta = torch.rand(length, 32, device="cuda").bfloat16()
            state = torch.randn(3, 32, 128, 128, device="cuda") * 0.1
            original = state.clone()
            expected = scan(raw_k, v, g, beta, state[1])
            cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            n_chunks = (length + 63) // 64
            chunks = torch.tensor(
                [[0, i] for i in range(n_chunks)], device="cuda", dtype=torch.int32
            )
            k = l2norm_fwd(raw_k[None], output_dtype=torch.float32)
            gc = chunk_local_cumsum(
                g[None],
                64,
                cu_seqlens=cu,
                chunk_indices=chunks,
                output_dtype=torch.float64,
            )
            w, u, _ = chunk_gated_delta_rule_fwd_intra(
                k,
                v[None].float(),
                gc,
                beta[None].float(),
                cu_seqlens=cu,
                chunk_indices=chunks,
                use_fp32_math=True,
            )
            h, new_v = chunk_gated_delta_rule_fwd_h(
                k,
                w,
                u,
                gc,
                initial_state=state,
                initial_state_indices=torch.tensor(
                    [1], device="cuda", dtype=torch.int32
                ),
                cu_seqlens=cu,
                chunk_indices=chunks,
                chunk_offsets=torch.tensor(
                    [0, n_chunks], device="cuda", dtype=torch.int32
                ),
                save_chunk_states=False,
                save_new_value=False,
                use_fp32_math=True,
            )
            self.assertIsNone(h)
            self.assertIsNone(new_v)
            self.assertLess(float((state[1] - expected).norm() / expected.norm()), 0.01)
            torch.testing.assert_close(state[1], expected, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(state[[0, 2]], original[[0, 2]], atol=0, rtol=0)

    @torch.inference_mode()
    def test_short_prefixes_zero_through_eight_and_graph(self):
        self._short_prefixes(poison_padding=False)

    @torch.inference_mode()
    def test_valid_lengths_skip_nan_padding_and_zero_capture(self):
        self._short_prefixes(poison_padding=True)

    def _short_prefixes(self, *, poison_padding):
        torch.manual_seed(84)
        bs = 10
        k = torch.randn(bs, 8, 16, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(bs, 8, 32, 128, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(bs, 8, 32, device="cuda") * 3
        beta = torch.rand(bs, 8, 32, device="cuda").bfloat16()
        lengths = torch.tensor(list(range(9)) + [8], device="cuda")
        valid = torch.arange(8, device="cuda")[None] < lengths[:, None]
        g = torch.where(valid[..., None], g, 0).contiguous()
        beta = torch.where(valid[..., None], beta, 0).contiguous()
        if poison_padding:
            k.masked_fill_(~valid[:, :, None, None], torch.nan)
            v.masked_fill_(~valid[:, :, None, None], torch.inf)
            g.masked_fill_(~valid[:, :, None], torch.nan)
            beta.masked_fill_(~valid[:, :, None], torch.nan)
        indices = torch.tensor(
            list(range(1, 10)) + [-1], device="cuda", dtype=torch.int32
        )
        state = torch.randn(12, 32, 128, 128, device="cuda") * 0.1
        original = state.clone()

        def update():
            return fused_recurrent_gated_delta_rule_update(
                k,
                k,
                v,
                g,
                beta,
                initial_state_source=state,
                initial_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
                disable_output_calculation=True,
                valid_token_lens=lengths if poison_padding else None,
            )

        update()
        expected = original.clone()
        for row, length in enumerate(range(9)):
            expected[row + 1] = scan(
                k[row, :length],
                v[row, :length],
                g[row, :length],
                beta[row, :length],
                original[row + 1],
            )
        torch.testing.assert_close(state, expected, atol=1e-5, rtol=1e-4)
        eager = state.clone()
        saved_lengths = lengths.clone()
        if poison_padding:
            state.copy_(original)
            lengths.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            update()
        if poison_padding:
            torch.testing.assert_close(state, original, atol=0, rtol=0)
            lengths.copy_(saved_lengths)
        state.copy_(original)
        graph.replay()
        torch.testing.assert_close(state, eager, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
