"""Current-prefix convolution updates, distinct from prior speculative offsets."""

import unittest

import torch
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestConvValidLengths(CustomTestCase):
    def test_bf16_convolution_rounds_before_silu(self):
        torch.manual_seed(74)
        x = torch.randn(2, 8, 2048, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        weight = torch.randn(2048, 4, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(2, 3, 2048, device="cuda", dtype=torch.bfloat16).transpose(
            1, 2
        )
        raw = causal_conv1d_update(x, state.clone(), weight, activation=None)
        actual = causal_conv1d_update(
            x, state.clone(), weight, activation="silu", round_before_activation=True
        )
        torch.testing.assert_close(
            actual, torch.nn.functional.silu(raw), atol=0, rtol=0
        )
        flat = x.transpose(1, 2).flatten(0, 1).T
        cu = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)
        raw = causal_conv1d_fn(flat, weight, None, None, cu, [8, 8], activation=None)
        actual = causal_conv1d_fn(
            flat,
            weight,
            None,
            None,
            cu,
            [8, 8],
            activation="silu",
            round_before_activation=True,
        )
        torch.testing.assert_close(
            actual, torch.nn.functional.silu(raw), atol=0, rtol=0
        )

    def test_all_commit_lengths_padding_and_graph(self):
        torch.manual_seed(72)
        batch, channels, steps, width = 11, 6144, 8, 4
        x = torch.randn(
            batch, steps, channels, device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2)
        weight = torch.randn(channels, width, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(
            batch + 2, width - 1, channels, device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2)
        original = state.clone()
        slots = torch.tensor(
            [8, 2, 7, 1, 5, 6, 3, 9, 4, -1, 10], device="cuda", dtype=torch.int32
        )
        lengths = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 0], device="cuda", dtype=torch.int32
        )
        for row, count in enumerate(lengths.cpu().tolist()):
            x[row, :, count:] = torch.nan
        args = dict(
            x=x,
            conv_state=state,
            weight=weight,
            activation="silu",
            conv_state_indices=slots,
            valid_token_lens=lengths,
        )
        actual = causal_conv1d_update(**args)
        for row, (slot, count) in enumerate(
            zip(slots.cpu().tolist(), lengths.cpu().tolist())
        ):
            if slot < 0:
                self.assertTrue(bool((actual[row] == 0).all()))
                continue
            if count:
                reference_state = original[slot : slot + 1].clone()
                expected = causal_conv1d_update(
                    x[row : row + 1, :, :count],
                    reference_state,
                    weight,
                    activation="silu",
                )
                torch.testing.assert_close(
                    actual[row : row + 1, :, :count], expected, atol=0, rtol=0
                )
                joined = torch.cat((original[slot], x[row, :, :count]), -1)
                torch.testing.assert_close(
                    state[slot], joined[:, -(width - 1) :], atol=0, rtol=0
                )
            else:
                torch.testing.assert_close(state[slot], original[slot], atol=0, rtol=0)
            self.assertTrue(bool((actual[row, :, count:] == 0).all()))
        torch.testing.assert_close(state[0], original[0], atol=0, rtol=0)
        torch.testing.assert_close(state[-1], original[-1], atol=0, rtol=0)
        state.copy_(original)
        saved = lengths.clone()
        lengths.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graphed = causal_conv1d_update(**args)
        torch.testing.assert_close(state, original, atol=0, rtol=0)
        lengths.copy_(saved)
        graph.replay()
        torch.testing.assert_close(graphed, actual, atol=0, rtol=0)

    def test_rejects_prior_offset_combination(self):
        x = torch.zeros(2, 16, 8, device="cuda", dtype=torch.bfloat16)
        state = torch.zeros(2, 16, 10, device="cuda", dtype=torch.bfloat16)
        weight = torch.zeros(16, 4, device="cuda", dtype=torch.bfloat16)
        counts = torch.ones(2, device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            causal_conv1d_update(
                x, state, weight, num_accept_tokens=counts, valid_token_lens=counts
            )


if __name__ == "__main__":
    unittest.main()
