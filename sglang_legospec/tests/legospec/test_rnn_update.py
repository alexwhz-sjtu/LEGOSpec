"""BF16 rounding/subnormal regression for the production RNN pointwise kernel."""

import unittest

import torch
from sglang.kernels.ops.activation import dspark_rnn_update
from sglang.test.test_utils import CustomTestCase


class TestRNNUpdate(CustomTestCase):
    def check_case(self, state, raw):
        g, c, o = raw.chunk(3, -1)
        gate, candidate = torch.sigmoid(g), torch.tanh(c)
        expected = (gate * state + (1.0 - gate) * candidate, torch.tanh(o))
        saved = [value.clone() for value in (state, raw)]
        actual = dspark_rnn_update(state, raw)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.all((a == b) | (a.isnan() & b.isnan())).item())
        for current, before in zip((state, raw), saved):
            self.assertTrue(
                torch.equal(current.view(torch.uint8), before.view(torch.uint8))
            )

    @torch.inference_mode()
    def test_all_bf16_patterns_and_subnormal_state(self):
        bits = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
            .reshape(-1, 256)
        )
        torch.manual_seed(42)
        self.check_case(
            torch.randn_like(bits).clamp_(-1, 1),
            torch.cat((bits, bits.flip(0), bits.roll(73, 0)), -1),
        )
        # The state may itself contain subnormals; keeping only tanh's small
        # inputs would not catch a future FTZ state-arithmetic regression.
        self.check_case(
            bits, torch.randn(bits.shape[0], 768, device="cuda", dtype=torch.bfloat16)
        )

    @torch.inference_mode()
    def test_captured_replay_reads_new_inputs_without_mutating_state(self):
        for bs in (1, 4, 16, 32):
            with self.subTest(batch=bs):
                state = torch.randn(bs, 256, device="cuda", dtype=torch.bfloat16)
                raw = torch.randn(bs, 768, device="cuda", dtype=torch.bfloat16)
                for _ in range(3):
                    dspark_rnn_update(state, raw)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = dspark_rnn_update(state, raw)
                raw.normal_()
                state.normal_()
                graph.replay()
                g, c, o = raw.chunk(3, -1)
                gate = torch.sigmoid(g)
                expected = (gate * state + (1.0 - gate) * torch.tanh(c), torch.tanh(o))
                self.assertTrue(
                    all(torch.equal(a, b) for a, b in zip(actual, expected))
                )

    def test_invalid_contracts_fail_before_launch_and_empty_batch_is_valid(self):
        state = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
        raw = torch.empty(0, 768, device="cuda", dtype=torch.bfloat16)
        self.assertEqual(
            [tuple(x.shape) for x in dspark_rnn_update(state, raw)],
            [(0, 256), (0, 256)],
        )
        with self.assertRaises(ValueError):
            dspark_rnn_update(state.float(), raw.float())
        with self.assertRaises(ValueError):
            dspark_rnn_update(
                torch.empty(2, 256, device="cuda", dtype=torch.bfloat16), raw
            )


if __name__ == "__main__":
    unittest.main()
