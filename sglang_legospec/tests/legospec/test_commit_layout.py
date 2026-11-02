"""Discrete confirmed-prefix clipping, independent of target KV allocation."""

import unittest

import torch
from sglang.kernels.ops.speculative.legospec_commit import legospec_commit_lens
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCommitLayout(CustomTestCase):
    def fixture(self):
        candidates = (
            torch.arange(8, device="cuda", dtype=torch.int64).expand(9, -1).contiguous()
        )
        positions = (
            (torch.arange(8, device="cuda", dtype=torch.int64) + 100)
            .expand(9, -1)
            .contiguous()
        )
        slots = torch.ones_like(positions)
        counts = torch.arange(9, device="cuda", dtype=torch.int32)
        limits = torch.full((9,), 200, device="cuda", dtype=torch.int64)
        stops = torch.full((9, 2), -1, device="cuda", dtype=torch.int64)
        out = torch.empty_like(counts)
        return candidates, positions, slots, counts, limits, stops, out

    def test_every_confirmed_length_and_eos_column(self):
        args = self.fixture()
        ids, pos, slots, counts, limits, stops, out = args
        for column in range(8):
            stops[:, 0].fill_(column)
            legospec_commit_lens(*args)
            expected = [min(n, column + 1) for n in range(9)]
            self.assertEqual(out.cpu().tolist(), expected)
        stops.fill_(-1)
        legospec_commit_lens(*args)
        torch.testing.assert_close(out, counts, atol=0, rtol=0)

    def test_budget_padding_and_graph_replay(self):
        args = self.fixture()
        ids, pos, slots, counts, limits, stops, out = args
        counts.fill_(8)
        limits.copy_(torch.arange(9, device="cuda", dtype=torch.int64) + 100)
        legospec_commit_lens(*args)
        self.assertEqual(out.cpu().tolist(), list(range(9)))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            legospec_commit_lens(*args)
        slots[3].zero_()
        slots[7].fill_(-1)
        counts[6].zero_()
        graph.replay()
        self.assertEqual(out.cpu().tolist(), [0, 1, 2, 0, 4, 5, 0, 0, 8])

    def test_queued_decode_after_stop_and_slot_reuse(self):
        args = self.fixture()
        ids, positions, slots, counts, limits, stops, out = args
        slots.copy_(torch.arange(1, 10, device="cuda")[:, None])
        terminated = torch.zeros(10, device="cuda", dtype=torch.int32)
        terminated[0] = 99
        counts.fill_(8)
        stops[:, 0].fill_(2)
        graph = torch.cuda.CUDAGraph()
        # Capture never commits tokens or latches a terminal condition.
        counts.zero_()
        with torch.cuda.graph(graph):
            legospec_commit_lens(*args, terminated)
        self.assertEqual(terminated.cpu().tolist(), [99] + [0] * 9)
        counts.fill_(8)
        graph.replay()
        self.assertEqual(out.cpu().tolist(), [3] * 9)
        self.assertEqual(terminated.cpu().tolist(), [99] + [1] * 9)
        # The overlap scheduler may already have issued this unused forward.
        positions.add_(8)
        stops.fill_(-1)
        graph.replay()
        self.assertEqual(out.cpu().tolist(), [0] * 9)
        terminated[4] = 0  # New-generation prefill resets only its own slot.
        limits[3] = 110
        graph.replay()
        self.assertEqual(out.cpu().tolist(), [0, 0, 0, 2, 0, 0, 0, 0, 0])
        self.assertEqual(int(terminated[4]), 1)
        self.assertEqual(int(terminated[0]), 99)


if __name__ == "__main__":
    unittest.main()
