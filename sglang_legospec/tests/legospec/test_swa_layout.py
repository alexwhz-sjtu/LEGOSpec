"""Discrete SWA address, anchored-window and partial-commit contracts."""

import unittest

import torch
from sglang.kernels.ops.attention.legospec_metadata import fill_legospec_block_indices
from sglang.kernels.ops.kvcache.legospec_ring import legospec_ring_locs
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSWALayout(CustomTestCase):
    def test_window_is_fixed_at_anchor_for_all_eight_queries(self):
        lens = torch.tensor(
            [511, 512, 513, 4096, 16384, 32768], device="cuda", dtype=torch.int32
        )
        slots = torch.tensor([5, 0, 2, 1, 4, 3], device="cuda", dtype=torch.int64)
        b, maximum = lens.numel(), 32768
        table = torch.arange(b * (maximum + 8), device="cuda", dtype=torch.int64).view(
            b, -1
        )
        current = torch.arange(b * 8, device="cuda", dtype=torch.int64) + 900000
        full_ptr = torch.cat((lens.new_zeros(1), (lens + 8).cumsum(0).int()))
        win_ptr = torch.cat(
            (lens.new_zeros(1), (lens.clamp_max(512) + 8).cumsum(0).int())
        )
        full = torch.empty(b * (maximum + 8), device="cuda", dtype=torch.int32)
        window = torch.empty(b * 520, device="cuda", dtype=torch.int32)
        scratch = torch.empty(b * 8, device="cuda", dtype=torch.int64)
        fill_legospec_block_indices(
            slots,
            lens,
            table,
            current,
            full_ptr,
            win_ptr,
            full,
            window,
            scratch,
            maximum,
        )
        full_tables = torch.full(
            (b, maximum + 8), -99, device="cuda", dtype=torch.int32
        )
        window_tables = torch.full((b, 520), -99, device="cuda", dtype=torch.int32)
        fill_legospec_block_indices(
            slots,
            lens,
            table,
            current,
            full_ptr,
            win_ptr,
            full,
            window,
            scratch,
            maximum,
            full_tables=full_tables,
            window_tables=window_tables,
        )
        for row, (n, slot) in enumerate(zip(lens.cpu().tolist(), slots.cpu().tolist())):
            f, w = int(full_ptr[row]), int(win_ptr[row])
            expected = torch.cat(
                (table[slot, :n], current[row * 8 : (row + 1) * 8])
            ).int()
            torch.testing.assert_close(full[f : f + n + 8], expected, atol=0, rtol=0)
            torch.testing.assert_close(
                full_tables[row, : n + 8], expected, atol=0, rtol=0
            )
            self.assertTrue(bool((full_tables[row, n + 8 :] == -99).all()))
            history = torch.arange(max(0, n - 512), n, device="cuda")
            ids = torch.cat(
                (
                    1 + slot * 520 + history % 512,
                    1 + slot * 520 + 512 + torch.arange(8, device="cuda"),
                )
            ).int()
            torch.testing.assert_close(window[w : w + len(ids)], ids, atol=0, rtol=0)
            torch.testing.assert_close(
                window_tables[row, : len(ids)], ids, atol=0, rtol=0
            )
            torch.testing.assert_close(
                scratch[row * 8 : (row + 1) * 8].int(), ids[-8:], atol=0, rtol=0
            )

    def test_commit_gates_and_prefill_wraparound(self):
        positions = torch.arange(1025, device="cuda", dtype=torch.int64)
        slots = torch.full_like(positions, 3)
        locs = legospec_ring_locs(slots, positions, torch.full_like(positions, 1024))
        self.assertTrue(bool((locs[:513] == -1).all()))
        self.assertEqual(len(locs[513:].unique()), 512)
        for valid in range(9):
            p = torch.arange(510, 518, device="cuda", dtype=torch.int64)
            actual = legospec_ring_locs(
                torch.ones_like(p),
                p,
                commit_lens=torch.tensor([valid], device="cuda", dtype=torch.int32),
            )
            self.assertTrue(bool((actual[valid:] == -1).all()))
            torch.testing.assert_close(
                actual[:valid], 1 + 520 + p[:valid] % 512, atol=0, rtol=0
            )


if __name__ == "__main__":
    unittest.main()
