"""Regressions for full-vocabulary selector and stable nucleus semantics."""

import unittest

import torch
from sglang.kernels.ops.sampling.stable_top_p import stable_nucleus_probs
from sglang.kernels.ops.speculative.legospec import selector_full_vocab_row
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestProposalSemantics(CustomTestCase):
    def test_stable_top_p_ties_and_crossing(self):
        # Frozen formal harness uses stable descending order, including crossing.
        for vocab in (8, 17, 151936):
            for seed in (0, 1):
                torch.manual_seed(seed)
                logits = torch.randn(3, vocab, device="cuda").bfloat16().float()
                logits[0].zero_()  # boundary ties must not all survive
                p = torch.tensor([0.95, 0.95, 1.0], device="cuda")
                ordered, indices = logits.sort(descending=True, stable=True)
                sorted_probs = ordered.softmax(-1)
                keep = sorted_probs.cumsum(-1) - sorted_probs < p[:, None]
                expected = torch.zeros_like(logits).scatter(
                    -1, indices, sorted_probs * keep
                )
                expected /= expected.sum(-1, keepdim=True)
                expected[2] = logits[2].softmax(-1)
                actual = stable_nucleus_probs(logits, p)
                self.assertTrue(
                    torch.equal(actual > 0, expected > 0),
                    f"support differs at vocab={vocab}, seed={seed}: "
                    f"actual={(actual > 0).sum(-1).tolist()}, "
                    f"expected={(expected > 0).sum(-1).tolist()}",
                )
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)

    def test_top_p_one_skips_truncation(self):
        logits = torch.tensor([[0.0, -10.0, -20.0, -30.0]], device="cuda")
        actual = stable_nucleus_probs(
            logits, torch.ones(1, device="cuda"), truncate=False
        )
        torch.testing.assert_close(actual, logits.softmax(-1), atol=0, rtol=0)
        self.assertTrue(bool((actual > 0).all()))

    def test_selector_missing_predecessor_preserves_full_vocab(self):
        base = torch.tensor([[9.0, 8.0, 7.0, 6.0], [9.0, 8.0, 7.0, 6.0]], device="cuda")
        candidates = torch.tensor([[0, 1], [0, 1]], device="cuda")
        predecessors = torch.tensor([[2, 3], [2, 3]], device="cuda")
        previous = torch.tensor([3, 0], device="cuda")
        delta = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]], device="cuda"
        )
        actual = selector_full_vocab_row(
            base, candidates, predecessors, previous, delta
        )
        torch.testing.assert_close(
            actual[0], torch.tensor([12.0, 12.0, 7.0, 6.0], device="cuda")
        )
        torch.testing.assert_close(actual[1], base[1], rtol=0, atol=0)
        self.assertGreater(float(actual[0].softmax(-1)[2]), 0)


if __name__ == "__main__":
    unittest.main()
