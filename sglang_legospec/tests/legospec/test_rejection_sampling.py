"""Independent controlled-uniform and distribution checks for native rejection."""

import unittest
from types import SimpleNamespace

import torch
from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    AcceptSampling,
    SoftmaxTemp,
)
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import accept_draft_tokens
from sglang.srt.speculative.legospec_proposal import _sample_q
from sglang.srt.speculative.legospec_rng import legospec_verify_uniforms
from sglang.test.test_utils import CustomTestCase


def verify(candidates, p, q, uniforms, final_uniforms):
    b, s = candidates.shape
    retrieve = torch.arange(b * s, device="cuda", dtype=torch.int64).reshape(b, s)
    predicts = torch.full((b * s,), -1, device="cuda", dtype=torch.int64)
    indices = torch.full_like(retrieve, -1)
    counts = torch.empty(b, device="cuda", dtype=torch.int32)
    chain_speculative_sampling_triton(
        predicts,
        indices,
        counts,
        candidates,
        retrieve,
        torch.empty_like(retrieve),
        torch.empty_like(retrieve),
        uniforms,
        final_uniforms,
        p,
        q,
        1.0,
        1.0,
        True,
    )
    return predicts.reshape(b, s), indices, counts


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRejectionSampling(CustomTestCase):
    def test_native_accept_consumes_supplied_coins_and_preserves_default_rng(self):
        """Delegating actual q must preserve residual, bonus, cutoff and stock RNG."""
        p = torch.tensor([[[0.4, 0.3, 0.2, 0.1]] * 4] * 3, device="cuda")
        p[1, 0] = torch.tensor([0.0, 0.2, 0.3, 0.5], device="cuda")
        p[2, 1] = torch.tensor([0.2, 0.0, 0.3, 0.5], device="cuda")
        q = torch.tensor([[[0.4, 0.3, 0.2, 0.1]] * 3] * 3, device="cuda")
        candidates = torch.tensor([[3, 0, 1, 2]] * 3, device="cuda", dtype=torch.int64)
        logits = p.log().reshape(-1, 4)
        sampling = SimpleNamespace(
            temperatures=torch.ones(3, 1, device="cuda"),
            need_top_k_sampling=False,
            need_top_p_sampling=False,
        )
        args = dict(
            candidates=candidates,
            target_logits=logits,
            draft_probs=q,
            sampling_info=sampling,
            draft_input=None,
            gamma=3,
            verify_num_draft_tokens=4,
        )
        native_p = SoftmaxTemp.execute(
            logits=logits, temperatures=sampling.temperatures, rows_per_request=4
        ).reshape(3, 4, 4)
        uniforms = torch.full((3, 3), 0.5, device="cuda")
        final_uniforms = torch.tensor([0.65, 0.25, 0.75], device="cuda")
        expected_tokens, _, expected_counts = verify(
            candidates, native_p, q, uniforms, final_uniforms
        )
        self.assertEqual(expected_counts.cpu().tolist(), [3, 0, 1])
        expected_bonus = expected_tokens.gather(
            1, expected_counts.long()[:, None]
        ).flatten()
        rng_before = torch.cuda.get_rng_state().clone()
        counts, bonus, trim = AcceptSampling.execute(
            **args,
            uniform_samples=uniforms,
            uniform_samples_for_final_sampling=final_uniforms,
        )
        torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
        torch.testing.assert_close(bonus, expected_bonus, atol=0, rtol=0)
        self.assertEqual(trim.cpu().tolist(), [0, 0, 0])
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(), rng_before))
        capped, capped_bonus, trim = AcceptSampling.execute(
            **args,
            uniform_samples=uniforms,
            uniform_samples_for_final_sampling=final_uniforms,
            cutoff_verify_lens=torch.tensor(
                [2, 1, 2], device="cuda", dtype=torch.int32
            ),
        )
        self.assertEqual(capped.cpu().tolist(), [1, 0, 1])
        self.assertEqual(trim.cpu().tolist(), [2, 0, 0])
        torch.testing.assert_close(
            capped_bonus,
            expected_tokens.gather(1, capped.long()[:, None]).flatten(),
            atol=0,
            rtol=0,
        )
        # Omitting the extension retains the original two torch.rand draws.
        torch.cuda.set_rng_state(rng_before)
        default_uniforms = torch.rand(3, 3, device="cuda")
        default_final = torch.rand(3, device="cuda")
        expected = tuple(
            t.clone()
            for t in AcceptSampling.execute(
                **args,
                uniform_samples=default_uniforms,
                uniform_samples_for_final_sampling=default_final,
            )
        )
        rng_after = torch.cuda.get_rng_state().clone()
        torch.cuda.set_rng_state(rng_before)
        actual = AcceptSampling.execute(**args)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(), rng_after))
        with self.assertRaisesRegex(ValueError, "both"):
            AcceptSampling.execute(**args, uniform_samples=uniforms)
        with self.assertRaisesRegex(ValueError, "shape"):
            AcceptSampling.execute(
                **args,
                uniform_samples=uniforms[:, :2],
                uniform_samples_for_final_sampling=final_uniforms,
            )

    def test_verification_rng_survives_reorder_and_subset(self):
        """A request's rejection and bonus coins cannot depend on batch membership."""
        seeds = torch.tensor([42, 99, 127], device="cuda", dtype=torch.int64)
        positions = torch.tensor([72, 80, 88], device="cuda", dtype=torch.int64)
        before = legospec_verify_uniforms(seeds, positions, 8)
        for order in ([2, 0, 1], [1], [2, 0]):
            indices = torch.tensor(order, device="cuda")
            after = legospec_verify_uniforms(seeds[indices], positions[indices], 8)
            for value, reference in zip(after, before):
                torch.testing.assert_close(value, reference[indices], atol=0, rtol=0)

    def test_actual_q_uses_native_accept_and_mixed_greedy_selection(self):
        """Recorded q is authoritative even without logits, including mixed batches."""
        candidates = torch.tensor([[3, 0, 1, 2]] * 3, device="cuda", dtype=torch.int64)
        logits = torch.tensor([[2.0, 0.0, 1.0, -1.0]] * 12, device="cuda")
        # Deliberately unrelated to target logits; there are no draft logits to
        # reconstruct. Native acceptance must consume these exact probabilities.
        q = torch.tensor([[[0.1, 0.4, 0.0, 0.5]] * 3] * 3, device="cuda")
        seeds = torch.tensor([42, 99, 127], device="cuda", dtype=torch.int64)
        positions = torch.tensor([72, 80, 88], device="cuda", dtype=torch.int64)
        sampling = SimpleNamespace(
            temperatures=torch.ones(3, 1, device="cuda"),
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            is_all_greedy=False,
            is_any_greedy=True,
        )
        uniforms, final_uniforms = legospec_verify_uniforms(seeds, positions, 4)
        common = dict(
            candidates=candidates,
            target_logits=logits,
            sampling_info=sampling,
            draft_input=None,
            gamma=3,
            verify_num_draft_tokens=4,
        )
        expected = tuple(
            t.clone()
            for t in AcceptSampling.execute(
                **common,
                draft_probs=q,
                uniform_samples=uniforms,
                uniform_samples_for_final_sampling=final_uniforms,
            )
        )
        actual = accept_draft_tokens(
            **common,
            draft_block=DraftBlockResult(
                draft_tokens=candidates[:, 1:],
                corrected_logits=None,
                greedy_mask=torch.tensor([True, False, False], device="cuda"),
                temperatures=sampling.temperatures,
                greedy_draft=False,
                proposal_probs=q,
                proposal_seeds=seeds,
                proposal_positions=positions,
            ),
        )
        # Greedy row accepts token 0, then rejects token 1 in favor of token 0.
        self.assertEqual([int(t[0]) for t in actual], [1, 0, 0])
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value[1:], reference[1:], atol=0, rtol=0)

    def test_native_proposal_draws_match_recorded_q(self):
        n = 100000
        q = torch.tensor([0.2, 0.0, 0.8], device="cuda").expand(n, -1).contiguous()
        seeds = torch.arange(n, device="cuda", dtype=torch.int64) + 987654
        positions = torch.full_like(seeds, 4096)
        ids = _sample_q(q, seeds, positions)
        counts = torch.bincount(ids, minlength=3).float() / n
        self.assertEqual(int((ids == 1).sum()), 0)
        torch.testing.assert_close(counts, q[0], atol=0.006, rtol=0)

    def test_controlled_accept_reject_residual_bonus(self):
        p = torch.tensor([[[0.4, 0.3, 0.2, 0.1]] * 4] * 3, device="cuda")
        q = torch.tensor([[[0.4, 0.3, 0.2, 0.1]] * 3] * 3, device="cuda")
        candidates = torch.tensor([[3, 0, 1, 2]] * 3, device="cuda", dtype=torch.int64)
        # Row 0 accepts all. Row 1 rejects at zero, row 2 at one.
        p[1, 0] = torch.tensor([0.0, 0.2, 0.3, 0.5], device="cuda")
        p[2, 1] = torch.tensor([0.2, 0.0, 0.3, 0.5], device="cuda")
        u = torch.full((3, 3), 0.5, device="cuda")
        uf = torch.tensor([0.65, 0.25, 0.75], device="cuda")
        tokens, indices, counts = verify(candidates, p, q, u, uf)
        p_cpu, q_cpu = p.cpu().double(), q.cpu().double()
        for b in range(3):
            correct = 0
            while correct < 3:
                t = int(candidates[b, correct + 1])
                if float(u[b, correct]) * float(q_cpu[b, correct, t]) >= float(
                    p_cpu[b, correct, t]
                ):
                    break
                correct += 1
            final = (
                p_cpu[b, correct]
                if correct == 3
                else (p_cpu[b, correct] - q_cpu[b, correct]).clamp_min(0)
            )
            cdf = (final / final.sum()).cumsum(0)
            bonus = int(torch.searchsorted(cdf, float(uf[b]), right=True))
            self.assertEqual(int(counts[b]), correct)
            self.assertEqual(int(tokens[b, correct]), bonus)
            self.assertEqual(
                indices[b, : correct + 1].cpu().tolist(),
                list(range(b * 4, b * 4 + correct + 1)),
            )
            self.assertEqual(tokens[b, :correct].cpu().tolist(), [0, 1, 2][:correct])

    def test_final_first_token_distribution_is_target(self):
        torch.manual_seed(902)
        n = 100000
        target = torch.tensor([0.15, 0.55, 0.30], device="cuda")
        proposal = torch.tensor([0.65, 0.10, 0.25], device="cuda")
        sampled = torch.multinomial(proposal, n, replacement=True)
        candidates = torch.stack((torch.zeros_like(sampled), sampled), -1)
        p = target.expand(n, 2, 3).contiguous()
        q = proposal.expand(n, 1, 3).contiguous()
        tokens, _, _ = verify(
            candidates,
            p,
            q,
            torch.rand(n, 1, device="cuda"),
            torch.rand(n, device="cuda"),
        )
        actual = torch.bincount(tokens[:, 0], minlength=3).float() / n
        torch.testing.assert_close(actual, target, atol=0.006, rtol=0)

    def test_request_rng_survives_batch_reorder(self):
        logits = torch.tensor(
            [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]], device="cuda"
        )
        seeds = torch.tensor([42, 99, 127], device="cuda", dtype=torch.int64)
        pos = torch.tensor([72, 80, 88], device="cuda", dtype=torch.int64)
        before = multinomial_with_seed(logits, seeds, pos)
        order = torch.tensor([2, 0, 1], device="cuda")
        after = multinomial_with_seed(logits[order], seeds[order], pos[order])
        torch.testing.assert_close(after, before[order], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
