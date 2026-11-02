"""Numerical and lifecycle checks for full-batch sampled-anchor OPD."""
import copy
from types import SimpleNamespace
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from litespec.core.composable.anchor_opd import sample_proposal, verify_sampled_blocks
from litespec.core.composable.anchor_opd_windows import prepare_anchor_opd_window
from litespec.core.composable.batch import prepare_layout
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.modeling.draft.composable import ComposableDraftConfig, ComposableDraftModel
from litespec.modeling.draft.composable.contracts import BackboneOutput, ModelInputs, SharedOps
from litespec.modeling.target.anchor_opd import Qwen3AnchorScorer


def target_model(device='cpu', dtype=torch.float32):
    torch.manual_seed(9)
    config = Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                        vocab_size=23, max_position_embeddings=128, attention_dropout=0.)
    config._attn_implementation = 'eager'
    return Qwen3ForCausalLM(config).to(device=device, dtype=dtype).requires_grad_(False).eval()


def fixture(proposal='parallel', device='cpu', dtype=torch.float32):
    target = target_model(device, dtype)
    prop = {'type': proposal, **({'rank': 4} if proposal == 'rnn' else {})}
    config = ComposableDraftConfig(hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_target_layers=2, vocab_size=23,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        block_size=4, attention_dropout=0.,
        composable_config={'context': {'type': 'dflash', 'target_layer_ids': [0]},
            'backbone': {'type': 'layer_stack', 'layers': [{'attention': 'gqa', 'access': 'dense'}]},
            'attention_configs': {'gqa': {}}, 'proposal': prop,
            'objective': {'type': 'opd_survival', 'loss_decay_gamma': None}})
    draft = ComposableDraftModel(config).to(device=device, dtype=dtype)
    wrapper = ComposableTrainingModel(draft, target.lm_head, target.model.embed_tokens, 3,
        num_anchors=4, objective_chunk_blocks=0, opd_data_mode='sampled_anchor').to(device)
    scorer = Qwen3AnchorScorer(target)
    ids = torch.randint(23, (2, 12), device=device)
    mask = torch.ones_like(ids).float()
    mask[1, 10:] = 0
    return wrapper, scorer, dict(input_ids=ids, loss_mask=mask,
        context_valid_mask=mask.bool(), target_hidden_states=None)


class AnchorOPDTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_shared_prefix_matches_independent_causal_forward(self):
        model = target_model()
        scorer = Qwen3AnchorScorer(model)
        ids = torch.randint(23, (2, 12))
        valid = torch.ones_like(ids).bool(); valid[1, 10:] = False
        anchors = torch.tensor([[1, 5, 7], [2, 6, 0]])
        keep = torch.tensor([[True, True, True], [True, True, False]])
        draft = torch.randint(23, (2, 3, 3))
        prefix = scorer.prefill_anchors(ids, valid, anchors, keep, capture_layer_ids=[0, 1])
        actual = scorer.score_sampled_blocks(prefix, draft)
        for n, a in keep.nonzero().tolist():
            length = anchors[n, a].item() + 1
            sequence = torch.cat((ids[n, :length], draft[n, a, :-1]))[None]
            expected = model.model(sequence, output_hidden_states=True).last_hidden_state[:, length - 1:]
            torch.testing.assert_close(actual[n, a], expected[0], atol=2e-6, rtol=2e-5)
        full = model.model(ids[:, :8], output_hidden_states=True)
        torch.testing.assert_close(prefix.context_hidden[0], full.hidden_states[1], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(model.model.norm(prefix.context_hidden[1]), full.last_hidden_state, atol=2e-6, rtol=2e-5)
        self.assertEqual(prefix.key_values[0][0].shape[0], 2)
        prefix.release(); self.assertEqual(prefix.key_values, [])

    def test_no_future_or_other_branch_leakage(self):
        scorer = Qwen3AnchorScorer(target_model())
        ids = torch.randint(23, (1, 12)); valid = torch.ones_like(ids).bool()
        anchors = torch.tensor([[2, 7]]); keep = torch.ones_like(anchors).bool()
        draft = torch.randint(23, (1, 2, 3))
        prefix = scorer.prefill_anchors(ids, valid, anchors, keep)
        before = scorer.score_sampled_blocks(prefix, draft)
        other = draft.clone(); other[:, 0] = (other[:, 0] + 1) % 23
        after = scorer.score_sampled_blocks(prefix, other)
        torch.testing.assert_close(before[:, 1], after[:, 1])
        changed = ids.clone(); changed[:, 3:] = (changed[:, 3:] + 1) % 23
        after = scorer.score_sampled_blocks(scorer.prefill_anchors(changed, valid, anchors, keep), draft)
        torch.testing.assert_close(before[:, 0], after[:, 0])

    def test_probabilistic_verifier_and_first_rejection(self):
        p = torch.tensor([[[[.2, .8], [.9, .1], [.5, .5]]]])
        ids = torch.zeros(1, 1, 3).long()
        logq = torch.tensor([[[.8, .9, .5]]]).log()
        # First acceptance probability is 0.25: it can reject even though the
        # same token is proposed. Later decisions cannot revive a prefix.
        for uniforms, expected in [([.1, .2, .3], 3), ([.3, .1, .1], 0)]:
            r = verify_sampled_blocks(p.log(), ids, logq, torch.tensor([[uniforms]]), torch.ones(1, 1).bool())
            self.assertEqual(r.item(), expected)
        with self.assertRaises(ValueError):
            verify_sampled_blocks(p.log(), ids, logq, torch.ones_like(logq), torch.ones(1, 1).bool())

    def test_parallel_rnn_sampling_replay_and_freezing(self):
        for proposal in ('parallel', 'rnn'):
            with self.subTest(proposal=proposal):
                wrapper, scorer, values = fixture(proposal)
                with torch.no_grad():
                    meta = wrapper(**values, opd_scorer=scorer, metadata_only=True)
                record = meta['sampled_opd_batch']
                captured = []
                hook = wrapper.draft_model.proposal.register_forward_hook(
                    lambda m, i, o: captured.append(o.draft_logits.detach()) if o.draft_logits is not None else None)
                output = wrapper(**values, sampled_opd_batch=record)
                hook.remove()
                replay = captured[-1].float().log_softmax(-1).gather(-1, record.draft_ids[..., None]).squeeze(-1)
                torch.testing.assert_close(replay, record.sampled_logq, atol=2e-6, rtol=2e-5)
                output.loss.backward()
                self.assertEqual((scorer.prefill_calls, scorer.branch_calls), (1, 1))
                self.assertTrue(all(p.grad is None for p in scorer.model.parameters()))
                self.assertGreater(wrapper.draft_model.context.projection.weight.grad.abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(output.loss))

    def test_cached_context_and_online_context_match(self):
        for proposal in ('parallel', 'rnn'):
            wrapper, scorer, values = fixture(proposal)
            with torch.no_grad():
                original = scorer.model.model(values['input_ids'], attention_mask=values['context_valid_mask'],
                                               output_hidden_states=True)
            torch.manual_seed(55)
            live = wrapper(**values, opd_scorer=scorer, metadata_only=True)['sampled_opd_batch']
            cached_values = dict(values, target_hidden_states=original.hidden_states[1],
                                 target_last_hidden_states=original.last_hidden_state)
            torch.manual_seed(55)
            cached = wrapper(**cached_values, opd_scorer=scorer, metadata_only=True)['sampled_opd_batch']
            torch.testing.assert_close(live.draft_ids, cached.draft_ids)
            torch.testing.assert_close(live.sampled_logq, cached.sampled_logq, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(live.teacher_hidden, cached.teacher_hidden, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(live.accepted_lengths, cached.accepted_lengths)

    def test_fast_forward_and_checkpoint_do_not_resample_target(self):
        wrapper, scorer, values = fixture('rnn')
        calls = []
        hook = wrapper.draft_model.backbone.register_forward_hook(lambda *args: calls.append(1))
        output = wrapper(**values, opd_scorer=scorer)
        self.assertEqual(len(calls), 1)
        output.loss.backward(); hook.remove()
        self.assertEqual((scorer.prefill_calls, scorer.branch_calls), (1, 1))

    def test_accumulation_collects_target_once_and_spills(self):
        wrapper, scorer, values = fixture('rnn')
        class Adapter:
            def _anchor_values(self, session, raw): return raw
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            session = SimpleNamespace(args=SimpleNamespace(accumulation_steps=2,
                window_buffer_mb=0, cache_dir=directory), fsdp_model=wrapper,
                anchor_scorer=scorer, process_group=None)
            records = prepare_anchor_opd_window(Adapter(), session, values, iter([values]), 2)
            self.assertTrue(all(item.spill_path for item in records))
            for item in records:
                saved = item.materialize()
                result = wrapper(input_ids=saved['input_ids'], loss_mask=saved['loss_mask'],
                    target_hidden_states=None, context_valid_mask=None,
                    sampled_opd_batch=saved['sampled_opd_batch'],
                    opd_global_denominators=item.opd_component_denominators,
                    opd_reference_samples=item.opd_reference_samples)
                result.loss.backward()
                item.release()
            self.assertEqual((scorer.prefill_calls, scorer.branch_calls), (2, 2))

    def test_generator_resume_determinism(self):
        wrapper, scorer, values = fixture()
        saved = torch.get_rng_state()
        a = wrapper(**values, opd_scorer=scorer, metadata_only=True)['sampled_opd_batch']
        after = torch.get_rng_state()
        torch.set_rng_state(saved)
        b = wrapper(**values, opd_scorer=scorer, metadata_only=True)['sampled_opd_batch']
        torch.testing.assert_close(a.draft_ids, b.draft_ids)
        torch.testing.assert_close(a.uniforms, b.uniforms)
        torch.testing.assert_close(a.layout.anchor_positions, b.layout.anchor_positions)
        torch.testing.assert_close(after, torch.get_rng_state())
        wrapper(**values, sampled_opd_batch=b)
        torch.testing.assert_close(after, torch.get_rng_state())

    def test_cached_and_live_context_loss_gradients_match(self):
        wrapper, scorer, values = fixture('rnn')
        with torch.no_grad():
            original = scorer.model.model(values['input_ids'], attention_mask=values['context_valid_mask'],
                                          output_hidden_states=True)
        cached_values = dict(values, target_hidden_states=original.hidden_states[1],
                             target_last_hidden_states=original.last_hidden_state)
        results = []
        for current in (values, cached_values):
            wrapper.zero_grad(set_to_none=True); torch.manual_seed(101)
            out = wrapper(**current, opd_scorer=scorer)
            out.loss.backward()
            results.append((out.loss.detach(), {n: p.grad.clone() for n, p in wrapper.named_parameters() if p.grad is not None}))
        torch.testing.assert_close(results[0][0], results[1][0], atol=3e-6, rtol=3e-5)
        for name in results[0][1]:
            torch.testing.assert_close(results[0][1][name], results[1][1][name], atol=3e-6, rtol=3e-5)

    def test_empty_rank_and_no_complete_blocks(self):
        wrapper, scorer, values = fixture()
        values['loss_mask'][1] = 0
        output = wrapper(**values, opd_scorer=scorer)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        values['loss_mask'].zero_()
        with self.assertRaisesRegex(ValueError, 'complete'):
            wrapper(**values, opd_scorer=scorer)

    def test_online_cli_defaults_and_explicit_chunk_rejection(self):
        from litespec.training.recipes.composable import parse_args
        from contextlib import redirect_stderr
        import io
        argv = ['--target-model-path', 'target', '--train-data-path', 'data.jsonl',
                '--output-dir', 'out', '--opd-data-mode', 'sampled_anchor',
                '--init-from', 'checkpoint', '--report-to', 'none']
        args = parse_args(argv=argv)
        self.assertEqual(args.objective_chunk_blocks, 0)
        self.assertEqual(args.opd_target_chunk_anchors, 0)
        self.assertEqual(args.num_anchors, 512)
        # The existing SGLangAnchorScorer supports the same collocated TP=1 path.
        sglang_args = parse_args(argv=argv + ['--target-model-backend', 'sglang'])
        self.assertEqual(sglang_args.target_model_backend, 'sglang')
        for extra in (['--objective-chunk-blocks', '32'], ['--opd-target-chunk-anchors', '128'],
                      ['--tp-size', '2']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv=argv + extra)

    def test_distribution_statistics_match_raw_loss_and_gradients(self):
        from litespec.core.composable.objectives import build_objective
        from litespec.core.composable.objectives.base import Supervision
        from litespec.core.composable.anchor_opd import full_batch_statistics
        objective = build_objective({'type': 'opd_survival'})
        logits = torch.randn(2, 3, 4, 23, requires_grad=True)
        teacher = torch.randn_like(logits)
        labels = torch.randint(23, (2, 3, 4)); mask = torch.ones_like(labels).float()
        r = torch.tensor([[0, 2, 4], [1, 3, 0]])
        results = []
        for precompute in (False, True):
            stats = full_batch_statistics(logits, teacher) if precompute else None
            output = objective(SimpleNamespace(draft_logits=logits),
                Supervision(labels, mask, teacher, opd_accepted_lengths=r, opd_distribution_terms=stats))
            results.append((output.loss, torch.autograd.grad(output.loss, logits)[0]))
        torch.testing.assert_close(results[0][0], results[1][0])
        torch.testing.assert_close(results[0][1], results[1][1])


if __name__ == '__main__':
    unittest.main()
