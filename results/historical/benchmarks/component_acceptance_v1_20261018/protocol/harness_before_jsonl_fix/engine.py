"""One final-proposal verifier for component acceptance benchmarks (batch one)."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import torch
from transformers import DynamicCache


def request_seed(sample_id, temperature, stream, seed=42):
    value = f'{seed}|{sample_id}|{temperature:g}|{stream}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'little') % (2**63-1)


def probabilities(logits, temperature, top_p=.95):
    """Nucleus includes the first token crossing p, in stable logit order."""
    if temperature == 0:
        return torch.zeros_like(logits, dtype=torch.float32).scatter_(
            -1, logits.argmax(-1, keepdim=True), 1.)
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError('Invalid sampling setting')
    ordered, indices = logits.float().div(temperature).sort(descending=True, stable=True)
    values = ordered.softmax(-1)
    keep = values.cumsum(-1) - values < top_p
    values = values * keep
    values = values / values.sum(-1, keepdim=True)
    return torch.zeros_like(values).scatter_(-1, indices, values)


def draw(probs, generator):
    return int(torch.multinomial(probs, 1, generator=generator).item())


def verify(proposal_ids, q_rows, p_rows, temperature, generator):
    """Returns accepted proposal prefix length and next target token."""
    for j, token in enumerate(proposal_ids):
        p, q = p_rows[j], q_rows[j]
        if temperature == 0:
            accepted = token == int(p.argmax())
        else:
            assert q[token] > 0, 'A sampled proposal has zero probability'
            accepted = bool(torch.rand((), device=p.device, generator=generator)
                            < torch.minimum(torch.ones_like(p[token]), p[token] / q[token]))
        if not accepted:
            residual = (p-q).clamp_min(0)
            # Positive rejection probability implies positive residual mass.
            if not torch.isfinite(residual).all() or residual.sum() <= 0:
                raise ArithmeticError('Invalid rejection residual')
            replacement = int(p.argmax()) if temperature == 0 else draw(residual/residual.sum(), generator)
            return j, replacement
    bonus = int(p_rows[-1].argmax()) if temperature == 0 else draw(p_rows[-1], generator)
    return len(proposal_ids), bonus


class TargetFeatures:
    """Capture selected decoder outputs, avoiding all-layer 32k hidden tuples."""
    def __init__(self, target, layer_ids):
        self.target, self.ids, self.values = target, layer_ids, {}
        self.handles = []
        for i in layer_ids:
            if i >= len(target.model.layers)-1:
                raise ValueError('Last-layer/final-norm feature requires a dedicated adapter')
            def capture(module, args, output, index=i):
                self.values[index] = output[0] if isinstance(output, tuple) else output
            self.handles.append(target.model.layers[i].register_forward_hook(capture))

    def forward(self, ids, cache, last_only=False):
        self.values.clear()
        out = self.target(input_ids=ids, past_key_values=cache, use_cache=True,
                          output_hidden_states=False, logits_to_keep=1 if last_only else 0)
        return out, torch.cat([self.values[i] for i in self.ids], -1)

    def close(self):
        for handle in self.handles:
            handle.remove()


class ComposableAdapter:
    def __init__(self, draft, target):
        from litespec.modeling.draft.composable import SharedOps
        self.draft = draft
        self.shared = SharedOps(target.get_input_embeddings(), target.get_output_embeddings())
        self.reset()

    def reset(self):
        from litespec.modeling.draft.composable import DraftState
        self.state = DraftState()

    def propose(self, anchor, position, context_hidden, context_positions, temperature,
                top_p, generator):
        from litespec.modeling.draft.composable import ModelInputs
        from litespec.modeling.draft.composable.contracts import BackboneOutput
        d = self.draft
        noise = torch.full((1, d.block_size), d.mask_token_id, device=context_hidden.device,
                           dtype=torch.long)
        noise[0, 0] = anchor
        inputs = ModelInputs(context_hidden, self.shared.embed_tokens(noise),
            torch.tensor([[position]], device=noise.device),
            torch.ones((1, 1), device=noise.device, dtype=torch.bool),
            context_position_ids=context_positions, anchor_token_ids=noise[:, :1],
            draft_state=self.state, mode='features')
        out = d(inputs, shared_ops=self.shared)
        self.state = out.draft_state
        features = BackboneOutput(out.backbone_hidden_states, out.draft_state)
        previous = torch.full((1, 1, d.block_size-1), anchor, device=noise.device, dtype=torch.long)
        rows, tokens = [], []
        for j in range(d.block_size-1):
            inputs.mode, inputs.previous_token_ids = 'teacher_forcing', previous
            scored = d.proposal(features, SimpleNamespace(inputs=inputs), self.shared)
            if scored.candidate_logits is not None:
                logits = scored.candidate_logits[0, 0, j]
                support = scored.candidate_ids[0, 0, j]
                q = torch.zeros(d.config.vocab_size, device=noise.device)
                q.scatter_(0, support, probabilities(logits, temperature, top_p))
            else:
                q = probabilities(scored.draft_logits[0, 0, j], temperature, top_p)
            token = int(q.argmax()) if temperature == 0 else draw(q, generator)
            tokens.append(token); rows.append(q)
            if j+1 < d.block_size-1:
                previous[0, 0, j+1] = token
        return tokens, rows


class NativeAdapter:
    """Native integrated DSpark/DFlash2/DFlash backbone, common final verifier."""
    def __init__(self, draft, target):
        self.draft, self.target = draft, target
        self.reset()

    def reset(self):
        self.cache = DynamicCache()

    def propose(self, anchor, position, context_hidden, context_positions, temperature,
                top_p, generator):
        d, target = self.draft, self.target
        noise = torch.full((1, d.block_size), d.mask_token_id, device=context_hidden.device,
                           dtype=torch.long)
        noise[0, 0] = anchor
        positions = torch.arange(self.cache.get_seq_length(), position+d.block_size,
                                 device=noise.device)[None]
        hidden = d(position_ids=positions, noise_embedding=target.get_input_embeddings()(noise),
                   target_hidden=context_hidden, past_key_values=self.cache,
                   use_cache=True, is_causal=False)[:, 1:]
        self.cache.crop(position)
        base = target.get_output_embeddings()(hidden)
        previous = torch.full((1, d.block_size-1), anchor, device=noise.device, dtype=torch.long)
        selector = getattr(d, 'candidate_selector', None)
        if selector is not None:
            unary, support = base.topk(selector.top_k, dim=-1)
        rows, tokens = [], []
        for j in range(d.block_size-1):
            if selector is not None:
                scores = selector.score_candidates(candidate_ids=support, unary_logits=unary,
                    hidden_states=hidden, predecessor_ids=previous)
                q = torch.zeros(d.config.vocab_size, device=noise.device)
                q.scatter_(0, support[0, j], probabilities(scores[0, j], temperature, top_p))
            else:
                scored = d.apply_logits_head(base, prev_token_ids=previous,
                    prev_token_embeddings=target.get_input_embeddings()(previous), hidden_states=hidden)
                q = probabilities(scored[0, j], temperature, top_p)
            token = int(q.argmax()) if temperature == 0 else draw(q, generator)
            tokens.append(token); rows.append(q)
            if j+1 < d.block_size-1:
                previous[0, j+1] = token
        return tokens, rows


@torch.inference_mode()
def generate(target, draft, input_ids, sample_id, temperature, eos_ids,
             max_new_tokens=512, top_p=.95, seed=42):
    if draft.block_size != 8:
        raise ValueError('Protocol v1 requires block size 8 (seven proposals)')
    device = input_ids.device
    proposal_rng = torch.Generator(device=device).manual_seed(request_seed(sample_id, temperature, 'proposal', seed))
    verifier_rng = torch.Generator(device=device).manual_seed(request_seed(sample_id, temperature, 'verifier', seed))
    adapter = ComposableAdapter(draft, target) if hasattr(draft.config, 'composable_config') else NativeAdapter(draft, target)
    capture = TargetFeatures(target, draft.target_layer_ids)
    cache = DynamicCache()
    generated, cycles = [], []
    try:
        out, context = capture.forward(input_ids, cache, last_only=True)
        p = probabilities(out.logits[0, -1], temperature, top_p)
        anchor = int(p.argmax()) if temperature == 0 else draw(p, verifier_rng)
        position = input_ids.shape[1]
        context_positions = torch.arange(position, device=device)[None]
        while len(generated) < max_new_tokens:
            if anchor in eos_ids:
                generated.append(anchor)
                break
            tokens, q_rows = adapter.propose(anchor, position, context, context_positions,
                                             temperature, top_p, proposal_rng)
            block = torch.tensor([[anchor]+tokens], device=device)
            verify_out, verified_hidden = capture.forward(block, cache)
            p_rows = [probabilities(row, temperature, top_p) for row in verify_out.logits[0]]
            accepted, next_anchor = verify(tokens, q_rows, p_rows, temperature, verifier_rng)
            committed = [anchor]+tokens[:accepted]
            eos_at = next((i for i,t in enumerate(committed) if t in eos_ids), None)
            budget_left = max_new_tokens-len(generated)
            logical = len(committed)
            actual = committed[:min(budget_left, eos_at+1 if eos_at is not None else logical)]
            terminal = eos_at is not None or logical > budget_left
            cycles.append({'accepted_draft_tokens':accepted, 'tau':1+accepted,
                           'committed_tokens':len(actual), 'censored':terminal,
                           'eos':eos_at is not None, 'budget_truncated':logical>budget_left})
            generated.extend(actual)
            if terminal or len(generated) >= max_new_tokens:
                break
            confirmed = 1+accepted
            cache.crop(position+confirmed)
            context = verified_hidden[:, :confirmed]
            context_positions = torch.arange(position, position+confirmed, device=device)[None]
            position += confirmed
            anchor = next_anchor
        full = [c for c in cycles if not c['censored']]
        return {'generated_ids':generated, 'cycles':cycles, 'complete_cycles':len(full),
                'generated_tokens_total':len(generated),
                'terminal_tokens_outside_cycles':len(generated)-sum(c['committed_tokens'] for c in cycles),
                'tau':sum(c['tau'] for c in full)/len(full) if full else None,
                'accepted_draft_tokens':sum(c['accepted_draft_tokens'] for c in full)/len(full) if full else None,
                'committed_per_cycle':sum(c['committed_tokens'] for c in cycles)/len(cycles) if cycles else None}
    finally:
        capture.close()
