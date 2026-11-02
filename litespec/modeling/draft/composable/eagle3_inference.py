"""EAGLE3 autoregressive drafting with separate target and draft KV caches.

Probability processing, RNG streams, rejection and cycle accounting match the
component-acceptance-v1 harness. Cache rebuild follows DeepSpec 005e03b.
"""

import hashlib

import torch
from transformers import DynamicCache

from .backbone.eagle3 import Eagle3State
from .contracts import SharedOps


def request_seed(sample_id, temperature, stream, seed=42):
    key = f"{seed}|{sample_id}|{temperature:g}|{stream}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)


def probabilities(logits, temperature, top_p=.95):
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling setting")
    if temperature == 0:
        return torch.zeros_like(logits, dtype=torch.float32).scatter_(
            -1, logits.argmax(-1, keepdim=True), 1.)
    ordered, indices = logits.float().div(temperature).sort(descending=True, stable=True)
    values = ordered.softmax(-1)
    values *= values.cumsum(-1) - values < top_p
    return torch.zeros_like(values).scatter_(-1, indices, values / values.sum(-1, keepdim=True))


def draw(probs, generator):
    return int(torch.multinomial(probs, 1, generator=generator).item())


def verify(proposal_ids, q_rows, p_rows, temperature, generator):
    for j, token in enumerate(proposal_ids):
        p, q = p_rows[j], q_rows[j]
        if temperature == 0:
            accepted = token == int(p.argmax())
        else:
            if q[token] <= 0:
                raise ValueError("Sampled proposal has zero probability")
            accepted = bool(torch.rand((), device=p.device, generator=generator)
                            < torch.minimum(torch.ones_like(p[token]), p[token] / q[token]))
        if not accepted:
            residual = (p - q).clamp_min(0)
            if not torch.isfinite(residual).all() or residual.sum() <= 0:
                raise ArithmeticError("Invalid rejection residual")
            replacement = int(p.argmax()) if temperature == 0 else draw(residual / residual.sum(), generator)
            return j, replacement
    bonus = int(p_rows[-1].argmax()) if temperature == 0 else draw(p_rows[-1], generator)
    return len(proposal_ids), bonus


class TargetFeatures:
    def __init__(self, target, layer_ids):
        self.target, self.ids, self.values, self.handles = target, layer_ids, {}, []
        layers = target.model.layers
        for i in layer_ids:
            if not 0 <= i < len(layers):
                self.close()
                raise ValueError("Target feature layer is out of range")
            def capture(module, args, output, index=i):
                self.values[index] = output[0] if isinstance(output, tuple) else output
            self.handles.append(layers[i].register_forward_hook(capture))

    def forward(self, ids, cache, last_only=False):
        self.values.clear()
        out = self.target(input_ids=ids, past_key_values=cache, use_cache=True,
                          output_hidden_states=False, logits_to_keep=1 if last_only else 0)
        return out, torch.cat([self.values[i] for i in self.ids], -1)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class Eagle3Adapter:
    def __init__(self, draft, target):
        if not draft.is_eagle3:
            raise ValueError("Eagle3Adapter requires an EAGLE3 backbone")
        self.draft = draft
        self.shared = SharedOps(target.get_input_embeddings(), target.get_output_embeddings())
        self.reset()

    def reset(self):
        self.state = Eagle3State()
        self.cache_len_before = 0

    def prefill(self, input_ids, target_hidden, anchor):
        shifted = torch.cat((input_ids[:, 1:], input_ids.new_tensor([[anchor]])), 1)
        self.draft.extend_eagle3(target_hidden, shifted, self.state, self.shared, target_features=True)

    def propose(self, temperature, top_p, generator):
        self.cache_len_before = self.state.length
        tokens, rows = [], []
        for step in range(self.draft.ttt_length):
            logits = self.draft.score_eagle3(self.state.hidden, self.shared).draft_logits[0, -1]
            q = probabilities(logits, temperature, top_p)
            token = int(q.argmax()) if temperature == 0 else draw(q, generator)
            tokens.append(token)
            rows.append(q)
            if step + 1 < self.draft.ttt_length:
                ids = torch.tensor([[token]], device=logits.device)
                self.draft.extend_eagle3(self.state.hidden, ids, self.state, self.shared)
        return tokens, rows

    def update(self, verified_hidden, accepted_tokens, next_anchor):
        # Discard every speculative entry, including entries for accepted IDs.
        self.state.crop(self.cache_len_before)
        ids = torch.tensor([list(accepted_tokens) + [next_anchor]], device=verified_hidden.device)
        self.draft.extend_eagle3(verified_hidden[:, :ids.shape[1]], ids,
                                self.state, self.shared, target_features=True)


@torch.inference_mode()
def generate_eagle3(target, draft, input_ids, sample_id="request", temperature=0.,
                    eos_ids=(), max_new_tokens=512, top_p=.95, seed=42):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("EAGLE3 inference requires one nonempty, unpadded prompt")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling setting")
    eos_ids = set(eos_ids or ())
    device = input_ids.device
    proposal_rng = torch.Generator(device=device).manual_seed(request_seed(sample_id, temperature, "proposal", seed))
    verifier_rng = torch.Generator(device=device).manual_seed(request_seed(sample_id, temperature, "verifier", seed))
    adapter, capture, cache = Eagle3Adapter(draft, target), TargetFeatures(target, draft.target_layer_ids), DynamicCache()
    generated, cycles = [], []
    try:
        if max_new_tokens:
            out, context = capture.forward(input_ids, cache, last_only=True)
            p = probabilities(out.logits[0, -1], temperature, top_p)
            anchor = int(p.argmax()) if temperature == 0 else draw(p, verifier_rng)
            position = input_ids.shape[1]
            if anchor not in eos_ids:
                adapter.prefill(input_ids, context, anchor)
            while len(generated) < max_new_tokens:
                if anchor in eos_ids:
                    generated.append(anchor)
                    break
                tokens, q_rows = adapter.propose(temperature, top_p, proposal_rng)
                block = torch.tensor([[anchor] + tokens], device=device)
                verified, verified_hidden = capture.forward(block, cache)
                p_rows = [probabilities(row, temperature, top_p) for row in verified.logits[0]]
                accepted, next_anchor = verify(tokens, q_rows, p_rows, temperature, verifier_rng)
                committed = [anchor] + tokens[:accepted]
                eos_at = next((i for i, token in enumerate(committed) if token in eos_ids), None)
                budget_left = max_new_tokens - len(generated)
                logical = len(committed)
                actual = committed[:min(budget_left, eos_at + 1 if eos_at is not None else logical)]
                terminal = eos_at is not None or logical > budget_left
                cycles.append(dict(accepted_draft_tokens=accepted, tau=1 + accepted,
                    committed_tokens=len(actual), censored=terminal,
                    eos=eos_at is not None, budget_truncated=logical > budget_left))
                generated.extend(actual)
                if terminal or len(generated) >= max_new_tokens:
                    break
                position += 1 + accepted
                cache.crop(position)
                if next_anchor not in eos_ids:
                    adapter.update(verified_hidden, tokens[:accepted], next_anchor)
                anchor = next_anchor
        full = [cycle for cycle in cycles if not cycle["censored"]]
        return dict(generated_ids=generated, cycles=cycles, complete_cycles=len(full),
            generated_tokens_total=len(generated),
            terminal_tokens_outside_cycles=len(generated) - sum(c["committed_tokens"] for c in cycles),
            tau=sum(c["tau"] for c in full) / len(full) if full else None,
            accepted_draft_tokens=sum(c["accepted_draft_tokens"] for c in full) / len(full) if full else None,
            committed_per_cycle=sum(c["committed_tokens"] for c in cycles) / len(cycles) if cycles else None)
    finally:
        capture.close()
