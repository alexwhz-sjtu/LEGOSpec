"""Parallel top-K transition lattices, with versioned proposal support."""

import torch

from ...dflash2 import CandidateSelector
from ..contracts import DraftOutput
from .parallel import ParallelProposal


class SelectorProposal(ParallelProposal):
    def __init__(self, config):
        super().__init__(config)
        recipe = config.composable_config["proposal"]
        self.candidate_selector = CandidateSelector(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            state_rank=int(recipe.get("rank", 256)),
            top_k=int(recipe.get("top_k", 16)),
            initializer_range=config.initializer_range,
        )
        self.stop_gradient = bool(recipe.get("stop_gradient", False))
        self.full_vocab_residual = recipe.get("selector_mode", "candidate_only") == "full_vocab_residual"

    def parallel_lattice(self, base_logits, hidden_states, anchor_token_ids, candidate_ids=None):
        """Compute every slot's K-by-K delta table before any path selection.

        Row p of slot i is conditioned on candidate p of slot i-1. Slot zero
        has a known anchor instead, repeated across rows. No sampled token is
        needed to construct the table, and no vocabulary-sized predecessor
        expansion or predecessor-conditioned rescoring is used.
        """
        if base_logits.ndim < 3 or base_logits.size(-2) < 1:
            raise ValueError("Selector lattice requires a non-empty complete block of slots")
        if candidate_ids is None:
            candidate_ids = base_logits.topk(self.candidate_selector.top_k, dim=-1).indices
        hidden = hidden_states.detach() if self.stop_gradient else hidden_states
        slots, k = candidate_ids.shape[-2:]
        # Zero unary extracts transition deltas without BF16 subtraction or
        # double addition. The shared scorer projects all slots in one call.
        delta = self.candidate_selector.build_lattice(
            candidate_ids=candidate_ids.reshape(-1, slots, k),
            unary_logits=base_logits.new_zeros(candidate_ids.shape).reshape(-1, slots, k),
            hidden_states=hidden.reshape(-1, slots, hidden.size(-1)),
            anchor_token_ids=anchor_token_ids.reshape(-1),
        )
        return delta.reshape(*candidate_ids.shape, k), candidate_ids

    @staticmethod
    def predecessor_rows(candidate_ids, predecessor_ids):
        """Locate true predecessors in the *previous* slot's candidate set."""
        k = candidate_ids.size(-1)
        known_anchor = predecessor_ids[..., :1, None].expand(*candidate_ids.shape[:-2], 1, k)
        previous_candidates = torch.cat((known_anchor, candidate_ids[..., :-1, :]), dim=-2)
        matches = previous_candidates == predecessor_ids[..., None]
        return matches.long().argmax(-1), matches.any(-1)

    def full_vocab_logits(self, base_logits, hidden_states, predecessor_ids, candidate_ids=None):
        """Gather precomputed rows; missing predecessors use zero correction.

        This scoring API consumes complete blocks, matching parallel inference.
        A current gold label outside top-K remains supervised. A predecessor
        outside the previous top-K removes only that slot's Selector correction.
        """
        lattice, candidate_ids = self.parallel_lattice(
            base_logits, hidden_states, predecessor_ids[..., 0], candidate_ids,
        )
        rows, covered = self.predecessor_rows(candidate_ids, predecessor_ids)
        k = candidate_ids.size(-1)
        delta = lattice.gather(-2, rows[..., None, None].expand(*rows.shape, 1, k)).squeeze(-2)
        delta = torch.where(covered[..., None], delta, torch.zeros_like(delta))
        return base_logits.float().scatter_add(-1, candidate_ids, delta.float()), candidate_ids

    def sample_candidates_parallel(self, base_logits, hidden_states, anchor_token_ids, temperature):
        """Legacy candidate-only distribution, using a parallel score table."""
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        lattice, candidate_ids = self.parallel_lattice(base_logits, hidden_states, anchor_token_ids)
        scores = base_logits.gather(-1, candidate_ids)[..., None, :] + lattice
        row = torch.zeros_like(anchor_token_ids)
        tokens, probabilities = [], []
        k = candidate_ids.size(-1)
        for position in range(base_logits.size(-2)):
            selected_scores = scores[:, position].gather(1, row[:, None, None].expand(-1, 1, k)).squeeze(1)
            probability = torch.softmax(selected_scores.float() / (temperature or 1.0), -1)
            row = selected_scores.argmax(-1) if temperature == 0 else torch.multinomial(probability, 1).squeeze(-1)
            tokens.append(candidate_ids[:, position].gather(1, row[:, None]).squeeze(1))
            probabilities.append(probability)
        return torch.stack(tokens, -1), torch.stack(probabilities, -2), candidate_ids

    def sample_full_vocab(self, base_logits, hidden_states, anchor_token_ids, temperature):
        """Sample actual full-vocabulary conditionals along the generated path."""
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        # All transition scores are ready before the first random draw. The
        # remaining loop only gathers rows, normalizes, and selects tokens.
        lattice, candidate_ids = self.parallel_lattice(base_logits, hidden_states, anchor_token_ids)
        previous = anchor_token_ids
        tokens, logits, probabilities = [], [], []
        k = candidate_ids.size(-1)
        for position in range(base_logits.size(-2)):
            if position == 0:
                delta = lattice[:, position, 0]
            else:
                matches = candidate_ids[:, position - 1] == previous[:, None]
                row_index = matches.long().argmax(-1)
                delta = lattice[:, position].gather(1, row_index[:, None, None].expand(-1, 1, k)).squeeze(1)
                delta = torch.where(matches.any(-1)[:, None], delta, torch.zeros_like(delta))
            row = base_logits[:, position].float().scatter_add(-1, candidate_ids[:, position], delta.float())
            probability = torch.softmax(row / (temperature or 1.0), -1)
            token = row.argmax(-1) if temperature == 0 else torch.multinomial(probability, 1).squeeze(-1)
            tokens.append(token); logits.append(row); probabilities.append(probability)
            previous = token
        return torch.stack(tokens, -1), torch.stack(logits, -2), torch.stack(probabilities, -2)

    def forward(self, features, context, shared_ops):
        if context.inputs.mode == "features":
            return super().forward(features, context, shared_ops)
        draft_hidden_states = self.prediction_hidden(features, context)
        base_logits = shared_ops.lm_head(draft_hidden_states)
        output = DraftOutput(
            draft_hidden_states,
            base_logits=base_logits,
            draft_state=features.draft_state,
        )
        if self.full_vocab_residual:
            if context.inputs.mode == "sample":
                if context.inputs.anchor_token_ids is None:
                    raise ValueError("Selector sampling requires anchor_token_ids")
                shape = draft_hidden_states.shape[:-1]
                tokens, logits, probabilities = self.sample_full_vocab(
                    base_logits.flatten(0, 1), draft_hidden_states.flatten(0, 1),
                    context.inputs.anchor_token_ids.flatten(), context.inputs.temperature,
                )
                output.draft_token_ids = tokens.reshape(shape)
                output.draft_logits = logits.reshape(*shape, -1)
                output.draft_probabilities = probabilities.reshape(*shape, -1)
            else:
                if context.inputs.previous_token_ids is None:
                    raise ValueError("Selector scoring requires previous_token_ids")
                output.draft_logits, output.candidate_ids = self.full_vocab_logits(
                    base_logits, draft_hidden_states, context.inputs.previous_token_ids,
                )
            # candidate_logits deliberately stays None: generic verifier and
            # accuracy code must consume the full-vocabulary draft_logits.
            return output
        candidate_logits, candidate_ids = base_logits.topk(
            self.candidate_selector.top_k, dim=-1
        )
        if context.inputs.mode == "sample":
            tokens, draft_probabilities, _ = self.sample_candidates_parallel(
                base_logits=base_logits.flatten(0, 1),
                hidden_states=draft_hidden_states.flatten(0, 1),
                anchor_token_ids=context.inputs.anchor_token_ids.flatten(),
                temperature=context.inputs.temperature,
            )
            output.draft_token_ids = tokens.reshape(draft_hidden_states.shape[:-1])
            draft_probabilities = draft_probabilities.reshape_as(candidate_logits)
            output.draft_probabilities = base_logits.new_zeros(
                base_logits.shape, dtype=torch.float32
            ).scatter(-1, candidate_ids, draft_probabilities)
        else:
            if context.inputs.previous_token_ids is None:
                raise ValueError("Selector scoring requires previous_token_ids")
            output.candidate_logits = self.candidate_selector.score_candidates(
                candidate_ids=candidate_ids,
                unary_logits=candidate_logits.detach()
                if self.stop_gradient
                else candidate_logits,
                hidden_states=draft_hidden_states.detach()
                if self.stop_gradient
                else draft_hidden_states,
                predecessor_ids=context.inputs.previous_token_ids,
            )
        if context.inputs.mode == "sample":
            output.draft_logits = output.draft_probabilities.log() * (
                context.inputs.temperature or 1.0
            )
        output.candidate_ids = candidate_ids
        return output
