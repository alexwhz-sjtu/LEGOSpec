"""Synchronous anchor OPD. Tokens, uniforms and verifier boundaries are replay data.

Shared sample/epoch layouts are supplied by the adapter, independently of all
model RNG. Each collected batch draws one checkpointed seed for proposal and
verifier generators. Legacy calls without a layout also derive their anchors
from that seed. Replaying a collected record draws no new sampling randomness.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint

from litespec.core.outputs import RatioTerms, TrainingOutput
from litespec.modeling.draft.composable.contracts import BackboneOutput, ModelInputs, SharedOps
from .batch import prepare_layout
from .component_reduction import combine_component_terms
from .objectives.base import Supervision
from .objectives.opd_survival import distribution_terms


_compiled_statistics = None


def full_batch_statistics(draft_logits, target_logits):
    global _compiled_statistics
    if not draft_logits.is_cuda:
        return distribution_terms(draft_logits, target_logits)
    if _compiled_statistics is None:
        _compiled_statistics = torch.compile(distribution_terms, fullgraph=True, dynamic=False)
    return _compiled_statistics(draft_logits, target_logits)


@dataclass
class SampledOPDBatch:
    layout: object
    context_hidden: torch.Tensor
    context_valid: torch.Tensor
    draft_ids: torch.Tensor
    sampled_logq: torch.Tensor
    uniforms: torch.Tensor
    teacher_hidden: torch.Tensor
    accepted_lengths: torch.Tensor
    seed: int
    label_teacher_hidden: torch.Tensor | None = None

    def to(self, device):
        return replace(self, layout=self.layout.to(device), **{
            key: value.to(device, non_blocking=True)
            for key, value in vars(self).items() if isinstance(value, torch.Tensor)})


def _generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def sample_proposal(model, hidden, anchors, lm_head, generator):
    """Compact full-vocabulary T=1 sampling, without retained vocab probabilities."""
    proposal = model.proposal
    logits = lm_head(hidden.detach())
    if model.config.composable_config['proposal']['type'] == 'parallel':
        logq = torch.log_softmax(logits.float(), -1)
        ids = torch.multinomial(logq.exp().reshape(-1, logq.size(-1)), 1,
                                generator=generator).reshape(logq.shape[:-1])
        return ids, logq.gather(-1, ids[..., None]).squeeze(-1)
    head = proposal.markov_head
    state = hidden.new_zeros(*hidden.shape[:2], head.markov_rank)
    previous, ids, selected = anchors, [], []
    for k in range(hidden.size(-2)):
        state, bias = head._rnn_step(state, head.get_prev_embeddings(previous), hidden[..., k, :].detach())
        logq = torch.log_softmax((logits[..., k, :] + bias).float(), -1)
        token = torch.multinomial(logq.exp().reshape(-1, logq.size(-1)), 1,
                                  generator=generator).reshape(previous.shape)
        ids.append(token)
        selected.append(logq.gather(-1, token[..., None]).squeeze(-1))
        previous = token
    return torch.stack(ids, -1), torch.stack(selected, -1)


@torch.no_grad()
def verify_sampled_blocks(teacher_logits, ids, sampled_logq, uniforms, keep):
    if teacher_logits.shape[:-1] != ids.shape or ids.shape != sampled_logq.shape or ids.shape != uniforms.shape:
        raise ValueError('Verifier logits, tokens, proposal logq and uniforms must align')
    if not torch.isfinite(sampled_logq).all() or ((uniforms < 0) | (uniforms >= 1)).any():
        raise ValueError('Verifier requires finite sampled logq and uniforms in [0,1)')
    scores = teacher_logits.float()
    logp = scores.gather(-1, ids[..., None]).squeeze(-1) - scores.logsumexp(-1)
    accepted = uniforms.log() < (logp - sampled_logq).clamp_max(0.)
    return accepted.long().cumprod(-1).sum(-1).masked_fill(~keep, 0)


def forward_anchor_opd(wrapper, input_ids, target_hidden_states, loss_mask, *,
                       context_valid_mask, target_last_hidden_states=None,
                       opd_scorer=None, sampled_opd_batch=None,
                       opd_global_denominators=None, opd_reference_samples=None,
                       metadata_only=False, layout=None, collect_detailed_metrics=False, **unused):
    model = wrapper.draft_model
    recipe = model.config.composable_config
    if recipe['proposal']['type'] not in ('parallel', 'rnn') or recipe['objective']['type'] != 'opd_survival':
        raise ValueError('Sampled anchor OPD requires Parallel/RNN and opd_survival')
    if wrapper.objective_chunk_blocks != 0:
        raise ValueError('Sampled anchor OPD processes all anchors: objective_chunk_blocks must be 0')
    if any(float(getattr(module, 'attention_dropout', 0.)) != 0 for module in model.modules()):
        raise ValueError('Sampled anchor OPD requires zero attention dropout')
    record = sampled_opd_batch
    prefix = None
    if record is None:
        if opd_scorer is None:
            raise ValueError('Fresh anchor OPD collection requires a target scorer')
        seed = int(torch.randint(2**31 - 3, (), device=input_ids.device).item())
        if layout is None:
            layout = prepare_layout(input_ids, loss_mask, model.block_size, wrapper.num_anchors,
                                    complete_blocks=True, generator=_generator(input_ids.device, seed))
        else:
            layout = layout.to(input_ids.device)
        context_valid_mask = context_valid_mask.bool()
        if target_hidden_states is None:
            prefix = opd_scorer.prefill_anchors(input_ids, context_valid_mask,
                layout.anchor_positions, layout.block_keep_mask, capture_layer_ids=model.target_layer_ids)
            target_hidden_states = torch.cat([prefix.context_hidden[i] for i in model.target_layer_ids], -1)
            prefix.context_hidden.clear()
            context_valid_mask = prefix.valid
    else:
        layout = record.layout
        target_hidden_states, context_valid_mask = record.context_hidden, record.context_valid
        seed = record.seed
    n, a = layout.anchor_positions.shape
    anchor_ids = input_ids.gather(1, layout.anchor_positions)
    noise = input_ids.new_full((n, a, model.block_size), wrapper.mask_token_id)
    noise[..., 0] = anchor_ids.masked_fill(~layout.block_keep_mask, wrapper.mask_token_id)
    inputs = ModelInputs(target_hidden_states=target_hidden_states,
        draft_embeddings=wrapper.embed_tokens(noise.flatten(1)),
        anchor_positions=layout.anchor_positions, block_keep_mask=layout.block_keep_mask,
        context_valid_mask=context_valid_mask, prediction_mask=layout.prediction_mask, mode='features')
    shared = SharedOps(wrapper.embed_tokens, wrapper.lm_head)
    try:
        prediction = model(inputs, shared_ops=shared)
        block_hidden = prediction.backbone_hidden_states.reshape(n, a, model.block_size, -1)
        if record is None:
            ids, logq = sample_proposal(model, prediction.draft_hidden_states, anchor_ids,
                                        wrapper.lm_head, _generator(input_ids.device, seed + 1))
            if prefix is None:
                prefix = opd_scorer.prefill_anchors(input_ids, context_valid_mask,
                    layout.anchor_positions, layout.block_keep_mask)
            first = None
            if target_last_hidden_states is not None and not getattr(opd_scorer, 'uses_live_anchor_hidden', False):
                first = target_last_hidden_states.gather(1, layout.anchor_positions[..., None].expand(
                    -1, -1, target_last_hidden_states.size(-1)))
            teacher_hidden = opd_scorer.score_sampled_blocks(prefix, ids, cached_anchor_hidden=first)
            label_source = (target_last_hidden_states if target_last_hidden_states is not None
                            else getattr(prefix, 'full_hidden', None))
            label_teacher = None
            if label_source is not None:
                label_indices = (layout.label_indices - 1).clamp(0, label_source.size(1) - 1)
                label_teacher = label_source.gather(1, label_indices.flatten(1)[..., None].expand(
                    -1, -1, label_source.size(-1))).reshape(*layout.labels.shape, -1)
            prefix.release()
            prefix = None
            uniforms = torch.rand(ids.shape, device=ids.device, generator=_generator(ids.device, seed + 2))
            with torch.no_grad():
                r = verify_sampled_blocks(wrapper.lm_head(teacher_hidden), ids, logq,
                                          uniforms, layout.block_keep_mask)
            record = SampledOPDBatch(layout, target_hidden_states, context_valid_mask,
                                     ids, logq, uniforms, teacher_hidden, r, seed, label_teacher)
    finally:
        if prefix is not None:
            prefix.release()
    objective = wrapper.loss_branches.objectives['final']
    if metadata_only:
        return {'sampled_opd_batch': record,
                'component_denominators': {'final': objective.component_denominators(
                    record.accepted_lengths, layout.prediction_mask)}, 'sample_count': n}
    previous = torch.cat((anchor_ids[..., None], record.draft_ids[..., :-1]), -1)
    replay_inputs = replace(inputs, previous_token_ids=previous, mode='teacher_forcing')

    def replay(hidden):
        output = model.proposal(BackboneOutput(hidden.flatten(1, 2)),
                               SimpleNamespace(inputs=replay_inputs), shared)
        with torch.no_grad():
            teacher_logits = wrapper.lm_head(record.teacher_hidden)
        terms = objective(output, Supervision(labels=record.draft_ids,
            prediction_mask=layout.prediction_mask, target_logits=teacher_logits,
            opd_accepted_lengths=record.accepted_lengths,
            opd_distribution_terms=full_batch_statistics(output.draft_logits, teacher_logits)))
        return {name: (term.numerator, term.denominator) for name, term in terms.component_terms.items()}

    values = (checkpoint(replay, block_hidden, use_reentrant=False)
              if wrapper.training and torch.is_grad_enabled() else replay(block_hidden))
    components = {'final': {name: RatioTerms(*value) for name, value in values.items()}}
    numerator, denominator, metrics = combine_component_terms(components,
        wrapper.loss_branches.objectives, wrapper.loss_branches.weights, n,
        wrapper.loss_process_group, opd_global_denominators, opd_reference_samples)
    blocks = layout.block_keep_mask.sum().float()
    metrics['rollout/accepted_length'] = RatioTerms(record.accepted_lengths.sum().float(), blocks)
    metrics['rollout/reachable_positions'] = RatioTerms(
        ((record.accepted_lengths + 1).clamp_max(model.block_size - 1) * layout.block_keep_mask).sum().float(), blocks)
    if collect_detailed_metrics:
        from .metrics import acceptance_metric_terms, final_token_accuracy_terms

        if record.label_teacher_hidden is None:
            raise ValueError('Label acceptance diagnostics require source-text teacher hidden states')
        with torch.no_grad():
            # Match ordinary SFT exactly: gold predecessors for conditional
            # rows/expected length, greedy free-running proposal vs data labels
            # for the walk. Sampled OPD tokens never become metric labels.
            label_inputs = replace(inputs, previous_token_ids=layout.previous_token_ids,
                                   mode='teacher_forcing')
            label_prediction = model.proposal(BackboneOutput(block_hidden.detach().flatten(1, 2)),
                SimpleNamespace(inputs=label_inputs), shared)
            label_logits = wrapper.lm_head(record.label_teacher_hidden)
            walk, expected, count = acceptance_metric_terms(model.proposal, label_prediction,
                label_logits, layout.labels, layout.previous_token_ids, layout.prediction_mask)
            correct, tokens = final_token_accuracy_terms(label_prediction, layout.labels, layout.prediction_mask)
        metrics['hard_label/block_accepted_length'] = RatioTerms(walk.float(), count.float())
        metrics['expected_accept_len'] = RatioTerms(expected.float(), count.float())
        metrics['hard_label/token_accuracy'] = RatioTerms(correct, tokens)
    return TrainingOutput(numerator / denominator.clamp_min(1),
                          RatioTerms(numerator, denominator), metrics)
