"""One FSDP forward owns all trainable fusion, backbone, heads and losses."""

from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from litespec.core.outputs import RatioTerms, TrainingOutput
from litespec.modeling.draft.composable.contracts import (
    BackboneOutput,
    ModelInputs,
    SharedOps,
)

from .batch import prepare_layout
from .component_reduction import combine_component_terms
from .metrics import acceptance_metric_terms, final_token_accuracy_terms
from .objectives.base import Supervision
from .objectives.selector import LossBranches


class ComposableTrainingModel(nn.Module):
    def __init__(
        self,
        draft_model,
        target_lm_head,
        target_embed_tokens,
        mask_token_id,
        num_anchors=32,
        objective_chunk_blocks=512,
        loss_process_group=None,
        **kwargs,
    ):
        super().__init__()
        removed = {"selector_loss_alpha", "selector_base_ce_alpha", "selector_warmup_ratio", "selector_ramp_ratio"} & kwargs.keys()
        if removed:
            raise ValueError("Removed Selector loss arguments; configure objective.base/final.weight instead")
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.mask_token_id = int(mask_token_id)
        self.num_anchors = int(num_anchors)
        self.objective_chunk_blocks = int(objective_chunk_blocks)
        self.loss_process_group = loss_process_group
        recipe = draft_model.config.composable_config
        self.loss_branches = LossBranches(recipe)
        draft_model.config.composable_config = self.loss_branches.recipe
        self.indexer_recipe = getattr(draft_model.config, "indexer_config", recipe.get("indexer"))
        self.selector_enabled = self.loss_branches.is_selector
        self.component_normalization = any(
            getattr(o, "needs_component_normalization", False)
            for o in self.loss_branches.objectives.values())
        if self.component_normalization and self.indexer_recipe is not None:
            raise ValueError("OPD-Survival with an indexer is not implemented")

    def forward(
        self,
        input_ids,
        target_hidden_states,
        loss_mask,
        target_last_hidden_states=None,
        layout=None,
        window_denominators=None,
        epoch_progress=0.0,
        collect_detailed_metrics=False,
        context_valid_mask=None,
        opd_accepted_lengths=None,
        opd_global_denominators=None,
        opd_reference_samples=None,
        metadata_only=False,
        **removed_loss_arguments,
    ):
        model = self.draft_model
        if metadata_only and not self.component_normalization:
            raise ValueError("Metadata forward requires a component objective")
        if removed_loss_arguments:
            raise ValueError("Removed loss arguments: " + ", ".join(removed_loss_arguments)
                             + "; configure objective.base/final.weight instead")
        # Teacher-free objectives do not acquire a teacher requirement just for
        # logging. Distribution acceptance diagnostics require available logits.
        acceptance_metrics = collect_detailed_metrics and target_last_hidden_states is not None
        if layout is None:
            layout = prepare_layout(
                input_ids,
                loss_mask,
                model.block_size,
                self.num_anchors,
                complete_blocks=self.loss_branches.requires_complete_blocks,
            )
        batch_size, num_anchors = layout.anchor_positions.shape
        noise_ids = input_ids.new_full(
            (batch_size, num_anchors, model.block_size), self.mask_token_id
        )
        noise_ids[..., 0] = input_ids.gather(1, layout.anchor_positions).masked_fill(
            ~layout.block_keep_mask, self.mask_token_id
        )
        inputs = ModelInputs(
            target_hidden_states=target_hidden_states,
            draft_embeddings=self.embed_tokens(noise_ids.flatten(1)),
            anchor_positions=layout.anchor_positions,
            block_keep_mask=layout.block_keep_mask,
            mode="features",
            context_valid_mask=context_valid_mask,
            prediction_mask=layout.prediction_mask,
            collect_indexer_teacher=self.indexer_recipe is not None,
        )
        prediction = model(
            inputs, shared_ops=SharedOps(self.embed_tokens, self.lm_head)
        )
        index_aux = prediction.indexer_aux
        if self.indexer_recipe is not None and self.indexer_recipe["stage"] == "warmup":
            if index_aux is None:
                raise ValueError("Warmup must return indexer auxiliary terms")
            numerator, denominator = index_aux.numerator, index_aux.denominator
            if window_denominators is not None:
                reference, _, global_index = window_denominators.to(numerator).unbind()
                numerator = reference * numerator / global_index.clamp_min(1)
                denominator = numerator.new_tensor(batch_size)
            return TrainingOutput(numerator / denominator.clamp_min(1), RatioTerms(numerator,denominator),
                {"objective/indexer_kl": RatioTerms(index_aux.numerator.detach(),index_aux.denominator.detach())})
        # Every checkpoint chunk owns whole blocks, preserving prefix objectives.
        chunk_size = self.objective_chunk_blocks or num_anchors
        zero = prediction.draft_hidden_states.float().sum() * 0.0
        names = list(self.loss_branches.objectives)
        numerators = {name: zero for name in names}
        denominators = {name: zero.detach().clone() for name in names}
        metric_numerators = {}
        correct_num = zero.detach().clone()
        accuracy_den = zero.detach().clone()
        walk_num = zero.detach().clone()
        expected_num = zero.detach().clone()
        block_den = zero.detach().clone()
        component_terms = {name: {} for name in names}
        component_sum_metrics = {}
        accepted_chunks = {name: [] for name in names
                           if getattr(self.loss_branches.objectives[name],
                                      "needs_component_normalization", False)}
        for start in range(0, num_anchors, chunk_size):
            end = min(start + chunk_size, num_anchors)
            block_hidden = prediction.backbone_hidden_states.reshape(
                batch_size, num_anchors, model.block_size, -1
            )[:, start:end]
            labels = layout.labels[:, start:end]
            previous = layout.previous_token_ids[:, start:end]
            mask = layout.prediction_mask[:, start:end]
            teacher_hidden = None
            if target_last_hidden_states is not None and (self.loss_branches.needs_teacher or acceptance_metrics):
                indices = (layout.label_indices[:, start:end] - 1).clamp_min(0)
                teacher_hidden = torch.gather(
                    target_last_hidden_states, 1,
                    indices.flatten(1)[..., None].expand(-1, -1, model.config.hidden_size),
                ).reshape(*labels.shape, -1)

            # Bind all layout slices: checkpoint replay must not read a later chunk.
            replay = (None if opd_accepted_lengths is None else
                      {name: r[:, start:end] for name, r in opd_accepted_lengths.items()})

            def make_chunk(labels, previous, mask, teacher_hidden, anchors, keep, replay):
                def run_chunk(hidden):
                    chunk_inputs = replace(inputs, anchor_positions=anchors,
                        block_keep_mask=keep, previous_token_ids=previous, mode="teacher_forcing")
                    output = model.proposal(BackboneOutput(hidden.flatten(1, 2)),
                        SimpleNamespace(inputs=chunk_inputs), SharedOps(self.embed_tokens, self.lm_head))
                    teacher_logits = None
                    if not metadata_only and (self.loss_branches.needs_teacher or acceptance_metrics):
                        if teacher_hidden is None:
                            raise ValueError("Active loss/metrics require target_last_hidden_states")
                        with torch.no_grad():
                            teacher_logits = self.lm_head(teacher_hidden)
                    supervision = Supervision(labels=labels, prediction_mask=mask,
                                              target_logits=teacher_logits, epoch_progress=epoch_progress)
                    if self.component_normalization:
                        components, accepted, sums = {}, {}, {}
                        for name in names:
                            objective = self.loss_branches.objectives[name]
                            branch_prediction = (replace(output, draft_logits=output.base_logits)
                                                 if name == 'base' else output)
                            r = None if replay is None else replay.get(name)
                            branch_supervision = replace(supervision, opd_accepted_lengths=r)
                            if metadata_only:
                                if getattr(objective, 'needs_component_normalization', False):
                                    r = objective.accepted_lengths(branch_prediction, branch_supervision)
                                    counts = objective.component_denominators(r, mask)
                                    accepted[name] = r
                                else:
                                    counts = {'token': objective.native_denominator(
                                        SimpleNamespace(prediction_mask=mask,
                                                        sample_prediction_counts=mask.flatten(1).sum(-1)))}
                                components[name] = {key: (hidden.float().sum() * 0., den)
                                                    for key, den in counts.items()}
                            else:
                                terms = objective(branch_prediction, branch_supervision)
                                native = terms.component_terms or {'token': terms.loss_terms}
                                components[name] = {key: (term.numerator, term.denominator)
                                                    for key, term in native.items()}
                                sums.update({name + '/' + key: value.detach()
                                             for key, value in terms.sum_metrics.items()})
                        with torch.no_grad():
                            correct, count = final_token_accuracy_terms(output, labels, mask)
                            if acceptance_metrics and not metadata_only:
                                walk, expected, blocks = acceptance_metric_terms(
                                    model.proposal, output, teacher_logits, labels, previous, mask)
                            else:
                                walk = expected = blocks = mask.new_zeros(())
                        return {'components': components, 'accepted': accepted,
                                'sums': sums, 'accuracy': (correct, count),
                                'acceptance': (walk, expected, blocks)}
                    values = []
                    for name in names:
                        terms = self.loss_branches.branch_output(name, output, supervision)
                        values.extend([terms.loss_terms.numerator, terms.loss_terms.denominator])
                        # Fixed slots keep checkpoint return structure deterministic.
                        for key in ("bv/mathematical_loss", "tv_survival/tv", "tv_survival/survival"):
                            metric = terms.ratio_metrics.get(key)
                            values.append(metric.numerator if metric is not None else mask.new_zeros(()))
                    with torch.no_grad():
                        correct, count = final_token_accuracy_terms(output, labels, mask)
                        if acceptance_metrics:
                            walk, expected, blocks = acceptance_metric_terms(
                                model.proposal, output, teacher_logits, labels, previous, mask)
                        else:
                            walk = expected = blocks = mask.new_zeros(())
                    return (*values, correct, count, walk, expected, blocks)
                return run_chunk

            operation = make_chunk(labels, previous, mask, teacher_hidden,
                                   layout.anchor_positions[:, start:end], layout.block_keep_mask[:, start:end], replay)
            values = (checkpoint(operation, block_hidden, use_reentrant=False)
                      if self.training and torch.is_grad_enabled() else operation(block_hidden))
            if self.component_normalization:
                for name, items in values['components'].items():
                    for key, (num, den) in items.items():
                        old = component_terms[name].get(key)
                        component_terms[name][key] = RatioTerms(
                            num if old is None else old.numerator + num,
                            den if old is None else old.denominator + den)
                for name, r in values['accepted'].items():
                    accepted_chunks[name].append(r.detach())
                for key, value in values['sums'].items():
                    component_sum_metrics[key] = component_sum_metrics.get(key, zero.detach()) + value
                correct, count = values['accuracy']
                walk, expected, blocks = values['acceptance']
                correct_num, accuracy_den = correct_num + correct, accuracy_den + count
                walk_num, expected_num, block_den = walk_num + walk, expected_num + expected, block_den + blocks
                continue
            for i, name in enumerate(names):
                offset = i * 5
                numerators[name] = numerators[name] + values[offset]
                if not self.loss_branches.objectives[name].sample_reduction or start == 0:
                    denominators[name] = denominators[name] + values[offset + 1]
                for j, key in enumerate(("bv/mathematical_loss", "tv_survival/tv", "tv_survival/survival")):
                    metric_numerators[name, key] = metric_numerators.get((name, key), zero.detach()) + values[offset + 2 + j]
            correct_num = correct_num + values[-5]
            accuracy_den = accuracy_den + values[-4]
            walk_num = walk_num + values[-3]
            expected_num = expected_num + values[-2]
            block_den = block_den + values[-1]

        if self.component_normalization:
            if metadata_only:
                return {'accepted_lengths': {name: torch.cat(chunks, dim=1)
                                              for name, chunks in accepted_chunks.items()},
                        'component_denominators': {name: {key: term.denominator.detach()
                            for key, term in items.items()} for name, items in component_terms.items()},
                        'sample_count': batch_size}
            if self.indexer_recipe is not None:
                raise ValueError("OPD-Survival with an indexer is not implemented")
            numerator, denominator, component_metrics = combine_component_terms(
                component_terms, self.loss_branches.objectives, self.loss_branches.weights,
                batch_size, self.loss_process_group, opd_global_denominators, opd_reference_samples)
            metrics = {'hard_label/token_accuracy': RatioTerms(correct_num.detach(), accuracy_den.detach()),
                       **component_metrics}
            if acceptance_metrics:
                metrics['hard_label/block_accepted_length'] = RatioTerms(walk_num.detach(), block_den.detach())
                metrics['expected_accept_len'] = RatioTerms(expected_num.detach(), block_den.detach())
            return TrainingOutput(numerator / denominator.clamp_min(1e-12),
                                  RatioTerms(numerator, denominator.detach()), metrics,
                                  sum_metrics=component_sum_metrics)
        if self.selector_enabled:
            # Order is reference sample count, final denominator, base denominator.
            if window_denominators is None:
                reference = zero.new_tensor(batch_size)
                global_denominators = denominators
            else:
                reference, final_global, base_global = window_denominators.to(zero).unbind()
                global_denominators = {"final": final_global, "base": base_global}
            numerator = reference * sum(
                self.loss_branches.weights[name] * numerators[name]
                / global_denominators[name].clamp_min(1e-12) for name in names)
            denominator = zero.new_tensor(batch_size)
        else:
            numerator, denominator = numerators['final'], denominators['final']
        if self.indexer_recipe is not None:
            if index_aux is None:
                raise ValueError("DSA training requires auxiliary statistics")
            if window_denominators is None:
                reference = zero.new_tensor(batch_size)
                main_global, index_global = denominators['final'], index_aux.denominator
            else:
                reference, main_global, index_global = window_denominators.to(zero).unbind()
            numerator = reference * (numerators['final'] / main_global.clamp_min(1e-12)
                + self.indexer_recipe.get("loss_weight", 1.0) * index_aux.numerator / index_global.clamp_min(1))
            denominator = zero.new_tensor(batch_size)
        metrics = {"hard_label/token_accuracy": RatioTerms(correct_num.detach(), accuracy_den.detach())}
        for name in names:
            metrics['objective/' + name] = RatioTerms(numerators[name].detach(), denominators[name].detach())
            objective = self.loss_branches.objectives[name]
            for key in ("bv/mathematical_loss", "tv_survival/tv", "tv_survival/survival"):
                if (key.startswith('bv/') and objective.recipe['type'] == 'bv') or (key.startswith('tv_survival/') and objective.recipe['type'] == 'tv_survival'):
                    metrics[name + '/' + key] = RatioTerms(metric_numerators[name, key].detach(), denominators[name].detach())
        if index_aux is not None:
            metrics['objective/indexer_kl'] = RatioTerms(index_aux.numerator.detach(), index_aux.denominator.detach())
        if acceptance_metrics:
            metrics['hard_label/block_accepted_length'] = RatioTerms(walk_num.detach(), block_den.detach())
            metrics['expected_accept_len'] = RatioTerms(expected_num.detach(), block_den.detach())
        return TrainingOutput(numerator / denominator.clamp_min(1e-12), RatioTerms(numerator, denominator.detach()), metrics)
