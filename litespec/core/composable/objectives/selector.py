"""Configured base/final branches; there is no candidate-only training loss."""

from copy import deepcopy
from dataclasses import replace

from torch import nn

from litespec.modeling.draft.composable.loss_config import normalize_training_recipe
from . import build_objective


class LossBranches(nn.Module):
    def __init__(self, recipe):
        super().__init__()
        self.recipe = normalize_training_recipe(recipe)
        self.is_selector = self.recipe['proposal']['type'] == 'selector'
        outer = self.recipe['objective']
        definitions = (
            [('final', outer['final']), ('base', outer['base'])]
            if self.is_selector else [('final', {**outer, 'weight': 1.0})]
        )
        self.weights = {}
        self.objectives = nn.ModuleDict()
        for name, definition in definitions:
            item = deepcopy(definition)
            weight = item.pop('weight')
            self.weights[name] = weight
            if not weight:
                continue
            item['loss_decay_gamma'] = outer['loss_decay_gamma']
            proposal_type = (
                'parallel' if name == 'base' else
                'selector_full_vocab' if self.is_selector else self.recipe['proposal']['type']
            )
            self.objectives[name] = build_objective(item, proposal_type)
        self.needs_teacher = any(o.needs_teacher for o in self.objectives.values())
        self.requires_complete_blocks = any(o.requires_complete_blocks for o in self.objectives.values())
        self.needs_bv = any(o.recipe['type'] == 'bv' for o in self.objectives.values())

    def branch_output(self, name, prediction, supervision):
        if name == 'base':
            prediction = replace(prediction, draft_logits=prediction.base_logits)
        return self.objectives[name](prediction, supervision)

    def native_denominators(self, layout):
        zero = layout.prediction_mask.new_zeros(())
        return {name: (self.objectives[name].native_denominator(layout)
                       if name in self.objectives else zero)
                for name in self.weights}
