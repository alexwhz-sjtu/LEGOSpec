"""Negative log overlap, globally normalized by decayed token weights."""

import torch

from .base import Objective


class NLAObjective(Objective):
    needs_teacher = True

    def forward(self, prediction, supervision):
        draft_probabilities = torch.softmax(prediction.draft_logits.float(), -1)
        target_probabilities = torch.softmax(
            supervision.target_logits.detach().float(), -1
        )
        token_losses = (
            -torch.minimum(target_probabilities, draft_probabilities)
            .sum(-1)
            .clamp_min(1e-5)
            .log()
        )
        weights = self.loss_weights(supervision)
        return self.output((token_losses * weights).sum(), weights.sum())
