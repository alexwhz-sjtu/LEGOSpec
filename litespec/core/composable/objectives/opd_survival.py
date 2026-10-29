"""Offline empirical OPD-Survival with separately normalized KL regions.

Accepted prefixes come from detached teacher-forced greedy matches. The
reachable survival prefix includes the first failure: H=min(r+1,K). This is a
fixed-data surrogate; it does not claim on-policy target verification.
"""

import torch

from litespec.core.outputs import RatioTerms, TrainingOutput
from .base import Objective


class OPDSurvivalObjective(Objective):
    needs_teacher = True
    requires_complete_blocks = True
    needs_component_normalization = True

    def __init__(self, recipe, proposal_type="parallel"):
        super().__init__(recipe, proposal_type)
        self.component_weights = {
            "acc": 1.0,
            "rej": self.recipe["beta_rej"],
            "survival": self.recipe["survival_weight"],
        }
        self.rejection_decay = self.recipe["rejection_decay"]

    @staticmethod
    def accepted_lengths(prediction, supervision):
        """Detached matches before the first failure, excluding the anchor."""
        if prediction.draft_logits is None:
            raise ValueError("opd_survival requires full-vocabulary draft logits")
        if prediction.draft_logits.shape[:-1] != supervision.labels.shape:
            raise ValueError("opd_survival labels and draft logits must align")
        if supervision.prediction_mask.shape != supervision.labels.shape:
            raise ValueError("opd_survival prediction mask and labels must align")
        complete = (supervision.prediction_mask > .5).all(-1)
        correct = prediction.draft_logits.detach().argmax(-1) == supervision.labels
        return correct.long().cumprod(-1).sum(-1).masked_fill(~complete, 0)

    def component_masks(self, accepted_lengths, prediction_mask):
        """Masks shared by the metadata prepass and differentiable objective."""
        complete = (prediction_mask > .5).all(-1)
        if accepted_lengths.shape != complete.shape:
            raise ValueError("opd_survival accepted lengths must match block axes")
        if accepted_lengths.dtype not in (torch.uint8, torch.int8, torch.int16,
                                           torch.int32, torch.int64):
            raise ValueError("opd_survival accepted lengths must be integer counts")
        k = prediction_mask.size(-1)
        r = accepted_lengths.detach()
        if ((r < 0) | (r > k)).any():
            raise ValueError("opd_survival accepted lengths must be within [0, K]")
        position = torch.arange(1, k + 1, device=prediction_mask.device)
        accepted = complete[..., None] & (position <= r[..., None])
        rejected = complete[..., None] & (position > r[..., None])
        reachable = complete[..., None] & (position <= (r + 1).clamp_max(k)[..., None])
        weights = self.rejection_decay ** (position.float() - 1.)
        return complete, accepted, rejected.to(torch.float32) * weights, reachable

    def component_denominators(self, accepted_lengths, prediction_mask):
        complete, accepted, rejected_weights, _ = self.component_masks(
            accepted_lengths, prediction_mask)
        return {
            "acc": accepted.sum().float(),
            "rej": rejected_weights.sum(),
            "survival": complete.sum().float() * prediction_mask.size(-1),
        }

    def native_denominator(self, layout):
        # Reference capacity only. Training must combine component_terms with
        # their own global optimizer-window denominators, never this one alone.
        return (layout.prediction_mask > .5).all(-1).sum().float() * layout.prediction_mask.size(-1)

    def forward(self, prediction, supervision):
        if supervision.target_logits is None:
            raise ValueError("opd_survival requires frozen teacher logits")
        if prediction.draft_logits is None:
            raise ValueError("opd_survival requires full-vocabulary draft logits")
        if prediction.draft_logits.shape != supervision.target_logits.shape:
            raise ValueError("opd_survival draft and teacher logits must align")
        if prediction.draft_logits.shape[:-1] != supervision.labels.shape:
            raise ValueError("opd_survival labels and logits must align")
        if supervision.prediction_mask.shape != supervision.labels.shape:
            raise ValueError("opd_survival prediction mask and labels must align")
        log_q = torch.log_softmax(prediction.draft_logits.float(), -1)
        log_p = torch.log_softmax(supervision.target_logits.detach().float(), -1)
        q, p = log_q.exp(), log_p.exp()
        r = getattr(supervision, "opd_accepted_lengths", None)
        if r is None:
            r = self.accepted_lengths(prediction, supervision)
        complete, accepted, rejected_weights, reachable = self.component_masks(
            r, supervision.prediction_mask)
        forward_kl = (p * (log_p - log_q)).sum(-1)
        reverse_kl = (q * (log_q - log_p)).sum(-1)
        overlap = (1. - .5 * (p - q).abs().sum(-1)).clamp(0., 1.)
        prefix = overlap.cumprod(-1)
        capacity = complete.sum().to(prefix) * prefix.size(-1)
        components = {
            "acc": RatioTerms((forward_kl * accepted).sum(), accepted.sum().to(prefix)),
            "rej": RatioTerms((reverse_kl * rejected_weights).sum(), rejected_weights.sum()),
            "survival": RatioTerms(capacity - (prefix * reachable).sum(), capacity),
        }
        loss = sum(self.component_weights[name] * terms.numerator
                   / terms.denominator.clamp_min(1e-12)
                   for name, terms in components.items())
        reference = loss.new_tensor(supervision.labels.size(0))
        metrics = {"opd_survival/" + name: terms.detached()
                   for name, terms in components.items()}
        accepted_count = accepted.sum().to(prefix)
        raw_counts = {
            "opd_survival/accepted_positions": accepted_count.detach(),
            "opd_survival/rejected_positions": (capacity - accepted_count).detach(),
            "opd_survival/reachable_positions": reachable.sum().to(prefix).detach(),
        }
        return TrainingOutput(loss, RatioTerms(loss * reference, reference),
                              metrics, sum_metrics=raw_counts,
                              component_terms=components)
