"""Standard TV plus a differentiable prefix-survival surrogate on fixed data.

Only complete supervised blocks contribute. K counts predictions, excluding the
anchor. This teacher-forced surrogate is not an on-policy accepted-length bound.
"""

import math

import torch

from .base import Objective


class TVSurvivalObjective(Objective):
    needs_teacher = True
    requires_complete_blocks = True

    def __init__(self, recipe, proposal_type="parallel"):
        super().__init__(recipe, proposal_type)
        weight = recipe.get("survival_weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("survival_weight must be finite and nonnegative")
        self.survival_weight = float(weight)
        if not math.isfinite(self.survival_weight) or self.survival_weight < 0:
            raise ValueError("survival_weight must be finite and nonnegative")
        # This block objective ignores fixed decay. A sibling Selector loss
        # may use the same outer gamma for its token-normalized objective.

    def native_denominator(self, layout):
        mask = layout.prediction_mask
        complete = (mask > 0.5).all(-1)
        return complete.sum().float() * mask.size(-1)

    def forward(self, prediction, supervision):
        if supervision.target_logits is None:
            raise ValueError("tv_survival requires frozen teacher logits")
        draft = torch.softmax(prediction.draft_logits.float(), dim=-1)
        target = torch.softmax(supervision.target_logits.detach().float(), dim=-1)
        tv = 0.5 * (target - draft).abs().sum(-1)
        overlap = (1.0 - tv).clamp(0.0, 1.0)
        survival_loss = 1.0 - overlap.cumprod(dim=-1)
        complete = (supervision.prediction_mask > 0.5).all(-1)
        valid = complete[..., None].to(tv)
        tv_numerator = (tv * valid).sum()
        survival_numerator = (survival_loss * valid).sum()
        denominator = complete.sum().to(tv) * tv.size(-1)
        return self.output(
            tv_numerator + self.survival_weight * survival_numerator,
            denominator,
            **{
                "tv_survival/tv": tv_numerator,
                "tv_survival/survival": survival_numerator,
            },
        )
