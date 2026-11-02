"""DFlash2 categorical CE on strict unary top-K support.

Gold tokens outside the support contribute zero Selector CE. The denominator
remains ALL valid weighted prediction tokens, matching the combined DFlash2
base + Selector training denominator, rather than reweighting by recall.
"""
import torch
from torch.nn import functional as F
from .base import Objective


class CandidateCEObjective(Objective):
    def forward(self, prediction, supervision):
        scores = prediction.candidate_logits.float()
        matches = prediction.candidate_ids == supervision.labels[..., None]
        covered = matches.any(-1)
        local_labels = matches.long().argmax(-1)
        losses = F.cross_entropy(scores.flatten(0, -2), local_labels.flatten(),
                                 reduction='none').reshape_as(supervision.prediction_mask)
        weights = self.loss_weights(supervision)
        # Multiplication retains a zero autograd path even for all-miss batches.
        return self.output((losses * weights * covered).sum(), weights.sum(),
                           candidate_recall=(weights * covered).sum())
