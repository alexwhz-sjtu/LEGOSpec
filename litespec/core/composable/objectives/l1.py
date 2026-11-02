"""Full distribution L1 distance, with the shared weighted-token reduction."""
from .tv import TVObjective


class L1Objective(TVObjective):
    def forward(self, prediction, supervision):
        output = super().forward(prediction, supervision)
        return self.output(2.0 * output.loss_terms.numerator, output.loss_terms.denominator)
