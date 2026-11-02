"""EAGLE3 target-feature fusion, without an additional context norm."""

from torch import nn

from ..contracts import ContextOutput
from .base import Context


class Eagle3Context(Context):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(
            len(config.composable_config["context"]["target_layer_ids"])
            * config.hidden_size, config.hidden_size, bias=False,
        )

    def project(self, hidden):
        return self.projection(hidden.flatten(2))

    def forward(self, inputs):
        return ContextOutput([self.project(inputs.target_hidden_states)], inputs)
