"""DFlare scalar/channelwise fusion; original scalar path: AngelSpec d3412bed."""

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from ..contracts import ContextOutput
from .base import Context


class HiddenGatedContext(Context):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        context = config.composable_config["context"]
        self.target_count = len(context["target_layer_ids"])
        self.channelwise = context.get("fusion") == "channel_softmax_shared_projection"
        shape = (config.num_hidden_layers, self.target_count)
        if self.channelwise:
            shape += (self.hidden_size,)
        self.layer_fusion_weights = nn.Parameter(
            torch.empty(shape)
        )
        if self.channelwise:
            self.shared_projection = nn.Linear(
                self.hidden_size, self.hidden_size, bias=False
            )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.reset_fusion_prior()

    def reset_fusion_prior(self):
        depth, count = self.layer_fusion_weights.shape[:2]
        with torch.no_grad():
            for index in range(depth):
                preferred = round(index / max(depth - 1, 1) * (count - 1))
                prior = -2.0 * (
                    torch.arange(count, device=self.layer_fusion_weights.device)
                    - preferred
                ).abs()
                if self.channelwise:
                    prior = prior[:, None].expand(count, self.hidden_size)
                self.layer_fusion_weights[index].copy_(prior)

    def forward(self, inputs):
        target_hidden_states = inputs.target_hidden_states.reshape(
            *inputs.target_hidden_states.shape[:2], self.target_count, self.hidden_size
        )
        if self.channelwise:
            # One distribution over target layers per draft layer/channel.
            # The single projection is shared by every draft layer and token.
            weights = self.layer_fusion_weights.softmax(dim=1).to(target_hidden_states.dtype)
            fused = torch.einsum("lkd,bskd->lbsd", weights, target_hidden_states)
            projected = self.norm(self.shared_projection(fused))
            return ContextOutput(list(projected.unbind(0)), inputs)
        weights = self.layer_fusion_weights.softmax(-1).to(target_hidden_states.dtype)
        return ContextOutput(
            [
                self.norm(torch.einsum("t,bstd->bsd", weight, target_hidden_states))
                for weight in weights
            ],
            inputs,
        )
