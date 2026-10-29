from dataclasses import replace

from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from ..contracts import BackboneOutput
from .base import Backbone
from .layer import DraftLayer
from .indexer import AnchorIndexer


class TransformerBackbone(Backbone):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DraftLayer(config, index, recipe)
                for index, recipe in enumerate(
                    config.composable_config["backbone"]["layers"]
                )
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.indexer = AnchorIndexer(config) if config.composable_config.get("indexer") else None

    def forward(self, context):
        draft_hidden_states = context.inputs.draft_embeddings
        inputs = context.inputs
        selection, teacher_sum, teacher_count = None, None, 0
        if self.indexer is not None and self.indexer.recipe.get("enabled", True):
            selection = self.indexer(inputs, self.layers[0])
            inputs = replace(inputs, selection=selection)
        for layer, context_hidden_states in zip(
            self.layers, context.context_hidden_states
        ):
            if self.gradient_checkpointing and self.training:
                draft_hidden_states = checkpoint(
                    layer,
                    draft_hidden_states,
                    context_hidden_states,
                    inputs,
                    use_reentrant=False,
                )
            else:
                draft_hidden_states = layer(
                    draft_hidden_states, context_hidden_states, inputs
                )
            if selection is not None:
                draft_hidden_states, teacher = draft_hidden_states
                if teacher is not None:
                    teacher_sum = teacher if teacher_sum is None else teacher_sum + teacher
                    teacher_count += 1
        auxiliary = None
        if teacher_count:
            auxiliary = self.indexer.auxiliary(selection, teacher_sum / teacher_count)
        return BackboneOutput(
            self.norm(draft_hidden_states), context.inputs.draft_state, auxiliary
        )
