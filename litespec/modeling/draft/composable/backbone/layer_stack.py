"""Arbitrary independent attention layers with atomic confirmed-state commits."""
from dataclasses import replace
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from ..contracts import BackboneOutput, DraftState
from .gdn import GDNBackbone, GDNLayer
from .gdn_events import build_event_layout
from .indexer import AnchorIndexer
from .layer import DraftLayer


class LayerStackBackbone(GDNBackbone):
    def __init__(self, config):
        nn.Module.__init__(self)
        recipes = config.composable_config["backbone"]["layers"]
        self.block_size = config.block_size
        self.layers = nn.ModuleList([
            GDNLayer(config, recipe["gdn_config"]) if recipe["attention"]["representation"] == "gdn"
            else DraftLayer(config, index, recipe)
            for index, recipe in enumerate(recipes)
        ])
        self.gdn_indices = {i for i, layer in enumerate(self.layers) if isinstance(layer, GDNLayer)}
        self.attention_indices = set(range(len(self.layers))) - self.gdn_indices
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.last_event_count = 0
        self.indexer = AnchorIndexer(config) if config.composable_config.get("indexer") else None

    def forward(self, context):
        inputs, original = context.inputs, context.inputs.draft_state
        if len(context.context_hidden_states) != len(self.layers):
            raise ValueError("Context mapping count must equal layer count")
        if original is not None:
            if (set(original.layers) - self.attention_indices or
                    (original.layers and set(original.layers) != self.attention_indices) or
                    (not self.gdn_indices and original.backbone_state is not None)):
                raise ValueError("Cache layers do not match this layer stack")
            if self.gdn_indices and ((original.backbone_state is None and original.layers) or
                                     (original.backbone_state is not None and set(original.layers) != self.attention_indices)):
                raise ValueError("Incomplete layer-stack cache")
            working = DraftState(dict(original.layers), original.backbone_state, original.indexer_state)
            docs = inputs.anchor_document_ids
            docs = torch.zeros_like(inputs.anchor_positions[:, 0]) if docs is None else docs[:, 0]
            # Switch documents without mutating the caller's cache on failure.
            working.layers = {i: replace(state, valid=state.valid & (state.document_ids == docs[:, None]))
                              for i, state in working.layers.items()}
            if working.indexer_state is not None:
                state = working.indexer_state
                working.indexer_state = replace(state, valid=state.valid & (state.document_ids == docs[:, None]))
            inputs = replace(inputs, draft_state=working)
        previous, updated = self.prepare_cache(inputs) if self.gdn_indices else (None, None)
        layout = build_event_layout(inputs, self.block_size) if self.gdn_indices else None
        self.last_event_count = 0 if layout is None else layout.num_events
        selection, teacher_sum, teacher_count = None, None, 0
        if self.indexer is not None and self.indexer.recipe.get("enabled", True):
            selection = self.indexer(inputs, self.layers[0])
            inputs = replace(inputs, selection=selection)
        hidden = inputs.draft_embeddings
        for index, (layer, features) in enumerate(zip(self.layers, context.context_hidden_states)):
            if isinstance(layer, GDNLayer):
                if self.gradient_checkpointing and self.training:
                    def operation(x, c, module=layer):
                        return module(x, c, layout, inputs=inputs)[0]
                    hidden = checkpoint(operation, hidden, features, use_reentrant=False)
                else:
                    hidden, state = layer(hidden, features, layout,
                                          None if previous is None else previous[index], inputs=inputs)
                    if updated is not None:
                        updated.layers[index] = state
            else:
                hidden = checkpoint(layer, hidden, features, inputs, use_reentrant=False) if self.gradient_checkpointing and self.training else layer(hidden, features, inputs)
                if selection is not None:
                    hidden, teacher = hidden
                    if teacher is not None:
                        teacher_sum = teacher if teacher_sum is None else teacher_sum + teacher
                        teacher_count += 1
        auxiliary = self.indexer.auxiliary(selection, teacher_sum / teacher_count) if teacher_count else None
        hidden = self.norm(hidden)
        if self.gdn_indices:
            keep = inputs.block_keep_mask.repeat_interleave(self.block_size, -1)
            hidden = hidden * keep[..., None].to(hidden.dtype)
        if original is not None:
            original.layers = inputs.draft_state.layers
            original.indexer_state = inputs.draft_state.indexer_state
            original.backbone_state = updated
        return BackboneOutput(hidden, original, auxiliary)
