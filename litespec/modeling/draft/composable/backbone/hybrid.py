"""Four context-memory GDN layers followed by one full DFlash MHA layer."""
from dataclasses import replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from ..contracts import BackboneOutput, DraftState
from .gdn import GDNBackbone, GDNLayer
from .gdn_events import build_event_layout
from .layer import DraftLayer


class GDNMHABackbone(GDNBackbone):
    def __init__(self, config):
        # Do not construct and discard a fifth GDN layer or consume its RNG.
        nn.Module.__init__(self)
        recipe = config.composable_config['backbone']
        self.block_size = config.block_size
        self.layers = nn.ModuleList([GDNLayer(config, recipe) for _ in range(4)] +
                                   [DraftLayer(config, 4, recipe['layers'][4])])
        self.attention_indices = {4}
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.last_event_count = 0

    def forward(self, context):
        inputs = context.inputs
        original = inputs.draft_state
        if original is not None:
            # Publish both cache types together after every layer succeeds.
            working = DraftState(dict(original.layers), original.backbone_state, original.indexer_state)
            old = original.backbone_state
            if (old is None and working.layers) or (old is not None and set(working.layers) != self.attention_indices):
                raise ValueError('Incomplete hybrid GDN/MHA cache')
            if old is not None and working.layers:
                docs = inputs.anchor_document_ids
                docs = torch.zeros_like(inputs.anchor_positions[:, 0]) if docs is None else docs[:, 0]
                reset = docs != old.document_ids.to(docs.device)
                working.layers = {i: replace(state, valid=state.valid & ~reset[:, None])
                                  for i, state in working.layers.items()}
            inputs = replace(inputs, draft_state=working)
        previous, updated = self.prepare_cache(inputs)
        layout = build_event_layout(inputs, self.block_size)
        self.last_event_count = layout.num_events
        hidden = inputs.draft_embeddings
        if len(context.context_hidden_states) != 5:
            raise ValueError('Hybrid context must provide five layer mappings')
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
            elif self.gradient_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, features, inputs, use_reentrant=False)
            else:
                hidden = layer(hidden, features, inputs)
        if original is not None:
            original.layers = inputs.draft_state.layers
            original.backbone_state = updated
        valid = inputs.block_keep_mask.repeat_interleave(self.block_size, -1)
        return BackboneOutput(self.norm(hidden) * valid[..., None].to(hidden.dtype), original)
