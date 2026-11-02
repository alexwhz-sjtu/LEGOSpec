"""Thin deployable orchestration of context, backbone and proposal."""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel

from ..dflash import build_target_layer_ids
from ..dflash2 import DFlashGroupedConv
from ..registry import register_draft
from .backbone import build_backbone
from .configuration import ComposableDraftConfig
from .contracts import DraftOutput
from .context import build_context
from .context.hidden_gated import HiddenGatedContext
from .proposal import build_proposal


@register_draft
class ComposableDraftModel(Qwen3PreTrainedModel):
    config_class = ComposableDraftConfig
    method_config_key = "composable_config"
    _no_split_modules = ["DraftLayer", "GDNLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        config.validate_recipe()
        recipe = config.composable_config
        self.is_eagle3 = recipe["backbone"]["type"] == "eagle3"
        if recipe["context"].get("target_layer_ids") is None:
            recipe["context"]["target_layer_ids"] = (
                list(range(config.num_target_layers))
                if config.num_target_layers < 4
                else sorted(
                    set(
                        build_target_layer_ids(
                            config.num_target_layers, 5 if self.is_eagle3 else config.num_hidden_layers
                        )
                    )
                )
            )
        config.validate_recipe()
        self.target_layer_ids = recipe["context"]["target_layer_ids"]
        self.block_size = config.block_size
        self.mask_token_id = recipe.get("mask_token_id")
        execution = config.runtime_config()
        self.context = build_context(execution)
        self.backbone = build_backbone(execution)
        if not self.is_eagle3:
            self.proposal = build_proposal(execution)
        self.post_init()
        # post_init initializes Linear modules; restore source-specific identity starts.
        for module in self.modules():
            if isinstance(module, DFlashGroupedConv):
                torch.nn.init.zeros_(module.kernel_projection.weight)
            if isinstance(module, HiddenGatedContext):
                module.reset_fusion_prior()
        head = getattr(getattr(self, "proposal", None), "markov_head", None)
        if head is not None:
            torch.nn.init.zeros_(head.markov_w2.weight)
        selector = getattr(getattr(self, "proposal", None), "candidate_selector", None)
        if selector is not None:
            torch.nn.init.zeros_(selector.successor_codebook)

    def forward(self, inputs, *, shared_ops=None):
        if self.is_eagle3:
            raise ValueError("EAGLE3 uses forward_ttt or extend_eagle3, not block proposal forward")
        if inputs.mode not in ("features", "teacher_forcing", "sample"):
            raise ValueError("mode must be features, teacher_forcing, or sample")
        if inputs.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.training and inputs.draft_state is not None:
            raise ValueError("Persistent inference state is not accepted in training")
        if inputs.draft_state is not None and inputs.anchor_positions.size(1) != 1:
            raise ValueError("Cached inference expects one current block per sample")
        context = self.context(inputs)
        features = self.backbone(context)
        output = self.proposal(features, context, shared_ops)
        output.backbone_hidden_states = features.draft_hidden_states
        output.indexer_aux = features.indexer_aux
        return output

    @property
    def ttt_length(self):
        return self.block_size - 1

    def forward_ttt(self, target_hidden_states, input_ids, valid, anchor_positions, shared_ops):
        if not self.is_eagle3:
            raise ValueError("forward_ttt requires EAGLE3")
        hidden = self.context.project(target_hidden_states)
        return self.backbone.ttt(hidden, input_ids, valid, shared_ops.embed_tokens, anchor_positions)

    def score_eagle3(self, hidden, shared_ops):
        return DraftOutput(hidden, draft_logits=shared_ops.lm_head(self.backbone.norm(hidden)))

    def extend_eagle3(self, hidden, token_ids, state, shared_ops, *, target_features=False):
        if target_features:
            hidden = self.context.project(hidden)
        return self.backbone.extend(hidden, shared_ops.embed_tokens(token_ids), state)

    def spec_generate(self, target, input_ids, max_new_tokens, stop_token_ids=(),
                      temperature=0., top_p=1., *, sample_id="request", seed=42):
        from .eagle3_inference import generate_eagle3
        if not self.is_eagle3:
            raise ValueError("Use the block benchmark adapter for non-EAGLE3 models")
        self.eval()
        result = generate_eagle3(target, self, input_ids, sample_id, temperature,
                                 stop_token_ids, max_new_tokens, top_p, seed)
        self._last_decode_stats = result
        return torch.cat((input_ids, input_ids.new_tensor([result["generated_ids"]])), 1)

    def get_last_decode_stats(self):
        return dict(self._last_decode_stats)
