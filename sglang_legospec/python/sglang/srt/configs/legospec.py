"""Inference configuration for frozen LEGOSpec composable checkpoints."""

from copy import deepcopy

from transformers import PretrainedConfig


class LEGOSpecDraftConfig(PretrainedConfig):
    model_type = "legospec"

    def __init__(self, composable_config=None, block_size=8, **kwargs):
        kwargs.pop("auto_map", None)
        super().__init__(**kwargs)
        self.composable_config = deepcopy(composable_config or {})
        self.block_size = int(block_size)
        self.architectures = ["LEGOSpecDraftModel"]
        if not self.composable_config:
            # Transformers constructs an empty config when computing diff dicts.
            return
        recipe = self.composable_config
        if self.block_size != 8:
            raise ValueError("LEGOSpec checkpoints require block_size=8")
        if recipe["context"]["type"] != "dflash":
            raise ValueError("LEGOSpec v1 requires dflash context")
        layers = recipe["backbone"]["layers"]
        if len(layers) != self.num_hidden_layers:
            raise ValueError("LEGOSpec layer recipe and num_hidden_layers differ")
        if recipe["proposal"]["type"] not in ("rnn", "selector", "residual_selector", "topk_selector"):
            raise ValueError("LEGOSpec v1 supports rnn and selector proposals")
        if (
            recipe["proposal"]["type"] in ("selector", "residual_selector", "topk_selector")
            and recipe["proposal"].get("selector_mode") not in ("full_vocab_residual", "candidate_only")
        ):
            raise ValueError("LEGOSpec requires an explicit Selector support mode")
        self.sample_from_anchor = False
        self.mask_token_id = int(recipe["mask_token_id"])
        self.target_layer_ids = list(recipe["context"]["target_layer_ids"])
        self.dflash_config = {
            "block_size": self.block_size,
            "target_layer_ids": self.target_layer_ids,
            "mask_token_id": self.mask_token_id,
        }
        self.proposal_type = recipe["proposal"]["type"]
