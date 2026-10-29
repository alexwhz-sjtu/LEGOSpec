"""Serializable shared attention recipe v3; runtime expansion is never saved."""
from copy import deepcopy
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from .schema_v3 import MLA_DEFAULTS, execution_options, normalize_v3, runtime_recipe


class ComposableDraftConfig(Qwen3Config):
    model_type = "litespec_composable"

    def __init__(self, composable_config=None, block_size=8, num_target_layers=None,
                 schema_version="composable_v3", target_attention=None,
                 execution_config=None, **kwargs):
        if schema_version != "composable_v3":
            raise ValueError("Only composable_v3 is supported; historical checkpoint compatibility is not provided")
        if composable_config is None:
            depth = kwargs.get("num_hidden_layers", 5)
            composable_config = {
                "context": {"type": "dflash"},
                "backbone": {"type": "layer_stack", "layers": [
                    {"attention": "gqa", "access": "dense"} for _ in range(depth)]},
                "proposal": {"type": "parallel"}, "objective": {"type": "hard_ce"},
                "attention_configs": {"gqa": {}}, "access_configs": {},
            }
        layers = composable_config.get("backbone", {}).get("layers", [])
        if "num_hidden_layers" not in kwargs:
            kwargs["num_hidden_layers"] = len(layers)
        super().__init__(**kwargs)
        self.block_size = block_size
        self.num_target_layers = num_target_layers if num_target_layers is not None else self.num_hidden_layers
        self.target_attention = deepcopy(target_attention if target_attention is not None else {
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
        })
        self.execution_config = execution_options(execution_config or {})
        self.composable_config = deepcopy(composable_config)
        self.schema_version = schema_version
        self.architectures = ["ComposableDraftModel"]
        self.validate_recipe()

    def validate_recipe(self):
        self.execution_config = execution_options(self.execution_config)
        self.composable_config = normalize_v3(self)

    @property
    def indexer_config(self):
        return self.composable_config["access_configs"].get("dsa")

    def runtime_config(self):
        result = deepcopy(self)
        result.composable_config = runtime_recipe(self)
        if self.indexer_config is not None:
            first = result.composable_config["backbone"]["layers"][0]["attention"]
            result.indexer_source_width = first["num_attention_heads"] * first["head_dim"]
        return result
