from .layer_stack import LayerStackBackbone
from .eagle3 import Eagle3Backbone


def build_backbone(config):
    if config.composable_config["backbone"]["type"] == "eagle3":
        return Eagle3Backbone(config)
    if config.composable_config["backbone"]["type"] != "layer_stack":
        raise ValueError("Only v3 layer_stack is supported")
    return LayerStackBackbone(config)
