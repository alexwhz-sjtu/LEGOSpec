from .layer_stack import LayerStackBackbone


def build_backbone(config):
    if config.composable_config["backbone"]["type"] != "layer_stack":
        raise ValueError("Only v3 layer_stack is supported")
    return LayerStackBackbone(config)
